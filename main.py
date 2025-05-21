from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import joblib
import requests
from fastapi.middleware.cors import CORSMiddleware
import datetime
import os
from typing import List, Dict

app = FastAPI()

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 前期开发阶段可设为 "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载模型
model = joblib.load("crop_model.pkl")

# API接口相关配置
API_URL = "http://iot-api.heclouds.com/datapoint/history-datapoints"
API_HEADERS = {
    "Accept": "application/json",
    "authorization": "version=2018-10-31&res=products%2Fch0NAXd446&et=1840265191&method=sha1&sign=PnvsKRBpkXxzKbW023%2B3oqVMlOY%3D"
}

# API请求参数 (修改为基础参数，limit将在函数中动态添加)
BASE_API_PARAMS = {
    "product_id": "ch0NAXd446",
    "device_name": "device1"
}

# 假设 CSV 文件在脚本的同级目录下
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")
IDEAL_CROP_CONDITIONS: Dict[str, Dict[str, float]] = {}
CROP_NAMES: List[str] = []

# OneNET 配置 (与之前相同)
# ONENET_API_KEY = "YOUR_ONENET_API_KEY" # 请替换为您的 OneNET Master API Key
# ONENET_DEVICE_ID = "YOUR_ONENET_DEVICE_ID" # 请替换为您的设备ID
# ONENET_API_URL = f"http://api.heclouds.com/devices/{ONENET_DEVICE_ID}/datapoints"

# 传感器键到CSV列名的映射 (用于预警功能)
# SENSOR_TO_CSV_MAP = {
# 'tu_nitrogen': 'N',
# 'tu_phosphorus': 'P',
# 'tu_potassium': 'K',
# 'temp': 'temperature', # 环境温度
# 'hum': 'humidity',     # 环境湿度
# 'ph': 'ph',            # 土壤pH (假设OneNET有此数据流或模拟)
# 'yuliang': 'rainfall'  # 降雨量
# }
# CSV列名到展示名称的映射
CSV_COLUMN_TO_DISPLAY_NAME = {
    'Nitrogen': '氮 (N)',       # 键更新以匹配CSV
    'phosphorus': '磷 (P)',   # 键更新以匹配CSV (小写p)
    'potassium': '钾 (K)',    # 键更新以匹配CSV (小写k)
    'temperature': '环境温度',
    'humidity': '环境湿度',
    'ph': '土壤pH值',
    'rainfall': '降雨量'
}

# API请求参数 (修改为基础参数，limit将在函数中动态添加)
# BASE_API_PARAMS = {
# "product_id": "ch0NAXd446",
# "device_name": "device1"
# }

# 请求数据模型
class CropInput(BaseModel):
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


# 新增数据模型用于作物建议接口
class CropAdviceInput(BaseModel):
    crop_type: str
    crop_age: float  # 农作物年龄，单位为月份


def get_sensor_data(limit: str = "1"): # 默认获取1条最新数据
    """从API获取传感器数据"""
    params = {**BASE_API_PARAMS, "limit": limit} # 合并基础参数和动态limit
    try:
        response = requests.get(API_URL, headers=API_HEADERS, params=params)
        response.raise_for_status()  # 检查请求是否成功

        data = response.json()

        if data.get("code") != 0:
            return {"error": f"API返回错误: {data.get('msg')}"}

        return data
    except Exception as e:
        return {"error": f"获取传感器数据失败: {str(e)}"}


def extract_model_inputs(api_data):
    """
    从API返回的数据中提取模型所需的输入参数
    模型需要的输入: Nitrogen, Phosphorus, Potassium, temperature, humidity, ph, rainfall
    """
    if not api_data or "data" not in api_data:
        return None

    # 创建一个字典来存储映射后的数据
    model_inputs = {}

    # 数据流映射关系
    mapping = {
        "tu_nitrogen": "Nitrogen",
        "tu_phosphorus": "Phosphorus",
        "tu_potassium": "Potassium",
        "temp": "temperature",
        "hum": "humidity",
        "yuliang": "rainfall"
    }

    # 从API数据中提取并映射数值
    for stream in api_data["data"]["datastreams"]:
        stream_id = stream["id"]
        if stream_id in mapping and stream["datapoints"] and len(stream["datapoints"]) > 0: # 确保datapoints存在且不为空
            # API默认返回数据是按时间倒序的，所以第一条就是最新的
            model_inputs[mapping[stream_id]] = float(stream["datapoints"][0]["value"])

    # 为pH值设置默认值，因为API数据中没有这个字段
    model_inputs["ph"] = 7.0  # pH的默认值

    # 检查是否获取了所有必要参数
    required_params = ["Nitrogen", "Phosphorus", "Potassium", "temperature", "humidity", "ph", "rainfall"]

    # 检查是否缺少其他必要参数
    missing_params = [param for param in required_params if param not in model_inputs]
    if missing_params:
        # 对于缺失的参数，设置默认值
        for param in missing_params:
            model_inputs[param] = 0

    return model_inputs


def generate_advice(input_data, predicted_crop, crop_age=None):
    """调用大模型生成种植建议"""
    # 构造提示词
    age_info = f"- 农作物年龄: {crop_age} 个月" if crop_age is not None else ""
    prompt = f"""
你是一位农业专家，请参考下面输入的土质参数、着重针对模型给出的推荐结果，向用户给出专业的种植建议：

1. 适合种植哪种作物
2. 基于推荐结果每一个输入参数给出详细的理由和解释
3. 种植周期与注意事项（考虑农作物年龄）
4. 浇水建议（频率、方式，结合农作物年龄）
5. 施肥建议（种类、用量、时间，结合农作物年龄）

输入数据如下：
- 氮（Nitrogen）: {input_data["Nitrogen"]}
- 磷（Phosphorus）: {input_data["Phosphorus"]}
- 钾（Potassium）: {input_data["Potassium"]}
- 温度: {input_data["temperature"]}℃
- 湿度: {input_data["humidity"]}%
- pH值: {input_data["ph"]}
- 降水量: {input_data["rainfall"]} mm
{age_info}

推荐作物：{predicted_crop}

注意 所有英文字段都需要翻译成中文回答
"""

    # 调用大模型 API
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-jyjwpeexxccslfnudqqvwdffkcikuwnjhduorariyqxpzhww",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "temperature": 0.1,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
        "tools": []
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response_json = response.json()
        advice = response_json['choices'][0]['message']['content']
        return advice
    except Exception as e:
        return f"生成建议失败: {str(e)}"


def generate_crop_advice(user_data, ideal_conditions, crop_type, crop_age):
    """调用大模型生成施肥和浇水建议"""
    # 构造提示词
    prompt = f"""
你是一位农业专家，请根据当前土壤和环境条件、农作物年龄以及适合指定作物的理想条件，为用户提供专业的施肥和浇水建议，充分考虑农作物的生长阶段,回答里面必须加上根据当前植物年龄在各个参数点上给出推荐建议，并需要给出理由。建议应包括：

1. 当前植物年龄所在的生长周期
2. 根据当前生长周期对当前土壤与理想条件的差异分析（每个参数的对比）
2. 根据当前生长周期的施肥建议（肥料种类、用量、施肥时间和方法，结合农作物年龄）
3. 根据当前生长周期的浇水建议（频率、方式、注意事项，结合农作物年龄）
4. 根据当前生长周期的其他改善土壤或环境的建议（如果适用）

当前土壤和环境条件：
- 氮（Nitrogen）: {user_data["Nitrogen"]}
- 磷（Phosphorus）: {user_data["Phosphorus"]}
- 钾（Potassium）: {user_data["Potassium"]}
- 温度: {user_data["temperature"]}℃
- 湿度: {user_data["humidity"]}%
- pH值: {user_data["ph"]}
- 降水量: {user_data["rainfall"]} mm
- 农作物年龄: {crop_age} 个月

适合 {crop_type} 的理想条件：
- 氮（Nitrogen）: {ideal_conditions["Nitrogen"]}
- 磷（Phosphorus）: {ideal_conditions["Phosphorus"]}
- 钾（Potassium）: {ideal_conditions["Potassium"]}
- 温度: {ideal_conditions["temperature"]}℃
- 湿度: {ideal_conditions["humidity"]}%
- pH值: {ideal_conditions["ph"]}
- 降水量: {ideal_conditions["rainfall"]} mm

注意：所有英文字段都需要翻译成中文回答，回答应简洁、实用且专业。
"""

    # 调用大模型 API
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-jyjwpeexxccslfnudqqvwdffkcikuwnjhduorariyqxpzhww",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "temperature": 0.1,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
        "tools": []
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response_json = response.json()
        advice = response_json['choices'][0]['message']['content']
        return advice
    except Exception as e:
        return f"生成施肥和浇水建议失败: {str(e)}"


# FastAPI 路由 - 手动输入数据进行预测
@app.post("/predict_crop")
def predict_crop(input_data: CropInput):
    # 转换为 DataFrame
    feature_order = ["Nitrogen", "Phosphorus", "Potassium", "temperature", "humidity", "ph", "rainfall"]
    input_dict = input_data.dict()
    ordered_data = {feature: input_dict[feature] for feature in feature_order}
    df_input = pd.DataFrame([ordered_data])

    # 模型预测
    predicted_crop = model.predict(df_input)[0]

    # 生成种植建议
    advice = generate_advice(input_dict, predicted_crop)

    return {
        "predicted_crop": predicted_crop,
        "input_data": input_dict,
        "agricultural_advice": advice
    }


# FastAPI 路由 - 从API获取数据进行预测
@app.get("/api_predict")
def api_predict():
    # 获取API数据 (调用默认参数，limit=1)
    api_data = get_sensor_data()

    if "error" in api_data:
        return {"error": api_data["error"]}

    # 提取模型输入数据
    model_inputs = extract_model_inputs(api_data)

    if not model_inputs:
        return {"error": "无法提取有效的模型输入数据"}

    # 确保特征的顺序与训练时一致
    feature_order = ["Nitrogen", "Phosphorus", "Potassium", "temperature", "humidity", "ph", "rainfall"]
    ordered_data = {feature: model_inputs[feature] for feature in feature_order}

    # 转换为DataFrame格式
    df_input = pd.DataFrame([ordered_data])

    # 模型预测
    try:
        predicted_crop = model.predict(df_input)[0]
    except Exception as e:
        return {"error": f"预测失败: {str(e)}"}

    # 生成农业建议
    advice = generate_advice(model_inputs, predicted_crop)

    # 获取数据的时间戳，格式化为易读的日期时间字符串
    timestamp = None
    if "data" in api_data and "datastreams" in api_data["data"] and len(api_data["data"]["datastreams"]) > 0:
        if "datapoints" in api_data["data"]["datastreams"][0] and len(
                api_data["data"]["datastreams"][0]["datapoints"]) > 0:
            timestamp = api_data["data"]["datastreams"][0]["datapoints"][0].get("at", None)

    # 当前时间作为默认值
    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "predicted_crop": predicted_crop,
        "input_data": model_inputs,
        "agricultural_advice": advice,
        "data_source": "物联网设备实时数据",
        "timestamp": timestamp
    }


# 新增路由 - 根据作物类型、年龄和物联网土壤数据生成施肥和浇水建议
@app.post("/crop_advice")
def crop_advice(input_data: CropAdviceInput):
    # 获取用户输入的作物类型和年龄
    crop_type = input_data.crop_type
    crop_age = input_data.crop_age

    # 获取API数据 (调用默认参数，limit=1，获取最新土壤数据)
    api_data = get_sensor_data()

    if "error" in api_data:
        return {"error": api_data["error"]}

    # 提取模型输入数据（当前土壤条件）
    current_conditions = extract_model_inputs(api_data)

    if not current_conditions:
        return {"error": "无法提取有效的土壤数据"}

    # 确保特征的顺序一致
    feature_order = ["Nitrogen", "Phosphorus", "Potassium", "temperature", "humidity", "ph", "rainfall"]
    ordered_data = {feature: current_conditions[feature] for feature in feature_order}

    # 模拟模型预测适合该作物的理想条件
    # 注：原模型预测作物而非条件，这里使用占位符，实际需反向映射或数据集
    # 过滤指定作物label

    # 读取CSV文件
    df = pd.read_csv("Crop_recommendation.csv")
    crop_data = df[df["label"].str.lower() == crop_type.lower()]

    if crop_data.empty:
        return {"error": f"未找到作物类型'{crop_type}'的推荐数据"}

    # 计算所需列的平均值，注意CSV列名和返回字段名对应关系
    # CSV列名：Nitrogen,phosphorus,potassium,...
    # 返回时大写首字母：Nitrogen,Phosphorus,Potassium
    ideal_conditions = {
        "Nitrogen": crop_data["Nitrogen"].mean(),
        "Phosphorus": crop_data["phosphorus"].mean(),
        "Potassium": crop_data["potassium"].mean(),
        "temperature": crop_data["temperature"].mean(),
        "humidity": crop_data["humidity"].mean(),
        "ph": crop_data["ph"].mean(),
        "rainfall": crop_data["rainfall"].mean(),
    }

    # 生成施肥和浇水建议
    advice = generate_crop_advice(current_conditions, ideal_conditions, crop_type, crop_age)

    # 获取数据的时间戳
    timestamp = None
    if "data" in api_data and "datastreams" in api_data["data"] and len(api_data["data"]["datastreams"]) > 0:
        if "datapoints" in api_data["data"]["datastreams"][0] and len(
                api_data["data"]["datastreams"][0]["datapoints"]) > 0:
            timestamp = api_data["data"]["datastreams"][0]["datapoints"][0].get("at", None)

    # 当前时间作为默认值
    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "crop_type": crop_type,
        "crop_age": crop_age,
        "current_conditions": ordered_data,
        "ideal_conditions": ideal_conditions,
        "agricultural_advice": advice,
        "data_source": "物联网设备实时数据",
        "timestamp": timestamp
    }


# 新增API端点：获取用于图表展示的传感器历史数据
@app.get("/api/sensor_chart_data")
async def get_sensor_chart_data_endpoint():
    # 获取最新的10条数据
    raw_sensor_data = get_sensor_data(limit="10")
    #print("DEBUG: raw_sensor_data from get_sensor_data():", raw_sensor_data) # 确认此函数返回

    if "error" in raw_sensor_data:
        print("ERROR in get_sensor_chart_data_endpoint: Error from get_sensor_data()", raw_sensor_data["error"])
        return {"error": raw_sensor_data["error"]}

    if not raw_sensor_data or "data" not in raw_sensor_data or not raw_sensor_data["data"].get("datastreams"):
        print("ERROR in get_sensor_chart_data_endpoint: Invalid raw_sensor_data structure (no data or datastreams)")
        return {"error": "未找到数据流或数据格式无效"}

    chart_data = {}
    chinese_labels = {
        "Nitrogen": "氮含量", "Phosphorus": "磷含量", "Potassium": "钾含量",
        "temperature": "温度", "humidity": "湿度", "ph": "pH值", "rainfall": "降水量",
        "tu_nitrogen": "土壤氮含量", "tu_phosphorus": "土壤磷含量",
        "tu_potassium": "土壤钾含量", 
        "temp": "环境温度", 
        "hum": "环境湿度",  
        "yuliang": "降水量", "guangmin": "光照强度",
        "tu_humidity": "土壤湿度", "tu_ec": "土壤EC值",
        "tu_temperature": "土壤温度", "tu_salt": "土壤盐分"
    }

    latest_timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # found_timestamp_for_all_data = False # 这个变量的逻辑之前有些复杂，我们简化处理
    absolute_latest_time_obj = None

    try:
        for stream_index, stream in enumerate(raw_sensor_data["data"]["datastreams"]):
            stream_id = stream.get("id", f"unknown_stream_{stream_index}")
            print(f"DEBUG: Processing stream {stream_index + 1}: ID = {stream_id}")
            points = []
            if "datapoints" in stream and stream["datapoints"]:
                # API返回数据点默认按时间倒序 (最新在前)。图表需要升序 (旧数据在前)。
                # 所以我们反转datapoints列表进行处理。
                for dp_index, dp in enumerate(reversed(stream["datapoints"])):
                    try:
                        # 尝试解析时间，支持带毫秒和不带毫秒的格式
                        time_str = dp["at"]
                        if "." in time_str:
                            current_dp_time_obj = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
                        else:
                            current_dp_time_obj = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                        
                        value = float(dp["value"])
                        points.append({"time": time_str, "value": value}) # 存储原始时间字符串给前端
                        
                        # 更新 overall latest timestamp
                        if absolute_latest_time_obj is None or current_dp_time_obj > absolute_latest_time_obj:
                            absolute_latest_time_obj = current_dp_time_obj

                    except KeyError as ke:
                        print(f"ERROR in stream '{stream_id}', datapoint {dp_index}: Missing key {ke}. Datapoint: {dp}")
                        #可以选择跳过这个数据点或采取其他措施
                        continue 
                    except ValueError as ve:
                        print(f"ERROR in stream '{stream_id}', datapoint {dp_index}: Value conversion error (e.g., float, datetime). Error: {ve}. Datapoint: {dp}")
                        continue 
                    except Exception as e_dp:
                        print(f"ERROR in stream '{stream_id}', datapoint {dp_index}: Unexpected error. Error: {e_dp}. Datapoint: {dp}")
                        continue
            else:
                print(f"DEBUG: Stream '{stream_id}' has no datapoints or 'datapoints' key is missing.")

            # 映射中文标签
            label_key_for_chinese = stream_id
            # (此处省略了之前的 reverse_mapping 逻辑，因为它依赖于 extract_model_inputs 的一个不太稳健的调用方式)
            # 可以简化为直接查找或提供一个更明确的映射表如果需要的话
            label = chinese_labels.get(label_key_for_chinese, stream_id)
            chart_data[label] = points
        
        if absolute_latest_time_obj:
            latest_timestamp_str = absolute_latest_time_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # 保留毫秒
        else:
            # 如果没有找到任何有效的时间戳（例如所有数据点都解析失败或没有数据点）
            latest_timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Fallback
            print("WARN: No valid datapoint timestamps found, using current time as fallback for latest_timestamp_str.")


        # print("DEBUG: 최종 chart_data:", chart_data)
        # print("DEBUG: 최종 latest_timestamp_str:", latest_timestamp_str)

        return {
            "chart_data": chart_data,
            "data_source": "物联网设备实时数据",
            "timestamp": latest_timestamp_str
        }

    except Exception as e_main:
        print(f"FATAL ERROR in get_sensor_chart_data_endpoint during main processing loop: {str(e_main)}")
        # 为了帮助诊断，可以考虑返回更详细的错误信息，但这可能暴露内部细节
        # import traceback
        # print(traceback.format_exc())
        # 对于生产环境，应该返回一个通用的服务器错误信息
        return {"error": f"处理图表数据时服务器内部发生严重错误: {str(e_main)}"} # 返回一个JSON响应体错误


# 新增API端点：获取天气数据 (以和风天气为例)
@app.get("/api/weather")
async def get_weather_data_qweather(city_location: str = "101030100"): # 默认为天津的Location ID
    API_KEY = "c3cba4e0b1734d78a214f9c176b56655"  # 用户提供的有效和风天气API Key
    
    if API_KEY == "YOUR_QWEATHER_API_KEY_PLACEHOLDER": # 保留此逻辑以防未来API Key失效或测试
        return {
            "city": "示例城市 (天津)", # 更新模拟数据城市
            "temperature": "18", # 模拟数据
            "feels_like": "17",
            "description": "多云",
            "icon": "101", 
            "wind_dir": "东南风",
            "wind_scale": "2",
            "humidity": "60",
            "precip": "0.0",
            "pressure": "1015",
            "vis": "15",
            "obs_time": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00"),
            "api_status": "使用占位API Key，此为模拟数据",
            "fxLink": "http://hfx.link/2ax1" # 模拟数据中的fxLink
        }

    BASE_URL = "https://pm6apvuwv7.re.qweatherapi.com/v7/weather/now"
    params = {
        "location": city_location,
        "key": API_KEY,
        "lang": "zh", # 返回中文天气描述
        "unit": "m"  # 使用公制单位 (温度摄氏度等)
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("code") == "200": # 和风天气API成功响应代码
            now = data["now"]
            return {
                "city_name_from_api": "天津",
                "temperature": now["temp"],
                "feels_like": now["feelsLike"],
                "description": now["text"],
                "icon": now["icon"], # 和风天气图标代码 (https://dev.qweather.com/docs/resource/icons/)
                "wind_dir": now["windDir"],
                "wind_scale": now["windScale"],
                "humidity": now["humidity"],
                "precip": now["precip"], # 当前小时累计降水量
                "pressure": now["pressure"],
                "vis": now["vis"], # 能见度 (公里)
                "obs_time": data.get("updateTime", now.get("obsTime")), #观测时间
                "api_status": "数据来自和风天气",
                "fxLink": data.get("fxLink") # 添加fxLink到真实API返回
            }
        else:
            return {"error": f"获取天气数据失败: 和风天气API错误 - {data.get('code')}", "api_response": data}
    except requests.exceptions.RequestException as e:
        return {"error": f"获取天气数据请求失败: {str(e)}"}
    except Exception as e: # 其他潜在错误
        return {"error": f"处理天气数据时发生未知错误: {str(e)}"}


@app.on_event("startup")
async def load_data_and_model():
    global IDEAL_CROP_CONDITIONS, CROP_NAMES
    try:
        if not os.path.exists(CSV_FILE_PATH):
            print(f"错误: CSV文件未找到路径 {CSV_FILE_PATH}")
            return

        df = pd.read_csv(CSV_FILE_PATH)
        # 计算每种作物的平均理想条件
        # 更新这里的列名以完全匹配CSV文件
        numeric_cols = ['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # 检查所有期望的列是否存在于DataFrame中
        missing_cols = [col for col in numeric_cols if col not in df.columns]
        if missing_cols:
            print(f"错误: CSV文件中缺少以下列: {', '.join(missing_cols)}. 请检查CSV文件列名是否与代码中的期望一致。")
            # 如果缺少关键列，IDEAL_CROP_CONDITIONS 将不会被正确填充
            # 后续依赖此数据的端点可能会失败或返回错误。
            return

        grouped = df.groupby('label')[numeric_cols].mean()
        IDEAL_CROP_CONDITIONS = grouped.to_dict(orient='index')
        
        # 确保 'label' 列存在
        if 'label' not in df.columns:
            print(f"错误: CSV文件中缺少 'label' 列。无法提取作物名称列表。")
            # CROP_NAMES 将为空，依赖此数据的端点可能会失败
            return
            
        CROP_NAMES = df['label'].unique().tolist()
        CROP_NAMES.sort() # 排序作物名称
        print("作物理想条件数据已加载。")
        print(f"共加载 {len(CROP_NAMES)}种作物。例如: {CROP_NAMES[:5]}...")
    except FileNotFoundError:
        print(f"错误: Crop_recommendation.csv 文件未在路径 {CSV_FILE_PATH} 找到。预警功能将不可用。")
    except Exception as e:
        print(f"加载或处理CSV时发生错误: {e}")


# 新增API端点：获取所有作物名称
@app.get("/api/crop_names")
async def get_crop_names():
    if not CROP_NAMES: # 如果CSV加载失败
        raise HTTPException(status_code=500, detail="作物数据未加载，请检查服务器日志。")
    return {"crop_names": CROP_NAMES}


# 新增API端点：作物环境预警
@app.get("/api/crop_warning")
async def crop_warning_system(crop_type: str = Query(..., title="作物种类")):
    if not IDEAL_CROP_CONDITIONS or not CROP_NAMES:
        raise HTTPException(status_code=500, detail="作物理想条件数据未加载，无法提供预警。")

    if crop_type not in IDEAL_CROP_CONDITIONS:
        raise HTTPException(status_code=404, detail=f"未找到作物 '{crop_type}' 的理想条件数据。可用作物: {CROP_NAMES}")

    ideal_conditions_for_crop = IDEAL_CROP_CONDITIONS[crop_type] # 这些键是CSV中的列名, e.g., 'Nitrogen', 'phosphorus'
    
    # 1. 从OneNET获取最新传感器数据 (使用已有的同步函数)
    api_data = get_sensor_data() # main.py中定义的get_sensor_data

    if "error" in api_data:
        return {
            "crop_type": crop_type,
            "ideal_conditions": {CSV_COLUMN_TO_DISPLAY_NAME.get(k, k): v for k, v in ideal_conditions_for_crop.items()},
            "current_conditions": None,
            "deviations": None,
            "assessment": {
                "level": "Error",
                "message": f"获取当前传感器数据失败: {api_data['error']}"
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    current_conditions_from_model_inputs = extract_model_inputs(api_data) # main.py中定义的extract_model_inputs

    if not current_conditions_from_model_inputs:
        return {
            "crop_type": crop_type,
            "ideal_conditions": {CSV_COLUMN_TO_DISPLAY_NAME.get(k, k): v for k, v in ideal_conditions_for_crop.items()},
            "current_conditions": None,
            "deviations": None,
            "assessment": {
                "level": "Error",
                "message": "无法从传感器数据中提取有效参数。"
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    # current_conditions_from_model_inputs 的键是 "Nitrogen", "Phosphorus", "Potassium" 等 (模型输入格式)
    # ideal_conditions_for_crop 的键是 "Nitrogen", "phosphorus", "potassium" 等 (CSV列名格式)
    # 我们需要一个映射，以便在比较时使用正确的键从 current_conditions_from_model_inputs 中取值
    CSV_TO_MODEL_INPUT_KEY_MAP = {
        'Nitrogen': 'Nitrogen',
        'phosphorus': 'Phosphorus', # CSV 'phosphorus' (小写p) -> model input 'Phosphorus' (大写P)
        'potassium': 'Potassium',   # CSV 'potassium' (小写k) -> model input 'Potassium' (大写K)
        'temperature': 'temperature',
        'humidity': 'humidity',
        'ph': 'ph',
        'rainfall': 'rainfall'
    }

    current_conditions_display_mapped = {} # 用于最终API输出的当前条件（带中文名）
    deviations = {}
    warnings_messages = [] # 重命名以避免与内置的 warnings 模块冲突
    overall_level = "Good"

    for param_csv_key, ideal_value in ideal_conditions_for_crop.items():
        model_input_key = CSV_TO_MODEL_INPUT_KEY_MAP.get(param_csv_key)
        current_value = None
        status_for_param = "Data missing" # 默认状态

        display_name = CSV_COLUMN_TO_DISPLAY_NAME.get(param_csv_key, param_csv_key)

        if model_input_key and model_input_key in current_conditions_from_model_inputs:
            raw_current_value = current_conditions_from_model_inputs[model_input_key]
            try:
                current_value = float(raw_current_value) # 确保是浮点数
                current_conditions_display_mapped[display_name] = round(current_value, 2)
                status_for_param = "Good" # 假设良好，后续会根据偏差调整
            except (ValueError, TypeError):
                current_value = None # 值无效
                current_conditions_display_mapped[display_name] = "N/A (invalid)"
                warnings_messages.append(f"{display_name}: 当前传感器值无效 ('{raw_current_value}')。")
        else:
            current_conditions_display_mapped[display_name] = "N/A (missing)"
            warnings_messages.append(f"{display_name}: 传感器数据缺失。")

        if current_value is None: # 如果数据缺失或无效
            deviations[display_name] = {"ideal": round(ideal_value, 2), "current": "N/A", "diff_percent": "N/A", "status": status_for_param}
            if overall_level != "Critical": overall_level = "Warning"
            continue

        diff = current_value - ideal_value
        diff_percent_val = "N/A"
        if ideal_value == 0:
            if diff != 0: diff_percent_val = "N/A (ideal is 0)"
            else: diff_percent_val = 0.0
        else:
            diff_percent_val = (diff / abs(ideal_value)) * 100
        
        # 阈值定义 (可以外部化或进一步细化)
        threshold_critical_percent = 50.0 # 例如 50%
        threshold_warning_percent = 25.0  # 例如 25%
        ph_abs_diff_critical = 0.7
        ph_abs_diff_warning = 0.4
        temp_abs_percent_critical = 30.0
        temp_abs_percent_warning = 15.0

        param_status_summary = "" # 用于构建警告信息中的参数部分

        if param_csv_key == 'ph':
            abs_diff_ph = abs(diff)
            if abs_diff_ph > ph_abs_diff_critical:
                status_for_param = "Critical"
                param_status_summary = f"{display_name} ({current_value:.2f}) 严重偏离理想值 ({ideal_value:.2f})"
            elif abs_diff_ph > ph_abs_diff_warning:
                status_for_param = "Warning"
                param_status_summary = f"{display_name} ({current_value:.2f}) 偏离理想值 ({ideal_value:.2f})"
        elif param_csv_key == 'temperature':
            if abs(diff_percent_val) > temp_abs_percent_critical:
                 status_for_param = "Critical"
                 param_status_summary = f"{display_name} ({current_value:.1f}°C) 严重偏离理想值 ({ideal_value:.1f}°C)"
            elif abs(diff_percent_val) > temp_abs_percent_warning:
                 status_for_param = "Warning"
                 param_status_summary = f"{display_name} ({current_value:.1f}°C) 偏离理想值 ({ideal_value:.1f}°C)"
        else: # 其他参数使用通用百分比阈值
            if abs(diff_percent_val) > threshold_critical_percent:
                status_for_param = "Critical"
                param_status_summary = f"{display_name} ({current_value:.2f}) 严重偏离理想值 ({ideal_value:.2f})"
            elif abs(diff_percent_val) > threshold_warning_percent:
                status_for_param = "Warning"
                param_status_summary = f"{display_name} ({current_value:.2f}) 偏离理想值 ({ideal_value:.2f})"
        
        if param_status_summary: # 如果有具体的偏差描述
            warnings_messages.append(param_status_summary + "。")

        deviations[display_name] = {
            "ideal": round(ideal_value, 2), 
            "current": round(current_value, 2), 
            "diff_percent": f"{diff_percent_val:.1f}%" if isinstance(diff_percent_val, float) else diff_percent_val,
            "status": status_for_param
        }
        
        if status_for_param == "Critical":
            overall_level = "Critical"
        elif status_for_param == "Warning" and overall_level == "Good": # 只有当之前是Good时才降为Warning
            overall_level = "Warning"

    assessment_message = ""
    critical_params = [name for name, dev_info in deviations.items() if dev_info['status'] == 'Critical']
    warning_params = [name for name, dev_info in deviations.items() if dev_info['status'] == 'Warning']
    missing_data_params = [name for name, dev_info in deviations.items() if dev_info['status'] == 'Data missing'] # 仅数据缺失

    if overall_level == "Critical":
        assessment_message = f"环境状况危急! {', '.join(critical_params)} 等参数严重偏离理想值。请立即检查并调整。"
    elif overall_level == "Warning":
        summary_parts = []
        if warning_params: summary_parts.append(f"{', '.join(warning_params)} 等参数偏离理想值")
        if missing_data_params: summary_parts.append(f"{', '.join(missing_data_params)} 等参数数据缺失")
        assessment_message = f"环境状况存在警告。{'; '.join(summary_parts)}。建议关注。"
    else: # Good
        assessment_message = f"当前环境条件对所选作物 ({crop_type}) 来说基本良好。"
    
    # 如果有具体的警告信息，优先展示它们
    if warnings_messages:
         assessment_message = "评估总结：\\n" + "\\n".join(warnings_messages)
    elif overall_level == "Good" : # 确保没有警告信息且整体良好时，才使用通用良好消息
         assessment_message = f"当前环境条件对 {crop_type} 整体良好，各项关键指标均在合理范围内。"


    return {
        "crop_type": crop_type,
        "ideal_conditions": {CSV_COLUMN_TO_DISPLAY_NAME.get(k, k): round(v,2) for k,v in ideal_conditions_for_crop.items()},
        "current_conditions": current_conditions_display_mapped, # 使用已映射好包含中文键的当前条件
        "deviations": deviations, # deviations 已经使用了 display_name作为键
        "assessment": {
            "level": overall_level,
            "message": assessment_message
        },
        "timestamp": datetime.datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)