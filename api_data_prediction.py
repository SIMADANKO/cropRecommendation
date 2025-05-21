import pandas as pd
import joblib
import requests
import json

# 加载训练好的模型
model = joblib.load("crop_model.pkl")

# API接口相关配置
API_URL = "http://iot-api.heclouds.com/datapoint/history-datapoints"
HEADERS = {
    "Accept": "application/json",
    "authorization": "version=2022-05-01&res=userid%2F441042&et=1747189050&method=sha1&sign=sFnbNMwr6CZPxzHU1M3sL1NBtlo%3D"
}

# API请求参数
params = {
    "product_id": "ch0NAXd446",
    "device_name": "device1",
    "limit": "1"  # 只获取最新的一条数据
}

def get_sensor_data():
    """从API获取传感器数据"""
    try:
        response = requests.get(API_URL, headers=HEADERS, params=params)
        response.raise_for_status()  # 检查请求是否成功
        
        data = response.json()
        
        if data.get("code") != 0:
            print(f"API返回错误: {data.get('msg')}")
            return None
            
        return data
    except Exception as e:
        print(f"获取传感器数据失败: {str(e)}")
        return None

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
        if stream_id in mapping and stream["datapoints"]:
            model_inputs[mapping[stream_id]] = float(stream["datapoints"][0]["value"])
    
    # 为pH值设置默认值，因为API数据中没有这个字段
    model_inputs["ph"] = 7.0  # pH的默认值
    
    # 检查是否获取了所有必要参数
    required_params = ["Nitrogen", "Phosphorus", "Potassium", "temperature", "humidity", "ph", "rainfall"]
    
    # 检查是否缺少其他必要参数
    missing_params = [param for param in required_params if param not in model_inputs]
    if missing_params:
        print(f"缺少必要参数: {', '.join(missing_params)}")
        # 对于缺失的参数，设置默认值
        for param in missing_params:
            model_inputs[param] = 0
    
    return model_inputs

def predict_crop(input_data):
    """使用模型预测作物"""
    if not input_data:
        return None
    
    try:
        # 确保特征的顺序与训练时一致
        # 训练时的特征顺序：Nitrogen, Phosphorus, Potassium, temperature, humidity, ph, rainfall
        feature_order = ["Nitrogen", "Phosphorus", "Potassium", "temperature", "humidity", "ph", "rainfall"]
        
        # 创建一个按照特定顺序排列的字典
        ordered_data = {feature: input_data[feature] for feature in feature_order}
        
        # 转换为DataFrame格式
        df_input = pd.DataFrame([ordered_data])
        
        # 确保列名和顺序与训练时一致
        print(f"特征顺序: {list(df_input.columns)}")
        
        # 模型预测
        predicted_crop = model.predict(df_input)[0]
        
        return {
            "predicted_crop": predicted_crop,
            "input_data": input_data
        }
    except Exception as e:
        print(f"预测失败: {str(e)}")
        return None

def generate_advice(prediction_result):
    """调用大模型生成种植建议"""
    if not prediction_result:
        return "无法生成建议，预测失败。"
    
    input_data = prediction_result["input_data"]
    predicted_crop = prediction_result["predicted_crop"]
    
    # 构造提示词
    prompt = f"""
你是一位农业专家，请参考下面输入的土质参数以及模型给出的推荐结果，向用户给出专业的种植建议：

1. 适合种植哪种作物
2. 基于每一个输入参数的对模型生成的推荐结果分别给出详细的理由和解释
3. 种植周期与注意事项
4. 浇水建议（频率、方式）
5. 施肥建议（种类、用量、时间）

输入数据如下：
- 氮（Nitrogen）: {input_data["Nitrogen"]}
- 磷（Phosphorus）: {input_data["Phosphorus"]}
- 钾（Potassium）: {input_data["Potassium"]}
- 温度: {input_data["temperature"]}℃
- 湿度: {input_data["humidity"]}%
- pH值: {input_data["ph"]}
- 降水量: {input_data["rainfall"]} mm

推荐作物：{predicted_crop}
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

def main():
    print("获取传感器数据...")
    api_data = get_sensor_data()
    
    if not api_data:
        print("无法获取传感器数据，程序终止。")
        return
    
    print("提取模型输入数据...")
    model_inputs = extract_model_inputs(api_data)
    
    if not model_inputs:
        print("无法提取有效的模型输入数据，程序终止。")
        return
    
    print("模型输入数据:")
    for key, value in model_inputs.items():
        print(f"- {key}: {value}")
    
    print("\n预测作物...")
    prediction_result = predict_crop(model_inputs)
    
    if not prediction_result:
        print("预测失败，程序终止。")
        return
    
    print(f"\n✅ 预测结果: {prediction_result['predicted_crop']}")
    
    print("\n生成农业建议...")
    advice = generate_advice(prediction_result)
    
    print("\n========== 种植建议 ==========")
    print(advice)
    print("==============================")

if __name__ == "__main__":
    main() 