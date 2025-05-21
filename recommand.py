import pandas as pd
import joblib
import requests

# 1. 加载模型进行作物预测
model = joblib.load("crop_model.pkl")

# 2. 定义输入数据
input_data = {
    "Nitrogen": 100.0,
    "Phosphorus": 70.0,
    "Potassium": 80.0,
    "temperature": 35.0,
    "humidity": 75.0,
    "ph": 7.0,
    "rainfall": 100.0
}

df_input = pd.DataFrame([input_data])
predicted_crop = model.predict(df_input)[0]

# 3. 构造 LLM 请求文本
llm_prompt = f"""
你是一位农业专家，请参考下面输入的土质参数以及模型给出的推荐结果，向用户给出专业的种植建议：

1. 适合种植哪种作物
2. 基于每一个输入参数的对模型生成的推荐结果分别给出详细的理由和解释
2. 种植周期与注意事项
3. 浇水建议（频率、方式）
4. 施肥建议（种类、用量、时间）

输入数据如下：
- 氮（Nitrogen）: {input_data['Nitrogen']}
- 磷（Phosphorus）: {input_data['Phosphorus']}
- 钾（Potassium）: {input_data['Potassium']}
- 温度: {input_data['temperature']}℃
- 湿度: {input_data['humidity']}%
- pH值: {input_data['ph']}
- 降水量: {input_data['rainfall']} mm

推荐作物：{predicted_crop}
"""

# 4. 请求 Qwen/QwQ-7B 接口生成建议
url = "https://api.siliconflow.cn/v1/chat/completions"
payload = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": llm_prompt
        }
    ],
    "stream": False,
    "max_tokens": 512,
    "enable_thinking": False,
    "thinking_budget": 4096,
    "min_p": 0.05,
    "stop": None,
    "temperature": 0.1,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "response_format": {"type": "text"},
    "tools": [
        {
            "type": "function",
            "function": {
                "description": "<string>",
                "name": "<string>",
                "parameters": {},
                "strict": False
            }
        }
    ]
}

headers = {
    "Authorization": "Bearer sk-jyjwpeexxccslfnudqqvwdffkcikuwnjhduorariyqxpzhww",  # 替换成你的真实token
    "Content-Type": "application/json"
}

# 5. 发送请求
response = requests.post(url, json=payload, headers=headers).json()
advice = response['choices'][0]['message']['content']

# 6. 输出结果
print(f"\n✅ 推荐种植作物：{predicted_crop}")

print(advice)

