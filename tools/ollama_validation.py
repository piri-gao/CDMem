# from openai import OpenAI
#
# client = OpenAI(
#     base_url="http://172.30.112.1:8001/v1",
#     api_key="ollama",
# )
#
#
# result = client.chat.completions.create(
#     messages=[
#         {'role': 'user', 'content': "Say this is not a test"},
#     ],
#     model="qwen2"
# )
#
# print(result)
# print('\n\n------------ response ------------\n\n')
# print(result.choices)#[0].message.content)

import requests
import json

# 定义请求的URL
url = "http://172.30.112.1:8001/api/generate"

# 定义请求的数据
# data = {
#     "model": "qwen2",
#     "prompt": "Why is the sky blue? Answer with one sentence.",
#     "stream": False,
#     "temperature": 0.1
# }

data = {
            "model": "qwen2",
            "prompt": "Why is the sky blue? Answer with one sentence.",
            "max_tokens": 1000,
            "temperature": 0.1,
            "stop_strs": ['he', 'she', 'him', 'his'],
            "top_p": 1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": False
        }

# 将数据转换为JSON格式
json_data = json.dumps(data)

# 设置请求头
headers = {
    "Content-Type": "application/json"
}

# 发送POST请求
response = requests.post(url, headers=headers, data=json_data)

# 打印响应内容
# print(response.status_code)
json_str = response.json()
print(json_str['response'])
