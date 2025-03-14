"""
测试API接口
"""
import requests
import json
import time

# 等待服务器启动
print("等待服务器启动...")
time.sleep(3)

# 测试根路径
print("\n测试根路径:")
try:
    response = requests.get("http://localhost:8000/")
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.json()}")
except Exception as e:
    print(f"请求失败: {e}")

# 测试健康检查
print("\n测试健康检查:")
try:
    response = requests.get("http://localhost:8000/health")
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.json()}")
except Exception as e:
    print(f"请求失败: {e}")

# 测试获取运输方式
print("\n测试获取运输方式:")
try:
    response = requests.get("http://localhost:8000/api/v1/quotes/transport-modes")
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.json()}")
except Exception as e:
    print(f"请求失败: {e}")

# 测试获取货物类型
print("\n测试获取货物类型:")
try:
    response = requests.get("http://localhost:8000/api/v1/quotes/cargo-types")
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.json()}")
except Exception as e:
    print(f"请求失败: {e}")

# 测试获取地点
print("\n测试获取地点:")
try:
    response = requests.get("http://localhost:8000/api/v1/quotes/locations")
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.json()}")
except Exception as e:
    print(f"请求失败: {e}")

# 测试价格预测
print("\n测试价格预测:")
try:
    data = {
        "origin_id": 1,
        "destination_id": 3,
        "transport_mode_id": 1,
        "cargo_type_id": 2,
        "weight": 500,
        "volume": 2.5,
        "special_requirements": ["fragile"]
    }
    response = requests.post(
        "http://localhost:8000/api/v1/quotes/predict",
        json=data
    )
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
except Exception as e:
    print(f"请求失败: {e}")

# 测试获取报价列表
print("\n测试获取报价列表:")
try:
    response = requests.get("http://localhost:8000/api/v1/quotes/quotes")
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
except Exception as e:
    print(f"请求失败: {e}")

print("\n测试完成!") 