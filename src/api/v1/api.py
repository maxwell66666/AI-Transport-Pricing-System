"""
API路由配置

此模块配置API路由，将各个端点注册到FastAPI应用中。
"""

from fastapi import APIRouter

from src.api.v1.endpoints import quotes

# 创建API路由器
api_router = APIRouter()

# 注册各个端点
api_router.include_router(quotes.router, prefix="/quotes", tags=["quotes"])

# 在这里可以添加更多的端点路由器
# 例如：
# api_router.include_router(users.router, prefix="/users", tags=["users"])
# api_router.include_router(auth.router, prefix="/auth", tags=["auth"]) 