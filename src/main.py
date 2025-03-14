"""
主应用模块

此模块是应用程序的入口点，负责创建和配置FastAPI应用实例。
"""

import logging
import os
import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# 使用绝对导入
from src.api.v1.api import api_router
from src.utils.config import settings
from src.routes.quotes import router as quotes_router

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="AI运输定价系统",
    description="智能运输定价和报价管理系统API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 允许前端访问
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册API路由
app.include_router(api_router, prefix="/api/v1")
app.include_router(quotes_router, prefix="/api")


@app.get("/", tags=["root"])
async def root():
    """
    根路径响应
    
    返回应用基本信息。
    """
    return {
        "message": "欢迎使用AI运输报价系统API",
        "version": "0.1.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.get("/health", tags=["health"])
async def health_check():
    """
    健康检查
    
    返回应用健康状态。
    """
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理器
    
    捕获并处理所有未捕获的异常。
    """
    logger.error(f"未捕获的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请稍后再试或联系管理员。"},
    )


# 如果直接运行此文件，则启动应用
if __name__ == "__main__":
    # 获取端口，默认为8000
    port = int(os.getenv("PORT", 8000))
    
    # 启动服务器
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
    )
    
    logger.info(f"应用已启动，访问地址: http://localhost:{port}") 