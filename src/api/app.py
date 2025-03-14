"""
主API应用程序

本模块创建FastAPI应用程序实例，配置中间件、路由和依赖项。
"""

import logging
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
import time
import uuid
from typing import Callable

from src.config import settings
from src.db.database import init_db, close_db_connection
from src.api.routes.quotes import router as quotes_router
from src.api.routes.auth import router as auth_router
from src.api.routes.base_data import router as base_data_router
from src.api.ml_routes import router as ml_router
from src.api.llm_routes import router as llm_router
from src.utils.logging import setup_logging, get_logger, RequestLogMiddleware

# 配置日志
setup_logging()
logger = get_logger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    docs_url=None,  # 禁用默认的Swagger UI路径
    redoc_url=None  # 禁用默认的ReDoc路径
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# 添加请求日志中间件
app.add_middleware(RequestLogMiddleware)

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行的操作"""
    logger.info("正在启动API服务器...")
    
    # 初始化数据库
    init_db()
    
    logger.info(f"API服务器启动完成。环境: {settings.ENVIRONMENT}")

# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行的操作"""
    logger.info("正在关闭API服务器...")
    
    # 关闭数据库连接
    close_db_connection()
    
    logger.info("API服务器关闭完成")

# 自定义Swagger UI路径
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """自定义Swagger UI页面"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API文档",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

# 健康检查端点
@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查端点，用于监控系统状态"""
    return {
        "status": "ok",
        "version": settings.PROJECT_VERSION,
        "environment": settings.ENVIRONMENT
    }

# 注册路由
app.include_router(
    auth_router,
    prefix=settings.API_PREFIX
)
app.include_router(
    quotes_router,
    prefix=settings.API_PREFIX
)
app.include_router(
    base_data_router,
    prefix=settings.API_PREFIX
)
app.include_router(
    ml_router,
    prefix=settings.API_PREFIX
)
app.include_router(
    llm_router,
    prefix=settings.API_PREFIX
)

# 挂载静态文件
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    logger.warning("静态文件目录不存在，跳过静态文件挂载")

# 根路径重定向到文档
@app.get("/", include_in_schema=False)
async def root():
    """根路径重定向到API文档"""
    return RedirectResponse(url="/docs")

# 全局异常处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """处理HTTP异常"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": getattr(request.state, "request_id", None)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """处理一般异常"""
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部错误",
            "request_id": getattr(request.state, "request_id", None)
        }
    ) 