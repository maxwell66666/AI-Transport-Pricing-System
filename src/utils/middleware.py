#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中间件模块

此模块定义了FastAPI应用中使用的中间件，用于处理请求和响应。
"""

import time
import uuid
from typing import Callable, Dict, Optional

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

from src.utils.config import settings
from src.utils.logging import log_request


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    请求ID中间件
    
    为每个请求添加唯一的请求ID，用于跟踪请求。
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求
        
        Args:
            request: 请求对象
            call_next: 下一个中间件或路由处理函数
            
        Returns:
            Response: 响应对象
        """
        # 生成请求ID或使用请求头中的请求ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # 记录请求开始
        start_time = time.time()
        
        # 处理请求
        try:
            response = await call_next(request)
            
            # 计算处理时间
            process_time = (time.time() - start_time) * 1000
            
            # 记录请求日志
            log_request(
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                processing_time=process_time,
                client_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("User-Agent")
            )
            
            # 添加请求ID到响应头
            response.headers["X-Request-ID"] = request_id
            
            return response
        except Exception as e:
            # 计算处理时间
            process_time = (time.time() - start_time) * 1000
            
            # 记录请求异常
            log_request(
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                status_code=500,
                processing_time=process_time,
                client_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("User-Agent"),
                error=str(e)
            )
            
            # 重新抛出异常
            raise


class ResponseTimeMiddleware(BaseHTTPMiddleware):
    """
    响应时间中间件
    
    添加响应时间头，用于性能监控。
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求
        
        Args:
            request: 请求对象
            call_next: 下一个中间件或路由处理函数
            
        Returns:
            Response: 响应对象
        """
        # 记录开始时间
        start_time = time.time()
        
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = (time.time() - start_time) * 1000
        
        # 添加处理时间到响应头
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        
        return response


def setup_middlewares(app: FastAPI) -> None:
    """
    设置中间件
    
    Args:
        app: FastAPI应用实例
    """
    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 添加请求ID中间件
    app.add_middleware(RequestIdMiddleware)
    
    # 添加响应时间中间件
    app.add_middleware(ResponseTimeMiddleware) 