#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志模块

此模块负责配置和管理系统日志，提供统一的日志记录功能。
支持控制台和文件日志，以及不同级别的日志记录。
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, Optional, Union

from src.utils.config import settings


# 日志格式
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
)

# 日志级别映射
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console: bool = True,
    detailed: bool = False,
) -> None:
    """
    设置日志系统
    
    Args:
        log_level: 日志级别，默认从配置中读取
        log_file: 日志文件路径，默认从配置中读取
        log_format: 日志格式，默认根据detailed参数决定
        max_file_size: 单个日志文件最大大小，默认10MB
        backup_count: 保留的日志文件数量，默认5个
        console: 是否输出到控制台，默认True
        detailed: 是否使用详细日志格式，默认False
    """
    # 获取日志级别
    log_level = log_level or settings.LOG_LEVEL
    level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
    
    # 获取日志格式
    if log_format is None:
        log_format = DETAILED_LOG_FORMAT if detailed else DEFAULT_LOG_FORMAT
    
    # 获取日志文件路径
    log_file = log_file or settings.LOG_FILE
    
    # 创建日志目录
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter(log_format)
    
    # 添加控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_file:
        # 使用RotatingFileHandler，按大小轮转
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    # 记录初始日志
    logging.info(f"日志系统初始化完成，级别: {log_level}, 文件: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logging.Logger: 日志记录器实例
    """
    return logging.getLogger(name)


class RequestIdFilter(logging.Filter):
    """
    请求ID过滤器，用于在日志中添加请求ID
    """
    
    def __init__(self, request_id: str = ""):
        super().__init__()
        self.request_id = request_id
    
    def filter(self, record):
        record.request_id = self.request_id
        return True


def get_request_logger(request_id: str) -> logging.Logger:
    """
    获取带有请求ID的日志记录器
    
    Args:
        request_id: 请求ID
        
    Returns:
        logging.Logger: 日志记录器实例
    """
    logger = logging.getLogger(f"request.{request_id}")
    
    # 添加请求ID过滤器
    for handler in logger.handlers:
        handler.addFilter(RequestIdFilter(request_id))
    
    return logger


def log_request(
    request_id: str,
    method: str,
    url: str,
    status_code: int,
    processing_time: float,
    user_id: Optional[str] = None,
    client_ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """
    记录API请求日志
    
    Args:
        request_id: 请求ID
        method: HTTP方法
        url: 请求URL
        status_code: 响应状态码
        processing_time: 处理时间(毫秒)
        user_id: 用户ID
        client_ip: 客户端IP
        user_agent: 用户代理
        error: 错误信息
    """
    logger = get_request_logger(request_id)
    
    log_data = {
        "request_id": request_id,
        "method": method,
        "url": url,
        "status_code": status_code,
        "processing_time_ms": processing_time,
        "timestamp": datetime.now().isoformat(),
    }
    
    if user_id:
        log_data["user_id"] = user_id
    
    if client_ip:
        log_data["client_ip"] = client_ip
    
    if user_agent:
        log_data["user_agent"] = user_agent
    
    if error:
        log_data["error"] = error
        logger.error(f"API请求失败: {log_data}")
    else:
        logger.info(f"API请求完成: {log_data}")


def log_error(
    error: Union[str, Exception],
    module: str = "",
    function: str = "",
    request_id: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> None:
    """
    记录错误日志
    
    Args:
        error: 错误信息或异常对象
        module: 模块名称
        function: 函数名称
        request_id: 请求ID
        extra: 额外信息
    """
    if request_id:
        logger = get_request_logger(request_id)
    else:
        logger = logging.getLogger("error")
    
    error_message = str(error)
    error_type = type(error).__name__ if isinstance(error, Exception) else "Error"
    
    log_data = {
        "error_type": error_type,
        "error_message": error_message,
        "timestamp": datetime.now().isoformat(),
    }
    
    if module:
        log_data["module"] = module
    
    if function:
        log_data["function"] = function
    
    if request_id:
        log_data["request_id"] = request_id
    
    if extra:
        log_data.update(extra)
    
    logger.error(f"错误: {log_data}")


# 初始化日志系统
if __name__ != "__main__":  # 避免在导入时初始化
    setup_logging() 