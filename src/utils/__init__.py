#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具模块包

此包包含各种工具函数和类，用于支持系统的各个部分。
"""

from src.utils.config import Settings, get_settings, settings
from src.utils.exceptions import (
    AuthenticationException, AuthorizationException, BadRequestException,
    BaseAppException, ConflictException, DatabaseException, ExternalServiceException,
    LLMException, MLException, NotFoundException, RateLimitException,
    ValidationException, format_validation_errors, setup_exception_handlers
)
from src.utils.helpers import (
    add_days, calculate_distance, format_currency, format_datetime, from_json,
    generate_api_key, generate_random_string, generate_request_id, generate_uuid,
    get_current_timestamp, get_file_extension, get_now, hash_password,
    is_valid_file_type, json_serial, paginate, parse_datetime, raise_http_exception,
    round_decimal, to_json, validate_email, validate_phone, verify_password
)
from src.utils.logging import (
    get_logger, get_request_logger, log_error, log_request, setup_logging
)
from src.utils.middleware import (
    RequestIdMiddleware, ResponseTimeMiddleware, setup_middlewares
)

__all__ = [
    # 配置模块
    "Settings",
    "get_settings",
    "settings",
    
    # 日志模块
    "get_logger",
    "get_request_logger",
    "log_error",
    "log_request",
    "setup_logging",
    
    # 工具函数模块
    "generate_uuid",
    "generate_request_id",
    "generate_random_string",
    "generate_api_key",
    "hash_password",
    "verify_password",
    "get_current_timestamp",
    "format_datetime",
    "parse_datetime",
    "get_now",
    "add_days",
    "json_serial",
    "to_json",
    "from_json",
    "round_decimal",
    "format_currency",
    "calculate_distance",
    "validate_email",
    "validate_phone",
    "get_file_extension",
    "is_valid_file_type",
    "raise_http_exception",
    "paginate",
    
    # 异常处理模块
    "BaseAppException",
    "ValidationException",
    "AuthenticationException",
    "AuthorizationException",
    "NotFoundException",
    "BadRequestException",
    "ConflictException",
    "RateLimitException",
    "DatabaseException",
    "ExternalServiceException",
    "LLMException",
    "MLException",
    "format_validation_errors",
    "setup_exception_handlers",
    
    # 中间件模块
    "RequestIdMiddleware",
    "ResponseTimeMiddleware",
    "setup_middlewares",
] 