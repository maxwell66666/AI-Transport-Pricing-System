#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
异常处理模块

此模块定义了系统中使用的自定义异常类和异常处理函数。
"""

from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.utils.logging import log_error


class BaseAppException(Exception):
    """基础应用异常类"""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "internal_error",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationException(BaseAppException):
    """验证异常"""
    
    def __init__(
        self,
        message: str = "数据验证失败",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="validation_error",
            details=details
        )


class AuthenticationException(BaseAppException):
    """认证异常"""
    
    def __init__(
        self,
        message: str = "认证失败",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="authentication_error",
            details=details
        )


class AuthorizationException(BaseAppException):
    """授权异常"""
    
    def __init__(
        self,
        message: str = "没有权限执行此操作",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="authorization_error",
            details=details
        )


class NotFoundException(BaseAppException):
    """资源不存在异常"""
    
    def __init__(
        self,
        message: str = "请求的资源不存在",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="not_found",
            details=details
        )


class BadRequestException(BaseAppException):
    """错误请求异常"""
    
    def __init__(
        self,
        message: str = "请求参数错误",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="bad_request",
            details=details
        )


class ConflictException(BaseAppException):
    """资源冲突异常"""
    
    def __init__(
        self,
        message: str = "资源冲突",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code="conflict",
            details=details
        )


class RateLimitException(BaseAppException):
    """速率限制异常"""
    
    def __init__(
        self,
        message: str = "请求频率超过限制",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="rate_limit_exceeded",
            details=details
        )


class DatabaseException(BaseAppException):
    """数据库异常"""
    
    def __init__(
        self,
        message: str = "数据库操作失败",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="database_error",
            details=details
        )


class ExternalServiceException(BaseAppException):
    """外部服务异常"""
    
    def __init__(
        self,
        message: str = "外部服务调用失败",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code="external_service_error",
            details=details
        )


class LLMException(BaseAppException):
    """LLM异常"""
    
    def __init__(
        self,
        message: str = "LLM服务调用失败",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code="llm_error",
            details=details
        )


class MLException(BaseAppException):
    """机器学习异常"""
    
    def __init__(
        self,
        message: str = "机器学习服务调用失败",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="ml_error",
            details=details
        )


def format_validation_errors(errors: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    格式化验证错误
    
    Args:
        errors: 验证错误列表
        
    Returns:
        Dict[str, List[str]]: 格式化后的验证错误
    """
    formatted_errors: Dict[str, List[str]] = {}
    
    for error in errors:
        loc = error.get("loc", [])
        field = ".".join(str(item) for item in loc)
        msg = error.get("msg", "")
        
        if field not in formatted_errors:
            formatted_errors[field] = []
        
        formatted_errors[field].append(msg)
    
    return formatted_errors


def setup_exception_handlers(app: FastAPI) -> None:
    """
    设置异常处理器
    
    Args:
        app: FastAPI应用实例
    """
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """处理HTTP异常"""
        # 记录错误日志
        log_error(
            error=exc,
            module="api",
            function="http_exception_handler",
            request_id=request.headers.get("X-Request-ID"),
            extra={"path": request.url.path, "method": request.method}
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"http_{exc.status_code}",
                    "message": exc.detail,
                }
            },
            headers=exc.headers
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """处理请求验证异常"""
        # 记录错误日志
        log_error(
            error=exc,
            module="api",
            function="validation_exception_handler",
            request_id=request.headers.get("X-Request-ID"),
            extra={"path": request.url.path, "method": request.method}
        )
        
        # 格式化验证错误
        errors = format_validation_errors(exc.errors())
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "请求数据验证失败",
                    "details": errors
                }
            }
        )
    
    @app.exception_handler(ValidationError)
    async def pydantic_validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
        """处理Pydantic验证异常"""
        # 记录错误日志
        log_error(
            error=exc,
            module="api",
            function="pydantic_validation_exception_handler",
            request_id=request.headers.get("X-Request-ID"),
            extra={"path": request.url.path, "method": request.method}
        )
        
        # 格式化验证错误
        errors = format_validation_errors(exc.errors())
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "数据验证失败",
                    "details": errors
                }
            }
        )
    
    @app.exception_handler(BaseAppException)
    async def app_exception_handler(request: Request, exc: BaseAppException) -> JSONResponse:
        """处理应用异常"""
        # 记录错误日志
        log_error(
            error=exc,
            module="api",
            function="app_exception_handler",
            request_id=request.headers.get("X-Request-ID"),
            extra={
                "path": request.url.path,
                "method": request.method,
                "error_code": exc.error_code,
                "details": exc.details
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """处理通用异常"""
        # 记录错误日志
        log_error(
            error=exc,
            module="api",
            function="general_exception_handler",
            request_id=request.headers.get("X-Request-ID"),
            extra={"path": request.url.path, "method": request.method}
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "服务器内部错误"
                }
            }
        )


class BaseError(Exception):
    """基础异常类"""
    def __init__(self, message: str = None):
        self.message = message
        super().__init__(self.message)

class ConfigError(BaseError):
    """配置错误"""
    pass

class DatabaseError(BaseError):
    """数据库错误"""
    pass

class APIError(BaseError):
    """API错误"""
    pass

class AuthenticationError(BaseError):
    """认证错误"""
    pass

class AuthorizationError(BaseError):
    """授权错误"""
    pass

class ValidationError(BaseError):
    """数据验证错误"""
    pass

class MLModelError(BaseError):
    """机器学习模型错误"""
    pass

class DataProcessingError(BaseError):
    """数据处理错误"""
    pass

class LLMError(BaseError):
    """LLM服务错误"""
    pass

class ExternalServiceError(BaseError):
    """外部服务错误"""
    pass

class ResourceNotFoundError(BaseError):
    """资源未找到错误"""
    pass

class ResourceExistsError(BaseError):
    """资源已存在错误"""
    pass

class FileOperationError(BaseError):
    """文件操作错误"""
    pass

class NetworkError(BaseError):
    """网络错误"""
    pass

class CacheError(BaseError):
    """缓存错误"""
    pass

class EmailError(BaseError):
    """邮件发送错误"""
    pass

class RateLimitError(BaseError):
    """速率限制错误"""
    pass

class BusinessLogicError(BaseError):
    """业务逻辑错误"""
    pass 