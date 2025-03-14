#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置模块

此模块负责加载和管理系统配置，支持从环境变量和配置文件中读取配置。
使用Pydantic的BaseSettings进行配置验证和类型转换。
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache

from pydantic import AnyHttpUrl, Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """系统配置类"""
    
    PROJECT_NAME: str = "AI Transport Pricing System"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # BACKEND_CORS_ORIGINS is a comma-separated list of origins
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """Validate CORS origins."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database settings
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "ai_transport"
    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None

    @field_validator("SQLALCHEMY_DATABASE_URI", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info) -> Any:
        if isinstance(v, str):
            return v
        values = info.data
        return PostgresDsn.build(
            scheme="postgresql",
            username=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )

    # OpenAI settings
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_API_BASE: str = Field("https://api.ffa.chat/v1", env="OPENAI_API_BASE")
    OPENAI_MODEL: str = Field("gpt-4", env="OPENAI_MODEL")

    # 基本配置
    ENVIRONMENT: str = Field("development", description="运行环境: development, testing, production")
    DEBUG: bool = Field(True, description="调试模式")
    LOG_LEVEL: str = Field("INFO", description="日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    LOG_FILE: str = Field("logs/app.log", description="日志文件路径")
    API_HOST: str = Field("0.0.0.0", description="API主机")
    API_PORT: int = Field(8000, description="API端口")
    
    # 数据库配置
    DATABASE_URL: str = Field("sqlite:///./transport_pricing.db", description="数据库URL")
    DATABASE_POOL_SIZE: int = Field(5, description="数据库连接池大小")
    DATABASE_TIMEOUT: int = Field(30, description="数据库连接超时时间(秒)")
    DATABASE_AUTO_CREATE_TABLES: bool = Field(True, description="是否在启动时自动创建表")
    
    # 安全配置
    SECRET_KEY: str = Field("your-secret-key-change-in-production", description="密钥(用于JWT令牌加密等)")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(43200, description="访问令牌过期时间(分钟)")
    REFRESH_TOKEN_EXPIRE_MINUTES: int = Field(129600, description="刷新令牌过期时间(分钟)")
    PASSWORD_HASH_ALGORITHM: str = Field("bcrypt", description="密码哈希算法")
    CORS_ORIGINS: List[str] = Field(["*"], description="CORS允许的源")
    ENABLE_HTTPS: bool = Field(False, description="是否启用HTTPS")
    SSL_CERT_FILE: Optional[str] = Field(None, description="SSL证书路径(如果启用HTTPS)")
    SSL_KEY_FILE: Optional[str] = Field(None, description="SSL密钥路径(如果启用HTTPS)")
    
    # 缓存配置
    CACHE_TYPE: str = Field("memory", description="缓存类型: memory, redis")
    REDIS_URL: Optional[str] = Field(None, description="Redis URL(如果使用Redis缓存)")
    CACHE_EXPIRATION: int = Field(3600, description="缓存过期时间(秒)")
    
    # 机器学习配置
    ML_ENABLED: bool = Field(True, description="是否启用机器学习")
    ML_MODEL_PATH: str = Field("data/models/price_prediction_model.pkl", description="模型文件路径")
    ML_SCALER_PATH: str = Field("data/models/feature_scaler.pkl", description="特征缩放器路径")
    ML_MODEL_UPDATE_FREQUENCY: int = Field(7, description="模型更新频率(天)")
    
    # LLM配置
    LLM_ENABLED: bool = Field(False, description="是否启用LLM")
    LLM_PROVIDER: str = Field("openai", description="LLM提供商: openai, azure, local")
    LLM_API_KEY: Optional[str] = Field(None, description="LLM API密钥")
    LLM_MODEL: str = Field("gpt-3.5-turbo", description="LLM模型名称")
    LLM_API_BASE: Optional[str] = Field(None, description="LLM API基础URL(Azure或本地部署时使用)")
    LLM_API_VERSION: Optional[str] = Field(None, description="LLM API版本(Azure时使用)")
    LLM_MAX_TOKENS: int = Field(1000, description="LLM最大令牌数")
    LLM_TEMPERATURE: float = Field(0.7, description="LLM温度参数(0-2)")
    LLM_TIMEOUT: int = Field(30, description="LLM超时时间(秒)")
    
    # 邮件配置
    SMTP_SERVER: Optional[str] = Field(None, description="SMTP服务器")
    SMTP_PORT: int = Field(587, description="SMTP端口")
    SMTP_USERNAME: Optional[str] = Field(None, description="SMTP用户名")
    SMTP_PASSWORD: Optional[str] = Field(None, description="SMTP密码")
    EMAIL_FROM: Optional[str] = Field(None, description="发件人邮箱")
    EMAIL_USE_TLS: bool = Field(True, description="是否使用TLS")
    
    # 监控配置
    ENABLE_PROMETHEUS: bool = Field(False, description="是否启用Prometheus指标")
    PROMETHEUS_PATH: str = Field("/metrics", description="Prometheus指标路径")
    ENABLE_HEALTH_CHECK: bool = Field(True, description="是否启用健康检查")
    HEALTH_CHECK_PATH: str = Field("/health", description="健康检查路径")
    
    # 其他配置
    DEFAULT_CURRENCY: str = Field("CNY", description="默认货币")
    DEFAULT_LANGUAGE: str = Field("zh-CN", description="默认语言")
    TIMEZONE: str = Field("Asia/Shanghai", description="时区")
    ENABLE_SWAGGER: bool = Field(True, description="是否启用Swagger UI")
    ENABLE_REDOC: bool = Field(True, description="是否启用ReDoc")
    STATIC_FILES_DIR: str = Field("static", description="静态文件目录")
    UPLOAD_FILES_DIR: str = Field("uploads", description="上传文件目录")
    MAX_UPLOAD_SIZE: int = Field(10, description="最大上传文件大小(MB)")
    
    # 配置文件
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
        populate_by_name=True,
        validate_default=True,
        env_prefix=""
    )
    
    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """验证数据库URL，如果是相对路径的SQLite，转换为绝对路径"""
        if v.startswith("sqlite:///./"):
            db_path = v[13:]
            if not os.path.isabs(db_path):
                return f"sqlite:///{os.path.join(ROOT_DIR, db_path)}"
        return v
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def validate_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """验证CORS_ORIGINS，支持字符串和列表格式"""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, str):
            import json
            return json.loads(v)
        return v
    
    @field_validator("LOG_FILE", mode="before")
    @classmethod
    def validate_log_file(cls, v: str) -> str:
        """验证日志文件路径，转换为绝对路径"""
        if not os.path.isabs(v):
            return os.path.join(ROOT_DIR, v)
        return v
    
    @field_validator("ML_MODEL_PATH", "ML_SCALER_PATH", mode="before")
    @classmethod
    def validate_ml_paths(cls, v: str) -> str:
        """验证机器学习模型路径，转换为绝对路径"""
        if not os.path.isabs(v):
            return os.path.join(ROOT_DIR, v)
        return v
    
    @field_validator("STATIC_FILES_DIR", "UPLOAD_FILES_DIR", mode="before")
    @classmethod
    def validate_dir_paths(cls, v: str) -> str:
        """验证目录路径，转换为绝对路径"""
        if not os.path.isabs(v):
            return os.path.join(ROOT_DIR, v)
        return v
    
    def get_database_url(self, testing: bool = False) -> str:
        """获取数据库URL，支持测试环境使用内存数据库"""
        if testing:
            return "sqlite:///:memory:"
        return self.DATABASE_URL
    
    def get_jwt_settings(self) -> Dict[str, Any]:
        """获取JWT设置"""
        return {
            "secret_key": self.SECRET_KEY,
            "algorithm": "HS256",
            "access_token_expire_minutes": self.ACCESS_TOKEN_EXPIRE_MINUTES,
            "refresh_token_expire_minutes": self.REFRESH_TOKEN_EXPIRE_MINUTES,
        }
    
    def get_llm_settings(self) -> Dict[str, Any]:
        """获取LLM设置"""
        return {
            "enabled": self.LLM_ENABLED,
            "provider": self.LLM_PROVIDER,
            "api_key": self.LLM_API_KEY,
            "model": self.LLM_MODEL,
            "api_base": self.LLM_API_BASE,
            "api_version": self.LLM_API_VERSION,
            "max_tokens": self.LLM_MAX_TOKENS,
            "temperature": self.LLM_TEMPERATURE,
            "timeout": self.LLM_TIMEOUT,
        }
    
    def get_email_settings(self) -> Dict[str, Any]:
        """获取邮件设置"""
        return {
            "smtp_server": self.SMTP_SERVER,
            "smtp_port": self.SMTP_PORT,
            "smtp_username": self.SMTP_USERNAME,
            "smtp_password": self.SMTP_PASSWORD,
            "email_from": self.EMAIL_FROM,
            "use_tls": self.EMAIL_USE_TLS,
        }
    
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.ENVIRONMENT.lower() == "development"
    
    def is_testing(self) -> bool:
        """是否为测试环境"""
        return self.ENVIRONMENT.lower() == "testing"
    
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.ENVIRONMENT.lower() == "production"


# 创建全局设置实例
settings = Settings()


@lru_cache()
def get_settings() -> Settings:
    """获取设置实例，用于依赖注入"""
    return settings 