"""
配置模块

本模块提供系统配置参数，支持从环境变量、配置文件等多种方式加载配置。
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """系统配置类"""
    
    # 基本配置
    PROJECT_NAME: str = "AI运输报价系统"
    PROJECT_VERSION: str = "0.1.0"
    PROJECT_DESCRIPTION: str = "基于AI的智能运输报价系统"
    
    # 环境配置
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    TESTING: bool = Field(default=False, env="TESTING")
    
    # 路径配置
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = Field(default=Path(__file__).parent.parent / "data")
    LOG_DIR: Path = Field(default=Path(__file__).parent.parent / "logs")
    
    # 日志配置
    LOG_LEVEL: str = Field(default="info", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # 数据库配置
    DATABASE_URL: str = Field(
        default="sqlite:///./transport_pricing.db",
        env="DATABASE_URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=5, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")
    DATABASE_POOL_RECYCLE: int = Field(default=3600, env="DATABASE_POOL_RECYCLE")
    
    # API配置
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_PREFIX: str = Field(default="/api", env="API_PREFIX")
    
    # 安全配置
    SECRET_KEY: str = Field(
        default="09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7",
        env="SECRET_KEY"
    )
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30 * 24 * 60,  # 30天
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    
    # CORS配置
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        env="CORS_ORIGINS"
    )
    CORS_METHODS: List[str] = Field(
        default=["*"],
        env="CORS_METHODS"
    )
    CORS_HEADERS: List[str] = Field(
        default=["*"],
        env="CORS_HEADERS"
    )
    
    # 机器学习配置
    ML_MODEL_DIR: Path = Field(default=Path(__file__).parent.parent / "models")
    ML_DATA_DIR: Path = Field(default=Path(__file__).parent.parent / "data")
    ML_DEFAULT_MODEL: str = Field(default="gradient_boosting")
    ML_TRAIN_TEST_SPLIT: float = Field(default=0.2)
    ML_RANDOM_STATE: int = Field(default=42)
    
    # LLM配置
    LLM_ENABLED: bool = Field(default=False, env="LLM_ENABLED")
    LLM_API_KEY: Optional[str] = Field(default=None, env="LLM_API_KEY")
    LLM_API_URL: Optional[str] = Field(default=None, env="LLM_API_URL")
    LLM_MODEL: str = Field(default="gpt-3.5-turbo", env="LLM_MODEL")
    
    # 缓存配置
    CACHE_ENABLED: bool = Field(default=True, env="CACHE_ENABLED")
    CACHE_URL: Optional[str] = Field(default=None, env="CACHE_URL")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1小时
    
    # 其他配置
    DEFAULT_CURRENCY: str = Field(default="CNY", env="DEFAULT_CURRENCY")
    
    @validator("DATA_DIR", pre=True, always=True)
    def set_data_dir(cls, v, values):
        """设置数据目录"""
        return v or values["BASE_DIR"] / "data"
    
    @validator("LOG_DIR", pre=True, always=True)
    def set_log_dir(cls, v, values):
        """设置日志目录"""
        return v or values["BASE_DIR"] / "logs"
    
    @validator("LOG_FILE", pre=True, always=True)
    def set_log_file(cls, v, values):
        """设置日志文件路径"""
        if v:
            return v
        
        log_dir = values.get("LOG_DIR") or Path("logs")
        os.makedirs(log_dir, exist_ok=True)
        return str(log_dir / "app.log")
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """解析CORS来源"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("CORS_METHODS", pre=True)
    def parse_cors_methods(cls, v):
        """解析CORS方法"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [method.strip() for method in v.split(",")]
        return v
    
    @validator("CORS_HEADERS", pre=True)
    def parse_cors_headers(cls, v):
        """解析CORS头"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [header.strip() for header in v.split(",")]
        return v
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "allow",
        "populate_by_name": True,
        "validate_default": True,
        "env_prefix": ""
    }


# 创建全局设置实例
settings = Settings()


def get_settings() -> Settings:
    """
    获取设置实例
    
    Returns:
        设置实例
    """
    return settings


def load_config_from_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    从文件加载配置
    
    Args:
        file_path: 配置文件路径
        
    Returns:
        配置字典
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path}")
    
    if file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    elif file_path.suffix.lower() in [".yaml", ".yml"]:
        try:
            import yaml
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("使用YAML配置文件需要安装PyYAML库")
    
    else:
        raise ValueError(f"不支持的配置文件格式: {file_path.suffix}")


def update_settings(config_dict: Dict[str, Any]) -> None:
    """
    更新设置
    
    Args:
        config_dict: 配置字典
    """
    global settings
    
    for key, value in config_dict.items():
        if hasattr(settings, key):
            setattr(settings, key, value)


def load_and_update_settings(file_path: Union[str, Path]) -> None:
    """
    加载并更新设置
    
    Args:
        file_path: 配置文件路径
    """
    config_dict = load_config_from_file(file_path)
    update_settings(config_dict) 