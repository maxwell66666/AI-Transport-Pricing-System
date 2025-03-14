"""
数据库连接和会话管理模块

本模块负责创建和管理数据库连接，提供数据库会话工厂和依赖注入函数。
支持多种数据库后端，包括PostgreSQL、SQLite和测试用的内存数据库。
"""

import os
from typing import Generator, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from src.config import settings

# 创建数据库引擎
if settings.TESTING:
    # 测试环境使用SQLite内存数据库
    SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, 
        connect_args={"check_same_thread": False}
    )
else:
    # 生产环境使用PostgreSQL
    SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_timeout=30,  # 默认超时时间
        pool_recycle=settings.DATABASE_POOL_RECYCLE,
        pool_pre_ping=True,
    )

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    提供数据库会话的依赖注入函数
    
    用于FastAPI的Depends依赖注入系统，确保会话在请求结束后正确关闭
    
    Yields:
        Session: SQLAlchemy会话对象
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    提供数据库会话的上下文管理器
    
    用于with语句，确保会话在代码块结束后正确关闭
    
    Yields:
        Session: SQLAlchemy会话对象
    
    Example:
        ```python
        with get_db_context() as db:
            db.query(User).all()
        ```
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    初始化数据库
    
    创建所有表和初始数据
    """
    from src.db.models import Base
    
    # 创建所有表
    Base.metadata.create_all(bind=engine)
    
    # 如果需要，可以在这里添加初始数据的创建


def close_db_connection() -> None:
    """
    关闭数据库连接
    
    在应用程序关闭时调用
    """
    if engine is not None:
        engine.dispose() 