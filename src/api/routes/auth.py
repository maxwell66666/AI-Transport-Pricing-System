"""
认证路由处理器

本模块提供用户认证相关的API路由，包括用户登录、注册、获取用户信息、API密钥管理等。
"""

from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from src.api.auth import (
    Token, UserCreate, UserInfo, APIKeyCreate, APIKeyInfo,
    authenticate_user, create_access_token, get_password_hash,
    get_current_active_user, get_current_admin_user
)
from src.config import settings
from src.db.database import get_db
from src.db.models import User
from src.db.repository import UserRepository, APIKeyRepository
from src.utils.logging import get_logger

# 创建路由器
router = APIRouter(prefix="/auth", tags=["认证"])

# 日志记录器
logger = get_logger(__name__)


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    用户登录获取访问令牌
    
    Args:
        form_data: OAuth2密码请求表单
        db: 数据库会话
        
    Returns:
        访问令牌
        
    Raises:
        HTTPException: 认证失败
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    
    if not user:
        logger.warning(f"用户登录失败: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    token_data = {
        "sub": user.username,
        "user_id": user.id,
        "is_admin": user.is_admin
    }
    
    access_token = create_access_token(
        data=token_data,
        expires_delta=access_token_expires
    )
    
    logger.info(f"用户登录成功: {user.username}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user_id": user.id,
        "username": user.username,
        "is_admin": user.is_admin
    }


@router.post("/register", response_model=UserInfo, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    用户注册
    
    Args:
        user_data: 用户注册数据
        db: 数据库会话
        
    Returns:
        用户信息
        
    Raises:
        HTTPException: 用户名或邮箱已存在
    """
    user_repo = UserRepository(db)
    
    # 检查用户名是否已存在
    if user_repo.get_by_username(user_data.username):
        logger.warning(f"用户注册失败，用户名已存在: {user_data.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已存在"
        )
    
    # 检查邮箱是否已存在
    if user_repo.get_by_email(user_data.email):
        logger.warning(f"用户注册失败，邮箱已存在: {user_data.email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="邮箱已存在"
        )
    
    # 创建用户
    hashed_password = get_password_hash(user_data.password)
    
    user = user_repo.create({
        "username": user_data.username,
        "email": user_data.email,
        "password": hashed_password,
        "full_name": user_data.full_name,
        "is_admin": user_data.is_admin,
        "is_active": True
    })
    
    logger.info(f"用户注册成功: {user.username}")
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "is_admin": user.is_admin,
        "created_at": user.created_at
    }


@router.get("/me", response_model=UserInfo)
async def get_user_me(current_user: User = Depends(get_current_active_user)):
    """
    获取当前用户信息
    
    Args:
        current_user: 当前用户
        
    Returns:
        用户信息
    """
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "is_active": current_user.is_active,
        "is_admin": current_user.is_admin,
        "created_at": current_user.created_at
    }


@router.get("/users", response_model=List[UserInfo])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    获取用户列表（仅管理员）
    
    Args:
        skip: 跳过记录数
        limit: 限制记录数
        current_user: 当前用户（管理员）
        db: 数据库会话
        
    Returns:
        用户列表
    """
    user_repo = UserRepository(db)
    users = user_repo.get_all(skip=skip, limit=limit)
    
    return [
        {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "is_admin": user.is_admin,
            "created_at": user.created_at
        }
        for user in users
    ]


@router.post("/api-keys", response_model=APIKeyInfo, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    api_key_data: APIKeyCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    创建API密钥
    
    Args:
        api_key_data: API密钥创建数据
        current_user: 当前用户
        db: 数据库会话
        
    Returns:
        API密钥信息
    """
    api_key_repo = APIKeyRepository(db)
    
    # 生成API密钥
    import secrets
    api_key = secrets.token_urlsafe(32)
    
    # 创建API密钥
    key = api_key_repo.create({
        "user_id": current_user.id,
        "name": api_key_data.name,
        "key": api_key,
        "is_active": True,
        "expires_at": api_key_data.expires_at
    })
    
    logger.info(f"用户 {current_user.username} 创建API密钥: {key.name}")
    
    return {
        "id": key.id,
        "user_id": key.user_id,
        "name": key.name,
        "key": key.key,
        "is_active": key.is_active,
        "created_at": key.created_at,
        "expires_at": key.expires_at
    }


@router.get("/api-keys", response_model=List[APIKeyInfo])
async def get_api_keys(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    获取当前用户的API密钥列表
    
    Args:
        current_user: 当前用户
        db: 数据库会话
        
    Returns:
        API密钥列表
    """
    api_key_repo = APIKeyRepository(db)
    keys = api_key_repo.get_by_user_id(current_user.id)
    
    return [
        {
            "id": key.id,
            "user_id": key.user_id,
            "name": key.name,
            "key": key.key,
            "is_active": key.is_active,
            "created_at": key.created_at,
            "expires_at": key.expires_at
        }
        for key in keys
    ]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    删除API密钥
    
    Args:
        key_id: API密钥ID
        current_user: 当前用户
        db: 数据库会话
        
    Raises:
        HTTPException: API密钥不存在或不属于当前用户
    """
    api_key_repo = APIKeyRepository(db)
    key = api_key_repo.get_by_id(key_id)
    
    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API密钥不存在"
        )
    
    if key.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权删除此API密钥"
        )
    
    api_key_repo.delete(key_id)
    logger.info(f"用户 {current_user.username} 删除API密钥: {key.name}")
    
    return None 