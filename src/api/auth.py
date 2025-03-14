"""
用户认证模块

本模块提供用户认证和授权功能，包括JWT令牌生成和验证、密码哈希等。
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.config import settings
from src.db.database import get_db
from src.db.models import User, APIKey
from src.db.repository import UserRepository, APIKeyRepository

# 密码上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2密码Bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")


# 令牌模型
class Token(BaseModel):
    """令牌模型"""
    access_token: str
    token_type: str
    expires_in: int
    user_id: int
    username: str
    is_admin: bool


# 令牌数据模型
class TokenData(BaseModel):
    """令牌数据模型"""
    username: Optional[str] = None
    user_id: Optional[int] = None
    is_admin: bool = False


# 用户登录模型
class UserLogin(BaseModel):
    """用户登录模型"""
    username: str
    password: str


# 用户注册模型
class UserCreate(BaseModel):
    """用户注册模型"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = Field(None, max_length=100)
    is_admin: bool = False


# 用户信息模型
class UserInfo(BaseModel):
    """用户信息模型"""
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_admin: bool
    created_at: datetime


# API密钥创建模型
class APIKeyCreate(BaseModel):
    """API密钥创建模型"""
    name: str = Field(..., min_length=1, max_length=100)
    expires_at: Optional[datetime] = None


# API密钥信息模型
class APIKeyInfo(BaseModel):
    """API密钥信息模型"""
    id: int
    user_id: int
    name: str
    key: str
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime] = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码
    
    Args:
        plain_password: 明文密码
        hashed_password: 哈希密码
        
    Returns:
        密码是否匹配
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    获取密码哈希
    
    Args:
        password: 明文密码
        
    Returns:
        哈希密码
    """
    return pwd_context.hash(password)


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """
    认证用户
    
    Args:
        db: 数据库会话
        username: 用户名
        password: 密码
        
    Returns:
        认证成功返回用户对象，否则返回None
    """
    user_repo = UserRepository(db)
    user = user_repo.get_by_username(username)
    
    if not user:
        return None
    
    if not verify_password(password, user.password):
        return None
    
    return user


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    创建访问令牌
    
    Args:
        data: 令牌数据
        expires_delta: 过期时间增量
        
    Returns:
        JWT令牌
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    return encoded_jwt


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    获取当前用户
    
    Args:
        token: JWT令牌
        db: 数据库会话
        
    Returns:
        当前用户对象
        
    Raises:
        HTTPException: 认证失败
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username = payload.get("sub")
        user_id = payload.get("user_id")
        
        if username is None or user_id is None:
            raise credentials_exception
        
        token_data = TokenData(username=username, user_id=user_id, is_admin=payload.get("is_admin", False))
    
    except JWTError:
        raise credentials_exception
    
    user_repo = UserRepository(db)
    user = user_repo.get_by_id(token_data.user_id)
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户已被禁用"
        )
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    获取当前活跃用户
    
    Args:
        current_user: 当前用户
        
    Returns:
        当前活跃用户
        
    Raises:
        HTTPException: 用户未激活
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="用户未激活")
    return current_user


async def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """
    获取当前管理员用户
    
    Args:
        current_user: 当前用户
        
    Returns:
        当前管理员用户
        
    Raises:
        HTTPException: 用户不是管理员
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足，需要管理员权限"
        )
    return current_user


def verify_api_key(api_key: str, db: Session) -> Optional[User]:
    """
    验证API密钥
    
    Args:
        api_key: API密钥
        db: 数据库会话
        
    Returns:
        验证成功返回用户对象，否则返回None
    """
    api_key_repo = APIKeyRepository(db)
    key = api_key_repo.get_by_key(api_key)
    
    if not key:
        return None
    
    if not key.is_active:
        return None
    
    if key.expires_at and key.expires_at < datetime.utcnow():
        return None
    
    user_repo = UserRepository(db)
    user = user_repo.get_by_id(key.user_id)
    
    if not user or not user.is_active:
        return None
    
    return user


async def get_api_key_user(
    api_key: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    获取API密钥对应的用户
    
    Args:
        api_key: API密钥
        db: 数据库会话
        
    Returns:
        API密钥对应的用户
        
    Raises:
        HTTPException: API密钥无效
    """
    user = verify_api_key(api_key, db)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的API密钥",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user 