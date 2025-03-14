#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数模块

此模块包含各种工具函数，用于支持系统的各个部分。
"""

import hashlib
import json
import os
import random
import re
import string
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytz
from fastapi import HTTPException, status

from src.utils.config import settings


def generate_uuid() -> str:
    """
    生成UUID字符串
    
    Returns:
        str: UUID字符串
    """
    return str(uuid.uuid4())


def generate_request_id() -> str:
    """
    生成请求ID
    
    Returns:
        str: 请求ID
    """
    return str(uuid.uuid4())


def generate_random_string(length: int = 8) -> str:
    """
    生成随机字符串
    
    Args:
        length: 字符串长度
        
    Returns:
        str: 随机字符串
    """
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def generate_api_key() -> str:
    """
    生成API密钥
    
    Returns:
        str: API密钥
    """
    return f"sk_{generate_random_string(32)}"


def hash_password(password: str) -> str:
    """
    对密码进行哈希处理
    
    Args:
        password: 原始密码
        
    Returns:
        str: 哈希后的密码
    """
    import bcrypt
    
    # 生成盐值
    salt = bcrypt.gensalt()
    
    # 对密码进行哈希
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码
    
    Args:
        plain_password: 原始密码
        hashed_password: 哈希后的密码
        
    Returns:
        bool: 密码是否匹配
    """
    import bcrypt
    
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )


def get_current_timestamp() -> int:
    """
    获取当前时间戳
    
    Returns:
        int: 当前时间戳（秒）
    """
    return int(datetime.now().timestamp())


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    格式化日期时间
    
    Args:
        dt: 日期时间对象
        format_str: 格式字符串
        
    Returns:
        str: 格式化后的日期时间字符串
    """
    return dt.strftime(format_str)


def parse_datetime(dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """
    解析日期时间字符串
    
    Args:
        dt_str: 日期时间字符串
        format_str: 格式字符串
        
    Returns:
        datetime: 日期时间对象
    """
    return datetime.strptime(dt_str, format_str)


def get_now(timezone: Optional[str] = None) -> datetime:
    """
    获取当前时间
    
    Args:
        timezone: 时区，默认为配置中的时区
        
    Returns:
        datetime: 当前时间
    """
    tz = pytz.timezone(timezone or settings.TIMEZONE)
    return datetime.now(tz)


def add_days(dt: datetime, days: int) -> datetime:
    """
    添加天数
    
    Args:
        dt: 日期时间对象
        days: 天数
        
    Returns:
        datetime: 新的日期时间对象
    """
    return dt + timedelta(days=days)


def json_serial(obj: Any) -> Any:
    """
    JSON序列化函数，用于处理datetime、date和Decimal类型
    
    Args:
        obj: 要序列化的对象
        
    Returns:
        Any: 可序列化的对象
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def to_json(obj: Any) -> str:
    """
    将对象转换为JSON字符串
    
    Args:
        obj: 要转换的对象
        
    Returns:
        str: JSON字符串
    """
    return json.dumps(obj, default=json_serial)


def from_json(json_str: str) -> Any:
    """
    将JSON字符串转换为对象
    
    Args:
        json_str: JSON字符串
        
    Returns:
        Any: 转换后的对象
    """
    return json.loads(json_str)


def round_decimal(value: Union[float, Decimal], places: int = 2) -> Decimal:
    """
    四舍五入小数
    
    Args:
        value: 要四舍五入的值
        places: 小数位数
        
    Returns:
        Decimal: 四舍五入后的值
    """
    if isinstance(value, float):
        value = Decimal(str(value))
    return value.quantize(Decimal(f"0.{'0' * places}"))


def format_currency(amount: Union[float, Decimal], currency: str = "CNY") -> str:
    """
    格式化货币
    
    Args:
        amount: 金额
        currency: 货币代码
        
    Returns:
        str: 格式化后的货币字符串
    """
    if isinstance(amount, float):
        amount = Decimal(str(amount))
    
    # 四舍五入到2位小数
    amount = round_decimal(amount, 2)
    
    # 根据货币代码格式化
    if currency == "CNY":
        return f"¥{amount:,.2f}"
    elif currency == "USD":
        return f"${amount:,.2f}"
    elif currency == "EUR":
        return f"€{amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def calculate_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    计算两点之间的距离（公里）
    
    使用Haversine公式计算两个坐标点之间的距离
    
    Args:
        lat1: 第一个点的纬度
        lon1: 第一个点的经度
        lat2: 第二个点的纬度
        lon2: 第二个点的经度
        
    Returns:
        float: 距离（公里）
    """
    from math import asin, cos, radians, sin, sqrt
    
    # 将经纬度转换为弧度
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球半径（公里）
    
    return c * r


def validate_email(email: str) -> bool:
    """
    验证电子邮件地址
    
    Args:
        email: 电子邮件地址
        
    Returns:
        bool: 是否有效
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """
    验证手机号码
    
    Args:
        phone: 手机号码
        
    Returns:
        bool: 是否有效
    """
    # 简单的中国手机号码验证
    pattern = r'^1[3-9]\d{9}$'
    return bool(re.match(pattern, phone))


def get_file_extension(filename: str) -> str:
    """
    获取文件扩展名
    
    Args:
        filename: 文件名
        
    Returns:
        str: 文件扩展名
    """
    return os.path.splitext(filename)[1].lower()


def is_valid_file_type(filename: str, allowed_types: List[str]) -> bool:
    """
    检查文件类型是否有效
    
    Args:
        filename: 文件名
        allowed_types: 允许的文件类型列表
        
    Returns:
        bool: 是否有效
    """
    ext = get_file_extension(filename)
    return ext in allowed_types


def raise_http_exception(
    status_code: int = status.HTTP_400_BAD_REQUEST,
    detail: str = "Bad Request",
    headers: Optional[Dict[str, str]] = None
) -> None:
    """
    抛出HTTP异常
    
    Args:
        status_code: HTTP状态码
        detail: 详细信息
        headers: 响应头
        
    Raises:
        HTTPException: HTTP异常
    """
    raise HTTPException(
        status_code=status_code,
        detail=detail,
        headers=headers
    )


def paginate(
    items: List[Any],
    page: int = 1,
    page_size: int = 10
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    分页函数
    
    Args:
        items: 要分页的项目列表
        page: 页码
        page_size: 每页大小
        
    Returns:
        Tuple[List[Any], Dict[str, Any]]: 分页后的项目列表和分页信息
    """
    total_items = len(items)
    total_pages = (total_items + page_size - 1) // page_size
    
    # 确保页码有效
    page = max(1, min(page, total_pages)) if total_pages > 0 else 1
    
    # 计算起始和结束索引
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_items)
    
    # 获取当前页的项目
    paginated_items = items[start_idx:end_idx]
    
    # 分页信息
    pagination = {
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_previous": page > 1,
        "has_next": page < total_pages,
    }
    
    return paginated_items, pagination 