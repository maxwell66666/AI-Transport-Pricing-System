"""
报价路由处理器

本模块提供运输报价相关的API路由，包括获取报价、创建报价、查询历史报价等。
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from src.api.auth import get_current_active_user, get_api_key_user
from src.config import settings
from src.core.pricing_engine import PricingEngine
from src.db.database import get_db
from src.db.models import User, Quote, QuoteDetail
from src.db.repository import (
    QuoteRepository, QuoteDetailRepository,
    TransportModeRepository, CargoTypeRepository,
    LocationRepository, DistanceMatrixRepository
)
from src.utils.logging import get_logger

# 创建路由器
router = APIRouter(prefix="/quotes", tags=["报价"])

# 日志记录器
logger = get_logger(__name__)


# 报价请求模型
class QuoteRequest(BaseModel):
    """报价请求模型"""
    origin_location_id: int = Field(..., description="起始地点ID")
    destination_location_id: int = Field(..., description="目的地点ID")
    transport_mode_id: int = Field(..., description="运输方式ID")
    cargo_type_id: int = Field(..., description="货物类型ID")
    weight: float = Field(..., ge=0, description="重量（千克）")
    volume: float = Field(..., ge=0, description="体积（立方米）")
    special_requirements: Optional[List[str]] = Field(None, description="特殊要求")
    use_ml: bool = Field(False, description="是否使用机器学习模型")
    use_llm: bool = Field(False, description="是否使用大语言模型分析")
    currency: str = Field(settings.DEFAULT_CURRENCY, description="货币")


# 报价详情模型
class QuoteDetailResponse(BaseModel):
    """报价详情模型"""
    id: int
    quote_id: int
    fee_type: str
    amount: float
    currency: str
    description: Optional[str] = None


# 报价响应模型
class QuoteResponse(BaseModel):
    """报价响应模型"""
    id: int
    origin_location_id: int
    origin_location_name: str
    destination_location_id: int
    destination_location_name: str
    transport_mode_id: int
    transport_mode_name: str
    cargo_type_id: int
    cargo_type_name: str
    weight: float
    volume: float
    distance: float
    typical_transit_time: int
    total_price: float
    currency: str
    quote_date: datetime
    expected_delivery_date: Optional[datetime] = None
    special_requirements: Optional[List[str]] = None
    is_ml_assisted: bool
    is_llm_assisted: bool
    details: List[QuoteDetailResponse]
    user_id: Optional[int] = None


# 报价历史查询参数模型
class QuoteHistoryParams(BaseModel):
    """报价历史查询参数模型"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    origin_location_id: Optional[int] = None
    destination_location_id: Optional[int] = None
    transport_mode_id: Optional[int] = None
    cargo_type_id: Optional[int] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None


@router.post("", response_model=QuoteResponse, status_code=status.HTTP_201_CREATED)
async def create_quote(
    quote_request: QuoteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    创建运输报价
    
    Args:
        quote_request: 报价请求
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        报价响应
        
    Raises:
        HTTPException: 请求参数无效或报价计算失败
    """
    # 验证请求参数
    transport_mode_repo = TransportModeRepository(db)
    cargo_type_repo = CargoTypeRepository(db)
    location_repo = LocationRepository(db)
    
    # 检查运输方式是否存在
    transport_mode = transport_mode_repo.get_by_id(quote_request.transport_mode_id)
    if not transport_mode:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"运输方式ID {quote_request.transport_mode_id} 不存在"
        )
    
    # 检查货物类型是否存在
    cargo_type = cargo_type_repo.get_by_id(quote_request.cargo_type_id)
    if not cargo_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"货物类型ID {quote_request.cargo_type_id} 不存在"
        )
    
    # 检查起始地点是否存在
    origin_location = location_repo.get_by_id(quote_request.origin_location_id)
    if not origin_location:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"起始地点ID {quote_request.origin_location_id} 不存在"
        )
    
    # 检查目的地点是否存在
    destination_location = location_repo.get_by_id(quote_request.destination_location_id)
    if not destination_location:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"目的地点ID {quote_request.destination_location_id} 不存在"
        )
    
    # 创建定价引擎
    pricing_engine = PricingEngine(db)
    
    try:
        # 计算报价
        price_result = pricing_engine.calculate_price(
            origin_location_id=quote_request.origin_location_id,
            destination_location_id=quote_request.destination_location_id,
            transport_mode_id=quote_request.transport_mode_id,
            cargo_type_id=quote_request.cargo_type_id,
            weight=quote_request.weight,
            volume=quote_request.volume,
            special_requirements=quote_request.special_requirements,
            use_ml=quote_request.use_ml and settings.ML_ENABLED,
            use_llm=quote_request.use_llm and settings.LLM_ENABLED,
            currency=quote_request.currency
        )
        
        # 创建报价记录
        quote_repo = QuoteRepository(db)
        quote_detail_repo = QuoteDetailRepository(db)
        
        # 计算预计交付日期
        expected_delivery_date = None
        if price_result.get("transit_time"):
            from datetime import timedelta
            expected_delivery_date = datetime.utcnow() + timedelta(days=price_result["transit_time"])
        
        # 创建报价
        quote = quote_repo.create({
            "user_id": current_user.id,
            "origin_location_id": quote_request.origin_location_id,
            "destination_location_id": quote_request.destination_location_id,
            "transport_mode_id": quote_request.transport_mode_id,
            "cargo_type_id": quote_request.cargo_type_id,
            "weight": quote_request.weight,
            "volume": quote_request.volume,
            "distance": price_result.get("distance", 0),
            "typical_transit_time": price_result.get("transit_time", 0),
            "total_price": price_result.get("total_price", 0),
            "currency": price_result.get("currency", quote_request.currency),
            "quote_date": datetime.utcnow(),
            "expected_delivery_date": expected_delivery_date,
            "special_requirements": quote_request.special_requirements,
            "is_ml_assisted": quote_request.use_ml and settings.ML_ENABLED,
            "is_llm_assisted": quote_request.use_llm and settings.LLM_ENABLED,
            "status": "active"
        })
        
        # 创建报价详情
        details = []
        for detail in price_result.get("details", []):
            quote_detail = quote_detail_repo.create({
                "quote_id": quote.id,
                "fee_type": detail.get("fee_type", "unknown"),
                "amount": detail.get("amount", 0),
                "currency": detail.get("currency", quote_request.currency),
                "description": detail.get("description")
            })
            details.append(quote_detail)
        
        # 构建响应
        response = {
            "id": quote.id,
            "origin_location_id": quote.origin_location_id,
            "origin_location_name": origin_location.name,
            "destination_location_id": quote.destination_location_id,
            "destination_location_name": destination_location.name,
            "transport_mode_id": quote.transport_mode_id,
            "transport_mode_name": transport_mode.name,
            "cargo_type_id": quote.cargo_type_id,
            "cargo_type_name": cargo_type.name,
            "weight": quote.weight,
            "volume": quote.volume,
            "distance": quote.distance,
            "typical_transit_time": quote.typical_transit_time,
            "total_price": quote.total_price,
            "currency": quote.currency,
            "quote_date": quote.quote_date,
            "expected_delivery_date": quote.expected_delivery_date,
            "special_requirements": quote.special_requirements,
            "is_ml_assisted": quote.is_ml_assisted,
            "is_llm_assisted": quote.is_llm_assisted,
            "details": [
                {
                    "id": detail.id,
                    "quote_id": detail.quote_id,
                    "fee_type": detail.fee_type,
                    "amount": detail.amount,
                    "currency": detail.currency,
                    "description": detail.description
                }
                for detail in details
            ],
            "user_id": quote.user_id
        }
        
        logger.info(f"用户 {current_user.username} 创建报价成功，ID: {quote.id}")
        
        return response
    
    except Exception as e:
        logger.error(f"报价计算失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"报价计算失败: {str(e)}"
        )


@router.get("/{quote_id}", response_model=QuoteResponse)
async def get_quote(
    quote_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    获取报价详情
    
    Args:
        quote_id: 报价ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        报价详情
        
    Raises:
        HTTPException: 报价不存在或无权访问
    """
    quote_repo = QuoteRepository(db)
    quote = quote_repo.get_by_id(quote_id)
    
    if not quote:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"报价ID {quote_id} 不存在"
        )
    
    # 检查权限
    if quote.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权访问此报价"
        )
    
    # 获取相关数据
    transport_mode_repo = TransportModeRepository(db)
    cargo_type_repo = CargoTypeRepository(db)
    location_repo = LocationRepository(db)
    quote_detail_repo = QuoteDetailRepository(db)
    
    transport_mode = transport_mode_repo.get_by_id(quote.transport_mode_id)
    cargo_type = cargo_type_repo.get_by_id(quote.cargo_type_id)
    origin_location = location_repo.get_by_id(quote.origin_location_id)
    destination_location = location_repo.get_by_id(quote.destination_location_id)
    details = quote_detail_repo.get_by_quote_id(quote.id)
    
    # 构建响应
    response = {
        "id": quote.id,
        "origin_location_id": quote.origin_location_id,
        "origin_location_name": origin_location.name,
        "destination_location_id": quote.destination_location_id,
        "destination_location_name": destination_location.name,
        "transport_mode_id": quote.transport_mode_id,
        "transport_mode_name": transport_mode.name,
        "cargo_type_id": quote.cargo_type_id,
        "cargo_type_name": cargo_type.name,
        "weight": quote.weight,
        "volume": quote.volume,
        "distance": quote.distance,
        "typical_transit_time": quote.typical_transit_time,
        "total_price": quote.total_price,
        "currency": quote.currency,
        "quote_date": quote.quote_date,
        "expected_delivery_date": quote.expected_delivery_date,
        "special_requirements": quote.special_requirements,
        "is_ml_assisted": quote.is_ml_assisted,
        "is_llm_assisted": quote.is_llm_assisted,
        "details": [
            {
                "id": detail.id,
                "quote_id": detail.quote_id,
                "fee_type": detail.fee_type,
                "amount": detail.amount,
                "currency": detail.currency,
                "description": detail.description
            }
            for detail in details
        ],
        "user_id": quote.user_id
    }
    
    return response


@router.get("", response_model=List[QuoteResponse])
async def get_quotes(
    skip: int = 0,
    limit: int = 100,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    origin_location_id: Optional[int] = None,
    destination_location_id: Optional[int] = None,
    transport_mode_id: Optional[int] = None,
    cargo_type_id: Optional[int] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    获取报价列表
    
    Args:
        skip: 跳过记录数
        limit: 限制记录数
        start_date: 开始日期
        end_date: 结束日期
        origin_location_id: 起始地点ID
        destination_location_id: 目的地点ID
        transport_mode_id: 运输方式ID
        cargo_type_id: 货物类型ID
        min_price: 最低价格
        max_price: 最高价格
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        报价列表
    """
    quote_repo = QuoteRepository(db)
    
    # 构建过滤条件
    filters = {}
    
    # 非管理员只能查看自己的报价
    if not current_user.is_admin:
        filters["user_id"] = current_user.id
    
    if start_date:
        filters["quote_date_gte"] = start_date
    
    if end_date:
        filters["quote_date_lte"] = end_date
    
    if origin_location_id:
        filters["origin_location_id"] = origin_location_id
    
    if destination_location_id:
        filters["destination_location_id"] = destination_location_id
    
    if transport_mode_id:
        filters["transport_mode_id"] = transport_mode_id
    
    if cargo_type_id:
        filters["cargo_type_id"] = cargo_type_id
    
    if min_price is not None:
        filters["total_price_gte"] = min_price
    
    if max_price is not None:
        filters["total_price_lte"] = max_price
    
    # 获取报价列表
    quotes = quote_repo.get_filtered(
        skip=skip,
        limit=limit,
        **filters
    )
    
    # 获取相关数据
    transport_mode_repo = TransportModeRepository(db)
    cargo_type_repo = CargoTypeRepository(db)
    location_repo = LocationRepository(db)
    quote_detail_repo = QuoteDetailRepository(db)
    
    # 预加载数据
    transport_modes = {tm.id: tm for tm in transport_mode_repo.get_all()}
    cargo_types = {ct.id: ct for ct in cargo_type_repo.get_all()}
    locations = {loc.id: loc for loc in location_repo.get_all()}
    
    # 构建响应
    response = []
    for quote in quotes:
        details = quote_detail_repo.get_by_quote_id(quote.id)
        
        response.append({
            "id": quote.id,
            "origin_location_id": quote.origin_location_id,
            "origin_location_name": locations.get(quote.origin_location_id, {}).name,
            "destination_location_id": quote.destination_location_id,
            "destination_location_name": locations.get(quote.destination_location_id, {}).name,
            "transport_mode_id": quote.transport_mode_id,
            "transport_mode_name": transport_modes.get(quote.transport_mode_id, {}).name,
            "cargo_type_id": quote.cargo_type_id,
            "cargo_type_name": cargo_types.get(quote.cargo_type_id, {}).name,
            "weight": quote.weight,
            "volume": quote.volume,
            "distance": quote.distance,
            "typical_transit_time": quote.typical_transit_time,
            "total_price": quote.total_price,
            "currency": quote.currency,
            "quote_date": quote.quote_date,
            "expected_delivery_date": quote.expected_delivery_date,
            "special_requirements": quote.special_requirements,
            "is_ml_assisted": quote.is_ml_assisted,
            "is_llm_assisted": quote.is_llm_assisted,
            "details": [
                {
                    "id": detail.id,
                    "quote_id": detail.quote_id,
                    "fee_type": detail.fee_type,
                    "amount": detail.amount,
                    "currency": detail.currency,
                    "description": detail.description
                }
                for detail in details
            ],
            "user_id": quote.user_id
        })
    
    return response


@router.delete("/{quote_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_quote(
    quote_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    删除报价
    
    Args:
        quote_id: 报价ID
        db: 数据库会话
        current_user: 当前用户
        
    Raises:
        HTTPException: 报价不存在或无权删除
    """
    quote_repo = QuoteRepository(db)
    quote = quote_repo.get_by_id(quote_id)
    
    if not quote:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"报价ID {quote_id} 不存在"
        )
    
    # 检查权限
    if quote.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权删除此报价"
        )
    
    # 删除报价详情
    quote_detail_repo = QuoteDetailRepository(db)
    details = quote_detail_repo.get_by_quote_id(quote.id)
    for detail in details:
        quote_detail_repo.delete(detail.id)
    
    # 删除报价
    quote_repo.delete(quote.id)
    
    logger.info(f"用户 {current_user.username} 删除报价，ID: {quote.id}")
    
    return None 