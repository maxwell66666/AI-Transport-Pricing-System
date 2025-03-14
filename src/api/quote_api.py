"""
报价API模块

本模块提供报价相关的API接口，包括创建报价、获取报价历史、分析报价等功能。
使用FastAPI框架实现RESTful API。
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from src.db.database import get_db
from src.db.models import Quote, QuoteDetail
from src.db.repository import QuoteRepository, QuoteDetailRepository
from src.core.pricing_engine import PricingEngine
from src.core.llm_service import LLMService
from src.core.ml_service import MLService
from src.data.data_processor import DataProcessor

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(
    prefix="/api/quotes",
    tags=["quotes"],
    responses={404: {"description": "Not found"}},
)


# 请求模型
class QuoteRequest(BaseModel):
    """报价请求模型"""
    origin_location_id: int = Field(..., description="起始地点ID")
    destination_location_id: int = Field(..., description="目的地点ID")
    transport_mode_id: int = Field(..., description="运输方式ID")
    cargo_type_id: int = Field(..., description="货物类型ID")
    weight: float = Field(..., description="重量(kg)")
    volume: float = Field(..., description="体积(m³)")
    special_requirements: Optional[str] = Field(None, description="特殊要求")
    use_ml: bool = Field(False, description="是否使用机器学习模型")
    use_llm: bool = Field(False, description="是否使用LLM分析")
    currency: str = Field("USD", description="货币")


# 响应模型
class QuoteResponse(BaseModel):
    """报价响应模型"""
    id: Optional[int] = Field(None, description="报价ID")
    origin_location_id: int = Field(..., description="起始地点ID")
    destination_location_id: int = Field(..., description="目的地点ID")
    transport_mode_id: int = Field(..., description="运输方式ID")
    cargo_type_id: int = Field(..., description="货物类型ID")
    weight: float = Field(..., description="重量(kg)")
    volume: float = Field(..., description="体积(m³)")
    distance: float = Field(..., description="距离(km)")
    typical_transit_time: float = Field(..., description="典型运输时间(天)")
    total_price: float = Field(..., description="总价")
    currency: str = Field(..., description="货币")
    quote_date: datetime = Field(..., description="报价日期")
    expected_delivery_date: str = Field(..., description="预计交付日期")
    special_requirements: Optional[str] = Field(None, description="特殊要求")
    is_llm_assisted: bool = Field(False, description="是否使用LLM辅助")
    details: List[Dict[str, Any]] = Field(..., description="报价明细")
    explanation: Optional[str] = Field(None, description="报价解释")
    origin_location: Dict[str, Any] = Field(..., description="起始地点信息")
    destination_location: Dict[str, Any] = Field(..., description="目的地点信息")
    transport_mode: Dict[str, Any] = Field(..., description="运输方式信息")
    cargo_type: Dict[str, Any] = Field(..., description="货物类型信息")


class QuoteListResponse(BaseModel):
    """报价列表响应模型"""
    total: int = Field(..., description="总数")
    items: List[QuoteResponse] = Field(..., description="报价列表")


@router.post("/", response_model=QuoteResponse, status_code=201)
async def create_quote(
    quote_request: QuoteRequest,
    db: Session = Depends(get_db)
):
    """
    创建新的运输报价
    
    根据提供的参数计算运输报价，并将结果保存到数据库
    """
    try:
        # 创建服务实例
        data_processor = DataProcessor(db)
        pricing_engine = PricingEngine(db)
        
        # 获取距离和运输时间
        distance, transit_time = data_processor.get_distance_and_transit_time(
            quote_request.origin_location_id,
            quote_request.destination_location_id,
            quote_request.transport_mode_id
        )
        
        # 计算报价
        pricing_result = pricing_engine.calculate_price(
            origin_location_id=quote_request.origin_location_id,
            destination_location_id=quote_request.destination_location_id,
            transport_mode_id=quote_request.transport_mode_id,
            cargo_type_id=quote_request.cargo_type_id,
            weight=quote_request.weight,
            volume=quote_request.volume,
            special_requirements=quote_request.special_requirements,
            use_ml=quote_request.use_ml,
            use_llm=quote_request.use_llm
        )
        
        # 创建报价记录
        quote_repo = QuoteRepository(db)
        quote_detail_repo = QuoteDetailRepository(db)
        
        # 准备报价数据
        quote_data = {
            "origin_location_id": quote_request.origin_location_id,
            "destination_location_id": quote_request.destination_location_id,
            "transport_mode_id": quote_request.transport_mode_id,
            "cargo_type_id": quote_request.cargo_type_id,
            "weight": quote_request.weight,
            "volume": quote_request.volume,
            "distance": pricing_result["distance"],
            "typical_transit_time": pricing_result["transit_time"],
            "total_price": pricing_result["total_price"],
            "currency": quote_request.currency,
            "quote_date": datetime.utcnow(),
            "special_requirements": quote_request.special_requirements,
            "is_llm_assisted": quote_request.use_llm
        }
        
        # 保存报价
        quote = quote_repo.create(quote_data)
        
        # 保存报价明细
        for detail in pricing_result["details"]:
            detail_data = {
                "quote_id": quote.id,
                "fee_type": detail["fee_type"],
                "amount": detail["amount"],
                "currency": detail["currency"],
                "description": detail["description"]
            }
            quote_detail_repo.create(detail_data)
        
        # 如果使用LLM，生成报价解释
        explanation = None
        if quote_request.use_llm:
            llm_service = LLMService(db)
            
            # 准备完整的报价数据
            quote_data = quote_repo.get_quote_with_details(quote.id)
            
            # 添加相关实体信息
            locations_df = data_processor.load_locations()
            transport_modes_df = data_processor.load_transport_modes()
            cargo_types_df = data_processor.load_cargo_types()
            
            # 获取起始地点信息
            origin_location = locations_df[locations_df['id'] == quote_request.origin_location_id].to_dict('records')[0] if not locations_df.empty else {}
            destination_location = locations_df[locations_df['id'] == quote_request.destination_location_id].to_dict('records')[0] if not locations_df.empty else {}
            
            # 获取运输方式信息
            transport_mode = transport_modes_df[transport_modes_df['id'] == quote_request.transport_mode_id].to_dict('records')[0] if not transport_modes_df.empty else {}
            
            # 获取货物类型信息
            cargo_type = cargo_types_df[cargo_types_df['id'] == quote_request.cargo_type_id].to_dict('records')[0] if not cargo_types_df.empty else {}
            
            # 计算预计交付日期
            quote_date = datetime.utcnow()
            expected_delivery_date = (quote_date + timedelta(days=float(pricing_result["transit_time"]))).strftime('%Y-%m-%d')
            
            # 准备完整的报价数据
            complete_quote_data = {
                "id": quote.id,
                "origin_location_id": quote_request.origin_location_id,
                "destination_location_id": quote_request.destination_location_id,
                "transport_mode_id": quote_request.transport_mode_id,
                "cargo_type_id": quote_request.cargo_type_id,
                "weight": quote_request.weight,
                "volume": quote_request.volume,
                "distance": pricing_result["distance"],
                "transit_time": pricing_result["transit_time"],
                "total_price": pricing_result["total_price"],
                "currency": quote_request.currency,
                "quote_date": quote_date.strftime('%Y-%m-%d'),
                "expected_delivery_date": expected_delivery_date,
                "special_requirements": quote_request.special_requirements,
                "is_llm_assisted": True,
                "details": pricing_result["details"],
                "origin_location": origin_location,
                "destination_location": destination_location,
                "transport_mode": transport_mode,
                "cargo_type": cargo_type
            }
            
            # 生成报价解释
            explanation = llm_service.explain_quote(complete_quote_data)
        
        # 构建响应
        response = {
            "id": quote.id,
            "origin_location_id": quote.origin_location_id,
            "destination_location_id": quote.destination_location_id,
            "transport_mode_id": quote.transport_mode_id,
            "cargo_type_id": quote.cargo_type_id,
            "weight": quote.weight,
            "volume": quote.volume,
            "distance": quote.distance,
            "typical_transit_time": quote.typical_transit_time,
            "total_price": quote.total_price,
            "currency": quote.currency,
            "quote_date": quote.quote_date,
            "expected_delivery_date": (quote.quote_date + timedelta(days=float(quote.typical_transit_time))).strftime('%Y-%m-%d'),
            "special_requirements": quote.special_requirements,
            "is_llm_assisted": quote.is_llm_assisted,
            "details": pricing_result["details"],
            "explanation": explanation,
            "origin_location": origin_location,
            "destination_location": destination_location,
            "transport_mode": transport_mode,
            "cargo_type": cargo_type
        }
        
        return response
    
    except Exception as e:
        logger.error(f"创建报价失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建报价失败: {str(e)}")


@router.get("/{quote_id}", response_model=QuoteResponse)
async def get_quote(
    quote_id: int = Path(..., description="报价ID"),
    db: Session = Depends(get_db)
):
    """
    获取指定ID的报价详情
    """
    try:
        quote_repo = QuoteRepository(db)
        quote = quote_repo.get_by_id(quote_id)
        
        if not quote:
            raise HTTPException(status_code=404, detail=f"报价ID {quote_id} 不存在")
        
        # 获取报价明细
        quote_with_details = quote_repo.get_quote_with_details(quote_id)
        
        # 获取相关实体信息
        data_processor = DataProcessor(db)
        locations_df = data_processor.load_locations()
        transport_modes_df = data_processor.load_transport_modes()
        cargo_types_df = data_processor.load_cargo_types()
        
        # 获取起始地点信息
        origin_location = locations_df[locations_df['id'] == quote.origin_location_id].to_dict('records')[0] if not locations_df.empty else {}
        destination_location = locations_df[locations_df['id'] == quote.destination_location_id].to_dict('records')[0] if not locations_df.empty else {}
        
        # 获取运输方式信息
        transport_mode = transport_modes_df[transport_modes_df['id'] == quote.transport_mode_id].to_dict('records')[0] if not transport_modes_df.empty else {}
        
        # 获取货物类型信息
        cargo_type = cargo_types_df[cargo_types_df['id'] == quote.cargo_type_id].to_dict('records')[0] if not cargo_types_df.empty else {}
        
        # 计算预计交付日期
        expected_delivery_date = (quote.quote_date + timedelta(days=float(quote.typical_transit_time))).strftime('%Y-%m-%d')
        
        # 构建响应
        response = {
            "id": quote.id,
            "origin_location_id": quote.origin_location_id,
            "destination_location_id": quote.destination_location_id,
            "transport_mode_id": quote.transport_mode_id,
            "cargo_type_id": quote.cargo_type_id,
            "weight": quote.weight,
            "volume": quote.volume,
            "distance": quote.distance,
            "typical_transit_time": quote.typical_transit_time,
            "total_price": quote.total_price,
            "currency": quote.currency,
            "quote_date": quote.quote_date,
            "expected_delivery_date": expected_delivery_date,
            "special_requirements": quote.special_requirements,
            "is_llm_assisted": quote.is_llm_assisted,
            "details": quote_with_details.get("details", []),
            "explanation": None,  # 这里可以添加报价解释，如果需要
            "origin_location": origin_location,
            "destination_location": destination_location,
            "transport_mode": transport_mode,
            "cargo_type": cargo_type
        }
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取报价失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取报价失败: {str(e)}")


@router.get("/", response_model=QuoteListResponse)
async def list_quotes(
    skip: int = Query(0, description="跳过的记录数"),
    limit: int = Query(10, description="返回的记录数"),
    db: Session = Depends(get_db)
):
    """
    获取报价列表
    """
    try:
        quote_repo = QuoteRepository(db)
        quotes = quote_repo.get_all(skip=skip, limit=limit)
        total = quote_repo.count()
        
        # 获取相关实体信息
        data_processor = DataProcessor(db)
        locations_df = data_processor.load_locations()
        transport_modes_df = data_processor.load_transport_modes()
        cargo_types_df = data_processor.load_cargo_types()
        
        # 构建响应
        items = []
        for quote in quotes:
            # 获取报价明细
            quote_with_details = quote_repo.get_quote_with_details(quote.id)
            
            # 获取起始地点信息
            origin_location = locations_df[locations_df['id'] == quote.origin_location_id].to_dict('records')[0] if not locations_df.empty else {}
            destination_location = locations_df[locations_df['id'] == quote.destination_location_id].to_dict('records')[0] if not locations_df.empty else {}
            
            # 获取运输方式信息
            transport_mode = transport_modes_df[transport_modes_df['id'] == quote.transport_mode_id].to_dict('records')[0] if not transport_modes_df.empty else {}
            
            # 获取货物类型信息
            cargo_type = cargo_types_df[cargo_types_df['id'] == quote.cargo_type_id].to_dict('records')[0] if not cargo_types_df.empty else {}
            
            # 计算预计交付日期
            expected_delivery_date = (quote.quote_date + timedelta(days=float(quote.typical_transit_time))).strftime('%Y-%m-%d')
            
            items.append({
                "id": quote.id,
                "origin_location_id": quote.origin_location_id,
                "destination_location_id": quote.destination_location_id,
                "transport_mode_id": quote.transport_mode_id,
                "cargo_type_id": quote.cargo_type_id,
                "weight": quote.weight,
                "volume": quote.volume,
                "distance": quote.distance,
                "typical_transit_time": quote.typical_transit_time,
                "total_price": quote.total_price,
                "currency": quote.currency,
                "quote_date": quote.quote_date,
                "expected_delivery_date": expected_delivery_date,
                "special_requirements": quote.special_requirements,
                "is_llm_assisted": quote.is_llm_assisted,
                "details": quote_with_details.get("details", []),
                "explanation": None,
                "origin_location": origin_location,
                "destination_location": destination_location,
                "transport_mode": transport_mode,
                "cargo_type": cargo_type
            })
        
        return {"total": total, "items": items}
    
    except Exception as e:
        logger.error(f"获取报价列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取报价列表失败: {str(e)}")


@router.post("/{quote_id}/explain", response_model=Dict[str, Any])
async def explain_quote(
    quote_id: int = Path(..., description="报价ID"),
    db: Session = Depends(get_db)
):
    """
    为指定的报价生成解释
    """
    try:
        quote_repo = QuoteRepository(db)
        quote = quote_repo.get_by_id(quote_id)
        
        if not quote:
            raise HTTPException(status_code=404, detail=f"报价ID {quote_id} 不存在")
        
        # 获取报价明细
        quote_with_details = quote_repo.get_quote_with_details(quote_id)
        
        # 获取相关实体信息
        data_processor = DataProcessor(db)
        locations_df = data_processor.load_locations()
        transport_modes_df = data_processor.load_transport_modes()
        cargo_types_df = data_processor.load_cargo_types()
        
        # 获取起始地点信息
        origin_location = locations_df[locations_df['id'] == quote.origin_location_id].to_dict('records')[0] if not locations_df.empty else {}
        destination_location = locations_df[locations_df['id'] == quote.destination_location_id].to_dict('records')[0] if not locations_df.empty else {}
        
        # 获取运输方式信息
        transport_mode = transport_modes_df[transport_modes_df['id'] == quote.transport_mode_id].to_dict('records')[0] if not transport_modes_df.empty else {}
        
        # 获取货物类型信息
        cargo_type = cargo_types_df[cargo_types_df['id'] == quote.cargo_type_id].to_dict('records')[0] if not cargo_types_df.empty else {}
        
        # 计算预计交付日期
        expected_delivery_date = (quote.quote_date + timedelta(days=float(quote.typical_transit_time))).strftime('%Y-%m-%d')
        
        # 准备完整的报价数据
        complete_quote_data = {
            "id": quote.id,
            "origin_location_id": quote.origin_location_id,
            "destination_location_id": quote.destination_location_id,
            "transport_mode_id": quote.transport_mode_id,
            "cargo_type_id": quote.cargo_type_id,
            "weight": quote.weight,
            "volume": quote.volume,
            "distance": quote.distance,
            "transit_time": quote.typical_transit_time,
            "total_price": quote.total_price,
            "currency": quote.currency,
            "quote_date": quote.quote_date.strftime('%Y-%m-%d'),
            "expected_delivery_date": expected_delivery_date,
            "special_requirements": quote.special_requirements,
            "is_llm_assisted": True,
            "details": quote_with_details.get("details", []),
            "origin_location": origin_location,
            "destination_location": destination_location,
            "transport_mode": transport_mode,
            "cargo_type": cargo_type
        }
        
        # 生成报价解释
        llm_service = LLMService(db)
        explanation = llm_service.explain_quote(complete_quote_data)
        
        return {"quote_id": quote_id, "explanation": explanation}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成报价解释失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成报价解释失败: {str(e)}")


@router.post("/{quote_id}/analyze", response_model=Dict[str, Any])
async def analyze_quote(
    quote_id: int = Path(..., description="报价ID"),
    db: Session = Depends(get_db)
):
    """
    分析指定的报价并提供建议
    """
    try:
        quote_repo = QuoteRepository(db)
        quote = quote_repo.get_by_id(quote_id)
        
        if not quote:
            raise HTTPException(status_code=404, detail=f"报价ID {quote_id} 不存在")
        
        # 获取报价明细
        quote_with_details = quote_repo.get_quote_with_details(quote_id)
        
        # 获取相关实体信息
        data_processor = DataProcessor(db)
        locations_df = data_processor.load_locations()
        transport_modes_df = data_processor.load_transport_modes()
        cargo_types_df = data_processor.load_cargo_types()
        
        # 获取起始地点信息
        origin_location = locations_df[locations_df['id'] == quote.origin_location_id].to_dict('records')[0] if not locations_df.empty else {}
        destination_location = locations_df[locations_df['id'] == quote.destination_location_id].to_dict('records')[0] if not locations_df.empty else {}
        
        # 获取运输方式信息
        transport_mode = transport_modes_df[transport_modes_df['id'] == quote.transport_mode_id].to_dict('records')[0] if not transport_modes_df.empty else {}
        
        # 获取货物类型信息
        cargo_type = cargo_types_df[cargo_types_df['id'] == quote.cargo_type_id].to_dict('records')[0] if not cargo_types_df.empty else {}
        
        # 查找相似报价
        similar_quotes = data_processor.find_similar_quotes(
            origin_id=quote.origin_location_id,
            destination_id=quote.destination_location_id,
            transport_mode_id=quote.transport_mode_id,
            cargo_type_id=quote.cargo_type_id,
            weight=quote.weight,
            volume=quote.volume,
            limit=5
        )
        
        # 准备价格明细
        price_details = quote_with_details.get("details", [])
        
        # 准备输入数据
        pricing_input = {
            "origin_location_id": quote.origin_location_id,
            "destination_location_id": quote.destination_location_id,
            "transport_mode_id": quote.transport_mode_id,
            "cargo_type_id": quote.cargo_type_id,
            "weight": quote.weight,
            "volume": quote.volume,
            "distance": quote.distance,
            "transit_time": quote.typical_transit_time,
            "origin_location": origin_location,
            "destination_location": destination_location,
            "transport_mode": transport_mode,
            "cargo_type": cargo_type,
            "special_requirements": quote.special_requirements
        }
        
        # 分析报价
        llm_service = LLMService(db)
        analysis_result = llm_service.analyze_quote(
            pricing_input=pricing_input,
            current_price=quote.total_price,
            price_details=price_details,
            similar_quotes=similar_quotes
        )
        
        if not analysis_result:
            analysis_result = {
                "analysis": "无法生成报价分析",
                "adjusted_price": quote.total_price,
                "adjustment_reason": "无法确定调整原因"
            }
        
        return {
            "quote_id": quote_id,
            "analysis": analysis_result.get("analysis", ""),
            "current_price": quote.total_price,
            "adjusted_price": analysis_result.get("adjusted_price", quote.total_price),
            "adjustment_reason": analysis_result.get("adjustment_reason", ""),
            "currency": quote.currency
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分析报价失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析报价失败: {str(e)}")


@router.get("/stats/summary", response_model=Dict[str, Any])
async def get_quote_stats(
    db: Session = Depends(get_db)
):
    """
    获取报价统计信息
    """
    try:
        quote_repo = QuoteRepository(db)
        
        # 获取总报价数
        total_quotes = quote_repo.count()
        
        # 获取最近30天的报价数
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_quotes = quote_repo.count_by_date_range(start_date=thirty_days_ago)
        
        # 获取平均价格
        avg_price = quote_repo.get_average_price()
        
        # 获取最高和最低价格
        max_price = quote_repo.get_max_price()
        min_price = quote_repo.get_min_price()
        
        # 获取按运输方式分组的报价数量
        quotes_by_mode = quote_repo.count_by_transport_mode()
        
        # 获取按货物类型分组的报价数量
        quotes_by_cargo = quote_repo.count_by_cargo_type()
        
        # 获取LLM辅助的报价数量
        llm_assisted_quotes = quote_repo.count_llm_assisted()
        
        return {
            "total_quotes": total_quotes,
            "recent_quotes": recent_quotes,
            "avg_price": avg_price,
            "max_price": max_price,
            "min_price": min_price,
            "quotes_by_mode": quotes_by_mode,
            "quotes_by_cargo": quotes_by_cargo,
            "llm_assisted_quotes": llm_assisted_quotes
        }
    
    except Exception as e:
        logger.error(f"获取报价统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取报价统计信息失败: {str(e)}") 