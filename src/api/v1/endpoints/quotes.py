"""
报价API接口

此模块提供报价相关的API接口，包括报价预测、报价查询等功能。
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta

# 替换旧的MLService导入为新的模型和服务
from src.models.art_transport_price_model import ArtTransportPriceModel
from src.services.transport_decision_service import TransportDecisionService
from src.preprocessing.art_transport_preprocessor import ArtTransportPreprocessor

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

# 替换旧的MLService实例化为新的模型和服务
price_model = ArtTransportPriceModel()
decision_service = TransportDecisionService()
preprocessor = ArtTransportPreprocessor()

# 定义请求和响应模型
class QuoteRequest(BaseModel):
    """报价请求模型"""
    origin_location_id: int = Field(..., description="起始地点ID")
    destination_location_id: int = Field(..., description="目的地点ID")
    transport_mode_id: int = Field(..., description="运输方式ID")
    cargo_type_id: int = Field(..., description="货物类型ID")
    weight: float = Field(..., description="重量(kg)")
    volume: float = Field(..., description="体积(m³)")
    distance: Optional[float] = Field(None, description="距离(km)，如果不提供将自动计算")
    typical_transit_time: Optional[int] = Field(None, description="典型运输时间(天)，如果不提供将自动计算")


class QuoteResponse(BaseModel):
    """报价响应模型"""
    predicted_price: float = Field(..., description="预测价格")
    currency: str = Field("USD", description="货币")
    model_used: str = Field(..., description="使用的模型")
    confidence: Optional[float] = Field(None, description="预测置信度")
    price_breakdown: Optional[Dict[str, float]] = Field(None, description="价格明细")


class TransportMode(BaseModel):
    """运输方式模型"""
    id: int
    name: str
    description: Optional[str] = None


class CargoType(BaseModel):
    """货物类型模型"""
    id: int
    name: str
    description: Optional[str] = None
    hazardous: bool = False


class Location(BaseModel):
    """地点模型"""
    id: int
    name: str
    country: str
    city: str
    latitude: float
    longitude: float


class PriceRequest(BaseModel):
    origin_id: int = Field(..., description="起始地点ID")
    destination_id: int = Field(..., description="目的地ID")
    transport_mode_id: int = Field(..., description="运输方式ID")
    cargo_type_id: int = Field(..., description="货物类型ID")
    weight: float = Field(..., description="重量(kg)")
    volume: float = Field(..., description="体积(m³)")
    special_requirements: List[str] = Field(default=[], description="特殊要求")


class PricePrediction(BaseModel):
    predicted_price: float
    confidence_interval: List[float]
    factors: dict
    currency: str = "CNY"


class Quote(BaseModel):
    id: int
    customer_id: int
    origin_id: int
    destination_id: int
    transport_mode_id: int
    cargo_type_id: int
    weight: float
    volume: float
    price: float
    currency: str
    created_at: datetime
    valid_until: datetime
    status: str


@router.get("/transport-modes", response_model=List[TransportMode])
async def get_transport_modes():
    """
    获取所有运输方式
    """
    # 暂时返回模拟数据
    return [
        {"id": 1, "name": "海运", "description": "通过海洋运输货物"},
        {"id": 2, "name": "空运", "description": "通过航空运输货物"},
        {"id": 3, "name": "陆运", "description": "通过陆地运输货物"},
        {"id": 4, "name": "铁路", "description": "通过铁路运输货物"},
        {"id": 5, "name": "多式联运", "description": "结合多种运输方式"}
    ]


@router.get("/cargo-types", response_model=List[CargoType])
async def get_cargo_types():
    """
    获取所有货物类型
    """
    # 暂时返回模拟数据
    return [
        {"id": 1, "name": "普通货物", "description": "一般商品", "hazardous": False},
        {"id": 2, "name": "易碎品", "description": "需要特殊包装的易碎物品", "hazardous": False},
        {"id": 3, "name": "危险品", "description": "危险物品", "hazardous": True},
        {"id": 4, "name": "冷藏品", "description": "需要温控的物品", "hazardous": False},
        {"id": 5, "name": "大型设备", "description": "大型机械设备", "hazardous": False}
    ]


@router.get("/locations", response_model=List[Location])
async def get_locations(
    country: Optional[str] = None,
    city: Optional[str] = None,
    search: Optional[str] = None
):
    """
    获取地点列表，支持筛选
    """
    # 暂时返回模拟数据
    locations = [
        {"id": 1, "name": "上海港", "country": "中国", "city": "上海", "latitude": 31.2304, "longitude": 121.4737},
        {"id": 2, "name": "深圳港", "country": "中国", "city": "深圳", "latitude": 22.5431, "longitude": 114.0579},
        {"id": 3, "name": "鹿特丹港", "country": "荷兰", "city": "鹿特丹", "latitude": 51.9225, "longitude": 4.47917},
        {"id": 4, "name": "新加坡港", "country": "新加坡", "city": "新加坡", "latitude": 1.29027, "longitude": 103.851},
        {"id": 5, "name": "洛杉矶港", "country": "美国", "city": "洛杉矶", "latitude": 33.7701, "longitude": -118.1937}
    ]
    
    # 应用筛选
    if country:
        locations = [loc for loc in locations if loc["country"] == country]
    if city:
        locations = [loc for loc in locations if loc["city"] == city]
    if search:
        locations = [loc for loc in locations if search.lower() in loc["name"].lower()]
        
    return locations


@router.post("/predict", response_model=PricePrediction)
async def predict_price(request: PriceRequest):
    """
    预测运输价格
    """
    try:
        logger.info(f"接收到价格预测请求: {request}")
        
        # 计算体积重量（按照航空运输标准：1立方米 = 167千克）
        volume_weight = request.volume * 167
        
        # 确定计费重量（取实际重量和体积重量的较大值）
        chargeable_weight = max(request.weight, volume_weight)
        
        # 设置截止日期（当前日期 + 14天）
        current_date = datetime.now()
        deadline = current_date + timedelta(days=14)
        
        # 假设尺寸（根据体积估算）
        # 假设长:宽:高 = 2:1:1
        volume_in_cm3 = request.volume * 1000000  # 立方米转立方厘米
        height = (volume_in_cm3 / 2) ** (1/3)
        width = height
        length = height * 2
        
        # 根据货物类型确定艺术品类型
        artwork_type_map = {
            1: "painting",    # 普通货物 -> 绘画
            2: "sculpture",   # 易碎品 -> 雕塑
            3: "installation", # 危险品 -> 装置艺术
            4: "painting",    # 冷藏品 -> 绘画
            5: "sculpture"    # 大型设备 -> 雕塑
        }
        artwork_type = artwork_type_map.get(request.cargo_type_id, "painting")
        
        # 使用新的模型和服务进行预测
        # 准备输入数据
        input_data = {
            "Origin": request.origin_id,
            "Destination": request.destination_id,
            "Transport_Mode": request.transport_mode_id,
            "Cargo_Type": request.cargo_type_id,
            "Weight": request.weight,
            "Volume": request.volume,
            "Special_Requirements": request.special_requirements,
            "artwork_value": 10000.0,  # 默认艺术品价值
            "days_to_deadline": 14,    # 默认截止日期天数
            "is_international": request.origin_id != request.destination_id,  # 根据起始地和目的地判断是否国际运输
            "chargeable_weight": chargeable_weight,  # 计费重量
            "deadline": deadline,  # 截止日期
            "Length": length,  # 长度（厘米）
            "Width": width,    # 宽度（厘米）
            "Height": height,  # 高度（厘米）
            "Actual_Weight": request.weight,  # 实际重量（千克）
            "requires_climate_control": "temperature_control" in request.special_requirements,
            "requires_custom_crating": "custom_crating" in request.special_requirements,
            "requires_art_handler": "art_handler" in request.special_requirements,
            "Origin_Country": "CN",  # 默认起始国家为中国
            "Destination_Country": "NL" if request.destination_id == 3 else "SG" if request.destination_id == 4 else "US" if request.destination_id == 5 else "CN",  # 根据目的地ID设置国家代码
            "Artwork_Type": artwork_type  # 艺术品类型
        }
        
        # 使用决策服务生成完整推荐
        # 添加默认距离参数
        recommendation = decision_service.generate_full_recommendation(
            artwork_data=input_data,
            distance_km=1000.0  # 默认距离为1000公里
        )
        
        # 提取价格预测结果
        prediction = {
            "price": recommendation["price_prediction_details"]["base_price"],
            "confidence_interval": [
                recommendation["price_prediction_details"].get("confidence_interval", {}).get("lower", 0),
                recommendation["price_prediction_details"].get("confidence_interval", {}).get("upper", 0)
            ],
            "feature_importance": recommendation.get("feature_importance", {})
        }
        
        return {
            "predicted_price": prediction["price"],
            "confidence_interval": prediction["confidence_interval"],
            "factors": prediction["feature_importance"],
            "currency": "CNY"  # 默认使用人民币
        }
        
    except Exception as e:
        logger.error(f"价格预测失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"价格预测失败: {str(e)}")


@router.get("/quotes", response_model=List[Quote])
async def get_quotes(
    customer_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100)
):
    """
    获取报价列表，支持筛选
    """
    # 暂时返回模拟数据
    quotes = [
        {
            "id": 1,
            "customer_id": 101,
            "origin_id": 1,
            "destination_id": 3,
            "transport_mode_id": 1,
            "cargo_type_id": 1,
            "weight": 1000.0,
            "volume": 5.0,
            "price": 4500.0,
            "currency": "CNY",
            "created_at": datetime.now(),
            "valid_until": datetime(2023, 12, 31),
            "status": "active"
        },
        {
            "id": 2,
            "customer_id": 102,
            "origin_id": 2,
            "destination_id": 4,
            "transport_mode_id": 2,
            "cargo_type_id": 2,
            "weight": 500.0,
            "volume": 2.5,
            "price": 6000.0,
            "currency": "CNY",
            "created_at": datetime.now(),
            "valid_until": datetime(2023, 12, 31),
            "status": "active"
        }
    ]
    
    # 应用筛选
    if customer_id:
        quotes = [q for q in quotes if q["customer_id"] == customer_id]
    if status:
        quotes = [q for q in quotes if q["status"] == status]
        
    # 应用限制
    quotes = quotes[:limit]
    
    return quotes 