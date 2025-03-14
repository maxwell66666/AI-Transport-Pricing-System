"""
机器学习服务API路由

此模块提供机器学习服务的API接口，包括价格预测、模型训练和分析功能。
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from src.ml.ml_service import MLService
from src.utils.exceptions import MLModelError, DataProcessingError
from src.db.database import get_db
from sqlalchemy.orm import Session
from src.utils.auth import get_current_user, get_current_active_user
from src.db.models import User
from src.config import settings

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(
    prefix="/api/ml",
    tags=["机器学习"],
    responses={404: {"description": "Not found"}},
)

# 创建MLService实例
ml_service = MLService()


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    健康检查端点
    
    Returns:
        包含服务状态和可用模型的字典
    """
    try:
        models = list(ml_service.models.keys())
        return {
            "status": "healthy",
            "available_models": models,
            "default_model": ml_service.default_model,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.post("/predict", response_model=Dict[str, Any])
async def predict_price(
    origin_id: int = Body(..., description="起始地点ID"),
    destination_id: int = Body(..., description="目的地ID"),
    transport_mode_id: int = Body(..., description="运输方式ID"),
    cargo_type_id: int = Body(..., description="货物类型ID"),
    weight: float = Body(..., description="重量（千克）"),
    volume: float = Body(..., description="体积（立方米）"),
    distance: Optional[float] = Body(None, description="距离（公里），如果为None，则从数据库中获取"),
    special_requirements: Optional[List[str]] = Body(None, description="特殊要求列表"),
    model_name: Optional[str] = Body(None, description="模型名称，如果为None，则使用默认模型"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    预测运输价格
    
    Args:
        origin_id: 起始地点ID
        destination_id: 目的地ID
        transport_mode_id: 运输方式ID
        cargo_type_id: 货物类型ID
        weight: 重量（千克）
        volume: 体积（立方米）
        distance: 距离（公里），如果为None，则从数据库中获取
        special_requirements: 特殊要求列表
        model_name: 模型名称，如果为None，则使用默认模型
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        包含预测价格和相关信息的字典
    """
    try:
        # 记录请求
        logger.info(f"用户 {current_user.username} 请求价格预测: origin_id={origin_id}, destination_id={destination_id}, transport_mode_id={transport_mode_id}, cargo_type_id={cargo_type_id}, weight={weight}, volume={volume}")
        
        # 调用MLService预测价格
        result = ml_service.predict_price(
            origin_id=origin_id,
            destination_id=destination_id,
            transport_mode_id=transport_mode_id,
            cargo_type_id=cargo_type_id,
            weight=weight,
            volume=volume,
            distance=distance,
            special_requirements=special_requirements,
            model_name=model_name
        )
        
        # 添加请求信息
        result["request"] = {
            "origin_id": origin_id,
            "destination_id": destination_id,
            "transport_mode_id": transport_mode_id,
            "cargo_type_id": cargo_type_id,
            "weight": weight,
            "volume": volume,
            "special_requirements": special_requirements or []
        }
        
        return result
        
    except MLModelError as e:
        logger.error(f"价格预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"价格预测失败: {str(e)}")
    except Exception as e:
        logger.error(f"价格预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@router.post("/train", response_model=Dict[str, Any])
async def train_model(
    model_name: str = Body("gradient_boosting", description="模型名称"),
    use_historical_data: bool = Body(True, description="是否使用历史数据"),
    test_size: float = Body(0.2, description="测试集比例"),
    random_state: int = Body(42, description="随机种子"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    训练新模型
    
    Args:
        model_name: 模型名称
        use_historical_data: 是否使用历史数据
        test_size: 测试集比例
        random_state: 随机种子
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        训练结果字典
    """
    # 检查用户权限
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="权限不足，只有管理员可以训练模型")
    
    try:
        # 记录请求
        logger.info(f"用户 {current_user.username} 请求训练模型: model_name={model_name}, use_historical_data={use_historical_data}")
        
        # 调用MLService训练模型
        result = ml_service.train_model(
            model_name=model_name,
            use_historical_data=use_historical_data,
            test_size=test_size,
            random_state=random_state
        )
        
        return result
        
    except MLModelError as e:
        logger.error(f"模型训练失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型训练失败: {str(e)}")
    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@router.get("/analyze/factors", response_model=Dict[str, Any])
async def analyze_price_factors(
    quote_id: Optional[int] = Query(None, description="报价ID，如果提供，则分析特定报价的因素"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    分析影响价格的因素
    
    Args:
        quote_id: 报价ID，如果提供，则分析特定报价的因素
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        分析结果字典
    """
    try:
        # 记录请求
        logger.info(f"用户 {current_user.username} 请求分析价格因素: quote_id={quote_id}")
        
        # 调用MLService分析价格因素
        result = ml_service.analyze_price_factors(quote_id=quote_id)
        
        return result
        
    except MLModelError as e:
        logger.error(f"分析价格因素失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析价格因素失败: {str(e)}")
    except Exception as e:
        logger.error(f"分析价格因素失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@router.get("/trends", response_model=Dict[str, Any])
async def get_price_trends(
    origin_id: Optional[int] = Query(None, description="起始地点ID"),
    destination_id: Optional[int] = Query(None, description="目的地ID"),
    transport_mode_id: Optional[int] = Query(None, description="运输方式ID"),
    cargo_type_id: Optional[int] = Query(None, description="货物类型ID"),
    period: str = Query("monthly", description="周期（daily, weekly, monthly, quarterly, yearly）"),
    months: int = Query(12, description="月数"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    获取价格趋势
    
    Args:
        origin_id: 起始地点ID
        destination_id: 目的地ID
        transport_mode_id: 运输方式ID
        cargo_type_id: 货物类型ID
        period: 周期（daily, weekly, monthly, quarterly, yearly）
        months: 月数
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        价格趋势字典
    """
    try:
        # 记录请求
        logger.info(f"用户 {current_user.username} 请求价格趋势: origin_id={origin_id}, destination_id={destination_id}, transport_mode_id={transport_mode_id}, cargo_type_id={cargo_type_id}, period={period}, months={months}")
        
        # 调用MLService获取价格趋势
        result = ml_service.get_price_trends(
            origin_id=origin_id,
            destination_id=destination_id,
            transport_mode_id=transport_mode_id,
            cargo_type_id=cargo_type_id,
            period=period,
            months=months
        )
        
        return result
        
    except MLModelError as e:
        logger.error(f"获取价格趋势失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取价格趋势失败: {str(e)}")
    except Exception as e:
        logger.error(f"获取价格趋势失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@router.get("/models", response_model=Dict[str, Any])
async def list_models(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    列出所有可用模型
    
    Args:
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        包含可用模型和指标的字典
    """
    try:
        # 记录请求
        logger.info(f"用户 {current_user.username} 请求列出可用模型")
        
        # 获取所有可用模型
        models = list(ml_service.models.keys())
        
        # 获取模型指标
        metrics = {}
        for model_name in models:
            if model_name in ml_service.model_metrics:
                metrics[model_name] = ml_service.model_metrics[model_name]
        
        return {
            "models": models,
            "default_model": ml_service.default_model,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"列出可用模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}") 