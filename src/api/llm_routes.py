"""
大型语言模型服务API路由

此模块提供大型语言模型服务的API接口，包括报价解释和优化功能。
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from src.llm.llm_service import LLMService
from src.utils.exceptions import LLMServiceError
from src.db.database import get_db
from sqlalchemy.orm import Session
from src.utils.auth import get_current_user, get_current_active_user
from src.db.models import User
from src.config import settings

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(
    prefix="/api/llm",
    tags=["大型语言模型"],
    responses={404: {"description": "Not found"}},
)

# 创建LLMService实例
llm_service = LLMService()


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    健康检查端点
    
    Returns:
        包含服务状态的字典
    """
    try:
        return {
            "status": "healthy",
            "model": llm_service.model_name,
            "api_base_url": llm_service.api_base_url,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.post("/explain", response_model=Dict[str, Any])
async def explain_quote(
    quote_id: int = Body(..., description="报价ID"),
    use_mock: bool = Body(False, description="是否使用模拟API（用于开发和测试）"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    解释运输报价
    
    Args:
        quote_id: 报价ID
        use_mock: 是否使用模拟API
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        包含解释内容和元数据的字典
    """
    try:
        # 记录请求
        logger.info(f"用户 {current_user.username} 请求解释报价: quote_id={quote_id}")
        
        # 调用LLMService解释报价
        result = llm_service.explain_quote(
            quote_id=quote_id,
            db=db,
            use_mock=use_mock or settings.ENVIRONMENT == "development"
        )
        
        return result
        
    except LLMServiceError as e:
        logger.error(f"解释报价失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"解释报价失败: {str(e)}")
    except Exception as e:
        logger.error(f"解释报价失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@router.post("/optimize", response_model=Dict[str, Any])
async def optimize_price(
    quote_id: int = Body(..., description="报价ID"),
    use_mock: bool = Body(False, description="是否使用模拟API（用于开发和测试）"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    优化运输价格
    
    Args:
        quote_id: 报价ID
        use_mock: 是否使用模拟API
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        包含优化建议和元数据的字典
    """
    try:
        # 记录请求
        logger.info(f"用户 {current_user.username} 请求优化价格: quote_id={quote_id}")
        
        # 调用LLMService优化价格
        result = llm_service.optimize_price(
            quote_id=quote_id,
            db=db,
            use_mock=use_mock or settings.ENVIRONMENT == "development"
        )
        
        return result
        
    except LLMServiceError as e:
        logger.error(f"优化价格失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"优化价格失败: {str(e)}")
    except Exception as e:
        logger.error(f"优化价格失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@router.post("/compare", response_model=Dict[str, Any])
async def compare_quotes(
    quote_id1: int = Body(..., description="第一个报价ID"),
    quote_id2: int = Body(..., description="第二个报价ID"),
    use_mock: bool = Body(False, description="是否使用模拟API（用于开发和测试）"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    比较两个运输报价
    
    Args:
        quote_id1: 第一个报价ID
        quote_id2: 第二个报价ID
        use_mock: 是否使用模拟API
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        包含比较分析和元数据的字典
    """
    try:
        # 记录请求
        logger.info(f"用户 {current_user.username} 请求比较报价: quote_id1={quote_id1}, quote_id2={quote_id2}")
        
        # 调用LLMService比较报价
        result = llm_service.compare_quotes(
            quote_id1=quote_id1,
            quote_id2=quote_id2,
            db=db,
            use_mock=use_mock or settings.ENVIRONMENT == "development"
        )
        
        return result
        
    except LLMServiceError as e:
        logger.error(f"比较报价失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"比较报价失败: {str(e)}")
    except Exception as e:
        logger.error(f"比较报价失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@router.get("/interactions/{quote_id}", response_model=List[Dict[str, Any]])
async def get_quote_interactions(
    quote_id: int = Path(..., description="报价ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    获取报价的LLM交互历史
    
    Args:
        quote_id: 报价ID
        db: 数据库会话
        current_user: 当前用户
        
    Returns:
        LLM交互历史列表
    """
    try:
        # 记录请求
        logger.info(f"用户 {current_user.username} 请求报价交互历史: quote_id={quote_id}")
        
        # 从数据库获取交互历史
        from src.db.repository import LLMInteractionRepository
        interaction_repo = LLMInteractionRepository(db)
        interactions = interaction_repo.get_by_quote_id(quote_id)
        
        # 格式化响应
        result = []
        for interaction in interactions:
            prompt = interaction.prompt
            result.append({
                "id": interaction.id,
                "quote_id": interaction.quote_id,
                "related_quote_id": interaction.related_quote_id,
                "prompt_type": prompt.type if prompt else "未知",
                "prompt_text": interaction.prompt_text,
                "response_text": interaction.response_text,
                "model": interaction.model,
                "token_usage": {
                    "prompt_tokens": interaction.prompt_tokens,
                    "completion_tokens": interaction.completion_tokens,
                    "total_tokens": interaction.total_tokens
                },
                "created_at": interaction.created_at.isoformat()
            })
        
        return result
        
    except Exception as e:
        logger.error(f"获取报价交互历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}") 