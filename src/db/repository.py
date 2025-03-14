"""
数据库仓库模块

本模块提供了一系列仓库类，用于处理数据库操作。
每个仓库类对应一个数据库模型，提供CRUD操作和其他特定的业务逻辑。
"""

from typing import List, Optional, Dict, Any, Generic, TypeVar, Type
from datetime import datetime
import uuid

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, not_, func
from fastapi.encoders import jsonable_encoder

from src.db.models import (
    User, APIKey, TransportMode, CargoType, Location, 
    DistanceMatrix, Quote, QuoteDetail, RuleCategory, 
    PricingRule, LLMPrompt, LLMInteraction, AuditLog
)

# 定义泛型类型变量
ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    基础仓库类，提供通用的CRUD操作
    
    泛型参数:
        ModelType: SQLAlchemy模型类型
        CreateSchemaType: 创建模型的Pydantic模式类型
        UpdateSchemaType: 更新模型的Pydantic模式类型
    """
    
    def __init__(self, model: Type[ModelType]):
        """
        初始化仓库
        
        Args:
            model: SQLAlchemy模型类
        """
        self.model = model
    
    def get(self, db: Session, id: int) -> Optional[ModelType]:
        """
        通过ID获取单个对象
        
        Args:
            db: 数据库会话
            id: 对象ID
            
        Returns:
            找到的对象，如果不存在则返回None
        """
        return db.query(self.model).filter(self.model.id == id).first()
    
    def get_multi(
        self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> List[ModelType]:
        """
        获取多个对象
        
        Args:
            db: 数据库会话
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            对象列表
        """
        return db.query(self.model).offset(skip).limit(limit).all()
    
    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        """
        创建新对象
        
        Args:
            db: 数据库会话
            obj_in: 创建对象的数据
            
        Returns:
            创建的对象
        """
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def update(
        self, db: Session, *, db_obj: ModelType, obj_in: UpdateSchemaType
    ) -> ModelType:
        """
        更新对象
        
        Args:
            db: 数据库会话
            db_obj: 要更新的数据库对象
            obj_in: 更新的数据
            
        Returns:
            更新后的对象
        """
        obj_data = jsonable_encoder(db_obj)
        update_data = obj_in.dict(exclude_unset=True)
        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def remove(self, db: Session, *, id: int) -> ModelType:
        """
        删除对象
        
        Args:
            db: 数据库会话
            id: 对象ID
            
        Returns:
            删除的对象
        """
        obj = db.query(self.model).get(id)
        db.delete(obj)
        db.commit()
        return obj


class UserRepository(BaseRepository[User, Any, Any]):
    """用户仓库类"""
    
    def __init__(self):
        super().__init__(User)
    
    def get_by_email(self, db: Session, *, email: str) -> Optional[User]:
        """
        通过邮箱获取用户
        
        Args:
            db: 数据库会话
            email: 用户邮箱
            
        Returns:
            找到的用户，如果不存在则返回None
        """
        return db.query(User).filter(User.email == email).first()
    
    def get_by_username(self, db: Session, *, username: str) -> Optional[User]:
        """
        通过用户名获取用户
        
        Args:
            db: 数据库会话
            username: 用户名
            
        Returns:
            找到的用户，如果不存在则返回None
        """
        return db.query(User).filter(User.username == username).first()


class APIKeyRepository(BaseRepository[APIKey, Any, Any]):
    """API密钥仓库类"""
    
    def __init__(self):
        super().__init__(APIKey)
    
    def get_by_key(self, db: Session, *, key: str) -> Optional[APIKey]:
        """
        通过密钥获取API密钥
        
        Args:
            db: 数据库会话
            key: API密钥
            
        Returns:
            找到的API密钥，如果不存在则返回None
        """
        return db.query(APIKey).filter(APIKey.key == key).first()
    
    def get_active_by_user(self, db: Session, *, user_id: int) -> List[APIKey]:
        """
        获取用户的所有活跃API密钥
        
        Args:
            db: 数据库会话
            user_id: 用户ID
            
        Returns:
            API密钥列表
        """
        return db.query(APIKey).filter(
            APIKey.user_id == user_id,
            APIKey.is_active == True,
            or_(
                APIKey.expires_at.is_(None),
                APIKey.expires_at > datetime.utcnow()
            )
        ).all()


class TransportModeRepository(BaseRepository[TransportMode, Any, Any]):
    """运输方式仓库类"""
    
    def __init__(self):
        super().__init__(TransportMode)
    
    def get_active(self, db: Session) -> List[TransportMode]:
        """
        获取所有活跃的运输方式
        
        Args:
            db: 数据库会话
            
        Returns:
            运输方式列表
        """
        return db.query(TransportMode).filter(TransportMode.is_active == True).all()


class CargoTypeRepository(BaseRepository[CargoType, Any, Any]):
    """货物类型仓库类"""
    
    def __init__(self):
        super().__init__(CargoType)
    
    def get_dangerous(self, db: Session) -> List[CargoType]:
        """
        获取所有危险货物类型
        
        Args:
            db: 数据库会话
            
        Returns:
            货物类型列表
        """
        return db.query(CargoType).filter(CargoType.is_dangerous == True).all()
    
    def get_temperature_controlled(self, db: Session) -> List[CargoType]:
        """
        获取所有需要温控的货物类型
        
        Args:
            db: 数据库会话
            
        Returns:
            货物类型列表
        """
        return db.query(CargoType).filter(CargoType.requires_temperature_control == True).all()


class LocationRepository(BaseRepository[Location, Any, Any]):
    """位置仓库类"""
    
    def __init__(self):
        super().__init__(Location)
    
    def get_ports(self, db: Session) -> List[Location]:
        """
        获取所有港口
        
        Args:
            db: 数据库会话
            
        Returns:
            位置列表
        """
        return db.query(Location).filter(Location.is_port == True).all()
    
    def get_airports(self, db: Session) -> List[Location]:
        """
        获取所有机场
        
        Args:
            db: 数据库会话
            
        Returns:
            位置列表
        """
        return db.query(Location).filter(Location.is_airport == True).all()
    
    def search(self, db: Session, *, query: str) -> List[Location]:
        """
        搜索位置
        
        Args:
            db: 数据库会话
            query: 搜索关键词
            
        Returns:
            位置列表
        """
        search_pattern = f"%{query}%"
        return db.query(Location).filter(
            or_(
                Location.name.ilike(search_pattern),
                Location.city.ilike(search_pattern),
                Location.country.ilike(search_pattern)
            )
        ).all()


class DistanceMatrixRepository(BaseRepository[DistanceMatrix, Any, Any]):
    """距离矩阵仓库类"""
    
    def __init__(self):
        super().__init__(DistanceMatrix)
    
    def get_by_locations_and_mode(
        self, db: Session, *, 
        origin_id: int, destination_id: int, mode_id: int
    ) -> Optional[DistanceMatrix]:
        """
        通过起始地、目的地和运输方式获取距离矩阵
        
        Args:
            db: 数据库会话
            origin_id: 起始地ID
            destination_id: 目的地ID
            mode_id: 运输方式ID
            
        Returns:
            找到的距离矩阵，如果不存在则返回None
        """
        return db.query(DistanceMatrix).filter(
            DistanceMatrix.origin_location_id == origin_id,
            DistanceMatrix.destination_location_id == destination_id,
            DistanceMatrix.transport_mode_id == mode_id
        ).first()


class QuoteRepository(BaseRepository[Quote, Any, Any]):
    """报价仓库类"""
    
    def __init__(self):
        super().__init__(Quote)
    
    def get_by_quote_number(self, db: Session, *, quote_number: str) -> Optional[Quote]:
        """
        通过报价编号获取报价
        
        Args:
            db: 数据库会话
            quote_number: 报价编号
            
        Returns:
            找到的报价，如果不存在则返回None
        """
        return db.query(Quote).filter(Quote.quote_number == quote_number).first()
    
    def get_by_user(self, db: Session, *, user_id: int, skip: int = 0, limit: int = 100) -> List[Quote]:
        """
        获取用户的所有报价
        
        Args:
            db: 数据库会话
            user_id: 用户ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            报价列表
        """
        return db.query(Quote).filter(Quote.user_id == user_id).offset(skip).limit(limit).all()
    
    def get_by_status(self, db: Session, *, status: str, skip: int = 0, limit: int = 100) -> List[Quote]:
        """
        通过状态获取报价
        
        Args:
            db: 数据库会话
            status: 报价状态
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            报价列表
        """
        return db.query(Quote).filter(Quote.status == status).offset(skip).limit(limit).all()
    
    def get_similar_quotes(
        self, db: Session, *, 
        origin_id: int, destination_id: int, 
        transport_mode_id: int, cargo_type_id: int,
        limit: int = 5
    ) -> List[Quote]:
        """
        获取相似的报价
        
        Args:
            db: 数据库会话
            origin_id: 起始地ID
            destination_id: 目的地ID
            transport_mode_id: 运输方式ID
            cargo_type_id: 货物类型ID
            limit: 返回的最大记录数
            
        Returns:
            报价列表
        """
        return db.query(Quote).filter(
            Quote.origin_location_id == origin_id,
            Quote.destination_location_id == destination_id,
            Quote.transport_mode_id == transport_mode_id,
            Quote.cargo_type_id == cargo_type_id,
            Quote.status == "confirmed"
        ).order_by(Quote.created_at.desc()).limit(limit).all()


class QuoteDetailRepository(BaseRepository[QuoteDetail, Any, Any]):
    """报价明细仓库类"""
    
    def __init__(self):
        super().__init__(QuoteDetail)
    
    def get_by_quote(self, db: Session, *, quote_id: int) -> List[QuoteDetail]:
        """
        获取报价的所有明细
        
        Args:
            db: 数据库会话
            quote_id: 报价ID
            
        Returns:
            报价明细列表
        """
        return db.query(QuoteDetail).filter(QuoteDetail.quote_id == quote_id).all()


class RuleCategoryRepository(BaseRepository[RuleCategory, Any, Any]):
    """规则类别仓库类"""
    
    def __init__(self):
        super().__init__(RuleCategory)


class PricingRuleRepository(BaseRepository[PricingRule, Any, Any]):
    """定价规则仓库类"""
    
    def __init__(self):
        super().__init__(PricingRule)
    
    def get_active_rules(self, db: Session) -> List[PricingRule]:
        """
        获取所有活跃的定价规则
        
        Args:
            db: 数据库会话
            
        Returns:
            定价规则列表
        """
        return db.query(PricingRule).filter(
            PricingRule.is_active == True
        ).order_by(PricingRule.priority).all()
    
    def get_by_category(self, db: Session, *, category_id: int) -> List[PricingRule]:
        """
        获取特定类别的所有定价规则
        
        Args:
            db: 数据库会话
            category_id: 规则类别ID
            
        Returns:
            定价规则列表
        """
        return db.query(PricingRule).filter(
            PricingRule.rule_category_id == category_id,
            PricingRule.is_active == True
        ).order_by(PricingRule.priority).all()


class LLMPromptRepository(BaseRepository[LLMPrompt, Any, Any]):
    """LLM提示模板仓库类"""
    
    def __init__(self):
        super().__init__(LLMPrompt)
    
    def get_by_purpose(self, db: Session, *, purpose: str) -> Optional[LLMPrompt]:
        """
        通过用途获取LLM提示模板
        
        Args:
            db: 数据库会话
            purpose: 提示模板用途
            
        Returns:
            找到的LLM提示模板，如果不存在则返回None
        """
        return db.query(LLMPrompt).filter(
            LLMPrompt.purpose == purpose,
            LLMPrompt.is_active == True
        ).first()


class LLMInteractionRepository(BaseRepository[LLMInteraction, Any, Any]):
    """LLM交互记录仓库类"""
    
    def __init__(self):
        super().__init__(LLMInteraction)
    
    def get_by_quote(self, db: Session, *, quote_id: int) -> List[LLMInteraction]:
        """
        获取报价的所有LLM交互记录
        
        Args:
            db: 数据库会话
            quote_id: 报价ID
            
        Returns:
            LLM交互记录列表
        """
        return db.query(LLMInteraction).filter(
            LLMInteraction.quote_id == quote_id
        ).order_by(LLMInteraction.created_at).all()


class AuditLogRepository(BaseRepository[AuditLog, Any, Any]):
    """审计日志仓库类"""
    
    def __init__(self):
        super().__init__(AuditLog)
    
    def get_by_user(self, db: Session, *, user_id: int, skip: int = 0, limit: int = 100) -> List[AuditLog]:
        """
        获取用户的所有审计日志
        
        Args:
            db: 数据库会话
            user_id: 用户ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            审计日志列表
        """
        return db.query(AuditLog).filter(
            AuditLog.user_id == user_id
        ).order_by(AuditLog.created_at.desc()).offset(skip).limit(limit).all()
    
    def get_by_entity(
        self, db: Session, *, 
        entity_type: str, entity_id: int, 
        skip: int = 0, limit: int = 100
    ) -> List[AuditLog]:
        """
        获取实体的所有审计日志
        
        Args:
            db: 数据库会话
            entity_type: 实体类型
            entity_id: 实体ID
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            审计日志列表
        """
        return db.query(AuditLog).filter(
            AuditLog.entity_type == entity_type,
            AuditLog.entity_id == entity_id
        ).order_by(AuditLog.created_at.desc()).offset(skip).limit(limit).all()


# 创建仓库实例
user_repository = UserRepository()
api_key_repository = APIKeyRepository()
transport_mode_repository = TransportModeRepository()
cargo_type_repository = CargoTypeRepository()
location_repository = LocationRepository()
distance_matrix_repository = DistanceMatrixRepository()
quote_repository = QuoteRepository()
quote_detail_repository = QuoteDetailRepository()
rule_category_repository = RuleCategoryRepository()
pricing_rule_repository = PricingRuleRepository()
llm_prompt_repository = LLMPromptRepository()
llm_interaction_repository = LLMInteractionRepository()
audit_log_repository = AuditLogRepository() 