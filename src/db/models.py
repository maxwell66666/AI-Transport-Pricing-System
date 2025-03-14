"""
数据库模型定义模块

本模块定义了AI运输报价系统所需的所有数据库模型，使用SQLAlchemy ORM框架。
模型设计遵循了关系数据库的最佳实践，包括适当的关系、约束和索引。

主要模型包括：
- 用户和认证相关模型
- 运输相关基础数据模型
- 报价和报价明细模型
- 规则引擎相关模型
- LLM集成相关模型
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    ForeignKey, Text, JSON, Enum, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship, declarative_base
import enum

Base = declarative_base()


class UserRole(enum.Enum):
    """用户角色枚举"""
    ADMIN = "admin"
    MANAGER = "manager"
    OPERATOR = "operator"
    CUSTOMER = "customer"
    GUEST = "guest"


class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)
    full_name = Column(String(100))
    role = Column(Enum(UserRole), default=UserRole.GUEST)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    quotes = relationship("Quote", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user")

    def __repr__(self):
        return f"<User {self.username}>"


class APIKey(Base):
    """API密钥模型"""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(50))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)

    # 关系
    user = relationship("User", back_populates="api_keys")

    def __repr__(self):
        return f"<APIKey {self.name}>"


class TransportMode(Base):
    """运输方式模型"""
    __tablename__ = "transport_modes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    quotes = relationship("Quote", back_populates="transport_mode")
    distance_matrices = relationship("DistanceMatrix", back_populates="transport_mode")

    def __repr__(self):
        return f"<TransportMode {self.name}>"


class CargoType(Base):
    """货物类型模型"""
    __tablename__ = "cargo_types"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    is_dangerous = Column(Boolean, default=False)
    requires_temperature_control = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    quotes = relationship("Quote", back_populates="cargo_type")

    def __repr__(self):
        return f"<CargoType {self.name}>"


class Location(Base):
    """位置模型"""
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    country = Column(String(50), nullable=False)
    city = Column(String(50), nullable=False)
    address = Column(String(200))
    postal_code = Column(String(20))
    latitude = Column(Float)
    longitude = Column(Float)
    is_port = Column(Boolean, default=False)
    is_airport = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    origin_quotes = relationship("Quote", foreign_keys="Quote.origin_location_id", back_populates="origin_location")
    destination_quotes = relationship("Quote", foreign_keys="Quote.destination_location_id", back_populates="destination_location")
    origin_distance_matrices = relationship("DistanceMatrix", foreign_keys="DistanceMatrix.origin_location_id", back_populates="origin_location")
    destination_distance_matrices = relationship("DistanceMatrix", foreign_keys="DistanceMatrix.destination_location_id", back_populates="destination_location")

    # 索引
    __table_args__ = (
        Index("idx_location_country_city", "country", "city"),
        Index("idx_location_port", "is_port"),
        Index("idx_location_airport", "is_airport"),
    )

    def __repr__(self):
        return f"<Location {self.name}, {self.city}, {self.country}>"


class DistanceMatrix(Base):
    """距离矩阵模型"""
    __tablename__ = "distance_matrix"

    id = Column(Integer, primary_key=True, index=True)
    origin_location_id = Column(Integer, ForeignKey("locations.id"), nullable=False)
    destination_location_id = Column(Integer, ForeignKey("locations.id"), nullable=False)
    transport_mode_id = Column(Integer, ForeignKey("transport_modes.id"), nullable=False)
    distance = Column(Float, nullable=False)  # 单位：公里
    typical_transit_time = Column(Float, nullable=False)  # 单位：天
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    origin_location = relationship("Location", foreign_keys=[origin_location_id], back_populates="origin_distance_matrices")
    destination_location = relationship("Location", foreign_keys=[destination_location_id], back_populates="destination_distance_matrices")
    transport_mode = relationship("TransportMode", back_populates="distance_matrices")

    # 唯一约束
    __table_args__ = (
        UniqueConstraint("origin_location_id", "destination_location_id", "transport_mode_id", name="uq_distance_matrix"),
    )

    def __repr__(self):
        return f"<DistanceMatrix {self.origin_location_id} to {self.destination_location_id} via {self.transport_mode_id}>"


class Quote(Base):
    """报价模型"""
    __tablename__ = "quotes"

    id = Column(Integer, primary_key=True, index=True)
    quote_number = Column(String(20), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    origin_location_id = Column(Integer, ForeignKey("locations.id"), nullable=False)
    destination_location_id = Column(Integer, ForeignKey("locations.id"), nullable=False)
    transport_mode_id = Column(Integer, ForeignKey("transport_modes.id"), nullable=False)
    cargo_type_id = Column(Integer, ForeignKey("cargo_types.id"), nullable=False)
    weight = Column(Float, nullable=False)  # 单位：公斤
    volume = Column(Float, nullable=False)  # 单位：立方米
    distance = Column(Float)  # 单位：公里
    typical_transit_time = Column(Float)  # 单位：天
    expected_delivery_date = Column(DateTime)
    special_requirements = Column(Text)
    status = Column(String(20), default="pending")  # pending, confirmed, cancelled
    total_price = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    is_llm_assisted = Column(Boolean, default=False)
    llm_analysis = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    user = relationship("User", back_populates="quotes")
    origin_location = relationship("Location", foreign_keys=[origin_location_id], back_populates="origin_quotes")
    destination_location = relationship("Location", foreign_keys=[destination_location_id], back_populates="destination_quotes")
    transport_mode = relationship("TransportMode", back_populates="quotes")
    cargo_type = relationship("CargoType", back_populates="quotes")
    details = relationship("QuoteDetail", back_populates="quote", cascade="all, delete-orphan")

    # 索引
    __table_args__ = (
        Index("idx_quote_date", "created_at"),
        Index("idx_quote_status", "status"),
    )

    def __repr__(self):
        return f"<Quote {self.quote_number}>"


class QuoteDetail(Base):
    """报价明细模型"""
    __tablename__ = "quote_details"

    id = Column(Integer, primary_key=True, index=True)
    quote_id = Column(Integer, ForeignKey("quotes.id"), nullable=False)
    fee_type = Column(String(50), nullable=False)  # base_cost, fuel_surcharge, handling_fee, etc.
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    description = Column(Text)
    is_taxable = Column(Boolean, default=True)
    tax_rate = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    quote = relationship("Quote", back_populates="details")

    def __repr__(self):
        return f"<QuoteDetail {self.fee_type} for Quote {self.quote_id}>"


class RuleCategory(Base):
    """规则类别模型"""
    __tablename__ = "rule_categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    rules = relationship("PricingRule", back_populates="category")

    def __repr__(self):
        return f"<RuleCategory {self.name}>"


class PricingRule(Base):
    """定价规则模型"""
    __tablename__ = "pricing_rules"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    rule_category_id = Column(Integer, ForeignKey("rule_categories.id"), nullable=False)
    rule_definition = Column(JSON, nullable=False)  # 存储规则定义的JSON
    priority = Column(Integer, default=100)  # 规则优先级，数字越小优先级越高
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    category = relationship("RuleCategory", back_populates="rules")

    # 索引
    __table_args__ = (
        Index("idx_rule_priority", "priority"),
        Index("idx_rule_active", "is_active"),
    )

    def __repr__(self):
        return f"<PricingRule {self.name}>"


class LLMPrompt(Base):
    """LLM提示模板模型"""
    __tablename__ = "llm_prompts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    prompt_template = Column(Text, nullable=False)
    purpose = Column(String(50), nullable=False)  # quote_adjustment, case_matching, etc.
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 索引
    __table_args__ = (
        Index("idx_prompt_purpose", "purpose"),
        Index("idx_prompt_active", "is_active"),
    )

    def __repr__(self):
        return f"<LLMPrompt {self.name}>"


class LLMInteraction(Base):
    """LLM交互记录模型"""
    __tablename__ = "llm_interactions"

    id = Column(Integer, primary_key=True, index=True)
    quote_id = Column(Integer, ForeignKey("quotes.id"), nullable=True)
    prompt_id = Column(Integer, ForeignKey("llm_prompts.id"), nullable=True)
    input_data = Column(JSON)
    prompt_used = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    model_used = Column(String(50), nullable=False)
    tokens_used = Column(Integer)
    processing_time = Column(Float)  # 单位：秒
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<LLMInteraction {self.id} for Quote {self.quote_id}>"


class AuditLog(Base):
    """审计日志模型"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(50), nullable=False)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(Integer)
    details = Column(JSON)
    ip_address = Column(String(50))
    user_agent = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)

    # 索引
    __table_args__ = (
        Index("idx_audit_action", "action"),
        Index("idx_audit_entity", "entity_type", "entity_id"),
        Index("idx_audit_date", "created_at"),
    )

    def __repr__(self):
        return f"<AuditLog {self.action} on {self.entity_type} {self.entity_id}>" 