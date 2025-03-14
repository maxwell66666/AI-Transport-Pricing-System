#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例数据加载脚本

此脚本用于加载示例数据到数据库，用于开发和测试目的。
脚本会从data/sample_data目录读取CSV文件，并将数据导入到数据库中。
"""

import csv
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from sqlalchemy.orm import Session

from src.db.database import get_engine
from src.db.models import (
    CargoType, DistanceMatrix, Location, PricingRule, Quote, 
    QuoteDetail, RuleCategory, TransportMode
)
from src.utils.logging import setup_logging

# 设置日志
logger = logging.getLogger(__name__)

# 示例数据目录
SAMPLE_DATA_DIR = ROOT_DIR / "data" / "sample_data"


def load_distance_matrix(session: Session) -> None:
    """加载距离矩阵数据"""
    logger.info("正在加载距离矩阵数据...")
    
    # 检查是否已存在距离矩阵数据
    existing_count = session.query(DistanceMatrix).count()
    if existing_count > 0:
        logger.info(f"已存在{existing_count}条距离矩阵数据，跳过加载")
        return
    
    # 读取CSV文件
    csv_file = SAMPLE_DATA_DIR / "distance_matrix.csv"
    if not csv_file.exists():
        logger.warning(f"距离矩阵数据文件不存在: {csv_file}")
        return
    
    # 获取位置和运输方式的映射
    locations = {loc.code: loc.id for loc in session.query(Location).all()}
    transport_modes = {tm.code: tm.id for tm in session.query(TransportMode).all()}
    
    # 读取CSV数据并导入
    distance_matrices = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # 获取位置和运输方式ID
                origin_code = row["origin_code"]
                destination_code = row["destination_code"]
                transport_mode_code = row["transport_mode_code"]
                
                if origin_code not in locations:
                    logger.warning(f"未知的起始位置代码: {origin_code}")
                    continue
                if destination_code not in locations:
                    logger.warning(f"未知的目的地位置代码: {destination_code}")
                    continue
                if transport_mode_code not in transport_modes:
                    logger.warning(f"未知的运输方式代码: {transport_mode_code}")
                    continue
                
                # 创建距离矩阵记录
                distance_matrix = DistanceMatrix(
                    origin_location_id=locations[origin_code],
                    destination_location_id=locations[destination_code],
                    transport_mode_id=transport_modes[transport_mode_code],
                    distance=float(row["distance"]),
                    typical_transit_time=int(row["typical_transit_time"]),
                    base_cost=float(row["base_cost"]),
                    cost_per_kg=float(row["cost_per_kg"]),
                    cost_per_cbm=float(row["cost_per_cbm"]),
                    currency=row["currency"],
                    notes=row.get("notes", ""),
                )
                distance_matrices.append(distance_matrix)
            except Exception as e:
                logger.error(f"处理距离矩阵数据行时出错: {e}, 行: {row}")
    
    # 批量插入数据
    if distance_matrices:
        session.add_all(distance_matrices)
        session.commit()
        logger.info(f"已加载{len(distance_matrices)}条距离矩阵数据")
    else:
        logger.warning("没有有效的距离矩阵数据可加载")


def load_pricing_rules(session: Session) -> None:
    """加载定价规则数据"""
    logger.info("正在加载定价规则数据...")
    
    # 检查是否已存在定价规则数据
    existing_count = session.query(PricingRule).count()
    if existing_count > 0:
        logger.info(f"已存在{existing_count}条定价规则数据，跳过加载")
        return
    
    # 创建规则类别
    rule_categories = [
        RuleCategory(name="基础费用", code="BASE_COST", description="基础运输费用计算规则"),
        RuleCategory(name="重量费用", code="WEIGHT_COST", description="基于重量的费用计算规则"),
        RuleCategory(name="体积费用", code="VOLUME_COST", description="基于体积的费用计算规则"),
        RuleCategory(name="特殊要求", code="SPECIAL_REQUIREMENTS", description="特殊要求的附加费用规则"),
        RuleCategory(name="货物类型", code="CARGO_TYPE", description="基于货物类型的费用调整规则"),
        RuleCategory(name="运输方式", code="TRANSPORT_MODE", description="基于运输方式的费用调整规则"),
        RuleCategory(name="距离调整", code="DISTANCE_ADJUSTMENT", description="基于距离的费用调整规则"),
        RuleCategory(name="季节性调整", code="SEASONAL_ADJUSTMENT", description="基于季节的费用调整规则"),
        RuleCategory(name="折扣", code="DISCOUNT", description="各类折扣规则"),
    ]
    session.add_all(rule_categories)
    session.commit()
    
    # 获取规则类别的映射
    categories = {cat.code: cat.id for cat in rule_categories}
    
    # 创建示例定价规则
    pricing_rules = [
        # 基础费用规则
        PricingRule(
            name="基础距离费用",
            code="BASE_DISTANCE_COST",
            category_id=categories["BASE_COST"],
            description="基于距离的基础费用",
            rule_type="formula",
            rule_data={"formula": "distance * base_cost"},
            priority=100,
            is_active=True,
        ),
        PricingRule(
            name="重量费用",
            code="WEIGHT_COST",
            category_id=categories["WEIGHT_COST"],
            description="基于重量的费用",
            rule_type="formula",
            rule_data={"formula": "weight * cost_per_kg"},
            priority=90,
            is_active=True,
        ),
        PricingRule(
            name="体积费用",
            code="VOLUME_COST",
            category_id=categories["VOLUME_COST"],
            description="基于体积的费用",
            rule_type="formula",
            rule_data={"formula": "volume * cost_per_cbm"},
            priority=80,
            is_active=True,
        ),
        
        # 特殊要求规则
        PricingRule(
            name="温控要求",
            code="TEMPERATURE_CONTROL",
            category_id=categories["SPECIAL_REQUIREMENTS"],
            description="温度控制要求的附加费用",
            rule_type="fixed",
            rule_data={"amount": 500, "fee_type": "temperature_control_fee"},
            conditions={"special_requirements": ["需要温控", "温控"]},
            priority=70,
            is_active=True,
        ),
        PricingRule(
            name="加急处理",
            code="EXPRESS_HANDLING",
            category_id=categories["SPECIAL_REQUIREMENTS"],
            description="加急处理的附加费用",
            rule_type="percentage",
            rule_data={"percentage": 15, "fee_type": "express_handling_fee"},
            conditions={"special_requirements": ["加急", "紧急"]},
            priority=70,
            is_active=True,
        ),
        
        # 货物类型规则
        PricingRule(
            name="危险品附加费",
            code="DANGEROUS_GOODS_SURCHARGE",
            category_id=categories["CARGO_TYPE"],
            description="危险品运输的附加费用",
            rule_type="percentage",
            rule_data={"percentage": 30, "fee_type": "dangerous_goods_fee"},
            conditions={"cargo_type_id": [3]},  # 危险品ID
            priority=60,
            is_active=True,
        ),
        PricingRule(
            name="冷藏品附加费",
            code="REFRIGERATED_GOODS_SURCHARGE",
            category_id=categories["CARGO_TYPE"],
            description="冷藏品运输的附加费用",
            rule_type="percentage",
            rule_data={"percentage": 20, "fee_type": "refrigerated_goods_fee"},
            conditions={"cargo_type_id": [4]},  # 冷藏品ID
            priority=60,
            is_active=True,
        ),
        
        # 运输方式规则
        PricingRule(
            name="空运燃油附加费",
            code="AIR_FUEL_SURCHARGE",
            category_id=categories["TRANSPORT_MODE"],
            description="空运的燃油附加费",
            rule_type="percentage",
            rule_data={"percentage": 10, "fee_type": "fuel_surcharge"},
            conditions={"transport_mode_id": [4]},  # 空运ID
            priority=50,
            is_active=True,
        ),
        PricingRule(
            name="海运文件费",
            code="SEA_DOCUMENTATION_FEE",
            category_id=categories["TRANSPORT_MODE"],
            description="海运的文件处理费",
            rule_type="fixed",
            rule_data={"amount": 200, "fee_type": "documentation_fee"},
            conditions={"transport_mode_id": [3]},  # 海运ID
            priority=50,
            is_active=True,
        ),
        
        # 距离调整规则
        PricingRule(
            name="长距离折扣",
            code="LONG_DISTANCE_DISCOUNT",
            category_id=categories["DISTANCE_ADJUSTMENT"],
            description="长距离运输的折扣",
            rule_type="percentage",
            rule_data={"percentage": -5, "fee_type": "distance_discount"},
            conditions={"distance": {"min": 1000}},
            priority=40,
            is_active=True,
        ),
        
        # 季节性调整规则
        PricingRule(
            name="春节旺季附加费",
            code="SPRING_FESTIVAL_SURCHARGE",
            category_id=categories["SEASONAL_ADJUSTMENT"],
            description="春节期间的旺季附加费",
            rule_type="percentage",
            rule_data={"percentage": 20, "fee_type": "seasonal_surcharge"},
            conditions={"date": {"month": [1, 2]}},
            priority=30,
            is_active=True,
        ),
        
        # 折扣规则
        PricingRule(
            name="大客户折扣",
            code="VIP_CUSTOMER_DISCOUNT",
            category_id=categories["DISCOUNT"],
            description="VIP客户的折扣",
            rule_type="percentage",
            rule_data={"percentage": -10, "fee_type": "vip_discount"},
            conditions={"user_id": [1]},  # 管理员用户ID
            priority=20,
            is_active=True,
        ),
    ]
    
    session.add_all(pricing_rules)
    session.commit()
    logger.info(f"已加载{len(pricing_rules)}条定价规则数据和{len(rule_categories)}个规则类别")


def load_quotes(session: Session) -> None:
    """加载报价数据"""
    logger.info("正在加载报价数据...")
    
    # 检查是否已存在报价数据
    existing_count = session.query(Quote).count()
    if existing_count > 0:
        logger.info(f"已存在{existing_count}条报价数据，跳过加载")
        return
    
    # 读取CSV文件
    csv_file = SAMPLE_DATA_DIR / "quotes.csv"
    if not csv_file.exists():
        logger.warning(f"报价数据文件不存在: {csv_file}")
        return
    
    # 获取位置、运输方式和货物类型的映射
    locations = {loc.code: loc.id for loc in session.query(Location).all()}
    transport_modes = {tm.code: tm.id for tm in session.query(TransportMode).all()}
    cargo_types = {ct.code: ct.id for ct in session.query(CargoType).all()}
    
    # 读取CSV数据并导入
    quotes = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # 解析日期
                quote_date = datetime.strptime(row["quote_date"], "%Y-%m-%d")
                
                # 解析特殊要求
                special_requirements = row["special_requirements"].split(",") if row["special_requirements"] else []
                
                # 创建报价记录
                quote = Quote(
                    id=row["id"],
                    user_id=1,  # 默认为管理员用户
                    origin_location_id=int(row["origin_location_id"]),
                    destination_location_id=int(row["destination_location_id"]),
                    transport_mode_id=int(row["transport_mode_id"]),
                    cargo_type_id=int(row["cargo_type_id"]),
                    weight=float(row["weight"]),
                    volume=float(row["volume"]),
                    distance=float(row["distance"]),
                    typical_transit_time=int(row["typical_transit_time"]),
                    total_price=float(row["total_price"]),
                    currency=row["currency"],
                    quote_date=quote_date,
                    expected_delivery_date=quote_date + timedelta(days=int(row["typical_transit_time"])),
                    special_requirements=special_requirements,
                    status="completed",
                    is_ml_assisted=row["is_ml_assisted"].lower() == "true",
                    is_llm_assisted=row["is_llm_assisted"].lower() == "true",
                )
                quotes.append(quote)
            except Exception as e:
                logger.error(f"处理报价数据行时出错: {e}, 行: {row}")
    
    # 批量插入数据
    if quotes:
        session.add_all(quotes)
        session.commit()
        logger.info(f"已加载{len(quotes)}条报价数据")
    else:
        logger.warning("没有有效的报价数据可加载")


def load_quote_details(session: Session) -> None:
    """加载报价明细数据"""
    logger.info("正在加载报价明细数据...")
    
    # 检查是否已存在报价明细数据
    existing_count = session.query(QuoteDetail).count()
    if existing_count > 0:
        logger.info(f"已存在{existing_count}条报价明细数据，跳过加载")
        return
    
    # 读取CSV文件
    csv_file = SAMPLE_DATA_DIR / "quote_details.csv"
    if not csv_file.exists():
        logger.warning(f"报价明细数据文件不存在: {csv_file}")
        return
    
    # 读取CSV数据并导入
    quote_details = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # 创建报价明细记录
                quote_detail = QuoteDetail(
                    id=row["id"],
                    quote_id=row["quote_id"],
                    fee_type=row["fee_type"],
                    amount=float(row["amount"]),
                    currency=row["currency"],
                    description=row["description"],
                )
                quote_details.append(quote_detail)
            except Exception as e:
                logger.error(f"处理报价明细数据行时出错: {e}, 行: {row}")
    
    # 批量插入数据
    if quote_details:
        session.add_all(quote_details)
        session.commit()
        logger.info(f"已加载{len(quote_details)}条报价明细数据")
    else:
        logger.warning("没有有效的报价明细数据可加载")


def main():
    """主函数"""
    # 设置日志
    setup_logging()
    
    logger.info("开始加载示例数据...")
    
    try:
        # 获取数据库引擎
        engine = get_engine()
        
        # 创建会话
        with Session(engine) as session:
            # 加载距离矩阵数据
            load_distance_matrix(session)
            
            # 加载定价规则数据
            load_pricing_rules(session)
            
            # 加载报价数据
            load_quotes(session)
            
            # 加载报价明细数据
            load_quote_details(session)
        
        logger.info("示例数据加载完成")
        
    except Exception as e:
        logger.error(f"加载示例数据时出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 