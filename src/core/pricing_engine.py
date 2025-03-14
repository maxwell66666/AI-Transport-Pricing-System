"""
核心定价引擎模块

本模块实现了运输报价系统的核心定价逻辑，包括基于规则的定价和基于机器学习的定价。
定价引擎可以根据各种因素计算运输报价，包括距离、重量、体积、运输方式、货物类型等。
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import json
import uuid

from sqlalchemy.orm import Session

from src.db.models import (
    Quote, QuoteDetail, TransportMode, CargoType, Location, 
    DistanceMatrix, PricingRule, RuleCategory
)
from src.db.repository import (
    quote_repository, quote_detail_repository, transport_mode_repository,
    cargo_type_repository, location_repository, distance_matrix_repository,
    pricing_rule_repository, rule_category_repository
)
from src.ml.price_predictor import PricePredictor
from src.core.llm_service import LLMService

logger = logging.getLogger(__name__)


class PricingEngine:
    """
    定价引擎类
    
    负责计算运输报价，包括基于规则的定价和基于机器学习的定价。
    """
    
    def __init__(self, db: Session):
        """
        初始化定价引擎
        
        Args:
            db: 数据库会话
        """
        self.db = db
        self.price_predictor = PricePredictor()
        self.llm_service = LLMService(db)
        self._load_rules()
    
    def _load_rules(self) -> None:
        """
        加载定价规则
        
        从数据库加载所有活跃的定价规则
        """
        self.rules = pricing_rule_repository.get_active_rules(self.db)
        logger.info(f"Loaded {len(self.rules)} pricing rules")
    
    def calculate_price(
        self, 
        origin_location_id: int,
        destination_location_id: int,
        transport_mode_id: int,
        cargo_type_id: int,
        weight: float,
        volume: float,
        special_requirements: Optional[str] = None,
        use_ml_model: bool = True,
        use_llm: bool = False
    ) -> Dict[str, Any]:
        """
        计算运输报价
        
        Args:
            origin_location_id: 起始地ID
            destination_location_id: 目的地ID
            transport_mode_id: 运输方式ID
            cargo_type_id: 货物类型ID
            weight: 重量（公斤）
            volume: 体积（立方米）
            special_requirements: 特殊要求
            use_ml_model: 是否使用机器学习模型
            use_llm: 是否使用LLM进行分析和调整
            
        Returns:
            报价结果，包括总价和明细
        """
        # 获取基础数据
        origin = location_repository.get(self.db, id=origin_location_id)
        destination = location_repository.get(self.db, id=destination_location_id)
        transport_mode = transport_mode_repository.get(self.db, id=transport_mode_id)
        cargo_type = cargo_type_repository.get(self.db, id=cargo_type_id)
        
        if not all([origin, destination, transport_mode, cargo_type]):
            raise ValueError("Invalid input parameters: one or more entities not found")
        
        # 获取距离和运输时间
        distance_matrix = distance_matrix_repository.get_by_locations_and_mode(
            self.db, 
            origin_id=origin_location_id, 
            destination_id=destination_location_id,
            mode_id=transport_mode_id
        )
        
        if not distance_matrix:
            raise ValueError(f"No distance matrix found for the given route and transport mode")
        
        distance = distance_matrix.distance
        transit_time = distance_matrix.typical_transit_time
        
        # 准备定价输入数据
        pricing_input = {
            "origin_location": {
                "id": origin.id,
                "name": origin.name,
                "country": origin.country,
                "city": origin.city,
                "is_port": origin.is_port,
                "is_airport": origin.is_airport
            },
            "destination_location": {
                "id": destination.id,
                "name": destination.name,
                "country": destination.country,
                "city": destination.city,
                "is_port": destination.is_port,
                "is_airport": destination.is_airport
            },
            "transport_mode": {
                "id": transport_mode.id,
                "name": transport_mode.name,
                "description": transport_mode.description
            },
            "cargo_type": {
                "id": cargo_type.id,
                "name": cargo_type.name,
                "description": cargo_type.description,
                "is_dangerous": cargo_type.is_dangerous,
                "requires_temperature_control": cargo_type.requires_temperature_control
            },
            "weight": weight,
            "volume": volume,
            "distance": distance,
            "transit_time": transit_time,
            "special_requirements": special_requirements
        }
        
        # 使用规则引擎计算基础价格
        rule_based_price, price_details = self._calculate_rule_based_price(pricing_input)
        
        # 如果需要，使用机器学习模型调整价格
        ml_adjusted_price = None
        if use_ml_model:
            ml_adjusted_price = self._calculate_ml_based_price(pricing_input, rule_based_price)
            
            # 如果ML模型给出了有效的价格，使用它来调整基础价格
            if ml_adjusted_price is not None:
                # 计算调整因子
                adjustment_factor = ml_adjusted_price / rule_based_price
                
                # 调整所有价格明细
                for detail in price_details:
                    detail["amount"] = round(detail["amount"] * adjustment_factor, 2)
                
                # 更新总价
                rule_based_price = ml_adjusted_price
        
        # 如果需要，使用LLM进行分析和调整
        llm_analysis = None
        llm_adjusted_price = None
        if use_llm:
            llm_analysis, llm_adjusted_price = self._get_llm_analysis_and_adjustment(
                pricing_input, rule_based_price, price_details
            )
            
            # 如果LLM给出了有效的价格调整，使用它
            if llm_adjusted_price is not None:
                # 计算调整因子
                adjustment_factor = llm_adjusted_price / rule_based_price
                
                # 调整所有价格明细
                for detail in price_details:
                    detail["amount"] = round(detail["amount"] * adjustment_factor, 2)
                
                # 更新总价
                rule_based_price = llm_adjusted_price
        
        # 准备最终报价结果
        result = {
            "total_price": round(rule_based_price, 2),
            "currency": "USD",  # 默认使用美元
            "details": price_details,
            "distance": distance,
            "transit_time": transit_time,
            "expected_delivery_date": (datetime.utcnow().date().isoformat() if transit_time < 1 
                                      else (datetime.utcnow().date() + transit_time).isoformat()),
            "ml_model_used": use_ml_model,
            "llm_assisted": use_llm
        }
        
        if llm_analysis:
            result["llm_analysis"] = llm_analysis
        
        return result
    
    def _calculate_rule_based_price(self, pricing_input: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        使用规则引擎计算价格
        
        Args:
            pricing_input: 定价输入数据
            
        Returns:
            总价和价格明细列表
        """
        # 初始化价格明细
        price_details = []
        
        # 计算基础运输成本
        base_cost = self._apply_base_cost_rules(pricing_input)
        price_details.append({
            "fee_type": "base_cost",
            "amount": base_cost,
            "currency": "USD",
            "description": "基础运输成本"
        })
        
        # 计算燃油附加费
        fuel_surcharge = base_cost * 0.1  # 默认为基础成本的10%
        price_details.append({
            "fee_type": "fuel_surcharge",
            "amount": fuel_surcharge,
            "currency": "USD",
            "description": "燃油附加费"
        })
        
        # 计算操作费
        handling_fee = base_cost * 0.05  # 默认为基础成本的5%
        price_details.append({
            "fee_type": "handling_fee",
            "amount": handling_fee,
            "currency": "USD",
            "description": "操作费"
        })
        
        # 计算文件费
        documentation_fee = base_cost * 0.05  # 默认为基础成本的5%
        price_details.append({
            "fee_type": "documentation_fee",
            "amount": documentation_fee,
            "currency": "USD",
            "description": "文件费"
        })
        
        # 计算特殊费用
        special_fees = self._calculate_special_fees(pricing_input, base_cost)
        price_details.extend(special_fees)
        
        # 计算总价
        total_price = sum(detail["amount"] for detail in price_details)
        
        # 应用折扣规则
        discount, discount_details = self._apply_discount_rules(pricing_input, total_price)
        if discount > 0:
            price_details.extend(discount_details)
            total_price -= discount
        
        return total_price, price_details
    
    def _apply_base_cost_rules(self, pricing_input: Dict[str, Any]) -> float:
        """
        应用基础成本规则
        
        Args:
            pricing_input: 定价输入数据
            
        Returns:
            基础成本
        """
        # 获取输入参数
        transport_mode = pricing_input["transport_mode"]
        weight = pricing_input["weight"]
        volume = pricing_input["volume"]
        distance = pricing_input["distance"]
        
        # 基于运输方式的基础成本计算
        if transport_mode["name"] == "海运整柜":
            # 海运整柜按体积和距离计算
            base_cost = volume * 200 + distance * 0.5
        elif transport_mode["name"] == "海运拼箱":
            # 海运拼箱按重量和体积计算，取较大值
            weight_based = weight * 2
            volume_based = volume * 300
            base_cost = max(weight_based, volume_based) + distance * 0.3
        elif transport_mode["name"] == "空运":
            # 空运主要按重量计算，但也考虑体积重
            volume_weight = volume * 167  # 体积重转换因子
            chargeable_weight = max(weight, volume_weight)
            base_cost = chargeable_weight * 5 + distance * 0.1
        elif transport_mode["name"] == "铁路运输":
            # 铁路运输按重量和距离计算
            base_cost = weight * 1.5 + distance * 0.8
        elif transport_mode["name"] == "公路运输":
            # 公路运输按重量和距离计算
            base_cost = weight * 1.2 + distance * 1.0
        else:
            # 默认计算方法
            base_cost = weight * 2 + volume * 250 + distance * 0.5
        
        # 确保基础成本不低于最小值
        min_cost = 500
        base_cost = max(base_cost, min_cost)
        
        return round(base_cost, 2)
    
    def _calculate_special_fees(self, pricing_input: Dict[str, Any], base_cost: float) -> List[Dict[str, Any]]:
        """
        计算特殊费用
        
        Args:
            pricing_input: 定价输入数据
            base_cost: 基础成本
            
        Returns:
            特殊费用列表
        """
        special_fees = []
        cargo_type = pricing_input["cargo_type"]
        special_requirements = pricing_input.get("special_requirements", "")
        
        # 危险品处理费
        if cargo_type["is_dangerous"]:
            dangerous_goods_fee = base_cost * 0.15
            special_fees.append({
                "fee_type": "dangerous_goods_fee",
                "amount": dangerous_goods_fee,
                "currency": "USD",
                "description": "危险品处理费"
            })
        
        # 温控费用
        if cargo_type["requires_temperature_control"]:
            temperature_control_fee = base_cost * 0.12
            special_fees.append({
                "fee_type": "temperature_control_fee",
                "amount": temperature_control_fee,
                "currency": "USD",
                "description": "温控费用"
            })
        
        # 根据特殊要求添加费用
        if special_requirements:
            if "贵重物品" in special_requirements:
                valuable_goods_fee = base_cost * 0.1
                special_fees.append({
                    "fee_type": "valuable_goods_fee",
                    "amount": valuable_goods_fee,
                    "currency": "USD",
                    "description": "贵重物品处理费"
                })
            
            if "大型设备" in special_requirements:
                oversized_cargo_fee = base_cost * 0.2
                special_fees.append({
                    "fee_type": "oversized_cargo_fee",
                    "amount": oversized_cargo_fee,
                    "currency": "USD",
                    "description": "大型货物处理费"
                })
            
            if "加急" in special_requirements:
                express_fee = base_cost * 0.3
                special_fees.append({
                    "fee_type": "express_fee",
                    "amount": express_fee,
                    "currency": "USD",
                    "description": "加急处理费"
                })
        
        return special_fees
    
    def _apply_discount_rules(self, pricing_input: Dict[str, Any], total_price: float) -> Tuple[float, List[Dict[str, Any]]]:
        """
        应用折扣规则
        
        Args:
            pricing_input: 定价输入数据
            total_price: 总价
            
        Returns:
            折扣金额和折扣明细列表
        """
        discount = 0
        discount_details = []
        
        # 大重量折扣
        weight = pricing_input["weight"]
        if weight >= 10000:
            weight_discount = total_price * 0.05
            discount += weight_discount
            discount_details.append({
                "fee_type": "weight_discount",
                "amount": -weight_discount,  # 负值表示折扣
                "currency": "USD",
                "description": "大重量折扣"
            })
        
        # 大体积折扣
        volume = pricing_input["volume"]
        if volume >= 50:
            volume_discount = total_price * 0.03
            discount += volume_discount
            discount_details.append({
                "fee_type": "volume_discount",
                "amount": -volume_discount,  # 负值表示折扣
                "currency": "USD",
                "description": "大体积折扣"
            })
        
        return discount, discount_details
    
    def _calculate_ml_based_price(self, pricing_input: Dict[str, Any], rule_based_price: float) -> Optional[float]:
        """
        使用机器学习模型计算价格
        
        Args:
            pricing_input: 定价输入数据
            rule_based_price: 基于规则的价格
            
        Returns:
            机器学习模型计算的价格，如果无法计算则返回None
        """
        try:
            # 准备模型输入特征
            features = {
                "origin_country": pricing_input["origin_location"]["country"],
                "origin_is_port": int(pricing_input["origin_location"]["is_port"]),
                "origin_is_airport": int(pricing_input["origin_location"]["is_airport"]),
                "destination_country": pricing_input["destination_location"]["country"],
                "destination_is_port": int(pricing_input["destination_location"]["is_port"]),
                "destination_is_airport": int(pricing_input["destination_location"]["is_airport"]),
                "transport_mode": pricing_input["transport_mode"]["name"],
                "cargo_type": pricing_input["cargo_type"]["name"],
                "is_dangerous": int(pricing_input["cargo_type"]["is_dangerous"]),
                "requires_temperature_control": int(pricing_input["cargo_type"]["requires_temperature_control"]),
                "weight": pricing_input["weight"],
                "volume": pricing_input["volume"],
                "distance": pricing_input["distance"],
                "transit_time": pricing_input["transit_time"]
            }
            
            # 使用模型预测价格
            predicted_price = self.price_predictor.predict(features)
            
            # 如果预测价格与规则价格相差太大，可能是异常值，此时返回None
            if predicted_price < rule_based_price * 0.5 or predicted_price > rule_based_price * 2:
                logger.warning(f"ML predicted price ({predicted_price}) differs significantly from rule-based price ({rule_based_price})")
                return None
            
            return predicted_price
        
        except Exception as e:
            logger.error(f"Error in ML-based pricing: {str(e)}")
            return None
    
    def _get_llm_analysis_and_adjustment(
        self, 
        pricing_input: Dict[str, Any], 
        current_price: float,
        price_details: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        使用LLM进行分析和价格调整
        
        Args:
            pricing_input: 定价输入数据
            current_price: 当前价格
            price_details: 价格明细
            
        Returns:
            LLM分析结果和调整后的价格
        """
        try:
            # 获取相似报价
            similar_quotes = quote_repository.get_similar_quotes(
                self.db,
                origin_id=pricing_input["origin_location"]["id"],
                destination_id=pricing_input["destination_location"]["id"],
                transport_mode_id=pricing_input["transport_mode"]["id"],
                cargo_type_id=pricing_input["cargo_type"]["id"],
                limit=3
            )
            
            # 准备相似报价数据
            similar_quotes_data = []
            for quote in similar_quotes:
                details = quote_detail_repository.get_by_quote(self.db, quote_id=quote.id)
                similar_quotes_data.append({
                    "quote_number": quote.quote_number,
                    "weight": quote.weight,
                    "volume": quote.volume,
                    "distance": quote.distance,
                    "total_price": quote.total_price,
                    "currency": quote.currency,
                    "quote_date": quote.created_at.date().isoformat(),
                    "details": [
                        {
                            "fee_type": detail.fee_type,
                            "amount": detail.amount,
                            "currency": detail.currency,
                            "description": detail.description
                        }
                        for detail in details
                    ]
                })
            
            # 使用LLM服务进行分析
            analysis_result = self.llm_service.analyze_quote(
                pricing_input=pricing_input,
                current_price=current_price,
                price_details=price_details,
                similar_quotes=similar_quotes_data
            )
            
            if not analysis_result:
                return None, None
            
            analysis = analysis_result.get("analysis")
            adjusted_price = analysis_result.get("adjusted_price")
            
            return analysis, adjusted_price
        
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return None, None
    
    def create_quote(
        self,
        user_id: int,
        origin_location_id: int,
        destination_location_id: int,
        transport_mode_id: int,
        cargo_type_id: int,
        weight: float,
        volume: float,
        special_requirements: Optional[str] = None,
        use_ml_model: bool = True,
        use_llm: bool = False
    ) -> Quote:
        """
        创建新的报价
        
        Args:
            user_id: 用户ID
            origin_location_id: 起始地ID
            destination_location_id: 目的地ID
            transport_mode_id: 运输方式ID
            cargo_type_id: 货物类型ID
            weight: 重量（公斤）
            volume: 体积（立方米）
            special_requirements: 特殊要求
            use_ml_model: 是否使用机器学习模型
            use_llm: 是否使用LLM进行分析和调整
            
        Returns:
            创建的报价对象
        """
        # 计算报价
        quote_result = self.calculate_price(
            origin_location_id=origin_location_id,
            destination_location_id=destination_location_id,
            transport_mode_id=transport_mode_id,
            cargo_type_id=cargo_type_id,
            weight=weight,
            volume=volume,
            special_requirements=special_requirements,
            use_ml_model=use_ml_model,
            use_llm=use_llm
        )
        
        # 生成报价编号
        quote_number = f"Q{datetime.utcnow().strftime('%Y%m%d')}{str(uuid.uuid4())[:8].upper()}"
        
        # 创建报价对象
        quote_data = {
            "quote_number": quote_number,
            "user_id": user_id,
            "origin_location_id": origin_location_id,
            "destination_location_id": destination_location_id,
            "transport_mode_id": transport_mode_id,
            "cargo_type_id": cargo_type_id,
            "weight": weight,
            "volume": volume,
            "distance": quote_result["distance"],
            "typical_transit_time": quote_result["transit_time"],
            "expected_delivery_date": datetime.fromisoformat(quote_result["expected_delivery_date"]),
            "special_requirements": special_requirements,
            "status": "pending",
            "total_price": quote_result["total_price"],
            "currency": quote_result["currency"],
            "is_llm_assisted": use_llm,
            "llm_analysis": quote_result.get("llm_analysis")
        }
        
        quote = Quote(**quote_data)
        self.db.add(quote)
        self.db.flush()  # 获取ID但不提交事务
        
        # 创建报价明细
        for detail in quote_result["details"]:
            quote_detail = QuoteDetail(
                quote_id=quote.id,
                fee_type=detail["fee_type"],
                amount=detail["amount"],
                currency=detail["currency"],
                description=detail["description"],
                is_taxable=True,
                tax_rate=0.0
            )
            self.db.add(quote_detail)
        
        # 提交事务
        self.db.commit()
        self.db.refresh(quote)
        
        return quote 