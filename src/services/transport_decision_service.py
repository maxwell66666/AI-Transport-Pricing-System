"""
艺术品运输决策支持服务
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

from src.preprocessing.art_transport_preprocessor import ArtTransportPreprocessor
from src.models.art_transport_price_model import ArtTransportPriceModel

class TransportDecisionService:
    """艺术品运输决策支持服务"""
    
    def __init__(self):
        """初始化服务"""
        # 风险权重
        self.risk_weights = {
            'value': 0.3,
            'fragility': 0.2,
            'time_pressure': 0.2,
            'distance': 0.15,
            'special_handling': 0.15
        }
        
        # 运输模式阈值
        self.transport_thresholds = {
            'high_value': 100000.0,  # 高价值艺术品阈值
            'urgent_days': 7,        # 紧急运输天数阈值
            'long_distance': 5000.0  # 长距离运输阈值（公里）
        }
        
        # 初始化预处理器和价格模型
        self.preprocessor = ArtTransportPreprocessor()
        self.price_model = ArtTransportPriceModel()
        
        # 使用简单的测试数据训练价格模型
        self._train_price_model_with_sample_data()
    
    def _train_price_model_with_sample_data(self):
        """使用样本数据训练价格模型"""
        import numpy as np
        
        # 创建简单的训练数据
        np.random.seed(42)
        n_samples = 100
        
        # 创建特征数据
        train_data = pd.DataFrame({
            'Artwork_Value_Normalized': np.random.uniform(0, 1, n_samples),
            'Weight_Normalized': np.random.uniform(0, 1, n_samples),
            'Is_International': np.random.randint(0, 2, n_samples),
            'same_continent': np.random.randint(0, 2, n_samples),
            'is_2d': np.random.randint(0, 2, n_samples),
            'fragility': np.random.uniform(0, 1, n_samples),
            'Requires_Climate_Control': np.random.randint(0, 2, n_samples),
            'Requires_Custom_Crating': np.random.randint(0, 2, n_samples),
            'Requires_Art_Handler': np.random.randint(0, 2, n_samples),
            'Type_painting': np.random.randint(0, 2, n_samples),
            'Type_sculpture': np.random.randint(0, 2, n_samples),
            'Type_installation': np.random.randint(0, 2, n_samples)
        })
        
        # 创建目标变量（价格）
        train_prices = pd.Series(
            5000 + 
            10000 * train_data['Artwork_Value_Normalized'] +
            5000 * train_data['Weight_Normalized'] +
            3000 * train_data['Is_International'] +
            1000 * train_data['same_continent'] +
            500 * train_data['is_2d'] +
            2000 * train_data['fragility'] +
            1500 * train_data['Requires_Climate_Control'] +
            2000 * train_data['Requires_Custom_Crating'] +
            1000 * train_data['Requires_Art_Handler'] +
            np.random.normal(0, 500, n_samples)
        )
        
        # 训练模型
        self.price_model.train(train_data, train_prices)
    
    def calculate_risk_score(self,
                           artwork_value: float,
                           fragility_score: int,
                           days_to_deadline: int,
                           distance_km: float,
                           special_handling_count: int) -> Dict[str, Any]:
        """
        计算风险评分
        
        Args:
            artwork_value: 艺术品价值
            fragility_score: 易碎性评分（1-5）
            days_to_deadline: 距离截止日期的天数
            distance_km: 运输距离（公里）
            special_handling_count: 特殊处理需求数量
            
        Returns:
            Dict[str, Any]: 风险评估结果
        """
        # 标准化各个风险因素
        max_value = 1000000.0  # 假设最大艺术品价值为100万
        value_score = min(artwork_value / max_value, 1.0)
        
        fragility_score = (fragility_score - 1) / 4  # 转换到0-1范围
        
        time_pressure = max(0, min(1, 1 - days_to_deadline / 30))  # 假设30天为最长期限
        
        max_distance = 10000.0  # 假设最大运输距离为10000公里
        distance_score = min(distance_km / max_distance, 1.0)
        
        max_special_handling = 3  # 最大特殊处理需求数量
        special_handling_score = special_handling_count / max_special_handling
        
        # 计算加权风险评分
        risk_components = {
            'value_risk': value_score * self.risk_weights['value'],
            'fragility_risk': fragility_score * self.risk_weights['fragility'],
            'time_risk': time_pressure * self.risk_weights['time_pressure'],
            'distance_risk': distance_score * self.risk_weights['distance'],
            'special_handling_risk': special_handling_score * self.risk_weights['special_handling']
        }
        
        total_risk = sum(risk_components.values())
        
        # 确定风险等级
        risk_level = '低风险'
        if total_risk > 0.7:
            risk_level = '高风险'
        elif total_risk > 0.4:
            risk_level = '中等风险'
        
        return {
            'total_score': total_risk,
            'components': risk_components,
            'risk_level': risk_level
        }
    
    def recommend_transport_mode(self,
                               is_international: bool,
                               artwork_value: float,
                               weight: float,
                               days_to_deadline: int,
                               distance_km: float) -> Dict[str, Any]:
        """
        推荐运输模式
        
        Args:
            is_international: 是否国际运输
            artwork_value: 艺术品价值
            weight: 计费重量
            days_to_deadline: 距离截止日期的天数
            distance_km: 运输距离
            
        Returns:
            Dict[str, Any]: 运输模式建议
        """
        reasons = []
        
        # 国际运输只能选择空运
        if is_international:
            reasons.append("国际运输需要使用空运")
            return {
                'recommended_mode': '国际空运',
                'reasons': reasons
            }
        
        # 高价值艺术品建议使用空运
        if artwork_value >= self.transport_thresholds['high_value']:
            reasons.append(f"艺术品价值超过{self.transport_thresholds['high_value']}，建议使用空运")
            return {
                'recommended_mode': '国内空运',
                'reasons': reasons
            }
        
        # 紧急运输使用空运
        if days_to_deadline <= self.transport_thresholds['urgent_days']:
            reasons.append(f"距离截止日期不足{self.transport_thresholds['urgent_days']}天，建议使用空运")
            return {
                'recommended_mode': '国内空运',
                'reasons': reasons
            }
        
        # 长距离运输建议使用空运
        if distance_km >= self.transport_thresholds['long_distance']:
            reasons.append(f"运输距离超过{self.transport_thresholds['long_distance']}公里，建议使用空运")
            return {
                'recommended_mode': '国内空运',
                'reasons': reasons
            }
        
        # 其他情况使用陆运
        reasons.append("常规运输可以使用陆运")
        return {
            'recommended_mode': '国内陆运',
            'reasons': reasons
        }
    
    def generate_time_plan(self,
                          transport_mode: str,
                          distance_km: float,
                          requires_special_handling: bool,
                          deadline: datetime) -> Dict[str, Any]:
        """
        生成时间规划
        
        Args:
            transport_mode: 运输模式
            distance_km: 运输距离
            requires_special_handling: 是否需要特殊处理
            deadline: 截止日期
            
        Returns:
            Dict[str, Any]: 时间规划
        """
        # 基础运输时间估算（天）
        transport_days = {
            '国际空运': distance_km / 800,  # 假设每天飞行800公里
            '国内空运': distance_km / 1000,  # 假设每天飞行1000公里
            '国内陆运': distance_km / 400   # 假设每天行驶400公里
        }
        
        base_transport_days = round(transport_days[transport_mode])
        
        # 特殊处理时间
        special_handling_days = 2 if requires_special_handling else 0
        
        # 清关时间（仅国际运输）
        customs_days = 3 if transport_mode == '国际空运' else 0
        
        # 总预计天数
        total_days = base_transport_days + special_handling_days + customs_days
        
        # 建议开始日期（考虑1天缓冲）
        buffer_days = 1
        latest_start = deadline - timedelta(days=total_days)
        recommended_start = latest_start - timedelta(days=buffer_days)
        
        return {
            'total_estimated_days': total_days,
            'breakdown': {
                'transport_days': base_transport_days,
                'special_handling_days': special_handling_days,
                'customs_days': customs_days,
                'buffer_days': buffer_days
            },
            'recommended_start_date': recommended_start,
            'latest_start_date': latest_start,
            'deadline': deadline
        }
    
    def estimate_total_cost(self,
                          base_price: float,
                          artwork_value: float,
                          special_handling_cost: float,
                          is_express: bool) -> Dict[str, Any]:
        """
        估算总成本
        
        Args:
            base_price: 基础运输价格
            artwork_value: 艺术品价值
            special_handling_cost: 特殊处理成本
            is_express: 是否快递服务
            
        Returns:
            Dict[str, Any]: 成本估算
        """
        # 保险费率（艺术品价值的0.5%）
        insurance_rate = 0.005
        insurance_cost = artwork_value * insurance_rate
        
        # 快递服务附加费（基础价格的30%）
        express_surcharge = base_price * 0.3 if is_express else 0
        
        # 计算总成本
        total_cost = (
            base_price +
            insurance_cost +
            special_handling_cost +
            express_surcharge
        )
        
        return {
            'total_cost': total_cost,
            'breakdown': {
                'base_price': base_price,
                'insurance_cost': insurance_cost,
                'special_handling_cost': special_handling_cost,
                'express_surcharge': express_surcharge
            },
            'insurance_details': {
                'rate': insurance_rate,
                'coverage_amount': artwork_value * 1.1  # 保险覆盖额度（110%艺术品价值）
            }
        }
    
    def generate_full_recommendation(self,
                                   artwork_data: Dict,
                                   distance_km: float) -> Dict[str, Any]:
        """
        生成完整的运输建议
        
        Args:
            artwork_data: 艺术品运输数据
            distance_km: 运输距离
            
        Returns:
            Dict: 完整的运输建议
        """
        # 确保artwork_data中的键名符合预处理器的期望
        # 如果需要，进行键名转换
        standardized_data = {}
        for key, value in artwork_data.items():
            standardized_data[key.lower() if key not in self.preprocessor.column_mapping else key] = value
        
        # 1. 计算风险评分
        risk_assessment = self.calculate_risk_score(
            artwork_value=standardized_data['artwork_value'],
            fragility_score=standardized_data.get('fragility_score', 3),
            days_to_deadline=standardized_data['days_to_deadline'],
            distance_km=distance_km,
            special_handling_count=sum([
                standardized_data.get('requires_climate_control', False),
                standardized_data.get('requires_custom_crating', False),
                standardized_data.get('requires_art_handler', False)
            ])
        )
        
        # 2. 推荐运输模式
        transport_recommendation = self.recommend_transport_mode(
            is_international=standardized_data['is_international'],
            artwork_value=standardized_data['artwork_value'],
            weight=standardized_data['chargeable_weight'],
            days_to_deadline=standardized_data['days_to_deadline'],
            distance_km=distance_km
        )
        
        # 3. 生成时间规划
        time_plan = self.generate_time_plan(
            transport_mode=transport_recommendation['recommended_mode'],
            distance_km=distance_km,
            requires_special_handling=any([
                standardized_data.get('requires_climate_control', False),
                standardized_data.get('requires_custom_crating', False),
                standardized_data.get('requires_art_handler', False)
            ]),
            deadline=standardized_data['deadline']
        )
        
        # 4. 预处理数据
        processed_data = self.preprocessor.preprocess_dataframe(
            pd.DataFrame([standardized_data])
        )
        
        # 5. 预测基础运输价格
        features = self.price_model.prepare_features(processed_data)
        price_prediction = self.price_model.predict(features)
        
        # 获取集成模型的预测结果
        base_price = price_prediction['ensemble'].iloc[0]
        
        # 计算置信区间（使用两个模型预测的差异作为不确定性度量）
        ridge_pred = price_prediction['ridge'].iloc[0]
        gb_pred = price_prediction['gb'].iloc[0]
        uncertainty = abs(ridge_pred - gb_pred) / 2
        
        # 创建符合测试预期的价格预测详情
        price_prediction_details = {
            'base_price': base_price,
            'ridge_prediction': ridge_pred,
            'gb_prediction': gb_pred,
            'confidence_interval': {
                'lower': base_price - uncertainty,
                'upper': base_price + uncertainty
            }
        }
        
        # 6. 估算总成本
        cost_estimation = self.estimate_total_cost(
            base_price=base_price,
            artwork_value=standardized_data['artwork_value'],
            special_handling_cost=processed_data['Special_Handling_Cost'].iloc[0] if 'Special_Handling_Cost' in processed_data.columns else sum([
                500 if standardized_data.get('requires_climate_control', False) else 0,
                1200 if standardized_data.get('requires_custom_crating', False) else 0,
                800 if standardized_data.get('requires_art_handler', False) else 0
            ]),
            is_express=standardized_data.get('days_to_deadline', 14) <= 7
        )
        
        return {
            'risk_assessment': risk_assessment,
            'transport_recommendation': transport_recommendation,
            'time_plan': time_plan,
            'cost_estimation': cost_estimation,
            'price_prediction_details': price_prediction_details
        } 