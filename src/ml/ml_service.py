"""
机器学习服务模块

此模块提供机器学习服务，用于集成机器学习模型到API中，提供价格预测和分析功能。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import os
import json
import time
from datetime import datetime

from src.ml.price_predictor import PricePredictor
from src.ml.data_processor import DataProcessor
from src.db.models import Quote, QuoteDetail, TransportMode, CargoType, Location, DistanceMatrix
from src.db.repository import QuoteRepository, TransportModeRepository, CargoTypeRepository, LocationRepository
from src.utils.exceptions import MLModelError, DataProcessingError
from src.config import settings

# 配置日志
logger = logging.getLogger(__name__)


class MLService:
    """机器学习服务类，提供价格预测和分析功能"""
    
    def __init__(self, model_dir: str = None):
        """
        初始化机器学习服务
        
        Args:
            model_dir: 模型目录路径，如果为None，则使用配置中的路径
        """
        self.model_dir = model_dir or settings.ML_MODEL_DIR
        self.data_dir = settings.DATA_DIR
        self.predictor = None
        self.models = {}
        self.model_metrics = {}
        self.default_model = settings.ML_DEFAULT_MODEL
        
        # 确保模型目录存在
        Path(self.model_dir).mkdir(exist_ok=True, parents=True)
        
        # 初始化预测器
        self._initialize_predictor()
    
    def _initialize_predictor(self) -> None:
        """初始化价格预测器"""
        try:
            logger.info("初始化价格预测器")
            self.predictor = PricePredictor(data_dir=self.data_dir, model_dir=self.model_dir)
            
            # 加载所有可用模型
            self._load_available_models()
            
            logger.info(f"价格预测器初始化完成，已加载 {len(self.models)} 个模型")
        except Exception as e:
            logger.error(f"初始化价格预测器失败: {str(e)}")
            raise MLModelError(f"初始化价格预测器失败: {str(e)}")
    
    def _load_available_models(self) -> None:
        """加载所有可用的模型"""
        try:
            # 获取模型目录中的所有模型文件
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
            
            for model_file in model_files:
                model_name = model_file.replace('.pkl', '')
                model = self.predictor.load_model(model_name)
                if model:
                    self.models[model_name] = model
                    logger.info(f"已加载模型: {model_name}")
                    
                    # 加载模型指标（如果存在）
                    metrics_file = Path(self.model_dir) / f"{model_name}_metrics.json"
                    if metrics_file.exists():
                        with open(metrics_file, 'r') as f:
                            self.model_metrics[model_name] = json.load(f)
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise MLModelError(f"加载模型失败: {str(e)}")
    
    def predict_price(self, 
                      origin_id: int, 
                      destination_id: int,
                      transport_mode_id: int,
                      cargo_type_id: int,
                      weight: float,
                      volume: float,
                      distance: float = None,
                      special_requirements: List[str] = None,
                      model_name: str = None) -> Dict[str, Any]:
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
            
        Returns:
            包含预测价格和相关信息的字典
        """
        start_time = time.time()
        
        try:
            # 使用默认模型（如果未指定）
            if model_name is None:
                model_name = self.default_model
                
            # 检查模型是否已加载
            if model_name not in self.models:
                logger.warning(f"模型 {model_name} 未加载，尝试加载")
                model = self.predictor.load_model(model_name)
                if model:
                    self.models[model_name] = model
                else:
                    raise MLModelError(f"模型 {model_name} 不存在")
            
            # 准备特征
            features = self._prepare_prediction_features(
                origin_id=origin_id,
                destination_id=destination_id,
                transport_mode_id=transport_mode_id,
                cargo_type_id=cargo_type_id,
                weight=weight,
                volume=volume,
                distance=distance,
                special_requirements=special_requirements
            )
            
            # 预测价格
            predicted_price = self.predictor.predict_price(model_name, features)
            
            # 计算置信区间（简化版）
            confidence_interval = self._calculate_confidence_interval(model_name, predicted_price)
            
            # 获取特征重要性（如果可用）
            feature_importance = self._get_feature_importance(model_name)
            
            # 构建响应
            response = {
                "predicted_price": round(predicted_price, 2),
                "currency": "CNY",
                "confidence_interval": confidence_interval,
                "model_used": model_name,
                "feature_importance": feature_importance,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            logger.info(f"价格预测完成: {response}")
            return response
            
        except Exception as e:
            logger.error(f"价格预测失败: {str(e)}")
            raise MLModelError(f"价格预测失败: {str(e)}")
    
    def _prepare_prediction_features(self, 
                                    origin_id: int, 
                                    destination_id: int,
                                    transport_mode_id: int,
                                    cargo_type_id: int,
                                    weight: float,
                                    volume: float,
                                    distance: float = None,
                                    special_requirements: List[str] = None) -> Dict[str, Any]:
        """
        准备预测所需的特征
        
        Args:
            origin_id: 起始地点ID
            destination_id: 目的地ID
            transport_mode_id: 运输方式ID
            cargo_type_id: 货物类型ID
            weight: 重量（千克）
            volume: 体积（立方米）
            distance: 距离（公里）
            special_requirements: 特殊要求列表
            
        Returns:
            特征字典
        """
        try:
            # 如果未提供距离，则从数据库或CSV文件中获取
            if distance is None:
                try:
                    # 尝试从数据库获取距离
                    from src.db.database import get_db
                    
                    db = next(get_db())
                    distance_record = db.query(DistanceMatrix).filter(
                        DistanceMatrix.origin_location_id == origin_id,
                        DistanceMatrix.destination_location_id == destination_id,
                        DistanceMatrix.transport_mode_id == transport_mode_id
                    ).first()
                    
                    if distance_record:
                        distance = distance_record.distance
                        typical_transit_time = distance_record.typical_transit_time
                    else:
                        # 如果数据库中没有找到，使用默认值
                        distance = 500.0
                        typical_transit_time = self._estimate_transit_time(distance, transport_mode_id)
                except Exception as e:
                    logger.warning(f"从数据库获取距离失败，尝试从CSV文件获取: {e}")
                    
                    # 尝试从CSV文件获取距离
                    try:
                        csv_file = Path(self.data_dir) / "sample_data" / "distance_matrix.csv"
                        if csv_file.exists():
                            df = pd.read_csv(csv_file)
                            
                            # 查找匹配的记录
                            mask = (
                                (df['origin_location_id'] == origin_id) & 
                                (df['destination_location_id'] == destination_id) & 
                                (df['transport_mode_id'] == transport_mode_id)
                            )
                            
                            matching_rows = df[mask]
                            
                            if not matching_rows.empty:
                                row = matching_rows.iloc[0]
                                distance = row['distance']
                                typical_transit_time = row['typical_transit_time']
                            else:
                                # 如果没有找到完全匹配的记录，使用默认值
                                distance = 500.0
                                typical_transit_time = self._estimate_transit_time(distance, transport_mode_id)
                        else:
                            # 如果CSV文件不存在，使用默认值
                            distance = 500.0
                            typical_transit_time = self._estimate_transit_time(distance, transport_mode_id)
                    except Exception as e:
                        logger.warning(f"从CSV文件获取距离失败，使用默认值: {e}")
                        distance = 500.0
                        typical_transit_time = self._estimate_transit_time(distance, transport_mode_id)
            else:
                # 如果提供了距离，估算运输时间
                typical_transit_time = self._estimate_transit_time(distance, transport_mode_id)
            
            # 处理特殊要求
            has_refrigeration = False
            has_fragile = False
            has_dangerous = False
            has_express = False
            
            if special_requirements:
                has_refrigeration = "refrigeration" in special_requirements
                has_fragile = "fragile" in special_requirements
                has_dangerous = "dangerous" in special_requirements
                has_express = "express" in special_requirements
            
            # 构建特征字典
            features = {
                "origin_location_id": origin_id,
                "destination_location_id": destination_id,
                "transport_mode_id": transport_mode_id,
                "cargo_type_id": cargo_type_id,
                "weight": weight,
                "volume": volume,
                "distance": distance,
                "typical_transit_time": typical_transit_time,
                "weight_volume_ratio": weight / max(volume, 0.001),  # 避免除以零
                "has_refrigeration": has_refrigeration,
                "has_fragile": has_fragile,
                "has_dangerous": has_dangerous,
                "has_express": has_express,
                # 添加时间相关特征
                "month": datetime.now().month,
                "day_of_week": datetime.now().weekday(),
                "is_holiday": False  # 简化版：假设不是假日
            }
            
            return features
            
        except Exception as e:
            logger.error(f"准备预测特征失败: {str(e)}")
            raise DataProcessingError(f"准备预测特征失败: {str(e)}")
    
    def _calculate_confidence_interval(self, model_name: str, predicted_price: float) -> Dict[str, float]:
        """
        计算预测价格的置信区间
        
        Args:
            model_name: 模型名称
            predicted_price: 预测价格
            
        Returns:
            包含下限和上限的字典
        """
        # 简化版：使用固定的误差百分比
        # 在实际应用中，应该基于模型的不确定性估计
        error_percentage = 0.1
        
        # 如果有模型指标，使用MAE作为误差估计
        if model_name in self.model_metrics and "mae" in self.model_metrics[model_name]:
            mae = self.model_metrics[model_name]["mae"]
            error_percentage = mae / max(predicted_price, 1.0)
        
        lower_bound = predicted_price * (1 - error_percentage)
        upper_bound = predicted_price * (1 + error_percentage)
        
        return {
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "confidence_level": "90%"  # 简化版：固定置信水平
        }
    
    def _get_feature_importance(self, model_name: str) -> List[Dict[str, Any]]:
        """
        获取特征重要性
        
        Args:
            model_name: 模型名称
            
        Returns:
            特征重要性列表
        """
        # 简化版：返回固定的特征重要性
        # 在实际应用中，应该从模型中提取
        default_importance = [
            {"feature": "distance", "importance": 0.35},
            {"feature": "weight", "importance": 0.25},
            {"feature": "transport_mode_id", "importance": 0.15},
            {"feature": "cargo_type_id", "importance": 0.10},
            {"feature": "volume", "importance": 0.10},
            {"feature": "has_express", "importance": 0.05}
        ]
        
        # 如果有模型指标，使用存储的特征重要性
        if model_name in self.model_metrics and "feature_importance" in self.model_metrics[model_name]:
            return self.model_metrics[model_name]["feature_importance"]
        
        return default_importance
    
    def train_model(self, 
                   model_name: str = "gradient_boosting",
                   use_historical_data: bool = True,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Dict[str, Any]:
        """
        训练新模型
        
        Args:
            model_name: 模型名称
            use_historical_data: 是否使用历史数据
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            训练结果字典
        """
        try:
            logger.info(f"开始训练模型: {model_name}")
            
            # 准备训练数据
            if use_historical_data:
                # 从数据库加载历史报价数据
                X, y = self._load_historical_data()
            else:
                # 使用预处理的数据
                X, y = self.predictor.prepare_features()
            
            # 训练模型
            if model_name == "all":
                # 训练所有模型
                results = self.predictor.train_models(X, y)
                
                # 更新模型和指标
                for model_name, model_info in results.items():
                    self.models[model_name] = model_info["pipeline"]
                    self.model_metrics[model_name] = {
                        "r2": model_info["r2"],
                        "mae": model_info["mae"],
                        "rmse": model_info["rmse"],
                        "feature_importance": model_info.get("feature_importance", [])
                    }
                
                # 保存模型和指标
                self.predictor._save_models()
                self._save_model_metrics()
                
                return {
                    "status": "success",
                    "message": f"成功训练 {len(results)} 个模型",
                    "models": list(results.keys()),
                    "best_model": max(results.items(), key=lambda x: x[1]["r2"])[0],
                    "metrics": {name: {"r2": info["r2"], "mae": info["mae"], "rmse": info["rmse"]} 
                               for name, info in results.items()}
                }
            else:
                # 训练单个模型
                # 这里简化处理，实际应该调用predictor的特定方法
                results = self.predictor.train_models(X, y)
                
                if model_name in results:
                    model_info = results[model_name]
                    self.models[model_name] = model_info["pipeline"]
                    self.model_metrics[model_name] = {
                        "r2": model_info["r2"],
                        "mae": model_info["mae"],
                        "rmse": model_info["rmse"],
                        "feature_importance": model_info.get("feature_importance", [])
                    }
                    
                    # 保存模型和指标
                    self.predictor._save_models()
                    self._save_model_metrics()
                    
                    return {
                        "status": "success",
                        "message": f"成功训练模型 {model_name}",
                        "model": model_name,
                        "metrics": {
                            "r2": model_info["r2"],
                            "mae": model_info["mae"],
                            "rmse": model_info["rmse"]
                        }
                    }
                else:
                    raise MLModelError(f"模型 {model_name} 训练失败")
                
        except Exception as e:
            logger.error(f"训练模型失败: {str(e)}")
            raise MLModelError(f"训练模型失败: {str(e)}")
    
    def _load_historical_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        从数据库加载历史报价数据
        
        Returns:
            特征数据框和目标变量系列
        """
        try:
            # 这里应该从数据库中加载历史报价数据
            # 简化版：使用预处理的数据
            return self.predictor.prepare_features()
        except Exception as e:
            logger.error(f"加载历史数据失败: {str(e)}")
            raise DataProcessingError(f"加载历史数据失败: {str(e)}")
    
    def _save_model_metrics(self) -> None:
        """保存模型指标到文件"""
        try:
            for model_name, metrics in self.model_metrics.items():
                metrics_file = Path(self.model_dir) / f"{model_name}_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"已保存模型 {model_name} 的指标")
        except Exception as e:
            logger.error(f"保存模型指标失败: {str(e)}")
    
    def analyze_price_factors(self, quote_id: int = None) -> Dict[str, Any]:
        """
        分析影响价格的因素
        
        Args:
            quote_id: 报价ID，如果提供，则分析特定报价的因素
            
        Returns:
            分析结果字典
        """
        try:
            # 如果提供了报价ID，分析特定报价
            if quote_id:
                return self._analyze_specific_quote(quote_id)
            
            # 否则，提供一般性分析
            return self._analyze_general_factors()
            
        except Exception as e:
            logger.error(f"分析价格因素失败: {str(e)}")
            raise MLModelError(f"分析价格因素失败: {str(e)}")
    
    def _analyze_specific_quote(self, quote_id: int) -> Dict[str, Any]:
        """
        分析特定报价的因素
        
        Args:
            quote_id: 报价ID
            
        Returns:
            分析结果字典
        """
        # 这里应该从数据库中获取报价详情，并分析影响因素
        # 简化版：返回固定的分析结果
        return {
            "quote_id": quote_id,
            "base_price_factors": {
                "distance": {"value": 500, "impact": "高", "contribution": 0.35},
                "weight": {"value": 1000, "impact": "中", "contribution": 0.25},
                "transport_mode": {"value": "公路运输", "impact": "中", "contribution": 0.15}
            },
            "special_factors": {
                "express_delivery": {"applied": True, "impact": "+15%"},
                "fragile_cargo": {"applied": False, "impact": "0%"}
            },
            "seasonal_factors": {
                "peak_season": {"applied": False, "impact": "0%"},
                "fuel_price": {"applied": True, "impact": "+5%"}
            },
            "recommendations": [
                "选择铁路运输可节省约20%的成本",
                "避开周一发货可减少约5%的费用",
                "增加货物量可获得批量折扣"
            ]
        }
    
    def _analyze_general_factors(self) -> Dict[str, Any]:
        """
        分析一般性价格因素
        
        Returns:
            分析结果字典
        """
        # 简化版：返回固定的分析结果
        return {
            "primary_factors": [
                {"factor": "距离", "average_impact": "每增加100公里增加约10%的成本"},
                {"factor": "重量", "average_impact": "每增加1000千克增加约15%的成本"},
                {"factor": "运输方式", "average_impact": "航空运输比公路运输贵约300%"}
            ],
            "secondary_factors": [
                {"factor": "季节性", "average_impact": "旺季价格上涨约20%"},
                {"factor": "燃油价格", "average_impact": "燃油价格每上涨10%，运输成本上涨约3%"},
                {"factor": "特殊要求", "average_impact": "冷藏要求增加约25%的成本"}
            ],
            "cost_saving_tips": [
                "提前7天预订可节省约8%的成本",
                "选择非高峰时段可节省约10%的成本",
                "合并小件货物可节省约15%的成本"
            ]
        }
    
    def get_price_trends(self, 
                        origin_id: int = None, 
                        destination_id: int = None,
                        transport_mode_id: int = None,
                        cargo_type_id: int = None,
                        period: str = "monthly",
                        months: int = 12) -> Dict[str, Any]:
        """
        获取价格趋势
        
        Args:
            origin_id: 起始地点ID
            destination_id: 目的地ID
            transport_mode_id: 运输方式ID
            cargo_type_id: 货物类型ID
            period: 周期（daily, weekly, monthly, quarterly, yearly）
            months: 月数
            
        Returns:
            价格趋势字典
        """
        try:
            # 这里应该从数据库中获取历史价格数据，并计算趋势
            # 简化版：返回模拟的价格趋势
            
            # 生成时间点
            if period == "monthly":
                time_points = [f"2023-{i:02d}" for i in range(1, 13)]
            elif period == "quarterly":
                time_points = [f"2023-Q{i}" for i in range(1, 5)]
            elif period == "yearly":
                time_points = [str(year) for year in range(2018, 2024)]
            else:
                time_points = [f"2023-{i:02d}" for i in range(1, 13)]
            
            # 生成价格数据（模拟季节性波动）
            base_price = 1000
            seasonal_factor = np.sin(np.linspace(0, 2*np.pi, len(time_points)))
            trend_factor = np.linspace(0, 0.2, len(time_points))  # 上升趋势
            random_factor = np.random.normal(0, 0.05, len(time_points))  # 随机波动
            
            prices = base_price * (1 + 0.15 * seasonal_factor + trend_factor + random_factor)
            prices = [round(p, 2) for p in prices]
            
            # 计算同比和环比变化
            yoy_change = round((prices[-1] / prices[0] - 1) * 100, 2) if len(prices) > 1 else 0
            mom_change = round((prices[-1] / prices[-2] - 1) * 100, 2) if len(prices) > 1 else 0
            
            # 预测未来3个时间点
            future_points = [f"预测-{i+1}" for i in range(3)]
            future_seasonal = np.sin(np.linspace(2*np.pi, 3*np.pi, 3))
            future_trend = np.linspace(trend_factor[-1], trend_factor[-1]+0.1, 3)
            future_random = np.random.normal(0, 0.03, 3)
            
            future_prices = base_price * (1 + 0.15 * future_seasonal + future_trend + future_random)
            future_prices = [round(p, 2) for p in future_prices]
            
            return {
                "filter_criteria": {
                    "origin_id": origin_id,
                    "destination_id": destination_id,
                    "transport_mode_id": transport_mode_id,
                    "cargo_type_id": cargo_type_id,
                    "period": period,
                    "months": months
                },
                "trend_data": {
                    "time_points": time_points,
                    "prices": prices,
                    "currency": "CNY"
                },
                "forecast": {
                    "time_points": future_points,
                    "prices": future_prices,
                    "confidence_interval": {
                        "lower": [round(p*0.9, 2) for p in future_prices],
                        "upper": [round(p*1.1, 2) for p in future_prices]
                    }
                },
                "summary": {
                    "average_price": round(sum(prices) / len(prices), 2),
                    "min_price": round(min(prices), 2),
                    "max_price": round(max(prices), 2),
                    "yoy_change": f"{yoy_change}%",
                    "mom_change": f"{mom_change}%",
                    "trend_direction": "上升" if yoy_change > 0 else "下降",
                    "seasonal_pattern": "有明显季节性波动，第二季度和第四季度价格较高"
                }
            }
            
        except Exception as e:
            logger.error(f"获取价格趋势失败: {str(e)}")
            raise MLModelError(f"获取价格趋势失败: {str(e)}")
    
    def _estimate_transit_time(self, distance: float, transport_mode_id: int) -> float:
        """
        估算运输时间（天）
        
        Args:
            distance: 距离（公里）
            transport_mode_id: 运输方式ID
            
        Returns:
            估算的运输时间（天）
        """
        # 根据运输方式和距离估算运输时间
        # 这里使用简化的估算方法，实际应用中可以使用更复杂的模型
        if transport_mode_id == 1:  # 假设1是公路运输
            # 公路运输：平均速度 60 km/h，每天行驶 10 小时
            return distance / (60 * 10)
        elif transport_mode_id == 2:  # 假设2是铁路运输
            # 铁路运输：平均速度 80 km/h，每天行驶 20 小时
            return distance / (80 * 20)
        elif transport_mode_id == 3:  # 假设3是海运
            # 海运：平均速度 30 km/h，每天行驶 24 小时
            return distance / (30 * 24)
        elif transport_mode_id == 4:  # 假设4是空运
            # 空运：平均速度 800 km/h，加上装卸时间 0.5 天
            return distance / (800 * 10) + 0.5
        else:
            # 默认情况
            return distance / 500  # 简单估算 