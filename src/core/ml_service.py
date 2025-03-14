"""
机器学习服务模块

本模块提供机器学习模型的训练、评估和预测功能，用于运输价格预测。
支持多种模型类型，包括线性回归、随机森林、梯度提升树和神经网络。
"""

import os
import pickle
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from src.config import settings

logger = logging.getLogger(__name__)


class MLService:
    """
    机器学习服务类
    
    提供机器学习模型的训练、评估和预测功能，用于运输价格预测。
    """
    
    def __init__(self):
        """初始化机器学习服务"""
        self.model_path = Path(settings.ML_MODEL_PATH)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.preprocessor = None
        self.feature_importance = None
        self.model_info = None
        
        # 尝试加载已有模型
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        加载已训练的模型
        
        Returns:
            是否成功加载模型
        """
        model_file = self.model_path / "price_prediction_model.pkl"
        preprocessor_file = self.model_path / "price_prediction_preprocessor.pkl"
        model_info_file = self.model_path / "price_prediction_model_info.json"
        
        if not model_file.exists() or not preprocessor_file.exists():
            logger.info("未找到已训练的模型文件，需要先训练模型")
            return False
        
        try:
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(preprocessor_file, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            if model_info_file.exists():
                with open(model_info_file, 'r', encoding='utf-8') as f:
                    self.model_info = json.load(f)
                    
                    # 如果存在特征重要性信息，加载它
                    if 'feature_importance' in self.model_info:
                        self.feature_importance = self.model_info['feature_importance']
            
            logger.info(f"成功加载模型: {type(self.model).__name__}")
            return True
        
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return False
    
    def _save_model(self) -> bool:
        """
        保存训练好的模型
        
        Returns:
            是否成功保存模型
        """
        if self.model is None or self.preprocessor is None:
            logger.error("没有可保存的模型")
            return False
        
        try:
            model_file = self.model_path / "price_prediction_model.pkl"
            preprocessor_file = self.model_path / "price_prediction_preprocessor.pkl"
            model_info_file = self.model_path / "price_prediction_model_info.json"
            
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(preprocessor_file, 'wb') as f:
                pickle.dump(self.preprocessor, f)
            
            # 保存模型信息
            if self.model_info:
                with open(model_info_file, 'w', encoding='utf-8') as f:
                    json.dump(self.model_info, f, ensure_ascii=False, indent=2)
            
            logger.info(f"模型保存成功: {model_file}")
            return True
        
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            return False
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练数据
        
        Args:
            data: 原始数据DataFrame
            
        Returns:
            特征数据和目标变量
        """
        # 确保数据中包含必要的列
        required_columns = [
            'origin_location_id', 'destination_location_id', 'transport_mode_id', 
            'cargo_type_id', 'weight', 'volume', 'distance', 'total_price'
        ]
        
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据中缺少必要的列: {col}")
        
        # 提取特征和目标变量
        X = data.drop(['total_price', 'id', 'quote_date', 'currency', 'special_requirements', 'is_llm_assisted'], 
                      axis=1, errors='ignore')
        y = data['total_price']
        
        return X, y
    
    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        创建数据预处理器
        
        Args:
            X: 特征数据
            
        Returns:
            列转换器
        """
        # 识别数值特征和分类特征
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 对于ID类型的列，也视为分类特征
        id_features = [col for col in X.columns if col.endswith('_id') and col in numeric_features]
        for feat in id_features:
            numeric_features.remove(feat)
            categorical_features.append(feat)
        
        # 创建预处理管道
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 组合预处理器
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # 丢弃其他列
        )
        
        return preprocessor
    
    def train(
        self, 
        data: pd.DataFrame, 
        model_type: str = 'random_forest',
        test_size: float = 0.2,
        random_state: int = 42,
        hyperparameter_tuning: bool = False
    ) -> Dict[str, Any]:
        """
        训练价格预测模型
        
        Args:
            data: 训练数据
            model_type: 模型类型，可选 'linear', 'random_forest', 'gradient_boosting', 'neural_network'
            test_size: 测试集比例
            random_state: 随机种子
            hyperparameter_tuning: 是否进行超参数调优
            
        Returns:
            训练结果，包括评估指标
        """
        logger.info(f"开始训练 {model_type} 模型")
        
        # 准备数据
        X, y = self._prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # 创建预处理器
        self.preprocessor = self._create_preprocessor(X)
        
        # 选择模型
        if model_type == 'linear':
            model = LinearRegression()
            param_grid = {}  # 线性回归没有需要调优的超参数
        
        elif model_type == 'random_forest':
            model = RandomForestRegressor(random_state=random_state)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=random_state)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        
        elif model_type == 'neural_network':
            model = MLPRegressor(random_state=random_state, max_iter=1000)
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 创建完整的管道
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', model)
        ])
        
        # 训练模型（可选超参数调优）
        start_time = datetime.now()
        
        if hyperparameter_tuning and param_grid:
            logger.info("开始超参数调优")
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logger.info(f"最佳参数: {best_params}")
        else:
            self.model = pipeline.fit(X_train, y_train)
            best_params = {}
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 计算特征重要性（如果模型支持）
        self.feature_importance = self._get_feature_importance()
        
        # 保存模型信息
        self.model_info = {
            'model_type': model_type,
            'training_date': datetime.now().isoformat(),
            'training_data_size': len(data),
            'training_time_seconds': training_time,
            'test_size': test_size,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'best_params': best_params
        }
        
        if self.feature_importance:
            self.model_info['feature_importance'] = self.feature_importance
        
        # 保存模型
        self._save_model()
        
        # 返回训练结果
        result = {
            'model_type': model_type,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'training_time_seconds': training_time,
            'feature_importance': self.feature_importance
        }
        
        logger.info(f"模型训练完成，RMSE: {rmse:.2f}, R²: {r2:.4f}")
        return result
    
    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典，如果模型不支持则返回None
        """
        if self.model is None:
            return None
        
        # 获取最终模型（如果是Pipeline）
        if isinstance(self.model, Pipeline):
            model = self.model.named_steps['model']
        else:
            model = self.model
        
        # 检查模型是否支持特征重要性
        if not hasattr(model, 'feature_importances_'):
            return None
        
        # 获取特征名称
        feature_names = []
        
        # 如果是Pipeline，从preprocessor中获取特征名称
        if isinstance(self.model, Pipeline) and 'preprocessor' in self.model.named_steps:
            preprocessor = self.model.named_steps['preprocessor']
            
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
            else:
                # 旧版本sklearn的兼容处理
                feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
        else:
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
        
        # 创建特征重要性字典
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        
        # 按重要性排序
        importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
        
        return importance_dict
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用模型进行价格预测
        
        Args:
            features: 特征数据字典
            
        Returns:
            预测结果，包括预测价格和置信度
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先训练模型")
        
        try:
            # 将特征字典转换为DataFrame
            df = pd.DataFrame([features])
            
            # 进行预测
            predicted_price = self.model.predict(df)[0]
            
            # 计算置信度（简单实现，实际应用中可能需要更复杂的方法）
            confidence = 0.8  # 默认置信度
            
            if self.model_info and 'metrics' in self.model_info:
                # 基于模型R²调整置信度
                r2 = self.model_info['metrics'].get('r2', 0)
                confidence = max(0.5, min(0.95, r2))
            
            return {
                'predicted_price': float(predicted_price),
                'confidence': confidence,
                'model_type': self.model_info.get('model_type') if self.model_info else None
            }
        
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            raise
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        评估模型在测试数据上的性能
        
        Args:
            test_data: 测试数据
            
        Returns:
            评估结果，包括各种指标
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先训练模型")
        
        # 准备数据
        X, y = self._prepare_data(test_data)
        
        # 进行预测
        y_pred = self.model.predict(X)
        
        # 计算评估指标
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # 计算预测误差百分比
        error_percent = np.abs((y - y_pred) / y) * 100
        mean_error_percent = np.mean(error_percent)
        median_error_percent = np.median(error_percent)
        
        # 计算预测在不同误差范围内的比例
        within_5_percent = np.mean(error_percent <= 5)
        within_10_percent = np.mean(error_percent <= 10)
        within_20_percent = np.mean(error_percent <= 20)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mean_error_percent': mean_error_percent,
            'median_error_percent': median_error_percent,
            'within_5_percent': within_5_percent,
            'within_10_percent': within_10_percent,
            'within_20_percent': within_20_percent
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        if self.model is None:
            return {'status': 'not_trained'}
        
        info = {
            'status': 'trained',
            'model_type': type(self.model).__name__
        }
        
        if self.model_info:
            info.update(self.model_info)
        
        return info
    
    def prepare_training_data(self, quotes_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        准备训练数据
        
        Args:
            quotes_data: 报价数据列表
            
        Returns:
            处理后的DataFrame
        """
        # 转换为DataFrame
        df = pd.DataFrame(quotes_data)
        
        # 确保数据类型正确
        numeric_cols = ['weight', 'volume', 'distance', 'typical_transit_time', 'total_price']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 处理日期列
        if 'quote_date' in df.columns:
            df['quote_date'] = pd.to_datetime(df['quote_date'], errors='coerce')
            
            # 提取日期特征
            df['quote_year'] = df['quote_date'].dt.year
            df['quote_month'] = df['quote_date'].dt.month
            df['quote_day'] = df['quote_date'].dt.day
            df['quote_dayofweek'] = df['quote_date'].dt.dayofweek
        
        # 处理缺失值
        df = df.dropna(subset=['total_price'])  # 删除目标变量缺失的行
        
        # 对于其他缺失值，可以根据具体情况填充
        df = df.fillna({
            'weight': df['weight'].median(),
            'volume': df['volume'].median(),
            'distance': df['distance'].median(),
            'typical_transit_time': df['typical_transit_time'].median()
        })
        
        return df 