"""
报价预测模型模块

此模块包含用于预测运输报价的机器学习模型。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import pickle
import os
from typing import Dict, List, Optional, Union, Tuple, Any

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

from src.ml.data_processor import DataProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PricePredictor:
    """报价预测模型类，用于预测运输报价"""
    
    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        """
        初始化报价预测器
        
        Args:
            data_dir: 数据目录路径
            model_dir: 模型保存目录
        """
        self.data_processor = DataProcessor(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # 模型和预处理器
        self.models = {}
        self.preprocessor = None
        self.feature_importances = {}
        
        # 加载处理后的数据
        self.processed_data = {}
        self._load_processed_data()
    
    def _load_processed_data(self) -> None:
        """加载处理后的数据"""
        # 尝试从处理后的文件加载数据
        processed_dir = self.data_processor.processed_dir
        if processed_dir.exists():
            csv_files = list(processed_dir.glob("processed_*.csv"))
            if csv_files:
                for file_path in csv_files:
                    name = file_path.stem.replace("processed_", "")
                    self.processed_data[name] = pd.read_csv(file_path)
                logger.info(f"从文件加载了 {len(self.processed_data)} 个处理后的数据表")
                return
        
        # 如果没有处理后的文件，则执行处理流程
        logger.info("未找到处理后的数据文件，执行数据处理流程")
        self.processed_data = self.data_processor.process_pipeline()
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备特征和目标变量
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: 特征DataFrame和目标Series
        """
        if 'quotes_enriched' not in self.processed_data:
            logger.error("未找到富集的报价数据，无法准备特征")
            return pd.DataFrame(), pd.Series()
        
        # 使用富集的报价数据
        df = self.processed_data['quotes_enriched'].copy()
        
        # 选择特征
        features = [
            'origin_location_id', 'destination_location_id', 'transport_mode_id',
            'cargo_type_id', 'weight', 'volume', 'distance', 'typical_transit_time'
        ]
        
        # 选择目标变量
        target = 'total_price'
        
        # 检查是否有缺失值
        missing_values = df[features + [target]].isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"数据中存在缺失值:\n{missing_values[missing_values > 0]}")
        
        # 移除包含缺失值的行
        df_clean = df[features + [target]].dropna()
        
        if len(df_clean) < len(df):
            logger.info(f"移除了 {len(df) - len(df_clean)} 行包含缺失值的数据")
        
        X = df_clean[features]
        y = df_clean[target]
        
        logger.info(f"准备了 {len(X)} 行数据，{len(features)} 个特征")
        return X, y
    
    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        构建特征预处理器
        
        Args:
            X: 特征DataFrame
            
        Returns:
            ColumnTransformer: 特征预处理器
        """
        # 区分数值特征和分类特征
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 对于本例，我们将ID视为分类特征
        id_features = [col for col in X.columns if col.endswith('_id')]
        for feature in id_features:
            if feature in numeric_features:
                numeric_features.remove(feature)
                categorical_features.append(feature)
        
        # 构建预处理器
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        logger.info(f"构建了特征预处理器，处理 {len(numeric_features)} 个数值特征和 {len(categorical_features)} 个分类特征")
        return preprocessor
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        训练多个模型并评估性能
        
        Args:
            X: 特征DataFrame
            y: 目标Series
            
        Returns:
            Dict[str, Any]: 包含训练好的模型和评估结果的字典
        """
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 构建预处理器
        self.preprocessor = self.build_preprocessor(X)
        
        # 定义要训练的模型
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42)
        }
        
        # 训练和评估每个模型
        results = {}
        
        for name, model in models.items():
            logger.info(f"训练模型: {name}")
            
            # 构建完整的管道
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('model', model)
            ])
            
            # 训练模型
            pipeline.fit(X_train, y_train)
            
            # 在测试集上评估
            y_pred = pipeline.predict(X_test)
            
            # 计算评估指标
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 保存模型和结果
            self.models[name] = pipeline
            
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model': pipeline
            }
            
            logger.info(f"模型 {name} 评估结果: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
            
            # 对于树模型，提取特征重要性
            if name in ['random_forest', 'gradient_boosting']:
                self._extract_feature_importance(name, pipeline, X.columns)
        
        # 绘制模型性能比较图
        self._plot_model_comparison(results)
        
        # 保存模型
        self._save_models()
        
        return results
    
    def _extract_feature_importance(self, model_name: str, pipeline: Pipeline, feature_names: List[str]) -> None:
        """
        提取特征重要性
        
        Args:
            model_name: 模型名称
            pipeline: 训练好的管道
            feature_names: 原始特征名称
        """
        model = pipeline.named_steps['model']
        
        if hasattr(model, 'feature_importances_'):
            # 获取预处理后的特征名称
            preprocessor = pipeline.named_steps['preprocessor']
            
            # 对于ColumnTransformer，我们需要获取转换后的特征名称
            # 这部分比较复杂，因为OneHotEncoder会创建新的特征名称
            # 这里我们简化处理，直接使用特征重要性值和索引
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # 保存特征重要性
            self.feature_importances[model_name] = {
                'importances': importances,
                'indices': indices
            }
            
            # 绘制特征重要性图
            plt.figure(figsize=(12, 6))
            plt.title(f'{model_name} - 特征重要性')
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), range(1, len(importances) + 1))
            plt.xlabel('特征排名')
            plt.ylabel('重要性')
            plt.tight_layout()
            plt.savefig(self.model_dir / f"{model_name}_feature_importance.png", dpi=300)
            plt.close()
            
            # 保存前10个最重要的特征
            top_n = min(10, len(importances))
            with open(self.model_dir / f"{model_name}_top_features.txt", 'w', encoding='utf-8') as f:
                f.write(f"# {model_name} - 前{top_n}个最重要的特征\n\n")
                for i in range(top_n):
                    f.write(f"{i+1}. 特征索引 {indices[i]}: {importances[indices[i]]:.4f}\n")
    
    def _plot_model_comparison(self, results: Dict[str, Dict]) -> None:
        """
        绘制模型性能比较图
        
        Args:
            results: 包含模型评估结果的字典
        """
        model_names = list(results.keys())
        rmse_values = [results[name]['rmse'] for name in model_names]
        mae_values = [results[name]['mae'] for name in model_names]
        r2_values = [results[name]['r2'] for name in model_names]
        
        # 绘制RMSE和MAE比较图
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        bars = plt.bar(model_names, rmse_values)
        plt.title('模型RMSE比较（越低越好）')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.subplot(2, 1, 2)
        bars = plt.bar(model_names, mae_values)
        plt.title('模型MAE比较（越低越好）')
        plt.ylabel('MAE')
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.model_dir / "model_error_comparison.png", dpi=300)
        plt.close()
        
        # 绘制R²比较图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, r2_values)
        plt.title('模型R²比较（越高越好）')
        plt.ylabel('R²')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.model_dir / "model_r2_comparison.png", dpi=300)
        plt.close()
        
        # 保存比较结果
        comparison_df = pd.DataFrame({
            'model': model_names,
            'rmse': rmse_values,
            'mae': mae_values,
            'r2': r2_values
        })
        comparison_df.to_csv(self.model_dir / "model_comparison.csv", index=False)
    
    def _save_models(self) -> None:
        """保存训练好的模型"""
        for name, model in self.models.items():
            model_path = self.model_dir / f"{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"保存模型 {name} 到 {model_path}")
    
    def load_model(self, model_name: str) -> Optional[Pipeline]:
        """
        加载保存的模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            Optional[Pipeline]: 加载的模型，如果加载失败则返回None
        """
        model_path = self.model_dir / f"{model_name}.pkl"
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"加载模型 {model_name} 从 {model_path}")
            return model
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            return None
    
    def predict_price(self, model_name: str, features: Dict) -> Optional[float]:
        """
        使用指定模型预测价格
        
        Args:
            model_name: 模型名称
            features: 包含特征值的字典
            
        Returns:
            Optional[float]: 预测的价格，如果预测失败则返回None
        """
        try:
            # 加载模型（如果尚未加载）
            if model_name not in self.models:
                model = self.load_model(model_name)
                if model is None:
                    logger.error(f"无法加载模型: {model_name}")
                    return None
                self.models[model_name] = model
            
            # 确保特征包含所有必要的字段
            required_features = [
                'origin_location_id', 'destination_location_id', 'transport_mode_id',
                'cargo_type_id', 'weight', 'volume', 'distance', 'typical_transit_time'
            ]
            
            # 创建特征字典的副本，以便修改
            features_copy = features.copy()
            
            # 检查是否有特征名称不匹配的情况
            feature_mapping = {
                'origin_id': 'origin_location_id',
                'destination_id': 'destination_location_id',
                'transport_mode': 'transport_mode_id',
                'cargo_type': 'cargo_type_id'
            }
            
            # 应用特征映射
            for old_name, new_name in feature_mapping.items():
                if old_name in features_copy and new_name not in features_copy:
                    features_copy[new_name] = features_copy[old_name]
                    logger.info(f"将特征 {old_name} 映射到 {new_name}")
            
            # 检查是否缺少必要的特征
            missing_features = [f for f in required_features if f not in features_copy]
            if missing_features:
                logger.warning(f"缺少必要的特征: {missing_features}")
                
                # 如果缺少 typical_transit_time，添加默认值
                if 'typical_transit_time' in missing_features and 'distance' in features_copy:
                    # 简单估算运输时间
                    distance = features_copy['distance']
                    transport_mode_id = features_copy['transport_mode_id']
                    
                    if transport_mode_id == 1:  # 公路
                        features_copy['typical_transit_time'] = distance / 600
                    elif transport_mode_id == 2:  # 铁路
                        features_copy['typical_transit_time'] = distance / 1600
                    elif transport_mode_id == 3:  # 海运
                        features_copy['typical_transit_time'] = distance / 720
                    elif transport_mode_id == 4:  # 空运
                        features_copy['typical_transit_time'] = distance / 8000 + 0.5
                    else:
                        features_copy['typical_transit_time'] = distance / 500
                    
                    logger.info(f"估算 typical_transit_time: {features_copy['typical_transit_time']}")
                
                # 检查是否仍然缺少必要的特征
                missing_features = [f for f in required_features if f not in features_copy]
                if missing_features:
                    logger.error(f"仍然缺少必要的特征: {missing_features}")
                    return None
            
            # 将特征字典转换为DataFrame
            features_df = pd.DataFrame([features_copy])
            
            # 确保DataFrame包含所有必要的特征列
            for feature in required_features:
                if feature not in features_df.columns:
                    logger.error(f"缺少必要的特征列: {feature}")
                    return None
            
            # 使用模型预测
            try:
                predicted_price = self.models[model_name].predict(features_df)[0]
                logger.info(f"预测价格: {predicted_price}")
                return predicted_price
            except Exception as e:
                logger.error(f"预测过程中出错: {e}")
                return None
        except Exception as e:
            logger.error(f"预测价格时出错: {e}")
            return None
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict:
        """
        对指定模型进行超参数调优
        
        Args:
            X: 特征DataFrame
            y: 目标Series
            model_name: 模型名称
            
        Returns:
            Dict: 包含最佳参数和评估结果的字典
        """
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 确保预处理器已构建
        if self.preprocessor is None:
            self.preprocessor = self.build_preprocessor(X)
        
        # 定义参数网格
        param_grids = {
            'linear_regression': {},  # 线性回归没有需要调优的超参数
            'ridge': {
                'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            'random_forest': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            }
        }
        
        if model_name not in param_grids:
            logger.error(f"不支持的模型名称: {model_name}")
            return {}
        
        # 获取基础模型
        base_models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42)
        }
        
        if model_name not in base_models:
            logger.error(f"不支持的模型名称: {model_name}")
            return {}
        
        # 构建完整的管道
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', base_models[model_name])
        ])
        
        # 使用网格搜索进行超参数调优
        logger.info(f"开始对模型 {model_name} 进行超参数调优")
        grid_search = GridSearchCV(
            pipeline,
            param_grids[model_name],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 获取最佳参数和性能
        best_params = grid_search.best_params_
        best_score = np.sqrt(-grid_search.best_score_)
        
        logger.info(f"模型 {model_name} 的最佳参数: {best_params}")
        logger.info(f"模型 {model_name} 的最佳RMSE: {best_score:.2f}")
        
        # 使用最佳参数在测试集上评估
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 保存调优后的模型
        tuned_model_name = f"{model_name}_tuned"
        self.models[tuned_model_name] = best_model
        
        # 保存模型
        model_path = self.model_dir / f"{tuned_model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        logger.info(f"保存调优后的模型 {tuned_model_name} 到 {model_path}")
        
        # 返回结果
        results = {
            'best_params': best_params,
            'best_score': best_score,
            'test_rmse': rmse,
            'test_mae': mae,
            'test_r2': r2,
            'model': best_model
        }
        
        return results
    
    def run_training_pipeline(self) -> Dict:
        """
        执行完整的模型训练流程
        
        Returns:
            Dict: 包含训练结果的字典
        """
        # 准备特征和目标变量
        X, y = self.prepare_features()
        if X.empty or y.empty:
            logger.error("特征准备失败，无法训练模型")
            return {}
        
        # 训练多个模型并评估
        results = self.train_models(X, y)
        
        # 找出性能最好的模型
        best_model_name = None
        best_r2 = -float('inf')
        
        for name, result in results.items():
            if result['r2'] > best_r2:
                best_r2 = result['r2']
                best_model_name = name
        
        if best_model_name:
            logger.info(f"性能最好的模型是 {best_model_name}，R²={best_r2:.4f}")
            
            # 对最佳模型进行超参数调优
            tuned_results = self.tune_hyperparameters(X, y, best_model_name)
            
            # 比较调优前后的性能
            if tuned_results and 'test_r2' in tuned_results:
                tuned_r2 = tuned_results['test_r2']
                logger.info(f"调优后的模型 R²={tuned_r2:.4f}，相比原始模型 {'+' if tuned_r2 > best_r2 else ''}{tuned_r2 - best_r2:.4f}")
        
        return results


if __name__ == "__main__":
    """直接运行此模块时执行模型训练流程"""
    predictor = PricePredictor()
    results = predictor.run_training_pipeline()
    print(f"模型训练完成，结果保存在 {predictor.model_dir} 目录") 