"""
艺术品运输价格预测模型测试模块
"""
import unittest
import numpy as np
import pandas as pd
from src.models.art_transport_price_model import ArtTransportPriceModel

class TestArtTransportPriceModel(unittest.TestCase):
    """测试艺术品运输价格预测模型"""
    
    def setUp(self):
        """测试前准备"""
        self.model = ArtTransportPriceModel()
        
        # 创建测试数据
        np.random.seed(42)
        n_samples = 100
        
        self.test_data = pd.DataFrame({
            'Chargeable_Weight': np.random.uniform(10, 1000, n_samples),
            'Volume': np.random.uniform(1000, 100000, n_samples),
            'Is_International': np.random.randint(0, 2, n_samples),
            'Value_Category': np.random.randint(1, 6, n_samples),
            'Special_Handling_Cost': np.random.uniform(0, 2500, n_samples),
            'Days_To_Deadline': np.random.randint(1, 30, n_samples),
            'Recommend_Express': np.random.randint(0, 2, n_samples),
            'Type_painting': np.random.randint(0, 2, n_samples),
            'Type_sculpture': np.random.randint(0, 2, n_samples),
            'Type_installation': np.random.randint(0, 2, n_samples)
        })
        
        # 创建模拟的价格数据（基于特征的线性组合加上随机噪声）
        self.test_prices = (
            2 * self.test_data['Chargeable_Weight'] +
            0.01 * self.test_data['Volume'] +
            5000 * self.test_data['Is_International'] +
            1000 * self.test_data['Value_Category'] +
            1.5 * self.test_data['Special_Handling_Cost'] +
            100 * self.test_data['Days_To_Deadline'] +
            2000 * self.test_data['Recommend_Express'] +
            np.random.normal(0, 1000, n_samples)
        )
    
    def test_prepare_features(self):
        """测试特征准备"""
        features = self.model.prepare_features(self.test_data)
        
        # 检查是否包含所有预期的特征列
        expected_columns = set(self.model.feature_columns + 
                             ['Type_painting', 'Type_sculpture', 'Type_installation'])
        self.assertEqual(set(features.columns), expected_columns)
        
        # 检查数据类型
        self.assertTrue(all(features.dtypes != 'object'))
    
    def test_train(self):
        """测试模型训练"""
        # 准备特征
        X = self.model.prepare_features(self.test_data)
        y = self.test_prices
        
        # 训练模型
        metrics = self.model.train(X, y)
        
        # 检查训练指标
        self.assertIn('ridge', metrics)
        self.assertIn('gb', metrics)
        
        for model_type in ['ridge', 'gb']:
            self.assertIn('train_mse', metrics[model_type])
            self.assertIn('train_r2', metrics[model_type])
            self.assertIn('cv_scores', metrics[model_type])
            
            # R2分数应该大于0（模型比平均值预测更好）
            self.assertGreater(metrics[model_type]['train_r2'], 0)
    
    def test_predict(self):
        """测试预测功能"""
        # 准备数据并训练模型
        X = self.model.prepare_features(self.test_data)
        y = self.test_prices
        self.model.train(X, y)
        
        # 进行预测
        predictions = self.model.predict(X)
        
        # 检查预测结果
        self.assertIn('ridge', predictions)
        self.assertIn('gb', predictions)
        
        for model_type in ['ridge', 'gb']:
            # 检查预测值的形状
            self.assertEqual(len(predictions[model_type]), len(X))
            # 检查预测值是否为数值型
            self.assertTrue(np.issubdtype(predictions[model_type].dtype, np.number))
    
    def test_predict_with_confidence(self):
        """测试带置信区间的预测"""
        # 准备数据并训练模型
        X = self.model.prepare_features(self.test_data)
        y = self.test_prices
        self.model.train(X, y)
        
        # 进行预测
        result = self.model.predict_with_confidence(X)
        
        # 检查结果结构
        self.assertIn('prediction', result)
        self.assertIn('confidence_interval', result)
        self.assertIn('individual_predictions', result)
        
        # 检查置信区间
        self.assertIn('lower', result['confidence_interval'])
        self.assertIn('upper', result['confidence_interval'])
        
        # 验证置信区间的合理性
        self.assertTrue(all(
            result['confidence_interval']['lower'] <= result['prediction']
        ))
        self.assertTrue(all(
            result['confidence_interval']['upper'] >= result['prediction']
        ))
    
    def test_feature_importance(self):
        """测试特征重要性计算"""
        # 准备数据并训练模型
        X = self.model.prepare_features(self.test_data)
        y = self.test_prices
        self.model.train(X, y)
        
        # 获取特征重要性
        importance = self.model.get_feature_importance()
        
        # 检查结果
        self.assertIn('ridge', importance)
        self.assertIn('gb', importance)
        
        for model_type in ['ridge', 'gb']:
            # 检查是否包含所有特征
            self.assertEqual(len(importance[model_type]), len(self.model.feature_columns))
            # 检查是否按重要性排序
            self.assertTrue(importance[model_type].is_monotonic_decreasing)
    
    def test_save_load_models(self, tmp_path='tmp'):
        """测试模型的保存和加载"""
        import os
        import shutil
        
        # 创建临时目录
        os.makedirs(tmp_path, exist_ok=True)
        
        try:
            # 准备数据并训练模型
            X = self.model.prepare_features(self.test_data)
            y = self.test_prices
            self.model.train(X, y)
            
            # 保存模型
            self.model.save_models(f"{tmp_path}/model")
            
            # 检查模型文件是否存在
            self.assertTrue(os.path.exists(f"{tmp_path}/model_ridge.joblib"))
            self.assertTrue(os.path.exists(f"{tmp_path}/model_gb.joblib"))
            
            # 加载模型
            loaded_model = ArtTransportPriceModel.load_models(f"{tmp_path}/model")
            
            # 比较原始模型和加载的模型的预测结果
            original_pred = self.model.predict(X)
            loaded_pred = loaded_model.predict(X)
            
            np.testing.assert_array_almost_equal(
                original_pred['ridge'],
                loaded_pred['ridge']
            )
            np.testing.assert_array_almost_equal(
                original_pred['gb'],
                loaded_pred['gb']
            )
        
        finally:
            # 清理临时文件
            shutil.rmtree(tmp_path)

if __name__ == '__main__':
    unittest.main() 