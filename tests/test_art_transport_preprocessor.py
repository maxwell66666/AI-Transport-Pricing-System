"""
艺术品运输预处理器测试模块
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.preprocessing.art_transport_preprocessor import ArtTransportPreprocessor

class TestArtTransportPreprocessor(unittest.TestCase):
    """测试艺术品运输预处理器"""
    
    def setUp(self):
        """测试前准备"""
        self.preprocessor = ArtTransportPreprocessor()
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'Length': [100, 200, 150],
            'Width': [50, 100, 75],
            'Height': [30, 60, 45],
            'Actual_Weight': [20, 40, 30],
            'Artwork_Value': [5000, 75000, 200000],
            'Origin_Country': ['CN', 'CN', 'US'],
            'Destination_Country': ['CN', 'US', 'JP'],
            'Artwork_Type': ['painting', 'sculpture', 'installation'],
            'Requires_Climate_Control': [True, False, True],
            'Requires_Custom_Crating': [False, True, True],
            'Requires_Art_Handler': [True, True, False],
            'Quote_Date': ['2024-03-01', '2024-03-02', '2024-03-03'],
            'Deadline': ['2024-03-10', '2024-03-15', '2024-03-05']
        })
    
    def test_calculate_volume_weight(self):
        """测试体积重量计算"""
        length, width, height = 100, 50, 30
        expected_volume_weight = (length * width * height) / 6000
        result = self.preprocessor.calculate_volume_weight(length, width, height)
        self.assertEqual(result, expected_volume_weight)
    
    def test_determine_chargeable_weight(self):
        """测试计费重量确定"""
        actual_weight = 20
        volume_weight = 25
        result = self.preprocessor.determine_chargeable_weight(
            actual_weight, volume_weight
        )
        self.assertEqual(result, volume_weight)
    
    def test_calculate_insurance_amount(self):
        """测试保险金额计算"""
        artwork_value = 10000
        expected_insurance = artwork_value * 1.1
        result = self.preprocessor.calculate_insurance_amount(artwork_value)
        self.assertEqual(result, expected_insurance)
    
    def test_calculate_special_handling_cost(self):
        """测试特殊处理成本计算"""
        # 测试所有特殊处理都需要的情况
        result = self.preprocessor.calculate_special_handling_cost(
            True, True, True
        )
        expected_cost = (
            self.preprocessor.special_handling_costs['climate_control'] +
            self.preprocessor.special_handling_costs['custom_crating'] +
            self.preprocessor.special_handling_costs['art_handler']
        )
        self.assertEqual(result, expected_cost)
        
        # 测试不需要任何特殊处理的情况
        result = self.preprocessor.calculate_special_handling_cost(
            False, False, False
        )
        self.assertEqual(result, 0.0)
    
    def test_calculate_days_to_deadline(self):
        """测试截止日期天数计算"""
        quote_date = '2024-03-01'
        deadline = '2024-03-10'
        expected_days = 9
        result = self.preprocessor.calculate_days_to_deadline(
            quote_date, deadline
        )
        self.assertEqual(result, expected_days)
    
    def test_preprocess_dataframe(self):
        """测试数据框预处理"""
        result_df = self.preprocessor.preprocess_dataframe(self.test_data)
        
        # 打印结果数据框的列名，用于调试
        print("\n预处理后的列名:", list(result_df.columns))
        
        # 检查是否包含所有预期的新列
        expected_columns = {
            'Volume',
            'Volume_Weight',
            'Chargeable_Weight',
            'Is_International',
            'Value_Category',
            'Suggested_Insurance',
            'Special_Handling_Cost',
            'Days_To_Deadline',
            'Recommend_Express',
            'Type_painting',
            'Type_sculpture',
            'Type_installation'
        }
        
        # 打印缺失的列，用于调试
        missing_columns = [col for col in expected_columns if col not in result_df.columns]
        print("缺失的列:", missing_columns)
        
        self.assertTrue(all(col in result_df.columns for col in expected_columns))
        
        # 检查国际运输标志
        self.assertEqual(result_df.iloc[0]['Is_International'], 0)  # 国内运输
        self.assertEqual(result_df.iloc[1]['Is_International'], 1)  # 国际运输
        
        # 检查艺术品类型独热编码
        self.assertEqual(result_df.iloc[0]['Type_painting'], 1)
        self.assertEqual(result_df.iloc[1]['Type_sculpture'], 1)
        self.assertEqual(result_df.iloc[2]['Type_installation'], 1)
        
        # 检查快递服务推荐
        self.assertTrue(all(isinstance(x, (int, np.integer)) 
                          for x in result_df['Recommend_Express']))

if __name__ == '__main__':
    unittest.main() 