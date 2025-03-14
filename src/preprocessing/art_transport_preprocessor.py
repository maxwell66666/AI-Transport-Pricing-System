"""
艺术品运输数据预处理器
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List
from datetime import datetime

class ArtTransportPreprocessor:
    """艺术品运输数据预处理器类"""
    
    def __init__(self):
        """初始化预处理器"""
        self.column_mapping = {
            'artwork_value': 'Artwork_Value',
            'fragility_score': 'Fragility_Score',
            'days_to_deadline': 'Days_To_Deadline',
            'is_international': 'Is_International',
            'chargeable_weight': 'Chargeable_Weight',
            'requires_climate_control': 'Requires_Climate_Control',
            'requires_custom_crating': 'Requires_Custom_Crating',
            'requires_art_handler': 'Requires_Art_Handler',
            'Length': 'Length',
            'Width': 'Width',
            'Height': 'Height',
            'Actual_Weight': 'Actual_Weight',
            'Origin_Country': 'Origin_Country',
            'Destination_Country': 'Destination_Country',
            'Artwork_Type': 'Artwork_Type',
            'Quote_Date': 'Quote_Date',
            'Deadline': 'Deadline'
        }
        
        # 特殊处理成本
        self.special_handling_costs = {
            'climate_control': 500.0,
            'custom_crating': 1200.0,
            'art_handler': 800.0
        }
        
        # 价值类别阈值
        self.value_thresholds = [
            (0, 5000, 1),       # 0-5000: 类别1
            (5000, 20000, 2),    # 5000-20000: 类别2
            (20000, 50000, 3),   # 20000-50000: 类别3
            (50000, 100000, 4),  # 50000-100000: 类别4
            (100000, float('inf'), 5)  # >100000: 类别5
        ]
    
    def calculate_volume(self, length: float, width: float, height: float) -> float:
        """
        计算体积
        
        Args:
            length: 长度（厘米）
            width: 宽度（厘米）
            height: 高度（厘米）
            
        Returns:
            float: 体积（立方厘米）
        """
        return length * width * height
    
    def calculate_volume_weight(self, length: float, width: float, height: float) -> float:
        """
        计算体积重量
        
        Args:
            length: 长度（厘米）
            width: 宽度（厘米）
            height: 高度（厘米）
            
        Returns:
            float: 体积重量（千克）
        """
        # 体积重量系数：1立方米 = 167千克，或者6000立方厘米 = 1千克
        volume = self.calculate_volume(length, width, height)
        return volume / 6000
    
    def determine_chargeable_weight(self, actual_weight: float, volume_weight: float) -> float:
        """
        确定计费重量
        
        Args:
            actual_weight: 实际重量（千克）
            volume_weight: 体积重量（千克）
            
        Returns:
            float: 计费重量（千克）
        """
        return max(actual_weight, volume_weight)
    
    def calculate_insurance_amount(self, artwork_value: float) -> float:
        """
        计算建议保险金额
        
        Args:
            artwork_value: 艺术品价值
            
        Returns:
            float: 建议保险金额（艺术品价值的110%）
        """
        return artwork_value * 1.1
    
    def calculate_special_handling_cost(self, 
                                      requires_climate_control: bool, 
                                      requires_custom_crating: bool, 
                                      requires_art_handler: bool) -> float:
        """
        计算特殊处理成本
        
        Args:
            requires_climate_control: 是否需要温控
            requires_custom_crating: 是否需要定制包装
            requires_art_handler: 是否需要艺术品处理人员
            
        Returns:
            float: 特殊处理总成本
        """
        total_cost = 0.0
        
        if requires_climate_control:
            total_cost += self.special_handling_costs['climate_control']
        
        if requires_custom_crating:
            total_cost += self.special_handling_costs['custom_crating']
        
        if requires_art_handler:
            total_cost += self.special_handling_costs['art_handler']
        
        return total_cost
    
    def calculate_days_to_deadline(self, quote_date: str, deadline: str) -> int:
        """
        计算距离截止日期的天数
        
        Args:
            quote_date: 报价日期（格式：YYYY-MM-DD）
            deadline: 截止日期（格式：YYYY-MM-DD）
            
        Returns:
            int: 距离截止日期的天数
        """
        quote_date_obj = datetime.strptime(quote_date, '%Y-%m-%d')
        deadline_obj = datetime.strptime(deadline, '%Y-%m-%d')
        
        delta = deadline_obj - quote_date_obj
        return delta.days
    
    def determine_value_category(self, artwork_value: float) -> int:
        """
        确定艺术品价值类别
        
        Args:
            artwork_value: 艺术品价值
            
        Returns:
            int: 价值类别（1-5）
        """
        for min_val, max_val, category in self.value_thresholds:
            if min_val <= artwork_value < max_val:
                return category
        
        return 5  # 默认最高类别
    
    def standardize_artwork_value(self, value: float) -> float:
        """
        标准化艺术品价值
        
        Args:
            value: 艺术品价值
            
        Returns:
            float: 标准化后的价值（0-1之间）
        """
        # 使用对数变换处理大范围的价值
        log_value = np.log1p(value)
        # 假设最大价值为1亿
        max_log_value = np.log1p(100000000)
        return log_value / max_log_value
    
    def standardize_weight(self, weight: float) -> float:
        """
        标准化重量
        
        Args:
            weight: 重量（千克）
            
        Returns:
            float: 标准化后的重量（0-1之间）
        """
        # 假设最大重量为10000千克
        return min(weight / 10000, 1)
    
    def encode_countries(self, origin: str, destination: str) -> Dict[str, int]:
        """
        编码国家信息
        
        Args:
            origin: 起始国家代码
            destination: 目的国家代码
            
        Returns:
            Dict[str, int]: 编码后的国家信息
        """
        # 简单的示例编码，实际应使用更复杂的编码方案
        return {
            'same_continent': 1 if origin[:2] == destination[:2] else 0,
            'Is_International': 1 if origin != destination else 0  # 注意这里使用大写I
        }
    
    def encode_artwork_type(self, artwork_type: str) -> Dict[str, int]:
        """
        编码艺术品类型
        
        Args:
            artwork_type: 艺术品类型
            
        Returns:
            Dict[str, int]: 编码后的艺术品类型信息
        """
        type_categories = {
            'painting': {'is_2d': 1, 'fragility': 0.8},
            'sculpture': {'is_2d': 0, 'fragility': 0.9},
            'installation': {'is_2d': 0, 'fragility': 1.0},
            'photography': {'is_2d': 1, 'fragility': 0.6},
            'print': {'is_2d': 1, 'fragility': 0.5}
        }
        return type_categories.get(artwork_type.lower(), {'is_2d': 0, 'fragility': 0.7})
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据框
        
        Args:
            df: 输入数据框
            
        Returns:
            pd.DataFrame: 预处理后的数据框
        """
        # 重命名列
        df = df.rename(columns=self.column_mapping)
        
        # 计算体积和重量相关指标
        df['Volume'] = df.apply(lambda row: self.calculate_volume(
            row['Length'], row['Width'], row['Height']
        ), axis=1)
        
        df['Volume_Weight'] = df.apply(lambda row: self.calculate_volume_weight(
            row['Length'], row['Width'], row['Height']
        ), axis=1)
        
        df['Chargeable_Weight'] = df.apply(lambda row: self.determine_chargeable_weight(
            row['Actual_Weight'], row['Volume_Weight']
        ), axis=1)
        
        # 标准化数值特征
        df['Artwork_Value_Normalized'] = df['Artwork_Value'].apply(self.standardize_artwork_value)
        df['Weight_Normalized'] = df['Chargeable_Weight'].apply(self.standardize_weight)
        
        # 处理国家信息
        country_info = df.apply(lambda row: self.encode_countries(
            row['Origin_Country'], row['Destination_Country']
        ), axis=1)
        df = pd.concat([df, pd.DataFrame(country_info.tolist())], axis=1)
        
        # 处理艺术品类型
        artwork_type_info = df['Artwork_Type'].apply(self.encode_artwork_type)
        df = pd.concat([df, pd.DataFrame(artwork_type_info.tolist())], axis=1)
        
        # 添加艺术品类型独热编码
        artwork_types = ['painting', 'sculpture', 'installation']
        for art_type in artwork_types:
            df[f'Type_{art_type}'] = df['Artwork_Type'].apply(
                lambda x: 1 if x.lower() == art_type else 0
            )
        
        # 计算价值类别
        df['Value_Category'] = df['Artwork_Value'].apply(self.determine_value_category)
        
        # 计算建议保险金额
        df['Suggested_Insurance'] = df['Artwork_Value'].apply(self.calculate_insurance_amount)
        
        # 计算特殊处理成本
        df['Special_Handling_Cost'] = df.apply(
            lambda row: self.calculate_special_handling_cost(
                row.get('Requires_Climate_Control', False),
                row.get('Requires_Custom_Crating', False),
                row.get('Requires_Art_Handler', False)
            ), axis=1
        )
        
        # 计算距离截止日期的天数（如果有相关列）
        if 'Quote_Date' in df.columns and 'Deadline' in df.columns:
            df['Days_To_Deadline'] = df.apply(
                lambda row: self.calculate_days_to_deadline(
                    row['Quote_Date'], row['Deadline']
                ), axis=1
            )
        
        # 添加快递服务推荐
        df['Recommend_Express'] = df.apply(
            lambda row: 1 if row.get('Days_To_Deadline', 14) <= 7 else 0, 
            axis=1
        )
        
        return df
    
    def preprocess_single_item(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理单个艺术品数据
        
        Args:
            data: 输入数据字典
            
        Returns:
            Dict[str, Any]: 预处理后的数据字典
        """
        # 创建单行数据框
        df = pd.DataFrame([data])
        
        # 使用数据框预处理方法
        processed_df = self.preprocess_dataframe(df)
        
        # 转换回字典
        return processed_df.iloc[0].to_dict()
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据字典
            
        Returns:
            bool: 数据是否有效
        """
        required_fields = [
            'artwork_value',
            'Length',
            'Width',
            'Height',
            'Actual_Weight',
            'Origin_Country',
            'Destination_Country',
            'Artwork_Type'
        ]
        
        # 检查必需字段
        for field in required_fields:
            if field not in data:
                return False
        
        # 验证数值字段为正数
        numeric_fields = ['artwork_value', 'Length', 'Width', 'Height', 'Actual_Weight']
        for field in numeric_fields:
            if not isinstance(data[field], (int, float)) or data[field] <= 0:
                return False
        
        # 验证国家代码
        if not isinstance(data['Origin_Country'], str) or not isinstance(data['Destination_Country'], str):
            return False
        
        # 验证艺术品类型
        valid_types = ['painting', 'sculpture', 'installation', 'photography', 'print']
        if data['Artwork_Type'].lower() not in valid_types:
            return False
        
        return True 