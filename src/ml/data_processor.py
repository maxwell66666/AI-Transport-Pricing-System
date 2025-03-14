"""
数据处理模块

此模块包含用于清洗和预处理历史运输报价数据的功能。
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理类，用于清洗和预处理历史运输报价数据"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.sample_dir = self.data_dir / "sample_data"
        
        # 确保目录存在
        self.raw_dir.mkdir(exist_ok=True, parents=True)
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        
        # 数据缓存
        self._transport_modes = None
        self._cargo_types = None
        self._locations = None
        self._distance_matrix = None
        self._quotes = None
    
    def load_sample_data(self) -> Dict:
        """
        加载样本数据
        
        Returns:
            Dict: 包含样本数据的字典
        """
        sample_file = self.sample_dir / "sample_transport_data.json"
        if not sample_file.exists():
            logger.error(f"样本数据文件不存在: {sample_file}")
            return {}
        
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载样本数据: {len(data)} 个类别")
            return data
        except Exception as e:
            logger.error(f"加载样本数据时出错: {e}")
            return {}
    
    def extract_dataframes(self, data: Dict) -> Dict[str, pd.DataFrame]:
        """
        从JSON数据中提取DataFrame
        
        Args:
            data: 包含数据的字典
            
        Returns:
            Dict[str, pd.DataFrame]: 包含各个数据表的字典
        """
        dataframes = {}
        
        # 提取运输方式数据
        if 'transport_modes' in data:
            dataframes['transport_modes'] = pd.DataFrame(data['transport_modes'])
            self._transport_modes = dataframes['transport_modes']
        
        # 提取货物类型数据
        if 'cargo_types' in data:
            dataframes['cargo_types'] = pd.DataFrame(data['cargo_types'])
            self._cargo_types = dataframes['cargo_types']
        
        # 提取地点数据
        if 'locations' in data:
            dataframes['locations'] = pd.DataFrame(data['locations'])
            self._locations = dataframes['locations']
        
        # 提取距离矩阵数据
        if 'distance_matrix' in data:
            dataframes['distance_matrix'] = pd.DataFrame(data['distance_matrix'])
            self._distance_matrix = dataframes['distance_matrix']
        else:
            # 尝试从CSV文件加载距离矩阵数据
            try:
                csv_file = self.sample_dir / "distance_matrix.csv"
                if csv_file.exists():
                    distance_df = pd.read_csv(csv_file)
                    dataframes['distance_matrix'] = distance_df
                    self._distance_matrix = distance_df
                    logger.info(f"从CSV文件加载距离矩阵数据: {len(distance_df)} 条记录")
            except Exception as e:
                logger.warning(f"从CSV文件加载距离矩阵数据失败: {e}")
        
        # 提取报价数据
        if 'sample_quotes' in data:
            # 提取基本报价信息
            quotes = []
            details = []
            
            for quote in data['sample_quotes']:
                quote_details = quote.pop('details', [])
                quotes.append(quote)
                
                # 处理报价明细
                for detail in quote_details:
                    details.append(detail)
            
            dataframes['quotes'] = pd.DataFrame(quotes)
            dataframes['quote_details'] = pd.DataFrame(details)
            self._quotes = dataframes['quotes']
        
        logger.info(f"成功提取数据表: {list(dataframes.keys())}")
        return dataframes
    
    def clean_data(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        清洗数据
        
        Args:
            dataframes: 包含各个数据表的字典
            
        Returns:
            Dict[str, pd.DataFrame]: 清洗后的数据表字典
        """
        cleaned = {}
        
        for name, df in dataframes.items():
            # 复制DataFrame以避免修改原始数据
            cleaned_df = df.copy()
            
            # 处理缺失值
            if name == 'quotes':
                # 对于报价数据，使用适当的默认值填充缺失值
                cleaned_df['special_requirements'].fillna('None', inplace=True)
                cleaned_df['is_llm_assisted'].fillna(False, inplace=True)
            
            # 处理日期列
            date_columns = [col for col in cleaned_df.columns if 'date' in col.lower() or 'time' in col.lower()]
            for col in date_columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
            
            # 处理异常值
            if name == 'quotes':
                # 移除异常的重量和体积值（例如，负值或极端值）
                cleaned_df = cleaned_df[cleaned_df['weight'] > 0]
                cleaned_df = cleaned_df[cleaned_df['volume'] > 0]
                
                # 计算单位体积重量，用于检测异常值
                cleaned_df['volume_weight_ratio'] = cleaned_df['weight'] / cleaned_df['volume']
                
                # 移除异常的比率值（例如，极端值）
                q1 = cleaned_df['volume_weight_ratio'].quantile(0.25)
                q3 = cleaned_df['volume_weight_ratio'].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                cleaned_df = cleaned_df[
                    (cleaned_df['volume_weight_ratio'] >= lower_bound) & 
                    (cleaned_df['volume_weight_ratio'] <= upper_bound)
                ]
                
                # 移除临时列
                cleaned_df.drop('volume_weight_ratio', axis=1, inplace=True)
            
            cleaned[name] = cleaned_df
            logger.info(f"清洗数据表 {name}: 从 {len(df)} 行到 {len(cleaned_df)} 行")
        
        return cleaned
    
    def preprocess_data(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        预处理数据，包括特征工程和数据转换
        
        Args:
            dataframes: 包含各个数据表的字典
            
        Returns:
            Dict[str, pd.DataFrame]: 预处理后的数据表字典
        """
        preprocessed = {}
        
        # 处理报价数据
        if 'quotes' in dataframes:
            quotes_df = dataframes['quotes'].copy()
            
            # 尝试合并距离信息
            try:
                if 'distance_matrix' in dataframes and not dataframes['distance_matrix'].empty:
                    distance_df = dataframes['distance_matrix'].copy()
                    
                    # 合并距离信息
                    quotes_with_distance = quotes_df.merge(
                        distance_df,
                        how='left',
                        left_on=['origin_location_id', 'destination_location_id', 'transport_mode_id'],
                        right_on=['origin_location_id', 'destination_location_id', 'transport_mode_id']
                    )
                    
                    # 检查是否有缺失的距离值
                    missing_distance = quotes_with_distance['distance'].isna().sum()
                    if missing_distance > 0:
                        logger.warning(f"有 {missing_distance} 条报价记录缺少距离信息，使用默认值填充")
                        # 填充缺失的距离值
                        quotes_with_distance['distance'].fillna(500.0, inplace=True)
                        
                    # 检查是否有缺失的运输时间值
                    missing_time = quotes_with_distance['typical_transit_time'].isna().sum()
                    if missing_time > 0:
                        logger.warning(f"有 {missing_time} 条报价记录缺少运输时间信息，使用默认值填充")
                        # 填充缺失的运输时间值
                        quotes_with_distance['typical_transit_time'].fillna(5.0, inplace=True)
                else:
                    logger.warning("距离矩阵数据不可用，使用默认值")
                    # 如果没有距离矩阵数据，添加默认的距离和运输时间列
                    quotes_with_distance = quotes_df.copy()
                    quotes_with_distance['distance'] = 500.0
                    quotes_with_distance['typical_transit_time'] = 5.0
            except Exception as e:
                logger.error(f"合并距离信息时出错: {e}")
                # 如果合并出错，使用默认值
                quotes_with_distance = quotes_df.copy()
                quotes_with_distance['distance'] = 500.0
                quotes_with_distance['typical_transit_time'] = 5.0
            
            # 计算单位距离价格
            quotes_with_distance['price_per_km'] = quotes_with_distance['total_price'] / quotes_with_distance['distance']
            
            # 计算单位重量价格
            quotes_with_distance['price_per_kg'] = quotes_with_distance['total_price'] / quotes_with_distance['weight']
            
            # 计算单位体积价格
            quotes_with_distance['price_per_cbm'] = quotes_with_distance['total_price'] / quotes_with_distance['volume']
            
            # 计算密度（重量/体积）
            quotes_with_distance['density'] = quotes_with_distance['weight'] / quotes_with_distance['volume']
            
            preprocessed['quotes_enriched'] = quotes_with_distance
            logger.info(f"预处理报价数据: 添加了距离和价格特征")
        
        # 处理其他数据表
        for name, df in dataframes.items():
            if name not in preprocessed:
                preprocessed[name] = df.copy()
        
        return preprocessed
    
    def save_processed_data(self, dataframes: Dict[str, pd.DataFrame], prefix: str = "processed_") -> None:
        """
        保存处理后的数据
        
        Args:
            dataframes: 包含各个数据表的字典
            prefix: 文件名前缀
        """
        for name, df in dataframes.items():
            file_path = self.processed_dir / f"{prefix}{name}.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"保存处理后的数据到: {file_path}")
    
    def process_pipeline(self) -> Dict[str, pd.DataFrame]:
        """
        执行完整的数据处理流程
        
        Returns:
            Dict[str, pd.DataFrame]: 处理后的数据表字典
        """
        # 加载样本数据
        raw_data = self.load_sample_data()
        if not raw_data:
            logger.error("无法加载样本数据，处理流程终止")
            return {}
        
        # 提取DataFrame
        dataframes = self.extract_dataframes(raw_data)
        
        # 清洗数据
        cleaned_dataframes = self.clean_data(dataframes)
        
        # 预处理数据
        preprocessed_dataframes = self.preprocess_data(cleaned_dataframes)
        
        # 保存处理后的数据
        self.save_processed_data(preprocessed_dataframes)
        
        return preprocessed_dataframes
    
    def get_transport_modes(self) -> pd.DataFrame:
        """获取运输方式数据"""
        if self._transport_modes is None:
            data = self.load_sample_data()
            self.extract_dataframes(data)
        return self._transport_modes
    
    def get_cargo_types(self) -> pd.DataFrame:
        """获取货物类型数据"""
        if self._cargo_types is None:
            data = self.load_sample_data()
            self.extract_dataframes(data)
        return self._cargo_types
    
    def get_locations(self) -> pd.DataFrame:
        """获取地点数据"""
        if self._locations is None:
            data = self.load_sample_data()
            self.extract_dataframes(data)
        return self._locations
    
    def get_distance_matrix(self) -> pd.DataFrame:
        """获取距离矩阵数据"""
        if self._distance_matrix is None:
            # 首先尝试从JSON数据加载
            data = self.load_sample_data()
            self.extract_dataframes(data)
            
            # 如果仍然没有距离矩阵数据，尝试直接从CSV文件加载
            if self._distance_matrix is None:
                try:
                    csv_file = self.sample_dir / "distance_matrix.csv"
                    if csv_file.exists():
                        self._distance_matrix = pd.read_csv(csv_file)
                        logger.info(f"直接从CSV文件加载距离矩阵数据: {len(self._distance_matrix)} 条记录")
                except Exception as e:
                    logger.warning(f"直接从CSV文件加载距离矩阵数据失败: {e}")
                    # 创建一个空的DataFrame作为后备
                    self._distance_matrix = pd.DataFrame(columns=[
                        'origin_location_id', 'destination_location_id', 
                        'transport_mode_id', 'distance', 'typical_transit_time'
                    ])
        
        return self._distance_matrix
    
    def get_quotes(self) -> pd.DataFrame:
        """获取报价数据"""
        if self._quotes is None:
            data = self.load_sample_data()
            self.extract_dataframes(data)
        return self._quotes


if __name__ == "__main__":
    """直接运行此模块时执行数据处理流程"""
    processor = DataProcessor()
    processed_data = processor.process_pipeline()
    print(f"处理完成，共处理 {len(processed_data)} 个数据表") 