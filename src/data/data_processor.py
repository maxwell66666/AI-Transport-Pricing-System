"""
数据处理模块

本模块负责加载、处理和转换数据，为模型训练和预测提供数据支持。
支持从CSV、JSON和数据库中加载数据，并进行必要的预处理和特征工程。
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

from sqlalchemy.orm import Session

from src.config import settings
from src.db.models import (
    Location, TransportMode, CargoType, DistanceMatrix, 
    Quote, QuoteDetail
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    数据处理类
    
    负责加载、处理和转换数据，为模型训练和预测提供数据支持。
    """
    
    def __init__(self, db: Optional[Session] = None):
        """
        初始化数据处理器
        
        Args:
            db: 数据库会话，如果提供则从数据库加载数据，否则从文件加载
        """
        self.db = db
        self.data_dir = Path(settings.DATA_DIR)
        self.sample_data_dir = Path(settings.SAMPLE_DATA_DIR)
        self.processed_data_dir = Path(settings.PROCESSED_DATA_DIR)
        self.raw_data_dir = Path(settings.RAW_DATA_DIR)
        
        # 确保目录存在
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据缓存
        self._locations_cache = None
        self._transport_modes_cache = None
        self._cargo_types_cache = None
        self._distance_matrix_cache = None
    
    def load_locations(self, force_reload: bool = False) -> pd.DataFrame:
        """
        加载位置数据
        
        Args:
            force_reload: 是否强制重新加载数据
            
        Returns:
            位置数据DataFrame
        """
        if self._locations_cache is not None and not force_reload:
            return self._locations_cache
        
        try:
            if self.db:
                # 从数据库加载
                locations = self.db.query(Location).all()
                data = [loc.__dict__ for loc in locations]
                for item in data:
                    item.pop('_sa_instance_state', None)
                df = pd.DataFrame(data)
            else:
                # 从文件加载
                file_path = self.sample_data_dir / "locations.csv"
                df = pd.read_csv(file_path)
            
            self._locations_cache = df
            logger.info(f"成功加载位置数据: {len(df)}条记录")
            return df
        
        except Exception as e:
            logger.error(f"加载位置数据失败: {str(e)}")
            # 返回空DataFrame
            return pd.DataFrame(columns=['id', 'name', 'country', 'city', 'is_port', 'is_airport'])
    
    def load_transport_modes(self, force_reload: bool = False) -> pd.DataFrame:
        """
        加载运输方式数据
        
        Args:
            force_reload: 是否强制重新加载数据
            
        Returns:
            运输方式数据DataFrame
        """
        if self._transport_modes_cache is not None and not force_reload:
            return self._transport_modes_cache
        
        try:
            if self.db:
                # 从数据库加载
                modes = self.db.query(TransportMode).all()
                data = [mode.__dict__ for mode in modes]
                for item in data:
                    item.pop('_sa_instance_state', None)
                df = pd.DataFrame(data)
            else:
                # 从文件加载
                file_path = self.sample_data_dir / "transport_modes.csv"
                df = pd.read_csv(file_path)
            
            self._transport_modes_cache = df
            logger.info(f"成功加载运输方式数据: {len(df)}条记录")
            return df
        
        except Exception as e:
            logger.error(f"加载运输方式数据失败: {str(e)}")
            # 返回空DataFrame
            return pd.DataFrame(columns=['id', 'name', 'description', 'is_active'])
    
    def load_cargo_types(self, force_reload: bool = False) -> pd.DataFrame:
        """
        加载货物类型数据
        
        Args:
            force_reload: 是否强制重新加载数据
            
        Returns:
            货物类型数据DataFrame
        """
        if self._cargo_types_cache is not None and not force_reload:
            return self._cargo_types_cache
        
        try:
            if self.db:
                # 从数据库加载
                types = self.db.query(CargoType).all()
                data = [type_.__dict__ for type_ in types]
                for item in data:
                    item.pop('_sa_instance_state', None)
                df = pd.DataFrame(data)
            else:
                # 从文件加载
                file_path = self.sample_data_dir / "cargo_types.csv"
                df = pd.read_csv(file_path)
            
            self._cargo_types_cache = df
            logger.info(f"成功加载货物类型数据: {len(df)}条记录")
            return df
        
        except Exception as e:
            logger.error(f"加载货物类型数据失败: {str(e)}")
            # 返回空DataFrame
            return pd.DataFrame(columns=['id', 'name', 'description', 'is_dangerous', 'requires_temperature_control'])
    
    def load_distance_matrix(self, force_reload: bool = False) -> pd.DataFrame:
        """
        加载距离矩阵数据
        
        Args:
            force_reload: 是否强制重新加载数据
            
        Returns:
            距离矩阵数据DataFrame
        """
        if self._distance_matrix_cache is not None and not force_reload:
            return self._distance_matrix_cache
        
        try:
            if self.db:
                # 从数据库加载
                distances = self.db.query(DistanceMatrix).all()
                data = [dist.__dict__ for dist in distances]
                for item in data:
                    item.pop('_sa_instance_state', None)
                df = pd.DataFrame(data)
            else:
                # 从文件加载
                file_path = self.sample_data_dir / "distance_matrix.csv"
                df = pd.read_csv(file_path)
            
            self._distance_matrix_cache = df
            logger.info(f"成功加载距离矩阵数据: {len(df)}条记录")
            return df
        
        except Exception as e:
            logger.error(f"加载距离矩阵数据失败: {str(e)}")
            # 返回空DataFrame
            return pd.DataFrame(columns=['origin_location_id', 'destination_location_id', 'transport_mode_id', 'distance', 'typical_transit_time'])
    
    def load_quotes(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        加载报价数据
        
        Args:
            limit: 限制加载的记录数量
            
        Returns:
            报价数据DataFrame
        """
        try:
            if self.db:
                # 从数据库加载
                query = self.db.query(Quote)
                if limit:
                    query = query.limit(limit)
                quotes = query.all()
                
                data = []
                for quote in quotes:
                    quote_dict = quote.__dict__.copy()
                    quote_dict.pop('_sa_instance_state', None)
                    
                    # 加载报价明细
                    details = self.db.query(QuoteDetail).filter(QuoteDetail.quote_id == quote.id).all()
                    quote_dict['details'] = [
                        {k: v for k, v in detail.__dict__.items() if k != '_sa_instance_state'}
                        for detail in details
                    ]
                    
                    data.append(quote_dict)
                
                df = pd.DataFrame(data)
            else:
                # 从文件加载
                file_path = self.sample_data_dir / "quotes.csv"
                df = pd.read_csv(file_path)
                
                # 加载报价明细
                details_path = self.sample_data_dir / "quote_details.csv"
                if details_path.exists():
                    details_df = pd.read_csv(details_path)
                    # 将明细数据转换为嵌套结构
                    details_grouped = details_df.groupby('quote_id')
                    df['details'] = df['id'].apply(
                        lambda qid: details_grouped.get_group(qid).to_dict('records') 
                        if qid in details_grouped.groups else []
                    )
            
            # 限制记录数量
            if limit and len(df) > limit:
                df = df.head(limit)
            
            logger.info(f"成功加载报价数据: {len(df)}条记录")
            return df
        
        except Exception as e:
            logger.error(f"加载报价数据失败: {str(e)}")
            # 返回空DataFrame
            return pd.DataFrame(columns=['id', 'origin_location_id', 'destination_location_id', 'transport_mode_id', 
                                        'cargo_type_id', 'weight', 'volume', 'distance', 'typical_transit_time', 
                                        'total_price', 'currency', 'quote_date', 'special_requirements', 'is_llm_assisted'])
    
    def prepare_training_data(self) -> pd.DataFrame:
        """
        准备模型训练数据
        
        Returns:
            处理后的训练数据DataFrame
        """
        # 加载基础数据
        quotes_df = self.load_quotes()
        locations_df = self.load_locations()
        transport_modes_df = self.load_transport_modes()
        cargo_types_df = self.load_cargo_types()
        
        if quotes_df.empty:
            logger.error("无法准备训练数据：报价数据为空")
            return pd.DataFrame()
        
        # 确保数据类型正确
        numeric_cols = ['weight', 'volume', 'distance', 'typical_transit_time', 'total_price']
        for col in numeric_cols:
            if col in quotes_df.columns:
                quotes_df[col] = pd.to_numeric(quotes_df[col], errors='coerce')
        
        # 处理日期列
        if 'quote_date' in quotes_df.columns:
            quotes_df['quote_date'] = pd.to_datetime(quotes_df['quote_date'], errors='coerce')
            
            # 提取日期特征
            quotes_df['quote_year'] = quotes_df['quote_date'].dt.year
            quotes_df['quote_month'] = quotes_df['quote_date'].dt.month
            quotes_df['quote_day'] = quotes_df['quote_date'].dt.day
            quotes_df['quote_dayofweek'] = quotes_df['quote_date'].dt.dayofweek
        
        # 处理缺失值
        quotes_df = quotes_df.dropna(subset=['total_price'])  # 删除目标变量缺失的行
        
        # 对于其他缺失值，可以根据具体情况填充
        quotes_df = quotes_df.fillna({
            'weight': quotes_df['weight'].median(),
            'volume': quotes_df['volume'].median(),
            'distance': quotes_df['distance'].median(),
            'typical_transit_time': quotes_df['typical_transit_time'].median()
        })
        
        # 添加特征：是否危险品
        if not cargo_types_df.empty and 'cargo_type_id' in quotes_df.columns:
            cargo_type_map = dict(zip(cargo_types_df['id'], cargo_types_df['is_dangerous']))
            quotes_df['is_dangerous_cargo'] = quotes_df['cargo_type_id'].map(cargo_type_map).fillna(False)
        
        # 添加特征：是否需要温控
        if not cargo_types_df.empty and 'cargo_type_id' in quotes_df.columns:
            temp_control_map = dict(zip(cargo_types_df['id'], cargo_types_df['requires_temperature_control']))
            quotes_df['requires_temp_control'] = quotes_df['cargo_type_id'].map(temp_control_map).fillna(False)
        
        # 添加特征：重量体积比
        if 'weight' in quotes_df.columns and 'volume' in quotes_df.columns:
            quotes_df['weight_volume_ratio'] = quotes_df['weight'] / quotes_df['volume'].replace(0, 0.001)
        
        # 添加特征：单位距离价格
        if 'total_price' in quotes_df.columns and 'distance' in quotes_df.columns:
            quotes_df['price_per_km'] = quotes_df['total_price'] / quotes_df['distance'].replace(0, 0.001)
        
        # 添加特征：单位重量价格
        if 'total_price' in quotes_df.columns and 'weight' in quotes_df.columns:
            quotes_df['price_per_kg'] = quotes_df['total_price'] / quotes_df['weight'].replace(0, 0.001)
        
        # 添加特征：单位体积价格
        if 'total_price' in quotes_df.columns and 'volume' in quotes_df.columns:
            quotes_df['price_per_cbm'] = quotes_df['total_price'] / quotes_df['volume'].replace(0, 0.001)
        
        # 保存处理后的数据
        processed_file = self.processed_data_dir / "training_data.csv"
        quotes_df.to_csv(processed_file, index=False)
        logger.info(f"训练数据已保存至: {processed_file}")
        
        return quotes_df
    
    def get_distance_and_transit_time(
        self, 
        origin_id: int, 
        destination_id: int, 
        transport_mode_id: int
    ) -> Tuple[float, float]:
        """
        获取两地点间的距离和运输时间
        
        Args:
            origin_id: 起始地点ID
            destination_id: 目的地点ID
            transport_mode_id: 运输方式ID
            
        Returns:
            距离和运输时间的元组
        """
        distance_df = self.load_distance_matrix()
        
        # 查找匹配的记录
        mask = (
            (distance_df['origin_location_id'] == origin_id) & 
            (distance_df['destination_location_id'] == destination_id) & 
            (distance_df['transport_mode_id'] == transport_mode_id)
        )
        
        matching_rows = distance_df[mask]
        
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            return row['distance'], row['typical_transit_time']
        
        # 如果没有找到完全匹配的记录，尝试找到相同起始地和目的地的其他运输方式
        mask = (
            (distance_df['origin_location_id'] == origin_id) & 
            (distance_df['destination_location_id'] == destination_id)
        )
        
        similar_rows = distance_df[mask]
        
        if not similar_rows.empty:
            # 使用平均值或第一条记录
            distance = similar_rows['distance'].mean()
            transit_time = similar_rows['typical_transit_time'].mean()
            
            # 根据运输方式调整
            transport_modes = self.load_transport_modes()
            if not transport_modes.empty:
                mode_row = transport_modes[transport_modes['id'] == transport_mode_id]
                if not mode_row.empty:
                    mode_name = mode_row.iloc[0]['name']
                    # 根据运输方式名称调整时间
                    if '空运' in mode_name or '航空' in mode_name:
                        transit_time = min(transit_time, 7)  # 空运通常更快
                    elif '海运' in mode_name:
                        transit_time = max(transit_time, 14)  # 海运通常更慢
            
            return distance, transit_time
        
        # 如果仍然没有找到，返回默认值或估计值
        # 这里可以实现更复杂的估算逻辑，例如基于地理坐标计算
        logger.warning(f"未找到距离数据: 起点={origin_id}, 终点={destination_id}, 运输方式={transport_mode_id}")
        return 1000.0, 14.0  # 默认值
    
    def find_similar_quotes(
        self, 
        origin_id: int, 
        destination_id: int, 
        transport_mode_id: int,
        cargo_type_id: int,
        weight: float,
        volume: float,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        查找相似的历史报价
        
        Args:
            origin_id: 起始地点ID
            destination_id: 目的地点ID
            transport_mode_id: 运输方式ID
            cargo_type_id: 货物类型ID
            weight: 重量
            volume: 体积
            limit: 返回结果数量限制
            
        Returns:
            相似报价列表
        """
        quotes_df = self.load_quotes()
        
        if quotes_df.empty:
            return []
        
        # 计算相似度分数
        quotes_df['similarity_score'] = 0
        
        # 相同起始地和目的地加分
        if 'origin_location_id' in quotes_df.columns and 'destination_location_id' in quotes_df.columns:
            quotes_df.loc[quotes_df['origin_location_id'] == origin_id, 'similarity_score'] += 3
            quotes_df.loc[quotes_df['destination_location_id'] == destination_id, 'similarity_score'] += 3
        
        # 相同运输方式加分
        if 'transport_mode_id' in quotes_df.columns:
            quotes_df.loc[quotes_df['transport_mode_id'] == transport_mode_id, 'similarity_score'] += 2
        
        # 相同货物类型加分
        if 'cargo_type_id' in quotes_df.columns:
            quotes_df.loc[quotes_df['cargo_type_id'] == cargo_type_id, 'similarity_score'] += 2
        
        # 重量和体积相似度
        if 'weight' in quotes_df.columns and weight > 0:
            quotes_df['weight_diff'] = abs(quotes_df['weight'] - weight) / weight
            quotes_df.loc[quotes_df['weight_diff'] <= 0.1, 'similarity_score'] += 2  # 差异小于10%
            quotes_df.loc[(quotes_df['weight_diff'] > 0.1) & (quotes_df['weight_diff'] <= 0.3), 'similarity_score'] += 1  # 差异10%-30%
        
        if 'volume' in quotes_df.columns and volume > 0:
            quotes_df['volume_diff'] = abs(quotes_df['volume'] - volume) / volume
            quotes_df.loc[quotes_df['volume_diff'] <= 0.1, 'similarity_score'] += 2  # 差异小于10%
            quotes_df.loc[(quotes_df['volume_diff'] > 0.1) & (quotes_df['volume_diff'] <= 0.3), 'similarity_score'] += 1  # 差异10%-30%
        
        # 按相似度分数排序并限制结果数量
        similar_quotes = quotes_df.sort_values('similarity_score', ascending=False).head(limit)
        
        # 转换为字典列表
        result = []
        for _, row in similar_quotes.iterrows():
            quote_dict = row.to_dict()
            
            # 移除不需要的列
            for col in ['similarity_score', 'weight_diff', 'volume_diff']:
                if col in quote_dict:
                    quote_dict.pop(col)
            
            # 确保日期格式正确
            if 'quote_date' in quote_dict and pd.notna(quote_dict['quote_date']):
                if isinstance(quote_dict['quote_date'], pd.Timestamp):
                    quote_dict['quote_date'] = quote_dict['quote_date'].strftime('%Y-%m-%d')
            
            result.append(quote_dict)
        
        return result
    
    def export_data_to_json(self, file_path: Optional[str] = None) -> str:
        """
        将数据导出为JSON格式
        
        Args:
            file_path: 导出文件路径，如果不提供则使用默认路径
            
        Returns:
            导出文件的路径
        """
        if file_path is None:
            file_path = self.processed_data_dir / f"export_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            file_path = Path(file_path)
        
        # 准备导出数据
        export_data = {
            "locations": self.load_locations().to_dict('records'),
            "transport_modes": self.load_transport_modes().to_dict('records'),
            "cargo_types": self.load_cargo_types().to_dict('records'),
            "distance_matrix": self.load_distance_matrix().to_dict('records'),
            "quotes": self.load_quotes().to_dict('records'),
            "export_date": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        # 导出到文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已导出至: {file_path}")
        return str(file_path)
    
    def import_data_from_json(self, file_path: str) -> bool:
        """
        从JSON文件导入数据
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            是否成功导入
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"文件不存在: {file_path}")
                return False
            
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据结构
            required_keys = ['locations', 'transport_modes', 'cargo_types', 'distance_matrix', 'quotes']
            for key in required_keys:
                if key not in data:
                    logger.error(f"导入数据缺少必要的键: {key}")
                    return False
            
            # 将数据保存为CSV文件
            for key in required_keys:
                if data[key]:
                    df = pd.DataFrame(data[key])
                    csv_path = self.raw_data_dir / f"{key}.csv"
                    df.to_csv(csv_path, index=False)
                    logger.info(f"已导入 {len(df)} 条 {key} 数据至 {csv_path}")
            
            # 清除缓存
            self._locations_cache = None
            self._transport_modes_cache = None
            self._cargo_types_cache = None
            self._distance_matrix_cache = None
            
            return True
        
        except Exception as e:
            logger.error(f"导入数据失败: {str(e)}")
            return False 