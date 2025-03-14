"""
探索性数据分析模块

此模块包含用于分析运输报价数据的功能，识别影响报价的关键因素。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple
import os

from src.ml.data_processor import DataProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class ExploratoryAnalysis:
    """探索性数据分析类，用于分析运输报价数据"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "data/analysis"):
        """
        初始化数据分析器
        
        Args:
            data_dir: 数据目录路径
            output_dir: 分析结果输出目录
        """
        self.data_processor = DataProcessor(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
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
    
    def generate_summary_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        生成数据摘要统计
        
        Returns:
            Dict[str, pd.DataFrame]: 包含各个数据表摘要统计的字典
        """
        summary = {}
        
        for name, df in self.processed_data.items():
            # 对于数值列，计算统计摘要
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                summary[f"{name}_numeric"] = df[numeric_cols].describe()
            
            # 对于分类列，计算频率统计
            categorical_cols = df.select_dtypes(include=['object']).columns
            if not categorical_cols.empty:
                cat_summary = {}
                for col in categorical_cols:
                    cat_summary[col] = df[col].value_counts().reset_index()
                    cat_summary[col].columns = ['value', 'count']
                summary[f"{name}_categorical"] = cat_summary
        
        # 保存摘要统计
        for name, stats in summary.items():
            if isinstance(stats, pd.DataFrame):
                stats.to_csv(self.output_dir / f"{name}_summary.csv")
            elif isinstance(stats, dict):
                for col, df in stats.items():
                    df.to_csv(self.output_dir / f"{name}_{col}_summary.csv", index=False)
        
        logger.info(f"生成了 {len(summary)} 个摘要统计表")
        return summary
    
    def analyze_price_factors(self) -> pd.DataFrame:
        """
        分析影响价格的因素
        
        Returns:
            pd.DataFrame: 包含价格因素分析结果的DataFrame
        """
        if 'quotes_enriched' not in self.processed_data:
            logger.error("未找到富集的报价数据，无法分析价格因素")
            return pd.DataFrame()
        
        quotes_df = self.processed_data['quotes_enriched']
        
        # 计算各个因素与总价的相关性
        numeric_cols = quotes_df.select_dtypes(include=[np.number]).columns
        correlation = quotes_df[numeric_cols].corr()['total_price'].sort_values(ascending=False)
        
        # 保存相关性结果
        correlation_df = pd.DataFrame({
            'factor': correlation.index,
            'correlation': correlation.values
        })
        correlation_df.to_csv(self.output_dir / "price_factors_correlation.csv", index=False)
        
        # 绘制相关性热图
        plt.figure(figsize=(12, 10))
        corr_matrix = quotes_df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('特征相关性热图')
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_heatmap.png", dpi=300)
        plt.close()
        
        logger.info(f"分析了 {len(correlation)} 个价格因素的相关性")
        return correlation_df
    
    def analyze_transport_modes(self) -> None:
        """分析不同运输方式的价格和时效"""
        if 'quotes_enriched' not in self.processed_data or 'transport_modes' not in self.processed_data:
            logger.error("未找到必要的数据，无法分析运输方式")
            return
        
        quotes_df = self.processed_data['quotes_enriched']
        modes_df = self.processed_data['transport_modes']
        
        # 合并运输方式名称
        df = quotes_df.merge(modes_df, left_on='transport_mode_id', right_on='id', suffixes=('', '_mode'))
        
        # 按运输方式分组，计算价格统计
        mode_stats = df.groupby('name').agg({
            'total_price': ['mean', 'median', 'std', 'min', 'max', 'count'],
            'price_per_km': ['mean', 'median', 'std'],
            'price_per_kg': ['mean', 'median', 'std'],
            'price_per_cbm': ['mean', 'median', 'std'],
            'typical_transit_time': ['mean', 'median', 'min', 'max']
        })
        
        # 保存统计结果
        mode_stats.to_csv(self.output_dir / "transport_mode_stats.csv")
        
        # 绘制运输方式价格箱线图
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='name', y='total_price', data=df)
        plt.title('不同运输方式的价格分布')
        plt.xlabel('运输方式')
        plt.ylabel('总价格')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "transport_mode_price_boxplot.png", dpi=300)
        plt.close()
        
        # 绘制运输方式单位价格条形图
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.barplot(x='name', y='price_per_km', data=df)
        plt.title('不同运输方式的单位距离价格')
        plt.xlabel('运输方式')
        plt.ylabel('单位距离价格 (价格/公里)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        sns.barplot(x='name', y='price_per_kg', data=df)
        plt.title('不同运输方式的单位重量价格')
        plt.xlabel('运输方式')
        plt.ylabel('单位重量价格 (价格/公斤)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        sns.barplot(x='name', y='price_per_cbm', data=df)
        plt.title('不同运输方式的单位体积价格')
        plt.xlabel('运输方式')
        plt.ylabel('单位体积价格 (价格/立方米)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        sns.barplot(x='name', y='typical_transit_time', data=df)
        plt.title('不同运输方式的典型运输时间')
        plt.xlabel('运输方式')
        plt.ylabel('运输时间 (天)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "transport_mode_unit_prices.png", dpi=300)
        plt.close()
        
        logger.info(f"分析了 {len(mode_stats)} 种运输方式的价格和时效")
    
    def analyze_routes(self) -> None:
        """分析不同路线的价格和时效"""
        if 'quotes_enriched' not in self.processed_data or 'locations' not in self.processed_data:
            logger.error("未找到必要的数据，无法分析路线")
            return
        
        quotes_df = self.processed_data['quotes_enriched']
        locations_df = self.processed_data['locations']
        
        # 合并起点和终点信息
        df = quotes_df.merge(
            locations_df, 
            left_on='origin_location_id', 
            right_on='id', 
            suffixes=('', '_origin')
        )
        df = df.merge(
            locations_df, 
            left_on='destination_location_id', 
            right_on='id', 
            suffixes=('', '_dest')
        )
        
        # 创建路线标识
        df['route'] = df['name'] + ' → ' + df['name_dest']
        
        # 按路线分组，计算价格统计
        route_stats = df.groupby('route').agg({
            'total_price': ['mean', 'median', 'std', 'min', 'max', 'count'],
            'price_per_km': ['mean', 'median', 'std'],
            'distance': ['mean'],
            'typical_transit_time': ['mean', 'median', 'min', 'max']
        })
        
        # 保存统计结果
        route_stats.to_csv(self.output_dir / "route_stats.csv")
        
        # 获取前10条最常用路线
        top_routes = df['route'].value_counts().head(10).index
        top_routes_df = df[df['route'].isin(top_routes)]
        
        # 绘制热门路线价格箱线图
        plt.figure(figsize=(14, 7))
        sns.boxplot(x='route', y='total_price', data=top_routes_df)
        plt.title('热门路线的价格分布')
        plt.xlabel('路线')
        plt.ylabel('总价格')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "top_routes_price_boxplot.png", dpi=300)
        plt.close()
        
        # 绘制热门路线单位价格和时效对比图
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.barplot(x='route', y='price_per_km', data=top_routes_df)
        plt.title('热门路线的单位距离价格')
        plt.xlabel('路线')
        plt.ylabel('单位距离价格 (价格/公里)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        sns.barplot(x='route', y='distance', data=top_routes_df)
        plt.title('热门路线的距离')
        plt.xlabel('路线')
        plt.ylabel('距离 (公里)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        sns.barplot(x='route', y='total_price', data=top_routes_df)
        plt.title('热门路线的平均价格')
        plt.xlabel('路线')
        plt.ylabel('平均价格')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        sns.barplot(x='route', y='typical_transit_time', data=top_routes_df)
        plt.title('热门路线的典型运输时间')
        plt.xlabel('路线')
        plt.ylabel('运输时间 (天)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "top_routes_comparison.png", dpi=300)
        plt.close()
        
        logger.info(f"分析了 {len(route_stats)} 条路线的价格和时效")
    
    def analyze_cargo_types(self) -> None:
        """分析不同货物类型的价格"""
        if 'quotes_enriched' not in self.processed_data or 'cargo_types' not in self.processed_data:
            logger.error("未找到必要的数据，无法分析货物类型")
            return
        
        quotes_df = self.processed_data['quotes_enriched']
        cargo_df = self.processed_data['cargo_types']
        
        # 合并货物类型信息
        df = quotes_df.merge(cargo_df, left_on='cargo_type_id', right_on='id', suffixes=('', '_cargo'))
        
        # 按货物类型分组，计算价格统计
        cargo_stats = df.groupby('name').agg({
            'total_price': ['mean', 'median', 'std', 'min', 'max', 'count'],
            'price_per_kg': ['mean', 'median', 'std'],
            'price_per_cbm': ['mean', 'median', 'std'],
            'weight': ['mean', 'median', 'min', 'max'],
            'volume': ['mean', 'median', 'min', 'max'],
            'density': ['mean', 'median']
        })
        
        # 保存统计结果
        cargo_stats.to_csv(self.output_dir / "cargo_type_stats.csv")
        
        # 绘制货物类型价格箱线图
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='name', y='total_price', data=df)
        plt.title('不同货物类型的价格分布')
        plt.xlabel('货物类型')
        plt.ylabel('总价格')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "cargo_type_price_boxplot.png", dpi=300)
        plt.close()
        
        # 绘制货物类型单位价格对比图
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.barplot(x='name', y='price_per_kg', data=df)
        plt.title('不同货物类型的单位重量价格')
        plt.xlabel('货物类型')
        plt.ylabel('单位重量价格 (价格/公斤)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        sns.barplot(x='name', y='price_per_cbm', data=df)
        plt.title('不同货物类型的单位体积价格')
        plt.xlabel('货物类型')
        plt.ylabel('单位体积价格 (价格/立方米)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        sns.barplot(x='name', y='weight', data=df)
        plt.title('不同货物类型的平均重量')
        plt.xlabel('货物类型')
        plt.ylabel('平均重量 (公斤)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        sns.barplot(x='name', y='volume', data=df)
        plt.title('不同货物类型的平均体积')
        plt.xlabel('货物类型')
        plt.ylabel('平均体积 (立方米)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "cargo_type_comparison.png", dpi=300)
        plt.close()
        
        # 分析危险品和温控货物的价格差异
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(x='is_dangerous', y='total_price', data=df)
        plt.title('危险品与非危险品的价格对比')
        plt.xlabel('是否危险品')
        plt.ylabel('总价格')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x='requires_temperature_control', y='total_price', data=df)
        plt.title('温控与非温控货物的价格对比')
        plt.xlabel('是否需要温控')
        plt.ylabel('总价格')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "special_cargo_price_comparison.png", dpi=300)
        plt.close()
        
        logger.info(f"分析了 {len(cargo_stats)} 种货物类型的价格")
    
    def analyze_weight_volume_relationship(self) -> None:
        """分析重量、体积与价格的关系"""
        if 'quotes_enriched' not in self.processed_data:
            logger.error("未找到富集的报价数据，无法分析重量和体积关系")
            return
        
        quotes_df = self.processed_data['quotes_enriched']
        
        # 绘制重量与价格的散点图
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='weight', y='total_price', hue='transport_mode_id', data=quotes_df)
        plt.title('重量与价格的关系')
        plt.xlabel('重量 (公斤)')
        plt.ylabel('总价格')
        plt.tight_layout()
        plt.savefig(self.output_dir / "weight_price_scatter.png", dpi=300)
        plt.close()
        
        # 绘制体积与价格的散点图
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='volume', y='total_price', hue='transport_mode_id', data=quotes_df)
        plt.title('体积与价格的关系')
        plt.xlabel('体积 (立方米)')
        plt.ylabel('总价格')
        plt.tight_layout()
        plt.savefig(self.output_dir / "volume_price_scatter.png", dpi=300)
        plt.close()
        
        # 绘制密度与单位价格的关系
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='density', y='price_per_kg', hue='transport_mode_id', data=quotes_df)
        plt.title('密度与单位重量价格的关系')
        plt.xlabel('密度 (公斤/立方米)')
        plt.ylabel('单位重量价格 (价格/公斤)')
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x='density', y='price_per_cbm', hue='transport_mode_id', data=quotes_df)
        plt.title('密度与单位体积价格的关系')
        plt.xlabel('密度 (公斤/立方米)')
        plt.ylabel('单位体积价格 (价格/立方米)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "density_unit_price_scatter.png", dpi=300)
        plt.close()
        
        # 计算重量、体积与价格的相关性
        corr_data = quotes_df[['weight', 'volume', 'density', 'total_price', 'price_per_kg', 'price_per_cbm']]
        correlation = corr_data.corr()
        
        # 绘制相关性热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('重量、体积与价格的相关性')
        plt.tight_layout()
        plt.savefig(self.output_dir / "weight_volume_price_correlation.png", dpi=300)
        plt.close()
        
        logger.info("分析了重量、体积与价格的关系")
    
    def analyze_distance_relationship(self) -> None:
        """分析距离与价格的关系"""
        if 'quotes_enriched' not in self.processed_data:
            logger.error("未找到富集的报价数据，无法分析距离关系")
            return
        
        quotes_df = self.processed_data['quotes_enriched']
        
        # 绘制距离与价格的散点图
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='distance', y='total_price', hue='transport_mode_id', data=quotes_df)
        plt.title('距离与价格的关系')
        plt.xlabel('距离 (公里)')
        plt.ylabel('总价格')
        plt.tight_layout()
        plt.savefig(self.output_dir / "distance_price_scatter.png", dpi=300)
        plt.close()
        
        # 绘制距离与单位价格的关系
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='distance', y='price_per_km', hue='transport_mode_id', data=quotes_df)
        plt.title('距离与单位距离价格的关系')
        plt.xlabel('距离 (公里)')
        plt.ylabel('单位距离价格 (价格/公里)')
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x='distance', y='typical_transit_time', hue='transport_mode_id', data=quotes_df)
        plt.title('距离与运输时间的关系')
        plt.xlabel('距离 (公里)')
        plt.ylabel('运输时间 (天)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "distance_unit_price_scatter.png", dpi=300)
        plt.close()
        
        # 按距离分组，计算平均价格
        quotes_df['distance_bin'] = pd.cut(quotes_df['distance'], bins=10)
        distance_price = quotes_df.groupby(['distance_bin', 'transport_mode_id']).agg({
            'total_price': 'mean',
            'price_per_km': 'mean',
            'typical_transit_time': 'mean'
        }).reset_index()
        
        # 绘制距离分组的平均价格
        plt.figure(figsize=(12, 8))
        sns.lineplot(x='distance_bin', y='total_price', hue='transport_mode_id', data=distance_price, marker='o')
        plt.title('不同距离范围的平均价格')
        plt.xlabel('距离范围 (公里)')
        plt.ylabel('平均价格')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "distance_bin_avg_price.png", dpi=300)
        plt.close()
        
        logger.info("分析了距离与价格的关系")
    
    def generate_key_findings(self) -> Dict:
        """
        生成关键发现
        
        Returns:
            Dict: 包含关键发现的字典
        """
        findings = {}
        
        # 分析价格因素
        price_factors = self.analyze_price_factors()
        if not price_factors.empty:
            # 获取前5个最重要的价格因素
            top_factors = price_factors.head(5)
            findings['top_price_factors'] = top_factors.to_dict('records')
        
        # 分析运输方式
        self.analyze_transport_modes()
        
        # 分析路线
        self.analyze_routes()
        
        # 分析货物类型
        self.analyze_cargo_types()
        
        # 分析重量和体积
        self.analyze_weight_volume_relationship()
        
        # 分析距离
        self.analyze_distance_relationship()
        
        # 保存关键发现
        with open(self.output_dir / "key_findings.txt", 'w', encoding='utf-8') as f:
            f.write("# 运输报价数据分析 - 关键发现\n\n")
            
            f.write("## 价格影响因素\n")
            if 'top_price_factors' in findings:
                f.write("最重要的价格影响因素（与总价的相关性）：\n")
                for factor in findings['top_price_factors']:
                    f.write(f"- {factor['factor']}: {factor['correlation']:.4f}\n")
            
            f.write("\n## 运输方式分析\n")
            f.write("不同运输方式的价格和时效特点：\n")
            f.write("- 详见 transport_mode_stats.csv 和相关图表\n")
            
            f.write("\n## 路线分析\n")
            f.write("热门路线的价格和时效特点：\n")
            f.write("- 详见 route_stats.csv 和相关图表\n")
            
            f.write("\n## 货物类型分析\n")
            f.write("不同货物类型的价格特点：\n")
            f.write("- 详见 cargo_type_stats.csv 和相关图表\n")
            
            f.write("\n## 重量和体积分析\n")
            f.write("重量、体积与价格的关系：\n")
            f.write("- 详见相关散点图和相关性分析\n")
            
            f.write("\n## 距离分析\n")
            f.write("距离与价格的关系：\n")
            f.write("- 详见相关散点图和分组分析\n")
        
        logger.info("生成了关键发现报告")
        return findings
    
    def run_analysis(self) -> None:
        """执行完整的数据分析流程"""
        # 生成摘要统计
        self.generate_summary_statistics()
        
        # 生成关键发现
        self.generate_key_findings()
        
        logger.info("完成了所有数据分析")


if __name__ == "__main__":
    """直接运行此模块时执行数据分析流程"""
    analyzer = ExploratoryAnalysis()
    analyzer.run_analysis()
    print(f"分析完成，结果保存在 {analyzer.output_dir} 目录") 