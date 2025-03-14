#!/usr/bin/env python
"""
运行数据处理和分析流程的脚本

此脚本执行以下步骤：
1. 数据处理：清洗和预处理历史运输报价数据
2. 探索性数据分析：分析影响报价的关键因素
3. 模型训练：训练和评估报价预测模型
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.data_processor import DataProcessor
from src.ml.exploratory_analysis import ExploratoryAnalysis
from src.ml.price_predictor import PricePredictor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行数据处理和分析流程')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='数据目录路径 (默认: data)')
    parser.add_argument('--output-dir', type=str, default='data/analysis',
                        help='分析结果输出目录 (默认: data/analysis)')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='模型保存目录 (默认: models)')
    parser.add_argument('--skip-processing', action='store_true',
                        help='跳过数据处理步骤')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='跳过探索性数据分析步骤')
    parser.add_argument('--skip-modeling', action='store_true',
                        help='跳过模型训练步骤')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建必要的目录
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 步骤1：数据处理
    if not args.skip_processing:
        logger.info("开始数据处理...")
        processor = DataProcessor(args.data_dir)
        processed_data = processor.process_pipeline()
        logger.info(f"数据处理完成，共处理 {len(processed_data)} 个数据表")
    else:
        logger.info("跳过数据处理步骤")
    
    # 步骤2：探索性数据分析
    if not args.skip_analysis:
        logger.info("开始探索性数据分析...")
        analyzer = ExploratoryAnalysis(args.data_dir, args.output_dir)
        analyzer.run_analysis()
        logger.info(f"探索性数据分析完成，结果保存在 {args.output_dir} 目录")
    else:
        logger.info("跳过探索性数据分析步骤")
    
    # 步骤3：模型训练
    if not args.skip_modeling:
        logger.info("开始模型训练...")
        predictor = PricePredictor(args.data_dir, args.model_dir)
        results = predictor.run_training_pipeline()
        logger.info(f"模型训练完成，结果保存在 {args.model_dir} 目录")
    else:
        logger.info("跳过模型训练步骤")
    
    logger.info("所有步骤完成")


if __name__ == "__main__":
    main() 