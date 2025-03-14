#!/usr/bin/env python
"""
Web应用启动脚本

此脚本用于启动AI运输报价系统的Web界面。
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 将项目根目录添加到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动AI运输报价系统Web界面")
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="服务器主机地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8050, 
        help="服务器端口 (默认: 8050)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="启用调试模式"
    )
    parser.add_argument(
        "--api-url", 
        type=str, 
        default="http://localhost:8000/api/v1", 
        help="API服务URL (默认: http://localhost:8000/api/v1)"
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置环境变量
    os.environ["API_BASE_URL"] = args.api_url
    
    # 导入Web应用
    from src.web.app import app
    
    # 启动Web应用
    logger.info(f"正在启动Web应用，地址: {args.host}:{args.port}")
    logger.info(f"API服务URL: {args.api_url}")
    
    app.run_server(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main() 