#!/usr/bin/env python
"""
API启动脚本

本脚本用于启动FastAPI服务，支持开发和生产环境。
"""

import os
import sys
import argparse
import logging
import uvicorn
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.config import settings, configure_logging

# 配置日志
configure_logging()
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动AI运输报价系统API服务")
    
    parser.add_argument(
        "--host", 
        type=str, 
        default=settings.API_HOST,
        help=f"服务器主机地址 (默认: {settings.API_HOST})"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=settings.API_PORT,
        help=f"服务器端口 (默认: {settings.API_PORT})"
    )
    
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="启用自动重载 (开发模式)"
    )
    
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1,
        help="工作进程数量 (默认: 1)"
    )
    
    parser.add_argument(
        "--env", 
        type=str, 
        choices=["development", "production", "testing"],
        default=settings.ENVIRONMENT,
        help=f"运行环境 (默认: {settings.ENVIRONMENT})"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置环境变量
    os.environ["ENVIRONMENT"] = args.env
    
    # 打印启动信息
    logger.info(f"启动AI运输报价系统API服务")
    logger.info(f"环境: {args.env}")
    logger.info(f"主机: {args.host}")
    logger.info(f"端口: {args.port}")
    logger.info(f"自动重载: {'启用' if args.reload else '禁用'}")
    logger.info(f"工作进程: {args.workers}")
    
    # 启动服务器
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,  # reload模式下只能使用1个工作进程
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main() 