#!/usr/bin/env python
"""
API服务启动脚本

此脚本用于启动AI运输报价系统的API服务。
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
    parser = argparse.ArgumentParser(description="启动AI运输报价系统API服务")
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="服务器主机地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="服务器端口 (默认: 8000)"
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
        help="工作进程数 (默认: 1)"
    )
    parser.add_argument(
        "--env-file", 
        type=str, 
        default=".env", 
        help="环境变量文件路径 (默认: .env)"
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 检查环境变量文件
    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = ROOT_DIR / env_path
    
    if not env_path.exists():
        logger.warning(f"环境变量文件 {env_path} 不存在，将使用默认配置")
    else:
        logger.info(f"使用环境变量文件: {env_path}")
        # 设置环境变量，让应用知道使用哪个环境变量文件
        os.environ["ENV_FILE"] = str(env_path)
    
    # 设置其他环境变量
    if args.reload:
        os.environ["DEBUG"] = "true"
    
    # 导入uvicorn
    import uvicorn
    
    # 启动服务器
    logger.info(f"正在启动API服务，地址: {args.host}:{args.port}")
    uvicorn.run(
        "src.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main() 