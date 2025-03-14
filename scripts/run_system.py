#!/usr/bin/env python
"""
系统启动脚本

此脚本用于一键启动AI运输报价系统的所有组件，包括API服务和Web应用。
"""

import os
import sys
import argparse
import logging
import subprocess
import time
import signal
import threading
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

# 进程列表
processes = []


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动AI运输报价系统")
    parser.add_argument(
        "--api-host", 
        type=str, 
        default="0.0.0.0", 
        help="API服务主机地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--api-port", 
        type=int, 
        default=8000, 
        help="API服务端口 (默认: 8000)"
    )
    parser.add_argument(
        "--web-host", 
        type=str, 
        default="0.0.0.0", 
        help="Web应用主机地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--web-port", 
        type=int, 
        default=8050, 
        help="Web应用端口 (默认: 8050)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="启用调试模式"
    )
    parser.add_argument(
        "--env-file", 
        type=str, 
        default=".env", 
        help="环境变量文件路径 (默认: .env)"
    )
    return parser.parse_args()


def start_api_server(args):
    """启动API服务"""
    logger.info("正在启动API服务...")
    
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_api_server.py"),
        "--host", args.api_host,
        "--port", str(args.api_port),
    ]
    
    if args.debug:
        cmd.append("--reload")
    
    if args.env_file:
        cmd.extend(["--env-file", args.env_file])
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    
    processes.append(process)
    
    # 启动线程读取输出
    threading.Thread(target=read_output, args=(process, "API服务"), daemon=True).start()
    
    logger.info(f"API服务已启动，地址: http://{args.api_host}:{args.api_port}")
    logger.info(f"API文档地址: http://{args.api_host}:{args.api_port}/docs")
    
    return process


def start_web_app(args):
    """启动Web应用"""
    logger.info("正在启动Web应用...")
    
    # 设置环境变量
    os.environ["API_BASE_URL"] = f"http://{args.api_host}:{args.api_port}/api/v1"
    
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_web_app.py"),
        "--host", args.web_host,
        "--port", str(args.web_port),
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    
    processes.append(process)
    
    # 启动线程读取输出
    threading.Thread(target=read_output, args=(process, "Web应用"), daemon=True).start()
    
    logger.info(f"Web应用已启动，地址: http://{args.web_host}:{args.web_port}")
    
    return process


def read_output(process, name):
    """读取进程输出"""
    for line in process.stdout:
        logger.info(f"[{name}] {line.strip()}")
    
    for line in process.stderr:
        logger.error(f"[{name}] {line.strip()}")


def signal_handler(sig, frame):
    """信号处理函数"""
    logger.info("正在关闭系统...")
    
    for process in processes:
        process.terminate()
    
    logger.info("系统已关闭")
    sys.exit(0)


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
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动API服务
    api_process = start_api_server(args)
    
    # 等待API服务启动
    time.sleep(2)
    
    # 启动Web应用
    web_process = start_web_app(args)
    
    # 等待进程结束
    try:
        api_process.wait()
        web_process.wait()
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭系统...")
        api_process.terminate()
        web_process.terminate()
        logger.info("系统已关闭")


if __name__ == "__main__":
    main() 