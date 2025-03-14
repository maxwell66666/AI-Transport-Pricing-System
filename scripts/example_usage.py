#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目初始化工具使用示例

此脚本展示了如何使用项目初始化工具创建不同类型的项目。
"""

import os
import subprocess
import sys
from pathlib import Path


def create_example_projects():
    """创建示例项目"""
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent.resolve()
    
    # 项目初始化脚本路径
    init_script = script_dir / "init_project.py"
    
    # 示例项目目录
    examples_dir = script_dir.parent / "examples"
    os.makedirs(examples_dir, exist_ok=True)
    
    # 创建默认项目
    print("\n=== 创建默认项目 ===")
    subprocess.run([
        sys.executable, 
        str(init_script), 
        "example_default_project",
        "--path", 
        str(examples_dir)
    ])
    
    # 创建数据分析项目
    print("\n=== 创建数据分析项目 ===")
    subprocess.run([
        sys.executable, 
        str(init_script), 
        "example_data_analysis_project",
        "--path", 
        str(examples_dir),
        "--type",
        "data_analysis"
    ])
    
    # 创建Web应用项目
    print("\n=== 创建Web应用项目 ===")
    subprocess.run([
        sys.executable, 
        str(init_script), 
        "example_web_app_project",
        "--path", 
        str(examples_dir),
        "--type",
        "web_app"
    ])
    
    print("\n所有示例项目已创建完成！")
    print(f"项目位置: {examples_dir}")


if __name__ == "__main__":
    create_example_projects() 