#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行Web仪表盘

此脚本用于启动基于Dash的Web仪表盘，提供AI运输报价系统的可视化界面。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# 导入仪表盘应用
from src.web.dashboard import app

if __name__ == "__main__":
    # 获取端口，默认为8050
    port = int(os.getenv("DASHBOARD_PORT", 8050))
    
    print(f"启动Web仪表盘，访问地址: http://localhost:{port}")
    
    # 启动Dash应用
    app.run_server(
        debug=True,
        host="0.0.0.0",
        port=port
    ) 