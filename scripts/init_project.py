#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目初始化脚本

此脚本用于按照Cursor项目系统规则快速设置项目结构。
它将创建标准目录结构、初始化配置文件，并设置基本的开发环境。
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any


class ProjectInitializer:
    """
    项目初始化器类
    
    用于创建符合Cursor项目系统规则的项目结构。
    """
    
    def __init__(self, project_name: str, project_path: str, project_type: str = "default"):
        """
        初始化项目初始化器
        
        参数:
            project_name: 项目名称
            project_path: 项目路径
            project_type: 项目类型 (default, data_analysis, web_app)
        """
        self.project_name = project_name
        self.project_path = Path(project_path).resolve()
        self.project_type = project_type
        
        # 基本目录结构
        self.base_dirs = [
            "src",
            "data/raw",
            "data/processed",
            "data/output",
            "tests/fixtures",
            "docs/api",
            "docs/user",
            "docs/dev",
            "scripts",
        ]
        
        # 项目类型特定目录
        self.type_specific_dirs = {
            "data_analysis": [
                "notebooks",
                "src/visualization",
                "src/models",
                "src/preprocessing",
            ],
            "web_app": [
                "web/assets",
                "web/components",
                "web/layouts",
                "src/api",
                "src/services",
            ]
        }
        
        # 基本文件模板
        self.base_files = {
            "README.md": self._get_readme_template,
            "pyproject.toml": self._get_pyproject_template,
            ".gitignore": self._get_gitignore_template,
            "src/__init__.py": self._get_init_template,
            "src/main.py": self._get_main_template,
            "src/config.py": self._get_config_template,
            "tests/__init__.py": lambda: "# 测试包初始化文件\n",
            "todo.md": self._get_todo_template,
            "project_summary.md": self._get_project_summary_template,
            "CHANGELOG.md": self._get_changelog_template,
        }
        
        # 项目类型特定文件
        self.type_specific_files = {
            "data_analysis": {
                "src/visualization/__init__.py": lambda: "# 可视化模块初始化文件\n",
                "src/models/__init__.py": lambda: "# 模型模块初始化文件\n",
                "src/preprocessing/__init__.py": lambda: "# 数据预处理模块初始化文件\n",
                "notebooks/example.ipynb": self._get_notebook_template,
            },
            "web_app": {
                "web/app.py": self._get_web_app_template,
                "web/components/__init__.py": lambda: "# 组件模块初始化文件\n",
                "web/layouts/__init__.py": lambda: "# 布局模块初始化文件\n",
                "src/api/__init__.py": lambda: "# API模块初始化文件\n",
                "src/services/__init__.py": lambda: "# 服务模块初始化文件\n",
            }
        }
    
    def create_project(self) -> None:
        """创建项目结构"""
        print(f"正在创建项目 '{self.project_name}' 在 {self.project_path}")
        
        # 创建项目目录
        self._create_directories()
        
        # 创建项目文件
        self._create_files()
        
        print(f"项目 '{self.project_name}' 创建成功！")
        print(f"项目路径: {self.project_path}")
        print("\n下一步:")
        print("1. 进入项目目录: cd", self.project_path)
        print("2. 创建虚拟环境: python -m venv .venv")
        print("3. 激活虚拟环境:")
        print("   - Windows: .venv\\Scripts\\activate")
        print("   - Linux/Mac: source .venv/bin/activate")
        print("4. 安装依赖: uv pip install -e .")
    
    def _create_directories(self) -> None:
        """创建项目目录结构"""
        # 创建基本目录
        for dir_path in self.base_dirs:
            self._create_directory(dir_path)
        
        # 创建项目类型特定目录
        if self.project_type in self.type_specific_dirs:
            for dir_path in self.type_specific_dirs[self.project_type]:
                self._create_directory(dir_path)
    
    def _create_directory(self, dir_path: str) -> None:
        """创建目录"""
        full_path = self.project_path / dir_path
        os.makedirs(full_path, exist_ok=True)
        print(f"创建目录: {full_path}")
    
    def _create_files(self) -> None:
        """创建项目文件"""
        # 创建基本文件
        for file_path, template_func in self.base_files.items():
            self._create_file(file_path, template_func())
        
        # 创建项目类型特定文件
        if self.project_type in self.type_specific_files:
            for file_path, template_func in self.type_specific_files[self.project_type].items():
                self._create_file(file_path, template_func())
    
    def _create_file(self, file_path: str, content: str) -> None:
        """创建文件"""
        full_path = self.project_path / file_path
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"创建文件: {full_path}")
    
    def _get_readme_template(self) -> str:
        """获取README.md模板"""
        return f"""# {self.project_name}

## 项目概述

这是一个使用Cursor项目系统规则创建的项目。

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd {self.project_name}

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\\Scripts\\activate
# Linux/Mac
source .venv/bin/activate

# 安装依赖
uv pip install -e .
```

## 使用方法

```python
from src import main

# 使用示例代码
```

## 项目结构

```
{self.project_name}/
├── src/                    # 源代码目录
├── data/                   # 数据文件目录
├── tests/                  # 测试代码
├── docs/                   # 文档
{'├── notebooks/             # Jupyter笔记本' if self.project_type == 'data_analysis' else ''}
{'├── web/                   # Web应用' if self.project_type == 'web_app' else ''}
├── scripts/                # 脚本文件
├── pyproject.toml          # 项目配置
├── README.md               # 项目说明
└── ...
```

## 许可证

[MIT](LICENSE)
"""
    
    def _get_pyproject_template(self) -> str:
        """获取pyproject.toml模板"""
        return f"""[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{self.project_name}"
version = "0.1.0"
description = "A project created with Cursor project system rules"
readme = "README.md"
requires-python = ">=3.10"
license = {{text = "MIT"}}
authors = [
    {{name = "Your Name", email = "your.email@example.com"}}
]
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "plotly>=5.14.0" if self.project_type == 'web_app' else "",
    "dash>=2.9.0" if self.project_type == 'web_app' else "",
    "matplotlib>=3.7.0" if self.project_type == 'data_analysis' else "",
    "seaborn>=0.12.0" if self.project_type == 'data_analysis' else "",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.3.1",
    "black>=23.3.0",
    "isort>=5.12.0",
    "mypy>=1.3.0",
    "pre-commit>=3.3.2",
]

[tool.setuptools]
packages = {{find = {{include = ["src"]}}}}

[tool.black]
line-length = 88
target-version = ["py310"]
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
"""
    
    def _get_gitignore_template(self) -> str:
        """获取.gitignore模板"""
        return """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS specific
.DS_Store

# Project specific
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
data/output/*
!data/output/.gitkeep
"""
    
    def _get_init_template(self) -> str:
        """获取__init__.py模板"""
        return f"""# {self.project_name} 包初始化文件

__version__ = "0.1.0"
"""
    
    def _get_main_template(self) -> str:
        """获取main.py模板"""
        return f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
{self.project_name} 主模块

此模块包含项目的主要功能入口点。
\"\"\"

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from . import config


def setup_logging() -> None:
    \"\"\"设置日志配置\"\"\"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_args() -> argparse.Namespace:
    \"\"\"解析命令行参数\"\"\"
    parser = argparse.ArgumentParser(description="{self.project_name}")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    return parser.parse_args()


def main() -> None:
    \"\"\"主函数\"\"\"
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.getLogger().setLevel(log_level)
    
    # 加载配置
    cfg = config.load_config(args.config)
    
    # 主要功能实现
    logging.info("开始执行...")
    
    # TODO: 实现主要功能
    
    logging.info("执行完成")


if __name__ == "__main__":
    setup_logging()
    main()
"""
    
    def _get_config_template(self) -> str:
        """获取config.py模板"""
        return """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
配置模块

此模块负责加载和管理项目配置。
\"\"\"

import logging
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    \"\"\"
    加载配置文件
    
    参数:
        config_path: 配置文件路径
        
    返回:
        配置字典
    
    异常:
        FileNotFoundError: 当配置文件不存在时
        yaml.YAMLError: 当配置文件格式不正确时
    \"\"\"
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logging.info(f"已加载配置文件: {config_path}")
        return config
    except yaml.YAMLError as e:
        logging.error(f"配置文件格式错误: {e}")
        raise
"""
    
    def _get_todo_template(self) -> str:
        """获取todo.md模板"""
        return f"""# {self.project_name} 任务清单

## 待完成任务
- [ ] 完善项目文档
- [ ] 实现核心功能
- [ ] 编写单元测试
- [ ] 设置CI/CD流程
- [ ] 优化性能

## 进行中任务
- 项目初始化和结构设置

## 已完成任务
- [x] 创建项目结构
"""
    
    def _get_project_summary_template(self) -> str:
        """获取project_summary.md模板"""
        return f"""# {self.project_name} - 项目摘要

## 项目概述

{self.project_name} 是一个使用Cursor项目系统规则创建的项目。

## 项目目标

1. 目标1
2. 目标2
3. 目标3

## 技术选型

1. **编程语言**：Python 3.12
2. **包管理**：uv
3. **数据处理**：NumPy和Pandas
{"4. **Web框架**：Dash" if self.project_type == 'web_app' else ""}
{"4. **数据可视化**：Matplotlib和Seaborn" if self.project_type == 'data_analysis' else ""}
5. **测试框架**：pytest

## 架构设计

本项目采用模块化架构，将不同功能分离到独立的模块中。

## 关键决策

1. 决策1
2. 决策2
3. 决策3

## 项目里程碑

1. **规划阶段**（已完成）：确定项目范围和目标
2. **开发阶段**（进行中）：实现核心功能
3. **测试阶段**（待开始）：确保功能正常工作
4. **部署阶段**（待开始）：将项目部署到生产环境
5. **维护阶段**（待开始）：修复问题并添加新功能

## 项目状态

当前项目处于开发初期阶段。
"""
    
    def _get_changelog_template(self) -> str:
        """获取CHANGELOG.md模板"""
        return f"""# 变更日志

所有项目的显著变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增
- 初始项目结构

### 变更
- 无

### 修复
- 无

## [0.1.0] - {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}
- 初始版本
"""
    
    def _get_notebook_template(self) -> str:
        """获取Jupyter笔记本模板"""
        return """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 示例笔记本\n",
    "\n",
    "这是一个示例Jupyter笔记本，用于数据分析和可视化。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 导入必要的库\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# 设置绘图样式\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "sns.set_context('notebook')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 数据加载\n",
    "\n",
    "在这一部分，我们将加载数据并进行初步探索。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 加载数据\n",
    "# df = pd.read_csv('../data/raw/example.csv')\n",
    "\n",
    "# 创建示例数据\n",
    "df = pd.DataFrame({\n",
    "    'x': np.random.normal(0, 1, 100),\n",
    "    'y': np.random.normal(0, 1, 100),\n",
    "    'category': np.random.choice(['A', 'B', 'C'], 100)\n",
    "})\n",
    "\n",
    "# 显示数据前几行\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 数据分析\n",
    "\n",
    "在这一部分，我们将对数据进行分析。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 数据描述性统计\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 数据可视化\n",
    "\n",
    "在这一部分，我们将创建一些可视化图表。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 创建散点图\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.scatterplot(data=df, x='x', y='y', hue='category')\n",
    "plt.title('散点图示例')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 结论\n",
    "\n",
    "在这一部分，我们将总结分析结果。"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
    
    def _get_web_app_template(self) -> str:
        """获取Web应用模板"""
        return f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
{self.project_name} Web应用

此模块包含Dash Web应用的入口点。
\"\"\"

import os
import sys
import logging
from pathlib import Path

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入项目模块
from src import config


# 初始化Dash应用
app = dash.Dash(
    __name__,
    meta_tags=[{{"name": "viewport", "content": "width=device-width, initial-scale=1"}}],
    suppress_callback_exceptions=True
)
app.title = "{self.project_name}"
server = app.server


# 定义应用布局
app.layout = html.Div([
    # 页面标题
    html.H1("{self.project_name}", className="app-header"),
    
    # 内容容器
    html.Div([
        # 侧边栏
        html.Div([
            html.H2("控制面板"),
            html.Hr(),
            
            # 控制组件
            html.Label("选择数据集:"),
            dcc.Dropdown(
                id="dataset-dropdown",
                options=[
                    {{"label": "数据集1", "value": "dataset1"}},
                    {{"label": "数据集2", "value": "dataset2"}},
                ],
                value="dataset1"
            ),
            
            html.Br(),
            
            html.Label("选择图表类型:"),
            dcc.RadioItems(
                id="chart-type",
                options=[
                    {{"label": "折线图", "value": "line"}},
                    {{"label": "柱状图", "value": "bar"}},
                    {{"label": "散点图", "value": "scatter"}}
                ],
                value="line"
            ),
            
            html.Br(),
            
            html.Button("更新", id="update-button", n_clicks=0)
        ], className="sidebar"),
        
        # 主内容区
        html.Div([
            # 图表容器
            html.Div([
                dcc.Graph(id="main-chart")
            ], className="chart-container"),
            
            # 数据表格
            html.Div([
                html.H3("数据表格"),
                html.Div(id="data-table")
            ], className="table-container")
        ], className="main-content")
    ], className="content-container")
], className="app-container")


# 回调函数
@app.callback(
    [Output("main-chart", "figure"),
     Output("data-table", "children")],
    [Input("update-button", "n_clicks")],
    [State("dataset-dropdown", "value"),
     State("chart-type", "value")]
)
def update_output(n_clicks, dataset, chart_type):
    \"\"\"更新图表和数据表格\"\"\"
    # 创建示例数据
    df = pd.DataFrame({{
        "x": range(10),
        "y1": [i**2 for i in range(10)],
        "y2": [10*i for i in range(10)]
    }})
    
    # 根据选择的图表类型创建图表
    if chart_type == "line":
        fig = px.line(df, x="x", y=["y1", "y2"], title=f"{{dataset}} - 折线图")
    elif chart_type == "bar":
        fig = px.bar(df, x="x", y=["y1", "y2"], title=f"{{dataset}} - 柱状图")
    else:  # scatter
        fig = px.scatter(df, x="x", y=["y1", "y2"], title=f"{{dataset}} - 散点图")
    
    # 更新图表布局
    fig.update_layout(
        template="plotly_white",
        xaxis_title="X轴",
        yaxis_title="Y轴",
        legend_title="数据系列"
    )
    
    # 创建数据表格
    table = html.Table(
        # 表头
        [html.Tr([html.Th(col) for col in df.columns])] +
        # 表格内容
        [html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(min(5, len(df)))]
    )
    
    return fig, table


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行应用
    app.run_server(debug=True, host="0.0.0.0", port=8050)
"""


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Cursor项目初始化工具")
    parser.add_argument("project_name", help="项目名称")
    parser.add_argument("--path", default=".", help="项目路径 (默认: 当前目录)")
    parser.add_argument(
        "--type", 
        choices=["default", "data_analysis", "web_app"], 
        default="default", 
        help="项目类型 (默认: default)"
    )
    
    args = parser.parse_args()
    
    # 创建项目路径
    project_path = os.path.join(args.path, args.project_name)
    
    # 检查项目路径是否已存在
    if os.path.exists(project_path):
        print(f"错误: 路径 '{project_path}' 已存在")
        sys.exit(1)
    
    # 初始化项目
    initializer = ProjectInitializer(args.project_name, project_path, args.type)
    initializer.create_project()


if __name__ == "__main__":
    main() 