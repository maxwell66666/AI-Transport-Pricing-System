#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目适配器脚本

此脚本用于将现有项目转换为符合Cursor项目系统规则的项目。
它会分析现有项目结构，并进行必要的调整。
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set


class ProjectAdapter:
    """
    项目适配器类
    
    用于将现有项目转换为符合Cursor项目系统规则的项目。
    """
    
    def __init__(self, project_path: str, project_type: str = "auto", backup: bool = True):
        """
        初始化项目适配器
        
        参数:
            project_path: 项目路径
            project_type: 项目类型 (auto, default, data_analysis, web_app)
            backup: 是否备份原项目
        """
        self.project_path = Path(project_path).resolve()
        self.project_type = project_type
        self.backup = backup
        self.project_name = self.project_path.name
        
        # 标准目录结构
        self.standard_dirs = {
            "default": [
                "src",
                "data/raw",
                "data/processed",
                "data/output",
                "tests/fixtures",
                "docs/api",
                "docs/user",
                "docs/dev",
                "scripts",
            ],
            "data_analysis": [
                "src",
                "data/raw",
                "data/processed",
                "data/output",
                "tests/fixtures",
                "docs/api",
                "docs/user",
                "docs/dev",
                "scripts",
                "notebooks",
                "src/visualization",
                "src/models",
                "src/preprocessing",
            ],
            "web_app": [
                "src",
                "data/raw",
                "data/processed",
                "data/output",
                "tests/fixtures",
                "docs/api",
                "docs/user",
                "docs/dev",
                "scripts",
                "web/assets",
                "web/components",
                "web/layouts",
                "src/api",
                "src/services",
            ]
        }
        
        # 标准文件
        self.standard_files = {
            "default": [
                "README.md",
                "pyproject.toml",
                ".gitignore",
                "todo.md",
                "project_summary.md",
                "CHANGELOG.md",
            ],
            "data_analysis": [
                "README.md",
                "pyproject.toml",
                ".gitignore",
                "todo.md",
                "project_summary.md",
                "CHANGELOG.md",
                "notebooks/example.ipynb",
            ],
            "web_app": [
                "README.md",
                "pyproject.toml",
                ".gitignore",
                "todo.md",
                "project_summary.md",
                "CHANGELOG.md",
                "web/app.py",
            ]
        }
        
        # 文件模板
        self.file_templates = {
            "todo.md": self._get_todo_template,
            "project_summary.md": self._get_project_summary_template,
            "CHANGELOG.md": self._get_changelog_template,
        }
    
    def adapt_project(self) -> None:
        """将项目适配为符合Cursor项目系统规则的项目"""
        print(f"正在适配项目: {self.project_path}")
        
        # 自动检测项目类型
        if self.project_type == "auto":
            self.project_type = self._detect_project_type()
            print(f"检测到项目类型: {self.project_type}")
        
        # 备份项目
        if self.backup:
            self._backup_project()
        
        # 创建缺失的目录
        self._create_missing_directories()
        
        # 创建缺失的文件
        self._create_missing_files()
        
        # 调整项目结构
        self._adjust_project_structure()
        
        print(f"项目适配完成: {self.project_path}")
    
    def _detect_project_type(self) -> str:
        """检测项目类型"""
        # 检查是否有notebooks目录或.ipynb文件
        has_notebooks = (self.project_path / "notebooks").exists() or any(
            f.suffix == ".ipynb" for f in self.project_path.glob("**/*.ipynb")
        )
        
        # 检查是否有web相关目录或文件
        has_web = (self.project_path / "web").exists() or any(
            f.name in ["app.py", "server.py", "index.html"] 
            for f in self.project_path.glob("**/*")
        )
        
        # 根据特征确定项目类型
        if has_notebooks and not has_web:
            return "data_analysis"
        elif has_web and not has_notebooks:
            return "web_app"
        elif has_notebooks and has_web:
            # 如果两者都有，根据文件数量决定
            notebook_count = len(list(self.project_path.glob("**/*.ipynb")))
            web_file_count = len(list(self.project_path.glob("**/*.html"))) + len(list(self.project_path.glob("**/*.js")))
            
            return "data_analysis" if notebook_count > web_file_count else "web_app"
        else:
            return "default"
    
    def _backup_project(self) -> None:
        """备份项目"""
        backup_dir = self.project_path.parent / f"{self.project_path.name}_backup"
        
        # 如果备份目录已存在，添加数字后缀
        if backup_dir.exists():
            i = 1
            while (self.project_path.parent / f"{self.project_path.name}_backup_{i}").exists():
                i += 1
            backup_dir = self.project_path.parent / f"{self.project_path.name}_backup_{i}"
        
        print(f"正在备份项目到: {backup_dir}")
        shutil.copytree(self.project_path, backup_dir)
    
    def _create_missing_directories(self) -> None:
        """创建缺失的目录"""
        for dir_path in self.standard_dirs[self.project_type]:
            full_path = self.project_path / dir_path
            if not full_path.exists():
                os.makedirs(full_path, exist_ok=True)
                print(f"创建目录: {full_path}")
    
    def _create_missing_files(self) -> None:
        """创建缺失的文件"""
        for file_path in self.standard_files[self.project_type]:
            full_path = self.project_path / file_path
            
            # 如果文件不存在且有模板，则创建
            if not full_path.exists() and file_path in self.file_templates:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(self.file_templates[file_path]())
                
                print(f"创建文件: {full_path}")
    
    def _adjust_project_structure(self) -> None:
        """调整项目结构"""
        # 检查是否有Python文件在根目录，应该移动到src目录
        python_files = [f for f in self.project_path.glob("*.py") 
                       if f.name not in ["setup.py", "pyproject.toml"]]
        
        if python_files:
            src_dir = self.project_path / "src"
            os.makedirs(src_dir, exist_ok=True)
            
            # 确保src目录有__init__.py
            init_file = src_dir / "__init__.py"
            if not init_file.exists():
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {self.project_name} 包初始化文件\n\n__version__ = \"0.1.0\"\n")
            
            # 移动Python文件到src目录
            for py_file in python_files:
                target_file = src_dir / py_file.name
                if not target_file.exists():
                    shutil.move(py_file, target_file)
                    print(f"移动文件: {py_file} -> {target_file}")
        
        # 检查是否有数据文件在根目录，应该移动到data目录
        data_extensions = [".csv", ".json", ".xlsx", ".xls", ".parquet", ".feather", ".pickle", ".pkl"]
        data_files = [f for f in self.project_path.glob("*.*") 
                     if f.suffix.lower() in data_extensions]
        
        if data_files:
            data_dir = self.project_path / "data/raw"
            os.makedirs(data_dir, exist_ok=True)
            
            # 移动数据文件到data/raw目录
            for data_file in data_files:
                target_file = data_dir / data_file.name
                if not target_file.exists():
                    shutil.move(data_file, target_file)
                    print(f"移动文件: {data_file} -> {target_file}")
        
        # 检查是否有Jupyter笔记本在根目录，应该移动到notebooks目录
        notebook_files = [f for f in self.project_path.glob("*.ipynb")]
        
        if notebook_files and self.project_type in ["data_analysis", "default"]:
            notebooks_dir = self.project_path / "notebooks"
            os.makedirs(notebooks_dir, exist_ok=True)
            
            # 移动笔记本文件到notebooks目录
            for nb_file in notebook_files:
                target_file = notebooks_dir / nb_file.name
                if not target_file.exists():
                    shutil.move(nb_file, target_file)
                    print(f"移动文件: {nb_file} -> {target_file}")
    
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
- 项目适配和结构调整

## 已完成任务
- [x] 初始化项目结构
"""
    
    def _get_project_summary_template(self) -> str:
        """获取project_summary.md模板"""
        return f"""# {self.project_name} - 项目摘要

## 项目概述

{self.project_name} 是一个使用Cursor项目系统规则适配的项目。

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

当前项目处于开发阶段。
"""
    
    def _get_changelog_template(self) -> str:
        """获取CHANGELOG.md模板"""
        return f"""# 变更日志

所有项目的显著变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增
- 项目结构调整，符合Cursor项目系统规则

### 变更
- 无

### 修复
- 无

## [0.1.0] - {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}
- 初始版本
"""


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Cursor项目适配工具")
    parser.add_argument("project_path", help="项目路径")
    parser.add_argument(
        "--type", 
        choices=["auto", "default", "data_analysis", "web_app"], 
        default="auto", 
        help="项目类型 (默认: auto)"
    )
    parser.add_argument(
        "--no-backup", 
        action="store_true", 
        help="不备份原项目"
    )
    
    args = parser.parse_args()
    
    # 检查项目路径是否存在
    if not os.path.exists(args.project_path):
        print(f"错误: 路径 '{args.project_path}' 不存在")
        sys.exit(1)
    
    # 适配项目
    adapter = ProjectAdapter(args.project_path, args.type, not args.no_backup)
    adapter.adapt_project()


if __name__ == "__main__":
    main() 