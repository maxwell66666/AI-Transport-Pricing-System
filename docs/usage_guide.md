# Cursor项目系统规则使用指南

本文档提供了关于如何使用Cursor项目系统规则和相关工具的详细说明。

## 目录

1. [简介](#简介)
2. [项目结构规范](#项目结构规范)
3. [代码编写规范](#代码编写规范)
4. [工具使用指南](#工具使用指南)
   - [项目初始化工具](#项目初始化工具)
   - [项目适配工具](#项目适配工具)
5. [最佳实践](#最佳实践)
6. [常见问题](#常见问题)

## 简介

Cursor项目系统规则是一套用于在Cursor IDE中进行项目开发的规范和工具集。它旨在提高代码编写和改进的效率，使项目结构更加完善。这些规则特别适用于使用Python进行数据分析和Web应用开发的项目。

## 项目结构规范

Cursor项目系统规则定义了一个标准的项目结构，包括以下主要组件：

- `src/`: 源代码目录
- `data/`: 数据文件目录
- `tests/`: 测试代码目录
- `docs/`: 文档目录
- `scripts/`: 脚本文件目录
- 以及其他特定于项目类型的目录

详细的项目结构规范可以在[cursor_project_rules.md](../cursor_project_rules.md)文件中找到。

## 代码编写规范

Cursor项目系统规则包含了详细的代码编写规范，涵盖以下方面：

- Python代码规范
- 数据分析代码规范
- Web应用代码规范
- 版本控制规范
- 文档规范
- 测试规范

详细的代码编写规范可以在[cursor_project_rules.md](../cursor_project_rules.md)文件中找到。

## 工具使用指南

Cursor项目系统规则提供了两个主要工具来帮助您快速设置和调整项目：

### 项目初始化工具

项目初始化工具(`scripts/init_project.py`)用于创建符合Cursor项目系统规则的新项目。

#### 基本用法

```bash
python scripts/init_project.py <project_name> [--path <path>] [--type <type>]
```

#### 参数说明

- `project_name`: 项目名称（必需）
- `--path`: 项目路径（可选，默认为当前目录）
- `--type`: 项目类型（可选，默认为"default"）
  - `default`: 默认项目类型
  - `data_analysis`: 数据分析项目
  - `web_app`: Web应用项目

#### 示例

创建一个默认项目：

```bash
python scripts/init_project.py my_project
```

创建一个数据分析项目：

```bash
python scripts/init_project.py my_data_project --type data_analysis
```

创建一个Web应用项目：

```bash
python scripts/init_project.py my_web_app --type web_app --path /path/to/projects
```

### 项目适配工具

项目适配工具(`scripts/adapt_project.py`)用于将现有项目转换为符合Cursor项目系统规则的项目。

#### 基本用法

```bash
python scripts/adapt_project.py <project_path> [--type <type>] [--no-backup]
```

#### 参数说明

- `project_path`: 项目路径（必需）
- `--type`: 项目类型（可选，默认为"auto"）
  - `auto`: 自动检测项目类型
  - `default`: 默认项目类型
  - `data_analysis`: 数据分析项目
  - `web_app`: Web应用项目
- `--no-backup`: 不备份原项目（可选，默认会创建备份）

#### 示例

适配一个现有项目，自动检测项目类型：

```bash
python scripts/adapt_project.py /path/to/existing_project
```

适配一个现有项目为数据分析项目：

```bash
python scripts/adapt_project.py /path/to/existing_project --type data_analysis
```

适配一个现有项目为Web应用项目，不创建备份：

```bash
python scripts/adapt_project.py /path/to/existing_project --type web_app --no-backup
```

## 最佳实践

使用Cursor项目系统规则时，建议遵循以下最佳实践：

1. **使用虚拟环境**：为每个项目创建独立的虚拟环境，避免依赖冲突。
2. **使用uv管理包**：uv提供更快的包安装速度和更准确的依赖解析。
3. **遵循类型注解**：使用类型注解提高代码可读性和减少运行时错误。
4. **编写详细文档**：为模块、类和函数编写详细的文档字符串。
5. **定期更新todo.md**：使用todo.md跟踪任务进度，定期更新任务状态。
6. **使用版本控制**：使用Git进行版本控制，遵循约定式提交格式。
7. **编写单元测试**：为核心功能编写单元测试，确保代码质量。
8. **使用Cursor的AI辅助功能**：充分利用Cursor IDE的AI辅助功能生成和改进代码。

## 常见问题

### 如何在现有项目中应用Cursor项目系统规则？

使用项目适配工具(`scripts/adapt_project.py`)将现有项目转换为符合Cursor项目系统规则的项目。

### 如何选择合适的项目类型？

- 如果项目主要涉及数据处理、分析和可视化，选择`data_analysis`类型。
- 如果项目主要是一个Web应用，选择`web_app`类型。
- 如果不确定或项目是通用Python项目，选择`default`类型。

### 如何自定义项目结构？

虽然Cursor项目系统规则提供了标准的项目结构，但您可以根据项目需求进行适当调整。关键是保持结构的一致性和可维护性。

### 如何处理项目依赖？

使用`pyproject.toml`文件管理项目依赖，并使用uv进行包管理。这样可以确保依赖的一致性和可重现性。

### 如何集成CI/CD流程？

您可以使用GitHub Actions、GitLab CI或其他CI/CD工具，配置自动化测试、代码质量检查和部署流程。在项目根目录创建相应的配置文件即可。 