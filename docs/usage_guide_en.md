# Cursor Project System Rules Usage Guide

This document provides detailed instructions on how to use the Cursor Project System Rules and related tools.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure Specifications](#project-structure-specifications)
3. [Coding Standards](#coding-standards)
4. [Tool Usage Guide](#tool-usage-guide)
   - [Project Initialization Tool](#project-initialization-tool)
   - [Project Adaptation Tool](#project-adaptation-tool)
5. [Best Practices](#best-practices)
6. [Frequently Asked Questions](#frequently-asked-questions)

## Introduction

The Cursor Project System Rules are a set of standards and tools for project development in the Cursor IDE. They aim to improve code writing and enhancement efficiency, making project structures more robust. These rules are particularly suitable for projects using Python for data analysis and web application development.

## Project Structure Specifications

The Cursor Project System Rules define a standard project structure, including the following main components:

- `src/`: Source code directory
- `data/`: Data files directory
- `tests/`: Test code directory
- `docs/`: Documentation directory
- `scripts/`: Script files directory
- And other project type-specific directories

Detailed project structure specifications can be found in the [cursor_project_rules_en.md](../cursor_project_rules_en.md) file.

## Coding Standards

The Cursor Project System Rules include detailed coding standards covering the following aspects:

- Python code standards
- Data analysis code standards
- Web application code standards
- Version control standards
- Documentation standards
- Testing standards

Detailed coding standards can be found in the [cursor_project_rules_en.md](../cursor_project_rules_en.md) file.

## Tool Usage Guide

The Cursor Project System Rules provide two main tools to help you quickly set up and adjust projects:

### Project Initialization Tool

The Project Initialization Tool (`scripts/init_project.py`) is used to create new projects that comply with the Cursor Project System Rules.

#### Basic Usage

```bash
python scripts/init_project.py <project_name> [--path <path>] [--type <type>]
```

#### Parameter Description

- `project_name`: Project name (required)
- `--path`: Project path (optional, default is the current directory)
- `--type`: Project type (optional, default is "default")
  - `default`: Default project type
  - `data_analysis`: Data analysis project
  - `web_app`: Web application project

#### Examples

Create a default project:

```bash
python scripts/init_project.py my_project
```

Create a data analysis project:

```bash
python scripts/init_project.py my_data_project --type data_analysis
```

Create a web application project:

```bash
python scripts/init_project.py my_web_app --type web_app --path /path/to/projects
```

### Project Adaptation Tool

The Project Adaptation Tool (`scripts/adapt_project.py`) is used to convert existing projects to comply with the Cursor Project System Rules.

#### Basic Usage

```bash
python scripts/adapt_project.py <project_path> [--type <type>] [--no-backup]
```

#### Parameter Description

- `project_path`: Project path (required)
- `--type`: Project type (optional, default is "auto")
  - `auto`: Automatically detect project type
  - `default`: Default project type
  - `data_analysis`: Data analysis project
  - `web_app`: Web application project
- `--no-backup`: Do not backup the original project (optional, default is to create a backup)

#### Examples

Adapt an existing project, automatically detecting the project type:

```bash
python scripts/adapt_project.py /path/to/existing_project
```

Adapt an existing project as a data analysis project:

```bash
python scripts/adapt_project.py /path/to/existing_project --type data_analysis
```

Adapt an existing project as a web application project, without creating a backup:

```bash
python scripts/adapt_project.py /path/to/existing_project --type web_app --no-backup
```

## Best Practices

When using the Cursor Project System Rules, it is recommended to follow these best practices:

1. **Use virtual environments**: Create independent virtual environments for each project to avoid dependency conflicts.
2. **Use uv to manage packages**: uv provides faster package installation and more accurate dependency resolution.
3. **Follow type annotations**: Use type annotations to improve code readability and reduce runtime errors.
4. **Write detailed documentation**: Write detailed docstrings for modules, classes, and functions.
5. **Regularly update todo.md**: Use todo.md to track task progress and regularly update task status.
6. **Use version control**: Use Git for version control and follow the conventional commit format.
7. **Write unit tests**: Write unit tests for core functionality to ensure code quality.
8. **Use Cursor's AI assistance features**: Make full use of Cursor IDE's AI assistance features to generate and improve code.

## Frequently Asked Questions

### How do I apply the Cursor Project System Rules to an existing project?

Use the Project Adaptation Tool (`scripts/adapt_project.py`) to convert an existing project to comply with the Cursor Project System Rules.

### How do I choose the appropriate project type?

- If the project mainly involves data processing, analysis, and visualization, choose the `data_analysis` type.
- If the project is mainly a web application, choose the `web_app` type.
- If you are unsure or the project is a general Python project, choose the `default` type.

### How do I customize the project structure?

Although the Cursor Project System Rules provide a standard project structure, you can adjust it according to project requirements. The key is to maintain consistency and maintainability of the structure.

### How do I manage project dependencies?

Use the `pyproject.toml` file to manage project dependencies and use uv for package management. This ensures consistency and reproducibility of dependencies.

### How do I integrate CI/CD processes?

You can use GitHub Actions, GitLab CI, or other CI/CD tools to configure automated testing, code quality checks, and deployment processes. Create the corresponding configuration files in the project root directory. 