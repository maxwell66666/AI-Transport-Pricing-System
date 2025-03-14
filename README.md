# AI运输报价系统

## 项目概述

AI运输报价系统是一个基于机器学习的智能运输定价和决策支持系统，专为艺术品运输设计。系统能够根据多种因素（如运输方式、货物类型、重量、体积等）预测运输价格，并提供最佳运输方案建议。

## 主要功能

- **价格预测**：基于机器学习模型预测运输价格
- **运输方案推荐**：根据货物特性和客户需求推荐最佳运输方式
- **风险评估**：评估不同运输方案的风险
- **时间规划**：提供运输时间估计和规划
- **成本估算**：详细的运输成本明细

## 技术栈

- **后端**：Python, FastAPI
- **数据库**：SQLite (开发), PostgreSQL (生产)
- **机器学习**：Scikit-learn, Pandas, NumPy
- **API文档**：Swagger UI, ReDoc

## 安装与运行

### 环境要求

- Python 3.8+
- pip 或 poetry

### 安装依赖

```bash
pip install -r requirements.txt
```

或使用poetry:

```bash
poetry install
```

### 配置环境变量

复制`.env.example`文件为`.env`并根据需要修改配置。

### 初始化数据库

```bash
python -m src.scripts.init_project
```

### 运行应用

```bash
python -m uvicorn src.main:app --reload
```

应用将在 http://localhost:8000 运行。

## API文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 最近更新

### 2023-03-14
- 修复了价格预测API的多个问题：
  - 添加了`chargeable_weight`计算逻辑
  - 添加了`deadline`字段
  - 添加了`Length`, `Width`, `Height`尺寸计算
  - 添加了`Artwork_Type`字段
  - 修正了价格预测结果的提取方式，从`price_prediction`改为`price_prediction_details`
- 所有测试现在都能成功通过
- API接口现在可以正常工作

## 开发者指南

### 项目结构

```
AI_Transport_Pricing_System/
├── data/                  # 数据文件
├── src/                   # 源代码
│   ├── api/               # API定义
│   ├── core/              # 核心功能
│   ├── db/                # 数据库模型和操作
│   ├── models/            # 机器学习模型
│   ├── preprocessing/     # 数据预处理
│   ├── services/          # 业务服务
│   └── utils/             # 工具函数
├── tests/                 # 测试代码
├── .env.example           # 环境变量示例
├── requirements.txt       # 依赖列表
└── README.md              # 项目说明
```

### 添加新功能

1. 在相应模块中实现功能
2. 添加单元测试
3. 更新API文档
4. 提交PR

## 许可证

[MIT](LICENSE)

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。 