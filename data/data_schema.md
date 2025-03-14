# AI辅助运输报价系统 - 数据结构定义

本文档定义了AI辅助运输报价系统所需的主要数据结构和数据模型。

## 1. 数据库架构概览

系统使用关系型数据库存储结构化数据，主要包括以下数据类别：

1. 用户和权限数据
2. 运输报价数据
3. 规则和配置数据
4. 参考数据（如地点、货物类型等）
5. 系统日志和审计数据

## 2. 主要数据表定义

### 2.1 用户和权限数据

#### 2.1.1 users（用户表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 用户ID | 主键, 自增 |
| username | VARCHAR(50) | 用户名 | 唯一, 非空 |
| email | VARCHAR(100) | 电子邮件 | 唯一, 非空 |
| password_hash | VARCHAR(255) | 密码哈希 | 非空 |
| full_name | VARCHAR(100) | 用户全名 | 非空 |
| role_id | INT | 角色ID | 外键(roles.id), 非空 |
| is_active | BOOLEAN | 是否激活 | 默认true |
| last_login | TIMESTAMP | 最后登录时间 | 可空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |
| updated_at | TIMESTAMP | 更新时间 | 非空, 默认当前时间 |

#### 2.1.2 roles（角色表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 角色ID | 主键, 自增 |
| name | VARCHAR(50) | 角色名称 | 唯一, 非空 |
| description | VARCHAR(255) | 角色描述 | 可空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |
| updated_at | TIMESTAMP | 更新时间 | 非空, 默认当前时间 |

#### 2.1.3 permissions（权限表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 权限ID | 主键, 自增 |
| name | VARCHAR(50) | 权限名称 | 唯一, 非空 |
| description | VARCHAR(255) | 权限描述 | 可空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |
| updated_at | TIMESTAMP | 更新时间 | 非空, 默认当前时间 |

#### 2.1.4 role_permissions（角色权限关联表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| role_id | INT | 角色ID | 主键, 外键(roles.id) |
| permission_id | INT | 权限ID | 主键, 外键(permissions.id) |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |

### 2.2 运输报价数据

#### 2.2.1 quotes（报价表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 报价ID | 主键, 自增 |
| quote_number | VARCHAR(20) | 报价编号 | 唯一, 非空 |
| user_id | INT | 创建用户ID | 外键(users.id), 非空 |
| customer_id | INT | 客户ID | 外键(customers.id), 可空 |
| origin_location_id | INT | 起始地点ID | 外键(locations.id), 非空 |
| destination_location_id | INT | 目的地点ID | 外键(locations.id), 非空 |
| cargo_type_id | INT | 货物类型ID | 外键(cargo_types.id), 非空 |
| weight | DECIMAL(10,2) | 重量(kg) | 非空 |
| volume | DECIMAL(10,2) | 体积(m³) | 非空 |
| transport_mode_id | INT | 运输方式ID | 外键(transport_modes.id), 非空 |
| expected_delivery_date | DATE | 期望交付日期 | 可空 |
| special_requirements | TEXT | 特殊要求 | 可空 |
| status | VARCHAR(20) | 状态 | 非空, 默认'draft' |
| total_price | DECIMAL(12,2) | 总价格 | 非空 |
| currency | VARCHAR(3) | 货币代码 | 非空, 默认'USD' |
| is_llm_assisted | BOOLEAN | 是否使用LLM辅助 | 非空, 默认false |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |
| updated_at | TIMESTAMP | 更新时间 | 非空, 默认当前时间 |

#### 2.2.2 quote_details（报价明细表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 明细ID | 主键, 自增 |
| quote_id | INT | 报价ID | 外键(quotes.id), 非空 |
| description | VARCHAR(255) | 费用描述 | 非空 |
| category | VARCHAR(50) | 费用类别 | 非空 |
| amount | DECIMAL(10,2) | 金额 | 非空 |
| is_taxable | BOOLEAN | 是否应税 | 非空, 默认false |
| tax_rate | DECIMAL(5,2) | 税率 | 可空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |
| updated_at | TIMESTAMP | 更新时间 | 非空, 默认当前时间 |

#### 2.2.3 quote_options（报价选项表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 选项ID | 主键, 自增 |
| quote_id | INT | 报价ID | 外键(quotes.id), 非空 |
| transport_mode_id | INT | 运输方式ID | 外键(transport_modes.id), 非空 |
| transit_time | INT | 运输时间(天) | 非空 |
| total_price | DECIMAL(12,2) | 总价格 | 非空 |
| is_recommended | BOOLEAN | 是否推荐 | 非空, 默认false |
| recommendation_reason | TEXT | 推荐理由 | 可空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |
| updated_at | TIMESTAMP | 更新时间 | 非空, 默认当前时间 |

#### 2.2.4 quote_adjustments（报价调整表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 调整ID | 主键, 自增 |
| quote_id | INT | 报价ID | 外键(quotes.id), 非空 |
| user_id | INT | 调整用户ID | 外键(users.id), 非空 |
| original_price | DECIMAL(12,2) | 原始价格 | 非空 |
| adjusted_price | DECIMAL(12,2) | 调整后价格 | 非空 |
| adjustment_reason | TEXT | 调整原因 | 非空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |

#### 2.2.5 customers（客户表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 客户ID | 主键, 自增 |
| name | VARCHAR(100) | 客户名称 | 非空 |
| contact_person | VARCHAR(100) | 联系人 | 可空 |
| email | VARCHAR(100) | 电子邮件 | 可空 |
| phone | VARCHAR(20) | 电话 | 可空 |
| address | TEXT | 地址 | 可空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |
| updated_at | TIMESTAMP | 更新时间 | 非空, 默认当前时间 |

### 2.3 规则和配置数据

#### 2.3.1 rules（规则表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 规则ID | 主键, 自增 |
| name | VARCHAR(100) | 规则名称 | 非空 |
| description | TEXT | 规则描述 | 可空 |
| rule_category_id | INT | 规则类别ID | 外键(rule_categories.id), 非空 |
| rule_definition | JSON | 规则定义 | 非空 |
| priority | INT | 优先级 | 非空, 默认0 |
| is_active | BOOLEAN | 是否激活 | 非空, 默认true |
| created_by | INT | 创建用户ID | 外键(users.id), 非空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |
| updated_at | TIMESTAMP | 更新时间 | 非空, 默认当前时间 |

#### 2.3.2 rule_categories（规则类别表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 类别ID | 主键, 自增 |
| name | VARCHAR(50) | 类别名称 | 唯一, 非空 |
| description | VARCHAR(255) | 类别描述 | 可空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |
| updated_at | TIMESTAMP | 更新时间 | 非空, 默认当前时间 |

#### 2.3.3 rule_history（规则历史表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 历史ID | 主键, 自增 |
| rule_id | INT | 规则ID | 外键(rules.id), 非空 |
| user_id | INT | 修改用户ID | 外键(users.id), 非空 |
| change_type | VARCHAR(20) | 变更类型 | 非空 |
| old_definition | JSON | 旧规则定义 | 可空 |
| new_definition | JSON | 新规则定义 | 可空 |
| change_reason | TEXT | 变更原因 | 可空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |

#### 2.3.4 llm_prompts（LLM提示模板表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 提示ID | 主键, 自增 |
| name | VARCHAR(100) | 提示名称 | 非空 |
| description | TEXT | 提示描述 | 可空 |
| prompt_template | TEXT | 提示模板 | 非空 |
| purpose | VARCHAR(50) | 用途 | 非空 |
| is_active | BOOLEAN | 是否激活 | 非空, 默认true |
| created_by | INT | 创建用户ID | 外键(users.id), 非空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |
| updated_at | TIMESTAMP | 更新时间 | 非空, 默认当前时间 |

### 2.4 参考数据

#### 2.4.1 locations（地点表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 地点ID | 主键, 自增 |
| name | VARCHAR(100) | 地点名称 | 非空 |
| country | VARCHAR(50) | 国家 | 非空 |
| city | VARCHAR(50) | 城市 | 非空 |
| address | TEXT | 详细地址 | 可空 |
| postal_code | VARCHAR(20) | 邮政编码 | 可空 |
| latitude | DECIMAL(10,6) | 纬度 | 可空 |
| longitude | DECIMAL(10,6) | 经度 | 可空 |
| is_port | BOOLEAN | 是否港口 | 非空, 默认false |
| is_airport | BOOLEAN | 是否机场 | 非空, 默认false |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |
| updated_at | TIMESTAMP | 更新时间 | 非空, 默认当前时间 |

#### 2.4.2 transport_modes（运输方式表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 运输方式ID | 主键, 自增 |
| name | VARCHAR(50) | 运输方式名称 | 唯一, 非空 |
| description | VARCHAR(255) | 描述 | 可空 |
| is_active | BOOLEAN | 是否激活 | 非空, 默认true |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |
| updated_at | TIMESTAMP | 更新时间 | 非空, 默认当前时间 |

#### 2.4.3 cargo_types（货物类型表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 货物类型ID | 主键, 自增 |
| name | VARCHAR(50) | 货物类型名称 | 唯一, 非空 |
| description | VARCHAR(255) | 描述 | 可空 |
| is_dangerous | BOOLEAN | 是否危险品 | 非空, 默认false |
| requires_temperature_control | BOOLEAN | 是否需要温控 | 非空, 默认false |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |
| updated_at | TIMESTAMP | 更新时间 | 非空, 默认当前时间 |

#### 2.4.4 distance_matrix（距离矩阵表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 距离ID | 主键, 自增 |
| origin_location_id | INT | 起始地点ID | 外键(locations.id), 非空 |
| destination_location_id | INT | 目的地点ID | 外键(locations.id), 非空 |
| transport_mode_id | INT | 运输方式ID | 外键(transport_modes.id), 非空 |
| distance | DECIMAL(10,2) | 距离(km) | 非空 |
| typical_transit_time | INT | 典型运输时间(天) | 非空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |
| updated_at | TIMESTAMP | 更新时间 | 非空, 默认当前时间 |

### 2.5 系统日志和审计数据

#### 2.5.1 system_logs（系统日志表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 日志ID | 主键, 自增 |
| log_level | VARCHAR(10) | 日志级别 | 非空 |
| component | VARCHAR(50) | 组件 | 非空 |
| message | TEXT | 日志消息 | 非空 |
| details | JSON | 详细信息 | 可空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |

#### 2.5.2 user_activity_logs（用户活动日志表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 日志ID | 主键, 自增 |
| user_id | INT | 用户ID | 外键(users.id), 可空 |
| activity_type | VARCHAR(50) | 活动类型 | 非空 |
| resource_type | VARCHAR(50) | 资源类型 | 可空 |
| resource_id | INT | 资源ID | 可空 |
| details | JSON | 详细信息 | 可空 |
| ip_address | VARCHAR(45) | IP地址 | 可空 |
| user_agent | VARCHAR(255) | 用户代理 | 可空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |

#### 2.5.3 llm_api_calls（LLM API调用日志表）

| 字段名 | 数据类型 | 描述 | 约束 |
|--------|----------|------|------|
| id | INT | 日志ID | 主键, 自增 |
| user_id | INT | 用户ID | 外键(users.id), 可空 |
| prompt_id | INT | 提示ID | 外键(llm_prompts.id), 可空 |
| request_data | JSON | 请求数据 | 非空 |
| response_data | JSON | 响应数据 | 可空 |
| tokens_used | INT | 使用的令牌数 | 可空 |
| processing_time | INT | 处理时间(ms) | 可空 |
| status | VARCHAR(20) | 状态 | 非空 |
| error_message | TEXT | 错误消息 | 可空 |
| created_at | TIMESTAMP | 创建时间 | 非空, 默认当前时间 |

## 3. 数据关系图

```
users 1--* quotes (创建者)
users 1--* quote_adjustments (调整者)
users 1--* rules (创建者)
users 1--* llm_prompts (创建者)
users 1--* user_activity_logs

roles 1--* users
roles *--* permissions (通过role_permissions)

quotes 1--* quote_details
quotes 1--* quote_options
quotes 1--* quote_adjustments
quotes *--1 customers
quotes *--1 locations (起点)
quotes *--1 locations (终点)
quotes *--1 cargo_types
quotes *--1 transport_modes

rules *--1 rule_categories
rules 1--* rule_history

llm_prompts 1--* llm_api_calls

locations *--* locations (通过distance_matrix)
```

## 4. 示例数据

系统将包含一些预定义的示例数据，用于测试和演示目的。示例数据将包括：

1. 预定义的用户角色和权限
2. 常用的运输方式
3. 常见的货物类型
4. 主要城市和港口的位置信息
5. 基本的报价规则

示例数据将存储在SQL脚本中，可以在系统初始化时导入。

## 5. 数据迁移和版本控制

系统将使用数据库迁移工具（如Flyway或Liquibase）来管理数据库架构的变更和版本控制。每个迁移脚本将包含：

1. 版本号
2. 描述
3. 变更SQL
4. 回滚SQL（如果适用）

这将确保数据库架构的变更可以被追踪和版本控制，并且可以在不同环境中一致地应用。 