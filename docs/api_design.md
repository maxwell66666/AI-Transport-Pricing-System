# AI辅助运输报价系统 - API设计文档

本文档定义了AI辅助运输报价系统的API接口设计，包括RESTful API端点、请求/响应格式、认证方式和错误处理。

## 1. API概述

### 1.1 基本信息

- **基础URL**: `https://api.transport-pricing-system.com/v1`
- **API版本**: v1
- **数据格式**: JSON
- **认证方式**: JWT (JSON Web Token)
- **跨域支持**: 支持CORS

### 1.2 API分组

API接口按功能分为以下几组：

1. **认证API**: 用户登录、注册、令牌刷新等
2. **用户API**: 用户管理、角色和权限
3. **报价API**: 报价创建、查询、更新和删除
4. **客户API**: 客户信息管理
5. **规则API**: 定价规则管理
6. **参考数据API**: 地点、运输方式、货物类型等
7. **AI辅助API**: AI辅助报价、规则提取等
8. **分析API**: 数据分析和报告

## 2. 认证与授权

### 2.1 认证流程

1. 客户端通过`/auth/login`端点提交用户名和密码
2. 服务器验证凭据并返回访问令牌(access token)和刷新令牌(refresh token)
3. 客户端在后续请求的Authorization头中包含访问令牌
4. 当访问令牌过期时，客户端使用刷新令牌获取新的访问令牌

### 2.2 认证API端点

#### 2.2.1 用户登录

```
POST /auth/login
```

请求体:
```json
{
  "username": "user@example.com",
  "password": "password123"
}
```

响应:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "user": {
    "id": 1,
    "username": "user@example.com",
    "full_name": "John Doe",
    "role": "manager"
  }
}
```

#### 2.2.2 令牌刷新

```
POST /auth/refresh
```

请求体:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

响应:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

#### 2.2.3 用户注销

```
POST /auth/logout
```

请求头:
```
Authorization: Bearer {access_token}
```

响应:
```json
{
  "message": "Successfully logged out"
}
```

### 2.3 授权控制

API使用基于角色的访问控制(RBAC)，每个端点都有相应的权限要求。常见角色包括：

- **admin**: 系统管理员，拥有所有权限
- **manager**: 管理人员，可以管理报价、规则和用户
- **operator**: 操作人员，可以创建和查看报价
- **viewer**: 查看者，只能查看报价和报告

## 3. 报价API

### 3.1 创建报价

```
POST /quotes
```

请求头:
```
Authorization: Bearer {access_token}
```

请求体:
```json
{
  "customer_id": 1,
  "origin_location_id": 1,
  "destination_location_id": 2,
  "cargo_type_id": 1,
  "weight": 5000.00,
  "volume": 20.00,
  "transport_mode_id": 2,
  "expected_delivery_date": "2023-04-15",
  "special_requirements": "None",
  "currency": "USD",
  "use_ai_assistance": true
}
```

响应:
```json
{
  "id": 1,
  "quote_number": "Q20230001",
  "status": "draft",
  "total_price": 8500.00,
  "currency": "USD",
  "is_llm_assisted": true,
  "created_at": "2023-03-01T10:00:00Z",
  "details": [
    {
      "id": 1,
      "description": "Base shipping cost",
      "category": "base",
      "amount": 7000.00
    },
    {
      "id": 2,
      "description": "Fuel surcharge",
      "category": "surcharge",
      "amount": 700.00
    },
    {
      "id": 3,
      "description": "Documentation fee",
      "category": "fee",
      "amount": 150.00
    },
    {
      "id": 4,
      "description": "Insurance",
      "category": "insurance",
      "amount": 650.00
    }
  ],
  "customer": {
    "id": 1,
    "name": "Acme Corporation"
  },
  "origin_location": {
    "id": 1,
    "name": "Shanghai Port"
  },
  "destination_location": {
    "id": 2,
    "name": "Los Angeles Port"
  },
  "cargo_type": {
    "id": 1,
    "name": "General Cargo"
  },
  "transport_mode": {
    "id": 2,
    "name": "Sea Freight"
  }
}
```

### 3.2 获取报价列表

```
GET /quotes
```

请求头:
```
Authorization: Bearer {access_token}
```

查询参数:
```
page=1
per_page=10
status=draft,confirmed
sort=created_at
order=desc
customer_id=1
```

响应:
```json
{
  "data": [
    {
      "id": 1,
      "quote_number": "Q20230001",
      "customer": {
        "id": 1,
        "name": "Acme Corporation"
      },
      "origin_location": {
        "id": 1,
        "name": "Shanghai Port"
      },
      "destination_location": {
        "id": 2,
        "name": "Los Angeles Port"
      },
      "status": "confirmed",
      "total_price": 8500.00,
      "currency": "USD",
      "created_at": "2023-03-01T10:00:00Z"
    },
    // 更多报价...
  ],
  "meta": {
    "current_page": 1,
    "per_page": 10,
    "total_pages": 5,
    "total_count": 42
  }
}
```

### 3.3 获取报价详情

```
GET /quotes/{id}
```

请求头:
```
Authorization: Bearer {access_token}
```

响应:
```json
{
  "id": 1,
  "quote_number": "Q20230001",
  "user_id": 1,
  "customer_id": 1,
  "origin_location_id": 1,
  "destination_location_id": 2,
  "cargo_type_id": 1,
  "weight": 5000.00,
  "volume": 20.00,
  "transport_mode_id": 2,
  "expected_delivery_date": "2023-04-15",
  "special_requirements": "None",
  "status": "confirmed",
  "total_price": 8500.00,
  "currency": "USD",
  "is_llm_assisted": false,
  "created_at": "2023-03-01T10:00:00Z",
  "updated_at": "2023-03-01T14:30:00Z",
  "details": [
    {
      "id": 1,
      "description": "Base shipping cost",
      "category": "base",
      "amount": 7000.00,
      "is_taxable": true,
      "tax_rate": 0.00
    },
    // 更多费用明细...
  ],
  "options": [
    {
      "id": 1,
      "transport_mode_id": 2,
      "transit_time": 18,
      "total_price": 8500.00,
      "is_recommended": true,
      "recommendation_reason": "Most cost-effective option"
    },
    // 更多选项...
  ],
  "adjustments": [
    {
      "id": 1,
      "user_id": 2,
      "original_price": 8700.00,
      "adjusted_price": 8500.00,
      "adjustment_reason": "Volume discount applied",
      "created_at": "2023-03-01T13:45:00Z"
    }
  ],
  "customer": {
    "id": 1,
    "name": "Acme Corporation",
    "contact_person": "John Smith",
    "email": "john@acme.com",
    "phone": "+1-123-456-7890"
  },
  "origin_location": {
    "id": 1,
    "name": "Shanghai Port",
    "country": "China",
    "city": "Shanghai"
  },
  "destination_location": {
    "id": 2,
    "name": "Los Angeles Port",
    "country": "USA",
    "city": "Los Angeles"
  },
  "cargo_type": {
    "id": 1,
    "name": "General Cargo",
    "description": "Standard non-hazardous goods"
  },
  "transport_mode": {
    "id": 2,
    "name": "Sea Freight",
    "description": "Transportation of goods by sea"
  },
  "created_by": {
    "id": 1,
    "username": "user@example.com",
    "full_name": "John Doe"
  }
}
```

### 3.4 更新报价

```
PUT /quotes/{id}
```

请求头:
```
Authorization: Bearer {access_token}
```

请求体:
```json
{
  "weight": 5200.00,
  "volume": 21.50,
  "special_requirements": "Handle with care",
  "expected_delivery_date": "2023-04-20"
}
```

响应:
```json
{
  "id": 1,
  "quote_number": "Q20230001",
  "status": "draft",
  "total_price": 8800.00,
  // 其他字段...
}
```

### 3.5 确认报价

```
POST /quotes/{id}/confirm
```

请求头:
```
Authorization: Bearer {access_token}
```

响应:
```json
{
  "id": 1,
  "quote_number": "Q20230001",
  "status": "confirmed",
  "confirmed_at": "2023-03-02T09:15:00Z",
  "confirmed_by": {
    "id": 1,
    "username": "user@example.com",
    "full_name": "John Doe"
  }
}
```

### 3.6 删除报价

```
DELETE /quotes/{id}
```

请求头:
```
Authorization: Bearer {access_token}
```

响应:
```json
{
  "message": "Quote successfully deleted"
}
```

## 4. 规则API

### 4.1 创建规则

```
POST /rules
```

请求头:
```
Authorization: Bearer {access_token}
```

请求体:
```json
{
  "name": "Sea Freight Base Rate",
  "description": "Base rate calculation for sea freight",
  "rule_category_id": 1,
  "rule_definition": {
    "condition": {
      "transport_mode_id": 2
    },
    "formula": "weight * 1.2 + volume * 250",
    "min_charge": 500
  },
  "priority": 100,
  "is_active": true
}
```

响应:
```json
{
  "id": 1,
  "name": "Sea Freight Base Rate",
  "description": "Base rate calculation for sea freight",
  "rule_category_id": 1,
  "rule_definition": {
    "condition": {
      "transport_mode_id": 2
    },
    "formula": "weight * 1.2 + volume * 250",
    "min_charge": 500
  },
  "priority": 100,
  "is_active": true,
  "created_by": 1,
  "created_at": "2023-03-01T10:00:00Z",
  "updated_at": "2023-03-01T10:00:00Z"
}
```

### 4.2 获取规则列表

```
GET /rules
```

请求头:
```
Authorization: Bearer {access_token}
```

查询参数:
```
page=1
per_page=10
category_id=1
is_active=true
```

响应:
```json
{
  "data": [
    {
      "id": 1,
      "name": "Sea Freight Base Rate",
      "rule_category": {
        "id": 1,
        "name": "Base Pricing"
      },
      "priority": 100,
      "is_active": true,
      "created_at": "2023-03-01T10:00:00Z"
    },
    // 更多规则...
  ],
  "meta": {
    "current_page": 1,
    "per_page": 10,
    "total_pages": 3,
    "total_count": 25
  }
}
```

### 4.3 获取规则详情

```
GET /rules/{id}
```

请求头:
```
Authorization: Bearer {access_token}
```

响应:
```json
{
  "id": 1,
  "name": "Sea Freight Base Rate",
  "description": "Base rate calculation for sea freight",
  "rule_category": {
    "id": 1,
    "name": "Base Pricing",
    "description": "Rules for calculating base shipping costs"
  },
  "rule_definition": {
    "condition": {
      "transport_mode_id": 2
    },
    "formula": "weight * 1.2 + volume * 250",
    "min_charge": 500
  },
  "priority": 100,
  "is_active": true,
  "created_by": {
    "id": 1,
    "username": "user@example.com",
    "full_name": "John Doe"
  },
  "created_at": "2023-03-01T10:00:00Z",
  "updated_at": "2023-03-01T10:00:00Z",
  "history": [
    {
      "id": 1,
      "user_id": 1,
      "change_type": "create",
      "change_reason": "Initial creation",
      "created_at": "2023-03-01T10:00:00Z"
    }
  ]
}
```

### 4.4 更新规则

```
PUT /rules/{id}
```

请求头:
```
Authorization: Bearer {access_token}
```

请求体:
```json
{
  "description": "Updated base rate calculation for sea freight",
  "rule_definition": {
    "condition": {
      "transport_mode_id": 2
    },
    "formula": "weight * 1.3 + volume * 260",
    "min_charge": 550
  },
  "priority": 110,
  "is_active": true
}
```

响应:
```json
{
  "id": 1,
  "name": "Sea Freight Base Rate",
  "description": "Updated base rate calculation for sea freight",
  "rule_category_id": 1,
  "rule_definition": {
    "condition": {
      "transport_mode_id": 2
    },
    "formula": "weight * 1.3 + volume * 260",
    "min_charge": 550
  },
  "priority": 110,
  "is_active": true,
  "updated_at": "2023-03-02T14:30:00Z"
}
```

### 4.5 测试规则

```
POST /rules/test
```

请求头:
```
Authorization: Bearer {access_token}
```

请求体:
```json
{
  "rule_definition": {
    "condition": {
      "transport_mode_id": 2
    },
    "formula": "weight * 1.3 + volume * 260",
    "min_charge": 550
  },
  "test_data": {
    "weight": 5000,
    "volume": 20,
    "transport_mode_id": 2
  }
}
```

响应:
```json
{
  "result": 11700.0,
  "calculation_steps": [
    "weight * 1.3 = 5000 * 1.3 = 6500",
    "volume * 260 = 20 * 260 = 5200",
    "6500 + 5200 = 11700",
    "11700 > min_charge(550), so result = 11700"
  ],
  "is_condition_met": true
}
```

## 5. AI辅助API

### 5.1 AI辅助报价生成

```
POST /ai/generate-quote
```

请求头:
```
Authorization: Bearer {access_token}
```

请求体:
```json
{
  "origin_location_id": 1,
  "destination_location_id": 2,
  "cargo_type_id": 1,
  "weight": 5000.00,
  "volume": 20.00,
  "transport_mode_id": 2,
  "special_requirements": "None"
}
```

响应:
```json
{
  "quote": {
    "base_price": 7000.00,
    "details": [
      {
        "description": "Base shipping cost",
        "category": "base",
        "amount": 7000.00
      },
      {
        "description": "Fuel surcharge",
        "category": "surcharge",
        "amount": 700.00
      },
      {
        "description": "Documentation fee",
        "category": "fee",
        "amount": 150.00
      },
      {
        "description": "Insurance",
        "category": "insurance",
        "amount": 650.00
      }
    ],
    "total_price": 8500.00,
    "currency": "USD",
    "estimated_transit_time": 18
  },
  "explanation": "This quote is based on current market rates for sea freight between Shanghai and Los Angeles. The base rate is calculated using our standard formula for general cargo. A 10% fuel surcharge is applied due to current fuel prices. Documentation and insurance fees are standard for this route.",
  "similar_cases": [
    {
      "quote_number": "Q20230010",
      "similarity_score": 0.92,
      "key_differences": "Similar weight but slightly different volume",
      "price": 8300.00
    },
    {
      "quote_number": "Q20230015",
      "similarity_score": 0.85,
      "key_differences": "Same route but different cargo type",
      "price": 8700.00
    }
  ],
  "options": [
    {
      "transport_mode_id": 2,
      "name": "Sea Freight (Standard)",
      "transit_time": 18,
      "total_price": 8500.00,
      "is_recommended": true,
      "recommendation_reason": "Most cost-effective option"
    },
    {
      "transport_mode_id": 1,
      "name": "Air Freight",
      "transit_time": 3,
      "total_price": 22500.00,
      "is_recommended": false,
      "recommendation_reason": null
    }
  ]
}
```

### 5.2 AI规则提取

```
POST /ai/extract-rules
```

请求头:
```
Authorization: Bearer {access_token}
```

请求体:
```json
{
  "data_source": "historical_quotes",
  "filters": {
    "date_range": {
      "start": "2022-01-01",
      "end": "2022-12-31"
    },
    "transport_mode_id": 2
  },
  "extraction_parameters": {
    "confidence_threshold": 0.7,
    "max_rules": 5
  }
}
```

响应:
```json
{
  "extracted_rules": [
    {
      "name": "Sea Freight Base Rate",
      "description": "Base rate calculation for sea freight",
      "rule_category_id": 1,
      "rule_definition": {
        "condition": {
          "transport_mode_id": 2
        },
        "formula": "weight * 1.2 + volume * 250",
        "min_charge": 500
      },
      "confidence": 0.95,
      "supporting_data_count": 120,
      "suggested_priority": 100
    },
    {
      "name": "Heavy Cargo Surcharge",
      "description": "Additional charge for heavy cargo",
      "rule_category_id": 2,
      "rule_definition": {
        "condition": {
          "weight": {
            "operator": ">=",
            "value": 10000
          },
          "transport_mode_id": 2
        },
        "formula": "base_price * 0.15",
        "min_charge": 1000
      },
      "confidence": 0.85,
      "supporting_data_count": 45,
      "suggested_priority": 50
    },
    // 更多提取的规则...
  ],
  "analysis_summary": "Analysis based on 250 historical quotes. Strong patterns identified for base pricing and several surcharges. Recommend manual review of the Heavy Cargo Surcharge rule as there are some outliers in the data."
}
```

### 5.3 AI报价解释

```
POST /ai/explain-quote
```

请求头:
```
Authorization: Bearer {access_token}
```

请求体:
```json
{
  "quote_id": 1,
  "explanation_type": "customer"
}
```

响应:
```json
{
  "explanation": {
    "introduction": "Thank you for choosing our logistics services. We're pleased to provide you with a competitive quote for shipping your general cargo from Shanghai to Los Angeles.",
    "price_breakdown": "Your quote of $8,500.00 includes a base shipping cost of $7,000.00, which is calculated based on the weight and volume of your shipment. Additionally, there is a fuel surcharge of $700.00, a documentation fee of $150.00, and insurance coverage of $650.00.",
    "transit_information": "The estimated transit time for this shipment is 18 days by sea freight. Your cargo is expected to arrive by April 15, 2023, based on your requested delivery date.",
    "next_steps": "To proceed with this quote, please click the 'Accept Quote' button. If you have any questions or need adjustments, please contact your account manager or reply to the quote email."
  },
  "customer_friendly_format": true,
  "language": "en"
}
```

## 6. 参考数据API

### 6.1 获取地点列表

```
GET /locations
```

请求头:
```
Authorization: Bearer {access_token}
```

查询参数:
```
page=1
per_page=10
country=China
is_port=true
search=Shanghai
```

响应:
```json
{
  "data": [
    {
      "id": 1,
      "name": "Shanghai Port",
      "country": "China",
      "city": "Shanghai",
      "is_port": true,
      "is_airport": false
    },
    // 更多地点...
  ],
  "meta": {
    "current_page": 1,
    "per_page": 10,
    "total_pages": 5,
    "total_count": 42
  }
}
```

### 6.2 获取运输方式列表

```
GET /transport-modes
```

请求头:
```
Authorization: Bearer {access_token}
```

响应:
```json
{
  "data": [
    {
      "id": 1,
      "name": "Air Freight",
      "description": "Transportation of goods by air",
      "is_active": true
    },
    {
      "id": 2,
      "name": "Sea Freight",
      "description": "Transportation of goods by sea",
      "is_active": true
    },
    // 更多运输方式...
  ]
}
```

### 6.3 获取货物类型列表

```
GET /cargo-types
```

请求头:
```
Authorization: Bearer {access_token}
```

响应:
```json
{
  "data": [
    {
      "id": 1,
      "name": "General Cargo",
      "description": "Standard non-hazardous goods",
      "is_dangerous": false,
      "requires_temperature_control": false
    },
    // 更多货物类型...
  ]
}
```

## 7. 错误处理

### 7.1 错误响应格式

所有API错误都将返回一个一致的JSON格式：

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested resource was not found",
    "details": "Quote with ID 999 does not exist",
    "status": 404
  }
}
```

### 7.2 常见错误代码

| 错误代码 | HTTP状态码 | 描述 |
|---------|-----------|------|
| UNAUTHORIZED | 401 | 未提供有效的认证凭据 |
| FORBIDDEN | 403 | 没有权限访问请求的资源 |
| RESOURCE_NOT_FOUND | 404 | 请求的资源不存在 |
| VALIDATION_ERROR | 422 | 请求数据验证失败 |
| INTERNAL_SERVER_ERROR | 500 | 服务器内部错误 |

### 7.3 验证错误示例

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "The request data is invalid",
    "details": {
      "weight": ["Weight is required"],
      "volume": ["Volume must be a positive number"]
    },
    "status": 422
  }
}
```

## 8. 分页和过滤

### 8.1 分页

列表API支持标准分页参数：

- `page`: 页码，从1开始
- `per_page`: 每页记录数，默认10，最大100

### 8.2 排序

列表API支持排序参数：

- `sort`: 排序字段，如`created_at`
- `order`: 排序方向，`asc`或`desc`

### 8.3 过滤

列表API支持特定于资源的过滤参数，例如：

- 报价列表：`status`, `customer_id`, `date_range`
- 规则列表：`category_id`, `is_active`

## 9. 速率限制

API实施速率限制以防止滥用：

- 标准用户：每分钟60个请求
- 高级用户：每分钟120个请求

超过限制时，API将返回429状态码和以下响应：

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": "Please try again after 30 seconds",
    "status": 429
  }
}
```

响应头将包含速率限制信息：

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1614556800
```

## 10. 版本控制

### 10.1 API版本策略

- 版本包含在URL路径中：`/v1/quotes`
- 主要版本（如v1到v2）可能包含不兼容的更改
- 次要更新在同一主要版本内向后兼容

### 10.2 弃用政策

- API版本弃用将提前90天通知
- 弃用的API版本将继续运行至少6个月
- 弃用通知将通过电子邮件和API响应头发送 