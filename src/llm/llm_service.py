"""
大型语言模型服务模块

此模块提供大型语言模型服务，用于报价解释和优化。
"""

import logging
import json
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import requests
from pathlib import Path

from src.config import settings
from src.db.models import LLMPrompt, LLMInteraction, Quote
from src.db.repository import LLMPromptRepository, LLMInteractionRepository, QuoteRepository
from src.utils.exceptions import LLMServiceError
from sqlalchemy.orm import Session

# 配置日志
logger = logging.getLogger(__name__)


class LLMService:
    """大型语言模型服务类，提供报价解释和优化功能"""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """
        初始化LLM服务
        
        Args:
            model_name: 模型名称，如果为None，则使用配置中的默认模型
            api_key: API密钥，如果为None，则使用配置中的密钥
        """
        self.model_name = model_name or settings.LLM_DEFAULT_MODEL
        self.api_key = api_key or settings.LLM_API_KEY
        self.api_base_url = settings.LLM_API_BASE_URL
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.temperature = settings.LLM_TEMPERATURE
        self.prompt_templates = {}
        
        # 加载提示模板
        self._load_prompt_templates()
    
    def _load_prompt_templates(self) -> None:
        """加载提示模板"""
        try:
            # 从配置文件加载提示模板
            templates_file = Path(settings.LLM_TEMPLATES_FILE)
            if templates_file.exists():
                with open(templates_file, 'r', encoding='utf-8') as f:
                    self.prompt_templates = json.load(f)
                logger.info(f"已加载 {len(self.prompt_templates)} 个提示模板")
            else:
                # 使用默认模板
                self.prompt_templates = {
                    "quote_explanation": "请解释以下运输报价的详细信息，包括价格构成、影响因素和可能的优化建议：\n\n{quote_details}\n\n请提供详细的解释，使客户能够理解价格的构成和影响因素。",
                    "price_optimization": "请分析以下运输报价，并提供优化建议，以降低成本或提高服务质量：\n\n{quote_details}\n\n请提供具体的优化建议，包括可能的替代方案、时间调整或其他可以降低成本的因素。",
                    "comparison_analysis": "请比较以下两个运输报价，分析它们的差异、优缺点，并给出建议：\n\n报价1：{quote1_details}\n\n报价2：{quote2_details}\n\n请详细分析两个报价的差异，并根据客户需求给出建议。"
                }
                logger.info("使用默认提示模板")
        except Exception as e:
            logger.error(f"加载提示模板失败: {str(e)}")
            # 使用基本模板
            self.prompt_templates = {
                "quote_explanation": "请解释以下运输报价：\n\n{quote_details}"
            }
    
    def _format_prompt(self, template_name: str, **kwargs) -> str:
        """
        格式化提示模板
        
        Args:
            template_name: 模板名称
            **kwargs: 模板参数
            
        Returns:
            格式化后的提示
        """
        if template_name not in self.prompt_templates:
            raise LLMServiceError(f"提示模板 {template_name} 不存在")
        
        template = self.prompt_templates[template_name]
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise LLMServiceError(f"提示模板参数错误: {str(e)}")
    
    def _call_llm_api(self, prompt: str) -> Dict[str, Any]:
        """
        调用LLM API
        
        Args:
            prompt: 提示文本
            
        Returns:
            API响应
        """
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 准备请求数据
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "你是一个专业的物流和运输顾问，擅长解释运输报价和提供优化建议。"},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            # 发送请求
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30  # 30秒超时
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 记录响应
            logger.info(f"LLM API调用成功，处理时间: {processing_time:.2f}秒")
            
            return {
                "content": result["choices"][0]["message"]["content"],
                "model": result.get("model", self.model_name),
                "processing_time": processing_time,
                "token_usage": result.get("usage", {})
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API调用失败: {str(e)}")
            raise LLMServiceError(f"LLM API调用失败: {str(e)}")
        except (KeyError, IndexError) as e:
            logger.error(f"LLM API响应解析失败: {str(e)}")
            raise LLMServiceError(f"LLM API响应解析失败: {str(e)}")
        except Exception as e:
            logger.error(f"LLM API调用异常: {str(e)}")
            raise LLMServiceError(f"LLM API调用异常: {str(e)}")
    
    def _mock_llm_api(self, prompt: str) -> Dict[str, Any]:
        """
        模拟LLM API调用（用于开发和测试）
        
        Args:
            prompt: 提示文本
            
        Returns:
            模拟的API响应
        """
        # 记录开始时间
        start_time = time.time()
        
        # 根据提示类型生成不同的模拟响应
        if "解释" in prompt or "explanation" in prompt:
            content = """
## 运输报价解释

这份运输报价基于以下几个主要因素:

1. **基础运费**: 根据距离(500公里)和货物重量(1000千克)计算的基本费用，约占总价的60%。
2. **特殊要求附加费**: 
   - 冷藏服务: +15%
   - 快速配送: +10%
3. **燃油附加费**: 当前燃油价格导致5%的附加费。
4. **季节性因素**: 非旺季运输，获得了8%的折扣。

### 影响因素分析
- **距离**是最主要的价格因素，每增加100公里约增加10%的成本。
- **货物重量**是第二大因素，特别是对于公路运输。
- **冷藏要求**显著增加了能源消耗，导致较高的附加费。

### 优化建议
1. 考虑将部分非紧急货物改为普通配送，可节省10%费用。
2. 如果可以灵活安排发货时间，选择周二至周四发货可能获得更好的价格。
3. 增加单次运输量可获得批量折扣，建议合并小批量订单。
            """
        elif "优化" in prompt or "optimization" in prompt:
            content = """
## 价格优化建议

分析您的运输需求后，我建议以下优化措施:

1. **运输方式调整**:
   - 将当前的公路运输改为铁路运输，可节省约18%的成本，虽然会增加1-2天的运输时间。
   - 对于不超过200公里的短途运输，考虑使用本地配送服务，可节省12-15%。

2. **时间优化**:
   - 避开周一和周五的高峰期发货，选择周三发货可降低约5%的费用。
   - 提前7天预订可获得早鸟折扣，约为总价的3-5%。

3. **货物整合**:
   - 将多个小批量货物合并为一个大订单，可获得8-10%的批量折扣。
   - 优化包装以减少体积，可降低基于体积的计费。

4. **合同策略**:
   - 考虑与承运商签订长期合同，可获得稳定的折扣率，通常为5-8%。
   - 探索返程货物机会，利用回程车辆可节省高达25%的成本。

实施这些建议，预计可以降低总运输成本15-20%，同时保持服务质量。
            """
        elif "比较" in prompt or "comparison" in prompt:
            content = """
## 运输报价比较分析

### 报价1 vs 报价2

| 因素 | 报价1 | 报价2 | 差异 |
|------|-------|-------|------|
| 基本费用 | ¥3,500 | ¥3,200 | 报价2低8.6% |
| 运输时间 | 3天 | 5天 | 报价1快40% |
| 特殊服务 | 包含冷藏 | 不含冷藏 | 功能差异 |
| 保险覆盖 | 全额 | 基础保障 | 保障差异 |
| 燃油附加费 | 包含 | 单独计费 | 计费结构不同 |

### 优缺点分析

**报价1优势**:
- 运输时间更短，适合时效性要求高的货物
- 包含全面的保险保障，降低风险
- 一价全包，无隐藏费用

**报价1劣势**:
- 总体价格较高
- 灵活性较低，无法调整服务项目

**报价2优势**:
- 基础价格更低，成本效益高
- 服务可定制，按需付费
- 适合预算有限的情况

**报价2劣势**:
- 运输时间较长
- 基础保障有限，可能需要额外购买保险
- 最终总价可能因附加费增加

### 建议

根据您的需求:
- 如果货物对时效性要求高或需要冷藏，建议选择报价1
- 如果预算有限且可以接受较长的运输时间，报价2更经济
- 考虑与报价2供应商协商，增加必要的服务项目，可能仍比报价1更具成本效益
            """
        else:
            content = """
## 运输报价分析

感谢您提供的运输报价信息。基于分析，我有以下几点观察和建议:

1. **价格构成**:
   - 基础运费占总价约65%
   - 各种附加费用占约25%
   - 税费和其他费用占约10%

2. **市场定位**:
   - 此报价处于市场中等偏上水平
   - 对于所提供的服务内容，价格合理

3. **建议**:
   - 考虑调整发货时间以获得更优惠的价格
   - 评估是否所有特殊服务都必要
   - 探索长期合作可能带来的折扣机会

如需更详细的分析，请提供更多关于货物类型、运输距离和特殊要求的信息。
            """
        
        # 计算处理时间（模拟延迟）
        time.sleep(0.5)  # 模拟0.5秒的API延迟
        processing_time = time.time() - start_time
        
        return {
            "content": content.strip(),
            "model": f"{self.model_name}-mock",
            "processing_time": processing_time,
            "token_usage": {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(content) // 4,
                "total_tokens": (len(prompt) + len(content)) // 4
            }
        }
    
    def explain_quote(self, 
                     quote_id: int, 
                     db: Session,
                     use_mock: bool = False) -> Dict[str, Any]:
        """
        解释运输报价
        
        Args:
            quote_id: 报价ID
            db: 数据库会话
            use_mock: 是否使用模拟API（用于开发和测试）
            
        Returns:
            包含解释内容和元数据的字典
        """
        try:
            # 获取报价详情
            quote_repo = QuoteRepository(db)
            quote = quote_repo.get_by_id(quote_id)
            
            if not quote:
                raise LLMServiceError(f"报价 {quote_id} 不存在")
            
            # 准备报价详情
            quote_details = self._prepare_quote_details(quote)
            
            # 格式化提示
            prompt = self._format_prompt("quote_explanation", quote_details=quote_details)
            
            # 调用LLM API
            if use_mock or settings.ENVIRONMENT == "development":
                result = self._mock_llm_api(prompt)
            else:
                result = self._call_llm_api(prompt)
            
            # 记录交互
            interaction_id = self._record_interaction(
                db=db,
                prompt_type="quote_explanation",
                prompt=prompt,
                response=result["content"],
                quote_id=quote_id,
                model=result["model"],
                token_usage=result["token_usage"]
            )
            
            # 构建响应
            response = {
                "explanation": result["content"],
                "quote_id": quote_id,
                "model_used": result["model"],
                "processing_time_ms": round(result["processing_time"] * 1000, 2),
                "interaction_id": interaction_id
            }
            
            return response
            
        except LLMServiceError as e:
            logger.error(f"解释报价失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"解释报价失败: {str(e)}")
            raise LLMServiceError(f"解释报价失败: {str(e)}")
    
    def optimize_price(self, 
                      quote_id: int, 
                      db: Session,
                      use_mock: bool = False) -> Dict[str, Any]:
        """
        优化运输价格
        
        Args:
            quote_id: 报价ID
            db: 数据库会话
            use_mock: 是否使用模拟API（用于开发和测试）
            
        Returns:
            包含优化建议和元数据的字典
        """
        try:
            # 获取报价详情
            quote_repo = QuoteRepository(db)
            quote = quote_repo.get_by_id(quote_id)
            
            if not quote:
                raise LLMServiceError(f"报价 {quote_id} 不存在")
            
            # 准备报价详情
            quote_details = self._prepare_quote_details(quote)
            
            # 格式化提示
            prompt = self._format_prompt("price_optimization", quote_details=quote_details)
            
            # 调用LLM API
            if use_mock or settings.ENVIRONMENT == "development":
                result = self._mock_llm_api(prompt)
            else:
                result = self._call_llm_api(prompt)
            
            # 记录交互
            interaction_id = self._record_interaction(
                db=db,
                prompt_type="price_optimization",
                prompt=prompt,
                response=result["content"],
                quote_id=quote_id,
                model=result["model"],
                token_usage=result["token_usage"]
            )
            
            # 构建响应
            response = {
                "optimization_suggestions": result["content"],
                "quote_id": quote_id,
                "model_used": result["model"],
                "processing_time_ms": round(result["processing_time"] * 1000, 2),
                "interaction_id": interaction_id
            }
            
            return response
            
        except LLMServiceError as e:
            logger.error(f"优化价格失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"优化价格失败: {str(e)}")
            raise LLMServiceError(f"优化价格失败: {str(e)}")
    
    def compare_quotes(self, 
                      quote_id1: int, 
                      quote_id2: int, 
                      db: Session,
                      use_mock: bool = False) -> Dict[str, Any]:
        """
        比较两个运输报价
        
        Args:
            quote_id1: 第一个报价ID
            quote_id2: 第二个报价ID
            db: 数据库会话
            use_mock: 是否使用模拟API（用于开发和测试）
            
        Returns:
            包含比较分析和元数据的字典
        """
        try:
            # 获取报价详情
            quote_repo = QuoteRepository(db)
            quote1 = quote_repo.get_by_id(quote_id1)
            quote2 = quote_repo.get_by_id(quote_id2)
            
            if not quote1:
                raise LLMServiceError(f"报价 {quote_id1} 不存在")
            if not quote2:
                raise LLMServiceError(f"报价 {quote_id2} 不存在")
            
            # 准备报价详情
            quote1_details = self._prepare_quote_details(quote1)
            quote2_details = self._prepare_quote_details(quote2)
            
            # 格式化提示
            prompt = self._format_prompt(
                "comparison_analysis", 
                quote1_details=quote1_details,
                quote2_details=quote2_details
            )
            
            # 调用LLM API
            if use_mock or settings.ENVIRONMENT == "development":
                result = self._mock_llm_api(prompt)
            else:
                result = self._call_llm_api(prompt)
            
            # 记录交互
            interaction_id = self._record_interaction(
                db=db,
                prompt_type="comparison_analysis",
                prompt=prompt,
                response=result["content"],
                quote_id=quote_id1,  # 使用第一个报价ID作为主要关联
                related_quote_id=quote_id2,
                model=result["model"],
                token_usage=result["token_usage"]
            )
            
            # 构建响应
            response = {
                "comparison_analysis": result["content"],
                "quote_ids": [quote_id1, quote_id2],
                "model_used": result["model"],
                "processing_time_ms": round(result["processing_time"] * 1000, 2),
                "interaction_id": interaction_id
            }
            
            return response
            
        except LLMServiceError as e:
            logger.error(f"比较报价失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"比较报价失败: {str(e)}")
            raise LLMServiceError(f"比较报价失败: {str(e)}")
    
    def _prepare_quote_details(self, quote: Quote) -> str:
        """
        准备报价详情
        
        Args:
            quote: 报价对象
            
        Returns:
            格式化的报价详情字符串
        """
        # 获取关联对象
        origin = quote.origin.name if quote.origin else "未知起点"
        destination = quote.destination.name if quote.destination else "未知终点"
        transport_mode = quote.transport_mode.name if quote.transport_mode else "未知运输方式"
        cargo_type = quote.cargo_type.name if quote.cargo_type else "未知货物类型"
        
        # 构建详情字符串
        details = f"""报价ID: {quote.id}
客户: {quote.user.username if quote.user else '未知客户'}
创建时间: {quote.created_at.strftime('%Y-%m-%d %H:%M:%S')}
状态: {quote.status}

起点: {origin}
终点: {destination}
距离: {quote.distance} 公里
运输方式: {transport_mode}
货物类型: {cargo_type}
重量: {quote.weight} 千克
体积: {quote.volume} 立方米

总价: {quote.total_price} {quote.currency}
预计运输时间: {quote.transit_time} 小时
预计送达日期: {quote.expected_delivery_date.strftime('%Y-%m-%d') if quote.expected_delivery_date else '未知'}

特殊要求: {', '.join(quote.special_requirements) if quote.special_requirements else '无'}
"""

        # 添加报价明细
        if quote.details:
            details += "\n报价明细:\n"
            for i, detail in enumerate(quote.details, 1):
                details += f"{i}. {detail.description}: {detail.amount} {quote.currency}\n"
        
        return details
    
    def _record_interaction(self, 
                           db: Session,
                           prompt_type: str,
                           prompt: str,
                           response: str,
                           quote_id: int,
                           model: str,
                           token_usage: Dict[str, int],
                           related_quote_id: int = None) -> int:
        """
        记录LLM交互
        
        Args:
            db: 数据库会话
            prompt_type: 提示类型
            prompt: 提示文本
            response: 响应文本
            quote_id: 报价ID
            model: 模型名称
            token_usage: 令牌使用情况
            related_quote_id: 相关报价ID（用于比较分析）
            
        Returns:
            交互记录ID
        """
        try:
            # 获取或创建提示记录
            prompt_repo = LLMPromptRepository(db)
            prompt_record = prompt_repo.get_by_type(prompt_type)
            
            if not prompt_record:
                prompt_record = prompt_repo.create({
                    "type": prompt_type,
                    "template": self.prompt_templates.get(prompt_type, ""),
                    "description": f"用于{prompt_type}的提示模板",
                    "created_at": datetime.now()
                })
            
            # 创建交互记录
            interaction_repo = LLMInteractionRepository(db)
            interaction = interaction_repo.create({
                "prompt_id": prompt_record.id,
                "quote_id": quote_id,
                "related_quote_id": related_quote_id,
                "prompt_text": prompt,
                "response_text": response,
                "model": model,
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "completion_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
                "created_at": datetime.now()
            })
            
            return interaction.id
            
        except Exception as e:
            logger.error(f"记录LLM交互失败: {str(e)}")
            # 不抛出异常，避免影响主要功能
            return None 