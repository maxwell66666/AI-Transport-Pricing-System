"""
LLM服务模块

本模块提供了与大型语言模型(LLM)交互的服务，用于智能报价分析、调整和解释。
支持多种LLM提供商，包括OpenAI、Azure OpenAI和本地部署的模型。
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import time
from datetime import datetime

from sqlalchemy.orm import Session

from src.config import settings
from src.db.models import LLMPrompt, LLMInteraction
from src.db.repository import llm_prompt_repository, llm_interaction_repository

# 根据配置选择LLM提供商
if settings.LLM_PROVIDER.lower() == "openai":
    import openai
    openai.api_key = settings.OPENAI_API_KEY
elif settings.LLM_PROVIDER.lower() == "azure":
    import openai
    openai.api_type = "azure"
    openai.api_key = settings.AZURE_OPENAI_API_KEY
    openai.api_base = settings.AZURE_OPENAI_API_BASE
    openai.api_version = settings.AZURE_OPENAI_API_VERSION
else:
    # 默认使用本地模型
    pass

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM服务类
    
    提供与大型语言模型交互的服务，用于智能报价分析、调整和解释。
    """
    
    def __init__(self, db: Session):
        """
        初始化LLM服务
        
        Args:
            db: 数据库会话
        """
        self.db = db
        self.model = settings.LLM_MODEL
        self.provider = settings.LLM_PROVIDER.lower()
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """
        加载提示模板
        
        从数据库加载所有活跃的LLM提示模板
        """
        self.prompts = {}
        prompts = self.db.query(LLMPrompt).filter(LLMPrompt.is_active == True).all()
        for prompt in prompts:
            self.prompts[prompt.purpose] = prompt.prompt_template
        
        logger.info(f"Loaded {len(self.prompts)} LLM prompt templates")
    
    def _call_openai(self, prompt: str, max_tokens: int = 1000) -> Tuple[str, Dict[str, Any]]:
        """
        调用OpenAI API
        
        Args:
            prompt: 提示文本
            max_tokens: 最大生成令牌数
            
        Returns:
            生成的文本和API响应元数据
        """
        start_time = time.time()
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的物流和运输定价专家，擅长分析运输报价并提供建议。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.2,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            processing_time = time.time() - start_time
            content = response.choices[0].message.content
            metadata = {
                "model": self.model,
                "tokens_used": response.usage.total_tokens,
                "processing_time": processing_time
            }
            
            return content, metadata
        
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
    
    def _call_local_model(self, prompt: str, max_tokens: int = 1000) -> Tuple[str, Dict[str, Any]]:
        """
        调用本地模型
        
        Args:
            prompt: 提示文本
            max_tokens: 最大生成令牌数
            
        Returns:
            生成的文本和API响应元数据
        """
        # 这里应该实现与本地模型的交互
        # 由于本地模型的实现可能各不相同，这里只提供一个示例框架
        start_time = time.time()
        try:
            # 假设本地模型通过某种方式提供服务
            # 实际实现需要根据具体的本地模型部署方式来定
            content = "这是本地模型的示例响应。在实际实现中，这里应该返回真实的模型输出。"
            processing_time = time.time() - start_time
            
            metadata = {
                "model": "local-model",
                "tokens_used": len(prompt.split()) + len(content.split()),  # 简单估计
                "processing_time": processing_time
            }
            
            return content, metadata
        
        except Exception as e:
            logger.error(f"Error calling local model: {str(e)}")
            raise
    
    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> Tuple[str, Dict[str, Any]]:
        """
        调用语言模型
        
        根据配置选择合适的LLM提供商
        
        Args:
            prompt: 提示文本
            max_tokens: 最大生成令牌数
            
        Returns:
            生成的文本和API响应元数据
        """
        if self.provider == "openai" or self.provider == "azure":
            return self._call_openai(prompt, max_tokens)
        else:
            return self._call_local_model(prompt, max_tokens)
    
    def _log_interaction(
        self, 
        prompt: str, 
        response: str, 
        metadata: Dict[str, Any], 
        quote_id: Optional[int] = None,
        prompt_id: Optional[int] = None
    ) -> None:
        """
        记录LLM交互
        
        Args:
            prompt: 提示文本
            response: 模型响应
            metadata: 响应元数据
            quote_id: 相关报价ID
            prompt_id: 相关提示模板ID
        """
        try:
            interaction = LLMInteraction(
                quote_id=quote_id,
                prompt_id=prompt_id,
                input_data={"prompt": prompt},
                prompt_used=prompt,
                response=response,
                model_used=metadata.get("model", self.model),
                tokens_used=metadata.get("tokens_used", 0),
                processing_time=metadata.get("processing_time", 0),
                created_at=datetime.utcnow()
            )
            
            self.db.add(interaction)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error logging LLM interaction: {str(e)}")
    
    def analyze_quote(
        self,
        pricing_input: Dict[str, Any],
        current_price: float,
        price_details: List[Dict[str, Any]],
        similar_quotes: List[Dict[str, Any]] = []
    ) -> Optional[Dict[str, Any]]:
        """
        分析报价并提供建议
        
        Args:
            pricing_input: 定价输入数据
            current_price: 当前计算的价格
            price_details: 价格明细
            similar_quotes: 相似报价列表
            
        Returns:
            分析结果，包括分析文本和调整后的价格
        """
        try:
            # 获取报价分析提示模板
            prompt_template = self.prompts.get("quote_adjustment")
            if not prompt_template:
                logger.warning("Quote adjustment prompt template not found")
                prompt_template = """
                你是一个专业的物流和运输定价专家，请分析以下运输报价并提供建议：
                
                运输信息：
                - 起始地：{{origin}}
                - 目的地：{{destination}}
                - 运输方式：{{transport_mode}}
                - 货物类型：{{cargo_type}}
                - 重量：{{weight}} 公斤
                - 体积：{{volume}} 立方米
                - 距离：{{distance}} 公里
                - 运输时间：{{transit_time}} 天
                - 特殊要求：{{special_requirements}}
                
                当前报价：{{current_price}} {{currency}}
                
                价格明细：
                {{price_details}}
                
                相似历史报价：
                {{similar_quotes}}
                
                请提供：
                1. 对当前报价的分析，包括各项费用的合理性
                2. 基于市场情况和历史数据的价格调整建议
                3. 调整后的建议总价
                4. 任何其他可能影响定价的因素
                
                请以JSON格式返回你的分析结果，包含以下字段：
                {
                    "analysis": "你的详细分析",
                    "adjusted_price": 调整后的价格（数字）,
                    "adjustment_reason": "价格调整的原因"
                }
                """
            
            # 准备提示参数
            origin = f"{pricing_input['origin_location']['name']}, {pricing_input['origin_location']['city']}, {pricing_input['origin_location']['country']}"
            destination = f"{pricing_input['destination_location']['name']}, {pricing_input['destination_location']['city']}, {pricing_input['destination_location']['country']}"
            
            price_details_text = "\n".join([
                f"- {detail['description']}: {detail['amount']} {detail['currency']}"
                for detail in price_details
            ])
            
            similar_quotes_text = ""
            if similar_quotes:
                similar_quotes_text = "历史相似报价：\n"
                for i, quote in enumerate(similar_quotes, 1):
                    similar_quotes_text += f"报价 {i}:\n"
                    similar_quotes_text += f"- 日期: {quote['quote_date']}\n"
                    similar_quotes_text += f"- 重量: {quote['weight']} 公斤\n"
                    similar_quotes_text += f"- 体积: {quote['volume']} 立方米\n"
                    similar_quotes_text += f"- 总价: {quote['total_price']} {quote['currency']}\n"
                    similar_quotes_text += "- 明细:\n"
                    for detail in quote['details']:
                        similar_quotes_text += f"  * {detail['description']}: {detail['amount']} {detail['currency']}\n"
                    similar_quotes_text += "\n"
            
            # 填充提示模板
            prompt = prompt_template.replace("{{origin}}", origin)
            prompt = prompt.replace("{{destination}}", destination)
            prompt = prompt.replace("{{transport_mode}}", pricing_input['transport_mode']['name'])
            prompt = prompt.replace("{{cargo_type}}", pricing_input['cargo_type']['name'])
            prompt = prompt.replace("{{weight}}", str(pricing_input['weight']))
            prompt = prompt.replace("{{volume}}", str(pricing_input['volume']))
            prompt = prompt.replace("{{distance}}", str(pricing_input['distance']))
            prompt = prompt.replace("{{transit_time}}", str(pricing_input['transit_time']))
            prompt = prompt.replace("{{special_requirements}}", pricing_input.get('special_requirements', '无'))
            prompt = prompt.replace("{{current_price}}", str(current_price))
            prompt = prompt.replace("{{currency}}", "USD")
            prompt = prompt.replace("{{price_details}}", price_details_text)
            prompt = prompt.replace("{{similar_quotes}}", similar_quotes_text)
            
            # 调用LLM
            response, metadata = self._call_llm(prompt)
            
            # 记录交互
            self._log_interaction(prompt, response, metadata)
            
            # 解析响应
            try:
                # 尝试直接解析JSON
                result = json.loads(response)
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试从文本中提取JSON部分
                try:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        result = json.loads(json_str)
                    else:
                        # 如果无法提取JSON，创建一个基本结果
                        result = {
                            "analysis": response,
                            "adjusted_price": current_price  # 保持原价
                        }
                except Exception as e:
                    logger.error(f"Error parsing LLM response as JSON: {str(e)}")
                    result = {
                        "analysis": response,
                        "adjusted_price": current_price  # 保持原价
                    }
            
            return result
        
        except Exception as e:
            logger.error(f"Error in quote analysis: {str(e)}")
            return None
    
    def explain_quote(self, quote_data: Dict[str, Any]) -> Optional[str]:
        """
        生成报价解释
        
        Args:
            quote_data: 报价数据
            
        Returns:
            报价解释文本
        """
        try:
            # 获取报价解释提示模板
            prompt_template = self.prompts.get("quote_explanation")
            if not prompt_template:
                logger.warning("Quote explanation prompt template not found")
                prompt_template = """
                你是一个专业的物流客户服务代表，请为以下运输报价生成一个清晰、友好的解释：
                
                运输信息：
                - 起始地：{{origin}}
                - 目的地：{{destination}}
                - 运输方式：{{transport_mode}}
                - 货物类型：{{cargo_type}}
                - 重量：{{weight}} 公斤
                - 体积：{{volume}} 立方米
                - 距离：{{distance}} 公里
                - 预计运输时间：{{transit_time}} 天
                - 预计交付日期：{{delivery_date}}
                - 特殊要求：{{special_requirements}}
                
                报价明细：
                {{price_details}}
                
                总价：{{total_price}} {{currency}}
                
                请提供：
                1. 友好的介绍
                2. 对每个费用组成部分的清晰解释
                3. 对任何特殊费用或折扣的解释
                4. 关于预计交付时间的信息
                5. 客户的后续步骤
                
                请使用专业但易于理解的语言，避免使用过多的行业术语。
                """
            
            # 准备提示参数
            origin = f"{quote_data['origin_location']['name']}, {quote_data['origin_location']['city']}, {quote_data['origin_location']['country']}"
            destination = f"{quote_data['destination_location']['name']}, {quote_data['destination_location']['city']}, {quote_data['destination_location']['country']}"
            
            price_details_text = "\n".join([
                f"- {detail['description']}: {detail['amount']} {detail['currency']}"
                for detail in quote_data['details']
            ])
            
            # 填充提示模板
            prompt = prompt_template.replace("{{origin}}", origin)
            prompt = prompt.replace("{{destination}}", destination)
            prompt = prompt.replace("{{transport_mode}}", quote_data['transport_mode']['name'])
            prompt = prompt.replace("{{cargo_type}}", quote_data['cargo_type']['name'])
            prompt = prompt.replace("{{weight}}", str(quote_data['weight']))
            prompt = prompt.replace("{{volume}}", str(quote_data['volume']))
            prompt = prompt.replace("{{distance}}", str(quote_data['distance']))
            prompt = prompt.replace("{{transit_time}}", str(quote_data['transit_time']))
            prompt = prompt.replace("{{delivery_date}}", quote_data['expected_delivery_date'])
            prompt = prompt.replace("{{special_requirements}}", quote_data.get('special_requirements', '无'))
            prompt = prompt.replace("{{price_details}}", price_details_text)
            prompt = prompt.replace("{{total_price}}", str(quote_data['total_price']))
            prompt = prompt.replace("{{currency}}", quote_data['currency'])
            
            # 调用LLM
            response, metadata = self._call_llm(prompt)
            
            # 记录交互
            self._log_interaction(prompt, response, metadata, quote_id=quote_data.get('id'))
            
            return response
        
        except Exception as e:
            logger.error(f"Error in quote explanation: {str(e)}")
            return None
    
    def extract_rules_from_data(self, historical_data: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        从历史数据中提取定价规则
        
        Args:
            historical_data: 历史报价数据
            
        Returns:
            提取的规则列表
        """
        try:
            # 获取规则提取提示模板
            prompt_template = self.prompts.get("rule_extraction")
            if not prompt_template:
                logger.warning("Rule extraction prompt template not found")
                prompt_template = """
                你是一个物流定价分析师。请分析以下历史运输报价数据，提取可能的定价规则或模式：
                
                {{historical_data}}
                
                请提供：
                1. 识别出的定价模式或规则
                2. 每个规则的公式或逻辑
                3. 每个规则应适用的条件
                4. 每个规则的置信度
                5. 每个规则的建议优先级
                
                请以JSON格式返回你的分析结果，格式如下：
                {
                    "rules": [
                        {
                            "name": "规则名称",
                            "description": "规则描述",
                            "formula": "规则公式或逻辑",
                            "conditions": "适用条件",
                            "confidence": 置信度（0-1之间的数字）,
                            "priority": 优先级（数字，越小优先级越高）
                        },
                        ...
                    ]
                }
                """
            
            # 准备历史数据文本
            historical_data_text = json.dumps(historical_data, ensure_ascii=False, indent=2)
            
            # 填充提示模板
            prompt = prompt_template.replace("{{historical_data}}", historical_data_text)
            
            # 调用LLM
            response, metadata = self._call_llm(prompt, max_tokens=2000)
            
            # 记录交互
            self._log_interaction(prompt, response, metadata)
            
            # 解析响应
            try:
                # 尝试直接解析JSON
                result = json.loads(response)
                return result.get("rules", [])
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试从文本中提取JSON部分
                try:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        result = json.loads(json_str)
                        return result.get("rules", [])
                    else:
                        logger.error("Could not extract JSON from LLM response")
                        return None
                except Exception as e:
                    logger.error(f"Error parsing LLM response as JSON: {str(e)}")
                    return None
        
        except Exception as e:
            logger.error(f"Error in rule extraction: {str(e)}")
            return None 