"""LLM service module for quote analysis and optimization."""
from typing import Dict, Any, List
import openai
from src.config.llm_config import llm_config

class LLMService:
    """Service for handling LLM-based quote analysis and optimization."""
    
    def __init__(self):
        """Initialize LLM service with configuration."""
        config = llm_config.get_openai_config()
        openai.api_key = config["api_key"]
        openai.api_base = config["api_base"]
        self.model = config["model"]
    
    async def analyze_quote(self, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a transportation quote using LLM."""
        prompt = self._create_analysis_prompt(quote_data)
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的运输定价分析师。"},
                    {"role": "user", "content": prompt}
                ]
            )
            return {
                "analysis": response.choices[0].message.content,
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    async def optimize_quote(self, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a transportation quote using LLM."""
        prompt = self._create_optimization_prompt(quote_data)
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的运输定价优化专家。"},
                    {"role": "user", "content": prompt}
                ]
            )
            return {
                "optimization": response.choices[0].message.content,
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def _create_analysis_prompt(self, quote_data: Dict[str, Any]) -> str:
        """Create prompt for quote analysis."""
        return f"""请分析以下运输报价:
运输类型: {quote_data.get('transport_type', 'N/A')}
起点: {quote_data.get('origin', 'N/A')}
终点: {quote_data.get('destination', 'N/A')}
货物类型: {quote_data.get('cargo_type', 'N/A')}
重量: {quote_data.get('weight', 'N/A')}
体积: {quote_data.get('volume', 'N/A')}
报价金额: {quote_data.get('price', 'N/A')}

请提供:
1. 价格合理性分析
2. 影响因素分析
3. 市场对比分析
4. 潜在风险分析
"""

    def _create_optimization_prompt(self, quote_data: Dict[str, Any]) -> str:
        """Create prompt for quote optimization."""
        return f"""请为以下运输报价提供优化建议:
运输类型: {quote_data.get('transport_type', 'N/A')}
起点: {quote_data.get('origin', 'N/A')}
终点: {quote_data.get('destination', 'N/A')}
货物类型: {quote_data.get('cargo_type', 'N/A')}
重量: {quote_data.get('weight', 'N/A')}
体积: {quote_data.get('volume', 'N/A')}
报价金额: {quote_data.get('price', 'N/A')}

请提供:
1. 成本优化建议
2. 路线优化建议
3. 时间优化建议
4. 服务优化建议
""" 