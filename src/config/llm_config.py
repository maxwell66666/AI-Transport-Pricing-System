"""
LLM服务配置模块
"""
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    """LLM配置类"""
    model_name: str = Field(default="gpt-3.5-turbo", env="LLM_MODEL_NAME")
    api_key: str = Field(default="", env="OPENAI_API_KEY")
    temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    max_tokens: int = Field(default=2000, env="LLM_MAX_TOKENS")
    top_p: float = Field(default=0.9, env="LLM_TOP_P")
    frequency_penalty: float = Field(default=0.0, env="LLM_FREQUENCY_PENALTY")
    presence_penalty: float = Field(default=0.0, env="LLM_PRESENCE_PENALTY")

llm_config = LLMConfig() 