from typing import Dict, List, Optional, Union
import pdfplumber
import email
import re
from pathlib import Path
from datetime import datetime
# 恢复langchain相关的导入
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class DataProcessor:
    """历史报价数据处理器"""
    
    def __init__(self, llm_api_key: Optional[str] = None):
        """
        初始化数据处理器
        
        Args:
            llm_api_key: OpenAI API密钥（可选）
        """
        # 恢复使用langchain相关的代码
        self.llm = OpenAI(api_key=llm_api_key) if llm_api_key else None
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # 定义提示模板
        self.extract_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            从以下文本中提取运输报价相关信息：
            
            {text}
            
            请以JSON格式返回以下信息：
            - 报价日期
            - 起始地
            - 目的地
            - 货物类型
            - 重量（千克）
            - 体积（立方米）
            - 报价金额（元）
            - 服务等级
            - 预计送达时间
            """
        )
        
    def process_pdf(self, file_path: Union[str, Path]) -> Dict:
        """
        处理PDF格式的报价文件
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            提取的结构化数据
        """
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
                
        return self._extract_quote_info(text)
    
    def process_email(self, email_content: str) -> Dict:
        """
        处理邮件格式的报价
        
        Args:
            email_content: 原始邮件内容
            
        Returns:
            提取的结构化数据
        """
        msg = email.message_from_string(email_content)
        text = ""
        
        # 提取邮件正文
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    text += part.get_payload(decode=True).decode() + "\n"
        else:
            text = msg.get_payload(decode=True).decode()
            
        return self._extract_quote_info(text)
    
    def _extract_quote_info(self, text: str) -> Dict:
        """
        从文本中提取报价信息
        
        Args:
            text: 待处理的文本
            
        Returns:
            提取的结构化数据
        """
        # 使用正则表达式进行初步信息提取
        info = {
            "quote_date": None,
            "origin": None,
            "destination": None,
            "goods_type": None,
            "weight": None,
            "volume": None,
            "price": None,
            "service_level": None,
            "estimated_delivery": None
        }
        
        # 尝试使用正则表达式匹配
        date_pattern = r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}'
        price_pattern = r'(?:价格|报价|费用|金额)[：:]\s*¥?\s*(\d+(?:\.\d{2})?)'
        weight_pattern = r'(?:重量)[：:]\s*(\d+(?:\.\d{2})?)\s*(?:kg|千克)'
        volume_pattern = r'(?:体积)[：:]\s*(\d+(?:\.\d{2})?)\s*(?:m3|立方米)'
        
        # 提取日期
        date_match = re.search(date_pattern, text)
        if date_match:
            info["quote_date"] = date_match.group()
            
        # 提取价格
        price_match = re.search(price_pattern, text)
        if price_match:
            info["price"] = float(price_match.group(1))
            
        # 提取重量
        weight_match = re.search(weight_pattern, text)
        if weight_match:
            info["weight"] = float(weight_match.group(1))
            
        # 提取体积
        volume_match = re.search(volume_pattern, text)
        if volume_match:
            info["volume"] = float(volume_match.group(1))
            
        # 如果有LLM支持，使用LLM提取更多信息
        if self.llm:
            try:
                chunks = self.text_splitter.split_text(text)
                chain = LLMChain(llm=self.llm, prompt=self.extract_prompt)
                
                for chunk in chunks:
                    result = chain.run(text=chunk)
                    # 解析LLM返回的JSON结果并更新info字典
                    llm_info = eval(result)
                    for key, value in llm_info.items():
                        if not info.get(key) and value:
                            info[key] = value
            except Exception as e:
                print(f"LLM处理出错: {str(e)}")
                
        return info
    
    def batch_process(self, file_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        批量处理多个文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            处理结果列表
        """
        results = []
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if path.suffix.lower() == '.pdf':
                    result = self.process_pdf(path)
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        result = self.process_email(f.read())
                results.append(result)
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
                continue
        return results 