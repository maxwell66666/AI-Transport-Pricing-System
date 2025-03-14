from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import tempfile
import os
from pathlib import Path
# 恢复导入
from ..utils.data_processor import DataProcessor

router = APIRouter()
# 恢复DataProcessor的实例化
data_processor = DataProcessor()

@router.post("/quotes/import")
async def import_quotes(files: List[UploadFile] = File(...)):
    """
    导入历史报价文件
    """
    # 恢复原始代码
    results = []
    
    for file in files:
        try:
            # 创建临时文件
            suffix = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                # 写入上传的文件内容
                content = await file.read()
                temp_file.write(content)
                temp_file.flush()
                
                # 处理文件
                if suffix.lower() == '.pdf':
                    result = data_processor.process_pdf(temp_file.name)
                else:
                    # 对于邮件文件，先读取内容
                    with open(temp_file.name, 'r', encoding='utf-8') as f:
                        result = data_processor.process_email(f.read())
                        
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "data": result
                })
                
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
        finally:
            # 清理临时文件
            if 'temp_file' in locals():
                os.unlink(temp_file.name)
    
    return {"results": results} 