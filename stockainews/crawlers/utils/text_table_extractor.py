import re
import json
from typing import List, Dict, Any, Optional
from stockainews.core.logger import setup_logger

logger = setup_logger(__name__)

class TextTableExtractor:
    """
    从 EnhancedDOMSerializer 生成的结构化文本中提取表格数据
    专门处理多表格混杂、需要特征识别的场景
    """

    @staticmethod
    def extract_data(
        text: str, 
        required_columns: List[str],
        block_name: str = "表格"
    ) -> List[Dict[str, Any]]:
        """
        从文本中识别并提取目标表格数据
        
        Args:
            text: 序列化后的文本内容
            required_columns: 必须包含的列名关键词（用于识别目标表格），如 ['公告标题', '公告日期']
            block_name: 表格块的前缀名称，默认为"表格"
            
        Returns:
            提取出的结构化数据列表。
        """
        if not text:
            return []

        # 1. 定位数据区域
        # 截取 "表格中的链接（重要，用于数据提取）：" 之后的内容
        start_marker = "表格中的链接（重要，用于数据提取）："
        content_to_parse = text
        if start_marker in text:
            content_to_parse = text.split(start_marker)[1]
        
        # 2. 分割各个表格块
        table_blocks = re.split(rf'{block_name}\d+（共\d+行数据）：', content_to_parse)
        
        target_data = []

        for block in table_blocks:
            if not block.strip():
                continue
                
            # 3. 特征识别：检查该表格块是否包含所有必须的列名
            is_target = True
            for col in required_columns:
                if f"{col}:" not in block:
                    is_target = False
                    break
            
            if not is_target:
                continue
                
            logger.debug(f"Found target table matching columns: {required_columns}")
            
            # 4. 解析行数据
            rows = re.split(r'\n\s*行\d+:', block)
            
            for row_text in rows:
                if not row_text.strip():
                    continue
                
                item = {}
                lines = row_text.split('\n')
                current_link_parsing = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line.startswith('链接:'):
                        current_link_parsing = True
                        continue
                        
                    if current_link_parsing:
                        link_match = re.search(r'->\s*(\S+)', line)
                        if link_match:
                            item['detail_url'] = link_match.group(1).strip()
                            current_link_parsing = False 
                    else:
                        if line.startswith('-'):
                            continue
                            
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            k, v = parts[0].strip(), parts[1].strip()
                            if k and v:
                                item[k] = v
                
                if item:
                    target_data.append(item)
            
            if target_data:
                break
            
        return target_data

    @staticmethod
    async def extract_with_llm(
        text: str,
        description: str,
        llm_wrapper: Any,
        prompt_template_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        使用 LLM 识别特征，然后提取表格数据
        
        Args:
            text: 序列化文本或HTML内容
            description: 目标表格描述（如"财务公告列表"）
            llm_wrapper: LLM适配器实例（DoubaoAdapter或LLMClientWrapper）
            prompt_template_path: Prompt 模板路径（已废弃，使用统一的PromptManager）
        """
        if not text:
            return []
            
        try:
            # 1. 准备 Prompt（使用统一的PromptManager）
            from stockainews.prompts import PromptManager
            
            content_snippet = text[:8000]
            if "表格中的链接（重要，用于数据提取）：" in text:
                content_snippet = text.split("表格中的链接（重要，用于数据提取）：")[1][:8000]
            
            # 使用统一的PromptManager加载提示词
            prompt_manager = PromptManager()
            try:
                template = prompt_manager.get("table_extraction")
                prompt = template.format(description=description, content_snippet=content_snippet)
            except (ValueError, KeyError):
                # 如果模板不存在，使用默认模板
                logger.warning("table_extraction模板不存在，使用默认模板")
                template = """
                任务：从文本中找到符合描述"{description}"的表格，返回其特征列名。
                文本：{content_snippet}
                返回JSON：{{"required_columns": ["列1", "列2"]}}
                """
                prompt = template.format(description=description, content_snippet=content_snippet)
            
            # 2. 调用 LLM
            # 支持两种类型的LLM适配器：
            # 1. LLMClientWrapper (旧接口): 有 client 和 _get_model_name() 方法
            # 2. DoubaoAdapter (新接口): 有 _async_client 和 model 属性，以及 generate() 方法
            from langchain_core.messages import HumanMessage
            
            result_text = None
            
            # 检查是否是 DoubaoAdapter (新接口)
            if hasattr(llm_wrapper, '_async_client') and hasattr(llm_wrapper, 'model'):
                # 使用 DoubaoAdapter 的异步接口
                try:
                    from langchain_core.messages import HumanMessage
                    messages = [HumanMessage(content=prompt)]
                    # DoubaoAdapter.generate() 支持 response_format 参数
                    chat_result = await llm_wrapper.generate(
                        messages=messages,
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                    # ChatResult.generations 是一个列表，每个元素是 ChatGeneration 列表
                    # ChatGeneration 有 message 属性，message.content 是文本内容
                    if chat_result.generations and len(chat_result.generations) > 0:
                        generation = chat_result.generations[0]
                        # ChatGeneration可能是单个对象或列表
                        if isinstance(generation, list) and len(generation) > 0:
                            result_text = generation[0].text
                        elif hasattr(generation, 'text'):
                            result_text = generation.text
                        elif hasattr(generation, 'message') and hasattr(generation.message, 'content'):
                            result_text = generation.message.content
                        else:
                            logger.error(f"Unexpected generation format: {type(generation)}, generation={generation}")
                            return []
                    else:
                        logger.error("No generations in ChatResult")
                        return []
                except Exception as e:
                    logger.error(f"DoubaoAdapter call failed: {e}", exc_info=True)
                    return []
            
            # 检查是否是 LLMClientWrapper (旧接口)
            elif hasattr(llm_wrapper, 'client') and hasattr(llm_wrapper, '_get_model_name'):
                if not llm_wrapper.client:
                    logger.error("LLM client not initialized")
                    return []
                
                import asyncio
                from functools import partial
                
                loop = asyncio.get_running_loop()
                model_name = llm_wrapper._get_model_name()
                
                # 在线程池中执行同步 API 调用，避免阻塞
                func = partial(
                    llm_wrapper.client.chat.completions.create,
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                response = await loop.run_in_executor(None, func)
                result_text = response.choices[0].message.content
            else:
                logger.error(f"Unsupported LLM wrapper type: {type(llm_wrapper)}")
                return []
            
            if not result_text:
                logger.error("LLM returned empty response")
                return []
            
            # 3. 解析结果
            try:
                result = json.loads(result_text)
                required_columns = result.get("required_columns", [])
                reasoning = result.get("reasoning", "")
                
                logger.info(f"LLM identified columns for '{description}': {required_columns} (Reason: {reasoning})")
                
                if not required_columns:
                    return []
                
                # 4. 执行提取
                return TextTableExtractor.extract_data(text, required_columns)
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response: {result_text}")
                return []
                
        except Exception as e:
            logger.error(f"Error in extract_with_llm: {e}")
            return []
