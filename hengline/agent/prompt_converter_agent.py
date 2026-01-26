"""
@FileName: prompt_converter_agent.py
@Description: 提示词转换智能体
@Author: HengLine
@Time: 2026/1/18 14:23
"""
from typing import Optional, Dict, Any


class PromptConverterAgent:
    """提示指令转换器"""
    def __init__(self, llm, config: Optional[Dict[str, Any]] = None):
        """
        初始化分镜生成智能体

        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        self.config = config or {}