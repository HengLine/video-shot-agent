# -*- coding: utf-8 -*-
"""
@FileName: shot_generator_agent.py
@Description: 分镜生成智能体
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from typing import Dict, Any, Optional

from hengline.prompts.prompts_manager import prompt_manager


class ShotGeneratorAgent:
    """分镜生成智能体"""

    def __init__(self, llm, config: Optional[Dict[str, Any]] = None):
        """
        初始化分镜生成智能体
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        self.shot_generation_template = prompt_manager.get_shot_generator_prompt()
        #############
        self.config = config or {}
