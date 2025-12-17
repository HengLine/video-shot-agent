# -*- coding: utf-8 -*-
"""
@FileName: script_parser_agent.py
@Description: LLM 剧本解析功能，通过LLM 将中文剧本转换为结构化动作序列
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from typing import Dict, Any, Optional

from hengline.agent.script_parser.script_base_parser import ScriptParser
from hengline.logger import debug
from hengline.prompts.prompts_manager import prompt_manager
from hengline.tools.json_parser_tool import parse_json_response


class ScriptLLMParser(ScriptParser):
    """优化版剧本解析智能体"""

    def __init__(self, llm, script_intel):
        """
        初始化剧本解析智能体
        
        Args:
            llm: 语言模型实例（推荐GPT-4o）
            script_intel: 嵌入模型实例
        """
        self.llm = llm
        self.script_intel = script_intel

    def parse_script_to_json(self, script_text: str, result: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        使用LLM直接解析剧本
        
        Args:
            script_text: 原始剧本文本
            
        Returns:
            解析结果或None
        """
        parser_prompt = prompt_manager.get_script_parser_prompt()

        # 调用LLM进行直接解析
        llm_result = self.llm.invoke(parser_prompt.format(script_text=script_text))
        debug(f"LLM直接解析结果:\n {llm_result}")

        # 解析LLM响应
        parsed_result = parse_json_response(llm_result)

        # LLM解析结果通常已经比较完整，只需要少量优化
        if parsed_result and parsed_result.get("scenes"):
            debug("LLM解析成功，应用最小化优化")
            # 对LLM结果应用轻量级优化
            for scene in parsed_result["scenes"]:
                # 确保必要字段存在
                if "atmosphere" not in scene:
                    scene["atmosphere"] = self._infer_atmosphere(scene)
                if "characters" not in scene:
                    scene["characters"] = []
                # 对动作进行基本排序优化
                # if "actions" in scene:
                #     scene["actions"] = self._reorder_actions_for_logic(scene["actions"])
            return parsed_result
        return None
