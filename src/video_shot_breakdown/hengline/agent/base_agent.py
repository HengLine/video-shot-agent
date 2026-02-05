"""
@FileName: llm_script_parser.py
@Description: 
@Author: HengLine
@Github: https://github.com/HengLine/video-shot-agent
@Time: 2026/1/9 21:23
"""
import time
from abc import ABC
from typing import Any, Dict

from video_shot_breakdown.hengline.client.client_factory import llm_chat_complete
from video_shot_breakdown.hengline.prompts.prompts_manager import prompt_manager
from video_shot_breakdown.hengline.tools.json_parser_tool import parse_json_response


class BaseAgent(ABC):

    def _get_prompt_template(self, key_name) -> str:
        """创建LLM提示词模板"""
        return prompt_manager.get_name_prompt(key_name)

    def _parse_llm_response(self, ai_response: str) -> Dict[str, Any]:
        """ 转换LLM响应，必要时需要重写该方法 """
        return parse_json_response(ai_response)

    def _call_llm_parse_with_retry(self, llm, system_prompt: str, user_prompt, max_retries: int = 2) -> Dict[str, Any] | None:
        """
            调用LLM，返回转换后的对象（支持重试）
            返回 dict
        """
        return self._parse_llm_response(self._call_llm_chat_with_retry(llm, system_prompt, user_prompt, max_retries))

    def _call_llm_chat_with_retry(self, llm, system_prompt: str, user_prompt, max_retries: int = 2) -> str | None:
        """
            调用LLM，直接返回json字符串（支持重试）
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        for attempt in range(max_retries):
            try:
                return llm_chat_complete(llm, messages)

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"LLM调用失败: {e}")
                time.sleep(1)

    def _call_llm_with_retry(self, llm, prompt: str, max_retries: int = 2) -> Any | None:
        """调用LLM，支持重试"""
        for attempt in range(max_retries):
            try:
                return llm.invoke(prompt)

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"LLM调用失败: {e}")
                time.sleep(1)
