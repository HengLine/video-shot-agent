"""
@FileName: llm_script_parser.py
@Description: 
@Author: HengLine
@Time: 2026/1/9 21:23
"""
import time
from typing import Any

from hengline.client.client_factory import llm_chat_complete


class BaseAgent:

    def _call_llm_chat_with_retry(self, llm, system_prompt: str, user_prompt, max_retries: int = 3) -> Any | None:
        """调用LLM，支持重试"""
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

    def _call_llm_with_retry(self, llm, prompt: str, max_retries: int = 3) -> Any | None:
        """调用LLM，支持重试"""
        for attempt in range(max_retries):
            try:
                return llm.invoke(prompt)

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"LLM调用失败: {e}")
                time.sleep(1)
