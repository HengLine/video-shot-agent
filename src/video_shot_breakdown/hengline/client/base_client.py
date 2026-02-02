"""
@FileName: base_client.py
@Description: 
@Author: HengLine
@Time: 2026/1/10 23:12
"""
from abc import ABC, abstractmethod

import aiohttp
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel


class BaseClient(ABC):

    @abstractmethod
    def llm_model(self) -> BaseLanguageModel:
        """返回 LangChain 兼容的 LLM 实例"""
        pass

    @abstractmethod
    def llm_embed(self) -> Embeddings:
        """返回文本的 embedding 向量"""
        pass

    def _get_model_kwargs(self):
        """返回模型参数字典"""
        pass

    def check_llm(self) -> bool:
        """ 检查 LLM 服务是否可用 """
        try:
            llm = self.llm_model()
            llm.invoke("Hello, world!")
            return True
        except Exception as e:
            print(f"LLM check failed: {e}")
            return False

    async def _check_llm_provider(self, base_url: str, api_key: str):
        """检查提供商"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"{base_url}/models",
                    headers=headers,
                    timeout=20
            ) as response:
                if response.status != 200:
                    raise ConnectionError(f"OpenAI API returned status {response.status}")

                data = await response.json()
                if "data" not in data:
                    raise ValueError("Invalid OpenAI API response")