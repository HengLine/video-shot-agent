"""
@FileName: base_client.py
@Description: 
@Author: HengLine
@Time: 2026/1/10 23:12
"""
from abc import ABC, abstractmethod

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
