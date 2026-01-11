"""
@FileName: ollama_client.py
@Description: 
@Author: HengLine
@Time: 2026/1/10 23:16
"""
from langchain_core.language_models import BaseLanguageModel
from langchain_ollama import ChatOllama

from base_client import BaseClient
from hengline.client.client_config import AIConfig


class OllamaClient(BaseClient):
    """Ollama 客户端实现"""

    def __init__(
            self,
            config: AIConfig
    ):
        self.config = config
        self.base_url = "http://localhost:11434"  # Ollama 默认本地地址

    def llm_model(self) -> BaseLanguageModel:
        return ChatOllama(
            model=self.config.model,
            temperature=self.config.temperature,
            base_url=self.base_url,
        )
