"""
@FileName: openai_client.py
@Description: 
@Author: HengLine
@Time: 2026/1/10 23:15
"""
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

from base_client import BaseClient
from hengline.client.client_config import AIConfig


class OpenAIClient(BaseClient):
    """OpenAI 客户端实现"""

    def __init__(
            self,
            config: AIConfig,
    ):
        self.config = config
        self.base_url = "https://api.openai.com/v1"

    def llm_model(self) -> BaseLanguageModel:
        return ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            api_key=self.config.api_key,
            base_url=self.base_url,
            max_retries=3,
            max_tokens=self.config.max_tokens,
        )
