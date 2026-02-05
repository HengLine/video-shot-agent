"""
@FileName: openai_client.py
@Description: 
@Author: HengLine
@Github: https://github.com/HengLine/video-shot-agent
@Time: 2026/1/10 23:15
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from video_shot_breakdown.hengline.client.base_client import BaseClient
from video_shot_breakdown.hengline.client.client_config import AIConfig


class OpenAIClient(BaseClient):
    """OpenAI 客户端实现"""

    def __init__(self, config: AIConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com/v1"

    def llm_model(self) -> BaseLanguageModel:
        return ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            api_key=self.config.api_key,
            base_url=self.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            max_tokens=self.config.max_tokens,
        )

    def llm_embed(self) -> Embeddings:
        return OpenAIEmbeddings(
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.base_url,
            dimensions=self.config.dimensions,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
