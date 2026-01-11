"""
@FileName: qwen_client.py
@Description: 
@Author: HengLine
@Time: 2026/1/10 23:13
"""

from langchain_community.chat_models import ChatTongyi  # Qwen via Tongyi
from langchain_core.language_models import BaseLanguageModel

from base_client import BaseClient
from hengline.client.client_config import AIConfig


class QwenClient(BaseClient):
    """Qwen LLM 客户端实现"""

    def __init__(
            self,
            config: AIConfig,
    ):
        self.config = config

    def llm_model(self) -> BaseLanguageModel:
        return ChatTongyi(
            model=self.config.model,
            model_kwargs=self._get_model_kwargs(),
            api_key=self.config.api_key,
            max_retries=3,
            streaming=False,
        )

    def _get_model_kwargs(self):
        """返回模型参数字典"""
        model_kwargs = {
            "temperature": self.config.temperature,
            # "top_p": config.top_p,
            # "presence_penalty": config.presence_penalty,
            # "frequency_penalty": config.frequency_penalty,
            "max_tokens": self.config.max_tokens,
        }
        return model_kwargs