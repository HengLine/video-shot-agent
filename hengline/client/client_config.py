"""
@FileName: client_type.py
@Description: 
@Author: HengLine
@Time: 2026/1/11 16:31
"""
from dataclasses import dataclass, Field
from enum import Enum, unique
from typing import Optional

from pydantic import SecretStr


@unique
class ClientType(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


@dataclass
class AIConfig:
    """AI配置"""
    model: str = "gpt-4"  # 或 "claude-3", "deepseek-chat"
    # base_url: str = ""  # 用于本地部署或特定API端点
    api_key: Optional[SecretStr] = None
    temperature: float = 0.2
    max_tokens: int = 4000
    system_prompt: str = ""
    json_mode: bool = True  # 强制JSON输出
    timeout: int = 60  # 请求超时时间，单位秒


