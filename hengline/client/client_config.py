"""
@FileName: client_type.py
@Description: 
@Author: HengLine
@Time: 2026/1/11 16:31
"""
from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional

from pydantic import SecretStr


@unique
class ClientType(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"

def get_client_type(client_type_str):
    for client_type in ClientType:
        if client_type.value == client_type_str.lower():
            return client_type

    raise ValueError(f"Invalid client_type: {client_type_str}")


@dataclass
class AIConfig:
    """AI配置"""
    model: str = "gpt-4"  # 或 "claude-3", "deepseek-chat"
    # base_url: str = ""  # 用于本地部署或特定API端点
    api_key: Optional[SecretStr] = None
    temperature: float = 0.1
    max_tokens: int = 5000
    json_mode: bool = True  # 强制JSON输出
    timeout: int = 60  # 请求超时时间，单位秒
    enable_cot: bool = True  # 启用思维链推理
    include_visual_hints: bool = True  # 包含视觉生成提示

    # 专业领域知识注入
    cinematic_knowledge: bool = True
    pacing_principles: bool = True
