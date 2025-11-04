"""
@FileName: embedding_client.py
@Description: 
@Author: HengLine
@Time: 2025/10/24 21:53
"""
from typing import Optional

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from config.config import get_embedding_config
from hengline.logger import debug, info, error
from utils.log_utils import print_log_exception

def get_embedding_client(
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
) -> BaseEmbedding:
    """
    获取嵌入模型实例

    Args:
        model_type: 模型类型，支持 "openai", "huggingface", "ollama"。如果为None，则从配置中读取
        model_name: 模型名称。如果为None，则从配置中读取
        **kwargs: 额外参数

    Returns:
        BaseEmbedding实例
    """
    try:
        # 从配置中获取嵌入模型设置
        embedding_config = get_embedding_config()
        debug(f"从配置中读取的embedding设置: {embedding_config}")

        # 如果没有指定model_type或model_name，则从配置中获取
        if model_type is None:
            model_type = embedding_config.get("provider", "openai")

        if model_name is None:
            model_name = embedding_config.get("model", "text-embedding-3-small")

        debug(f"获取嵌入模型: type={model_type}, name={model_name}")

        if model_type == "openai":
            # OpenAI嵌入模型
            # 确保kwargs是字典类型
            if not isinstance(kwargs, dict):
                kwargs = {}
            
            # 从配置中获取额外参数
            config_kwargs = {
                "base_url": embedding_config.get("base_url"),
                "api_key": embedding_config.get("api_key")
            }
            # 确保kwargs是字典
            if not isinstance(kwargs, dict):
                kwargs = {}
            # 合并配置参数和传入参数，传入参数优先级更高
            merged_kwargs = {**config_kwargs, **kwargs}
            # 移除None值
            merged_kwargs = {k: v for k, v in merged_kwargs.items() if v is not None}
            debug(f"OpenAI嵌入模型参数: {merged_kwargs}")

            return OpenAIEmbedding(
                model=model_name,
                **merged_kwargs
            )

        elif model_type == "huggingface":
            # HuggingFace嵌入模型
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            # 确保kwargs是字典类型
            if not isinstance(kwargs, dict):
                kwargs = {}
            
            # 从配置中获取额外参数
            config_kwargs = {
                # HuggingFace相关配置可以从embedding_config中获取
                "model_name": model_name,
                "token": embedding_config.get("token", 1024),
                "cache_folder": embedding_config.get("cache_folder", "../../data/cache"),
                "device": embedding_config.get("device", "cpu")
            }
            # 确保kwargs是字典
            if not isinstance(kwargs, dict):
                kwargs = {}
            # 合并配置参数和传入参数
            merged_kwargs = {**config_kwargs, **kwargs}
            merged_kwargs = {k: v for k, v in merged_kwargs.items() if v is not None}
            debug(f"HuggingFace嵌入模型参数: {merged_kwargs}")

            return HuggingFaceEmbedding(
                model_name=model_name,
                **merged_kwargs
            )

        elif model_type == "ollama":
            # Ollama本地嵌入模型
            from llama_index.embeddings.ollama import OllamaEmbedding
            # 确保kwargs是字典类型
            if not isinstance(kwargs, dict):
                kwargs = {}
            
            # 从配置中获取额外参数
            config_kwargs = {
                "base_url": embedding_config.get("base_url", "http://localhost:11434"),
                "request_timeout": embedding_config.get("timeout")
            }
            # 确保kwargs是字典
            if not isinstance(kwargs, dict):
                kwargs = {}
            # 合并配置参数和传入参数
            merged_kwargs = {**config_kwargs, **kwargs}
            merged_kwargs = {k: v for k, v in merged_kwargs.items() if v is not None}
            debug(f"Ollama嵌入模型参数: {merged_kwargs}")

            return OllamaEmbedding(
                model_name=model_name,
                **merged_kwargs
            )

        else:
            raise ValueError(f"不支持的嵌入模型类型: {model_type}")

    except Exception as e:
        print_log_exception()
        error(f"获取嵌入模型失败: {str(e)}, 将尝试使用默认的OpenAI嵌入模型")
        # 如果出错，尝试返回默认的OpenAI嵌入模型
        try:
            return OpenAIEmbedding(model="text-embedding-3-small")
        except Exception as default_error:
            error(f"默认模型初始化也失败: {str(default_error)}")
            raise RuntimeError("无法初始化任何嵌入模型")
