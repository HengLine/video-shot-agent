# -*- coding: utf-8 -*-
"""
@FileName: base_client.py
@Description: AI模型客户端基类
@Author: HengLine
@Time: 2025/10/6
"""

import os
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable

from hengline.client.openai_compat import OpenAICompatibleWrapper, BaseOpenAIResponse
from hengline.logger import info, error, warning


class BaseAIClient(ABC):
    """
    AI模型客户端基类
    提供所有AI客户端共享的基础功能
    """

    # 默认配置
    DEFAULT_BASE_URL = ""
    DEFAULT_MODEL = ""
    PROVIDER_NAME = ""
    API_KEY_ENV_VAR = ""

    @classmethod
    def create_client(cls, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        创建客户端实例的通用入口
        子类可以覆盖此方法或使用_get_client_implementation方法
        
        Args:
            config: 客户端配置参数
            
        Returns:
            配置好的客户端实例
        """
        config = config or {}

        # 获取配置信息
        api_key = cls._get_api_key(config)
        base_url = cls._get_base_url(config)

        # 验证配置
        cls._validate_config(api_key)

        # 获取具体实现
        return cls._get_client_implementation(api_key, base_url, config)

    @classmethod
    def _get_api_key(cls, config: Dict[str, Any]) -> str:
        """
        从环境变量或配置中获取API密钥
        
        Args:
            config: 配置字典
            
        Returns:
            API密钥
        """
        return os.environ.get(cls.API_KEY_ENV_VAR, config.get('api_key', ''))

    @classmethod
    def _get_base_url(cls, config: Dict[str, Any]) -> str:
        """
        获取基础URL
        
        Args:
            config: 配置字典
            
        Returns:
            基础URL
        """
        return config.get('base_url', cls.DEFAULT_BASE_URL)

    @classmethod
    def _validate_config(cls, api_key: str) -> None:
        """
        验证配置是否有效
        
        Args:
            api_key: API密钥
            
        Raises:
            ValueError: 当配置无效时
        """
        # 对于Ollama等本地模型，通常不需要API密钥
        if cls.PROVIDER_NAME != "ollama" and not api_key and cls.API_KEY_ENV_VAR:
            # 在开发环境中，如果使用OpenAI模型但没有API密钥，记录警告而不是抛出错误
            # 这样可以允许在无API密钥的情况下进行基本功能测试
            import os
            if os.environ.get("DEV_MODE") == "true":
                warning(f"警告：未配置{cls.PROVIDER_NAME}的API密钥，但在开发模式下允许继续")
            else:
                error(f"未配置{cls.PROVIDER_NAME}的API密钥")
                raise ValueError(f"未配置{cls.PROVIDER_NAME}的API密钥")

    @classmethod
    @abstractmethod
    def _get_client_implementation(cls, api_key: str, base_url: str, config: Dict[str, Any]) -> Any:
        """
        获取具体的客户端实现
        子类必须实现此方法
        
        Args:
            api_key: API密钥
            base_url: 基础URL
            config: 配置字典
            
        Returns:
            客户端实例
        """
        pass
        
    @classmethod
    def make_request(cls, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 60, retry_count: int = 3) -> requests.Response:
        """
        发送HTTP请求到AI服务提供商，支持重试机制
        
        Args:
            url: 请求URL
            headers: 请求头
            payload: 请求参数
            timeout: 请求超时时间（秒）
            retry_count: 请求失败后重试次数
            
        Returns:
            requests.Response对象
            
        Raises:
            requests.HTTPError: 当请求在重试后仍然失败时
        """
        import time
        import random
        
        last_exception = None
        for attempt in range(retry_count):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()  # 检查HTTP错误
                return response
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < retry_count - 1:
                    # 指数退避策略，增加随机抖动避免雪崩效应
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    info(f"请求失败 (尝试 {attempt + 1}/{retry_count}): {str(e)}，{delay:.2f}秒后重试")
                    time.sleep(delay)
                else:
                    error(f"请求在 {retry_count} 次尝试后失败: {str(e)}")
        
        # 如果所有重试都失败，抛出最后一个异常
        raise last_exception

    @classmethod
    def create_completion_handler(cls, api_key: str, base_url: str, config: Dict[str, Any] = None) -> Callable:
        """
        创建completion处理函数
        子类可以覆盖此方法提供特定的处理逻辑
        
        Args:
            api_key: API密钥
            base_url: 基础URL
            config: 配置字典
            
        Returns:
            completion处理函数
        """

        def handler(model: str = None, messages: list = None,
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None,
                    response_format: Optional[Dict] = None,
                    **kwargs) -> Any:
            """
            默认的completion处理函数
            子类应该覆盖此方法提供具体实现
            
            Args:
                model: 模型名称
                messages: 消息列表
                temperature: 温度参数
                max_tokens: 最大生成字数
                response_format: 响应格式要求
                **kwargs: 其他参数
                
            Returns:
                模型响应
            """
            raise NotImplementedError("子类必须实现具体的completion处理逻辑")

        return handler

    @classmethod
    def create_openai_compatible_wrapper(cls, handler: Callable) -> OpenAICompatibleWrapper:
        """
        创建OpenAI兼容的包装器
        
        Args:
            handler: completion处理函数
            
        Returns:
            OpenAI兼容的客户端包装器
        """
        wrapper = OpenAICompatibleWrapper(handler)
        info(f"成功创建{cls.PROVIDER_NAME}客户端（OpenAI兼容格式）")
        return wrapper

    @classmethod
    def create_response_from_content(cls, content: Any) -> BaseOpenAIResponse:
        """从内容创建一个符合OpenAI格式的响应对象
        
        Args:
            content: 响应内容，可以是字符串或对象
            
        Returns:
            符合OpenAI格式的响应对象
        """
        # 确保content是字符串
        if isinstance(content, str):
            content_str = content
        elif hasattr(content, 'content'):
            content_str = str(content.content) if content.content else ""
        elif hasattr(content, '__str__'):
            content_str = str(content)
        else:
            content_str = ""
        
        # 直接传递content字符串，符合BaseOpenAIResponse的构造函数
        response = BaseOpenAIResponse(content_str)
        # 确保response本身有content属性，方便直接访问
        response.content = content_str
        return response

    @classmethod
    def _build_common_payload(cls, model: str, messages: list, temperature: Optional[float] = None,
                              max_tokens: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        构建通用的请求参数
        
        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成字数
            **kwargs: 其他参数
            
        Returns:
            请求参数字典
        """
        payload = {
            "model": model or cls.DEFAULT_MODEL,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.1,
            "max_tokens": max_tokens if max_tokens is not None else 2000,
        }

        # 添加其他参数
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        return payload

    @classmethod
    def _build_common_headers(cls, api_key: str) -> Dict[str, str]:
        """
        构建通用的请求头
        
        Args:
            api_key: API密钥
            
        Returns:
            请求头字典
        """
        headers = {
            "Content-Type": "application/json",
        }

        # 如果有API密钥，添加到请求头
        if api_key and cls.API_KEY_ENV_VAR:
            headers["Authorization"] = f"Bearer {api_key}"

        return headers

    @staticmethod
    def convert_response(response: Any) -> str:
        """
        转换模型响应为文本内容
        子类应该覆盖此方法提供特定的转换逻辑
        
        Args:
            response: 模型响应对象
            
        Returns:
            提取的文本内容
        """
        raise NotImplementedError("子类必须实现具体的响应转换逻辑")

    @classmethod
    def get_default_model(cls) -> str:
        """
        获取默认模型名称
        
        Returns:
            默认模型名称
        """
        return cls.DEFAULT_MODEL
    
    @classmethod
    def get_langchain_llm(cls, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        获取LangChain兼容的LLM实例
        子类应该根据需要覆盖此方法以提供特定的LLM实现
        
        Args:
            config: 配置参数，包含model、temperature等
            
        Returns:
            LangChain兼容的LLM实例，如果不支持则返回None
        """
        raise NotImplementedError("子类必须实现获取LangChain LLM实例的方法")
