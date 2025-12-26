# -*- coding: utf-8 -*-
"""
OpenAI 模型客户端实现
"""

import os
from typing import Dict, Optional, Any, Callable

import httpx
from openai import OpenAI
from langchain_community.llms import OpenAI
from langchain_core.callbacks import CallbackManager

from hengline.client.base_client import BaseAIClient
from hengline.client.openai_compat import OpenAICompatibleWrapper, BaseOpenAIResponse
from hengline.logger import debug, info, error


class OpenAIClient(BaseAIClient):
    """
    OpenAI 客户端类
    提供 OpenAI 模型的访问功能
    """

    # OpenAI特定配置
    PROVIDER_NAME = "openai"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4"
    API_KEY_ENV_VAR = "OPENAI_API_KEY"

    @classmethod
    def _get_client_implementation(cls, api_key: str, base_url: str, config: Dict[str, Any]) -> OpenAICompatibleWrapper:
        """
        获取OpenAI客户端实现
        
        Args:
            api_key: API密钥
            base_url: 基础URL
            config: 配置字典
            
        Returns:
            OpenAI兼容的客户端实例
        """
        # 创建completion处理函数，确保传递config参数
        handler = cls.create_completion_handler(api_key, base_url, config)

        # 创建并返回OpenAI兼容的包装器
        return cls.create_openai_compatible_wrapper(handler)

    @classmethod
    def create_completion_handler(cls, api_key: str, base_url: str, config: Dict[str, Any] = None) -> Callable:
        """
        创建OpenAI的completion处理函数
        
        Args:
            api_key: API密钥
            base_url: 基础URL
            config: 配置字典
            
        Returns:
            completion处理函数
        """
        try:
            # 从配置中获取超时时间和重试次数
            timeout = config.get('timeout', 60)
            retry_count = config.get('retry_count', 3)

            # 创建 HTTP 客户端，设置适当的超时
            http_client = httpx.Client(
                timeout=timeout,
                follow_redirects=True
            )

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=http_client
            )

            def openai_completion_handler(model: str = None, messages: list = None,
                                          temperature: Optional[float] = None,
                                          max_tokens: Optional[int] = None,
                                          response_format: Optional[Dict] = None,
                                          **kwargs) -> BaseOpenAIResponse:
                """
                OpenAI模型调用处理函数
                
                Args:
                    model: 模型名称
                    messages: 消息列表
                    temperature: 温度参数
                    max_tokens: 最大生成字数
                    response_format: 响应格式要求
                    **kwargs: 其他参数
                    
                Returns:
                    BaseOpenAIResponse对象
                """
                try:
                    # 从配置中获取默认参数
                    default_temperature = config.get('temperature', 0.1)
                    default_max_tokens = config.get('max_tokens', 2000)
                    default_model = config.get('default_model', cls.DEFAULT_MODEL)

                    # 构建请求参数
                    payload = {
                        "model": model or default_model,
                        "messages": messages,
                        "temperature": temperature if temperature is not None else default_temperature,
                        "max_tokens": max_tokens if max_tokens is not None else default_max_tokens,
                    }

                    # 添加响应格式参数（如果提供）
                    if response_format:
                        payload["response_format"] = response_format

                    # 添加其他可选参数
                    for key, value in kwargs.items():
                        if value is not None:
                            payload[key] = value

                    # 发送请求
                    debug(f"向OpenAI发送请求: model={model}, temperature={temperature}")
                    response = client.chat.completions.create(**payload)

                    # 转换为OpenAI格式（实际上已经是兼容的）
                    content = cls.convert_response(response)

                    # 创建并返回响应对象
                    return cls.create_response_from_content(content)

                except Exception as e:
                    error(f"OpenAI调用失败: {str(e)}")
                    raise

            return openai_completion_handler

        except Exception as e:
            error(f"创建 OpenAI 客户端失败: {str(e)}")
            raise

    @staticmethod
    def convert_response(response: Any) -> str:
        """
        转换OpenAI响应为文本内容
        
        Args:
            response: OpenAI API响应对象
            
        Returns:
            提取的文本内容
        """
        try:
            # 处理 OpenAI SDK 返回的对象
            if response and hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    return choice.message.content

            # 处理字典格式的响应
            elif isinstance(response, dict):
                content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                if content:
                    return content

            error(f"OpenAI响应格式异常: {response}")
            return ""

        except Exception as e:
            error(f"转换OpenAI响应失败: {str(e)}")
            return ""

    @classmethod
    def get_langchain_llm(cls, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        获取OpenAI的LangChain LLM实例，优先使用langchain_community的OpenAI，失败则回退到langchain_openai的ChatOpenAI
        
        Args:
            config: 配置参数，包含model、temperature等
            
        Returns:
            LangChain的实例
        """
        config = config or {}

        # 获取API密钥和其他配置
        api_key = cls._get_api_key(config)
        model = config.get('model', cls.DEFAULT_MODEL)
        temperature = config.get('temperature', 0.7)
        max_tokens = config.get('max_tokens', 2000)
        timeout = config.get('timeout', 60)
        base_url = config.get('base_url')

        # 优先尝试使用langchain_community的OpenAI
        try:
            debug(f"优先尝试使用langchain_community的OpenAI，模型: {model}")

            # 构建参数字典
            llm_params = {
                'model': model,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'timeout': timeout,
                'callbacks': CallbackManager([])
            }

            # 根据可用性添加可选参数
            if api_key:
                llm_params['openai_api_key'] = api_key

            if base_url:
                llm_params['base_url'] = base_url

            # 添加其他可能的参数
            if config.get('top_p') is not None:
                llm_params['top_p'] = config.get('top_p')

            if config.get('frequency_penalty') is not None:
                llm_params['frequency_penalty'] = config.get('frequency_penalty')

            if config.get('presence_penalty') is not None:
                llm_params['presence_penalty'] = config.get('presence_penalty')

            # 创建OpenAI实例
            llm = OpenAI(**llm_params)

            # 验证实例是否成功创建
            if llm:
                debug(f"成功创建langchain_community的OpenAI实例，模型: {model}")
                return llm

        except Exception as e:
            error(f"创建langchain_community的OpenAI实例失败: {str(e)}")

        # langchain_community的OpenAI失败后，尝试使用langchain_openai的ChatOpenAI
        try:
            from langchain_openai import ChatOpenAI
            debug(f"langchain_community的OpenAI失败，回退到langchain_openai的ChatOpenAI，模型: {model}")

            # 构建参数字典
            chat_params = {
                'model': model,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'timeout': timeout
            }

            # 根据可用性添加可选参数
            if api_key:
                chat_params['openai_api_key'] = api_key

            if base_url:
                chat_params['openai_api_base'] = base_url

            # 添加其他可能的参数
            if config.get('top_p') is not None:
                chat_params['top_p'] = config.get('top_p')

            if config.get('frequency_penalty') is not None:
                chat_params['frequency_penalty'] = config.get('frequency_penalty')

            if config.get('presence_penalty') is not None:
                chat_params['presence_penalty'] = config.get('presence_penalty')

            # 创建ChatOpenAI实例
            llm = ChatOpenAI(**chat_params)

            # 验证实例是否成功创建
            if llm:
                debug(f"成功创建langchain_openai的ChatOpenAI实例，模型: {model}")
                return llm

        except ImportError as import_e:
            error(f"导入langchain_openai失败: {str(import_e)}")
        except Exception as e:
            error(f"创建langchain_openai的ChatOpenAI实例失败: {str(e)}")

        # 如果主要方法失败，尝试使用最小参数集的回退方案
        try:
            debug("使用最小参数集进行回退尝试")

            try:
                # 优先尝试最小参数的OpenAI
                minimal_params = {
                    'model': model,
                    'temperature': temperature
                }

                if api_key:
                    minimal_params['openai_api_key'] = api_key

                llm = OpenAI(**minimal_params)
                if llm:
                    debug(f"成功使用最小参数创建OpenAI实例")
                    return llm
            except Exception:
                # 如果OpenAI失败，尝试ChatOpenAI
                try:
                    from langchain_openai import ChatOpenAI
                    minimal_chat_params = {
                        'model': model,
                        'temperature': temperature
                    }

                    if api_key:
                        minimal_chat_params['openai_api_key'] = api_key

                    llm = ChatOpenAI(**minimal_chat_params)
                    if llm:
                        debug(f"成功使用最小参数创建ChatOpenAI实例")
                        return llm
                except Exception:
                    pass

        except Exception as fallback_e:
            error(f"回退尝试也失败: {str(fallback_e)}")

        error(f"无法创建OpenAI相关的LangChain LLM实例，模型: {model}")
        return None

    @classmethod
    def get_default_model(cls) -> str:
        """
        获取默认模型名称
        
        Returns:
            默认模型名称
        """
        # 允许通过环境变量覆盖默认模型
        return os.environ.get('OPENAI_DEFAULT_MODEL', cls.DEFAULT_MODEL)


def get_openai_client(config: Optional[Dict[str, Any]] = None) -> OpenAICompatibleWrapper:
    """
    获取OpenAI客户端实例
    
    Args:
        config: 配置参数
        
    Returns:
        OpenAI兼容的客户端实例
    """
    return OpenAIClient.create_client(config)


# 缓存的客户端实例
_cached_client = None
_cached_config = None


def get_cached_openai_client(config: Optional[Dict[str, Any]] = None) -> OpenAICompatibleWrapper:
    """
    获取缓存的OpenAI客户端实例
    
    Args:
        config: 配置参数
        
    Returns:
        OpenAI兼容的客户端实例
    """
    global _cached_client, _cached_config

    # 如果没有缓存或配置发生变化，则创建新实例
    if _cached_client is None or _cached_config != config:
        _cached_client = OpenAIClient.create_client(config)
        _cached_config = config

    return _cached_client


def create_openai_client_with_retry(max_retries: int = 3,
                                    retry_delay: float = 2.0,
                                    config: Optional[Dict[str, Any]] = None) -> OpenAICompatibleWrapper:
    """
    创建带重试机制的OpenAI客户端
    
    Args:
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
        config: 配置参数
        
    Returns:
        OpenAI兼容的客户端实例
    """
    import time

    for attempt in range(max_retries):
        try:
            client = OpenAIClient.create_client(config)
            info("成功创建OpenAI客户端")
            return client
        except Exception as e:
            if attempt == max_retries - 1:
                error(f"创建OpenAI客户端失败，已达到最大重试次数: {str(e)}")
                raise
            error(f"创建OpenAI客户端失败，将在{retry_delay}秒后重试: {str(e)}")
            time.sleep(retry_delay)
            retry_delay *= 1.5  # 指数退避

    raise RuntimeError("创建OpenAI客户端失败")


def analyze_with_openai(self, audio_path, user_query):
    """使用OpenAI Whisper API进行语音识别"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )

        return transcript
    except ImportError:
        print("请安装openai: pip install openai")
        return None
    except Exception as e:
        print(f"OpenAI API错误: {e}")
        return None
