# -*- coding: utf-8 -*-
"""
@FileName: qwen_client.py
@Description: 通义千问模型客户端
@Author: HengLine
@Time: 2025/10/6
"""

import os
from typing import Dict, Any, Optional, Callable

from langchain_community.llms import Tongyi
from langchain_core.callbacks import CallbackManager

from hengline.client.base_client import BaseAIClient
from hengline.client.openai_client import OpenAIClient
from hengline.client.openai_compat import OpenAICompatibleWrapper, BaseOpenAIResponse
from hengline.logger import debug, error


class QwenClient(BaseAIClient):
    """
    通义千问模型客户端
    提供通义千问模型的访问和响应处理
    """

    # 通义千问特定配置
    PROVIDER_NAME = "qwen"
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "qwen-plus"
    API_KEY_ENV_VAR = "QWEN_API_KEY"

    @classmethod
    def _get_client_implementation(cls, api_key: str, base_url: str, config: Dict[str, Any]) -> OpenAICompatibleWrapper:
        """
        获取通义千问客户端实现
        
        Args:
            api_key: API密钥
            base_url: 基础URL
            config: 配置字典
            
        Returns:
            OpenAI兼容的客户端实例
        """
        # 创建completion处理函数，并传递config参数
        handler = cls.create_completion_handler(api_key, base_url, config)

        # 创建并返回OpenAI兼容的包装器
        return cls.create_openai_compatible_wrapper(handler)

    @classmethod
    def create_completion_handler(cls, api_key: str, base_url: str, config: Dict[str, Any] = None) -> Callable:
        """
        创建通义千问的completion处理函数
        
        Args:
            api_key: API密钥
            base_url: 基础URL
            
        Returns:
            completion处理函数
        """

        def qwen_completion_handler(model: str = None, messages: list = None,
                                    temperature: Optional[float] = None,
                                    max_tokens: Optional[int] = None,
                                    response_format: Optional[Dict] = None,
                                    **kwargs) -> BaseOpenAIResponse:
            """
            通义千问模型调用处理函数
            
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
                # 从配置中获取超时时间和重试次数
                timeout = config.get('timeout', 60)
                retry_count = config.get('retry_count', 3)

                # 构建通义千问API请求参数
                payload = cls._build_qwen_payload(model, messages, temperature, max_tokens, config)

                # 构建请求头
                headers = cls._build_qwen_headers(api_key)

                # 发送请求，包含超时参数和重试次数
                debug(f"向通义千问发送请求: model={model}, temperature={temperature}, timeout={timeout}s, retry_count={retry_count}")
                response = cls.make_request(base_url, headers, payload, timeout=timeout, retry_count=retry_count)

                # 解析响应
                response_data = response.json()

                # 转换为OpenAI格式
                content = cls.convert_response(response_data)

                # 创建并返回响应对象
                return cls.create_response_from_content(content)

            except Exception as e:
                error(f"通义千问调用失败: {str(e)}")
                raise

        return qwen_completion_handler

    @classmethod
    def _build_qwen_payload(cls, model: Optional[str], messages: list,
                            temperature: Optional[float] = None,
                            max_tokens: Optional[int] = None,
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        构建通义千问特定的请求参数
        
        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成字数
            config: 配置字典，包含默认参数
            
        Returns:
            通义千问请求参数字典
        """
        config = config or {}
        # 从配置中获取默认值
        default_model = config.get('default_model', cls.DEFAULT_MODEL)
        default_temperature = config.get('temperature', 0.1)
        default_max_tokens = config.get('max_tokens', 2000)

        return {
            "model": model or default_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else default_max_tokens,
            "stream": False
        }

    @classmethod
    def _build_qwen_headers(cls, api_key: str) -> Dict[str, str]:
        """
        构建通义千问特定的请求头
        
        Args:
            api_key: API密钥
            
        Returns:
            通义千问请求头字典
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    @staticmethod
    def convert_response(response: Any) -> str:
        """
        转换通义千问响应为文本内容，支持多种响应格式
        
        Args:
            response: 通义千问API响应数据（可能是字典、对象或其他格式）
            
        Returns:
            提取的文本内容
        """
        try:
            # 处理字典类型响应
            if isinstance(response, dict):
                # 路径1: OpenAI兼容格式 (choices[0].message.content)
                choices = response.get('choices', [])
                if choices and isinstance(choices, list):
                    first_choice = choices[0]
                    if isinstance(first_choice, dict) and 'message' in first_choice:
                        content = first_choice['message'].get('content', '')
                        if content:
                            return content
                    # 也处理对象类型的choice
                    elif hasattr(first_choice, 'message') and hasattr(first_choice.message, 'content'):
                        return first_choice.message.content

                # 路径2: DashScope标准格式 (output.text)
                output = response.get('output')
                if output and isinstance(output, dict):
                    # 直接文本输出
                    text = output.get('text', '')
                    if text:
                        return text

                    # 带choices的输出
                    choices = output.get('choices')
                    if choices and isinstance(choices, list):
                        first_choice = choices[0]
                        if isinstance(first_choice, dict) and 'message' in first_choice:
                            return first_choice['message'].get('content', '')

            # 处理对象类型响应
            elif hasattr(response, 'choices') and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, 'message') and hasattr(first_choice.message, 'content'):
                    return first_choice.message.content

            # 处理其他可能的格式
            elif isinstance(response, str):
                return response

            # 未知格式，记录日志
            error(f"通义千问响应格式无法识别: {type(response).__name__} - {str(response)[:200]}...")
            return ""

        except Exception as e:
            error(f"转换通义千问响应失败: {str(e)}")
            # 尝试返回响应的字符串表示作为最后的备选
            try:
                return str(response) if response else ""
            except:
                return ""

    @classmethod
    def get_langchain_llm(cls, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        获取通义千问的LangChain LLM实例，优先使用langchain_community中的Tongyi实现
        
        Args:
            config: 配置参数，包含model、temperature等
            
        Returns:
            LangChain的实例
        """
        config = config or {}

        # 获取API密钥和其他配置
        api_key = cls._get_api_key(config)
        model = config.get('model', cls.DEFAULT_MODEL)
        temperature = config.get('temperature', 0.1)
        response_format = config.get('response_format', 'json_object')

        # 检查是否有API密钥可用
        has_api_key = api_key or 'DASHSCOPE_API_KEY' in os.environ
        
        if not has_api_key:
            # 在开发模式下，可以静默跳过，不尝试创建LLM实例
            debug(f"没有找到通义千问API密钥，跳过创建Tongyi实例")
            return None

        # 优先尝试使用langchain_community中的Tongyi实现
        try:
            debug(f"尝试使用langchain_community中的Tongyi实现通义千问，模型: {model}")

            # 构建Tongyi实例参数
            llm_params = {
                'model_name': model,
                'temperature': temperature,
                "response_format": {"type": response_format},
                'callbacks': CallbackManager([])
            }

            # 添加API密钥（通义千问使用dashscope_api_key参数名）
            dashscope_api_key = api_key or os.environ.get('DASHSCOPE_API_KEY')
            llm_params['dashscope_api_key'] = dashscope_api_key

            # 添加其他可选参数
            if config.get('max_tokens'):
                llm_params['max_tokens'] = config.get('max_tokens')

            # 创建并返回Tongyi实例
            llm = Tongyi(**llm_params)
            debug(f"成功创建Tongyi实例，模型: {model}")
            return llm

        except ImportError as import_e:
            error(f"导入Tongyi失败: {str(import_e)}")
        except Exception as tongyi_e:
            error(f"创建Tongyi实例失败: {str(tongyi_e)}")

        # 回退到使用OpenAI兼容方式 - 仅在有API密钥时尝试
        if has_api_key:
            try:
                debug("Tongyi实现失败，回退到OpenAI兼容方式")

                # 构建兼容的配置
                openai_config = {
                    'model': model,
                    'temperature': temperature,
                    'max_tokens': config.get('max_tokens', 2000),
                    'timeout': config.get('timeout', 60),
                    'api_key': api_key or os.environ.get('DASHSCOPE_API_KEY'),
                    'base_url': config.get('base_url', cls.DEFAULT_BASE_URL)
                }

                # 使用OpenAIClient获取LLM实例
                llm = OpenAIClient.get_langchain_llm(openai_config)
                if llm:
                    debug(f"成功使用OpenAIClient获取LLM实例，模型: {model}")
                    return llm

            except ImportError as import_e:
                error(f"导入OpenAIClient失败: {str(import_e)}")
            except Exception as openai_e:
                error(f"使用OpenAIClient获取LLM实例失败: {str(openai_e)}")

        # 没有API密钥或所有尝试都失败时，返回None而不报错
        debug("无法创建通义千问的LLM实例（缺少API密钥或其他原因）")
        return None
