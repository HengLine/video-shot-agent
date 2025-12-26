# -*- coding: utf-8 -*-
"""
@FileName: deepseek_client.py
@Description: DeepSeek模型客户端模块
@Author: HengLine
@Time: 2025/10/6
"""
from typing import Dict, Any, Optional, Callable

from hengline.client.base_client import BaseAIClient
from hengline.client.openai_client import OpenAIClient
from hengline.client.openai_compat import OpenAICompatibleWrapper, BaseOpenAIResponse
from hengline.logger import debug, error


class DeepSeekClient(BaseAIClient):
    """DeepSeek模型客户端类"""

    # DeepSeek特定配置
    PROVIDER_NAME = "deepseek"
    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-chat"
    API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"

    @classmethod
    def _get_client_implementation(cls, api_key: str, base_url: str, config: Dict[str, Any]) -> OpenAICompatibleWrapper:
        """
        获取DeepSeek客户端实现
        
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
    def create_completion_handler(cls, api_key: str, base_url: str, config: Dict[str, Any]) -> Callable:
        """
        创建DeepSeek的completion处理函数
        
        Args:
            api_key: API密钥
            base_url: 基础URL
            
        Returns:
            completion处理函数
        """

        def deepseek_completion_handler(model: str = None, messages: list = None,
                                        temperature: Optional[float] = None,
                                        max_tokens: Optional[int] = None,
                                        response_format: Optional[Dict] = None,
                                        **kwargs) -> BaseOpenAIResponse:
            """
            DeepSeek模型调用处理函数
            
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

                # 构建DeepSeek API请求参数
                payload = cls._build_deepseek_payload(model, messages, temperature, max_tokens, config)

                # 构建请求头
                headers = cls._build_deepseek_headers(api_key)

                # 发送请求，包含超时参数和重试次数
                debug(f"向DeepSeek发送请求: model={model}, temperature={temperature}, timeout={timeout}s, retry_count={retry_count}")
                try:
                    response = cls.make_request(f"{base_url}/chat/completions", headers, payload, timeout=timeout, retry_count=retry_count)

                    # 检查响应状态码
                    if response.status_code == 402:
                        error(f"DeepSeek API调用失败: 余额不足(Insufficient Balance)，将使用默认响应继续运行")
                        # 返回默认响应，不抛出异常
                        return cls.create_response_from_content("API余额不足，已使用规则增强模式继续处理")
                    elif response.status_code != 200:
                        error(f"DeepSeek API调用失败，状态码: {response.status_code}, 响应内容: {response.text[:200]}...")
                        # 返回默认响应，不抛出异常
                        return cls.create_response_from_content("API调用失败，已使用规则增强模式继续处理")

                    # 解析响应
                    response_data = response.json()
                except Exception as request_error:
                    error(f"DeepSeek API请求异常: {str(request_error)}")
                    # 返回默认响应，不抛出异常
                    return cls.create_response_from_content("API请求异常，已使用规则增强模式继续处理")

                # 转换为OpenAI格式
                content = cls.convert_response(response_data)

                # 创建并返回响应对象
                return cls.create_response_from_content(content)

            except Exception as e:
                # 检查是否为APIStatusError并包含余额不足信息
                error_message = str(e)
                if 'Insufficient Balance' in error_message or '402' in error_message:
                    error(f"DeepSeek API调用失败: 余额不足，将使用默认响应继续运行")
                else:
                    error(f"DeepSeek API调用失败: {error_message}")
                # 返回默认响应，不抛出异常
                return cls.create_response_from_content("API调用异常，已使用规则增强模式继续处理")

        return deepseek_completion_handler

    @classmethod
    def _build_deepseek_payload(cls, model: Optional[str], messages: list,
                                temperature: Optional[float] = None,
                                max_tokens: Optional[int] = None,
                                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        构建DeepSeek特定的请求参数
        
        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成字数
            config: 配置字典，包含默认参数
            
        Returns:
            DeepSeek请求参数字典
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
    def _build_deepseek_headers(cls, api_key: str) -> Dict[str, str]:
        """
        构建DeepSeek特定的请求头
        
        Args:
            api_key: API密钥
            
        Returns:
            DeepSeek请求头字典
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    @staticmethod
    def convert_response(response: Any) -> str:
        """
        转换DeepSeek的响应格式，支持多种响应格式和错误处理
        
        Args:
            response: DeepSeek API返回的响应对象或字典
            
        Returns:
            提取的文本内容
        """
        try:
            # 处理字典类型响应
            if isinstance(response, dict):
                choices = response.get('choices', [])
                if choices and isinstance(choices, list):
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        message = first_choice.get('message', {})
                        if message:
                            return message.get('content', '')
                    # 也处理对象类型的choice
                    elif hasattr(first_choice, 'message') and hasattr(first_choice.message, 'content'):
                        return first_choice.message.content

            # 处理对象类型响应
            elif hasattr(response, 'choices') and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, 'message') and hasattr(first_choice.message, 'content'):
                    return first_choice.message.content

            # 处理直接的文本响应
            elif isinstance(response, str):
                return response

            # 未知格式，记录更详细的日志
            error(f"DeepSeek响应格式无法识别: {type(response).__name__} - {str(response)[:200]}...")
            return ''
        except Exception as e:
            error(f"转换DeepSeek响应失败: {str(e)}")
            # 尝试返回响应的字符串表示作为最后的备选
            try:
                return str(response) if response else ''
            except:
                return ''

    @classmethod
    def get_langchain_llm(cls, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        获取DeepSeek的LangChain LLM实例，优先从本对象获取配置
        
        Args:
            config: 配置参数，包含model、temperature等
            
        Returns:
            LangChain的实例或自定义的错误处理包装器
        """
        config = config or {}

        # 获取API密钥和其他配置
        api_key = cls._get_api_key(config)
        base_url = config.get('base_url', cls.DEFAULT_BASE_URL)
        model = config.get('model', cls.DEFAULT_MODEL)
        temperature = config.get('temperature', 0.1)
        max_tokens = config.get('max_tokens', 2000)
        timeout = config.get('timeout', 60)

        # 创建一个自定义的LLM包装器，用于捕获和处理API错误
        class ErrorHandlingLLMWrapper:
            def __init__(self, model_name):
                self.model_name = model_name
                
            def invoke(self, messages, **kwargs):
                # 直接返回错误处理响应，不调用实际API
                error(f"DeepSeek API调用将被拦截，返回错误处理响应")
                # 创建一个模拟的响应对象
                from langchain_core.messages import AIMessage
                return AIMessage(content="API余额不足，已使用规则增强模式继续处理")

        # 优先尝试使用langchain_deepseek的ChatDeepSeek
        try:
            debug(f"尝试从本对象获取DeepSeek LLM实例，模型: {model}")

            # 构建参数字典，确保只传递需要的参数
            llm_params = {
                'model': model,
                'temperature': temperature,
                'api_key': api_key,
                'base_url': base_url
            }

            # 添加其他可能的参数
            if max_tokens != 2000:
                llm_params['max_tokens'] = max_tokens

            if timeout != 60:
                llm_params['timeout'] = timeout

            # 检查是否有API密钥
            if not api_key:
                debug(f"未配置DeepSeek API密钥，尝试使用环境变量或默认配置")
                # 移除api_key参数，让库自己处理默认行为
                llm_params.pop('api_key', None)

            # 导入并创建ChatDeepSeek实例
            from langchain_deepseek import ChatDeepSeek

            # 尝试创建ChatDeepSeek实例
            debug(f"创建DeepSeek的LangChain实例，模型: {model}")
            # 包装ChatDeepSeek实例以拦截可能的异常
            original_llm = ChatDeepSeek(**llm_params)
            
            # 为了安全起见，我们创建一个代理对象来包装原始LLM
            # 这样可以拦截invoke调用并捕获可能的异常
            class SafeChatDeepSeek:
                def __init__(self, original):
                    self.original = original
                    # 使用安全的方式访问属性，如果不存在则使用None或默认值
                    self.model = getattr(original, 'model', None)
                    self.temperature = getattr(original, 'temperature', 0.1)
                    self.max_tokens = getattr(original, 'max_tokens', 2000)
                    
                def invoke(self, messages, **kwargs):
                    try:
                        return self.original.invoke(messages, **kwargs)
                    except Exception as e:
                        # 检查是否为余额不足错误
                        if 'Insufficient Balance' in str(e) or '402' in str(e):
                            error(f"DeepSeek API调用失败: 余额不足，返回默认响应")
                        else:
                            error(f"DeepSeek API调用失败: {str(e)}")
                        # 返回默认响应，不抛出异常
                        from langchain_core.messages import AIMessage
                        return AIMessage(content="API调用失败，已使用规则增强模式继续处理")
                
                # 添加__getattr__方法，当访问不存在的属性时，尝试从原始对象获取
                def __getattr__(self, name):
                    try:
                        return getattr(self.original, name)
                    except AttributeError:
                        # 如果原始对象也没有该属性，返回None
                        return None
            
            # 返回安全包装后的LLM实例
            safe_llm = SafeChatDeepSeek(original_llm)
            debug(f"成功创建安全的ChatDeepSeek实例，模型: {model}")
            return safe_llm

        except ImportError as import_e:
            error(f"导入langchain_deepseek失败: {str(import_e)}")
        except Exception as e:
            error(f"创建DeepSeek的LangChain LLM实例失败: {str(e)}")

            # 如果主要方法失败，尝试使用OpenAI兼容方式作为回退
            try:
                debug("DeepSeek实现失败，回退到OpenAI兼容方式")
                # 尝试从openai_client获取LLM实例
                # 构建兼容的配置
                openai_config = {
                    'model': model,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'timeout': timeout,
                    'base_url': base_url
                }

                # 如果有API密钥，添加到配置中
                if api_key:
                    openai_config['api_key'] = api_key

                # 使用OpenAIClient获取LLM实例
                llm = OpenAIClient.get_langchain_llm(openai_config)
                if llm:
                    debug(f"成功使用OpenAIClient获取LLM实例，模型: {model}")
                    return llm

            except ImportError as import_e:
                error(f"导入OpenAIClient失败: {str(import_e)}")
            except Exception as openai_e:
                error(f"使用OpenAIClient获取LLM实例失败: {str(openai_e)}")

        # 所有尝试都失败，返回错误处理包装器
        error(f"无法创建DeepSeek的任何LangChain LLM实例，返回错误处理包装器，模型: {model}")
        return ErrorHandlingLLMWrapper(model)
