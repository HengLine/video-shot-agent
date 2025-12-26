# -*- coding: utf-8 -*-
"""
@FileName: ollama_client.py
@Description: Ollama本地模型客户端
@Author: HengLine
@Time: 2025/10/6
"""

from typing import Dict, Any, Optional, Callable
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.callbacks import CallbackManager
from hengline.client.openai_client import OpenAIClient
from hengline.client.base_client import BaseAIClient
from hengline.client.openai_compat import OpenAICompatibleWrapper, BaseOpenAIResponse
from hengline.logger import debug, error


class OllamaClient(BaseAIClient):
    """
    Ollama本地模型客户端
    提供Ollama本地模型的访问和响应处理
    """

    # Ollama特定配置
    PROVIDER_NAME = "ollama"
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3"
    API_KEY_ENV_VAR = "OLLAMA_API_KEY"  # Ollama通常不需要API密钥，但为了接口一致性保留
    
    @staticmethod
    def make_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 60, retry_count:int = 3) -> Any:
        """
        发送API请求
        
        Args:
            url: 请求URL
            headers: 请求头
            payload: 请求体
            timeout: 请求超时时间（秒）
            
        Returns:
            API响应对象
        """
        # 导入requests库
        import requests

        # 发送POST请求，包含超时参数
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        
        # 检查响应状态
        response.raise_for_status()
        return response

    @classmethod
    def _get_client_implementation(cls, api_key: str, base_url: str, config: Dict[str, Any]) -> OpenAICompatibleWrapper:
        """
        获取Ollama客户端实现
        
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
        创建Ollama的completion处理函数
        
        Args:
            api_key: API密钥（Ollama通常不需要）
            base_url: 基础URL
            
        Returns:
            completion处理函数
        """

        def ollama_completion_handler(model: str = None, messages: list = None,
                                      temperature: Optional[float] = None,
                                      max_tokens: Optional[int] = None,
                                      response_format: Optional[Dict] = None,
                                      **kwargs) -> BaseOpenAIResponse:
            """
            Ollama模型调用处理函数
            
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
                
                # 构建Ollama API请求参数
                payload = cls._build_ollama_payload(model, messages, temperature, max_tokens, config)

                # 构建请求头（Ollama通常不需要Authorization）
                headers = cls._build_ollama_headers(api_key)

                # 发送请求，包含超时参数和重试次数
                debug(f"向Ollama发送请求: model={model}, temperature={temperature}, timeout={timeout}s, retry_count={retry_count}")
                response = cls.make_request(f"{base_url}/api/chat", headers, payload, timeout=timeout, retry_count=retry_count)

                # 解析响应
                response_data = response.json()

                # 转换为OpenAI格式
                content = cls.convert_response(response_data)

                # 创建并返回响应对象
                return cls.create_response_from_content(content)

            except Exception as e:
                error(f"Ollama调用失败: {str(e)}")
                raise

        return ollama_completion_handler

    @classmethod
    def _build_ollama_payload(cls, model: Optional[str], messages: list,
                              temperature: Optional[float] = None,
                              max_tokens: Optional[int] = None,
                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        构建Ollama特定的请求参数
        
        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成字数
            config: 配置字典，包含默认参数
            
        Returns:
            Ollama请求参数字典
        """
        config = config or {}
        # 从配置中获取默认值
        default_model = config.get('default_model', cls.DEFAULT_MODEL)
        default_temperature = config.get('temperature', 0.1)
        default_max_tokens = config.get('max_tokens', 2000)
        
        payload = {
            "model": model or default_model,
            "messages": messages,
            "stream": False
        }

        # 如果未提供温度参数，使用配置中的默认值
        if temperature is not None:
            payload['temperature'] = temperature
        elif default_temperature != 0.1:  # 仅当配置了非默认值时才设置
            payload['temperature'] = default_temperature
            
        # 如果未提供最大令牌数，使用配置中的默认值
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens
        elif default_max_tokens != 2000:  # 仅当配置了非默认值时才设置
            payload['max_tokens'] = default_max_tokens

        return payload

    @classmethod
    def _build_ollama_headers(cls, api_key: str) -> Dict[str, str]:
        """
        构建Ollama特定的请求头
        
        Args:
            api_key: API密钥（Ollama通常不需要）
            
        Returns:
            Ollama请求头字典
        """
        headers = {
            "Content-Type": "application/json"
        }

        # 如果提供了API密钥，则添加Authorization头
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        return headers

    @staticmethod
    def convert_response(response: Dict) -> str:
        """
        转换Ollama的响应格式
        
        Args:
            response: Ollama API返回的响应字典
            
        Returns:
            提取的文本内容
        """
        return response.get('message', {}).get('content', '')
    
    @classmethod
    def get_langchain_llm(cls, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        获取Ollama的LangChain LLM实例，优先从本对象获取配置
        
        Args:
            config: 配置参数，包含model、temperature、timeout等
            
        Returns:
            LangChain的实例
        """
        config = config or {}
        
        # 获取统一配置参数
        base_url = config.get('base_url', cls.DEFAULT_BASE_URL)
        model = config.get('default_model', config.get('model', cls.DEFAULT_MODEL))
        temperature = config.get('temperature', 0.1)
        timeout = config.get('timeout', 180)
        max_tokens = config.get('max_tokens', 2000)
        
        # 优先尝试使用langchain_ollama的ChatOllama
        try:
            debug(f"尝试从本对象获取Ollama LLM实例，模型: {model}")
            
            # 构建参数字典，确保只传递需要的参数
            llm_params = {
                'model': model,
                'temperature': temperature,
                'base_url': base_url,
                'timeout': timeout * 2,
                'repeat_penalty': 1.1,        # 重复惩罚
                # 'format': 'json',           # 关键参数：强制返回 JSON
                # 'num_ctx': 2048,              # 上下文长度，根据模型调整
                'num_thread': 4,              # 线程数，根据CPU核心数调整
                'stream': False,               # 非流式，一次性返回
                # 'stop': ["\n\n", "。", "！", "？"],  # 停止词，避免无限生成
                'callbacks': CallbackManager([])
            }
            
            # 添加其他可能的参数
            if max_tokens != 2000:
                llm_params['max_tokens'] = max_tokens
            
            # 添加其他可能的Ollama特定参数
            if config.get('keep_alive') is not None:
                llm_params['keep_alive'] = config.get('keep_alive')
            else:
                llm_params['keep_alive'] = -1
            
            # 最大生成长度
            if config.get('num_predict') is not None:
                llm_params['num_predict'] = config.get('num_predict')

            # 采样参数
            if config.get('top_k') is not None:
                llm_params['top_k'] = config.get('top_k')
            else:
                llm_params['top_k'] = 20
                
            # 核采样参数
            if config.get('top_p') is not None:
                llm_params['top_p'] = config.get('top_p')
            else:
                llm_params['top_p'] = 0.9

            # 创建Ollama实例
            debug(f"创建Ollama的LangChain实例，模型: {model}")
            # llm = Ollama(**llm_params)
            # 使用新的 OllamaLLM 类
            llm = OllamaLLM(**llm_params)

            # 验证实例是否成功创建
            if llm:
                debug(f"成功创建Ollama实例，模型: {model}")
                return llm
                
        except ImportError as import_e:
            error(f"导入langchain_ollama失败: {str(import_e)}")
        except Exception as e:
            error(f"创建Ollama的LangChain LLM实例失败: {str(e)}")
        
        # 如果主要方法失败，尝试使用OpenAI兼容方式作为回退
        try:
            debug("Ollama实现失败，回退到OpenAI兼容方式")
            
            # 构建兼容的配置
            openai_config = {
                'model': model,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'timeout': timeout,
                'base_url': base_url + "/v1"  # Ollama OpenAI兼容端点通常在/v1路径
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
        
        # 所有尝试都失败
        error(f"无法创建Ollama的任何LangChain LLM实例，模型: {model}")
        return None
