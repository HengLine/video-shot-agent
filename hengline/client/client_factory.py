# -*- coding: utf-8 -*-
"""
模型客户端工厂模块
负责根据配置创建不同类型的模型客户端
"""
from utils.log_utils import print_log_exception

"""
@FileName: client_factory.py
@Description: 客户端工厂模块，负责统一调用不同厂商的客户端实现
@Author: HengLine
@Time: 2025/10/6
"""
from typing import Any, Dict, Optional

from openai import OpenAI

from config.config import get_ai_config
from hengline.client.deepseek_client import DeepSeekClient
from hengline.client.ollama_client import OllamaClient
# 导入各个厂商的客户端实现
from hengline.client.openai_client import OpenAIClient
from hengline.client.qwen_client import QwenClient
from hengline.logger import error, debug, warning, info


class ClientFactory:
    """AI客户端工厂类，负责统一调用不同厂商的客户端实现"""

    # 支持的提供商列表
    SUPPORTED_PROVIDERS = ['openai', 'qwen', 'deepseek', 'ollama']

    @classmethod
    def create_client(cls, provider: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        根据提供商名称创建对应的AI客户端
        
        Args:
            provider: AI模型提供商名称
            config: 客户端配置参数
            
        Returns:
            配置好的AI客户端实例（确保返回OpenAI兼容格式）
        """
        # 确保提供商名称小写
        provider = provider.lower()
        if not provider or provider not in cls.SUPPORTED_PROVIDERS:
            error(f"不支持的AI模型提供商: {provider}")
            raise ValueError(f"不支持的AI模型提供商: {provider}")

        # 如果未提供配置，从全局配置中获取
        if config is None:
            global_config = get_ai_config()
            # 直接使用全局配置，因为get_ai_config()已经返回了完整的llm配置
            config = global_config.copy()

        # 根据提供商类型创建对应的客户端，传递完整配置（包括retry_count）
        if provider == 'openai':
            return cls._create_openai_client(config)
        elif provider == 'qwen':
            return QwenClient.create_client(config)
        elif provider == 'deepseek':
            return DeepSeekClient.create_client(config)
        elif provider == 'ollama':
            return OllamaClient.create_client(config)
            
    @classmethod
    def get_langchain_llm(cls, provider: str = None, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        获取LangChain兼容的LLM实例
        
        Args:
            provider: AI服务提供商名称
            config: 配置参数，包含model、temperature等
            
        Returns:
            LangChain兼容的LLM实例，如果不支持则返回None
        """
        config = config or {}
        
        # 如果没有指定提供商，尝试从配置中获取
        if not provider:
            provider = config.get('provider', 'openai')
        # 确保提供商名称小写
        provider = provider.lower()
        
        # 检查提供商是否支持
        if provider not in cls.SUPPORTED_PROVIDERS:
            warning(f"不支持的AI提供商: {provider}")
            return None
        
        # 获取对应的客户端类
        try:
            client_class = cls.get_provider_client_class(provider)
            return client_class.get_langchain_llm(config)
        except NotImplementedError:
            warning(f"{provider} 客户端未实现 get_langchain_llm 方法")
        except Exception as e:
            error(f"获取 {provider} 的LangChain LLM实例失败: {str(e)}")
        
        return None

    @staticmethod
    def _create_openai_client(config: Dict[str, Any]) -> OpenAI:
        """创建OpenAI客户端"""
        return OpenAIClient.create_client(config)

    @classmethod
    def get_supported_providers(cls) -> list:
        """获取支持的AI模型提供商列表"""
        return cls.SUPPORTED_PROVIDERS

    @classmethod
    def get_provider_client_class(cls, provider: str) -> Any:
        """
        获取指定提供商的客户端类
        
        Args:
            provider: AI模型提供商名称
            
        Returns:
            对应的客户端类
        """
        # 确保提供商名称小写
        provider = provider.lower()
        provider_map = {
            'openai': OpenAIClient,
            'qwen': QwenClient,
            'deepseek': DeepSeekClient,
            'ollama': OllamaClient
        }

        if provider not in provider_map:
            error(f"未找到提供商 {provider} 对应的客户端类")
            raise ValueError(f"未找到提供商 {provider} 对应的客户端类")

        return provider_map[provider]


# 创建全局工厂实例
ai_client_factory = ClientFactory()

def get_llm_client() -> Optional[Any]:
    """
    获取LangChain兼容的LLM客户端的便捷函数

    Returns:
        配置好的LangChain LLM实例，或None如果不支持
    """
    try:
        from config.config import get_ai_config

        # 获取配置
        ai_config = get_ai_config()
        provider = ai_config.get("provider", "openai").lower()
        model = ai_config.get("default_model", "gpt-4o")
        temperature = ai_config.get("temperature", 0.1)

        info(f"使用AI提供商: {provider}, 模型: {model}")

        # 创建完整的LLM配置
        llm_config = {
            'model': model,
            'temperature': temperature,
            **ai_config  # 包含API密钥等配置
        }

        # 使用client_factory获取对应的LangChain LLM实例
        llm = ai_client_factory.get_langchain_llm(provider=provider, config=llm_config)

        if not llm:
            warning(f"AI模型初始化失败（未能获取 {provider} 的LLM实例），系统将自动使用规则引擎模式继续工作")

        return llm
    except Exception as e:
        print_log_exception()
        error(f"AI模型初始化失败（错误: {str(e)}），系统将自动使用规则引擎模式继续工作")


def get_ai_client(provider: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    获取AI客户端的便捷函数
    
    Args:
        provider: AI模型提供商名称，如果为None则使用配置中的默认值
        config: 客户端配置参数
        
    Returns:
        配置好的AI客户端实例
    """
    # 如果未指定提供商，从配置中获取默认值
    if provider is None:
        provider = get_ai_config().get('provider', 'openai')
    # 确保提供商名称小写
    provider = provider.lower()

    # 使用工厂创建客户端
    return ai_client_factory.create_client(provider, config)


def convert_response(provider: str, response: Any) -> str:
    """
    转换特定提供商的响应格式为文本，增强格式兼容性和错误处理
    
    Args:
        provider: AI模型提供商名称
        response: API响应对象或字典
        
    Returns:
        提取的文本内容
    """
    try:
        # 确保提供商名称小写
        provider = provider.lower()
        # 空响应检查
        if response is None:
            debug(f"收到空响应，提供商: {provider}")
            return ''
        
        # 对于特定提供商，使用其专用的转换方法
        if provider in ['qwen', 'deepseek', 'ollama']:
            try:
                client_class = ai_client_factory.get_provider_client_class(provider)
                result = client_class.convert_response(response)
                # 验证结果非空
                if result:
                    return result
                debug(f"提供商{provider}的转换方法返回空结果")
            except Exception as e:
                error(f"调用{provider}的转换方法失败: {str(e)}")
                # 继续尝试通用转换方法
        
        # OpenAI和通用响应格式处理
        # 1. 处理对象类型响应
        if hasattr(response, 'choices') and response.choices:
            first_choice = response.choices[0]
            if hasattr(first_choice, 'message'):
                message = first_choice.message
                if hasattr(message, 'content'):
                    return message.content
                elif hasattr(message, 'text'):
                    return message.text
        
        # 2. 处理字典类型响应
        elif isinstance(response, dict):
            # 标准OpenAI格式
            choices = response.get('choices', [])
            if choices and isinstance(choices, list):
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get('message', {})
                    if isinstance(message, dict):
                        return message.get('content', '') or message.get('text', '')
                    elif hasattr(message, 'content'):
                        return message.content
            
            # 其他常见格式
            # 直接输出格式
            if 'content' in response:
                return response['content']
            
            # 输出对象格式
            output = response.get('output')
            if output:
                if isinstance(output, dict):
                    return output.get('text', '') or output.get('content', '')
                elif hasattr(output, 'text'):
                    return output.text
                elif hasattr(output, 'content'):
                    return output.content
        
        # 3. 直接文本响应
        elif isinstance(response, str):
            return response
        
        # 4. 处理其他可能的格式
        # 检查是否有直接的文本属性
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'content'):
            return response.content
        
        # 未知格式，记录详细信息以便调试
        error(f"无法识别的响应格式，提供商: {provider}, 类型: {type(response).__name__}, 内容: {str(response)[:200]}...")
        
        # 最后的备选方案：返回响应的字符串表示
        return str(response) if response else ''
        
    except Exception as e:
        error(f"转换响应时发生异常，提供商: {provider}, 错误: {str(e)}")
        # 安全地返回响应的字符串表示
        try:
            return str(response) if response else ''
        except:
            return ''
