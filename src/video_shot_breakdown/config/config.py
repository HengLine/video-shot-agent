# -*- coding: utf-8 -*-
"""
@FileName: config.py
@Description: 配置管理模块，负责读取和管理应用配置
@Author: HengLine
@Time: 2025/08 - 2025/11
"""
import json
import os
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from video_shot_breakdown.hengline.logger import error, warning, debug

# 加载.env文件中的环境变量
load_dotenv()

# 默认配置
DEFAULT_CONFIG = {
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1
    },
    "llm": {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "timeout": 60,
        "model_name": "gpt-4o",
        "response_format": "json_object",
        "temperature": 0.1,
        "max_tokens": 2000,
        "max_retries": 3
    },
    "embedding": {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model_name": "text-embedding-3-small",
        "dimensions": 1536,
        "timeout": 60,
        "max_retries": 3
    }
}

# 全局配置实例
_config_instance: Optional[Dict[str, Any]] = None


def get_config_path() -> str:
    """
    获取配置文件路径
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.json')
    return config_path


def _update_api_config_from_env(config: Dict[str, Any]) -> None:
    """
    从环境变量更新API配置
    优先从环境变量获取端口、主机等API配置
    """
    api_config = config.get("api", {})
    
    # 从环境变量获取端口配置，支持PORT或API_PORT
    port_env_var = os.environ.get("PORT")
    if not port_env_var:
        port_env_var = os.environ.get("API_PORT")
    
    if port_env_var:
        try:
            api_config["port"] = int(port_env_var)
        except ValueError:
            warning(f"无效的端口号: {port_env_var}，使用默认值")
    
    # 从环境变量获取主机配置
    host_env_var = os.environ.get("HOST")
    if not host_env_var:
        host_env_var = os.environ.get("API_HOST")
    
    if host_env_var:
        api_config["host"] = host_env_var
    
    # 从环境变量获取工作进程数
    workers_env_var = os.environ.get("WORKERS")
    if not workers_env_var:
        workers_env_var = os.environ.get("API_WORKERS")
    
    if workers_env_var:
        try:
            api_config["workers"] = int(workers_env_var)
        except ValueError:
            warning(f"无效的工作进程数: {workers_env_var}，使用默认值")


def _update_ai_config_from_env(config: Dict[str, Any]) -> None:
    """
    从环境变量更新AI配置
    根据LLM_PROVIDER环境变量动态加载相应的API配置
    确保环境变量中的provider值具有最高优先级
    """
    ai_config = config.get("llm", {})
    
    # 优先从环境变量获取AI提供商，如果不存在则使用配置中的值，最后使用默认值
    env_provider = os.environ.get("LLM_PROVIDER")
    if env_provider:
        # 环境变量存在，优先使用
        provider = env_provider.lower()
    else:
        # 环境变量不存在，使用配置中的值或默认值
        provider = ai_config.get("provider", "openai").lower()
    
    ai_config["provider"] = provider
    
    # 动态加载指定提供商的API配置
    # API密钥命名格式: {PROVIDER}_API_KEY
    # Base URL命名格式: {PROVIDER}_BASE_URL
    # 模型命名格式: {PROVIDER}_MODEL
    # 备用模型命名格式: {PROVIDER}_FALLBACK_MODEL
    provider_upper = provider.upper()
    api_key_env_var = f"{provider_upper}_API_KEY"
    base_url_env_var = f"{provider_upper}_BASE_URL"
    model_env_var = f"{provider_upper}_MODEL"
    fallback_model_env_var = f"{provider_upper}_FALLBACK_MODEL"
    
    # 加载API密钥
    if os.environ.get(api_key_env_var):
        ai_config["api_key"] = os.environ[api_key_env_var]

        if provider == "qwen":
            os.environ["DASHSCOPE_API_KEY"] = ai_config["api_key"]
    
    # 加载Base URL
    if os.environ.get(base_url_env_var):
        ai_config["base_url"] = os.environ[base_url_env_var]
    
    # 加载特定提供商的模型配置
    if os.environ.get(model_env_var):
        ai_config["model_name"] = os.environ[model_env_var]
    
    # 加载特定提供商的备用模型配置（可选）
    if os.environ.get(fallback_model_env_var):
        ai_config["fallback_model"] = os.environ[fallback_model_env_var]
    
    # 加载统一的超时时间配置
    if os.environ.get("LLM_API_TIMEOUT"):
        try:
            ai_config["timeout"] = int(os.environ["LLM_API_TIMEOUT"])
        except ValueError:
            warning("Invalid LLM_API_TIMEOUT value, using default")

    # 加载温度参数
    if os.environ.get("LLM_TEMPERATURE"):
        try:
            ai_config["temperature"] = float(os.environ["LLM_TEMPERATURE"])
        except ValueError:
            warning("Invalid LLM_TEMPERATURE value, using default")
    
    # 加载最大令牌数
    if os.environ.get("LLM_MAX_TOKENS"):
        try:
            ai_config["max_tokens"] = int(os.environ["LLM_MAX_TOKENS"])
        except ValueError:
            warning("Invalid LLM_MAX_TOKENS value, using default")
    
    # 加载重试次数
    if os.environ.get("LLM_MAX_RETRIES"):
        try:
            ai_config["max_retries"] = int(os.environ["LLM_MAX_RETRIES"])
        except ValueError:
            warning("Invalid LLM_MAX_RETRIES value, using default")


def _update_embedding_config_from_env(config: Dict[str, Any]) -> None:
    """
    从环境变量更新嵌入模型配置
    根据EMBEDDING_PROVIDER环境变量动态加载相应的API配置
    确保环境变量中的provider值具有最高优先级
    """
    # 确保配置中存在embedding字段
    if "embedding" not in config:
        config["embedding"] = {}
    
    embedding_config = config["embedding"]
    
    # 确保embedding_config是一个字典
    if not isinstance(embedding_config, dict):
        embedding_config = {}
        config["embedding"] = embedding_config
    
    # 优先从环境变量获取嵌入模型提供商
    if "EMBEDDING_PROVIDER" in os.environ:
        provider = os.environ["EMBEDDING_PROVIDER"].lower()
        embedding_config["provider"] = provider
    
    # 基础配置
    if "EMBEDDING_MODEL" in os.environ:
        embedding_config["model_name"] = os.environ["EMBEDDING_MODEL"]
    
    if "EMBEDDING_BASE_URL" in os.environ:
        embedding_config["base_url"] = os.environ["EMBEDDING_BASE_URL"]
    
    if "EMBEDDING_API_KEY" in os.environ:
        embedding_config["api_key"] = os.environ["EMBEDDING_API_KEY"]
    
    # 数字类型配置
    if "EMBEDDING_DIMENSIONS" in os.environ:
        try:
            embedding_config["dimensions"] = int(os.environ["EMBEDDING_DIMENSIONS"])
        except ValueError:
            warning(f"无效的嵌入维度值: {os.environ['EMBEDDING_DIMENSIONS']}")
    
    # 超时配置
    if "EMBEDDING_TIMEOUT" in os.environ:
        try:
            embedding_config["timeout"] = int(os.environ["EMBEDDING_TIMEOUT"])
        except ValueError:
            warning(f"无效的嵌入超时值: {os.environ['EMBEDDING_TIMEOUT']}")
    
    # 重试配置
    if "EMBEDDING_MAX_RETRIES" in os.environ:
        try:
            embedding_config["max_retries"] = int(os.environ["EMBEDDING_MAX_RETRIES"])
        except ValueError:
            warning(f"无效的嵌入重试次数: {os.environ['EMBEDDING_MAX_RETRIES']}")
    
    # 更新到原始配置中
    config["embedding"] = embedding_config


def get_settings_config() -> Dict[str, Any]:
    """
    获取应用设置配置
    """
    global _config_instance

    if _config_instance is not None:
        return _config_instance

    try:
        config_path = get_config_path()
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # 配置文件不存在，使用默认配置
            config = DEFAULT_CONFIG.copy()
            
            # 创建配置文件目录（如果不存在）
            config_dir = os.path.dirname(config_path)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            # 写入默认配置到文件
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
                
            warning(f"配置文件不存在，已创建默认配置文件: {config_path}")
    except Exception as e:
        error(f"加载配置文件失败: {str(e)}")
        # 使用默认配置
        config = DEFAULT_CONFIG.copy()
    
    # 合并默认配置和读取的配置，确保所有必要的配置项都存在
    merged_config = DEFAULT_CONFIG.copy()
    for key, value in config.items():
        if key in merged_config and isinstance(merged_config[key], dict):
            merged_config[key].update(value)
        else:
            merged_config[key] = value
    
    # 从环境变量更新配置 - 确保调用顺序正确
    _update_embedding_config_from_env(merged_config)
    _update_api_config_from_env(merged_config)
    _update_ai_config_from_env(merged_config)
    
    # 将环境变量中的调试模式设置到配置中
    debug_mode_env = os.environ.get("APP_DEBUG", "false").lower()
    if debug_mode_env in ["true", "1", "yes"]:
        merged_config["app"]["debug"] = True
    
    debug(f"配置加载完成，最终 Embedding 配置: {merged_config.get('embedding', {})}")
    debug(f"配置加载完成，最终 LLM 配置: {merged_config.get('llm', {})}")
    _config_instance = merged_config
    return merged_config


def get_app_root() -> str:
    """
    获取应用根目录
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_storyboard_config() -> Dict[str, Any]:
    """
    获取分镜生成配置
    """
    config = get_settings_config()
    return config.get("storyboard", {})


def get_llm_config() -> Dict[str, Any]:
    """
    获取AI模型配置
    """
    config = get_settings_config()
    return config.get("llm", {})


def is_debug_mode() -> bool:
    """
    检查是否为调试模式
    """
    config = get_settings_config()
    return config.get("app", {}).get("debug", False)


def get_supported_styles() -> list:
    """
    获取支持的风格列表
    """
    storyboard_config = get_storyboard_config()
    return storyboard_config.get("supported_styles", [])


def get_paths_config() -> Dict[str, str]:
    """
    获取路径配置
    """
    config = get_settings_config()
    return config.get("paths", {})


def get_data_paths() -> Dict[str, str]:
    """
    获取数据路径配置
    """
    paths_config = get_paths_config()
    app_root = get_app_root()
    
    # 确保路径是绝对路径
    data_paths = {}
    for key, path in paths_config.items():
        if path and not os.path.isabs(path):
            data_paths[key] = os.path.join(app_root, path)
        else:
            data_paths[key] = path
    
    return data_paths


def get_data_input_path() -> str:
    """
    获取数据输入路径
    """
    return get_data_paths()["data_input"]


def get_data_output_path() -> str:
    """
    获取数据输出路径
    """
    return get_data_paths()["data_output"]


def get_data_embeddings_path() -> str:
    """
    获取数据嵌入存储路径
    """
    return get_data_paths()["data_output"]

def get_embedding_config() -> Dict[str, Any]:
    """
    获取嵌入模型配置
    
    Returns:
        Dict[str, Any]: 嵌入模型的完整配置
    """
    config = get_settings_config()
    return config.get("embedding", {})
