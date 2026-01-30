"""
@FileName: config.py
@Description: 配置管理模块
@Author: HengLine
@Time: 2026/1/30 17:55
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic_settings import BaseSettings
from pydantic import Field, validator, ConfigDict
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()


class LLMConfig(BaseSettings):
    """LLM配置"""
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = Field(default="", env="OPENAI_API_KEY")
    timeout: int = 60
    model_name: str = "gpt-4o"
    response_format: str = "json_object"
    temperature: float = 0.1
    max_tokens: int = 2000
    max_retries: int = 3

    # 备用配置
    fallback_provider: Optional[str] = None
    deepseek_api_key: Optional[str] = Field(default=None, env="DEEPSEEK_API_KEY")
    wenxin_api_key: Optional[str] = Field(default=None, env="WENXIN_API_KEY")
    wenxin_secret_key: Optional[str] = Field(default=None, env="WENXIN_SECRET_KEY")
    ollama_base_url: Optional[str] = Field(default="http://localhost:11434")

    model_config = ConfigDict(env_prefix="LLM_", extra="ignore")


class EmbeddingConfig(BaseSettings):
    """嵌入模型配置"""
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = Field(default="", env="OPENAI_API_KEY")
    model_name: str = "text-embedding-3-small"
    dimensions: int = 1536
    timeout: int = 60
    max_retries: int = 3

    # 备用配置
    fallback_provider: Optional[str] = None
    huggingface_model: str = "BAAI/bge-small-zh-v1.5"

    model_config = ConfigDict(env_prefix="EMBEDDING_", extra="ignore")


class StoryboardConfig(BaseSettings):
    """分镜配置"""
    default_duration_per_shot: int = 5
    max_duration_deviation: float = 0.5
    max_retries: int = 2
    default_style: str = "realistic"
    supported_styles: list = ["realistic", "anime", "cinematic", "cartoon"]

    model_config = ConfigDict(env_prefix="STORYBOARD_", extra="ignore")


class PathsConfig(BaseSettings):
    """路径配置"""
    data_input: str = "/data/input"
    data_output: str = "/data/output"
    model_cache: str = "/data/models"
    embedding_cache: str = "/data/embeddings"

    @validator('*', pre=True)
    def ensure_path_exists(cls, v):
        """确保路径存在"""
        if isinstance(v, str):
            path = Path(v)
            path.mkdir(parents=True, exist_ok=True)
        return v

    model_config = ConfigDict(env_prefix="PATHS_", extra="ignore")


class APIConfig(BaseSettings):
    """API配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False

    model_config = ConfigDict(env_prefix="API_", extra="ignore")


class Settings(BaseSettings):
    """主配置"""
    app_name: str = "Script-to-Shot AI Agent"
    version: str = "1.0.0"
    debug: bool = False
    environment: str = Field(default="development", env="ENVIRONMENT")

    # 子配置
    api: APIConfig = Field(default_factory=APIConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storyboard: StoryboardConfig = Field(default_factory=StoryboardConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    # 从YAML文件加载的额外配置
    _extra_config: Dict[str, Any] = {}

    model_config = ConfigDict(env_prefix="APP_", extra="ignore")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 根据环境加载YAML配置
        self._load_yaml_config()

        # 更新配置
        if self._extra_config:
            self._update_from_yaml()

    def _load_yaml_config(self):
        """从YAML文件加载配置"""
        config_dir = Path(__file__).parent.parent / "config"

        # 确定环境配置文件
        if self.environment.lower() == "prod":
            config_file = config_dir / "production.yaml"
        else:
            config_file = config_dir / "development.yaml"

        # 加载日志配置
        log_config_file = config_dir / "logging.yaml"

        try:
            # 加载主配置
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self._extra_config = yaml.safe_load(f) or {}

            # 加载日志配置
            if log_config_file.exists():
                with open(log_config_file, 'r', encoding='utf-8') as f:
                    self._extra_config.setdefault('logging', {})
                    self._extra_config['logging']['yaml_config'] = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load YAML config: {e}")

    def _update_from_yaml(self):
        """用YAML配置更新当前配置"""
        # 这里可以根据需要实现具体的更新逻辑
        # 例如，合并配置或覆盖默认值
        pass

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self._extra_config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        if 'logging' in self._extra_config:
            return self._extra_config['logging'].get('yaml_config', {})
        return {}


# 全局配置实例
settings = Settings()