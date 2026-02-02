"""
@FileName: config.py
@Description: 配置管理模块 - 严格遵循 env > yaml > default 优先级
@Author: HengLine (优化版)
@Time: 2026/01
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from video_shot_breakdown.hengline.logger import debug, error, warning
from video_shot_breakdown.utils.path_utils import PathResolver

# ==================== 路径配置 ====================
# 确定项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
ENV_FILE = PROJECT_ROOT / ".env"

# 加载 .env 文件（如果存在）
load_dotenv(ENV_FILE)

debug(f"配置目录: {CONFIG_DIR}")
debug(f".env 文件: {ENV_FILE}")

# 确保目录存在
CONFIG_DIR.mkdir(exist_ok=True)


class LLMProviderConfig(BaseModel):
    """LLM提供商配置"""
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: SecretStr = Field(default=SecretStr(""))
    model_name: str = Field(default="gpt-4o")
    timeout: int = Field(default=60, ge=1)
    response_format: str = Field(default="json_object")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2000, ge=1)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: int = Field(default=1, ge=0, le=60)

    @field_validator("base_url", "model_name", mode="before")
    @classmethod
    def strip_whitespace(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v):
        """API密钥验证"""
        if isinstance(v, str) and v.startswith("$"):
            env_var = v[2:-1] if v.startswith("${") else v[1:]
            return os.getenv(env_var, "")
        return v


class EmbeddingProviderConfig(BaseModel):
    """嵌入模型提供商配置"""
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: SecretStr = Field(default=SecretStr(""))
    model_name: str = Field(default="text-embedding-3-small")
    device: str = Field(default="gpu")
    normalize_embeddings: bool = Field(default=True)
    dimensions: int = Field(default=1536, ge=1)
    timeout: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0, le=10)

    @field_validator("base_url", "model_name", mode="before")
    @classmethod
    def strip_whitespace(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v):
        """API密钥验证"""
        if isinstance(v, str) and v.startswith("$"):
            env_var = v[2:-1] if v.startswith("${") else v[1:]
            return os.getenv(env_var, "")
        return v


class LLMConfig(BaseModel):
    """LLM主配置"""
    default: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    fallback: Optional[LLMProviderConfig] = Field(default_factory=LLMProviderConfig)


class EmbeddingConfig(BaseModel):
    """嵌入模型主配置"""
    default: EmbeddingProviderConfig = Field(default_factory=EmbeddingProviderConfig)
    fallback: EmbeddingProviderConfig = Field(default_factory=EmbeddingProviderConfig)


class APIConfig(BaseModel):
    """API配置"""
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=10)
    reload: bool = Field(default=False)
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
        ]
    )
    api_prefix: str = Field(default="/api")
    docs_url: str = Field(default="/docs")


class AppConfig(BaseModel):
    """应用配置"""
    name: str = Field(default="Script-to-Shot AI Agent")
    version: str = Field(default="1.0.0")
    description: str = Field(default="脚本转分镜AI助手")
    environment: Literal["development", "production"] = Field(default="development")
    language: Literal["zh", "en"] = Field(default="zh")


class StoryboardGenerationConfig(BaseModel):
    """分镜生成配置"""
    default_duration_per_shot: int = Field(default=5, ge=1)
    max_duration_deviation: float = Field(default=0.5, ge=0.0)
    max_retries: int = Field(default=2, ge=0)
    default_style: str = Field(default="realistic")
    supported_styles: List[str] = Field(
        default_factory=lambda: [
            "realistic", "anime", "cinematic", "cartoon", "fantasy", "sci-fi", "documentary"
        ]
    )


class StoryboardStructureConfig(BaseModel):
    """分镜结构配置"""
    min_shots: int = Field(default=1, ge=1)
    max_shots: int = Field(default=50, ge=1)
    default_scenes_per_shot: int = Field(default=5, ge=1)
    enable_transitions: bool = Field(default=True)
    include_dialogue: bool = Field(default=True)
    include_camera_angles: bool = Field(default=True)


class StoryboardOutputConfig(BaseModel):
    """分镜输出配置"""
    format: Literal["json", "yaml", "xml"] = Field(default="json")
    include_timestamps: bool = Field(default=True)
    include_visual_descriptions: bool = Field(default=True)
    include_character_list: bool = Field(default=True)
    include_location_list: bool = Field(default=True)
    include_shot_duration: bool = Field(default=True)


class StoryboardConfig(BaseModel):
    """完整分镜配置"""
    generation: StoryboardGenerationConfig = Field(default_factory=StoryboardGenerationConfig)
    structure: StoryboardStructureConfig = Field(default_factory=StoryboardStructureConfig)
    output: StoryboardOutputConfig = Field(default_factory=StoryboardOutputConfig)


class PathsConfig(BaseModel):
    """路径配置"""
    data_input: str = Field(default="data/input")
    data_output: str = Field(default="data/output")
    model_cache: str = Field(default="data/models")
    embedding_cache: str = Field(default="data/embeddings")


# ==================== 统一的配置源 ====================
class UnifiedConfigSource(PydanticBaseSettingsSource):
    """统一的配置源：合并YAML和环境变量"""

    def __init__(self, settings_cls: type[BaseSettings]):
        super().__init__(settings_cls)
        self.yaml_config = self._load_yaml_config()
        self.env_config = self._load_env_config()

    def get_field_value(self, field: FieldInfo, field_name: str) -> Tuple[Any, str, bool]:
        return None, "", False

    def __call__(self) -> Dict[str, Any]:
        """合并YAML和环境变量配置"""
        # 深拷贝YAML配置作为基础
        config = self._deep_copy(self.yaml_config)

        # 用环境变量覆盖（环境变量优先级更高）
        config = self._merge_env_into_config(config, self.env_config)

        debug(f"配置合并: YAML配置项={len(self._flatten_dict(self.yaml_config))}, "
              f"环境变量配置项={len(self._flatten_dict(self.env_config))}")

        return config

    def _load_yaml_config(self) -> Dict[str, Any]:
        """加载YAML配置"""
        config = {}

        # 从环境变量获取环境
        env = os.getenv("ENVIRONMENT", "development").lower()

        # 1. 加载基础配置
        settings_file = CONFIG_DIR / "settings.yaml"
        if settings_file.exists():
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                    debug(f"加载 settings.yaml: {len(self._flatten_dict(config))} 个配置项")
            except Exception as e:
                error(f"加载 settings.yaml 失败: {e}")
        else:
            warning(" settings.yaml 不存在")

        # 2. 加载环境特定配置
        env_file = CONFIG_DIR / f"env/{env}.yaml"
        if env_file.exists():
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    env_config = yaml.safe_load(f) or {}
                    config = self._deep_merge(config, env_config)
                    debug(f"加载 {env_file.name}: {len(self._flatten_dict(env_config))} 个配置项")
            except Exception as e:
                error(f"加载 {env_file} 失败: {e}")

        return config

    def _load_env_config(self) -> Dict[str, Any]:
        """从环境变量加载配置"""
        env_config = {}

        # 遍历所有环境变量
        for env_key, env_value in os.environ.items():
            if env_value:
                # 转换为小写并分割（因为 case_sensitive=False）
                key_parts = env_key.lower().split('__')

                # 跳过不相关的环境变量
                if len(key_parts) < 2:  # 至少要有两级，如 llm__default
                    continue

                # 构建嵌套字典
                current = env_config
                for i, part in enumerate(key_parts[:-1]):
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # 设置值
                last_part = key_parts[-1]

                # 类型转换
                if env_value.lower() in ('true', 'false'):
                    current[last_part] = env_value.lower() == 'true'
                elif env_value.isdigit():
                    current[last_part] = int(env_value)
                else:
                    try:
                        # 尝试转换为浮点数
                        float_val = float(env_value)
                        current[last_part] = float_val
                    except ValueError:
                        # 保持字符串
                        current[last_part] = env_value

        return env_config

    def _deep_copy(self, data: Any) -> Any:
        """深拷贝"""
        if isinstance(data, dict):
            return {k: self._deep_copy(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deep_copy(item) for item in data]
        else:
            return data

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并配置"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _merge_env_into_config(self, config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """将环境变量配置合并到主配置中"""
        result = config.copy()

        for key, value in env_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_env_into_config(result[key], value)
            else:
                result[key] = value

        return result

    def _flatten_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """展平字典用于统计"""
        items = {}
        for k, v in d.items():
            if isinstance(v, dict):
                items.update({f"{k}.{subk}": subv for subk, subv in self._flatten_dict(v).items()})
            else:
                items[k] = v
        return items


# ==================== 主配置类 ====================
class Settings(BaseSettings):
    """主配置 - 优先级: 环境变量 > YAML > 模型默认值"""

    # 基础配置
    app: AppConfig = Field(default_factory=AppConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    # AI 模型配置
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embed: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    # 业务配置
    storyboard: StoryboardConfig = Field(default_factory=StoryboardConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    model_config = SettingsConfigDict(
        case_sensitive=False,  # 大小写不敏感
        env_file_encoding="utf-8",
        env_file=str(ENV_FILE),  # 明确指定.env文件路径
        extra="ignore",
        env_prefix="",  # 清除前缀
        env_ignore_empty=True,
        env_nested_delimiter="__",  # 使用双下划线表示嵌套
    )

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        自定义配置源加载顺序: 优先级链: YAML(settings.yaml → {env}.yaml) → .env → 系统env → init
        1. YAML 配置（基础默认值）
        2. 环境变量（系统环境变量 + .env 文件，覆盖 YAML）
        3. 初始化参数（代码中传入的 kwargs，最高优先级）
        """
        return (
            UnifiedConfigSource(settings_cls),  # 统一的配置源
            env_settings,  # 系统环境变量（覆盖）
            init_settings,  # 初始化参数（最高）
        )

    def get_llm_config(self, provider: str = "default") -> LLMProviderConfig:
        """获取LLM配置（安全返回，不暴露 SecretStr 原始值）"""
        return self.llm.fallback if provider == "fallback" else self.llm.default

    def get_embedding_config(self, provider: str = "default") -> EmbeddingProviderConfig:
        """获取嵌入模型配置"""
        return self.embedding.fallback if provider == "fallback" else self.embedding.default

    def get_config_summary(self) -> Dict[str, Any]:
        """获取当前配置摘要（用于日志/调试）"""
        return {
            "environment": self.app.environment,
            "language": self.app.language,
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "reload": self.api.reload,
            },
            "llm": {
                "default_model": self.llm.default.model_name,
                "fallback_model": self.llm.fallback.model_name if self.llm.fallback else None,
            },
            "embedding": {
                "default_model": self.embedding.default.model_name,
                "fallback_model": self.embedding.fallback.model_name if self.embedding.fallback else None,
            },
            "storyboard": {
                "max_shots": self.storyboard.structure.max_shots,
                "default_style": self.storyboard.generation.default_style,
            },
        }

    def get_supported_styles(self) -> list:
        """
        获取支持的风格列表
        """
        return self.storyboard.generation.supported_styles

    def get_data_paths(self) -> Dict[str, str]:
        """
        获取数据路径配置
        """
        paths_config = self.PathsConfig()
        app_root = PathResolver.get_project_root()

        # 确保路径是绝对路径
        data_paths = {}
        for key, path in paths_config.items():
            if path and not os.path.isabs(path):
                data_paths[key] = os.path.join(app_root, path)
            else:
                data_paths[key] = path

        return data_paths


# ==================== 全局配置实例 ====================
settings = Settings()

# ==================== 调试辅助 ====================

if __name__ == "__main__":
    # 打印配置摘要（调试用）

    print("✅ 配置加载成功")
    print(f"🌍 环境: {settings.app.environment}")
    print(f"🌐 API: http://{settings.api.host}:{settings.api.port}{settings.api.docs_url}")
    print(f"🤖 LLM: {settings.llm.default.model_name}")
    print(f"🤖 LLM fallback: {settings.llm.fallback.model_name}")
    print(f"🧠 Embedding: {settings.embed.default.model_name}")
    print(f"🧠 Embedding fallback: {settings.embed.fallback.device}")
    # print("\n📋 完整配置摘要:")
    # print(json.dumps(settings.get_config_summary(), indent=2, ensure_ascii=False))
