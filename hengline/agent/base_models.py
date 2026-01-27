"""
@FileName: base_models.py
@Description: 基本模型
@Author: HengLine
@Time: 2026/1/18 14:31
"""
import uuid
from datetime import datetime
from enum import Enum, unique
from typing import Optional

from pydantic import BaseModel, Field


# ==================== 基础枚举和类型定义 ====================
class ScriptType(Enum):
    """剧本格式类型"""
    NATURAL_LANGUAGE = "natural_language"  # 自然语言描述
    AI_STORYBOARD = "ai_storyboard"  # AI分镜脚本
    STRUCTURED_SCENE = "structured_scene"  # 结构化场景描述
    STANDARD_SCRIPT = "standard_script"  # 标准剧本格式
    DIALOGUE_ONLY = "dialogue_only"  # 纯对话
    MIXED_FORMAT = "mixed_format"  # 混合格式


class ElementType(str, Enum):
    SCENE = "scene"  # 场景描述
    DIALOGUE = "dialogue"  # 对话节点
    ACTION = "action"  # 动作节点
    TRANSITION = "transition"
    SILENCE = "silence"
    UNKNOWN = "unknown"


@unique
class AgentMode(Enum):
    """agent 实现的模式"""
    LLM = "llm"  # 基于 LLM 实现
    RULE = "rule"  # 基于本地规则


class VideoStyle(Enum):
    # 逼真
    REALISTIC = 'realistic'
    # 动漫
    ANIME = 'anime'
    # 电影
    CINEMATIC = 'cinematic'
    # 卡通
    CARTOON = 'cartoon'


class AIPlatform(str, Enum):
    """AI平台枚举"""
    RUNWAY_GEN2 = "runway_gen2"  # Runway Gen-2
    PIKA_LABS = "pika_labs"  # Pika Labs
    STABLE_VIDEO = "stable_video"  # Stable Video Diffusion
    LUMAS = "lumas"  # Luma AI
    KAIBER = "kaiber"  # Kaiber
    MOONVALLEY = "moonvalley"  # Moon Valley


@unique
class GenerationStatus(str, Enum):
    """生成状态"""
    PENDING = "pending"  # 等待生成
    GENERATING = "generating"  # 生成中
    SUCCESS = "success"  # 成功
    FAILED = "failed"  # 失败
    NEEDS_RETRY = "needs_retry"  # 需要重试
    NEEDS_MANUAL_FIX = "needs_manual_fix"  # 需要手动修复


# ==================== 基础模型类 ====================
class BaseMetadata(BaseModel):
    """时间戳混合类"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    version: str = Field(default="1.0", description="模型版本")
