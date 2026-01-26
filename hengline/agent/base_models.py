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
class DifficultyLevel(str, Enum):
    """生成难度等级"""
    LOW = "low"  # 低难度
    MEDIUM = "medium"  # 中难度
    HIGH = "high"  # 高难度
    EXTREME = "extreme"  # 极高难度


class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "low"  # 低风险
    MEDIUM = "medium"  # 中风险
    HIGH = "high"  # 高风险
    CRITICAL = "critical"  # 极高风险


class MotionType(str, Enum):
    """运动类型"""
    STATIC = "static"  # 静态
    SIMPLE = "simple"  # 简单运动
    COMPOUND = "compound"  # 复合运动
    COMPLEX = "complex"  # 复杂运动
    CONTINUOUS = "continuous"  # 连续运动


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
