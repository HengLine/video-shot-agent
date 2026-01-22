"""
@FileName: base_models.py
@Description: 基本模型
@Author: HengLine
@Time: 2026/1/18 14:31
"""
import uuid
from datetime import datetime
from enum import StrEnum, Enum, unique
from typing import Optional, Dict

from pydantic import BaseModel, Field


# ==================== 基础枚举和类型定义 ====================
class EmotionType(StrEnum):
    """情感类型枚举"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    EXCITED = "excited"
    FOCUSED = "focused"
    DETERMINED = "determined"
    URGENT = "urgent"
    SHOCKED = "shocked"
    CURIOUS = "curious"


class ShotType(str, Enum):
    """镜头类型枚举"""
    EXTREME_CLOSEUP = "extreme_closeup"  # 大特写
    CLOSEUP = "closeup"  # 特写
    MEDIUM_CLOSEUP = "medium_closeup"  # 中特写
    MEDIUM_SHOT = "medium_shot"  # 中景
    MEDIUM_FULL_SHOT = "medium_full_shot"  # 中全景
    FULL_SHOT = "full_shot"  # 全景
    WIDE_SHOT = "wide_shot"  # 广角
    EXTREME_WIDE_SHOT = "extreme_wide_shot"  # 超广角
    OVER_THE_SHOULDER = "over_the_shoulder"  # 过肩镜头
    POINT_OF_VIEW = "point_of_view"  # 主观镜头


class CameraMovement(str, Enum):
    """摄像机运动枚举"""
    FIXED = "fixed"  # 固定
    PAN_LEFT = "pan_left"  # 左摇
    PAN_RIGHT = "pan_right"  # 右摇
    TILT_UP = "tilt_up"  # 上摇
    TILT_DOWN = "tilt_down"  # 下摇
    DOLLY_IN = "dolly_in"  # 推
    DOLLY_OUT = "dolly_out"  # 拉
    TRACKING_LEFT = "tracking_left"  # 左跟
    TRACKING_RIGHT = "tracking_right"  # 右跟
    CRANE_UP = "crane_up"  # 升
    CRANE_DOWN = "crane_down"  # 降
    HANDHELD = "handheld"  # 手持
    ZOOM_IN = "zoom_in"  # 变焦推
    ZOOM_OUT = "zoom_out"  # 变焦拉


class CameraAngle(str, Enum):
    """摄像机角度枚举"""
    EYE_LEVEL = "eye_level"  # 平视
    LOW_ANGLE = "low_angle"  # 低角度
    HIGH_ANGLE = "high_angle"  # 高角度
    DUTCH_ANGLE = "dutch_angle"  # 荷兰角
    BIRDS_EYE = "birds_eye"  # 鸟瞰
    WORMS_EYE = "worms_eye"  # 虫瞰


class LightingType(str, Enum):
    """灯光类型枚举"""
    NATURAL = "natural"  # 自然光
    STUDIO = "studio"  # 影棚光
    PRACTICAL = "practical"  # 实用光
    BACKLIGHT = "backlight"  # 背光
    SIDELIGHT = "sidelight"  # 侧光
    TOPLIGHT = "toplight"  # 顶光
    UNDERLIGHT = "underlight"  # 底光
    RIM_LIGHT = "rim_light"  # 轮廓光
    KEY_LIGHT = "key_light"  # 主光
    FILL_LIGHT = "fill_light"  # 补光


class DepthOfField(str, Enum):
    """景深枚举"""
    SHALLOW = "shallow"  # 浅景深
    MEDIUM = "medium"  # 中景深
    DEEP = "deep"  # 深景深
    RACK_FOCUS = "rack_focus"  # 变焦


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


class TransitionType(str, Enum):
    """转场类型"""
    CUT = "cut"  # 切
    FADE_IN = "fade_in"  # 淡入
    FADE_OUT = "fade_out"  # 淡出
    DISSOLVE = "dissolve"  # 叠化
    WIPE = "wipe"  # 划变
    MATCH_CUT = "match_cut"  # 匹配剪辑
    JUMP_CUT = "jump_cut"  # 跳切
    MOTION_MATCH = "motion_match"  # 运动匹配
    INVISIBLE = "invisible"  # 无缝转场

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

class ColorInfo(BaseModel):
    """颜色信息"""
    hex_code: str = Field(..., description="十六进制颜色代码，如'#1a237e'")
    name: Optional[str] = Field(None, description="颜色名称")
    rgb: Optional[Dict[str, int]] = Field(None, description="RGB值")
    importance: float = Field(default=1.0, ge=0.0, le=1.0, description="颜色重要性权重")
    usage: str = Field(default="primary", description="使用场景：primary/secondary/accent")


class Vector3(BaseModel):
    """三维向量"""
    x: float = Field(default=0.0, description="X轴值")
    y: float = Field(default=0.0, description="Y轴值")
    z: float = Field(default=0.0, description="Z轴值")

class CameraState(BaseModel):
    """摄像机状态"""
    position: Vector3 = Field(default_factory=Vector3, description="摄像机位置")
    rotation: Vector3 = Field(default_factory=Vector3, description="摄像机旋转（欧拉角）")
    fov: float = Field(default=60.0, ge=30.0, le=120.0, description="视场角")
    focal_length: Optional[float] = Field(None, description="焦距（毫米）")
    aperture: Optional[float] = Field(None, description="光圈值")

class PoseDescription(BaseModel):
    """姿势描述"""
    joint_angles: Optional[Dict[str, Vector3]] = Field(None, description="关节角度")
    gaze_direction: Vector3 = Field(default_factory=Vector3, description="视线方向")
    facial_expression: str = Field(default="neutral", description="面部表情描述")
    body_orientation: Optional[float] = Field(None, ge=0.0, le=360.0, description="身体朝向（度）")
    center_of_gravity: Optional[Vector3] = Field(None, description="重心位置")