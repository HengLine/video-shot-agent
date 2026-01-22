"""
@FileName: shot_generator_models.py
@Description: 模型
@Author: HengLine
@Time: 2026/1/18 14:26
"""
from datetime import datetime
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field

from hengline.agent.base_models import EmotionType, MotionType, ShotType, CameraAngle, CameraMovement, DepthOfField, ColorInfo, BaseMetadata


class CameraMovement(BaseModel):
    """镜头运动描述"""
    movement_type: str = Field(
        default="static",
        description="运动类型：static/pan/tilt/zoom/dolly/track"
    )
    direction: Optional[str] = Field(
        default=None,
        description="方向：left/right/up/down/in/out"
    )
    speed: str = Field(
        default="normal",
        description="速度：slow/normal/fast"
    )
    duration: float = Field(
        default=0.0,
        description="运动持续时间"
    )


class ShotComposition(BaseModel):
    """镜头构图信息"""
    shot_type: str = Field(
        default="medium_shot",
        description="镜头类型：close_up/medium_shot/wide_shot/extreme_wide"
    )
    angle: str = Field(
        default="eye_level",
        description="角度：eye_level/low_angle/high_angle/dutch_angle"
    )
    focus: str = Field(
        default="character",
        description="焦点：character/object/environment"
    )
    depth_of_field: str = Field(
        default="normal",
        description="景深：shallow/normal/deep"
    )


class ShotElement(BaseModel):
    """镜头内的元素引用"""
    element_id: str = Field(..., description="引用的剧本元素ID")
    start_offset: float = Field(
        default=0.0,
        description="在镜头内的开始时间偏移"
    )
    duration: float = Field(
        default=0.0,
        description="在镜头内的持续时间"
    )
    importance: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="在镜头中的重要性权重"
    )


class ShotInfo(BaseModel):
    """镜头信息 - 阶段2输出"""
    id: str = Field(..., description="镜头唯一ID，格式：shot_001")

    # 基本信息
    scene_id: str = Field(..., description="所属场景ID")
    description: str = Field(..., description="镜头内容描述")

    # 时间信息
    start_time: float = Field(default=0.0, description="全局开始时间（秒）")
    end_time: float = Field(default=0.0, description="全局结束时间（秒）")
    duration: float = Field(default=0.0, description="镜头时长（秒）")

    # 视觉信息
    composition: ShotComposition = Field(
        default_factory=ShotComposition,
        description="镜头构图"
    )
    camera_movement: CameraMovement = Field(
        default_factory=CameraMovement,
        description="镜头运动"
    )

    # 内容信息
    main_character: Optional[str] = Field(
        default=None,
        description="主要角色（如有）"
    )
    key_action: Optional[str] = Field(
        default=None,
        description="关键动作描述"
    )

    # 情感和节奏
    emotion_intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="情感强度，0-1之间"
    )
    pace: str = Field(
        default="normal",
        description="节奏：slow/normal/fast"
    )

    # 元素引用
    elements: List[ShotElement] = Field(
        default_factory=list,
        description="镜头包含的剧本元素"
    )

    # 连续性锚点（简化版）
    continuity_anchors: Dict[str, Any] = Field(
        default_factory=lambda: {
            "character_states": {},
            "prop_locations": {},
            "scene_state": {}
        },
        description="镜头开始时的关键状态"
    )


class ShotSequence(BaseModel):
    """镜头序列 - 阶段2输出"""

    # 元数据
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "generated_at": datetime.now().isoformat(),
            "script_title": "Unknown",
            "total_duration": 0.0
        },
        description="序列元数据"
    )

    # 核心数据
    shots: List[ShotInfo] = Field(
        default_factory=list,
        description="镜头列表，按时间顺序排列"
    )

    # 引用信息
    source_script: Dict[str, Any] = Field(
        default_factory=dict,
        description="源剧本的引用信息"
    )

    # 统计数据
    stats: Dict[str, Any] = Field(
        default_factory=lambda: {
            "shot_count": 0,
            "avg_shot_duration": 0.0,
            "emotion_curve": [],
            "pace_changes": 0
        },
        description="镜头序列统计数据"
    )

    # 节奏分析
    rhythm_analysis: Dict[str, Any] = Field(
        default_factory=lambda: {
            "high_points": [],  # 高潮点时间
            "quiet_periods": [],  # 安静段落
            "transitions": []  # 转场点
        },
        description="节奏分析结果"
    )