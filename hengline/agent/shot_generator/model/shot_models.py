"""
@FileName: shot_models.py
@Description: 镜头相关模型
@Author: HengLine
@Time: 2026/1/5 23:05
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple

from .data_models import VisualEffect, GenerationMetadata, TechnicalSettings
from .style_models import LightingScheme, ColorPalette, StyleGuide
from hengline.agent.temporal_planner.temporal_planner_model import ContentType


class ShotSize(str, Enum):
    EXTREME_CLOSE_UP = "extreme_close_up"
    CLOSE_UP = "close_up"
    MEDIUM_CLOSE_UP = "medium_close_up"
    MEDIUM_SHOT = "medium_shot"
    MEDIUM_FULL_SHOT = "medium_full_shot"
    FULL_SHOT = "full_shot"
    WIDE_SHOT = "wide_shot"
    EXTREME_WIDE_SHOT = "extreme_wide_shot"


class CameraMovement(str, Enum):
    STATIC = "static"
    SLOW_PUSH_IN = "slow_push_in"
    SLOW_PULL_OUT = "slow_pull_out"
    SLOW_PAN = "slow_pan"
    SLOW_DOLLY = "slow_dolly"
    FAST_PUSH_IN = "fast_push_in"
    FAST_PULL_OUT = "fast_pull_out"
    FAST_PAN = "fast_pan"
    FAST_DOLLY = "fast_dolly"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    HANDHELD_SHAKY = "handheld_shaky"
    SMOOTH_TRACKING = "smooth_tracking"
    CIRCULAR_MOVEMENT = "circular_movement"
    CRANE_SHOT = "crane_shot"
    DRONE_SHOT = "drone_shot"
    QUICK_ZOOM = "quick_zoom"
    QUICK_PAN = "quick_pan"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"


@dataclass
class CameraParameters:
    """相机参数"""
    shot_size: ShotSize
    camera_movement: CameraMovement
    lens_focal_length: int  # mm
    aperture: str = "f/2.8"
    framerate: int = 24
    shutter_angle: float = 180.0
    camera_height: str = "eye_level"  # "low_angle", "high_angle", "eye_level"
    camera_distance: str = "medium"  # "close", "medium", "far"
    movement_speed: str = "slow"  # "static", "slow", "medium", "fast"
    movement_pattern: Optional[str] = None
    focus_technique: str = "rack_focus"
    depth_of_field: str = "shallow"


@dataclass
class ShotTransition:
    """镜头过渡"""
    transition_type: str  # "cut", "fade", "dissolve", "wipe"
    duration: float
    style: str = "smooth"
    direction: Optional[str] = None
    timing_curve: str = "ease_in_out"


@dataclass
class GenerationHints:
    """生成提示"""
    emphasis_keywords: List[str]
    avoid_keywords: List[str]
    reference_styles: List[str]
    visual_references: List[str]
    technical_requirements: Dict[str, Any]


@dataclass
class SoraPromptStructure:
    """Sora提示词结构"""
    subject_description: str
    style_enhancement: str
    technical_specs: str

    def compose_full_prompt(self, emphasize_constraints: bool = True) -> str:
        """组合完整提示词"""
        if emphasize_constraints:
            return f"{self.subject_description}. {self.style_enhancement}. {self.technical_specs}"
        else:
            return f"{self.style_enhancement}, featuring {self.subject_description}, {self.technical_specs}"


@dataclass
class SoraShot:
    """Sora镜头指令"""
    shot_id: str
    segment_id: str
    time_range: Tuple[float, float]
    duration: float

    # 提示词部分
    primary_prompt: str
    style_prompt: str
    technical_prompt: str
    full_sora_prompt: str

    # 约束满足记录
    satisfied_constraints: List[str]
    constraint_violations: List[str]
    constraint_compliance_score: float

    # 视觉参数
    camera_parameters: CameraParameters
    lighting_scheme: LightingScheme
    color_palette: ColorPalette

    # 特殊效果
    visual_effects: List[VisualEffect]
    transition_to_next: ShotTransition

    # 生成提示
    generation_hints: GenerationHints

    # 元数据
    content_type: ContentType
    emotional_tone: str
    action_intensity: float
    timestamp_generated: datetime = field(default_factory=datetime.now)
    version: str = "1.0"


@dataclass
class SoraReadyShots:
    """Sora就绪的分镜指令"""
    shot_sequence: List[SoraShot]       # 镜头序列
    technical_settings: TechnicalSettings   # 技术设置
    style_consistency: StyleGuide       # 风格一致性
    constraints_summary: Dict[str, List[str]]   # 约束总结
    generation_metadata: GenerationMetadata     # 生成元数据
    visual_appeal_score: float      # 视觉吸引力评分
    constraint_satisfaction: float  # 约束满足度评分
    timestamp: datetime = field(default_factory=datetime.now)

    def to_json(self) -> Dict[str, Any]:
        """转换为JSON格式"""
        return {
            "metadata": {
                "generated_at": self.timestamp.isoformat(),
                "total_shots": len(self.shot_sequence),
                "total_duration": sum(shot.duration for shot in self.shot_sequence),
                "visual_appeal_score": self.visual_appeal_score,
                "constraint_satisfaction": self.constraint_satisfaction
            },
            "technical_settings": {
                "resolution": self.technical_settings.resolution,
                "framerate": self.technical_settings.framerate,
                "render_engine": self.technical_settings.render_engine
            },
            "style_guide": {
                "visual_theme": self.style_consistency.visual_theme,
                "color_grading": self.style_consistency.color_grading
            },
            "shots": [
                {
                    "shot_id": shot.shot_id,
                    "time_range": shot.time_range,
                    "full_prompt": shot.full_sora_prompt,
                    "camera": {
                        "shot_size": shot.camera_parameters.shot_size.value,
                        "movement": shot.camera_parameters.camera_movement.value
                    }
                }
                for shot in self.shot_sequence
            ]
        }
