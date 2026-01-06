"""
@FileName: data_models.py
@Description: 所有数据类模型
@Author: HengLine
@Time: 2026/1/5 23:04
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from hengline.agent.continuity_guardian.continuity_guardian_model import AnchoredSegment, SegmentState
from hengline.agent.continuity_guardian.model.continuity_rule_guardian import ContinuityRuleSet


# ==================== 输入模型 ====================
@dataclass
class ContinuityAnchoredInput:
    """智能体3的完整输出"""
    anchored_segments: List[AnchoredSegment]
    continuity_rules: ContinuityRuleSet
    state_snapshots: Dict[str, SegmentState]
    validation_report: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== 输出模型 ====================
@dataclass
class VisualEffect:
    """视觉特效"""
    effect_type: str  # "lens_flare", "film_grain", "vignette", "bloom"
    intensity: float
    parameters: Dict[str, Any]


@dataclass
class TechnicalSettings:
    """技术设置"""
    resolution: str = "1920x1080"
    aspect_ratio: str = "16:9"
    framerate: int = 24
    bit_depth: int = 10
    color_space: str = "rec709"
    render_engine: str = "sora_v2"
    seed_value: Optional[int] = None
    cfg_scale: float = 7.5
    steps: int = 50
    sampler: str = "ddim"
    denoising_strength: float = 0.7
    upscale_factor: int = 1


@dataclass
class GenerationMetadata:
    """生成元数据"""
    generator_version: str
    processing_time: float
    total_segments: int
    total_shots: int
    constraint_satisfaction_rate: float
    style_consistency_score: float
    visual_appeal_score: float
    warnings: List[str]
    suggestions: List[str]
