"""
@FileName: continuity_guardian_model.py
@Description: 
@Author: HengLine
@Time: 2026/1/4 16:38
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any

from hengline.agent.temporal_planner.temporal_planner_model import TimeSegment
from .model.continuity_rule_guardian import GenerationHints
from .model.continuity_state_guardian import CharacterState, PropState, EnvironmentState
from .model.continuity_transition_guardian import KeyframeAnchor, TransitionInstruction
from .model.continuity_visual_guardian import VisualMatchRequirements, SpatialRelation, VisualSignature


class ContinuityLevel(Enum):
    """连续性检查级别"""
    CRITICAL = "critical"  # 关键连续性错误（角色、主要道具变化）
    MAJOR = "major"  # 主要问题（环境、显著特征变化）
    MINOR = "minor"  # 次要问题（细节不一致）
    COSMETIC = "cosmetic"  # 外观问题（视觉风格轻微变化）


class ProcessStage(Enum):
    """处理阶段枚举"""
    ANALYSIS = "analysis"
    STATE_TRACKING = "state_tracking"
    CONSTRAINT_GENERATION = "constraint_generation"
    PHYSICS_VALIDATION = "physics_validation"
    CONTINUITY_VALIDATION = "continuity_validation"
    COLLISION_DETECTION = "collision_detection"
    MOTION_ANALYSIS = "motion_analysis"
    FIX_SUGGESTION = "fix_suggestion"
    LEARNING = "learning"


@dataclass
class PipelineConfig:
    """流程管道配置"""
    # 各阶段开关
    enable_stages: Dict[ProcessStage, bool] = field(default_factory=lambda: {
        ProcessStage.ANALYSIS: True,
        ProcessStage.STATE_TRACKING: True,
        ProcessStage.CONSTRAINT_GENERATION: True,
        ProcessStage.PHYSICS_VALIDATION: True,
        ProcessStage.CONTINUITY_VALIDATION: True,
        ProcessStage.COLLISION_DETECTION: False,  # 默认关闭，需要时开启
        ProcessStage.MOTION_ANALYSIS: False,  # 默认关闭，需要时开启
        ProcessStage.FIX_SUGGESTION: True,
        ProcessStage.LEARNING: True
    })

    # 性能优化参数
    max_processing_time: float = 5.0  # 最大处理时间（秒）
    enable_parallel_processing: bool = False  # 是否启用并行处理
    batch_size: int = 10  # 批处理大小

    # 质量控制参数
    min_confidence_threshold: float = 0.7  # 最小置信度阈值
    enable_adaptive_threshold: bool = True  # 自适应阈值


@dataclass
class HardConstraint:
    """必须遵守的连续性约束"""

    constraint_id: str
    type: str  # "character_appearance" | "prop_state" | "environment" | "spatial"
    priority: int  # 1-10，10为最高

    # 约束内容
    description: str  # 人类可读的描述
    sora_instruction: str  # Sora能理解的指令

    # 作用范围
    applicable_segments: List[str]  # 适用的片段ID
    temporal_range: Optional[Tuple[float, float]]  # 时间范围

    # 验证信息
    is_enforced: bool = True  # 是否强制执行
    verification_method: str = "visual_check"  # 验证方法
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentState:
    """片段边界的状态快照"""

    timestamp: float  # 时间点
    segment_id: str  # 所属片段

    # 角色状态
    character_states: Dict[str, CharacterState]

    # 道具状态
    prop_states: Dict[str, PropState]

    # 环境状态
    environment_state: EnvironmentState

    # 空间关系
    spatial_relations: List[SpatialRelation]

    # 视觉特征
    visual_signatures: List[VisualSignature]


@dataclass
class AnchoredSegment:
    """带强约束的5秒片段"""

    # 基础信息
    base_segment: TimeSegment  # 原始时间片段
    segment_id: str  # 片段ID

    # 强制性约束（必须遵守）
    hard_constraints: List[HardConstraint]  # 硬约束

    # 视觉匹配要求
    visual_match_requirements: VisualMatchRequirements

    # 状态边界条件
    start_state: SegmentState  # 片段开始时的状态
    end_state: SegmentState  # 片段结束时的状态

    # 关键帧定义
    keyframes: List[KeyframeAnchor]  # 片段内部的关键帧

    # 过渡指令
    transition_to_next: TransitionInstruction  # 如何连接到下一片段

    # 生成提示
    generation_hints: GenerationHints  # 给Sora的生成提示
