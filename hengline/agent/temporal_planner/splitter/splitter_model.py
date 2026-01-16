"""
@FileName: splitter_model.py
@Description: 
@Author: HengLine
@Time: 2026/1/15 0:03
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Any, Dict, Optional, Tuple

from hengline.agent.temporal_planner.temporal_planner_model import TimelinePlan, TimeSegment


@dataclass
class SplitDecision:
    """切割决策"""
    element_id: str
    split_point: float  # 切割点（在元素内的相对时间）
    reason: str  # 切割原因
    quality_score: float = 0.0  # 切割质量得分

    # 切割点特征
    is_natural_boundary: bool = False  # 是否为自然边界
    visual_continuity: bool = True  # 是否保持视觉连贯性
    emotional_continuity: bool = True  # 是否保持情感连贯性


class SplitPriority(Enum):
    """分割优先级"""
    PRESERVE_DIALOGUE = 1      # 优先保持对话完整
    PRESERVE_EMOTIONAL_FLOW = 2 # 优先保持情感流动
    PRESERVE_ACTION_SEQUENCE = 3 # 优先保持动作序列
    PRESERVE_SCENE_CONTINUITY = 4 # 优先保持场景连续性
    BALANCE_DURATION = 5       # 平衡时长

@dataclass
class SplitDecision:
    """分割决策"""
    element_id: str
    split_point: float  # 分割点（秒）
    split_type: str     # "complete"完整保留, "split"分割, "delay"延迟到下段
    reason: str         # 分割原因
    narrative_score: float  # 叙事连贯性评分 (0-1)
    visual_consistency_score: float  # 视觉一致性评分 (0-1)
    quality_score: float  # 综合质量评分 (0-1)


# ==================== 连续性锚点系统（输出给智能体3） ====================

class AnchorType(Enum):
    """锚点类型"""
    CHARACTER_APPEARANCE = "character_appearance"
    CHARACTER_POSITION = "character_position"
    CHARACTER_POSTURE = "character_posture"
    CHARACTER_EXPRESSION = "character_expression"
    CHARACTER_GAZE = "character_gaze"
    PROP_STATE = "prop_state"
    PROP_POSITION = "prop_position"
    ENVIRONMENT_SETTING = "environment_setting"
    ENVIRONMENT_LIGHTING = "environment_lighting"
    SPATIAL_RELATION = "spatial_relation"
    VISUAL_COMPOSITION = "visual_composition"
    TRANSITION_REQUIREMENT = "transition_requirement"
    CAMERA_ANGLE = "camera_angle"


class AnchorPriority(Enum):
    """锚点优先级"""
    ABSOLUTE = 10  # 必须绝对遵守（如角色身份）
    CRITICAL = 9  # 必须遵守（明显不连贯）
    HIGH = 7  # 重要（观众会注意到）
    MEDIUM = 5  # 建议（提升质量）
    LOW = 3  # 可选（细微优化）
    INFORMATIONAL = 1  # 信息性（无强制要求）


class ConstraintType(Enum):
    """约束类型"""
    HARD = "hard"  # 硬约束，必须满足
    SOFT = "soft"  # 软约束，尽量满足
    REFERENCE = "reference"  # 参考性约束


@dataclass
class VisualConstraint:
    """视觉约束定义"""
    constraint_id: str = field(default_factory=lambda: f"vc_{uuid.uuid4().hex[:8]}")
    type: ConstraintType = ConstraintType.HARD
    description: str = ""
    sora_prompt: str = ""

    # 验证条件
    verification_method: str = "visual_inspection"
    tolerance_level: str = "strict"  # strict, moderate, flexible

    # 适用性
    applicable_elements: List[str] = field(default_factory=list)
    temporal_range: Optional[Tuple[float, float]] = None


@dataclass
class CharacterStateSnapshot:
    """角色状态快照"""
    character_name: str
    timestamp: float
    segment_id: str

    # 外观状态
    appearance: Dict[str, str] = field(default_factory=dict)
    clothing: Dict[str, str] = field(default_factory=dict)
    accessories: List[str] = field(default_factory=list)

    # 物理状态
    posture: str = "unknown"
    posture_details: Dict[str, Any] = field(default_factory=dict)
    location: str = "unknown"
    coordinates: Optional[Tuple[float, float, float]] = None

    # 交互状态
    interacting_with: List[str] = field(default_factory=list)
    holding_items: List[str] = field(default_factory=list)

    # 情绪状态
    facial_expression: str = "neutral"
    gaze_direction: str = "forward"
    emotional_state: str = "neutral"
    emotional_intensity: float = 0.5

    # 视觉特征
    visual_signature: str = ""
    key_visual_cues: List[str] = field(default_factory=list)


@dataclass
class PropStateSnapshot:
    """道具状态快照"""
    prop_name: str
    timestamp: float
    segment_id: str

    # 基本状态
    current_state: str = "normal"
    position: str = "unknown"
    position_details: Dict[str, Any] = field(default_factory=dict)
    owner: str = "none"

    # 视觉状态
    visual_description: str = ""
    material_texture: str = ""
    lighting_condition: str = "default"

    # 交互状态
    is_being_used: bool = False
    interaction_type: str = "none"
    interaction_target: str = ""


@dataclass
class EnvironmentSnapshot:
    """环境状态快照"""
    scene_id: str
    timestamp: float
    segment_id: str

    # 环境属性
    location: str = ""
    time_of_day: str = ""
    weather: str = ""
    lighting_condition: str = ""

    # 视觉特征
    color_palette: List[str] = field(default_factory=list)
    visual_atmosphere: str = ""
    key_visual_elements: List[str] = field(default_factory=list)

    # 物理环境
    spatial_dimensions: Dict[str, float] = field(default_factory=dict)
    camera_position: Optional[str] = None


@dataclass
class ContinuityAnchor:
    """连续性锚点"""
    anchor_id: str = field(default_factory=lambda: f"anchor_{uuid.uuid4().hex[:8]}")
    anchor_type: AnchorType
    priority: AnchorPriority
    constraint_type: ConstraintType = ConstraintType.HARD

    # 时间与范围
    creation_time: float = field(default_factory=lambda: datetime.now().timestamp())
    start_time: float = 0.0
    end_time: Optional[float] = None
    source_segment_id: str
    target_segment_ids: List[str] = field(default_factory=list)

    # 约束内容
    constraint_description: str
    sora_compatible_prompt: str
    visual_reference: str = ""
    reference_images: List[str] = field(default_factory=list)

    # 验证与执行
    verification_method: str = "visual_check"
    mandatory: bool = True
    confidence_score: float = 0.9

    # 状态变化
    state_change_before: Optional[Dict[str, Any]] = None
    state_change_after: Optional[Dict[str, Any]] = None

    # 元数据
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class StateTransition:
    """状态过渡定义"""
    transition_id: str = field(default_factory=lambda: f"trans_{uuid.uuid4().hex[:8]}")
    from_state: Dict[str, Any]
    to_state: Dict[str, Any]
    transition_type: str  # "continuous", "abrupt", "gradual"
    transition_duration: float = 0.0
    transition_curve: str = "linear"  # linear, ease_in, ease_out


@dataclass
class AnchoredTimeline:
    """带锚点的时序（输出给智能体3）"""
    timeline_plan: TimelinePlan
    anchored_segments: List[TimeSegment]

    # 锚点系统
    continuity_anchors: List[ContinuityAnchor]
    visual_constraints: Dict[str, VisualConstraint]

    # 状态跟踪
    character_state_history: Dict[str, List[CharacterStateSnapshot]]
    prop_state_history: Dict[str, List[PropStateSnapshot]]
    environment_state_history: Dict[str, List[EnvironmentSnapshot]]

    # 状态过渡
    state_transitions: List[StateTransition]

    # 验证与质量
    continuity_score: float = 0.0
    validation_report: Dict[str, Any] = field(default_factory=dict)
    detected_issues: List[Dict[str, Any]] = field(default_factory=list)

    # 元数据
    generation_timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    generator_version: str = "1.0.0"