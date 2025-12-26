"""
@FileName: temporal_planner_model.py
@Description: 时序规划模型定义模块
@Author: HengLine
@Time: 2025/12/20 18:35
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class DurationEstimation:
    """时长估算结果"""
    element_id: str
    element_type: str  # "dialogue" | "action" | "description"
    estimated_duration: float  # 单位：秒
    confidence: float  # 置信度 0-1
    factors_considered: List[str] = field(default_factory=list)
    adjustment_notes: str = ""


@dataclass
class TimeSegment:
    """5秒时间片段"""
    segment_id: str
    time_range: Tuple[float, float]  # (开始时间, 结束时间)
    duration: float

    # 内容
    visual_content: str
    audio_content: str
    events: List[str]  # 包含的事件ID

    # 关键元素
    key_elements: List[str]  # 关键视觉/逻辑元素

    # 连续性
    continuity_hooks: Dict[str, Any] = field(default_factory=dict)

    # 节奏
    pacing: str = "normal"  # slow | normal | fast | varying

    # 质量指标
    quality_score: float = 1.0


@dataclass
class ContinuityAnchor:
    """连续性锚点"""
    anchor_id: str
    type: str  # "visual_match" | "transition" | "keyframe" | "character_state"
    from_segment: str
    to_segment: str
    description: str
    priority: int  # 1-10，越高越重要

    # 可选的特定字段
    required_elements: List[str] = field(default_factory=list)
    prohibited_elements: List[str] = field(default_factory=list)
    timestamp: Optional[float] = None  # 对于keyframe类型
    transition_type: str = "cut"  # cut | dissolve | fade | match


@dataclass
class PacingAnalysis:
    """节奏分析"""
    overall_pace: str  # slow | medium | fast
    pace_variation: float  # 节奏变化程度 0-1
    emotional_arc: List[Dict]  # 情绪变化弧线
    key_moments: List[Dict]  # 关键时刻
    recommendations: List[str]  # 节奏优化建议


@dataclass
class TimelinePlan:
    """时序规划结果"""
    timeline_segments: List[TimeSegment]  # 5秒时间片段
    duration_estimations: Dict[str, DurationEstimation]  # 每个元素的时长估算
    continuity_anchors: List[ContinuityAnchor]  # 连续性锚点
    pacing_analysis: PacingAnalysis  # 节奏分析
    quality_metrics: Dict[str, float]  # 质量指标
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelineEvent:
    """时间线事件"""
    type: str  # "dialogue" | "action" | "description" | "scene_start"
    element_id: Optional[str] = None
    scene_id: str = ""
    start_time: float = 0.0
    duration: float = 0.0
    content: str = ""

    # 类型特定字段
    speaker: Optional[str] = None
    actor: Optional[str] = None
    emotion: Optional[str] = None
    intensity: Optional[int] = None