"""
@FileName: temporal_planner_model.py
@Description: 时序规划模型定义模块
@Author: HengLine
@Time: 2025/12/20 18:35
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

class ContentType(str, Enum):
    DIALOGUE_INTIMATE = "dialogue_intimate"  # 亲密对话场景
    ACTION_FAST = "action_fast"  # 快速动作场景
    EMOTIONAL_REVEAL = "emotional_reveal"  # 情感揭示时刻
    ESTABLISHING_SHOT = "establishing_shot"  # 场景建立镜头
    TRANSITION_SCENE = "transition_scene"  # 过渡场景
    MONTAGE = "montage"  # 蒙太奇序列
    FLASHBACK = "flashback"  # 闪回场景


@dataclass
class DurationEstimation:
    """时长估算结果"""
    element_id: str
    element_type: str  # "dialogue" | "action" | "description"
    estimated_duration: float  # 单位：秒
    confidence: float  # 置信度 0-1
    factors_considered: List[str] = field(default_factory=list)
    adjustment_notes: str = ""


"""
复杂度判断标准：
    simple（简单）:
    - 单个主体
    - 静态或简单移动
    - 单一背景
    - 示例：人物坐着说话
    
    medium（中等）:
    - 1-2个主体
    - 中等移动
    - 背景有细节
    - 示例：两人边走边谈
    
    complex（复杂）:
    - 多个主体（3+）
    - 复杂互动
    - 环境交互
    - 示例：人群中的追逐
    
    very_complex（非常复杂）:
    - 大量细节
    - 快速变化
    - 特效需求
    - 示例：爆炸战斗场景
    
动作强度，标度参考：
    1.0：静态对话、坐着谈话、平静状态
    1.5：缓慢走动、手势交流、轻度动作
    2.0：快走、激动对话、中等动作
    2.5：奔跑、打斗、快速追逐
    3.0：激烈战斗、极限运动、高速追逐
"""
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

    # 新增的属性
    content_type: Optional[str] = "dialogue_intimate"  # 识别片段的核心内容类型 ContentType
    # "happy" | "sad" | "tense" | "romantic" | "melancholy" | "excited" | "fearful" | "neutral" | "dreamy" | "suspenseful"
    emotional_tone: Optional[str] = "neutral"  # 定义片段的情绪氛围，影响色彩、灯光和音乐（间接）
    action_intensity: float = 1.0   # 量化片段的动作强度，从1.0（正常）到3.0（激烈）
    shot_complexity: str = "medium" # 评估单个5秒片段内部的视觉复杂度 （"simple" | "medium" | "complex" | "very_complex"）

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
