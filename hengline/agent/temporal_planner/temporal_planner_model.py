"""
@FileName: temporal_planner_model.py
@Description: 时序规划模型定义模块
@Author: HengLine
@Time: 2025/12/20 18:35
"""
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, unique
from typing import List, Dict, Any, Optional, Tuple


@unique
class ElementType(Enum):
    SCENE = "scene"
    DIALOGUE = "dialogue"
    ACTION = "action"
    TRANSITION = "transition"
    SILENCE = "silence"


@unique
class ContentType(Enum):
    """片段核心内容类型"""
    DIALOGUE_INTIMATE = "dialogue_intimate"  # 亲密对话
    DIALOGUE_CONFLICT = "dialogue_conflict"  # 冲突对话
    DIALOGUE_EXPOSITION = "dialogue_exposition"  # 交代信息
    ACTION_SIMPLE = "action_simple"  # 简单动作
    ACTION_COMPLEX = "action_complex"  # 复杂动作
    SCENE_ESTABLISHING = "scene_establishing"  # 场景建立
    SCENE_ATMOSPHERIC = "scene_atmospheric"  # 氛围营造
    TRANSITION = "transition"  # 转场
    CLIMACTIC = "climactic"  # 高潮时刻
    SILENT_EMOTIONAL = "silent_emotional"  # 沉默情感时刻
    REACTION_SHOT = "reaction_shot"  # 反应镜头


@unique
class EmotionalTone(Enum):
    """情绪氛围类型（影响色彩、灯光、音乐）"""
    HAPPY = "happy"  # 快乐
    SAD = "sad"  # 悲伤
    TENSE = "tense"  # 紧张
    ROMANTIC = "romantic"  # 浪漫
    MELANCHOLY = "melancholy"  # 忧郁
    EXCITED = "excited"  # 兴奋
    FEARFUL = "fearful"  # 恐惧
    NEUTRAL = "neutral"  # 中性
    DREAMY = "dreamy"  # 梦幻
    SUSPENSEFUL = "suspenseful"  # 悬疑
    LONELY = "lonely"  # 孤独
    NOSTALGIC = "nostalgic"  # 怀旧


@unique
class ShotComplexity(Enum):
    """视觉复杂度评估"""
    SIMPLE = "simple"  # 简单：单一主体，固定镜头
    MEDIUM = "medium"  # 中等：2-3个元素，简单运动
    COMPLEX = "complex"  # 复杂：多元素，相机运动，焦点变化
    VERY_COMPLEX = "very_complex"  # 非常复杂：多重运动，特效，复杂构图


class EstimationConfidence(Enum):
    """估算置信度级别"""
    HIGH = "high"  # 置信度 > 0.8
    MEDIUM = "medium"  # 置信度 0.5-0.8
    LOW = "low"  # 置信度 < 0.5


@dataclass
class DurationEstimation:
    """时长估算结果"""
    element_id: str
    element_type: ElementType
    original_duration: float  # 解析器原始估算
    estimated_duration: float  # 混合模型调整后
    # min_duration: float
    # max_duration: float
    confidence: float  # 置信度 0-1

    # 详细分解
    reasoning_breakdown: Dict[str, Any] = field(default_factory=dict)   # 估算依据
    duration_breakdown: Dict[str, float] = field(default_factory=dict)  # 时长分解
    visual_hints: Dict[str, Any] = field(default_factory=dict)  # 视觉线索
    key_factors: List[str] = field(default_factory=list)    # 关键影响因素
    pacing_notes: str = ""      # 节奏调整说明

    # 上下文因子（为节奏分析和分镜生成）
    emotional_weight: float = 1.0  # 情绪权重 (沉默、高潮>1.0)
    visual_complexity: float = 1.0  # 视觉复杂度
    pacing_factor: float = 1.0  # 节奏因子 (快速>1.0, 慢速<1.0)

    # 为智能体3准备的状态信息
    character_states: Dict[str, str] = field(default_factory=dict)  # 角色状态变化
    prop_states: Dict[str, str] = field(default_factory=dict)  # 道具状态变化
    continuity_requirements: Dict[str, Any] = field(default_factory=dict)  # 连续性信息
    shot_suggestions: List[str] = field(default_factory=list)
    emotional_trajectory: List[Dict] = field(default_factory=list)

    # 来源跟踪
    rule_based_estimate: Optional[float] = None  # 规则估算值（如果有）
    ai_based_estimate: Optional[float] = None  # AI估算值（如果有）
    adjustment_reason: str = ""     # 调整原因说明
    estimated_at: str = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "original_duration": round(self.original_duration, 2),
            "estimated_duration": round(self.estimated_duration, 2),
            "confidence": round(self.confidence, 2),
            "rule_based_estimate": round(self.rule_based_estimate, 2) if self.rule_based_estimate is not None else None,
            "ai_based_estimate": round(self.ai_based_estimate, 2) if self.ai_based_estimate is not None else None,
            "adjustment_reason": self.adjustment_reason,
            "emotional_weight": round(self.emotional_weight, 2),
            "visual_complexity": round(self.visual_complexity, 2),
            "pacing_factor": round(self.pacing_factor, 2),
            "character_states": self.character_states,
            "prop_states": self.prop_states,
            "reasoning_breakdown": self.reasoning_breakdown,
            "duration_breakdown": self.duration_breakdown,
            "emotional_trajectory": self.emotional_trajectory,
            "shot_suggestions": self.shot_suggestions,
            "visual_hints": self.visual_hints,
            "key_factors": self.key_factors,
            "pacing_notes": self.pacing_notes,
            "estimated_at": self.estimated_at
        }

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
    """5秒视频片段（核心输出单元）"""
    segment_id: str  # 如 "seg_001"
    time_range: Tuple[float, float]  # (开始时间, 结束时间)
    duration: float = 5.0  # 固定5秒

    # 内容组成
    visual_summary: str = None  # 视觉内容摘要（给分镜生成）
    contained_elements: List[Dict[str, Any]] = None  # 包含的元素ID与类型

    # === 新增属性：为后续智能体提供高级信号 ===
    content_type: Optional[ContentType] = None  # 核心内容类型识别
    emotional_tone: Optional[EmotionalTone] = EmotionalTone.NEUTRAL  # 情绪氛围
    action_intensity: float = 1.0  # 动作强度 1.0-3.0
    shot_complexity: ShotComplexity = ShotComplexity.MEDIUM  # 视觉复杂度

    # === 为智能体3准备的连贯性锚点 ===
    start_anchor: Dict[str, Any] = field(default_factory=dict)  # 开始状态约束
    end_anchor: Dict[str, Any] = field(default_factory=dict)  # 结束状态约束
    continuity_requirements: List[str] = field(default_factory=list)  # 必须保持的连续性

    # === 为智能体4准备的生成提示 ===
    shot_type_suggestion: str = ""  # 镜头类型建议
    lighting_suggestion: str = ""  # 灯光建议
    focus_elements: List[str] = field(default_factory=list)  # 焦点元素

    # === 新增：技术参数建议 ===
    camera_movement: str = "static"  # 相机运动：static, slow_pan, dolly_in, etc.
    color_palette_suggestion: str = ""  # 色彩调色板建议
    sound_design_hint: str = ""  # 音效设计提示

    # === 新增：风格一致性标记 ===
    style_consistency_tags: List[str] = field(default_factory=list)  # 风格标签

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典（用于JSON输出）"""
        return {
            "segment_id": self.segment_id,
            "time_range": self.time_range,
            "duration": self.duration,
            "visual_summary": self.visual_summary,
            "contained_elements": self.contained_elements,
            "content_type": self.content_type.value if self.content_type else None,
            "emotional_tone": self.emotional_tone.value,
            "action_intensity": self.action_intensity,
            "shot_complexity": self.shot_complexity.value,
            "start_anchor": self.start_anchor,
            "end_anchor": self.end_anchor,
            "continuity_requirements": self.continuity_requirements,
            "shot_type_suggestion": self.shot_type_suggestion,
            "lighting_suggestion": self.lighting_suggestion,
            "focus_elements": self.focus_elements,
            "camera_movement": self.camera_movement,
            "color_palette_suggestion": self.color_palette_suggestion,
            "sound_design_hint": self.sound_design_hint,
            "style_consistency_tags": self.style_consistency_tags
        }


@dataclass
class ContinuityAnchor:
    """连续性锚点"""
    anchor_id: str
    type: str  # "visual_match" | "transition" | "keyframe" | "character_state"
    from_segment: str
    to_segment: str
    time_point: float  # 时间点（秒）
    description: str
    priority: int  # 1-10，越高越重要

    # 可选的特定字段
    required_elements: List[str] = field(default_factory=list)
    prohibited_elements: List[str] = field(default_factory=list)
    timestamp: Optional[float] = None  # 对于keyframe类型
    transition_type: str = "cut"  # cut | dissolve | fade | match
    requirements: Dict[str, Any] = field(default_factory=dict)
    visual_constraints: Dict[str, Any] = field(default_factory=dict)
    character_continuity: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PacingAnalysis:
    """节奏分析"""
    pacing_profile: str     # 节奏类型描述
    intensity_curve: []    # 强度曲线描述
    emotional_arc: List[Dict]      # 情绪弧线描述
    visual_complexity: List[float]  # 视觉复杂度描述
    key_points: List[Dict]         # 关键节点描述
    recommendations: List[str]    # 优化建议描述
    analysis_notes: str     # 额外分析备注
    statistics: dict[str, float]    # 统计数据


@dataclass
class TimelinePlan:
    """时序规划结果"""
    timeline_segments: List[TimeSegment]  # 5秒时间片段
    duration_estimations: Dict[str, DurationEstimation]  # 每个元素的时长估算
    continuity_anchors: List[ContinuityAnchor]  # 连续性锚点
    pacing_analysis: PacingAnalysis  # 节奏分析
    # quality_metrics: Dict[str, float]  # 质量指标
    # 元数据
    total_duration: float
    segments_count: int
    elements_count: int

    estimations: Dict[str, Any] = field(default_factory=dict)  # 多模型估算结果
    script_summary: Dict[str, float] = field(default_factory=dict)  # 原始剧本摘要
    processing_stats: Dict[str, Any] = field(default_factory=dict)

    # 为分镜生成准备的全局参数
    global_visual_style: str = ""
    dominant_emotion: str = ""
    key_transition_points: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """转换为字典（优化版）"""
        return {
            "meta": {
                "total_duration": round(self.total_duration, 2),
                "segments_count": self.segments_count,
                "elements_count": self.elements_count,
                "generated_at": datetime.now().isoformat()
            },
            "segments": [seg.to_dict() for seg in self.timeline_segments],
            "estimations_summary": self._create_estimations_summary(),
            "pacing_analysis": self.pacing_analysis,
            "continuity_anchors": self.continuity_anchors
        }

    def _create_estimations_summary(self) -> Dict:
        """创建估算摘要"""
        summary = {
            "total_elements": len(self.duration_estimations),
            "by_type": defaultdict(int),
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "duration_adjustments": {"increased": 0, "decreased": 0, "unchanged": 0}
        }

        total_original = 0
        total_ai = 0

        for est in self.duration_estimations.values():
            # 按类型统计
            summary["by_type"][est.element_type.value] += 1

            # 置信度分布
            if est.confidence >= 0.8:
                summary["confidence_distribution"]["high"] += 1
            elif est.confidence >= 0.5:
                summary["confidence_distribution"]["medium"] += 1
            else:
                summary["confidence_distribution"]["low"] += 1

            # 时长调整统计
            diff = est.estimated_duration - est.original_duration
            if abs(diff) < 0.1:
                summary["duration_adjustments"]["unchanged"] += 1
            elif diff > 0:
                summary["duration_adjustments"]["increased"] += 1
            else:
                summary["duration_adjustments"]["decreased"] += 1

            total_original += est.original_duration
            total_ai += est.estimated_duration

        summary["total_original_duration"] = round(total_original, 2)
        summary["total_ai_adjusted_duration"] = round(total_ai, 2)
        summary["total_adjustment"] = round(total_ai - total_original, 2)

        return dict(summary)  # 转换回普通dict