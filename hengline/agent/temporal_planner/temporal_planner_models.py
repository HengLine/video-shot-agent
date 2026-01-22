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
from typing import List, Dict, Any, Optional, Tuple, Set

@unique
class ElementType(Enum):
    SCENE = "scene"
    DIALOGUE = "dialogue"
    ACTION = "action"
    TRANSITION = "transition"
    SILENCE = "silence"
    UNKNOWN = "unknown"


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


class EstimationSource(Enum):
    """估算来源"""
    AI_LLM = "ai_llm"
    LOCAL_RULE = "local_rule"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


####################################### 时长估算模型 #########################################
@dataclass
class DurationEstimation:
    """时长估算结果"""
    element_id: str
    element_type: ElementType
    original_duration: float  # 解析器原始估算
    estimated_duration: float  # 混合模型调整后

    # 质量指标
    confidence: float = 0.7  # 置信度 0-1
    min_duration: float = 0.0  # 最小可能时长
    max_duration: float = 0.0  # 最大可能时长

    # 来源信息
    llm_estimated: Optional[float] = None  # LLM估算值
    rule_estimated: Optional[float] = None  # 规则估算值
    estimator_source: EstimationSource = EstimationSource.HYBRID
    adjustment_reason: str = ""     # 调整原因说明

    # 详细分解
    reasoning_breakdown: Dict[str, Any] = field(default_factory=dict)   # 估算依据
    duration_breakdown: Dict[str, float] = field(default_factory=dict)  # 时长分解
    key_factors: List[str] = field(default_factory=list)    # 关键影响因素
    pacing_notes: str = ""      # 节奏调整说明

    # 上下文因子（为节奏分析和分镜生成）
    emotional_weight: float = 1.0  # 情绪权重 (沉默、高潮>1.0)
    visual_complexity: float = 1.0  # 视觉复杂度
    pacing_factor: float = 1.0  # 节奏因子 (快速>1.0, 慢速<1.0)

    # 为5秒分片准备的信息
    can_be_split: bool = True  # 是否可以被切割
    split_priority: int = 5  # 切割优先级（1-10，越低越优先不切割）
    key_moment_percentage: float = 0.5  # 关键时刻在元素中的位置（0-1）

    # 为智能体3准备的状态信息
    character_states: Dict[str, str] = field(default_factory=dict)  # 角色状态变化
    prop_states: Dict[str, str] = field(default_factory=dict)  # 道具状态变化
    continuity_requirements: Dict[str, Any] = field(default_factory=dict)  # 连续性信息
    shot_suggestions: List[str] = field(default_factory=list)
    visual_hints: Dict[str, Any] = field(default_factory=dict)  # 视觉线索
    emotional_trajectory: List[Dict] = field(default_factory=list)

    estimated_at: str = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "original_duration": round(self.original_duration, 2),
            "estimated_duration": round(self.estimated_duration, 2),
            "confidence": round(self.confidence, 2),
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "rule_estimated": round(self.rule_estimated, 2) if self.rule_estimated is not None else None,
            "llm_estimated": round(self.llm_estimated, 2) if self.llm_estimated is not None else None,
            "estimator_source": self.estimator_source.value,
            "adjustment_reason": self.adjustment_reason,
            "emotional_weight": round(self.emotional_weight, 2),
            "visual_complexity": round(self.visual_complexity, 2),
            "pacing_factor": round(self.pacing_factor, 2),
            "character_states": self.character_states,
            "prop_states": self.prop_states,
            "reasoning_breakdown": self.reasoning_breakdown,
            "duration_breakdown": self.duration_breakdown,
            "continuity_requirements": self.continuity_requirements,
            "emotional_trajectory": self.emotional_trajectory,
            "shot_suggestions": self.shot_suggestions,
            "visual_hints": self.visual_hints,
            "key_factors": self.key_factors,
            "pacing_notes": self.pacing_notes,
            "estimated_at": self.estimated_at
        }

@dataclass
class ContainedElement:
    """片段中包含的元素信息"""
    element_id: str
    element_type: ElementType
    start_offset: float          # 在片段内的开始时间偏移
    duration: float   # 在片段内的持续时间
    is_partial: bool = False     # 是否为部分元素
    partial_type: str = ""       # "start", "middle", "end"
    element_data: Dict[str, Any] = field(default_factory=dict)  # 原始元素数据


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
    element_coverage: List[str] = field(default_factory=dict)  # 元素ID，用于跟踪片段的内容组成，便于后续分析和状态跟踪
    visual_content: str = None  # 视觉内容摘要（给分镜生成）
    visual_consistency_tags: Set[str] = field(default_factory=list)  # 视觉一致性标签为AnchorGenerator提供视觉匹配的依据，确保环境、时间、天气等的一致性
    narrative_arc: str = None  # 叙事弧描述（给分镜生成），帮助理解片段的叙事目的，用于过渡分析和节奏控制
    contained_elements: List[ContainedElement] = None  # 包含的元素
    segment_type: str = "normal"  # normal, transition, climax, setup
    requires_special_attention: bool = False

    # === 为智能体3准备的连贯性锚点 ===
    start_anchor: Dict[str, Any] = field(default_factory=dict)  # 开始状态约束
    end_anchor: Dict[str, Any] = field(default_factory=dict)  # 结束状态约束
    continuity_requirements: List[str] = field(default_factory=list)  # 必须保持的连续性

    # === 为智能体4准备的生成提示 ===
    shot_type_suggestion: str = ""  # 镜头类型建议
    lighting_suggestion: str = ""  # 灯光建议
    focus_elements: List[str] = field(default_factory=list)  # 焦点元素
    camera_movement: str = "static"  # 相机运动：static, slow_pan, dolly_in, etc.
    color_palette_suggestion: str = ""  # 色彩调色板建议
    sound_design_hint: str = ""  # 音效设计提示

    content_type: Optional[ContentType] = None  # 核心内容类型识别
    emotional_tone: Optional[EmotionalTone] = EmotionalTone.NEUTRAL  # 情绪氛围
    action_intensity: float = 1.0  # 动作强度 1.0-3.0
    shot_complexity: ShotComplexity = ShotComplexity.MEDIUM  # 视觉复杂度

    # 片段质量指标
    pacing_score: float = 0.0  # 节奏得分 0-10
    completeness_score: float = 0.0  # 完整性得分 0-1
    split_quality: float = 0.0  # 切割质量得分 0-1

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典（用于JSON输出）"""
        return {
            "segment_id": self.segment_id,
            "time_range": self.time_range,
            "duration": self.duration,
            "visual_content": self.visual_content,
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
            "pacing_score": self.pacing_score
        }

    @property
    def start_time(self) -> float:
        """片段开始时间"""
        return self.time_range[0]

    @property
    def end_time(self) -> float:
        """片段结束时间"""
        return self.time_range[1]

    def contains_element(self, element_id: str) -> bool:
        """检查是否包含指定元素"""
        return any(elem.element_id == element_id for elem in self.contained_elements)

    def get_element_by_id(self, element_id: str) -> Optional[ContainedElement]:
        """根据ID获取元素"""
        for elem in self.contained_elements:
            if elem.element_id == element_id:
                return elem
        return None

###################################### 5秒片段分隔模型 #########################################
class CharacterState(BaseModel):
    """角色状态模型"""
    pose: str = Field(description="姿势描述")
    gaze_direction: str = Field(description="视线方向")
    facial_expression: EmotionType = Field(description="面部表情")
    props_in_use: List[str] = Field(default_factory=list, description="使用的道具")


class FragmentState(BaseModel):
    """片段状态模型"""
    camera: str = Field(description="摄像机状态")
    character: str = Field(description="角色状态")
    transition_hook: Optional[str] = Field(None, description="转场钩子")


class ContinuityRequirement(BaseModel):
    """连续性要求"""
    match_with_previous: Optional[str] = Field(None, description="与前一镜头匹配")
    prepare_for_next: Optional[str] = Field(None, description="为下一镜头准备")


class FrameLevelPlanning(BaseModel):
    """帧级别规划"""
    keyframes: List[Dict[str, Any]] = Field(default_factory=list, description="关键帧列表")
    motion_curves: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="运动曲线")


class GenerationConstraint(BaseModel):
    """生成约束"""
    max_motion_per_frame: Dict[str, str] = Field(default_factory=dict, description="每帧最大运动量")
    camera_movement_limits: Dict[str, str] = Field(default_factory=dict, description="摄像机运动限制")
    character_movement_limits: Dict[str, str] = Field(default_factory=dict, description="角色运动限制")


class PhysicsValidation(BaseModel):
    """物理验证"""
    momentum_check: Dict[str, Any] = Field(default_factory=dict, description="动量检查")
    cloth_simulation: Dict[str, Any] = Field(default_factory=dict, description="布料模拟")
    lighting_consistency: Dict[str, Any] = Field(default_factory=dict, description="光照一致性")


class Fragment(IdentifiableEntity, TemporalEntity):
    """片段模型（5秒内）"""
    parent_shot_id: str = Field(description="父镜头ID")
    is_split: bool = Field(False, description="是否由拆分产生")
    split_index: Optional[int] = Field(None, ge=1, description="拆分索引")
    total_splits: Optional[int] = Field(None, ge=2, description="总分拆数")

    # 描述信息
    description: str = Field(description="片段描述")
    action_unit: str = Field(description="动作单元")

    # 角色状态
    character_states: Dict[str, CharacterState] = Field(default_factory=dict, description="角色状态")

    # 起始结束状态
    start_state: FragmentState = Field(description="起始状态")
    end_state: FragmentState = Field(description="结束状态")

    # 连续性要求
    continuity_requirements: ContinuityRequirement = Field(default_factory=ContinuityRequirement, description="连续性要求")

    # 音频同步
    audio_sync: Optional[Dict[str, Any]] = Field(None, description="音频同步信息")

    # 新增：帧级别规划
    frame_level_planning: Optional[FrameLevelPlanning] = Field(None, description="帧级别规划")

    # 新增：生成约束
    generation_constraints: Optional[GenerationConstraint] = Field(None, description="生成约束")

    # 新增：物理验证
    physics_validation: Optional[PhysicsValidation] = Field(None, description="物理验证")

    class Config:
        validate_assignment = True

    @validator('duration')
    def validate_fragment_duration(cls, v):
        """验证片段时长不超过5秒"""
        if v > 5.0:
            raise ValueError("片段时长不能超过5秒")
        if v < 0.3:
            raise ValueError("片段时长不能小于0.3秒")
        return v

    @validator('split_index', 'total_splits')
    def validate_split_fields(cls, v, values, **kwargs):
        """验证拆分字段一致性"""
        if 'is_split' in values and values['is_split']:
            field_name = kwargs['field'].name
            if field_name == 'split_index' and v is None:
                raise ValueError("拆分片段必须指定split_index")
            if field_name == 'total_splits' and v is None:
                raise ValueError("拆分片段必须指定total_splits")
        return v


class FragmentTransition(BaseModel):
    """片段转场"""
    from_fragment: str = Field(description="来源片段ID")
    to_fragment: str = Field(description="目标片段ID")
    transition_type: TransitionType = Field(description="转场类型")
    description: str = Field(description="转场描述")
    duration: float = Field(default=0.5, ge=0, description="转场时长")
    audio_transition: Optional[str] = Field(None, description="音频转场")


class FragmentsModel(BaseMetadata):
    """5秒片段模型（Phase 4 输出）"""
    derived_from: str = Field(description="来源镜头拆分ID")
    adjustments_applied: str = Field(description="应用的连续性预审ID")

    # 技术参数
    max_duration_per_fragment: float = Field(default=5.0, gt=0, description="每片段最大时长")
    frame_rate: int = Field(default=30, gt=0, description="帧率")

    # 数据
    fragments: List[Fragment] = Field(default_factory=list, description="片段列表")
    fragment_transitions: List[FragmentTransition] = Field(default_factory=list, description="片段转场")

    class Config:
        model_type = "fragments_model"

    @property
    def total_fragments(self) -> int:
        """获取总片段数"""
        return len(self.fragments)

    @property
    def total_frames(self) -> int:
        """获取总帧数"""
        total_duration = sum(frag.duration for frag in self.fragments)
        return int(total_duration * self.frame_rate)

    def get_fragment_by_id(self, fragment_id: str) -> Optional[Fragment]:
        """根据ID获取片段"""
        for fragment in self.fragments:
            if fragment.id == fragment_id:
                return fragment
        return None

    def get_fragments_by_shot(self, shot_id: str) -> List[Fragment]:
        """获取镜头的所有片段"""
        return [frag for frag in self.fragments if frag.parent_shot_id == shot_id]
