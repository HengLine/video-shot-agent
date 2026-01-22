"""
@FileName: script_parser_models.py
@Description:  剧本解析相关模型
@Author: HengLine
@Time: 2026/1/19 21:44
"""
from typing import List, Dict, Optional, Any, Literal

from pydantic import BaseModel, Field

from hengline.agent.base_models import EmotionType, BaseMetadata
from hengline.agent.workflow.workflow_models import ScriptType


class Dialogue(BaseModel):
    """对话台词"""
    id: str = Field(..., description="台词ID")
    character_id: str = Field(..., description="角色ID")
    emotion: EmotionType = Field(default=EmotionType.NEUTRAL, description="情感类型")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="情感强度")
    duration_estimate: float = Field(default=2.0, gt=0.0, description="预计时长（秒）")
    volume_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="音量级别")
    timing_notes: Optional[str] = Field(None, description="时机说明")
    speaker: Optional[str] = Field(None, description="说话者名字，必须在 characters.name 中；若为沉默则为 null")
    content: str = Field(..., description="台词内容，沉默时为空字符串")
    target: Optional[str] = Field(None, description="对话对象，必须在 characters.name 中；若没有则为 null")
    parenthetical: Optional[str] = Field(None, description="剧本中的旁注，如 '声音微颤'")
    type: Literal["speech", "silence"] = Field("speech", description="类型：speech 或 silence")

    def to_dict(self) -> dict:
        """转换为字典表示"""
        return self.model_dump()


class Action(BaseModel):
    """动作描述"""
    id: str = Field(..., description="动作ID")
    character_id: str = Field(..., description="执行角色ID")
    actors: List[str] = Field(..., description="执行者列表：必须是角色名、道具名或固定实体（如 '手机'）")
    target: Optional[str] = Field(..., description="动作目标")
    # type: ActionType = Field(..., description="动作类型，必须为预定义枚举值")
    description: str = Field(..., description="动作描述文本")
    duration_estimate: float = Field(default=2.0, gt=0.0, description="预计时长（秒）")
    emotion: EmotionType = Field(default=EmotionType.NEUTRAL, description="伴随情感")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="情感强度")
    is_primary: bool = Field(default=True, description="是否主要动作")
    props_used: List[str] = Field(default_factory=list, description="使用的道具")

    def to_dict(self) -> dict:
        """转换为字典表示"""
        return self.model_dump()


class Character(BaseModel):
    """角色信息"""
    id: str = Field(..., description="角色ID")
    name: str = Field(..., description="角色名称")
    role: str = Field(..., description="角色身份/职业")
    gender: str = Field(..., description="性别")
    description: str = Field(..., description="详细描述")
    age_range: Optional[str] = Field(None, description="年龄范围")
    ethnicity: Optional[str] = Field(None, description="族裔")
    key_features: List[str] = Field(default_factory=list, description="关键特征")
    costume_description: Optional[str] = Field(None, description="服装描述")
    emotional_arc: List[EmotionType] = Field(default_factory=list, description="情感弧线")

    def to_dict(self) -> dict:
        """转换为字典表示"""
        return self.model_dump()


class Scene(BaseModel):
    """场景信息"""
    id: str = Field(..., description="场景ID")
    location: str = Field(..., description="场景地点")
    time_period: str = Field(default="daytime", description="时间段")
    weather: Optional[str] = Field(None, description="天气，如 '大雨滂沱'")
    mood: Optional[str] = Field(..., description="氛围关键词，如 '孤独紧张'")
    lighting_condition: str = Field(..., description="光照条件描述")
    emotional_tone: str = Field(..., description="整体情感基调")
    description: Optional[str] = Field(None, description="场景详细描述")
    characters: List[Character] = Field(default_factory=list, description="场景中的角色")
    props: List[str] = Field(default_factory=list, description="场景道具")
    dialogues: List[Dialogue] = Field(default_factory=list, description="对话")
    actions: List[Action] = Field(default_factory=list, description="动作")

    def to_dict(self) -> dict:
        """转换为字典表示"""
        return self.model_dump()


class UnifiedScript(BaseMetadata):
    """
    剧本解析模型 - 第一阶段输出
    将自然语言剧本解析为结构化数据
    """
    total_duration: float = Field(..., gt=0.0, description="总时长估计（秒）")
    total_scenes: int = Field(..., ge=1, description="场景总数")
    total_characters: int = Field(..., ge=0, description="角色总数")
    format_type: ScriptType = Field(..., description="剧本类型")
    scenes: List[Scene] = Field(..., min_length=1, description="所有场景信息")
    completeness_score: Optional[float] = Field(None, description="剧本解析评分总分")

    # 用于第二阶段的数据增强
    pacing_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="节奏分析结果（由智能体分析后填充）"
    )

    emotional_arc: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="情感弧线分析（由智能体分析后填充）"
    )

    generation_feasibility: Optional[Dict[str, Any]] = Field(
        None,
        description="生成可行性评估（由智能体分析后填充）"
    )


    def to_dict(self) -> dict:
        """转换为字典表示"""
        return self.model_dump()