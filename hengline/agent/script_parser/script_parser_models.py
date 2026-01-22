"""
@FileName: script_parser_models.py
@Description:  剧本解析相关模型
@Author: HengLine
@Time: 2026/1/19 21:44
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field


class BaseElement(BaseModel):
    """剧本元素基类 - 所有类型元素的共同字段"""
    id: str = Field(..., description="元素唯一标识，格式：elem_001")
    type: Literal["dialogue", "action", "scene_description"] = Field(
        ...,
        description="元素类型：对话/动作/场景描述"
    )
    sequence: int = Field(
        ...,
        description="全局顺序编号，从1开始，表示在剧本中的出现顺序"
    )

    # 通用时间信息
    estimated_duration: float = Field(
        default=3.0,
        description="预估持续时间（秒），基于简单规则估算"
    )

    # 元数据
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="解析置信度，0-1之间"
    )

    class Config:
        extra = "allow"  # 允许额外字段，便于扩展


class DialogueElement(BaseElement):
    """对话元素 - 扩展自ScriptElement"""
    type: Literal["dialogue"] = "dialogue"

    speaker: str = Field(..., description="说话者角色名")
    content: str = Field(..., description="对话内容文本")

    # 对话特有属性
    tone: str = Field(
        default="normal",
        description="语气：normal/angry/happy/sad/whisper/shout"
    )
    volume: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="音量级别，0-1之间"
    )
    target_character: Optional[str] = Field(
        default=None,
        description="对话目标角色（如有）"
    )

    # 覆盖基类的estimated_duration
    estimated_duration: float = Field(
        default_factory=lambda: 2.0,  # 对话默认2秒
        description="对话时长估算，基于字数自动计算"
    )


class ActionElement(BaseElement):
    """动作元素 - 扩展自ScriptElement"""
    type: Literal["action"] = "action"

    actors: List[str] = Field(
        ...,
        description="执行动作的角色列表，至少一个"
    )
    description: str = Field(..., description="动作描述文本")

    # 动作特有属性
    target_object: Optional[str] = Field(
        default=None,
        description="动作目标物体（如有）"
    )
    target_location: Optional[str] = Field(
        default=None,
        description="动作目标位置（如有）"
    )
    emotion: str = Field(
        default="neutral",
        description="伴随情绪：neutral/happy/angry/sad/fear"
    )
    intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="动作强度，0-1之间"
    )


class SceneDescriptionElement(BaseElement):
    """场景描述元素 - 扩展自ScriptElement"""
    type: Literal["scene_description"] = "scene_description"

    content: str = Field(..., description="场景描述文本")

    # 场景特有属性
    time_of_day: str = Field(
        default="day",
        description="时间：day/night/dusk/dawn"
    )
    weather: Optional[str] = Field(
        default=None,
        description="天气：sunny/rainy/cloudy/snowy"
    )
    lighting: str = Field(
        default="normal",
        description="光照：normal/dim/bright/dark"
    )
    mood: str = Field(
        default="neutral",
        description="氛围：neutral/tense/calm/mysterious"
    )


class CharacterInfo(BaseModel):
    """角色信息模型"""
    name: str = Field(..., description="角色名称")
    description: str = Field(
        default="",
        description="角色描述：外貌、特征等"
    )
    initial_costume: Optional[str] = Field(
        default=None,
        description="初始服装描述（如有）"
    )
    key_props: List[str] = Field(
        default_factory=list,
        description="关键道具列表"
    )
    voice_type: Optional[str] = Field(
        default=None,
        description="声音类型：male/female/child/elderly"
    )


class SceneInfo(BaseModel):
    """场景信息模型"""
    id: str = Field(..., description="场景唯一ID，格式：scene_001")
    location: str = Field(..., description="场景地点")
    description: str = Field(..., description="场景详细描述")

    # 时间信息
    time_of_day: str = Field(default="day", description="场景时间")
    estimated_duration: float = Field(
        default=0.0,
        description="场景预估总时长"
    )

    # 场景元素引用
    element_ids: List[str] = Field(
        default_factory=list,
        description="本场景包含的元素ID列表"
    )


class ParsedScript(BaseModel):
    """剧本解析结果 - 阶段1输出"""

    # 元数据
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "parsed_at": datetime.now().isoformat(),
            "model_version": "parser_v1.0",
            "source_type": "unknown"
        },
        description="解析元数据"
    )

    # 核心数据
    title: Optional[str] = Field(
        default=None,
        description="剧本标题（如能识别）"
    )
    characters: List[CharacterInfo] = Field(
        default_factory=list,
        description="角色列表"
    )
    scenes: List[SceneInfo] = Field(
        default_factory=list,
        description="场景列表"
    )

    # 扁平化的元素数组（按sequence排序）
    elements: List[BaseElement] = Field(
        default_factory=list,
        description="所有剧本元素，按出现顺序排列"
    )

    # 统计数据
    stats: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total_elements": 0,
            "total_duration": 0.0,
            "dialogue_count": 0,
            "action_count": 0
        },
        description="解析统计数据"
    )

    # 质量标记
    quality_flags: Dict[str, Any] = Field(
        default_factory=lambda: {
            "parsing_confidence": 0.8,
            "needs_review": False,
            "order_issues": []
        },
        description="质量标记和问题列表"
    )

    class Config:
        # 允许任意类型，便于序列化
        arbitrary_types_allowed = True

    def to_dict(self) -> dict:
        """转换为字典表示"""
        return self.model_dump()
