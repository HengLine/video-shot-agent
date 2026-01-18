"""
@FileName: script_parser_models.py
@Description: 
@Author: HengLine
@Time: 2026/1/9 21:28
"""
from enum import Enum
from typing import List, Optional, Literal, Set, Dict

from pydantic import BaseModel, Field, model_validator, field_validator

from hengline.agent.temporal_planner.analyzer.action_analyzer import ActionIntensityAnalyzer
from hengline.agent.workflow.workflow_models import ScriptType


# ==============================
# 核心数据模型
# ==============================

class Meta(BaseModel):
    """
    元信息：描述本次解析的上下文
    """
    schema_version: str = Field(default="1.1", description="Schema 版本号")
    time_unit: str = Field(default="seconds", description="时间单位，固定为 seconds")
    source_type: str = Field(default="screenplay_snippet", description="输入源类型")
    target_use: str = Field(default="text-to-video prompt generation", description="目标用途")
    frame_rate_assumption: Optional[str] = Field(default="24fps", description="假设帧率，如 '24fps'")
    script_format: Optional[ScriptType] = Field(default=None, description="推断的输入剧本格式")

# ==============================
# 道具模型
# ==============================
class Prop(BaseModel):
    """
    场景中的道具/环境对象
    """
    name: str = Field(..., description="道具名称，如 '旧羊毛毯'、'手机'")
    description: str = Field(..., description="道具描述")
    owner: str = Field(..., description="持有者")
    state: str = Field(..., description="当前状态，如 '凝出水雾'、'摊开'")
    position: str = Field(..., description="空间位置，如 '茶几上'、'林然肩上'")

# ==============================
# 场景模型
# ==============================
class Scene(BaseModel):
    """
    场景描述单元
    """
    scene_id: str = Field(..., pattern=r"^scene_\d+$", description="场景唯一ID，格式如 scene_1")
    order: int = Field(..., ge=1, description="场景顺序")
    location: str = Field(..., description="地点，如 '城市公寓客厅'")
    time_of_day: str = Field(..., description="时段，如 '深夜'")
    time_exact: Optional[str] = Field(None, description="精确时间，如 '23:00'")
    weather: Optional[str] = Field(None, description="天气，如 '大雨滂沱'")
    mood: Optional[str] = Field(..., description="氛围关键词，如 '孤独紧张'")
    summary: str = Field(..., description="场景摘要")
    description: str = Field(..., description="详细环境描述")
    key_visuals: List[str] = Field(..., description="本场景涉及的关键视觉元素")
    character_refs: List[str] = Field(..., description="本场景涉及的角色名列表")
    start_time: float = Field(..., ge=0, description="场景开始时间（秒）")
    end_time: float = Field(..., ge=0, description="场景结束时间（秒）")
    duration: float = Field(..., gt=0, description="持续时间（秒）")
    props: List[Prop] = Field(default_factory=list, description="场景内道具列表")

    def to_dict(self) -> dict:
        """转换为字典表示"""
        return self.model_dump()

# ==============================
# 角色模型
# ==============================
class Character(BaseModel):
    """
    角色信息
    """
    name: str = Field(..., min_length=1, description="角色唯一标识名，必须与 dialogues/actions 中的引用一致")
    age: Optional[int] = Field(..., description="年龄（如果可知）")
    gender: str = Field(..., description="性别")
    role: str = Field(..., description="角色定位，如 '女主角'")
    appearance: str = Field("", description="外貌细节，仅当剧本明确描述或匹配预设档案时填写")
    personality: Optional[str] = Field("", description="性格特点，仅当剧本明确描述或匹配预设档案时填写")
    state: Dict[str, str] = Field(
        default_factory=dict,
        description="当前状态，如 {'emotional': '震惊', 'physical': '颤抖'}"
    )

    def to_dict(self) -> dict:
        """转换为字典表示"""
        return self.model_dump()

# ==============================
# 对话模型
# ==============================
class Dialogue(BaseModel):
    """
    对话或沉默事件
    """
    dialogue_id: str = Field(..., pattern=r"^dial_\d+$", description="对话唯一ID")
    scene_ref: str = Field(..., description="所属场景ID，必须等于某个 scene.scene_id")
    speaker: Optional[str] = Field(None, description="说话者名字，必须在 characters.name 中；若为沉默则为 null")
    content: str = Field(..., description="台词内容，沉默时为空字符串")
    target: Optional[str] = Field(None, description="对话对象，必须在 characters.name 中；若没有则为 null")
    emotion: Optional[str] = Field(None, description="情绪标签")
    intensity: float = Field(1.0, description="强度 1.0-3.0")
    voice_quality: Optional[str] = Field(None, description="声音特质，如 '沙哑'、'低沉'")
    parenthetical: Optional[str] = Field(None, description="剧本中的旁注，如 '声音微颤'")
    type: Literal["speech", "silence"] = Field("speech", description="类型：speech 或 silence")
    time_offset: float = Field(..., ge=0, description="相对于场景开始的时间偏移（秒）")
    duration: Optional[float] = Field(None, gt=0, description="持续时间（秒），可选")
    timestamp: Optional[float] = Field(None, description="时间戳（秒）")

    def to_dict(self) -> dict:
        """转换为字典表示"""
        return self.model_dump()

    @field_validator('type', mode='before')
    @classmethod
    def infer_type_from_speaker(cls, v, info):
        """若未显式指定 type，根据 speaker 是否为 null 推断"""
        if v is None:
            speaker = info.data.get('speaker')
            return "silence" if speaker is None else "speech"
        return v

# ==============================
# 动作模型及枚举
# ==============================
class ActionType(str, Enum):
    """
    动作类型枚举，用于 actions[].type
    确保 LLM 输出的动作分类标准化，便于下游动画控制
    """
    POSTURE = "posture"  # 姿态（如蜷缩、坐直）
    GESTURE = "gesture"  # 手势（如悬停、收紧）
    FACIAL = "facial"  # 面部表情（如瞳孔收缩、流泪）
    GAZE = "gaze"  # 视线行为（如盯看）
    PHYSIOLOGICAL = "physiological"  # 生理反应（如喉头滚动、呼吸停滞）
    INTERACTION = "interaction"  # 与物体交互（如接听电话）
    DEVICE_ALERT = "device_alert"  # 设备触发（如手机震动）
    PROP_FILL = "prop_fill"  # 道具状态填充（较少用）
    PROP_FALL = "prop_fall"  # 道具掉落/滑落
    VOCAL = "vocal"  # 发声尝试（如张嘴无声）
    AUDIO = "audio"  # 环境音（如汽笛声）

class ActionIntensityLevel(Enum):
    """动作强度等级枚举"""
    VERY_LOW = 1.0    # 极低强度：静止、细微变化
    LOW = 1.3         # 低强度：日常动作
    MEDIUM = 1.7      # 中等强度：有目的性动作
    HIGH = 2.2        # 高强度：激烈动作
    VERY_HIGH = 2.8   # 极高强度：极限动作
    EXTREME = 3.0     # 极端强度：生死攸关

class Action(BaseModel):
    """
    原子化动作事件
    """
    action_id: str = Field(..., pattern=r"^act_\w+$", description="动作唯一ID")
    scene_ref: str = Field(..., description="所属场景ID，必须等于某个 scene.scene_id")
    actor: str = Field(..., min_length=1, description="执行者：必须是角色名、道具名或固定实体（如 '手机'）")
    # actors: List[str] = Field(..., description="执行者列表：必须是角色名、道具名或固定实体（如 '手机'）")
    target: Optional[str] = Field(..., description="动作目标")
    type: ActionType = Field(..., description="动作类型，必须为预定义枚举值")
    intensity: ActionIntensityLevel = Field(ActionIntensityLevel.LOW, description="动作强度，如 '轻微'、'剧烈'")
    is_crucial: bool = Field(False, description="是否为关键动作，影响剧情走向")
    description: str = Field(..., description="动作描述")
    time_offset: float = Field(..., ge=0, description="时间偏移（秒）")
    duration: float = Field(..., gt=0, description="持续时间（秒）")
    timestamp: Optional[float] = Field(None, description="时间戳（秒）")

    def to_dict(self) -> dict:
        """转换为字典表示"""
        return self.model_dump()

    def analyze_action(self, context: Dict):
        """分析动作强度和相关属性"""
        analyzer = ActionIntensityAnalyzer()

        # 分析强度
        intensity, breakdown = analyzer.analyze_intensity(
            self.description, self.action_type, context
        )

        self.intensity_score = intensity
        self.intensity_breakdown = breakdown

        # 确定强度等级
        if intensity < 1.3:
            self.intensity_level = "very_low"
        elif intensity < 1.7:
            self.intensity_level = "low"
        elif intensity < 2.2:
            self.intensity_level = "medium"
        elif intensity < 2.8:
            self.intensity_level = "high"
        else:
            self.intensity_level = "very_high"

        # 估算时长
        self.estimated_duration = analyzer.calculate_action_duration(
            self.description, self.action_type, intensity
        )

        # 建议镜头类型
        self.recommended_shot_types = self._get_recommended_shots(intensity)

        # 建议摄像机运动
        self.camera_movement_suggestions = self._get_camera_suggestions(intensity)

    def _get_recommended_shots(self, intensity: float) -> List[str]:
        """根据强度推荐镜头类型"""
        if intensity < 1.5:
            return ["close_up", "extreme_close_up", "static"]
        elif intensity < 2.0:
            return ["medium", "two_shot", "pan"]
        elif intensity < 2.5:
            return ["action_wide", "handheld", "tracking"]
        else:
            return ["extreme_close_up", "slow_motion", "wide"]

    def _get_camera_suggestions(self, intensity: float) -> List[str]:
        """根据强度推荐摄像机运动"""
        if intensity < 1.5:
            return ["static", "slow_pan", "slight_zoom"]
        elif intensity < 2.0:
            return ["pan", "tilt", "dolly_slow"]
        elif intensity < 2.5:
            return ["handheld", "fast_pan", "tracking", "dolly_fast"]
        else:
            return ["crane", "extreme_angle", "rapid_zoom", "shaky_cam"]


# ==============================
# 关系模型
# ==============================
class Relationship(BaseModel):
    """
    角色间关系（用于情感/叙事建模）
    """
    from_: str = Field(..., alias="from", description="关系发起方角色名")
    to: str = Field(..., description="关系接收方角色名")
    type: str = Field(..., description="关系类型，如 '重要过往关系'")
    current: str = Field(..., description="当前状态，如 '久别后联系'")
    implied_history: Optional[str] = Field(None, description="隐含历史背景")


# ==============================
# 根模型 + 跨字段校验
# ==============================

class UnifiedScript(BaseModel):
    """
    完整分镜结构根模型
    包含自定义校验器，确保跨对象引用一致性
    """
    meta: Meta = Field(..., alias="_meta", description="元信息")
    scenes: List[Scene] = Field(..., min_length=1, description="场景列表")
    characters: List[Character] = Field(..., min_length=1, description="角色列表")
    dialogues: List[Dialogue] = Field(..., description="对话/沉默事件列表")
    actions: List[Action] = Field(..., description="动作事件列表")
    relationships: List[Relationship] = Field(default_factory=list, description="角色关系列表")
    original_text: Optional[str] = Field(None, description="原始剧本文本（可选）")
    completeness_score: Optional[float] = Field(None, description="剧本解析评分总分")
    warnings: List[str] = Field(None, description="剧本解析评分建议列表")
    parsing_confidence: Dict[str, float] = Field(None, description="剧本解析置信度")

    def get_scene_by_id(self, scene_id: str) -> Optional[Scene]:
        """根据ID获取场景"""
        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        return None

    def get_dialogues_in_scene(self, scene_id: str) -> List[Dialogue]:
        """获取场景中的所有对话"""
        return [d for d in self.dialogues if d.scene_ref == scene_id]

    def get_actions_in_scene(self, scene_id: str) -> List[Action]:
        """获取场景中的所有动作"""
        return [a for a in self.actions if a.scene_ref == scene_id]

    def get_character_by_name(self, name: str) -> Optional[Character]:
        """根据名称获取角色"""
        for char in self.characters:
            if char.name == name:
                return char
        return None

    @model_validator(mode='after')
    def validate_cross_references(self) -> 'UnifiedScript':
        """
        全局校验器：确保所有引用有效
        """
        # 1. 提取所有合法引用池
        valid_character_names: Set[str] = {char.name for char in self.characters}
        valid_scene_ids: Set[str] = {scene.scene_id for scene in self.scenes}
        valid_prop_names: Set[str] = set()
        for scene in self.scenes:
            valid_prop_names.update(prop.name for prop in scene.props)
        # 允许的固定实体（设备/环境）
        fixed_entities = {"手机", "电视", "环境", "窗户"}

        # 2. 校验 dialogues.speaker
        for dial in self.dialogues:
            if dial.speaker is not None and dial.speaker not in valid_character_names:
                raise ValueError(
                    f"Dialogue speaker '{dial.speaker}' not found in characters. "
                    f"Valid names: {sorted(valid_character_names)}"
                )
            if dial.scene_ref not in valid_scene_ids:
                raise ValueError(f"Dialogue scene_ref '{dial.scene_ref}' not found in scenes.")

        # 3. 校验 actions.actor 和 scene_ref
        valid_actors = valid_character_names | valid_prop_names | fixed_entities
        for act in self.actions:
            if act.actor not in valid_actors:
                raise ValueError(
                    f"Action actor '{act.actor}' is not a valid character, prop, or fixed entity. "
                    f"Valid actors: {sorted(valid_actors)}"
                )
            if act.scene_ref not in valid_scene_ids:
                raise ValueError(f"Action scene_ref '{act.scene_ref}' not found in scenes.")

        # 4. 校验 relationships 中的角色名
        for rel in self.relationships:
            if rel.from_ not in valid_character_names:
                raise ValueError(f"Relationship 'from' role '{rel.from_}' not declared in characters.")
            if rel.to not in valid_character_names:
                raise ValueError(f"Relationship 'to' role '{rel.to}' not declared in characters.")

        return self

    @model_validator(mode='after')
    def validate_time_order(self) -> 'UnifiedScript':
        """
        （可选）校验时间轴合理性：同一场景内事件时间不倒序
        """
        # 按 scene_ref 分组
        from collections import defaultdict
        scene_events: Dict[str, List[tuple[float, str]]] = defaultdict(list)

        for dial in self.dialogues:
            scene_events[dial.scene_ref].append((dial.time_offset, "dialogue"))
        for act in self.actions:
            scene_events[act.scene_ref].append((act.time_offset, "action"))

        for scene_id, events in scene_events.items():
            times = [t for t, _ in sorted(events)]
            if times != sorted(times):
                raise ValueError(f"Time offsets in scene '{scene_id}' are not in chronological order.")

        return self


# ==============================
# 使用示例
# ==============================

if __name__ == "__main__":
    # 示例：从 JSON 字符串加载并验证
    sample_json_str = '''
    {
      "_meta": {
        "schema_version": "1.1",
        "time_unit": "seconds",
        "source_type": "screenplay_snippet",
        "target_use": "text-to-video prompt generation",
        "input_format": "natural_language"
      },
      "scenes": [
        {
          "scene_id": "scene_1",
          "order": 1,
          "location": "城市公寓客厅",
          "time_of_day": "深夜",
          "weather": "大雨滂沱",
          "mood": "孤独紧张",
          "summary": "林然深夜在家接到陈默的神秘来电",
          "description": "深夜11点，城市公寓客厅...",
          "character_refs": ["林然", "陈默"],
          "start_time": 0,
          "end_time": 30,
          "duration": 30,
          "props": [
            {"name": "手机", "state": "亮起‘未知号码’", "position": "茶几上"}
          ]
        }
      ],
      "characters": [
        {"name": "林然", "role": "女主角", "appearance_notes": "齐肩黑发，左侧深紫挑染...", "state": {"emotional": "震惊"}},
        {"name": "陈默", "role": "神秘来电者", "appearance_notes": "", "state": {"emotional": "疲惫"}}
      ],
      "dialogues": [
        {
          "dialogue_id": "dial_1",
          "scene_ref": "scene_1",
          "speaker": "陈默",
          "text": "是我。",
          "emotion": "疲惫沉重",
          "type": "speech",
          "time_offset": 18
        }
      ],
      "actions": [
        {
          "action_id": "act_1",
          "scene_ref": "scene_1",
          "actor": "林然",
          "type": "posture",
          "description": "蜷在沙发里",
          "time_offset": 0,
          "duration": 10
        }
      ],
      "relationships": []
    }
    '''

    try:
        storyboard = UnifiedScript.model_validate_json(sample_json_str)
        print("JSON 验证通过！")
        print(f"场景数: {len(storyboard.scenes)}")
        print(f"角色: {[c.name for c in storyboard.characters]}")
    except Exception as e:
        print(f"验证失败: {e}")
