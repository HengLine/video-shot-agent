"""
@FileName: shot_model.py
@Description: 
@Author: HengLine
@Time: 2026/1/17 22:03
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict

from hengline.agent.temporal_planner.temporal_planner_model import EstimationSource


class ShotType(Enum):
    """镜头类型枚举"""
    ESTABLISHING = "establishing"  # 建立镜头
    WIDE = "wide"  # 广角镜头/全景
    MEDIUM = "medium"  # 中景
    CLOSE_UP = "close_up"  # 特写
    EXTREME_CLOSE_UP = "extreme_close_up"  # 大特写
    TWO_SHOT = "two_shot"  # 双人镜头
    GROUP_SHOT = "group_shot"  # 群像镜头
    OVER_THE_SHOULDER = "ots"  # 过肩镜头
    POV = "pov"  # 主观视角
    REVERSE = "reverse"  # 反打镜头
    DOLLY = "dolly"  # 推拉镜头
    PAN = "pan"  # 平移镜头
    TILT = "tilt"  # 俯仰镜头
    ZOOM = "zoom"  # 变焦镜头
    HANDHELD = "handheld"  # 手持镜头
    STATIC = "static"  # 固定镜头
    REACTION = "reaction"  # 反应镜头
    DETAIL = "detail"  # 细节镜头
    INSERT = "insert"  # 插入镜头
    ACTION_WIDE = "action_wide"  # 动作广角
    ACTION_CLOSE = "action_close"  # 动作特写
    FLASHBACK = "flashback"  # 闪回镜头
    MONTAGE = "montage"  # 蒙太奇镜头


class CameraMovement(Enum):
    """摄像机运动类型"""
    NONE = "none"  # 无运动
    PAN_LEFT = "pan_left"  # 左摇
    PAN_RIGHT = "pan_right"  # 右摇
    TILT_UP = "tilt_up"  # 上摇
    TILT_DOWN = "tilt_down"  # 下摇
    DOLLY_IN = "dolly_in"  # 推进
    DOLLY_OUT = "dolly_out"  # 拉出
    TRACKING = "tracking"  # 跟拍
    CRANE_UP = "crane_up"  # 升降上
    CRANE_DOWN = "crane_down"  # 升降下
    ZOOM_IN = "zoom_in"  # 变焦进
    ZOOM_OUT = "zoom_out"  # 变焦出
    HANDHELD = "handheld"  # 手持晃动


class ShotPurpose(Enum):
    """镜头目的"""
    ESTABLISH_LOCATION = "establish_location"  # 建立空间
    SHOW_CHARACTER = "show_character"  # 展示角色
    DIALOGUE = "dialogue"  # 对话
    REACTION = "reaction"  # 反应
    ACTION = "action"  # 动作
    EMOTION = "emotion"  # 情感表达
    TRANSITION = "transition"  # 转场
    REVEAL = "reveal"  # 揭示
    ATMOSPHERE = "atmosphere"  # 氛围渲染
    DETAIL = "detail"  # 细节展示
    RELATIONSHIP = "relationship"
    CONFLICT = "conflict"


@dataclass
class Shot:
    """分镜头对象"""
    shot_id: str
    shot_type: ShotType
    description: str  # 镜头描述
    duration_estimate: float = 0.0  # 估算时长（秒）

    # 镜头属性
    purpose: ShotPurpose = ShotPurpose.SHOW_CHARACTER
    characters: List[str] = field(default_factory=list)  # 镜头中的角色
    key_visuals: List[str] = field(default_factory=list)  # 关键视觉元素
    camera_movement: CameraMovement = CameraMovement.NONE  # 摄像机运动
    framing: str = "normal"  # 构图
    focus: str = "deep"  # 焦点
    angle: str = "eye_level"  # 角度：eye_level, high_angle, low_angle, dutch_angle

    # 情感属性
    emotional_intensity: float = 1.0  # 情感强度 1.0-3.0
    pacing: str = "normal"  # 节奏：slow, normal, fast

    # 对话关联（如果适用）
    dialogue_text: str = ""  # 关联的对话文本
    dialogue_speaker: str = ""  # 说话者

    # 动作关联（如果适用）
    action_description: str = ""  # 动作描述
    action_type: str = ""  # 动作类型

    # 元数据
    sequence_number: int = 0  # 在序列中的序号
    is_transition: bool = False  # 是否是转场镜头
    notes: str = ""  # 备注

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "shot_id": self.shot_id,
            "shot_type": self.shot_type.value,
            "description": self.description,
            "duration_estimate": self.duration_estimate,
            "purpose": self.purpose.value,
            "characters": self.characters,
            "camera_movement": self.camera_movement.value,
            "emotional_intensity": self.emotional_intensity,
            "dialogue_text": self.dialogue_text,
            "dialogue_speaker": self.dialogue_speaker,
            "action_description": self.action_description,
            "sequence_number": self.sequence_number,
            "notes": self.notes
        }


@dataclass
class ShotGenerationResult:
    """分镜头生成结果"""
    scene_id: str
    shots: List[Shot] = field(default_factory=list)
    total_duration: float = 0.0
    shot_count: int = 0
    generation_method: EstimationSource = EstimationSource.FALLBACK  # rule_based, ai_enhanced, hybrid
    confidence: float = 0.0  # 生成置信度
    reasoning: str = ""  # 生成理由说明

    # 统计信息
    shot_type_distribution: Dict[str, int] = field(default_factory=dict)
    character_coverage: Dict[str, int] = field(default_factory=dict)

    # 元数据
    generated_at: str = ""

    def __post_init__(self):
        """初始化后处理"""
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()

        if self.shots:
            self.shot_count = len(self.shots)
            self.total_duration = sum(shot.duration_estimate for shot in self.shots)

            # 统计镜头类型分布
            for shot in self.shots:
                shot_type = shot.shot_type.value
                self.shot_type_distribution[shot_type] = self.shot_type_distribution.get(shot_type, 0) + 1

            # 统计角色覆盖
            for shot in self.shots:
                for char in shot.characters:
                    self.character_coverage[char] = self.character_coverage.get(char, 0) + 1

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "scene_id": self.scene_id,
            "shot_count": self.shot_count,
            "total_duration": self.total_duration,
            "generation_method": self.generation_method.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "shot_type_distribution": self.shot_type_distribution,
            "character_coverage": self.character_coverage,
            "shots": [shot.to_dict() for shot in self.shots],
            "generated_at": self.generated_at
        }