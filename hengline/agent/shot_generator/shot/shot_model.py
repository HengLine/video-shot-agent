"""
@FileName: shot_model.py
@Description: 
@Author: HengLine
@Time: 2026/1/17 22:03
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional


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
    description: str
    duration_estimate: float = 0.0
    purpose: ShotPurpose = ShotPurpose.SHOW_CHARACTER
    scene_id: str = ""

    # 内容
    characters: List[str] = field(default_factory=list)
    key_visuals: List[str] = field(default_factory=list)
    dialogue_id: Optional[str] = None
    action_id: Optional[str] = None

    # 技术参数
    camera_movement: CameraMovement = CameraMovement.NONE
    framing: str = "normal"
    focus: str = "deep"
    angle: str = "eye_level"

    # 情感与节奏
    emotional_intensity: float = 1.0
    pacing: str = "normal"

    # 元数据
    sequence_number: int = 0
    is_transition: bool = False
    notes: str = ""
    timestamp_in_scene: float = 0.0  # 在场景中的时间位置

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "shot_id": self.shot_id,
            "shot_type": self.shot_type.value,
            "description": self.description,
            "duration_estimate": self.duration_estimate,
            "purpose": self.purpose.value,
            "scene_id": self.scene_id,
            "characters": self.characters,
            "dialogue_id": self.dialogue_id,
            "action_id": self.action_id,
            "camera_movement": self.camera_movement.value,
            "emotional_intensity": self.emotional_intensity,
            "pacing": self.pacing,
            "sequence_number": self.sequence_number,
            "timestamp_in_scene": self.timestamp_in_scene,
            "notes": self.notes
        }


@dataclass
class SceneShotResult:
    """场景分镜结果"""
    scene_id: str
    scene_description: str
    shots: List[Shot] = field(default_factory=list)
    total_duration: float = 0.0
    shot_count: int = 0

    # 统计
    shot_type_distribution: Dict[str, int] = field(default_factory=dict)
    character_coverage: Dict[str, int] = field(default_factory=dict)

    # 元数据
    generation_method: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()

        if self.shots:
            self.shot_count = len(self.shots)
            self.total_duration = sum(s.duration_estimate for s in self.shots)

            # 排序镜头
            self.shots.sort(key=lambda x: x.sequence_number)

            # 统计
            for shot in self.shots:
                shot_type = shot.shot_type.value
                self.shot_type_distribution[shot_type] = self.shot_type_distribution.get(shot_type, 0) + 1

                for char in shot.characters:
                    self.character_coverage[char] = self.character_coverage.get(char, 0) + 1

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "scene_id": self.scene_id,
            "scene_description": self.scene_description,
            "shot_count": self.shot_count,
            "total_duration": self.total_duration,
            "generation_method": self.generation_method,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "shot_type_distribution": self.shot_type_distribution,
            "character_coverage": self.character_coverage,
            "shots": [shot.to_dict() for shot in self.shots],
            "generated_at": self.generated_at
        }


@dataclass
class ScriptShotResult:
    """整个剧本的分镜结果"""
    script_id: Optional[str] = None
    scene_results: Dict[str, SceneShotResult] = field(default_factory=dict)
    total_shots: int = 0
    total_duration: float = 0.0

    # 全局统计
    global_shot_distribution: Dict[str, int] = field(default_factory=dict)
    global_character_screen_time: Dict[str, float] = field(default_factory=dict)

    # 元数据
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()

        # 计算全局统计
        for scene_result in self.scene_results.values():
            self.total_shots += scene_result.shot_count
            self.total_duration += scene_result.total_duration

            # 合并镜头类型分布
            for shot_type, count in scene_result.shot_type_distribution.items():
                self.global_shot_distribution[shot_type] = self.global_shot_distribution.get(shot_type, 0) + count

            # 计算角色总出场时间
            for shot in scene_result.shots:
                for char in shot.characters:
                    self.global_character_screen_time[char] = self.global_character_screen_time.get(char, 0.0) + shot.duration_estimate

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "script_id": self.script_id,
            "total_shots": self.total_shots,
            "total_duration": self.total_duration,
            "global_shot_distribution": self.global_shot_distribution,
            "global_character_screen_time": self.global_character_screen_time,
            "scene_results": {k: v.to_dict() for k, v in self.scene_results.items()},
            "generated_at": self.generated_at
        }
