"""
@FileName: base_shot_generator.py
@Description: 
@Author: HengLine
@Time: 2026/1/17 22:03
"""
from abc import abstractmethod
from typing import List, Dict, Any

from hengline.agent.base_agent import BaseAgent
from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.shot.shot_model import ShotType, CameraMovement, Shot, ShotGenerationResult
from hengline.logger import error


class BaseShotGenerator(BaseAgent):
    """分镜头生成器基类"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._initialize_rules()

    def _initialize_rules(self):
        """初始化规则"""
        # 镜头类型基准时长（秒）
        self.shot_baselines = {
            ShotType.ESTABLISHING: 4.0,
            ShotType.WIDE: 3.5,
            ShotType.MEDIUM: 2.5,
            ShotType.CLOSE_UP: 2.0,
            ShotType.EXTREME_CLOSE_UP: 1.5,
            ShotType.TWO_SHOT: 3.0,
            ShotType.GROUP_SHOT: 4.0,
            ShotType.OVER_THE_SHOULDER: 2.8,
            ShotType.POV: 2.2,
            ShotType.REVERSE: 2.5,
            ShotType.REACTION: 2.0,
            ShotType.DETAIL: 1.8,
            ShotType.INSERT: 1.5,
            ShotType.ACTION_WIDE: 3.0,
            ShotType.ACTION_CLOSE: 2.2,
        }

        # 摄像机运动调整系数
        self.movement_factors = {
            CameraMovement.NONE: 1.0,
            CameraMovement.PAN_LEFT: 1.3,
            CameraMovement.PAN_RIGHT: 1.3,
            CameraMovement.TILT_UP: 1.2,
            CameraMovement.TILT_DOWN: 1.2,
            CameraMovement.DOLLY_IN: 1.4,
            CameraMovement.DOLLY_OUT: 1.4,
            CameraMovement.TRACKING: 1.5,
            CameraMovement.CRANE_UP: 1.6,
            CameraMovement.CRANE_DOWN: 1.6,
            CameraMovement.ZOOM_IN: 1.3,
            CameraMovement.ZOOM_OUT: 1.3,
            CameraMovement.HANDHELD: 1.2,
        }

        # 情感强度调整系数
        self.emotion_factors = {
            "neutral": 1.0,
            "calm": 1.0,
            "happy": 1.1,
            "excited": 1.2,
            "sad": 1.3,
            "angry": 1.2,
            "tense": 1.3,
            "fearful": 1.4,
            "romantic": 1.2,
        }

        # 节奏调整系数
        self.pacing_factors = {
            "slow": 1.3,
            "normal": 1.0,
            "fast": 0.8,
        }

    @abstractmethod
    def generate_shots(self, script: UnifiedScript) -> ShotGenerationResult:
        """生成分镜头 - 抽象方法"""
        pass

    def estimate_shot_duration(self, shot: Shot) -> float:
        """估算单个镜头时长"""
        try:
            # 基础时长
            base_duration = self.shot_baselines.get(
                shot.shot_type,
                self.shot_baselines[ShotType.MEDIUM]
            )

            # 摄像机运动调整
            movement_factor = self.movement_factors.get(
                shot.camera_movement, 1.0
            )
            base_duration *= movement_factor

            # 情感强度调整
            base_duration *= shot.emotional_intensity

            # 节奏调整
            pacing_factor = self.pacing_factors.get(shot.pacing, 1.0)
            base_duration *= pacing_factor

            # 对话内容调整（如果有对话）
            if shot.dialogue_text:
                word_count = len(shot.dialogue_text.split())
                dialogue_time = word_count * 0.3  # 每词0.3秒（包含情感停顿）
                base_duration = max(base_duration, dialogue_time)

            # 动作内容调整（如果有动作）
            if shot.action_description:
                # 简单动作估算
                action_words = len(shot.action_description.split())
                action_time = 1.0 + (action_words * 0.2)
                base_duration = max(base_duration, action_time)

            # 确保最小时长
            return round(max(0.5, base_duration), 2)

        except Exception as e:
            error(f"估算镜头时长失败: {e}")
            return 2.0  # 默认时长

    def _determine_scene_type(self, script: UnifiedScript) -> str:
        """确定场景类型"""
        # 基于脚本内容判断场景类型
        if not script.dialogues and not script.actions:
            return "establishing"
        elif len(script.dialogues) > len(script.actions) * 2:
            return "dialogue"
        elif len(script.actions) > len(script.dialogues) * 2:
            return "action"
        elif script.mood in ["tense", "suspense", "action"]:
            return "action"
        elif script.mood in ["emotional", "romantic", "dramatic"]:
            return "emotional"
        else:
            return "general"

    def _get_characters_for_shot(self, script: UnifiedScript,
                                 focus_on: str = None) -> List[str]:
        """获取镜头中的角色"""
        if focus_on:
            return [focus_on]

        # 默认使用所有场景角色
        return script.characters_in_scene.copy()

    def _create_shot_id(self, scene_id: str, seq_num: int) -> str:
        """创建镜头ID"""
        return f"{scene_id}_shot_{seq_num:03d}"
