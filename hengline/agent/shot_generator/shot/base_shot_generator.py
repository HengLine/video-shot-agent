"""
@FileName: base_shot_generator.py
@Description: 
@Author: HengLine
@Time: 2026/1/17 22:03
"""
from abc import abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Tuple

from hengline.agent.base_agent import BaseAgent
from hengline.agent.script_parser2.script_parser_models import UnifiedScript, Scene, Dialogue, Action
from hengline.agent.shot_generator.shot.shot_model import ShotType, CameraMovement, Shot, ShotPurpose, SceneShotResult, ScriptShotResult
from hengline.logger import error, info, debug


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
            ShotType.DOLLY: 3.2,
            ShotType.PAN: 3.0,
            ShotType.TILT: 2.8,
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
            "lonely": 1.3,
            "melancholy": 1.4,
            "suspense": 1.3,
        }

        # 节奏调整系数
        self.pacing_factors = {
            "slow": 1.3,
            "normal": 1.0,
            "fast": 0.8,
            "very_slow": 1.5,
            "very_fast": 0.7,
        }

    @abstractmethod
    def generate_for_scene(self, scene: Scene,
                           script: UnifiedScript) -> SceneShotResult:
        """为单个场景生成分镜头 - 抽象方法"""
        pass

    def generate_for_script(self, script: UnifiedScript) -> ScriptShotResult:
        """为整个剧本生成分镜头"""
        debug(f"开始为剧本生成分镜头，共{len(script.scenes)}个场景")

        scene_results = {}

        for scene in script.scenes:
            try:
                result = self.generate_for_scene(scene, script)
                scene_results[scene.scene_id] = result
                info(f"场景 {scene.scene_id} 分镜生成完成: {result.shot_count}镜头")
            except Exception as e:
                error(f"场景 {scene.scene_id} 分镜生成失败: {e}")
                # 创建降级结果
                scene_results[scene.scene_id] = self._create_fallback_result(scene)

        return ScriptShotResult(
            script_id=f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            scene_results=scene_results
        )

    def estimate_shot_duration(self, shot: Shot, context: Dict = None) -> float:
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

            # 确保最小时长
            return round(max(0.5, base_duration), 2)

        except Exception as e:
            error(f"估算镜头时长失败: {e}")
            return 2.0  # 默认时长

    def _get_scene_elements(self, scene: Scene, script: UnifiedScript) -> Tuple[List[Dialogue], List[Action]]:
        """获取场景相关的对话和动作"""
        scene_dialogues = [d for d in script.dialogues if d.scene_ref == scene.scene_id]
        scene_actions = [a for a in script.actions if a.scene_ref == scene.scene_id]

        # 按时间戳排序（如果有时戳）
        scene_dialogues.sort(key=lambda x: x.timestamp or 0)
        scene_actions.sort(key=lambda x: x.timestamp or 0)

        return scene_dialogues, scene_actions

    def _get_characters_in_scene(self, scene: Scene, script: UnifiedScript) -> List[str]:
        """获取场景中的角色"""
        if scene.character_refs:
            return scene.character_refs.copy()

        # 从对话和动作中推断角色
        characters = set()
        scene_dialogues, scene_actions = self._get_scene_elements(scene, script)

        for dialogue in scene_dialogues:
            characters.add(dialogue.speaker)

        for action in scene_actions:
            characters.update(action.actors)

        return list(characters)

    def _determine_scene_type(self, scene: Scene, dialogues: List[Dialogue],
                              actions: List[Action]) -> str:
        """确定场景类型"""
        if not dialogues and not actions:
            return "establishing"

        dialogue_ratio = len(dialogues) / max(1, len(dialogues) + len(actions))

        if dialogue_ratio > 0.7:
            return "dialogue"
        elif dialogue_ratio < 0.3:
            return "action"
        elif scene.mood in ["tense", "suspense", "action"]:
            return "action"
        elif scene.mood in ["emotional", "romantic", "dramatic", "sad", "lonely"]:
            return "emotional"
        else:
            return "general"

    def _create_shot_id(self, scene_id: str, seq_num: int) -> str:
        """创建镜头ID"""
        return f"{scene_id}_shot_{seq_num:03d}"

    def _create_fallback_result(self, scene: Scene) -> SceneShotResult:
        """创建降级结果"""
        shot = Shot(
            shot_id=self._create_shot_id(scene.scene_id, 1),
            shot_type=ShotType.ESTABLISHING,
            description=f"建立镜头: {scene.description[:50]}...",
            duration_estimate=3.0,
            purpose=ShotPurpose.ESTABLISH_LOCATION,
            scene_id=scene.scene_id,
            sequence_number=1
        )

        return SceneShotResult(
            scene_id=scene.scene_id,
            scene_description=scene.description,
            shots=[shot],
            generation_method="fallback",
            confidence=0.3,
            reasoning="分镜生成失败，使用降级方案"
        )
