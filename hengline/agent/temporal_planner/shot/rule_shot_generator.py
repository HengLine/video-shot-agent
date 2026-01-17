"""
@FileName: rule_shot_generator.py
@Description: 规则基生成器
@Author: HengLine
@Time: 2026/1/17 22:07
"""
from typing import List, Dict, Any

from hengline.agent.temporal_planner.temporal_planner_model import EstimationSource
from hengline.context_var import task_id_ctx
from hengline.logger import debug, info, error
from hengline.agent.temporal_planner.shot.shot_model import Shot, ShotType, CameraMovement, ShotPurpose
from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.shot.base_shot_generator import BaseShotGenerator
from hengline.agent.temporal_planner.shot.shot_model import ShotGenerationResult
from utils.counter_utils import ExpiringStringCounter
from utils.log_utils import print_log_exception


class RuleShotGenerator(BaseShotGenerator):
    """基于规则的分镜头生成器"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.scene_templates = self._load_scene_templates()
        self.counter = ExpiringStringCounter(7200)  # 2小时过期

    def _load_scene_templates(self) -> Dict:
        """加载场景模板"""
        return {
            "establishing": self._generate_establishing_scene,
            "dialogue": self._generate_dialogue_scene,
            "action": self._generate_action_scene,
            "emotional": self._generate_emotional_scene,
            "general": self._generate_general_scene,
        }

    def generate_shots(self, script: UnifiedScript) -> ShotGenerationResult:
        """生成分镜头"""
        debug(f"开始规则基分镜生成")

        try:
            # 确定场景类型
            scene_type = self._determine_scene_type(script)
            debug(f"场景类型: {scene_type}")

            # 使用对应的模板生成分镜头
            if scene_type in self.scene_templates:
                shots = self.scene_templates[scene_type](script)
            else:
                shots = self._generate_general_scene(script)

            # 为每个镜头估算时长并编号
            for i, shot in enumerate(shots):
                shot.shot_id = self._create_shot_id(script.scene_id, i + 1)
                shot.sequence_number = i + 1
                shot.duration_estimate = self.estimate_shot_duration(shot)

            # 创建结果
            result = ShotGenerationResult(
                scene_id=script.scene_id,
                shots=shots,
                generation_method=EstimationSource.LOCAL_RULE,
                confidence=self._calculate_confidence(script, shots),
                reasoning=f"基于规则生成，场景类型: {scene_type}"
            )

            info(f"规则基分镜生成完成: {result.shot_count}个镜头，总时长: {result.total_duration}s")
            return result

        except Exception as e:
            print_log_exception()
            error(f"规则基分镜生成失败: {e}")
            return self._create_fallback_result(script)

    def _generate_establishing_scene(self, script: UnifiedScript) -> List[Shot]:
        """生成建立场景的分镜头"""
        shots = []

        # 1. 建立镜头 - 展示环境
        shots.append(Shot(
            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
            shot_type=ShotType.ESTABLISHING,
            description=f"展示{script.location}，时间{script.time_of_day}，天气{script.weather}",
            purpose=ShotPurpose.ESTABLISH_LOCATION,
            key_visuals=script.key_visual_elements.copy(),
            camera_movement=CameraMovement.PAN_LEFT if script.location != "室内" else CameraMovement.NONE,
            emotional_intensity=1.0 if script.mood == "平静" else 1.3
        ))

        # 2. 如果有角色，添加角色引入镜头
        if script.characters_in_scene:
            for char in script.characters_in_scene:
                shots.append(Shot(
                    shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                    shot_type=ShotType.MEDIUM,
                    description=f"展示{char}在场景中的位置",
                    purpose=ShotPurpose.SHOW_CHARACTER,
                    characters=[char],
                    emotional_intensity=self._get_emotion_intensity(script.mood)
                ))

        # 3. 氛围渲染镜头
        if script.mood != "neutral":
            shots.append(Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=ShotType.WIDE,
                description=f"渲染{script.mood}氛围",
                purpose=ShotPurpose.ATMOSPHERE,
                camera_movement=CameraMovement.SLOW_PAN if hasattr(CameraMovement, 'SLOW_PAN') else CameraMovement.PAN_LEFT,
                emotional_intensity=self._get_emotion_intensity(script.mood) + 0.2
            ))

        return shots

    def _generate_dialogue_scene(self, script: UnifiedScript) -> List[Shot]:
        """生成对话场景的分镜头"""
        shots = []
        dialogue_count = len(script.dialogues)

        if dialogue_count == 0:
            return self._generate_general_scene(script)

        # 1. 建立镜头
        shots.append(self._create_establishing_shot(script))

        # 2. 处理每个对话单元
        for i, dialogue in enumerate(script.dialogues):
            # 说话者镜头
            shots.append(Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=ShotType.CLOSE_UP,
                description=f"{dialogue.speaker}说: {dialogue.text[:30]}...",
                purpose=ShotPurpose.DIALOGUE,
                characters=[dialogue.speaker],
                dialogue_text=dialogue.text,
                dialogue_speaker=dialogue.speaker,
                emotional_intensity=dialogue.intensity,
                pacing="slow" if dialogue.emotion in ["sad", "emotional"] else "normal"
            ))

            # 重要的对话添加反应镜头
            if dialogue.is_important or dialogue.intensity > 1.5:
                # 找到听众（排除说话者）
                listeners = [c for c in script.characters_in_scene if c != dialogue.speaker]
                if listeners:
                    shots.append(Shot(
                        shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                        shot_type=ShotType.REACTION,
                        description=f"展示听众对{dialogue.speaker}话语的反应",
                        purpose=ShotPurpose.REACTION,
                        characters=listeners,
                        emotional_intensity=dialogue.intensity * 0.8,
                        pacing="slow"  # 反应镜头通常较慢
                    ))

            # 对话间的过渡镜头（每隔2-3个对话）
            if i > 0 and i % 2 == 0:
                shots.append(self._create_transition_shot(script))

        # 3. 对话结束后的总结镜头
        shots.append(Shot(
            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
            shot_type=ShotType.MEDIUM if len(script.characters_in_scene) <= 2 else ShotType.GROUP_SHOT,
            description=f"对话结束，展示角色状态",
            purpose=ShotPurpose.REACTION,
            characters=script.characters_in_scene.copy(),
            emotional_intensity=script.importance * 0.7
        ))

        return shots

    def _generate_action_scene(self, script: UnifiedScript) -> List[Shot]:
        """生成动作场景的分镜头"""
        shots = []
        action_count = len(script.actions)

        if action_count == 0:
            return self._generate_general_scene(script)

        # 1. 快速建立镜头
        shots.append(Shot(
            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
            shot_type=ShotType.WIDE,
            description="动作场景建立",
            purpose=ShotPurpose.ESTABLISH_LOCATION,
            camera_movement=CameraMovement.PAN_LEFT,
            pacing="fast"
        ))

        # 2. 处理每个动作单元
        for i, action in enumerate(script.actions):
            # 根据动作类型选择镜头
            if action.intensity.value > 2.0 or action.is_crucial:
                # 关键动作：广角 + 特写组合
                shots.append(Shot(
                    shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                    shot_type=ShotType.ACTION_WIDE,
                    description=f"广角: {action.description}",
                    purpose=ShotPurpose.ACTION,
                    characters=[action.actor],
                    action_description=action.description,
                    action_type=action.type,
                    emotional_intensity=action.intensity.value,
                    pacing="fast",
                    camera_movement=CameraMovement.TRACKING if action.action_type in ["run", "chase"] else CameraMovement.NONE
                ))

                # 动作特写
                shots.append(Shot(
                    shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                    shot_type=ShotType.ACTION_CLOSE,
                    description=f"特写: {action.description}",
                    purpose=ShotPurpose.ACTION,
                    characters=[action.actor],
                    emotional_intensity=action.intensity.value * 1.2,
                    pacing="fast"
                ))
            else:
                # 普通动作：中景
                shots.append(Shot(
                    shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                    shot_type=ShotType.MEDIUM,
                    description=action.description,
                    purpose=ShotPurpose.ACTION,
                    characters=[action.actor],
                    action_description=action.description,
                    emotional_intensity=action.intensity.value,
                    pacing="normal"
                ))

            # 复杂动作可能需要反应镜头
            if action.is_crucial and i < len(script.actions) - 1:
                # 找到其他角色的反应
                other_chars = [c for c in script.characters_in_scene if c != action.actor]
                if other_chars:
                    shots.append(Shot(
                        shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                        shot_type=ShotType.REACTION,
                        description=f"其他角色对动作的反应",
                        purpose=ShotPurpose.REACTION,
                        characters=other_chars,
                        emotional_intensity=action.intensity.value * 0.9,
                        pacing="normal"
                    ))

        # 3. 动作结束镜头
        shots.append(Shot(
            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
            shot_type=ShotType.WIDE,
            description="动作结束，展示结果",
            purpose=ShotPurpose.REVEAL,
            characters=[action.name for action in script.characters.copy()],
            pacing="slow" if script.mood in ["tense", "dramatic"] else "normal",
            emotional_intensity=script.importance
        ))

        return shots

    def _generate_emotional_scene(self, script: UnifiedScript) -> List[Shot]:
        """生成情感场景的分镜头"""
        shots = []

        # 1. 氛围建立镜头
        shots.append(Shot(
            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
            shot_type=ShotType.ESTABLISHING,
            description=f"建立{script.mood}情感氛围",
            purpose=ShotPurpose.ATMOSPHERE,
            camera_movement=CameraMovement.SLOW_DOLLY_IN if hasattr(CameraMovement, 'SLOW_DOLLY_IN') else CameraMovement.DOLLY_IN,
            emotional_intensity=self._get_emotion_intensity(script.mood) + 0.3,
            pacing="slow"
        ))

        # 2. 角色情感特写
        for char in script.characters_in_scene:
            shots.append(Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=ShotType.EXTREME_CLOSE_UP,
                description=f"{char}的情感表达",
                purpose=ShotPurpose.EMOTION,
                characters=[char],
                emotional_intensity=self._get_emotion_intensity(script.mood) + 0.5,
                pacing="very_slow" if script.mood in ["sad", "romantic"] else "slow"
            ))

        # 3. 细节渲染镜头
        for visual in script.key_visual_elements[:3]:  # 最多3个关键视觉
            shots.append(Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=ShotType.DETAIL,
                description=f"情感细节: {visual}",
                purpose=ShotPurpose.DETAIL,
                key_visuals=[visual],
                emotional_intensity=self._get_emotion_intensity(script.mood),
                pacing="slow"
            ))

        # 4. 总结镜头
        if len(script.characters_in_scene) > 1:
            shots.append(Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=ShotType.TWO_SHOT if len(script.characters_in_scene) == 2 else ShotType.GROUP_SHOT,
                description=f"情感场景总结",
                purpose=ShotPurpose.EMOTION,
                characters=script.characters_in_scene.copy(),
                emotional_intensity=self._get_emotion_intensity(script.mood),
                pacing="slow"
            ))

        return shots

    def _generate_general_scene(self, script: UnifiedScript) -> List[Shot]:
        """生成通用场景的分镜头"""
        shots = []

        # 基础结构：建立 -> 展示 -> 细节 -> 结束
        shots.append(self._create_establishing_shot(script))

        # 角色展示
        for char in script.characters_in_scene:
            shots.append(Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=ShotType.MEDIUM,
                description=f"展示{char}",
                purpose=ShotPurpose.SHOW_CHARACTER,
                characters=[char]
            ))

        # 关键视觉元素
        for visual in script.key_visual_elements[:2]:
            shots.append(Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=ShotType.DETAIL,
                description=f"关键细节: {visual}",
                purpose=ShotPurpose.DETAIL,
                key_visuals=[visual]
            ))

        # 结束镜头
        shots.append(Shot(
            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
            shot_type=ShotType.WIDE,
            description="场景结束",
            purpose=ShotPurpose.TRANSITION,
            characters=script.characters_in_scene.copy() if script.characters_in_scene else []
        ))

        return shots

    def _create_establishing_shot(self, script: UnifiedScript) -> Shot:
        """创建建立镜头"""
        return Shot(
            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
            shot_type=ShotType.ESTABLISHING,
            description=f"建立{script.location}场景",
            purpose=ShotPurpose.ESTABLISH_LOCATION,
            key_visuals=script.key_visual_elements.copy()[:3],  # 最多3个关键视觉
            camera_movement=CameraMovement.PAN_LEFT if "外" in script.location else CameraMovement.NONE,
            emotional_intensity=1.0 if script.mood == "neutral" else 1.2
        )

    def _create_transition_shot(self, script: UnifiedScript) -> Shot:
        """创建过渡镜头"""
        return Shot(
            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
            shot_type=ShotType.WIDE,
            description="场景过渡",
            purpose=ShotPurpose.TRANSITION,
            characters=script.characters_in_scene.copy(),
            is_transition=True,
            pacing="normal"
        )

    def _get_emotion_intensity(self, mood: str) -> float:
        """获取情绪强度"""
        intensity_map = {
            "neutral": 1.0,
            "calm": 1.0,
            "peaceful": 1.0,
            "happy": 1.3,
            "joyful": 1.4,
            "excited": 1.5,
            "sad": 1.6,
            "melancholy": 1.7,
            "angry": 1.5,
            "furious": 1.8,
            "tense": 1.6,
            "suspense": 1.7,
            "fearful": 1.8,
            "romantic": 1.4,
            "dramatic": 1.5,
        }
        return intensity_map.get(mood, 1.2)

    def _calculate_confidence(self, script: UnifiedScript, shots: List[Shot]) -> float:
        """计算生成置信度"""
        confidence = 0.7  # 基础置信度

        # 根据场景复杂度调整
        total_elements = len(script.dialogues) + len(script.actions)
        if total_elements <= 3:
            confidence += 0.15  # 简单场景置信度高
        elif total_elements <= 8:
            confidence += 0.05  # 中等场景
        else:
            confidence -= 0.1  # 复杂场景置信度降低

        # 根据镜头数量合理性调整
        expected_shots = min(3 + total_elements * 0.8, 15)  # 预期镜头数
        actual_shots = len(shots)
        ratio = actual_shots / expected_shots if expected_shots > 0 else 1.0

        if 0.7 <= ratio <= 1.3:
            confidence += 0.1
        else:
            confidence -= 0.1

        return round(min(0.95, max(0.3, confidence)), 2)

    def _create_fallback_result(self, script: UnifiedScript) -> ShotGenerationResult:
        """创建降级结果"""
        shots = [self._create_establishing_shot(script)]
        for i, shot in enumerate(shots):
            shot.shot_id = self._create_shot_id(script.scene_id, i + 1)
            shot.sequence_number = i + 1
            shot.duration_estimate = 3.0  # 默认时长

        return ShotGenerationResult(
            scene_id=script.scene_id,
            shots=shots,
            generation_method=EstimationSource.FALLBACK,
            confidence=0.3,
            reasoning="规则生成失败，使用降级方案"
        )