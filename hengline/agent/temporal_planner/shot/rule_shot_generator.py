"""
@FileName: rule_shot_generator.py
@Description: 规则基生成器
@Author: HengLine
@Time: 2026/1/17 22:07
"""
from typing import List, Dict, Any

from hengline.agent.script_parser2.script_parser_models import UnifiedScript, Dialogue, Action, Scene
from hengline.agent.temporal_planner.shot.base_shot_generator import BaseShotGenerator
from hengline.agent.temporal_planner.shot.shot_model import Shot, ShotType, CameraMovement, ShotPurpose, SceneShotResult
from hengline.context_var import task_id_ctx
from hengline.logger import debug, info, error
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

    def generate_for_scene(self, scene: Scene, script: UnifiedScript) -> SceneShotResult:
        """为场景生成分镜头"""
        debug(f"开始规则基分镜生成: {scene.scene_id}")

        try:
            # 获取场景元素
            dialogues, actions = self._get_scene_elements(scene, script)
            characters = self._get_characters_in_scene(scene, script)

            # 确定场景类型
            scene_type = self._determine_scene_type(scene, dialogues, actions)
            debug(f"场景类型: {scene_type}")

            # 使用对应的模板生成分镜头
            if scene_type in self.scene_templates:
                shots = self.scene_templates[scene_type](scene, dialogues, actions, characters, script)
            else:
                shots = self._generate_general_scene(scene, dialogues, actions, characters, script)

            # 为每个镜头估算时长并编号
            timestamp_counter = 0.0
            for i, shot in enumerate(shots):
                shot.shot_id = self._create_shot_id(scene.scene_id, i + 1)
                shot.scene_id = scene.scene_id
                shot.sequence_number = i + 1
                shot.timestamp_in_scene = timestamp_counter
                shot.duration_estimate = self.estimate_shot_duration(shot)
                timestamp_counter += shot.duration_estimate

            # 创建结果
            result = SceneShotResult(
                scene_id=scene.scene_id,
                scene_description=scene.description,
                shots=shots,
                generation_method="rule_based",
                confidence=self._calculate_confidence(scene, dialogues, actions, shots),
                reasoning=f"基于规则生成，场景类型: {scene_type}，对话: {len(dialogues)}个，动作: {len(actions)}个"
            )

            info(f"规则基分镜生成完成: {result.shot_count}个镜头，总时长: {result.total_duration}s")
            return result

        except Exception as e:
            print_log_exception()
            error(f"规则基分镜生成失败: {e}")
            return self._create_fallback_result(scene)

    def _generate_establishing_scene(self, scene: Scene, dialogues: List[Dialogue],
                                     actions: List[Action], characters: List[str],
                                     script: UnifiedScript) -> List[Shot]:
        """生成建立场景的分镜头"""
        shots = []

        # 1. 建立镜头
        key_visuals = scene.key_visuals or []
        shots.append(Shot(
            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
            shot_type=ShotType.ESTABLISHING,
            description=f"展示{scene.location}，{scene.time_of_day}，{scene.weather}",
            purpose=ShotPurpose.ESTABLISH_LOCATION,
            key_visuals=key_visuals[:3],  # 最多3个关键视觉
            camera_movement=CameraMovement.PAN_LEFT if "外" in scene.location else CameraMovement.NONE,
            emotional_intensity=self._get_emotion_intensity(scene.mood),
            pacing="slow"
        ))

        # 2. 角色引入
        for char in characters:
            shots.append(Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=ShotType.MEDIUM,
                description=f"展示{char}在场景中",
                purpose=ShotPurpose.SHOW_CHARACTER,
                characters=[char],
                emotional_intensity=1.0
            ))

        return shots

    def _generate_dialogue_scene(self, scene: Scene, dialogues: List[Dialogue],
                                 actions: List[Action], characters: List[str],
                                 script: UnifiedScript) -> List[Shot]:
        """生成对话场景的分镜头"""
        shots = []

        # 1. 建立镜头
        shots.append(self._create_scene_establishing_shot(scene))

        # 2. 处理每个对话
        dialogue_shots = []
        for dialogue in dialogues:
            # 说话者镜头
            shot = Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=ShotType.CLOSE_UP,
                description=f"{dialogue.speaker}: {dialogue.text[:40]}...",
                purpose=ShotPurpose.DIALOGUE,
                characters=[dialogue.speaker],
                dialogue_id=dialogue.dialogue_id,
                emotional_intensity=dialogue.intensity,
                pacing="slow" if dialogue.emotion in ["sad", "emotional"] else "normal"
            )
            dialogue_shots.append(shot)

            # 如果是重要对话，添加听众反应镜头
            if dialogue.is_important or dialogue.intensity > 1.5:
                # 找到听众
                listeners = [c for c in characters if c != dialogue.speaker]
                if listeners:
                    reaction_shot = Shot(
                        shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                        shot_type=ShotType.REACTION,
                        description=f"{'、'.join(listeners)}的反应",
                        purpose=ShotPurpose.REACTION,
                        characters=listeners,
                        emotional_intensity=dialogue.intensity * 0.8,
                        pacing="slow"
                    )
                    dialogue_shots.append(reaction_shot)

        # 3. 处理动作（如果有）
        action_shots = []
        for action in actions:
            if action.actors:
                shot = Shot(
                    shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                    shot_type=ShotType.MEDIUM,
                    description=action.description[:50],
                    purpose=ShotPurpose.ACTION,
                    characters=action.actors.copy(),
                    action_id=action.action_id,
                    emotional_intensity=action.intensity.value,
                    pacing="fast" if action.intensity.value > 1.5 else "normal"
                )
                action_shots.append(shot)

        # 4. 交织对话和动作镜头
        shots.extend(self._interweave_dialogue_action(dialogue_shots, action_shots, scene))

        # 5. 结束镜头
        if characters:
            shot_type = ShotType.TWO_SHOT if len(characters) == 2 else ShotType.GROUP_SHOT
            shots.append(Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=shot_type,
                description=f"对话结束，{scene.mood}氛围",
                purpose=ShotPurpose.EMOTION,
                characters=characters.copy(),
                emotional_intensity=self._get_emotion_intensity(scene.mood),
                pacing="slow"
            ))

        return shots

    def _generate_action_scene(self, scene: Scene, dialogues: List[Dialogue],
                               actions: List[Action], characters: List[str],
                               script: UnifiedScript) -> List[Shot]:
        """生成动作场景的分镜头"""
        shots = []

        # 1. 快速建立镜头
        shots.append(Shot(
            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
            shot_type=ShotType.WIDE,
            description=f"动作场景: {scene.location}",
            purpose=ShotPurpose.ESTABLISH_LOCATION,
            camera_movement=CameraMovement.PAN_LEFT,
            pacing="fast"
        ))

        # 2. 处理关键动作
        for action in actions:
            if action.is_crucial:
                # 关键动作：广角+特写组合
                shots.append(Shot(
                    shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                    shot_type=ShotType.ACTION_WIDE,
                    description=f"{action.actors[0] if action.actors else '场景'}: {action.description[:40]}...",
                    purpose=ShotPurpose.ACTION,
                    characters=action.actors.copy(),
                    action_id=action.action_id,
                    camera_movement=CameraMovement.TRACKING if action.action_type in ["run", "chase"] else CameraMovement.NONE,
                    emotional_intensity=action.intensity.value,
                    pacing="fast"
                ))

                shots.append(Shot(
                    shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                    shot_type=ShotType.ACTION_CLOSE,
                    description=f"动作特写: {action.description[:40]}...",
                    purpose=ShotPurpose.DETAIL,
                    characters=action.actors.copy() if action.actors else [],
                    emotional_intensity=action.intensity.value * 1.2,
                    pacing="very_fast" if action.intensity.value > 2.0 else "fast"
                ))
            else:
                # 普通动作
                shots.append(Shot(
                    shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                    shot_type=ShotType.MEDIUM,
                    description=action.description[:50],
                    purpose=ShotPurpose.ACTION,
                    characters=[action.actor],
                    action_id=action.action_id,
                    emotional_intensity=action.intensity.value,
                    pacing="normal"
                ))

        # 3. 处理对话（如果有）
        for dialogue in dialogues:
            shots.append(Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=ShotType.CLOSE_UP,
                description=f"{dialogue.speaker}: {dialogue.content[:30]}...",
                purpose=ShotPurpose.DIALOGUE,
                characters=[dialogue.speaker],
                dialogue_id=dialogue.dialogue_id,
                emotional_intensity=dialogue.intensity,
                pacing="normal"
            ))

        # 4. 结束镜头
        shots.append(Shot(
            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
            shot_type=ShotType.WIDE,
            description=f"动作结束，{scene.mood}氛围",
            purpose=ShotPurpose.REVEAL,
            characters=characters.copy(),
            emotional_intensity=self._get_emotion_intensity(scene.mood),
            pacing="slow" if scene.mood in ["tense", "dramatic"] else "normal"
        ))

        return shots

    def _generate_emotional_scene(self, scene: Scene, dialogues: List[Dialogue],
                                  actions: List[Action], characters: List[str],
                                  script: UnifiedScript) -> List[Shot]:
        """生成情感场景的分镜头"""
        shots = []

        # 1. 氛围建立
        shots.append(Shot(
            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
            shot_type=ShotType.ESTABLISHING,
            description=f"建立{scene.mood}氛围: {scene.location}",
            purpose=ShotPurpose.ATMOSPHERE,
            camera_movement=CameraMovement.DOLLY_IN,
            emotional_intensity=self._get_emotion_intensity(scene.mood) + 0.3,
            pacing="very_slow" if scene.mood in ["sad", "lonely", "romantic"] else "slow"
        ))

        # 2. 角色情感特写
        for char in characters:
            shots.append(Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=ShotType.EXTREME_CLOSE_UP,
                description=f"{char}的情感表达",
                purpose=ShotPurpose.EMOTION,
                characters=[char],
                emotional_intensity=self._get_emotion_intensity(scene.mood) + 0.5,
                pacing="very_slow"
            ))

        # 3. 对话处理（情感对话需要更长的镜头）
        for dialogue in dialogues:
            shots.append(Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=ShotType.CLOSE_UP,
                description=f"{dialogue.speaker}: {dialogue.text[:30]}...",
                purpose=ShotPurpose.DIALOGUE,
                characters=[dialogue.speaker],
                dialogue_id=dialogue.dialogue_id,
                emotional_intensity=dialogue.intensity,
                pacing="very_slow" if dialogue.emotion in ["sad", "emotional"] else "slow"
            ))

        # 4. 关键视觉细节
        key_visuals = scene.key_visuals or []
        for visual in key_visuals[:2]:  # 最多2个关键视觉
            shots.append(Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=ShotType.DETAIL,
                description=f"情感细节: {visual}",
                purpose=ShotPurpose.DETAIL,
                key_visuals=[visual],
                emotional_intensity=self._get_emotion_intensity(scene.mood),
                pacing="very_slow"
            ))

        # 5. 关系镜头（如果有多角色）
        if len(characters) > 1:
            # 检查角色关系
            relationships = script.relationships
            if relationships:
                for rel in relationships:
                    if rel.character_a in characters and rel.character_b in characters:
                        shots.append(Shot(
                            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                            shot_type=ShotType.TWO_SHOT,
                            description=f"{rel.character_a}和{rel.character_b}的关系互动",
                            purpose=ShotPurpose.RELATIONSHIP,
                            characters=[rel.character_a, rel.character_b],
                            emotional_intensity=rel.intensity,
                            pacing="slow"
                        ))

        return shots

    def _generate_general_scene(self, scene: Scene, dialogues: List[Dialogue],
                                actions: List[Action], characters: List[str],
                                script: UnifiedScript) -> List[Shot]:
        """生成通用场景的分镜头"""
        shots = []

        # 1. 建立镜头
        shots.append(self._create_scene_establishing_shot(scene))

        # 2. 角色展示
        for char in characters:
            shots.append(Shot(
                shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                shot_type=ShotType.MEDIUM,
                description=f"展示{char}",
                purpose=ShotPurpose.SHOW_CHARACTER,
                characters=[char]
            ))

        # 3. 交替处理对话和动作
        all_elements = self._merge_and_sort_elements(dialogues, actions)

        for element in all_elements:
            if isinstance(element, Dialogue):
                shots.append(Shot(
                    shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                    shot_type=ShotType.CLOSE_UP,
                    description=f"{element.speaker}: {element.text[:30]}...",
                    purpose=ShotPurpose.DIALOGUE,
                    characters=[element.speaker],
                    dialogue_id=element.dialogue_id,
                    emotional_intensity=element.intensity,
                    pacing="normal"
                ))
            else:  # Action
                if element.actors:
                    shots.append(Shot(
                        shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
                        shot_type=ShotType.MEDIUM,
                        description=element.description[:40],
                        purpose=ShotPurpose.ACTION,
                        characters=element.actors.copy(),
                        action_id=element.action_id,
                        emotional_intensity=element.intensity,
                        pacing="normal"
                    ))

        # 4. 结束镜头
        shots.append(Shot(
            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
            shot_type=ShotType.WIDE,
            description="场景结束",
            purpose=ShotPurpose.TRANSITION,
            characters=characters.copy() if characters else [],
            is_transition=True
        ))

        return shots

    def _create_scene_establishing_shot(self, scene: Scene) -> Shot:
        """创建场景建立镜头"""
        key_visuals = scene.key_visuals or []
        return Shot(
            shot_id=f"shot_{self.counter.get_next(task_id_ctx.get())}",
            shot_type=ShotType.ESTABLISHING,
            description=f"建立{scene.location}场景",
            purpose=ShotPurpose.ESTABLISH_LOCATION,
            key_visuals=key_visuals[:3],
            camera_movement=CameraMovement.PAN_LEFT if "外" in scene.location else CameraMovement.NONE,
            emotional_intensity=self._get_emotion_intensity(scene.mood),
            pacing="slow"
        )

    def _interweave_dialogue_action(self, dialogue_shots: List[Shot],
                                    action_shots: List[Shot],
                                    scene: Scene) -> List[Shot]:
        """交织对话和动作镜头"""
        shots = []

        # 根据场景氛围决定交织策略
        if scene.mood in ["tense", "action", "suspense"]:
            # 紧张场景：动作优先，对话快速穿插
            for i, action_shot in enumerate(action_shots):
                shots.append(action_shot)
                # 每隔一个动作插入一个对话
                if i < len(dialogue_shots):
                    shots.append(dialogue_shots[i])
        else:
            # 普通场景：对话优先，动作作为打断
            for i, dialogue_shot in enumerate(dialogue_shots):
                shots.append(dialogue_shot)
                # 重要对话后插入动作
                if i < len(action_shots) and dialogue_shot.emotional_intensity > 1.5:
                    shots.append(action_shots[i])

        # 确保所有镜头都被包含
        used_dialogue = min(len(dialogue_shots), len(shots) // 2)
        used_action = min(len(action_shots), len(shots) - used_dialogue)

        # 添加剩余的镜头
        if used_dialogue < len(dialogue_shots):
            shots.extend(dialogue_shots[used_dialogue:])
        if used_action < len(action_shots):
            shots.extend(action_shots[used_action:])

        return shots

    def _merge_and_sort_elements(self, dialogues: List[Dialogue],
                                 actions: List[Action]) -> List:
        """合并并按时间戳排序对话和动作"""
        all_elements = []

        for dialogue in dialogues:
            all_elements.append((dialogue.timestamp or 0, dialogue))

        for action in actions:
            all_elements.append((action.timestamp or 0, action))

        # 按时间戳排序
        all_elements.sort(key=lambda x: x[0])

        return [element for _, element in all_elements]

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
            "lonely": 1.6,
        }
        return intensity_map.get(mood, 1.2)

    def _calculate_confidence(self, scene: Scene, dialogues: List[Dialogue],
                              actions: List[Action], shots: List[Shot]) -> float:
        """计算生成置信度"""
        confidence = 0.7

        # 元素丰富度
        total_elements = len(dialogues) + len(actions)
        if total_elements == 0:
            confidence += 0.2  # 纯建立场景容易处理
        elif 1 <= total_elements <= 5:
            confidence += 0.1
        elif total_elements > 10:
            confidence -= 0.1

        # 场景信息完整性
        if scene.description and scene.location and scene.mood:
            confidence += 0.1

        # 角色明确性
        if scene.character_refs or (dialogues and dialogues[0].speaker):
            confidence += 0.05

        return round(min(0.95, max(0.3, confidence)), 2)
