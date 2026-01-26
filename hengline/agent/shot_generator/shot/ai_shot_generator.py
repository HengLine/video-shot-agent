"""
@FileName: ai_shot_generator.py
@Description: AI增强生成器
@Author: HengLine
@Time: 2026/1/17 22:09
"""
import json
from typing import Dict, Any, List

from hengline.agent.script_parser2.script_parser_models import UnifiedScript, Dialogue, Action, Scene
from hengline.agent.shot_generator.shot.rule_shot_generator import RuleShotGenerator
from hengline.agent.shot_generator.shot.shot_model import Shot, ShotType, ShotPurpose, SceneShotResult
from hengline.logger import debug, info, warning, error
from utils.log_utils import print_log_exception


class AIEnhancedShotGenerator(RuleShotGenerator):
    """AI增强的分镜头生成器"""

    def __init__(self, config: Dict[str, Any] = None,
                 llm=None,  # AI服务接口
                 auto_ai_threshold: float = 0.65):
        super().__init__(config)
        self.llm = llm
        self.auto_ai_threshold = auto_ai_threshold

        # AI优化配置
        self.ai_optimize_scene_types = ["dialogue", "emotional", "complex"]

    def generate_for_scene(self, scene: Scene, script: UnifiedScript) -> SceneShotResult:
        """生成分镜头（AI增强版）"""
        # 1. 先生成规则基结果
        rule_result = super().generate_for_scene(scene, script)

        # 2. 判断是否需要AI增强
        if self._should_use_ai(scene, script, rule_result):
            debug(f"使用AI优化场景 {scene.scene_id}")

            try:
                # 3. AI增强
                ai_shots = self._ai_enhance_shots(scene, script, rule_result.shots)

                # 4. 融合结果
                final_shots = self._merge_ai_suggestions(rule_result.shots, ai_shots, scene)

                # 5. 更新镜头属性
                timestamp_counter = 0.0
                for i, shot in enumerate(final_shots):
                    shot.shot_id = self._create_shot_id(scene.scene_id, i + 1)
                    shot.scene_id = scene.scene_id
                    shot.sequence_number = i + 1
                    shot.timestamp_in_scene = timestamp_counter
                    if shot.duration_estimate == 0:
                        shot.duration_estimate = self.estimate_shot_duration(shot)
                    timestamp_counter += shot.duration_estimate

                # 6. 创建AI增强结果
                result = SceneShotResult(
                    scene_id=scene.scene_id,
                    scene_description=scene.description,
                    shots=final_shots,
                    generation_method="ai_enhanced",
                    confidence=self._calculate_ai_confidence(rule_result.confidence),
                    reasoning=f"规则基+AI优化，原始置信度: {rule_result.confidence:.2f}"
                )

                info(f"AI增强分镜生成完成: {result.shot_count}镜头，置信度: {result.confidence}")
                return result

            except Exception as e:
                error(f"AI增强失败，使用规则基结果: {e}")
                print_log_exception()
                # 返回规则基结果
                rule_result.generation_method = "rule_based_after_ai_fail"
                return rule_result

        # 不需要AI增强
        rule_result.generation_method = "rule_based_only"
        return rule_result

    def _should_use_ai(self, scene: Scene, script: UnifiedScript,
                       rule_result: SceneShotResult) -> bool:
        """判断是否应该使用AI"""
        if not self.llm:
            return False

        # 规则基置信度太低
        if rule_result.confidence < self.auto_ai_threshold:
            return True

        # 获取场景元素
        dialogues, actions = self._get_scene_elements(scene, script)

        # 复杂场景类型
        scene_type = self._determine_scene_type(scene, dialogues, actions)
        if scene_type in self.ai_optimize_scene_types:
            return True

        # 多角色互动场景
        characters = self._get_characters_in_scene(scene, script)
        if len(characters) >= 3:
            return True

        # 重要场景
        if scene.importance and scene.importance > 1.5:
            return True

        return False

    def _ai_enhance_shots(self, scene: Scene, script: UnifiedScript,
                          base_shots: List[Shot]) -> List[Shot]:
        """使用AI增强分镜头"""
        # 构建AI提示词
        prompt = self._build_ai_prompt(scene, script, base_shots)

        # 调用AI服务
        ai_response = self._call_llm_with_retry(self.llm, prompt)

        # 解析AI响应
        ai_shots = self._parse_ai_response(ai_response, scene, script)

        return ai_shots

    def _build_ai_prompt(self, scene: Scene, script: UnifiedScript,
                         base_shots: List[Shot]) -> str:
        """构建AI提示词"""
        # 获取场景相关信息
        dialogues, actions = self._get_scene_elements(scene, script)
        characters = self._get_characters_in_scene(scene, script)

        # 构建场景信息
        scene_info = f"""
场景ID: {scene.scene_id}
场景描述: {scene.description}
地点: {scene.location}
时间: {scene.time_of_day}
天气: {scene.weather}
氛围/情绪: {scene.mood}
重要性: {scene.importance or 1.0}

角色 ({len(characters)}人): {', '.join(characters)}

对话 ({len(dialogues)}个):
{self._format_dialogues_for_ai(dialogues)}

动作 ({len(actions)}个):
{self._format_actions_for_ai(actions)}
"""

        # 构建现有分镜头信息
        shots_info = "当前分镜头方案:\n"
        for i, shot in enumerate(base_shots[:10]):  # 最多显示10个
            shots_info += f"{i + 1}. {shot.shot_type.value}: {shot.description}\n"
            if shot.characters:
                shots_info += f"   角色: {', '.join(shot.characters)}\n"
            if shot.pacing != "normal":
                shots_info += f"   节奏: {shot.pacing}\n"
            shots_info += f"   时长: {shot.duration_estimate}s\n\n"

        # 构建完整提示词
        prompt = f"""作为专业的分镜师，请优化以下场景的分镜头设计。

{scene_info}

{shots_info}

优化要求：
1. 叙事清晰：确保观众能理解场景发展
2. 情感传达：有效表现"{scene.mood}"氛围
3. 节奏控制：目标节奏为"{"流畅自然"}"
4. 视觉多样：避免镜头类型单一
5. 角色平衡：确保主要角色有足够镜头

请提供优化后的分镜头列表，每个镜头包含：
- 镜头类型
- 简短描述
- 主要角色
- 节奏建议 (slow/normal/fast)
- 情感强度 (1.0-3.0)

请保持镜头数量在合理范围内 ({max(3, len(base_shots) - 2)}-{min(15, len(base_shots) + 3)}个)。"""

        return prompt

    def _format_dialogues_for_ai(self, dialogues: List[Dialogue]) -> str:
        """为AI格式化对话"""
        if not dialogues:
            return "  无对话"

        lines = []
        for i, d in enumerate(dialogues):
            lines.append(f"  {i + 1}. {d.speaker}: {d.text[:60]}{'...' if len(d.text) > 60 else ''}")
            if d.emotion != "neutral" or d.is_important:
                lines.append(f"     情绪: {d.emotion}, 重要: {d.is_important}")

        return "\n".join(lines)

    def _format_actions_for_ai(self, actions: List[Action]) -> str:
        """为AI格式化动作"""
        if not actions:
            return "  无动作"

        lines = []
        for i, a in enumerate(actions):
            lines.append(f"  {i + 1}. {', '.join(a.actors) if a.actors else '场景'}: {a.description[:60]}")
            if a.is_crucial:
                lines.append(f"     关键动作，类型: {a.action_type}")

        return "\n".join(lines)

    def _parse_ai_response(self, ai_response: str, scene: Scene,
                           script: UnifiedScript) -> List[Shot]:
        """解析AI响应"""
        # 这里简化实现，实际需要根据AI返回格式解析
        # 假设AI返回JSON格式

        try:
            # 尝试解析为JSON
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                ai_data = json.loads(json_match.group())
                return self._parse_ai_json(ai_data, scene)
        except Exception as e:
            error(f"AI响应JSON解析失败: {e}")
            pass

        # 如果无法解析为JSON，使用简单解析
        return self._parse_ai_text(ai_response, scene)

    def _parse_ai_json(self, ai_data: Dict, scene: Scene) -> List[Shot]:
        """解析AI的JSON响应"""
        shots = []

        if "shots" in ai_data:
            for i, shot_data in enumerate(ai_data["shots"]):
                try:
                    shot = Shot(
                        shot_id=f"{scene.scene_id}_ai_{i + 1}",
                        shot_type=self._parse_shot_type(shot_data.get("type", "medium")),
                        description=shot_data.get("description", f"AI建议镜头 {i + 1}"),
                        purpose=self._parse_shot_purpose(shot_data.get("purpose", "show_character")),
                        characters=shot_data.get("characters", []),
                        emotional_intensity=float(shot_data.get("emotional_intensity", 1.0)),
                        pacing=shot_data.get("pacing", "normal"),
                        notes=shot_data.get("notes", "AI建议")
                    )
                    shots.append(shot)
                except Exception as e:
                    warning(f"解析AI镜头失败: {e}")

        return shots if shots else [self._create_scene_establishing_shot(scene)]

    def _parse_ai_text(self, text: str, scene: Scene) -> List[Shot]:
        """解析AI的文本响应"""
        shots = []
        lines = text.strip().split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # 寻找镜头描述行
            if line and any(marker in line.lower() for marker in ["镜头", "shot", "scene", "close", "wide", "medium"]):
                # 简单解析
                shot_type = ShotType.MEDIUM
                if "广角" in line or "wide" in line.lower():
                    shot_type = ShotType.WIDE
                elif "特写" in line or "close" in line.lower():
                    shot_type = ShotType.CLOSE_UP
                elif "建立" in line or "establish" in line.lower():
                    shot_type = ShotType.ESTABLISHING

                description = line
                if ':' in line:
                    description = line.split(':', 1)[1].strip()
                elif ']' in line:
                    description = line.split(']', 1)[1].strip()

                shots.append(Shot(
                    shot_id=f"{scene.scene_id}_ai_{len(shots) + 1}",
                    shot_type=shot_type,
                    description=description[:100],
                    purpose=ShotPurpose.SHOW_CHARACTER,
                    notes="AI文本解析"
                ))

            i += 1

        return shots if shots else [self._create_scene_establishing_shot(scene)]

    def _parse_shot_type(self, type_str: str) -> ShotType:
        """解析镜头类型字符串"""
        type_map = {
            "establishing": ShotType.ESTABLISHING,
            "wide": ShotType.WIDE,
            "medium": ShotType.MEDIUM,
            "close_up": ShotType.CLOSE_UP,
            "extreme_close_up": ShotType.EXTREME_CLOSE_UP,
            "two_shot": ShotType.TWO_SHOT,
            "group_shot": ShotType.GROUP_SHOT,
            "reaction": ShotType.REACTION,
            "detail": ShotType.DETAIL,
        }
        return type_map.get(type_str.lower(), ShotType.MEDIUM)

    def _parse_shot_purpose(self, purpose_str: str) -> ShotPurpose:
        """解析镜头目的"""
        purpose_map = {
            "establish_location": ShotPurpose.ESTABLISH_LOCATION,
            "show_character": ShotPurpose.SHOW_CHARACTER,
            "dialogue": ShotPurpose.DIALOGUE,
            "reaction": ShotPurpose.REACTION,
            "action": ShotPurpose.ACTION,
            "emotion": ShotPurpose.EMOTION,
        }
        return purpose_map.get(purpose_str.lower(), ShotPurpose.SHOW_CHARACTER)

    def _merge_ai_suggestions(self, base_shots: List[Shot],
                              ai_shots: List[Shot], scene: Scene) -> List[Shot]:
        """融合AI建议和基础分镜头"""
        if not ai_shots or len(ai_shots) < 2:
            return base_shots.copy()

        # 策略：保留基础镜头的建立和结束，用AI镜头替换中间部分
        final_shots = []

        # 保留第一个基础镜头（建立镜头）
        if base_shots and base_shots[0].purpose == ShotPurpose.ESTABLISH_LOCATION:
            final_shots.append(base_shots[0])
        elif ai_shots:
            # 如果没有建立镜头，从AI镜头中找一个
            establishing_shots = [s for s in ai_shots if s.purpose == ShotPurpose.ESTABLISH_LOCATION]
            if establishing_shots:
                final_shots.append(establishing_shots[0])
            else:
                final_shots.append(ai_shots[0])

        # 使用AI镜头的中间部分（排除第一个和最后一个）
        ai_middle = ai_shots[1:-1] if len(ai_shots) > 2 else ai_shots[1:]
        final_shots.extend(ai_middle)

        # 保留结束镜头
        if base_shots and len(base_shots) > 1 and base_shots[-1].is_transition:
            final_shots.append(base_shots[-1])
        elif ai_shots and len(ai_shots) > 0:
            final_shots.append(ai_shots[-1])

        return final_shots if final_shots else base_shots.copy()

    def _calculate_ai_confidence(self, base_confidence: float) -> float:
        """计算AI增强后的置信度"""
        ai_boost = min(0.15, (1.0 - base_confidence) * 0.25)
        return round(min(0.95, base_confidence + ai_boost), 2)
