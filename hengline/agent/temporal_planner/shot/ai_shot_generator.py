"""
@FileName: ai_shot_generator.py
@Description: AI增强生成器
@Author: HengLine
@Time: 2026/1/17 22:09
"""
from typing import Dict, Any, List

from hengline.agent.temporal_planner.temporal_planner_model import EstimationSource
from hengline.logger import debug, info, warning, error
from hengline.agent.temporal_planner.shot.shot_model import Shot, ShotType, CameraMovement, ShotPurpose
from hengline.agent.script_parser.script_parser_models import UnifiedScript, Dialogue, Action
from hengline.agent.temporal_planner.shot.rule_shot_generator import RuleShotGenerator
from hengline.agent.temporal_planner.shot.shot_model import ShotGenerationResult
from utils.log_utils import print_log_exception


class AIEnhancedShotGenerator(RuleShotGenerator):
    """AI增强的分镜头生成器"""

    def __init__(self, config: Dict[str, Any] = None,
                 llm=None,  # AI服务接口
                 use_ai_for_all: bool = False):
        super().__init__(config)
        self.llm = llm
        self.use_ai_for_all = use_ai_for_all

        # AI优化配置
        self.ai_optimize_scene_types = ["dialogue", "emotional", "complex"]
        self.min_confidence_for_ai = 0.6

    def generate_shots(self, script: UnifiedScript) -> ShotGenerationResult:
        """生成分镜头（AI增强版）"""
        debug(f"开始AI增强分镜生成")

        try:
            # 1. 先生成规则基分镜头
            rule_result = super().generate_shots(script)

            # 2. 判断是否需要AI增强
            if self._should_use_ai(script, rule_result):
                debug(f"使用AI优化分镜头")
                ai_enhanced_shots = self._enhance_with_ai(script, rule_result.shots)

                # 3. 融合AI建议
                final_shots = self._merge_ai_suggestions(rule_result.shots, ai_enhanced_shots)

                # 更新镜头时长和ID
                for i, shot in enumerate(final_shots):
                    if not shot.shot_id:
                        shot.shot_id = self._create_shot_id(script.scene_id, i + 1)
                    shot.sequence_number = i + 1
                    if shot.duration_estimate == 0:
                        shot.duration_estimate = self.estimate_shot_duration(shot)

                # 4. 创建AI增强结果
                result = ShotGenerationResult(
                    scene_id=script.scene_id,
                    shots=final_shots,
                    generation_method=EstimationSource.AI_LLM,
                    confidence=self._calculate_ai_confidence(rule_result.confidence),
                    reasoning=f"规则基+AI优化，原始置信度: {rule_result.confidence}"
                )

                info(f"AI增强分镜生成完成: {result.shot_count}个镜头，总时长: {result.total_duration}s")
                return result

            # 如果不使用AI，返回规则基结果
            rule_result.generation_method = EstimationSource.LOCAL_RULE
            debug(f"跳过AI优化，使用纯规则基结果")
            return rule_result

        except Exception as e:
            print_log_exception()
            error(f"AI增强分镜生成失败: {e}")
            # 降级到纯规则基
            warning(f"降级到规则基生成器")
            return super().generate_shots(script)

    def _should_use_ai(self, script: UnifiedScript, rule_result: ShotGenerationResult) -> bool:
        """判断是否应该使用AI"""
        if self.use_ai_for_all:
            return True

        # 规则基置信度太低，需要AI帮助
        if rule_result.confidence < self.min_confidence_for_ai:
            return True

        # 特定场景类型使用AI
        scene_type = self._determine_scene_type(script)
        if scene_type in self.ai_optimize_scene_types:
            return True

        # 复杂场景使用AI
        total_elements = len(script.dialogues) + len(script.actions)
        if total_elements > 8:  # 复杂场景
            return True

        # 重要场景使用AI
        if script.importance > 1.5:
            return True

        return False

    def _enhance_with_ai(self, script: UnifiedScript, base_shots: List[Shot]) -> List[Shot]:
        """使用AI增强分镜头"""
        if not self.llm:
            warning("AI服务未配置，跳过AI增强")
            return base_shots.copy()

        try:
            # 构建AI提示词
            prompt = self._build_ai_prompt(script, base_shots)

            # 调用AI服务
            ai_response = self._call_llm_with_retry(self.llm, prompt)

            # 解析AI响应
            enhanced_shots = self._parse_ai_response(ai_response, script)

            return enhanced_shots

        except Exception as e:
            error(f"AI增强失败: {e}")
            return base_shots.copy()

    def _build_ai_prompt(self, script: UnifiedScript, base_shots: List[Shot]) -> str:
        """构建AI提示词"""
        # 格式化脚本信息
        script_info = f"""
场景ID: {script.scene_id}
场景描述: {script.scene_description}
地点: {script.location}
时间: {script.time_of_day}
天气: {script.weather}
氛围/情绪: {script.mood}
场景重要性: {script.importance}
节奏目标: {script.pacing_target}

角色列表: {', '.join(script.characters_in_scene)}

对话单元 ({len(script.dialogues)}个):
{self._format_dialogues(script.dialogues)}

动作单元 ({len(script.actions)}个):
{self._format_actions(script.actions)}

关键视觉元素: {', '.join(script.key_visual_elements)}
"""

        # 格式化当前分镜头
        shots_info = "当前分镜头方案:\n"
        for i, shot in enumerate(base_shots):
            shots_info += f"{i + 1}. {shot.shot_type.value}: {shot.description}\n"
            if shot.characters:
                shots_info += f"   角色: {', '.join(shot.characters)}\n"
            if shot.dialogue_text:
                shots_info += f"   对话: {shot.dialogue_text[:50]}...\n"
            if shot.action_description:
                shots_info += f"   动作: {shot.action_description}\n"
            shots_info += f"   时长: {shot.duration_estimate}s\n\n"

        # 构建完整提示词
        prompt = f"""你是一位经验丰富的电影分镜师和导演。请优化以下场景的分镜头设计。

{script_info}

{shots_info}

请分析并优化分镜头方案，考虑以下方面：
1. 叙事清晰度：镜头是否能清晰传达故事？
2. 情感表达：是否能有效表现"{script.mood}"氛围？
3. 节奏控制：是否符合"{script.pacing_target}"的节奏目标？
4. 视觉多样性：镜头类型是否丰富多样？
5. 技术可行性：镜头是否在合理的技术范围内？

请提供优化后的分镜头列表，格式如下：
[序号]. [镜头类型]: [描述]
   目的: [镜头目的]
   角色: [角色列表]
   运动: [摄像机运动]
   情感强度: [1.0-3.0]
   节奏: [slow/normal/fast]
   备注: [可选备注]

请确保镜头数量在合理范围内（通常3-15个），并保持连贯性。"""

        return prompt

    def _format_dialogues(self, dialogues: List[Dialogue]) -> str:
        """格式化对话列表"""
        if not dialogues:
            return "无对话"

        result = []
        for i, d in enumerate(dialogues):
            result.append(f"  {i + 1}. {d.speaker}: {d.text[:50]}{'...' if len(d.text) > 50 else ''}")
            result.append(f"     情绪: {d.emotion}, 强度: {d.intensity}, 重要: {d.is_important}")

        return "\n".join(result)

    def _format_actions(self, actions: List[Action]) -> str:
        """格式化动作列表"""
        if not actions:
            return "无动作"

        result = []
        for i, a in enumerate(actions):
            result.append(f"  {i + 1}. {a.actor}: {a.description}")
            result.append(f"     类型: {a.action_type}, 强度: {a.intensity}, 关键: {a.is_crucial}")

        return "\n".join(result)

    def _parse_ai_response(self, ai_response: str, script: UnifiedScript) -> List[Shot]:
        """解析AI响应为Shot对象列表"""
        shots = []
        lines = ai_response.strip().split('\n')

        i = 0
        current_shot = None

        while i < len(lines):
            line = lines[i].strip()

            # 识别新的镜头开始（以数字开头）
            if line and line[0].isdigit() and '.' in line:
                # 保存上一个镜头
                if current_shot:
                    shots.append(current_shot)

                # 解析新镜头
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    shot_type_str = parts[0].split('.')[1].strip() if '.' in parts[0] else parts[0].strip()
                    description = parts[1].strip()

                    # 尝试解析镜头类型
                    shot_type = self._parse_shot_type(shot_type_str)

                    current_shot = Shot(
                        shot_id=f"shot_{script.scene_id}_{len(shots) + 1}",
                        shot_type=shot_type,
                        description=description,
                        purpose=ShotPurpose.SHOW_CHARACTER,  # 默认
                        emotional_intensity=1.0,
                        pacing="normal"
                    )

            # 解析镜头属性
            elif current_shot:
                if line.startswith("目的:"):
                    purpose_str = line.replace("目的:", "").strip()
                    current_shot.purpose = self._parse_shot_purpose(purpose_str)
                elif line.startswith("角色:"):
                    chars_str = line.replace("角色:", "").strip()
                    current_shot.characters = [c.strip() for c in chars_str.split(',') if c.strip()]
                elif line.startswith("运动:"):
                    movement_str = line.replace("运动:", "").strip()
                    current_shot.camera_movement = self._parse_camera_movement(movement_str)
                elif line.startswith("情感强度:"):
                    intensity_str = line.replace("情感强度:", "").strip()
                    try:
                        current_shot.emotional_intensity = float(intensity_str)
                    except:
                        pass
                elif line.startswith("节奏:"):
                    pacing_str = line.replace("节奏:", "").strip()
                    current_shot.pacing = pacing_str.lower()
                elif line.startswith("备注:"):
                    notes = line.replace("备注:", "").strip()
                    current_shot.notes = notes

            i += 1

        # 添加最后一个镜头
        if current_shot:
            shots.append(current_shot)

        # 如果没有解析到任何镜头，返回基础镜头
        if not shots:
            warning("AI响应解析失败，使用基础分镜头")
            return [self._create_establishing_shot(script)]

        return shots

    def _parse_shot_type(self, type_str: str) -> ShotType:
        """解析镜头类型字符串"""
        type_str = type_str.lower().replace(' ', '_').replace('-', '_')

        # 常见映射
        type_map = {
            "establishing": ShotType.ESTABLISHING,
            "wide": ShotType.WIDE,
            "wide_shot": ShotType.WIDE,
            "medium": ShotType.MEDIUM,
            "medium_shot": ShotType.MEDIUM,
            "close_up": ShotType.CLOSE_UP,
            "closeup": ShotType.CLOSE_UP,
            "extreme_close_up": ShotType.EXTREME_CLOSE_UP,
            "two_shot": ShotType.TWO_SHOT,
            "group_shot": ShotType.GROUP_SHOT,
            "over_the_shoulder": ShotType.OVER_THE_SHOULDER,
            "ots": ShotType.OVER_THE_SHOULDER,
            "pov": ShotType.POV,
            "reaction": ShotType.REACTION,
            "detail": ShotType.DETAIL,
            "insert": ShotType.INSERT,
        }

        return type_map.get(type_str, ShotType.MEDIUM)

    def _parse_shot_purpose(self, purpose_str: str) -> ShotPurpose:
        """解析镜头目的"""
        purpose_str = purpose_str.lower().replace(' ', '_')

        purpose_map = {
            "establish_location": ShotPurpose.ESTABLISH_LOCATION,
            "show_character": ShotPurpose.SHOW_CHARACTER,
            "dialogue": ShotPurpose.DIALOGUE,
            "reaction": ShotPurpose.REACTION,
            "action": ShotPurpose.ACTION,
            "emotion": ShotPurpose.EMOTION,
            "transition": ShotPurpose.TRANSITION,
            "reveal": ShotPurpose.REVEAL,
            "atmosphere": ShotPurpose.ATMOSPHERE,
            "detail": ShotPurpose.DETAIL,
        }

        return purpose_map.get(purpose_str, ShotPurpose.SHOW_CHARACTER)

    def _parse_camera_movement(self, movement_str: str) -> CameraMovement:
        """解析摄像机运动"""
        movement_str = movement_str.lower().replace(' ', '_')

        movement_map = {
            "none": CameraMovement.NONE,
            "static": CameraMovement.NONE,
            "pan_left": CameraMovement.PAN_LEFT,
            "pan_right": CameraMovement.PAN_RIGHT,
            "tilt_up": CameraMovement.TILT_UP,
            "tilt_down": CameraMovement.TILT_DOWN,
            "dolly_in": CameraMovement.DOLLY_IN,
            "dolly_out": CameraMovement.DOLLY_OUT,
            "tracking": CameraMovement.TRACKING,
            "handheld": CameraMovement.HANDHELD,
            "zoom_in": CameraMovement.ZOOM_IN,
            "zoom_out": CameraMovement.ZOOM_OUT,
        }

        return movement_map.get(movement_str, CameraMovement.NONE)

    def _merge_ai_suggestions(self, base_shots: List[Shot], ai_shots: List[Shot]) -> List[Shot]:
        """融合AI建议和基础分镜头"""
        if not ai_shots or len(ai_shots) < 2:
            return base_shots.copy()

        # 简单策略：用AI镜头替换部分基础镜头
        # 保持基础镜头的建立和结束部分，替换中间部分

        final_shots = []

        # 保留第一个基础镜头（通常是建立镜头）
        if base_shots:
            final_shots.append(base_shots[0])

        # 使用AI镜头的中间部分（排除第一个和最后一个）
        ai_middle = ai_shots[1:-1] if len(ai_shots) > 2 else ai_shots[1:]
        final_shots.extend(ai_middle)

        # 保留最后一个基础镜头（通常是结束镜头）
        if base_shots and len(base_shots) > 1:
            final_shots.append(base_shots[-1])
        elif ai_shots and len(ai_shots) > 0:
            final_shots.append(ai_shots[-1])

        # 如果没有镜头，创建默认
        if not final_shots:
            final_shots = base_shots.copy() if base_shots else ai_shots.copy()

        return final_shots

    def _calculate_ai_confidence(self, base_confidence: float) -> float:
        """计算AI增强后的置信度"""
        # AI增强通常会提高置信度，但幅度有限
        ai_boost = min(0.2, (1.0 - base_confidence) * 0.3)
        return round(min(0.95, base_confidence + ai_boost), 2)
