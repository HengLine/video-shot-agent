"""
@FileName: ai_dialogue_estimator.py
@Description: 对话时长估算器
@Author: HengLine
@Time: 2026/1/14 16:33
"""
import json
from typing import Dict, Any

from hengline.agent.script_parser.script_parser_models import Dialogue
from hengline.agent.temporal_planner.estimator.ai_base_estimator import BaseAIDurationEstimator
from hengline.agent.temporal_planner.estimator.base_estimator import EstimationErrorLevel
from hengline.agent.temporal_planner.temporal_planner_model import ElementType, DurationEstimation, EstimationSource


class AIDialogueDurationEstimator(BaseAIDurationEstimator):
    """对话时长估算器（包含沉默）"""

    def _get_element_type(self) -> ElementType:
        return ElementType.DIALOGUE

    def _get_id_value(self, dialogue_data: Dialogue) -> str:
        return dialogue_data.dialogue_id

    def estimate(self, dialogue_data: Dialogue, context: Dict = None) -> DurationEstimation:
        """估算对话时长（自动识别沉默）"""
        # 检查是否为沉默
        if self._is_silence(dialogue_data):
            # 使用专门的沉默估算逻辑
            return self._estimate_silence(dialogue_data, context)

        return self.estimate_with_context(dialogue_data, context)

    def _is_silence(self, dialogue_data: Dialogue) -> bool:
        """检查是否为沉默"""
        return (dialogue_data.type == "silence" or not dialogue_data.content.strip())

    def _estimate_silence(self, dialogue_data: Dialogue, context: Dict = None) -> DurationEstimation:
        """估算沉默时长"""
        # 创建专门的沉默估算器（或重用当前类）
        silence_estimator = AISilenceDurationEstimator(self.config)
        return silence_estimator.estimate(dialogue_data, context)

    def _generate_prompt(self, dialogue_data: Dialogue, context: Dict = None) -> str:
        """生成对话估算提示词"""
        return self.prompt_templates.dialogue_duration_prompt(dialogue_data, context)

    def _parse_ai_response(self, response: str, dialogue_data: Dialogue) -> dict[str, float | Any] | None:
        """解析对话AI响应"""
        try:
            cleaned_response = self._clean_json_response(response)
            data = json.loads(cleaned_response)

            return {
                "estimated_duration": float(data.get("estimated_duration", 2.0)),
                "confidence": float(data.get("confidence", 0.7)),
                "reasoning": data.get("reasoning_breakdown", {}),
                "visual_hints": data.get("visual_hints", {}),
                "speech_characteristics": data.get("speech_characteristics", {}),
                "duration_breakdown": data.get("duration_breakdown", {}),
                "key_factors": data.get("key_factors", []),
                "pacing_notes": data.get("pacing_notes", "")
            }

        except (json.JSONDecodeError, ValueError) as e:
            self._log_error(
                element_id=dialogue_data.dialogue_id,
                error_type="response_parse_error",
                message=f"对话响应解析失败: {str(e)}",
                level=EstimationErrorLevel.WARNING
            )
            return None

    def _validate_estimation(self, parsed_result: Dict, dialogue_data: Dialogue) -> DurationEstimation:
        """验证对话估算结果"""
        if parsed_result is None:
            return self._create_fallback_estimation(dialogue_data)

        duration = parsed_result.get("estimated_duration", 0)
        confidence = parsed_result.get("confidence", 0.0)

        # 验证词速合理性
        content = dialogue_data.content
        word_count = len(content.split())

        if word_count > 0:
            seconds_per_word = duration / word_count if word_count > 0 else 0

            if seconds_per_word < 0.1:
                duration = word_count * 0.4  # 调整为0.4秒/词
                confidence = min(confidence, 0.6)
                self._log_error(
                    element_id=dialogue_data.dialogue_id,
                    error_type="too_fast_speech",
                    message=f"语速过快: {seconds_per_word:.2f}秒/词",
                    level=EstimationErrorLevel.WARNING,
                    recovery_action="调整为基础语速"
                )
            elif seconds_per_word > 1.5:
                duration = word_count * 1.0
                confidence = min(confidence, 0.6)
                self._log_error(
                    element_id=dialogue_data.dialogue_id,
                    error_type="too_slow_speech",
                    message=f"语速过慢: {seconds_per_word:.2f}秒/词",
                    level=EstimationErrorLevel.WARNING,
                    recovery_action="限制为1.0秒/词"
                )

        # 创建DurationEstimation对象
        result = DurationEstimation(
            element_id=dialogue_data.dialogue_id,
            element_type=ElementType.DIALOGUE,
            original_duration=round(dialogue_data.duration, 2),
            estimated_duration=round(duration, 2),
            llm_estimated=round(duration, 2),
            estimator_source=EstimationSource.AI_LLM,
            confidence=round(confidence, 2),
            reasoning_breakdown=parsed_result.get("reasoning", {}),
            visual_hints=parsed_result.get("visual_hints", {}),
            duration_breakdown=parsed_result.get("duration_breakdown", {}),
            key_factors=parsed_result.get("key_factors", []),
            pacing_notes=parsed_result.get("pacing_notes", "")
        )

        return result

    def _create_fallback_estimation(self, dialogue_data: Dialogue, context: Dict = None) -> DurationEstimation:
        """对话降级估算"""
        emotion = dialogue_data.emotion

        word_count = len(dialogue_data.content.split())

        # 基础语速
        words_per_second = 2.5

        # 情绪调整
        if "微颤" in emotion or "哽咽" in emotion:
            words_per_second = 1.8
        elif "快速" in emotion:
            words_per_second = 3.2

        base_duration = word_count / words_per_second if words_per_second > 0 else 0

        # 添加停顿
        pause = 0.3 if word_count > 0 else 0

        total_duration = base_duration + pause

        return DurationEstimation(
            element_id=dialogue_data.dialogue_id,
            element_type=ElementType.DIALOGUE,
            original_duration=round(dialogue_data.duration, 2),
            estimated_duration=round(total_duration, 2),
            llm_estimated=round(total_duration, 2),
            estimator_source=EstimationSource.FALLBACK,
            confidence=0.4,
            reasoning_breakdown={"fallback_estimation": True, "method": "word_count_with_emotion"},
            visual_hints={"fallback": True},
            key_factors=["fallback_estimation"],
            pacing_notes="降级估算，基于词数和情绪"
        )

    def _enhance_estimation(self, result: DurationEstimation, dialogue_data: Dialogue) -> DurationEstimation:
        """增强对话估算"""
        # 计算情感权重
        emotion = dialogue_data.emotion
        content = dialogue_data.content

        emotional_weight = 1.0
        if "微颤" in emotion:
            emotional_weight += 0.5
        if "哽咽" in emotion:
            emotional_weight += 1.0
        if "陈默" in content:  # 关键名字
            emotional_weight += 0.3

        # 添加到reasoning
        if "reasoning" not in result.reasoning:
            result.reasoning = {}
        result.reasoning["emotional_weight"] = round(emotional_weight, 2)

        # 添加情感轨迹（如果AI没有提供）
        if not result.emotional_trajectory and emotional_weight > 1.5:
            result.emotional_trajectory = [
                {"time": 0.0, "emotion": "anticipation", "intensity": 5},
                {"time": result.estimated_duration * 0.3, "emotion": "expression", "intensity": 7},
                {"time": result.estimated_duration * 0.8, "emotion": "lingering", "intensity": 6}
            ]

        return result


class AISilenceDurationEstimator(AIDialogueDurationEstimator):
    """沉默时长估算器（继承自对话估算器）"""

    def _get_element_type(self) -> ElementType:
        return ElementType.SILENCE

    def _parse_ai_response(self, response: str, dialogue_data: Dialogue) -> Dict[str, Any] | None:
        """解析沉默AI响应"""
        try:
            cleaned_response = self._clean_json_response(response)
            data = json.loads(cleaned_response)

            return {
                "estimated_duration": float(data.get("estimated_duration", 2.5)),
                "confidence": float(data.get("confidence", 0.7)),
                "reasoning": data.get("reasoning_breakdown", {}),
                "visual_hints": data.get("visual_hints", {}),
                "silence_type": data.get("silence_type", "emotional"),
                "duration_breakdown": data.get("duration_breakdown", {}),
                "key_factors": data.get("key_factors", []),
                "pacing_notes": data.get("pacing_notes", ""),
                "continuity_requirements": data.get("continuity_requirements", [])
            }

        except (json.JSONDecodeError, ValueError) as e:
            self._log_error(
                element_id=dialogue_data.dialogue_id,
                error_type="response_parse_error",
                message=f"沉默响应解析失败: {str(e)}",
                level=EstimationErrorLevel.WARNING
            )
            return None

    def _validate_estimation(self, parsed_result: Dict, dialogue_data: Dialogue) -> DurationEstimation:
        """验证沉默估算结果"""
        if parsed_result is None:
            return self._create_fallback_estimation(dialogue_data)

        duration = parsed_result.get("estimated_duration", 0.0)
        confidence = parsed_result.get("confidence", 0.0)

        # 验证沉默时长合理性（通常1-8秒）
        if duration < 0.5:
            duration = 1.0
            confidence = min(confidence, 0.6)
            self._log_error(
                element_id=dialogue_data.dialogue_id,
                error_type="silence_too_short",
                message=f"沉默时长过短: {duration}秒",
                level=EstimationErrorLevel.WARNING,
                recovery_action="调整为1.0秒"
            )
        elif duration > 10:
            duration = min(duration, 8.0)
            confidence = min(confidence, 0.6)
            self._log_error(
                element_id=dialogue_data.dialogue_id,
                error_type="silence_too_long",
                message=f"沉默时长过长: {duration}秒",
                level=EstimationErrorLevel.WARNING,
                recovery_action="限制为8.0秒"
            )

        # 创建DurationEstimation对象
        result = DurationEstimation(
            element_id=dialogue_data.dialogue_id,
            element_type=ElementType.SILENCE,
            estimated_duration=round(duration, 2),
            llm_estimated=round(duration, 2),
            estimator_source=EstimationSource.AI_LLM,
            original_duration=round(dialogue_data.duration, 2),
            confidence=round(confidence, 2),
            reasoning_breakdown=parsed_result.get("reasoning", {}),
            visual_hints=parsed_result.get("visual_hints", {}),
            duration_breakdown=parsed_result.get("duration_breakdown", {}),
            key_factors=parsed_result.get("key_factors", []),
            pacing_notes=parsed_result.get("pacing_notes", ""),
            continuity_requirements=parsed_result.get("continuity_requirements", {})
        )

        return result

    def _create_fallback_estimation(self, dialogue_data: Dialogue, context: Dict = None) -> DurationEstimation:
        """沉默降级估算"""
        parenthetical = dialogue_data.parenthetical

        # 基于动作描述的沉默时长
        base_duration = 2.0

        if "张了张嘴" in parenthetical:
            base_duration = 3.0
        elif "震惊" in parenthetical or "愣住" in parenthetical:
            base_duration = 3.5

        return DurationEstimation(
            element_id=dialogue_data.dialogue_id,
            element_type=ElementType.SILENCE,
            original_duration=round(dialogue_data.duration, 2),
            estimated_duration=round(base_duration, 2),
            llm_estimated=round(base_duration, 2),
            estimator_source=EstimationSource.FALLBACK,
            confidence=0.5,
            reasoning_breakdown={"fallback_estimation": True, "method": "parenthetical_based"},
            visual_hints={"fallback": True, "suggested_shot_types": ["close_up"]},
            key_factors=["fallback_estimation"],
            pacing_notes="降级估算，基于动作描述"
        )
