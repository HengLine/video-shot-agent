"""
@FileName: ai_action_estimator.py
@Description: 动作时长估算器
@Author: HengLine
@Time: 2026/1/14 17:00
"""
import json
from typing import Dict, Any

from hengline.agent.script_parser2.script_parser_models import Action
from hengline.agent.temporal_planner.estimator.ai_base_estimator import BaseAIDurationEstimator
from hengline.agent.temporal_planner.estimator.base_estimator import EstimationErrorLevel
from hengline.agent.temporal_planner.temporal_planner_model import ElementType, DurationEstimation, EstimationSource


class AIActionDurationEstimator(BaseAIDurationEstimator):
    """动作时长估算器"""

    def _get_element_type(self) -> ElementType:
        return ElementType.ACTION

    def _get_id_value(self, action_data: Action) -> str:
        return action_data.action_id

    def _generate_prompt(self, action_data: Action, context: Dict = None) -> str:
        """生成动作估算提示词"""
        return self.prompt_templates.action_duration_prompt(action_data, context)

    def _parse_ai_response(self, response: str, action_data: Action) -> Dict[str, Any] | None:
        """解析动作AI响应"""
        try:
            cleaned_response = self._clean_json_response(response)
            data = json.loads(cleaned_response)

            return {
                "estimated_duration": float(data.get("estimated_duration", 1.5)),
                "confidence": float(data.get("confidence", 0.7)),
                "reasoning": data.get("reasoning_breakdown", {}),
                "visual_hints": data.get("visual_hints", {}),
                "action_components": data.get("action_components", []),
                "duration_breakdown": data.get("duration_breakdown", {}),
                "key_factors": data.get("key_factors", []),
                "pacing_notes": data.get("pacing_notes", ""),
                "continuity_requirements": data.get("continuity_requirements", [])
            }

        except (json.JSONDecodeError, ValueError) as e:
            self._log_error(
                element_id=action_data.action_id,
                error_type="response_parse_error",
                message=f"动作响应解析失败: {str(e)}",
                level=EstimationErrorLevel.WARNING
            )
            return None

    def _validate_estimation(self, parsed_result: Dict, action_data: Action) -> DurationEstimation:
        """验证动作估算结果"""
        if parsed_result is None:
            return self._create_fallback_estimation(action_data)

        duration = parsed_result.get("estimated_duration", 0.0)
        confidence = parsed_result.get("confidence", 0.0)

        # 验证动作时长合理性（通常0.5-8秒）
        if duration < 0.3:
            duration = 0.5
            confidence = min(confidence, 0.6)
            self._log_error(
                element_id=action_data.action_id,
                error_type="action_too_short",
                message=f"动作时长过短: {duration}秒",
                level=EstimationErrorLevel.WARNING,
                recovery_action="调整为0.5秒"
            )
        elif duration > 12:
            duration = min(duration, 10.0)
            confidence = min(confidence, 0.6)
            self._log_error(
                element_id=action_data.action_id,
                error_type="action_too_long",
                message=f"动作时长过长: {duration}秒",
                level=EstimationErrorLevel.WARNING,
                recovery_action="限制为10.0秒"
            )

        # 创建DurationEstimation对象
        result = DurationEstimation(
            element_id=action_data.action_id,
            element_type=ElementType.ACTION,
            estimated_duration=round(duration, 2),
            llm_estimated=round(duration, 2),
            estimator_source=EstimationSource.AI_LLM,
            original_duration=round(action_data.duration, 2),
            confidence=round(confidence, 2),
            reasoning_breakdown=parsed_result.get("reasoning", {}),
            visual_hints=parsed_result.get("visual_hints", {}),
            duration_breakdown=parsed_result.get("duration_breakdown", {}),
            key_factors=parsed_result.get("key_factors", []),
            pacing_notes=parsed_result.get("pacing_notes", ""),
            continuity_requirements=parsed_result.get("continuity_requirements", {})
        )

        return result

    def _create_fallback_estimation(self, action_data: Action, context: Dict = None) -> DurationEstimation:
        """动作降级估算"""

        word_count = len(action_data.description.split())

        # 基于类型的基础时长
        type_baselines = {
            "posture": 2.0,
            "gaze": 1.5,
            "gesture": 1.2,
            "facial": 1.0,
            "physiological": 0.8,
            "interaction": 1.5,
            "prop_fall": 1.0,
            "device_alert": 2.0
        }

        base_duration = type_baselines.get(action_data.type, 1.5)

        # 根据描述复杂度调整
        complexity_factor = min(word_count / 5.0, 3.0)

        total_duration = base_duration * complexity_factor

        return DurationEstimation(
            element_id=action_data.action_id,
            element_type=ElementType.ACTION,
            estimated_duration=round(total_duration, 2),
            llm_estimated=round(total_duration, 2),
            estimator_source=EstimationSource.FALLBACK,
            original_duration=round(action_data.duration, 2),
            confidence=0.4,
            reasoning_breakdown={"fallback_estimation": True, "method": "type_and_complexity"},
            visual_hints={"fallback": True},
            key_factors=["fallback_estimation"],
            pacing_notes="降级估算，基于类型和复杂度"
        )

    def _enhance_estimation(self, result: DurationEstimation, action_data: Action) -> DurationEstimation:
        """增强动作估算"""
        description = action_data.description

        # 计算复杂度得分
        word_count = len(description.split())
        length_score = min(word_count / 8.0, 3.0)

        components = result.reasoning.get("action_components", [])
        component_score = min(len(components) / 3.0, 2.0)

        complexity_score = (length_score + component_score) / 2.0

        # 添加到reasoning
        if "reasoning" not in result.reasoning:
            result.reasoning = {}
        result.reasoning["complexity_score"] = round(complexity_score, 2)

        # 检测关键动作
        key_action_indicators = ["按下接听键", "手指瞬间收紧", "泪水在眼眶中打转", "猛地坐直"]

        for indicator in key_action_indicators:
            if indicator in description:
                result.key_factors.append("关键转折动作")
                result.visual_hints["dramatic_emphasis"] = True
                result.visual_hints["slow_motion_consideration"] = True
                break

        # 提取连续性信息
        self._extract_continuity_info(result, action_data)

        return result

    def _extract_continuity_info(self, result: DurationEstimation, action_data: Action):
        """提取连续性信息"""
        actor = action_data.actor
        description = action_data.description

        state_changes = []

        if "坐直" in description:
            state_changes.append("posture:从蜷坐到挺直")
        if "手指收紧" in description:
            state_changes.append("hand_tension:放松到紧绷")
        if "泪水" in description:
            state_changes.append("emotional_state:震惊到悲伤")

        if "旧羊毛毯" in actor and "滑落" in description:
            state_changes.append("prop_position:从肩头到地板")

        if state_changes:
            result.continuity_requirements.extend(state_changes)
