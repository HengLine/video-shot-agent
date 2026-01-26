"""
@FileName: ai_scene_estimator.py
@Description: 场景时长估算器
@Author: HengLine
@Time: 2026/1/14 16:21
"""

import json
from typing import Dict, Any

from hengline.agent.script_parser2.script_parser_models import Scene
from hengline.agent.shot_generator.estimator.ai_base_estimator import BaseAIDurationEstimator
from hengline.agent.shot_generator.estimator.base_estimator import EstimationErrorLevel
from hengline.agent.temporal_planner.temporal_planner_model import ElementType, DurationEstimation, EstimationSource


class AISceneDurationEstimator(BaseAIDurationEstimator):
    """场景时长估算器"""

    def _get_element_type(self) -> ElementType:
        return ElementType.SCENE

    def _get_id_value(self, scene_data: Scene) -> str:
        return scene_data.scene_id

    def _generate_prompt(self, scene_data: Scene, context: Dict = None) -> str:
        """生成场景估算提示词"""
        return self.prompt_templates.scene_duration_prompt(scene_data, context)

    def _parse_ai_response(self, response: str, scene_data: Scene) -> dict[str, float | Any] | None:
        """解析场景AI响应"""
        try:
            cleaned_response = self._clean_json_response(response)
            data = json.loads(cleaned_response)

            return {
                "estimated_duration": float(data.get("estimated_duration", 3.0)),
                "confidence": float(data.get("confidence", 0.7)),
                "reasoning": data.get("reasoning_breakdown", {}),
                "visual_hints": data.get("visual_hints", {}),
                "duration_breakdown": data.get("duration_breakdown", {}),
                "key_factors": data.get("key_factors", []),
                "pacing_notes": data.get("pacing_notes", ""),
                "continuity_requirements": data.get("continuity_requirements", []),
                "shot_suggestions": data.get("shot_suggestions", [])
            }

        except (json.JSONDecodeError, ValueError) as e:
            self._log_error(
                element_id=scene_data.scene_id,
                error_type="response_parse_error",
                message=f"场景响应解析失败: {str(e)}",
                level=EstimationErrorLevel.WARNING
            )
            return None

    def _validate_estimation(self, parsed_result: Dict, scene_data: Scene) -> DurationEstimation:
        """验证场景估算结果"""
        if parsed_result is None:
            return self._create_fallback_estimation(scene_data)

        duration = parsed_result.get("estimated_duration", 0.0)
        confidence = parsed_result.get("confidence", 0.0)

        # 验证时长合理性
        if duration <= 0:
            duration = 4.0
            confidence = min(confidence, 0.5)
            self._log_error(
                element_id=scene_data.scene_id,
                error_type="invalid_duration",
                message=f"场景时长无效: {duration}秒",
                level=EstimationErrorLevel.WARNING,
                recovery_action="使用默认值4.0秒"
            )

        # 验证场景时长范围（通常1-15秒）
        if duration > 20:
            duration = min(duration, 15.0)
            confidence = min(confidence, 0.6)
            self._log_error(
                element_id=scene_data.scene_id,
                error_type="excessive_duration",
                message=f"场景时长过长: {duration}秒",
                level=EstimationErrorLevel.WARNING,
                recovery_action="限制为15.0秒"
            )

        # 创建DurationEstimation对象
        result = DurationEstimation(
            element_id=scene_data.scene_id,
            element_type=ElementType.SCENE,
            original_duration=round(scene_data.duration, 2),
            estimated_duration=round(duration, 2),
            llm_estimated=round(duration, 2),
            estimator_source=EstimationSource.AI_LLM,
            confidence=round(confidence, 2),
            reasoning_breakdown=parsed_result.get("reasoning", {}),
            visual_hints=parsed_result.get("visual_hints", {}),
            duration_breakdown=parsed_result.get("duration_breakdown", {}),
            key_factors=parsed_result.get("key_factors", []),
            pacing_notes=parsed_result.get("pacing_notes", ""),
            continuity_requirements=parsed_result.get("continuity_requirements", {}),
            shot_suggestions=parsed_result.get("shot_suggestions", [])
        )

        return result

    def _create_fallback_estimation(self, scene_data: Scene, context: Dict = None) -> DurationEstimation:
        """场景降级估算"""
        mood = scene_data.mood

        # 基于简单规则的降级估算
        word_count = len(scene_data.description.split())
        base_duration = word_count * 0.06

        # 关键视觉元素加成
        visual_bonus = len(scene_data.key_visuals) * 0.4

        # 情绪加成
        mood_bonus = 0
        if "紧张" in mood or "压抑" in mood:
            mood_bonus = 1.5
        elif "孤独" in mood:
            mood_bonus = 1.0

        total_duration = base_duration + visual_bonus + mood_bonus
        total_duration = max(2.0, min(total_duration, 12.0))

        return DurationEstimation(
            element_id=scene_data.scene_id,
            element_type=ElementType.SCENE,
            original_duration=round(scene_data.duration, 2),
            estimated_duration=round(total_duration, 2),
            llm_estimated=round(total_duration, 2),
            estimator_source=EstimationSource.FALLBACK,
            confidence=0.4,
            reasoning_breakdown={"fallback_estimation": True, "method": "word_count_based"},
            visual_hints={"fallback": True, "suggested_shot_types": ["establishing_shot"]},
            key_factors=["fallback_estimation"],
            pacing_notes="降级估算，建议人工审核"
        )

    def _enhance_estimation(self, result: DurationEstimation, scene_data: Scene) -> DurationEstimation:
        """增强场景估算"""
        # 计算视觉复杂度
        visual_complexity = len(scene_data.key_visuals) / 5.0  # 归一化

        # 添加到reasoning
        if "reasoning" not in result.reasoning:
            result.reasoning = {}
        result.reasoning["visual_complexity"] = round(visual_complexity, 2)

        # 根据氛围调整情感权重
        mood = scene_data.mood
        if any(emotion in mood for emotion in ["紧张", "压抑", "悲伤"]):
            # 添加情感轨迹
            if not result.emotional_trajectory:
                result.emotional_trajectory = [
                    {"time": 0.0, "emotion": "atmospheric", "intensity": 6},
                    {"time": result.estimated_duration * 0.5, "emotion": "immersive", "intensity": 7},
                    {"time": result.estimated_duration, "emotion": "lingering", "intensity": 5}
                ]

        return result
