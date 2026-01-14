"""
@FileName: timeline_planner_factory.py
@Description: 时序规划器工厂
@Author: HengLine
@Time: 2026/1/14 17:08
"""
from typing import Dict, Any

from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.estimator.ai_action_estimator import AIActionDurationEstimator
from hengline.agent.temporal_planner.estimator.ai_base_estimator import BaseAIDurationEstimator
from hengline.agent.temporal_planner.estimator.ai_dialogue_estimator import AIDialogueDurationEstimator, AISilenceDurationEstimator
from hengline.agent.temporal_planner.estimator.ai_scene_estimator import AISceneDurationEstimator
from hengline.agent.temporal_planner.temporal_planner_model import ElementType, DurationEstimation
from hengline.prompts.temporal_planner_prompt import PromptConfig


class TimelinePlannerFactory:
    """时序规划器工厂类"""

    _estimators = {}

    @classmethod
    def get_estimator(cls, element_type: ElementType, config: PromptConfig = None) -> BaseAIDurationEstimator:
        """获取指定类型的估算器"""
        if element_type not in cls._estimators:
            cls._estimators[element_type] = cls._create_estimator(element_type, config)

        return cls._estimators[element_type]

    @classmethod
    def _create_estimator(cls, element_type: ElementType, config: PromptConfig = None) -> BaseAIDurationEstimator:
        """创建估算器"""
        config = config or PromptConfig()

        if element_type == ElementType.SCENE:
            return AISceneDurationEstimator(config)
        elif element_type == ElementType.DIALOGUE:
            return AIDialogueDurationEstimator(config)
        elif element_type == ElementType.SILENCE:
            return AISilenceDurationEstimator(config)
        elif element_type == ElementType.ACTION:
            return AIActionDurationEstimator(config)
        else:
            raise ValueError(f"不支持的元素类型: {element_type}")

    @classmethod
    def estimate_element(cls, element_data: Any, element_type: ElementType = None,
                         context: Dict = None) -> DurationEstimation:
        """估算单个元素"""
        # 如果未指定元素类型，尝试自动推断
        if element_type is None:
            element_type = cls._infer_element_type(element_data)

        estimator = cls.get_estimator(element_type)
        return estimator.estimate(element_data, context)

    @classmethod
    def estimate_script(cls, script_data: UnifiedScript) -> Dict[str, DurationEstimation]:
        """估算整个剧本"""
        results = {}

        # 估算场景
        for scene in script_data.scenes:
            result = cls.estimate_element(scene, ElementType.SCENE)
            results[scene.scene_id] = result

        # 估算对话
        for dialogue in script_data.dialogues:
            result = cls.estimate_element(dialogue, ElementType.DIALOGUE)
            results[dialogue.dialogue_id] = result

        # 估算动作
        for action in script_data.actions:
            result = cls.estimate_element(action, ElementType.ACTION)
            results[action.action_id] = result

        return results

    @classmethod
    def _infer_element_type(cls, element_data: Any) -> ElementType:
        """推断元素类型"""
        model_fields = element_data.model_fields

        if "scene_id" in model_fields:
            return ElementType.SCENE
        elif "dialogue_id" in model_fields:
            # 检查是否为沉默
            if element_data.type == "silence" or not element_data.content.strip():
                return ElementType.SILENCE
            return ElementType.DIALOGUE
        elif "action_id" in model_fields:
            return ElementType.ACTION
        else:
            raise ValueError("无法推断元素类型")

    @classmethod
    def clear_all_cache(cls):
        """清空所有估算器的缓存"""
        for estimator in cls._estimators.values():
            estimator.clear_cache()

    @classmethod
    def get_error_summary(cls) -> Dict[str, Any]:
        """获取所有估算器的错误摘要"""
        all_errors = []

        for element_type, estimator in cls._estimators.items():
            errors = estimator.get_error_summary()
            if errors["total_errors"] > 0:
                all_errors.append({
                    "element_type": element_type.value,
                    **errors
                })

        return {
            "total_estimators": len(cls._estimators),
            "estimators_with_errors": len(all_errors),
            "errors_by_estimator": all_errors
        }
