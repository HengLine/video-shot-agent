"""
@FileName: timeline_planner_factory.py
@Description: 时序规划器工厂
@Author: HengLine
@Time: 2026/1/14 17:08
"""
from typing import Dict, Any

from hengline.agent.script_parser2.script_parser_models import UnifiedScript
from hengline.agent.shot_generator.estimator.ai_action_estimator import AIActionDurationEstimator
from hengline.agent.shot_generator.estimator.ai_base_estimator import BaseAIDurationEstimator
from hengline.agent.shot_generator.estimator.ai_dialogue_estimator import AIDialogueDurationEstimator, AISilenceDurationEstimator
from hengline.agent.shot_generator.estimator.ai_scene_estimator import AISceneDurationEstimator
from hengline.agent.shot_generator.estimator.rule_action_estimator import RuleActionDurationEstimator
from hengline.agent.shot_generator.estimator.rule_base_estimator import BaseRuleDurationEstimator
from hengline.agent.shot_generator.estimator.rule_dialogue_estimator import RuleDialogueDurationEstimator
from hengline.agent.shot_generator.estimator.rule_scene_estimator import RuleSceneDurationEstimator
from hengline.agent.temporal_planner.temporal_planner_model import ElementType, DurationEstimation, EstimationSource
from hengline.prompts.temporal_planner_prompt import PromptConfig


class DurationEstimatorFactory:
    """时序规划器工厂类"""

    _llm_estimators = {}
    _rule_estimators = {}

    @classmethod
    def get_llm_estimator(cls, llm, element_type: ElementType, config: PromptConfig = None) -> BaseAIDurationEstimator:
        """获取指定类型的估算器"""
        if element_type not in cls._llm_estimators:
            cls._llm_estimators[element_type] = cls._create_llm_estimator(llm, element_type, config)

        return cls._llm_estimators[element_type]

    @classmethod
    def _create_llm_estimator(cls, llm, element_type: ElementType, config: PromptConfig = None) -> BaseAIDurationEstimator:
        """创建估算器"""
        config = config or PromptConfig()

        if element_type == ElementType.SCENE:
            return AISceneDurationEstimator(llm, config)
        elif element_type == ElementType.DIALOGUE:
            return AIDialogueDurationEstimator(llm, config)
        elif element_type == ElementType.SILENCE:
            return AISilenceDurationEstimator(llm, config)
        elif element_type == ElementType.ACTION:
            return AIActionDurationEstimator(llm, config)
        else:
            raise ValueError(f"不支持的元素类型: {element_type}")

    @classmethod
    def get_rule_estimator(cls, element_type: ElementType) -> BaseAIDurationEstimator:
        """获取指定类型的估算器"""
        if element_type not in cls._rule_estimators:
            cls._rule_estimators[element_type] = cls._create_rule_estimator(element_type)

        return cls._rule_estimators[element_type]

    @classmethod
    def _create_rule_estimator(cls, element_type: ElementType) -> BaseRuleDurationEstimator:
        """创建估算器"""
        if element_type == ElementType.SCENE:
            return RuleSceneDurationEstimator()
        elif element_type == ElementType.DIALOGUE:
            return RuleDialogueDurationEstimator()
        elif element_type == ElementType.SILENCE:
            return RuleDialogueDurationEstimator()
        elif element_type == ElementType.ACTION:
            return RuleActionDurationEstimator()
        else:
            raise ValueError(f"不支持的元素类型: {element_type}")

    @classmethod
    def estimate_element(cls, element_data: Any, estimator_type: EstimationSource, element_type: ElementType = None,
                         context: Dict = None, llm=None) -> DurationEstimation:
        """估算单个元素"""
        # 如果未指定元素类型，尝试自动推断
        if element_type is None:
            element_type = cls._infer_element_type(element_data)

        if estimator_type == EstimationSource.AI_LLM:
            estimator = cls.get_llm_estimator(llm, element_type)
        else:
            estimator = cls.get_rule_estimator(element_type)

        return estimator.estimate(element_data, context)

    @classmethod
    def estimate_script(cls, script_data: UnifiedScript, estimator_type: EstimationSource, context: Dict = None, llm=None) -> Dict[str, DurationEstimation]:
        """估算整个剧本"""
        results = {}

        # 估算场景
        for scene in script_data.scenes:
            result = cls.estimate_element(scene, estimator_type, ElementType.SCENE, context, llm)
            results[scene.scene_id] = result

        # 估算对话
        for dialogue in script_data.dialogues:
            result = cls.estimate_element(dialogue, estimator_type, ElementType.DIALOGUE, context, llm)
            results[dialogue.dialogue_id] = result

        # 估算动作
        for action in script_data.actions:
            result = cls.estimate_element(action, estimator_type, ElementType.ACTION, context, llm)
            results[action.action_id] = result

        return results

    @classmethod
    def estimate_script_with_llm(cls, llm, script_data: UnifiedScript, context: Dict = None) -> Dict[str, DurationEstimation]:
        """使用LLM估算整个剧本"""
        return cls.estimate_script(script_data, EstimationSource.AI_LLM, context, llm)

    @classmethod
    def estimate_script_with_rules(cls, script_data: UnifiedScript, context: Dict = None) -> Dict[str, DurationEstimation]:
        """使用规则估算整个剧本"""
        return cls.estimate_script(script_data, EstimationSource.LOCAL_RULE, context)

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
        for estimator in cls._llm_estimators.values():
            estimator.clear_cache()
        for estimator in cls._rule_estimators.values():
            estimator.clear_cache()

    @classmethod
    def get_error_summary(cls) -> Dict[str, Any]:
        """获取所有估算器的错误摘要"""
        llm_all_errors = []
        rule_all_errors = []

        for element_type, estimator in cls._llm_estimators.items():
            errors = estimator.get_error_summary()
            if errors["total_errors"] > 0:
                llm_all_errors.append({
                    "element_type": element_type.value,
                    **errors
                })

        for element_type, estimator in cls._rule_estimators.items():
            errors = estimator.get_error_summary()
            if errors["total_errors"] > 0:
                rule_all_errors.append({
                    "element_type": element_type.value,
                    **errors
                })

        return {
            "llm_estimators": {
                "total_estimators": len(cls._llm_estimators),
                "estimators_with_errors": len(llm_all_errors),
                "errors_by_estimator": llm_all_errors
            },
            "rule_estimators": {
                "total_estimators": len(cls._rule_estimators),
                "estimators_with_errors": len(rule_all_errors),
                "errors_by_estimator": rule_all_errors
            }
        }


estimator_factory = DurationEstimatorFactory()
