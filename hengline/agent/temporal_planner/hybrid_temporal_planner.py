"""
@FileName: hybrid_temporal_planner.py
@Description: 混合时序规划器（组合LLM和规则规划器的结果）
@Author: HengLine
@Time: 2026/1/15 17:13
"""
from abc import ABC
from typing import Dict, Any, List

from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.base_temporal_planner import BaseTemporalPlanner
from hengline.agent.temporal_planner.llm_temporal_planner import LLMTemporalPlanner
from hengline.agent.temporal_planner.local_temporal_planner import LocalRuleTemporalPlanner
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation
from hengline.agent.temporal_planner.temporal_planner_model import EstimationSource, ElementType
from hengline.logger import info, warning, debug


class HybridTemporalPlanner(BaseTemporalPlanner, ABC):
    """混合时序规划器 - 合并LLM和规则结果"""

    def __init__(self, llm):
        super().__init__()
        # 初始化子规划器
        self.llm_planner = LLMTemporalPlanner(llm)
        self.rule_planner = LocalRuleTemporalPlanner()

        # 混合策略配置
        self.mixing_strategy = {
            "default_blend_ratio": 0.5,  # 默认LLM:规则 = 50:50
            "llm_priority_elements": ["scene", "dialogue", "silence"],  # LLM优先的元素类型
            "rule_priority_elements": ["action"],  # 规则优先的元素类型
            "confidence_threshold": 0.7,  # 置信度阈值
            "max_duration_diff_ratio": 0.5  # 最大时长差异比率
        }

    def estimate_all_elements(self, script_data: UnifiedScript) -> Dict[str, DurationEstimation]:
        """
        估算所有元素 - 实现基类抽象方法
        合并LLM和规则的结果
        """
        # 1. 获取LLM估算结果
        debug("开始LLM估算...")
        llm_estimations = self.llm_planner.estimate_all_elements(script_data)

        # 2. 获取规则估算结果
        debug("开始规则估算...")
        rule_estimations = self.rule_planner.estimate_all_elements(script_data)

        # 3. 合并两个结果
        info("开始合并估算结果...")
        merged_estimations = {}

        #
        self._merge_element_estimations(script_data.scenes, ElementType.SCENE, llm_estimations, rule_estimations, merged_estimations)
        self._merge_element_estimations(script_data.dialogues, ElementType.DIALOGUE, llm_estimations, rule_estimations, merged_estimations)
        self._merge_element_estimations(script_data.actions, ElementType.ACTION, llm_estimations, rule_estimations, merged_estimations)


        return merged_estimations

    def _merge_element_estimations(self, elements: List[Any], element_type: ElementType, llm_estimations, rule_estimations, merged_estimations):
        """
        合并指定类型元素的估算结果
        """
        for element in elements:
            if element_type == ElementType.SCENE:
                element_id = element.scene_id
            elif element_type == ElementType.DIALOGUE or element_type == ElementType.SILENCE:
                element_id = element.dialogue_id
            elif element_type == ElementType.ACTION:
                element_id = element.action_id
            elif element_type == ElementType.TRANSITION:
                element_id = element.transition_id
            else:
                continue  # 未知元素类型，跳过

            llm_est = llm_estimations.get(element_id)
            rule_est = rule_estimations.get(element_id)

            if llm_est and rule_est:
                # 两个都有值，按照规则合并
                merged = self._merge_estimations(llm_est, rule_est, element_id, element.duration, element_type)
            elif llm_est:
                # 只有LLM有值
                merged = llm_est
                merged.source = EstimationSource.LLM
            elif rule_est:
                # 只有规则有值
                merged = rule_est
                merged.source = EstimationSource.LOCAL_RULE
            else:
                # 两个都没有值，使用降级估算
                merged = self._create_fallback_estimation(element)

            merged_estimations[element_id] = merged

    def _merge_estimations(self, llm_est: DurationEstimation,
                           rule_est: DurationEstimation,
                           element_id: str, duration: float, element_type: ElementType) -> DurationEstimation:
        """
        合并LLM和规则估算结果
        按照指定规则：取中间值或非零值
        """
        llm_duration = llm_est.estimated_duration
        rule_duration = rule_est.estimated_duration
        llm_confidence = llm_est.confidence
        rule_confidence = rule_est.confidence

        # 应用合并规则
        if llm_duration > 0 and rule_duration > 0:
            # 两个都有值，取加权平均
            blend_ratio = self._get_blend_ratio(element_type, llm_confidence, rule_confidence)
            final_duration = (llm_duration * blend_ratio +
                              rule_duration * (1 - blend_ratio))

            # 检查差异是否过大
            diff_ratio = abs(llm_duration - rule_duration) / min(llm_duration, rule_duration)
            if diff_ratio > self.mixing_strategy["max_duration_diff_ratio"]:
                warning(f"估算差异过大 ({diff_ratio:.1%}): "
                        f"LLM={llm_duration:.2f}, 规则={rule_duration:.2f}")
                # 差异过大时，根据置信度选择
                if llm_confidence > rule_confidence:
                    final_duration = llm_duration
                else:
                    final_duration = rule_duration

            source = EstimationSource.HYBRID

        elif llm_duration > 0:
            # 只有LLM有值
            final_duration = llm_duration
            source = EstimationSource.LLM

        elif rule_duration > 0:
            # 只有规则有值
            final_duration = rule_duration
            source = EstimationSource.LOCAL_RULE

        else:
            # 两个都没有值，使用降级
            final_duration = self._get_default_duration(element_type)
            source = EstimationSource.FALLBACK

        # 合并置信度
        if llm_confidence > 0 and rule_confidence > 0:
            final_confidence = (llm_confidence + rule_confidence) / 2
        elif llm_confidence > 0:
            final_confidence = llm_confidence * 0.8  # 单一来源降低置信度
        elif rule_confidence > 0:
            final_confidence = rule_confidence * 0.8
        else:
            final_confidence = 0.5

        # 合并其他属性
        emotional_weight = max(llm_est.emotional_weight, rule_est.emotional_weight)
        visual_complexity = max(llm_est.visual_complexity, rule_est.visual_complexity)

        # 合并视觉提示（优先使用LLM的）
        visual_hints = llm_est.visual_hints or rule_est.visual_hints or {}

        # 合并连续性要求
        continuity_requirements = dict(set(
            llm_est.continuity_requirements + rule_est.continuity_requirements
        ))

        # 创建最终估算结果
        final_estimation = DurationEstimation(
            element_id=element_id,
            element_type=element_type,
            estimated_duration=final_duration,
            original_duration=duration,
            llm_estimated=llm_duration if llm_duration > 0 else None,
            rule_estimated=rule_duration if rule_duration > 0 else None,
            estimator_source=source,
            confidence=final_confidence,
            emotional_weight=emotional_weight,
            visual_complexity=visual_complexity,
            visual_hints=visual_hints,
            continuity_requirements=continuity_requirements,
            character_states={**llm_est.character_states, **rule_est.character_states},
            prop_states={**llm_est.prop_states, **rule_est.prop_states}
        )

        return final_estimation

    def _get_blend_ratio(self, element_type: ElementType,
                         llm_confidence: float, rule_confidence: float) -> float:
        """
        获取混合比例
        """
        # 根据元素类型决定基础比例
        if element_type in self.mixing_strategy["llm_priority_elements"]:
            base_ratio = 0.7  # LLM优先
        elif element_type in self.mixing_strategy["rule_priority_elements"]:
            base_ratio = 0.3  # 规则优先
        else:
            base_ratio = self.mixing_strategy["default_blend_ratio"]

        # 根据置信度调整
        if llm_confidence >= self.mixing_strategy["confidence_threshold"]:
            # LLM高置信度，增加LLM权重
            adjustment = (llm_confidence - 0.7) * 0.5
            base_ratio = min(base_ratio + adjustment, 0.9)

        if rule_confidence >= self.mixing_strategy["confidence_threshold"]:
            # 规则高置信度，增加规则权重
            adjustment = (rule_confidence - 0.7) * 0.5
            base_ratio = max(base_ratio - adjustment, 0.1)

        return base_ratio

    def _create_fallback_estimation(self, element: Any) -> DurationEstimation:
        """
        创建降级估算
        """
        fallback_duration = self._get_default_duration(element.element_type)

        return DurationEstimation(
            element_id=element.element_id,
            element_type=element.element_type,
            estimated_duration=fallback_duration,
            original_duration=element.original_duration,
            estimator_source=EstimationSource.FALLBACK,
            confidence=0.3
        )
