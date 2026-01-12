"""
@FileName: action_estimator.py
@Description: 动作时长估算器
@Author: HengLine
@Time: 2026/1/12 15:41
"""
from hengline.agent.script_parser.script_parser_models import Action
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, ElementType


class ActionDurationEstimator:
    """动作时长估算模型"""

    # 动作类型基准时长
    ACTION_BASELINE = {
        "simple_gesture": 1.2,  # 简单手势
        "basic_movement": 2.5,  # 基础移动
        "complex_action": 4.0,  # 复杂动作
        "interaction": 3.0,  # 交互动作
        "extended_sequence": 8.0,  # 扩展序列
    }

    def estimate(self, action: Action, context: EstimationContext) -> DurationEstimation:
        """估算动作时长"""

        # 1. 动作复杂度分析
        complexity = self._analyze_action_complexity(action.description)

        # 2. 物理可行性检查
        physics_check = self._physics_sanity_check(action.description, context)

        # 3. 时长估算
        if complexity in self.ACTION_BASELINE:
            base_duration = self.ACTION_BASELINE[complexity]
        else:
            base_duration = self._estimate_by_decomposition(action.description)

        # 4. 上下文调整
        adjusted_duration = self._context_adjustment(base_duration, context)

        return DurationEstimation(
            element_id=action.id,
            element_type=ElementType.ACTION,
            base_duration=adjusted_duration,
            min_duration=adjusted_duration * 0.6,
            max_duration=adjusted_duration * 1.4,
            confidence=physics_check.confidence,
            complexity_factor=complexity.score
        )