"""
@FileName: duration_estimator.py
@Description: 时长估算引擎
@Author: HengLine
@Time: 2025/12/20 21:23
"""
from typing import Dict

from hengline.logger import warning
from hengline.agent.script_parser.script_parser_model import UnifiedScript
from hengline.agent.temporal_planner.duration_plan.duration_calculator import RuleDurationCalculator
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation
from hengline.config.temporal_planner_config import DurationConfig


class DurationEstimator:
    """
    时长估算引擎 - 混合策略（规则 + AI微调）
    """

    def __init__(self, config: DurationConfig):
        self.config = config

        # 规则计算器
        self.rule_calculator = RuleDurationCalculator(config)

        # AI微调器（可选）
        # self.ai_adjuster = AIDurationAdjuster(config) if config.enable_ai_adjustment else None

    def estimate_all(self, script_input: UnifiedScript) -> Dict[str, DurationEstimation]:
        """
        估算所有元素的时长
        """
        estimations = {}
        if not script_input:
            warning("DurationEstimator: 没有提供脚本输入，无法进行时长估算。")
            return estimations

        # 1. 估算对话时长
        if script_input.dialogues:
            for dialogue in script_input.dialogues:
                estimation = self.rule_calculator.estimate_dialogue(dialogue.text, 1)
                estimations[dialogue.dialogue_id] = estimation

        # 2. 估算动作时长
        if script_input.actions:
            for action in script_input.actions:
                estimation = self.rule_calculator.estimate_action(action.type, action.description, 1)
                estimations[action.action_id] = estimation

        # 3. 估算场景描述时长
        if script_input.scenes:
            for scene in script_input.scenes:
                if scene.description:
                    estimation = self.rule_calculator.estimate_description(scene.description, scene)
                    estimations[f"scene_{scene.scene_id}_desc"] = estimation

        # 4. AI微调（只在需要时）
        # if self.ai_adjuster and self._needs_ai_adjustment(estimations, script_input):
        #     estimations = self.ai_adjuster.adjust(estimations, script_input)

        return estimations

    def _needs_ai_adjustment(self, estimations: Dict, script_input: UnifiedScript) -> bool:
        """
        判断是否需要AI微调
        """
        # 规则1：有复杂情感场景
        complex_emotions = ["愤怒", "悲伤", "震惊", "狂喜"]
        for dialogue in script_input.dialogues:
            if dialogue.emotion in complex_emotions:
                return True

        # 规则2：有复杂的动作序列
        complex_action_count = sum(1 for action in script_input.actions
                                   if action.intensity >= 4 or action.type in ["fight", "chase"])
        if complex_action_count >= 2:
            return True

        # 规则3：总时长超过阈值
        total_duration = sum(e.estimated_duration for e in estimations.values())
        if total_duration > self.config.ai_adjustment_threshold:
            return True

        return False
