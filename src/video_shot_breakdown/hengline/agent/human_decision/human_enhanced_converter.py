"""
@FileName: human_enhanced_converter.py
@Description: 
@Author: HengLine
@Time: 2026/2/5 17:47
"""
from typing import Dict, Any

from video_shot_breakdown.hengline.agent.human_decision.human_decision_converter import HumanDecisionConverter
from video_shot_breakdown.hengline.agent.workflow.workflow_models import DecisionState


class EnhancedHumanDecisionConverter(HumanDecisionConverter):
    """增强版转换器 - 支持更多功能"""

    def __init__(self):
        super().__init__()
        # 添加自定义映射
        self.INPUT_TO_STANDARD_MAP.update({
            # 中文支持
            "继续": "CONTINUE",
            "重试": "RETRY",
            "调整": "ADJUST",
            "修复": "FIX",
            "优化": "OPTIMIZE",
            "升级": "ESCALATE",
            "中止": "ABORT",
        })

    def suggest_decision(self, context: Dict[str, Any]) -> str:
        """根据上下文建议决策

        Args:
            context: 上下文信息

        Returns:
            str: 建议的标准化决策
        """
        has_errors = context.get("has_errors", False)
        is_timeout = context.get("is_timeout", False)
        retry_count = context.get("retry_count", 0)

        if is_timeout:
            return "CONTINUE"

        if has_errors:
            if retry_count >= 3:
                return "ESCALATE"
            else:
                return "RETRY"

        # 默认建议
        return "CONTINUE"

    def explain_decision(self, decision_state: DecisionState,
                         input_str: str = None) -> Dict[str, str]:
        """解释决策

        Args:
            decision_state: 决策状态
            input_str: 原始输入（可选）

        Returns:
            Dict: 解释信息
        """
        explanation = {
            "decision": decision_state.value,
            "description": self.get_decision_description(decision_state),
            "impact": "",
            "next_steps": "",
        }

        # 根据决策状态添加影响和下一步
        impacts = {
            DecisionState.SUCCESS: "继续生成最终视频",
            DecisionState.SHOULD_RETRY: "重新开始整个处理流程",
            DecisionState.NEEDS_ADJUSTMENT: "调整提示词后重新审查",
            DecisionState.NEEDS_FIX: "修复问题片段",
            DecisionState.ABORT_PROCESS: "立即停止所有处理",
        }

        next_steps = {
            DecisionState.SUCCESS: "进入视频生成阶段",
            DecisionState.SHOULD_RETRY: "返回剧本解析阶段",
            DecisionState.NEEDS_ADJUSTMENT: "进入提示词调整阶段",
            DecisionState.NEEDS_FIX: "进入片段修复阶段",
            DecisionState.ABORT_PROCESS: "清理资源并结束",
        }

        explanation["impact"] = impacts.get(decision_state, "继续处理")
        explanation["next_steps"] = next_steps.get(decision_state, "根据流程继续")

        if input_str:
            explanation["input"] = input_str
            explanation["normalized"] = self.normalize_input(input_str)

        return explanation
