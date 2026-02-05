"""
@FileName: human_decision_converter.py
@Description: 人工决策转换器
@Author: HengLine
@Time: 2026/2/5 17:09
"""
import time
from typing import Dict, Any, Optional

from video_shot_breakdown.hengline.agent.workflow.workflow_models import DecisionState
from video_shot_breakdown.logger import info, warning


class HumanDecisionConverter:
    """人工决策转换器

    职责：
    1. 将各种形式的人工输入标准化
    2. 将标准化输入映射到 DecisionState
    3. 提供输入验证和清理功能
    """

    # 输入到标准决策的映射
    INPUT_TO_STANDARD_MAP = {
        # 数字输入映射
        "1": "CONTINUE",
        "2": "APPROVE",
        "3": "RETRY",
        "4": "ADJUST",
        "5": "FIX",
        "6": "OPTIMIZE",
        "7": "ESCALATE",
        "8": "ABORT",

        # 常见输入变体
        "CONTINUE": "CONTINUE",
        "C": "CONTINUE",
        "GO": "CONTINUE",
        "OK": "CONTINUE",
        "YES": "CONTINUE",
        "Y": "CONTINUE",

        "APPROVE": "APPROVE",
        "A": "APPROVE",
        "PASS": "APPROVE",
        "ACCEPT": "APPROVE",

        "RETRY": "RETRY",
        "R": "RETRY",
        "RESTART": "RETRY",
        "TRY_AGAIN": "RETRY",
        "REPEAT": "RETRY",

        "ADJUST": "ADJUST",
        "MODIFY": "ADJUST",
        "TWEAK": "ADJUST",
        "CHANGE": "ADJUST",
        "EDIT": "ADJUST",

        "FIX": "FIX",
        "REPAIR": "FIX",
        "CORRECT": "FIX",
        "RESOLVE": "FIX",

        "OPTIMIZE": "OPTIMIZE",
        "IMPROVE": "OPTIMIZE",
        "ENHANCE": "OPTIMIZE",
        "REFINE": "OPTIMIZE",

        "ESCALATE": "ESCALATE",
        "E": "ESCALATE",
        "MANAGER": "ESCALATE",
        "SUPERVISOR": "ESCALATE",
        "HELP": "ESCALATE",

        "ABORT": "ABORT",
        "STOP": "ABORT",
        "CANCEL": "ABORT",
        "TERMINATE": "ABORT",
        "QUIT": "ABORT",
        "Q": "ABORT",
    }

    # 标准决策到 DecisionState 的映射
    STANDARD_TO_DECISION_MAP = {
        "CONTINUE": DecisionState.SUCCESS,
        "APPROVE": DecisionState.SUCCESS,
        "RETRY": DecisionState.SHOULD_RETRY,
        "ADJUST": DecisionState.NEEDS_ADJUSTMENT,  # -> CONVERT_PROMPT
        "FIX": DecisionState.NEEDS_FIX,  # -> SPLIT_VIDEO
        "OPTIMIZE": DecisionState.NEEDS_OPTIMIZATION,  # -> CONVERT_PROMPT
        "ESCALATE": DecisionState.REQUIRE_HUMAN,
        "ABORT": DecisionState.ABORT_PROCESS
    }

    def __init__(self):
        """初始化转换器"""
        self.conversion_history = []

    def normalize_input(self, raw_input: Any) -> str:
        """标准化输入

        Args:
            raw_input: 原始输入（字符串、数字等）

        Returns:
            str: 标准化的决策字符串
        """
        if raw_input is None:
            return "CONTINUE"

        # 转换为字符串并清理
        input_str = str(raw_input).strip().upper()

        # 空字符串处理
        if not input_str:
            return "CONTINUE"

        # 查找映射
        if input_str in self.INPUT_TO_STANDARD_MAP:
            standard = self.INPUT_TO_STANDARD_MAP[input_str]
            info(f"输入标准化: '{raw_input}' -> '{standard}'")
            return standard

        # 尝试部分匹配
        for key, standard in self.INPUT_TO_STANDARD_MAP.items():
            if key in input_str or input_str in key:
                info(f"输入部分匹配: '{raw_input}' -> '{standard}'")
                return standard

        # 未知输入，默认继续
        warning(f"未知输入: '{raw_input}'，使用默认值 'CONTINUE'")
        return "CONTINUE"

    def convert_to_decision_state(self, normalized_input: str,
                                  context: Optional[Dict[str, Any]] = None) -> DecisionState:
        """转换为决策状态

        Args:
            normalized_input: 标准化后的输入
            context: 转换上下文（可选）

        Returns:
            DecisionState: 决策状态
        """
        # 获取映射
        decision_state = self.STANDARD_TO_DECISION_MAP.get(
            normalized_input,
            DecisionState.SUCCESS  # 默认值
        )

        # 记录转换历史
        self.conversion_history.append({
            "timestamp": time.time(),
            "input": normalized_input,
            "decision": decision_state.value,
            "context": context
        })

        info(f"决策转换: '{normalized_input}' -> {decision_state.value}")

        return decision_state

    def validate_decision(self, decision_state: DecisionState,
                          context: Optional[Dict[str, Any]] = None) -> bool:
        """验证决策的合理性

        Args:
            decision_state: 要验证的决策状态
            context: 验证上下文

        Returns:
            bool: 是否合理
        """
        # 这里可以添加业务逻辑验证
        # 例如：在某些上下文中不允许某些决策

        if context and context.get("is_timeout", False):
            # 超时情况下，只能继续或中止
            valid_states = [DecisionState.SUCCESS, DecisionState.ABORT_PROCESS]
            if decision_state not in valid_states:
                warning(f"超时情况下不允许的决策: {decision_state.value}")
                return False

        return True

    def get_decision_description(self, decision_state: DecisionState) -> str:
        """获取决策描述

        Args:
            decision_state: 决策状态

        Returns:
            str: 人类可读的描述
        """
        descriptions = {
            DecisionState.SUCCESS: "继续流程",
            DecisionState.VALID: "验证通过",
            DecisionState.SHOULD_RETRY: "重新开始",
            DecisionState.NEEDS_ADJUSTMENT: "调整优化",
            DecisionState.NEEDS_FIX: "修复问题",
            DecisionState.NEEDS_OPTIMIZATION: "优化质量",
            DecisionState.RETRY_WITH_ADJUSTMENT: "调整后重试",
            DecisionState.REQUIRE_HUMAN: "需要人工干预",
            DecisionState.FAILED: "处理失败",
            DecisionState.CRITICAL_FAILURE: "严重失败",
            DecisionState.ABORT_PROCESS: "中止流程",
        }
        return descriptions.get(decision_state, "未知决策")
