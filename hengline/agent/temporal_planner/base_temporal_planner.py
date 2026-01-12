# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: 时序规划，负责将剧本按5秒粒度切分，估算动作时长
@Author: HengLine
@Time: 2025/10 - 2025/12
"""
from typing import Dict

from hengline.agent.base_agent import BaseAgent
from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.temporal_planner_model import TimelinePlan, EstimationErrorLevel, EstimationError


class TemporalPlanner(BaseAgent):
    """时序规划"""

    def plan_timeline(self, structured_script: UnifiedScript) -> TimelinePlan | None:
        """
        规划剧本的时序分段
        
        Args:
            structured_script: 结构化的剧本

        Returns:
            分段计划列表
        """
        pass

    def _log_error(self, error_log, element_id: str, error_type: str, message: str,
                   level: EstimationErrorLevel, recovery_action: str = "",
                   fallback_value: float = None):
        """记录错误"""
        error = EstimationError(
            element_id=element_id,
            error_type=error_type,
            message=message,
            level=level,
            recovery_action=recovery_action,
            fallback_value=fallback_value
        )

        error_log.append(error)

        # 打印错误信息（在实际系统中可能写入日志文件）
        print(f"[{level.value.upper()}] {error_type}: {message} (元素: {element_id})")
        if recovery_action:
            print(f"  恢复操作: {recovery_action}")
        if fallback_value is not None:
            print(f"  备用值: {fallback_value}")

    def get_error_summary(self, error_log) -> Dict:
        """获取错误摘要"""
        error_counts = {}
        for error in error_log:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1

        return {
            "total_errors": len(error_log),
            "error_by_type": error_counts,
            "errors_by_level": {
                "warning": len([e for e in error_log if e.level == EstimationErrorLevel.WARNING]),
                "error": len([e for e in error_log if e.level == EstimationErrorLevel.ERROR]),
                "critical": len([e for e in error_log if e.level == EstimationErrorLevel.CRITICAL])
            },
            "recovery_rate": len([e for e in error_log if e.recovery_action]) / len(error_log) if error_log else 0
        }
