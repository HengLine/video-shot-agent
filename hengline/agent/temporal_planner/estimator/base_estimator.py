"""
@FileName: base_estimator.py
@Description: 
@Author: HengLine
@Time: 2026/1/15 17:42
"""
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict

from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation


class EstimationErrorLevel(Enum):
    """错误级别"""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class EstimationError:
    """估算错误信息"""
    element_id: str
    error_type: str
    message: str
    level: EstimationErrorLevel
    recovery_action: str = ""
    fallback_value: Optional[float] = None
    timestamp: str = None



class BaseDurationEstimator(ABC):
    """时长估算器基类"""
    def __init__(self):
        self.error_log: List[EstimationError] = []
        self.cache: Dict[str, DurationEstimation] = {}

    def _log_error(self, element_id: str, error_type: str, message: str,
                   level: EstimationErrorLevel, recovery_action: str = "",
                   fallback_value: float = None):
        """记录错误"""
        error = EstimationError(
            element_id=element_id,
            error_type=error_type,
            message=message,
            level=level,
            recovery_action=recovery_action,
            fallback_value=fallback_value,
            timestamp=datetime.now().isoformat()
        )

        self.error_log.append(error)

        # 打印错误信息
        level_icon = {
            EstimationErrorLevel.WARNING: "⚠️",
            EstimationErrorLevel.ERROR: "❌",
            EstimationErrorLevel.CRITICAL: "🔥"
        }.get(level, "ℹ️")

        print(f"{level_icon} [{level.value.upper()}] {error_type}: {message}")
        if recovery_action:
            print(f"  恢复操作: {recovery_action}")
