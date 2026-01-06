"""
@FileName: check_models.py
@Description: 
@Author: HengLine
@Time: 2026/1/6 16:14
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from hengline.agent.shot_qa.model.issue_models import IssueSeverity


class CheckStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """检查结果基类"""
    check_id: str
    check_name: str
    check_description: str
    status: CheckStatus
    severity: IssueSeverity
    score: float  # 0-1

    # 详细信息
    details: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    affected_elements: List[str] = field(default_factory=list)

    # 时间信息
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0


@dataclass
class ContinuityCheckResult(CheckResult):
    """连续性检查结果"""

    # 连续性类型
    continuity_type: str = "time"  # "position", "appearance", "action", "time"

    # 具体问题
    position_discrepancy: Optional[float] = None  # 位置差异度
    appearance_changes: List[str] = field(default_factory=list)
    time_gap: Optional[float] = None
    action_discontinuity: Optional[str] = None

    # 片段信息
    previous_segment_id: Optional[str] = None
    current_segment_id: Optional[str] = None

    # 修复建议
    suggested_transition: Optional[str] = None
    recommended_adjustment: Optional[str] = None


@dataclass
class ConstraintCheckResult(CheckResult):
    """约束检查结果"""

    # 约束信息
    constraint_id: str = None
    constraint_type: str = None
    constraint_description: str = None
    constraint_priority: int = None

    # 检查详情
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    deviation: Optional[float] = None
    tolerance: Optional[float] = None

    # 满足状态
    is_satisfied: bool = False
    satisfaction_score: float = 0.0

    # 影响范围
    affected_shots: List[str] = field(default_factory=list)

    # 修复信息
    fix_complexity: str = "low"  # "low", "medium", "high"
    fix_suggestion: Optional[str] = None


@dataclass
class VisualQualityCheckResult(CheckResult):
    """视觉质量检查结果"""

    # 检查维度
    quality_dimension: str = "composition"  # "composition", "lighting", "color", "style"

    # 具体评估
    composition_score: Optional[float] = None
    lighting_consistency_score: Optional[float] = None
    color_harmony_score: Optional[float] = None
    style_consistency_score: Optional[float] = None

    # 问题详情
    composition_issues: List[str] = field(default_factory=list)
    lighting_issues: List[str] = field(default_factory=list)
    color_issues: List[str] = field(default_factory=list)
    style_issues: List[str] = field(default_factory=list)

    # 改进建议
    composition_suggestions: List[str] = field(default_factory=list)
    lighting_suggestions: List[str] = field(default_factory=list)
    color_suggestions: List[str] = field(default_factory=list)
    style_suggestions: List[str] = field(default_factory=list)


@dataclass
class TechnicalCheckResult(CheckResult):
    """技术检查结果"""

    # 检查类型
    technical_aspect: str = "prompt_quality"  # "prompt_quality", "camera_params", "feasibility"

    # 具体问题
    prompt_issues: List[str] = field(default_factory=list)
    camera_issues: List[str] = field(default_factory=list)
    feasibility_issues: List[str] = field(default_factory=list)

    # 技术指标
    prompt_length: Optional[int] = None
    prompt_clarity_score: Optional[float] = None
    camera_parameter_validity: Optional[bool] = None
    feasibility_score: Optional[float] = None

    # 修复建议
    prompt_improvements: List[str] = field(default_factory=list)
    camera_adjustments: List[str] = field(default_factory=list)
    feasibility_suggestions: List[str] = field(default_factory=list)
