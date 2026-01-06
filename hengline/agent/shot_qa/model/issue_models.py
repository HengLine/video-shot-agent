"""
@FileName: issue_models.py
@Description: 问题相关数据模型
@Author: HengLine
@Time: 2026/1/6 15:59
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List


class IssueSeverity(Enum):
    CRITICAL = "critical"  # 必须修复，否则视频无法使用
    HIGH = "high"  # 严重影响质量，建议修复
    MEDIUM = "medium"  # 影响质量，可以考虑修复
    LOW = "low"  # 轻微问题，可选修复
    INFO = "info"  # 信息性提示


@dataclass
class IssueBase:
    """问题基类"""

    issue_id: str
    issue_type: str
    severity: IssueSeverity
    title: str
    description: str

    # 问题详情
    location: str  # 问题位置，如 "shot_s001_shot1"
    root_cause: Optional[str] = None
    impact: Optional[str] = None

    # 关联信息
    related_checks: List[str] = field(default_factory=list)
    related_constraints: List[str] = field(default_factory=list)

    # 时间信息
    detected_at: datetime = field(default_factory=datetime.now)
    priority: int = 3  # 1-5，1最高


@dataclass
class CriticalIssue(IssueBase):
    """关键问题"""

    # 必须修复的问题
    must_fix: bool = True
    blocks_approval: bool = True

    # 修复信息
    suggested_fix: Optional[str] = None
    fix_effort: str = "medium"  # "low", "medium", "high"

    # 验证信息
    verification_method: Optional[str] = None
    verification_required: bool = True


@dataclass
class IssueWarning(IssueBase):
    """警告"""

    # 警告特性
    suggestion_type: str = "improvement"  # "improvement", "optimization", "warning"

    # 修复建议
    recommended_action: Optional[str] = None
    expected_benefit: Optional[str] = None

    # 可选性
    is_optional: bool = True
    suggested_priority: str = "medium"  # "low", "medium", "high"


@dataclass
class IssueSuggestion(IssueBase):
    """建议"""

    # 建议类型
    suggestion_category: str = "enhancement"  # "enhancement", "optimization", "alternative"

    # 具体建议
    proposed_solution: str = None
    rationale: Optional[str] = None

    # 影响评估
    expected_improvement: Optional[float] = None  # 0-1
    implementation_cost: str = "low"  # "low", "medium", "high"

    # 状态
    considered: bool = False
    implemented: bool = False
