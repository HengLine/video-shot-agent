"""
@FileName: continuity_guardian_report.py
@Description: 快照与报告
@Author: HengLine
@Time: 2026/1/4 17:41
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Any

from .continuity_state_guardian import CharacterState, PropState, EnvironmentState
from .continuity_visual_guardian import SpatialRelation
from ..continuity_guardian_model import ContinuityLevel


@dataclass
class StateSnapshot:
    """状态快照"""
    timestamp: datetime
    scene_id: str
    frame_number: int
    characters: Dict[str, CharacterState]
    props: Dict[str, PropState]
    environment: EnvironmentState
    spatial_relations: SpatialRelation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "scene_id": self.scene_id,
            "frame_number": self.frame_number,
            "character_count": len(self.characters),
            "prop_count": len(self.props),
            "metadata": self.metadata
        }


class ContinuityIssue:
    """连续性问题"""

    def __init__(self, issue_id: str, level: ContinuityLevel, description: str):
        self.issue_id = issue_id
        self.level = level
        self.description = description
        self.entity_type: str = ""  # character/prop/environment
        self.entity_id: str = ""
        self.frame_range: Tuple[int, int] = (0, 0)
        self.evidence: List[Dict] = []  # 证据数据
        self.suggested_fixes: List[str] = []  # 建议修复方案
        self.auto_fixable: bool = False  # 是否可自动修复

    def add_evidence(self, evidence_data: Dict):
        """添加证据"""
        self.evidence.append(evidence_data)

    def suggest_fix(self, fix_description: str):
        """添加修复建议"""
        self.suggested_fixes.append(fix_description)


class ValidationReport:
    """验证报告"""

    def __init__(self, validation_id: str):
        self.validation_id = validation_id
        self.timestamp = datetime.now()
        self.issues: List[ContinuityIssue] = []
        self.summary: Dict[str, int] = {
            "total_checks": 0,
            "passed": 0,
            "critical_issues": 0,
            "major_issues": 0,
            "minor_issues": 0,
            "cosmetic_issues": 0
        }
        self.recommendations: List[str] = []

    def add_issue(self, issue: ContinuityIssue):
        """添加问题"""
        self.issues.append(issue)
        # 更新统计
        if issue.level == ContinuityLevel.CRITICAL:
            self.summary["critical_issues"] += 1
        elif issue.level == ContinuityLevel.MAJOR:
            self.summary["major_issues"] += 1
        elif issue.level == ContinuityLevel.MINOR:
            self.summary["minor_issues"] += 1
        else:
            self.summary["cosmetic_issues"] += 1

    def add_recommendation(self, recommendation: str):
        """添加推荐"""
        self.recommendations.append(recommendation)

    def generate_summary(self) -> str:
        """生成摘要"""
        return (f"Validation Report: {self.validation_id}\n"
                f"Time: {self.timestamp}\n"
                f"Critical Issues: {self.summary['critical_issues']}\n"
                f"Major Issues: {self.summary['major_issues']}\n"
                f"Minor Issues: {self.summary['minor_issues']}\n"
                f"Cosmetic Issues: {self.summary['cosmetic_issues']}")
