"""
@FileName: continuity_guardian_report.py
@Description: 快照与报告
@Author: HengLine
@Time: 2026/1/4 17:41
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from .continuity_rule_guardian import ContinuityRuleSet
from .continuity_state_guardian import StateSnapshot
from ..continuity_guardian_model import ContinuityLevel, AnchoredSegment


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


class AutoFix:
    """自动修复引擎"""

    def __init__(self, rule_set: ContinuityRuleSet):
        self.rule_set = rule_set
        self.fix_strategies: Dict[str, callable] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """注册默认修复策略"""
        self.fix_strategies["character_appearance"] = self._fix_character_appearance
        self.fix_strategies["prop_position"] = self._fix_prop_position
        self.fix_strategies["environment_consistency"] = self._fix_environment_consistency

    def _fix_character_appearance(self, issue: ContinuityIssue, current_state: StateSnapshot) -> Dict[str, Any]:
        """修复角色外貌"""
        fix_details = {
            "action": "adjust_appearance",
            "entity_id": issue.entity_id,
            "changes": {},
            "confidence": 0.8
        }
        return fix_details

    def _fix_prop_position(self, issue: ContinuityIssue, current_state: StateSnapshot) -> Dict[str, Any]:
        """修复道具位置"""
        fix_details = {
            "action": "reposition_prop",
            "entity_id": issue.entity_id,
            "target_position": None,
            "confidence": 0.9
        }
        return fix_details

    def _fix_environment_consistency(self, issue: ContinuityIssue, current_state: StateSnapshot) -> Dict[str, Any]:
        """修复环境一致性"""
        fix_details = {
            "action": "adjust_environment",
            "parameter": issue.description.split(":")[-1].strip(),
            "confidence": 0.7
        }
        return fix_details

    def suggest_fix(self, issue: ContinuityIssue, current_state: StateSnapshot) -> Optional[Dict[str, Any]]:
        """建议修复方案"""
        if not issue.auto_fixable:
            return None

        # 根据问题类型选择修复策略
        fix_type = self._determine_fix_type(issue)
        if fix_type in self.fix_strategies:
            return self.fix_strategies[fix_type](issue, current_state)
        return None

    def _determine_fix_type(self, issue: ContinuityIssue) -> str:
        """确定修复类型"""
        # 简化的类型判断逻辑
        if "character" in issue.description.lower():
            return "character_appearance"
        elif "prop" in issue.description.lower():
            return "prop_position"
        elif "environment" in issue.description.lower():
            return "environment_consistency"
        return ""


@dataclass
class AnchoredTimeline:
    """带强约束的连续性规划"""
    # 核心：带约束的时间片段
    anchored_segments: List[AnchoredSegment]

    # 连续性规则系统
    continuity_rules: ContinuityRuleSet

    # 状态跟踪快照
    state_snapshots: Dict[str, StateSnapshot]  # key: timestamp

    # 验证报告
    validation_report: ValidationReport

    # 问题与修复
    detected_issues: List[ContinuityIssue]
    auto_fixes: List[AutoFix]

    # 质量指标
    continuity_score: float = 0  # 0-1，连贯性评分
