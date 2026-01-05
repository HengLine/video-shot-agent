"""
@FileName: continuity_guardian_autofix.py
@Description: 自动修复
@Author: HengLine
@Time: 2026/1/4 17:47
"""
from typing import Dict, Any, Optional

from .continuity_guardian_report import ContinuityIssue, StateSnapshot
from .continuity_rule_guardian import ContinuityRuleSet


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
