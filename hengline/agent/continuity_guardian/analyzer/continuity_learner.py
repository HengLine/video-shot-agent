"""
@FileName: continuity_learner.py
@Description: 
@Author: HengLine
@Time: 2026/1/5 15:46
"""
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime
import numpy as np

from hengline.agent.continuity_guardian.model.continuity_guard_guardian import GuardianConfig
from hengline.agent.continuity_guardian.model.continuity_guardian_report import ContinuityIssue


class ContinuityLearner:
    """连续性学习器"""

    def __init__(self, config: GuardianConfig):
        self.config = config
        self.pattern_database: Dict[str, List[Dict]] = defaultdict(list)
        self.mistake_history: List[Dict] = []
        self.success_patterns: List[Dict] = []
        self.learning_rate: float = 0.1
        self.confidence_threshold: float = 0.7

    def learn_from_issue(self, issue: ContinuityIssue,
                         fix_result: Dict[str, Any]) -> None:
        """从问题中学习"""
        learning_record = {
            "timestamp": datetime.now(),
            "issue_type": issue.description,
            "issue_level": issue.level.value,
            "fix_applied": fix_result.get("applied", False),
            "fix_effectiveness": fix_result.get("effectiveness", 0.0),
            "context": issue.evidence[-1] if issue.evidence else {}
        }

        self.mistake_history.append(learning_record)

        # 如果修复有效，记录成功模式
        if fix_result.get("applied", False) and fix_result.get("effectiveness", 0.0) > 0.7:
            success_pattern = {
                "issue_pattern": issue.description,
                "fix_strategy": fix_result.get("strategy", ""),
                "context_pattern": self._extract_pattern(issue.evidence),
                "confidence": fix_result.get("effectiveness", 0.0)
            }
            self.success_patterns.append(success_pattern)

    def predict_issues(self, scene_data: Dict) -> List[Dict]:
        """预测可能出现的问题"""
        predicted_issues = []

        # 基于历史模式预测
        for pattern in self.success_patterns:
            if self._pattern_matches(scene_data, pattern["context_pattern"]):
                predicted_issue = {
                    "type": pattern["issue_pattern"],
                    "confidence": pattern["confidence"],
                    "suggested_fix": pattern["fix_strategy"],
                    "prevention_advice": self._generate_prevention_advice(pattern)
                }
                predicted_issues.append(predicted_issue)

        return predicted_issues

    def _extract_pattern(self, evidence: List[Dict]) -> Dict[str, Any]:
        """从证据中提取模式"""
        pattern = {}

        for item in evidence:
            if "entity_type" in item:
                entity_type = item["entity_type"]
                if entity_type not in pattern:
                    pattern[entity_type] = []
                pattern[entity_type].append(item)

        return pattern

    def _pattern_matches(self, scene_data: Dict, pattern: Dict) -> bool:
        """检查模式是否匹配"""
        match_score = 0.0
        total_checks = 0

        for entity_type, pattern_items in pattern.items():
            if entity_type == "character":
                scene_items = scene_data.get("characters", [])
            elif entity_type == "prop":
                scene_items = scene_data.get("props", [])
            else:
                continue

            # 简化的模式匹配
            if len(scene_items) > 0:
                match_score += 0.3
                total_checks += 1

        if total_checks > 0:
            return match_score / total_checks > self.confidence_threshold
        return False

    def _generate_prevention_advice(self, pattern: Dict) -> List[str]:
        """生成预防建议"""
        advice = []

        issue_type = pattern.get("issue_pattern", "")

        if "position" in issue_type.lower():
            advice.append("确保位置变化不超过物理合理范围")
            advice.append("检查是否有支撑物体突然消失")

        if "appearance" in issue_type.lower():
            advice.append("记录角色服装和外观的基准状态")
            advice.append("渐进式变化而非突变")

        if "lighting" in issue_type.lower():
            advice.append("保持光源位置和参数一致")
            advice.append("使用渐变而非跳变的照明变化")

        return advice

    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计"""
        stats = {
            "total_issues_learned": len(self.mistake_history),
            "successful_fixes": len([r for r in self.mistake_history
                                     if r["fix_applied"] and r["fix_effectiveness"] > 0.7]),
            "success_patterns_count": len(self.success_patterns),
            "average_confidence": np.mean([p["confidence"] for p in self.success_patterns])
            if self.success_patterns else 0.0
        }

        # 问题类型分布
        issue_types = defaultdict(int)
        for record in self.mistake_history:
            issue_types[record["issue_type"]] += 1
        stats["issue_type_distribution"] = dict(issue_types)

        return stats