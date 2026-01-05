"""
@FileName: continuity_transition_guardian.py
@Description: 提示与规则
@Author: HengLine
@Time: 2026/1/4 17:36
"""
from typing import List, Dict, Any, Optional


class GenerationHints:
    """生成提示"""

    def __init__(self):
        self.continuity_constraints: List[str] = []  # 连续性约束
        self.style_guidelines: Dict[str, Any] = {}  # 风格指南
        self.previous_context: Dict[str, Any] = {}  # 先前上下文
        self.avoid_elements: List[str] = []  # 避免的元素
        self.required_elements: List[str] = []  # 必需的元素

    def add_constraint(self, constraint: str):
        """添加约束"""
        self.continuity_constraints.append(constraint)


class ContinuityRuleSet:
    """连续性规则集"""

    def __init__(self):
        self.rules: Dict[str, Dict] = {
            "character_appearance": {
                "description": "角色外貌一致性规则",
                "strictness": "high",
                "allowed_changes": ["emotion", "minor_dirt", "hair_movement"]
            },
            "prop_consistency": {
                "description": "道具连续性规则",
                "strictness": "medium",
                "track_movement": True
            },
            "environment_stability": {
                "description": "环境稳定性规则",
                "strictness": "high",
                "allowed_transitions": ["time_progression", "weather_changes"]
            },
            "spatial_continuity": {
                "description": "空间连续性规则",
                "strictness": "critical",
                "enforce_physics": True
            }
        }

    def get_rule(self, rule_name: str) -> Optional[Dict]:
        """获取规则"""
        return self.rules.get(rule_name)
