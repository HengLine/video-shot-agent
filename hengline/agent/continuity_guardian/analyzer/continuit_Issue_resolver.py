"""
@FileName: continuit_Issue_resolver.py
@Description: 
@Author: HengLine
@Time: 2026/1/4 18:07
"""
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

from hengline.agent.continuity_guardian.model.continuity_guardian_report import ContinuityIssue
from hengline.agent.continuity_guardian.model.continuity_rule_guardian import ContinuityRuleSet


class ContinuityIssueResolver:
    """连续性问题解决器"""

    def __init__(self, rule_set: ContinuityRuleSet):
        self.rule_set = rule_set
        self.resolution_strategies = self._initialize_strategies()
        self.fix_history: List[Dict[str, Any]] = []

    def _initialize_strategies(self) -> Dict[str, callable]:
        """初始化解决策略"""
        return {
            "character_appearance_inconsistency": self._resolve_character_appearance,
            "prop_position_inconsistency": self._resolve_prop_position,
            "environment_change_inconsistency": self._resolve_environment_change,
            "temporal_discontinuity": self._resolve_temporal_discontinuity,
            "spatial_discontinuity": self._resolve_spatial_discontinuity,
            "visual_style_inconsistency": self._resolve_visual_style
        }

    def resolve_issue(self, issue: ContinuityIssue, context: Dict[str, Any]) -> Dict[str, Any]:
        """解决问题"""
        resolution = {
            "issue_id": issue.issue_id,
            "timestamp": datetime.now(),
            "original_description": issue.description,
            "resolution_applied": False,
            "resolution_details": {},
            "confidence": 0.0
        }

        # 确定解决策略
        strategy = self._identify_resolution_strategy(issue)

        if strategy and strategy in self.resolution_strategies:
            try:
                resolution_details = self.resolution_strategies[strategy](issue, context)
                resolution["resolution_applied"] = True
                resolution["resolution_details"] = resolution_details
                resolution["confidence"] = resolution_details.get("confidence", 0.0)

                # 记录修复历史
                self.fix_history.append({
                    **resolution,
                    "context_snapshot": context.get("snapshot", {})
                })

            except Exception as e:
                resolution["error"] = str(e)
                resolution["resolution_applied"] = False

        else:
            resolution["resolution_applied"] = False
            resolution["reason"] = f"No strategy found for issue type: {issue.entity_type}"

        return resolution

    def _identify_resolution_strategy(self, issue: ContinuityIssue) -> Optional[str]:
        """识别解决策略"""
        issue_lower = issue.description.lower()

        if "character" in issue_lower and "appearance" in issue_lower:
            return "character_appearance_inconsistency"
        elif "prop" in issue_lower and "position" in issue_lower:
            return "prop_position_inconsistency"
        elif "environment" in issue_lower:
            return "environment_change_inconsistency"
        elif "time" in issue_lower or "temporal" in issue_lower:
            return "temporal_discontinuity"
        elif "space" in issue_lower or "position" in issue_lower:
            return "spatial_discontinuity"
        elif "visual" in issue_lower or "style" in issue_lower:
            return "visual_style_inconsistency"

        return None

    def _resolve_character_appearance(self, issue: ContinuityIssue, context: Dict[str, Any]) -> Dict[str, Any]:
        """解决角色外貌不一致"""
        character_id = issue.entity_id
        current_state = context.get("current_state", {})
        previous_state = context.get("previous_state", {})

        resolution = {
            "action": "smooth_appearance_transition",
            "parameters": {
                "character_id": character_id,
                "blend_frames": 3,
                "maintain_key_features": True
            },
            "confidence": 0.85
        }

        # 如果有前后状态，计算中间状态
        if current_state and previous_state:
            resolution["intermediate_states"] = self._calculate_intermediate_appearances(
                previous_state, current_state, character_id
            )

        return resolution

    def _resolve_prop_position(self, issue: ContinuityIssue, context: Dict[str, Any]) -> Dict[str, Any]:
        """解决道具位置不一致"""
        prop_id = issue.entity_id

        resolution = {
            "action": "adjust_prop_trajectory",
            "parameters": {
                "prop_id": prop_id,
                "smooth_movement": True,
                "respect_physics": True
            },
            "confidence": 0.9
        }

        # 添加物理约束
        if context.get("physics_constraints"):
            resolution["physics_constraints"] = context["physics_constraints"]

        return resolution

    def _resolve_environment_change(self, issue: ContinuityIssue, context: Dict[str, Any]) -> Dict[str, Any]:
        """解决环境变化不一致"""
        resolution = {
            "action": "gradual_environment_transition",
            "parameters": {
                "transition_type": "dissolve",
                "duration_frames": 5,
                "maintain_atmosphere": True
            },
            "confidence": 0.75
        }

        # 根据问题描述调整参数
        if "time" in issue.description.lower():
            resolution["parameters"]["transition_type"] = "time_lapse"
        elif "weather" in issue.description.lower():
            resolution["parameters"]["transition_type"] = "weather_transition"

        return resolution

    def _resolve_temporal_discontinuity(self, issue: ContinuityIssue, context: Dict[str, Any]) -> Dict[str, Any]:
        """解决时间不连续"""
        resolution = {
            "action": "insert_transition_sequence",
            "parameters": {
                "transition_effect": "fade",
                "duration": 2.0,  # 秒
                "add_temporal_markers": True
            },
            "confidence": 0.8
        }

        return resolution

    def _resolve_spatial_discontinuity(self, issue: ContinuityIssue, context: Dict[str, Any]) -> Dict[str, Any]:
        """解决空间不连续"""
        resolution = {
            "action": "adjust_camera_movement",
            "parameters": {
                "smooth_camera_path": True,
                "maintain_spatial_relationships": True,
                "add_establishing_shot": False
            },
            "confidence": 0.7
        }

        # 如果需要，添加建立镜头
        if "jump" in issue.description.lower() or "cut" in issue.description.lower():
            resolution["parameters"]["add_establishing_shot"] = True

        return resolution

    def _resolve_visual_style(self, issue: ContinuityIssue, context: Dict[str, Any]) -> Dict[str, Any]:
        """解决视觉风格不一致"""
        resolution = {
            "action": "apply_style_normalization",
            "parameters": {
                "color_correction": True,
                "contrast_adjustment": True,
                "filter_consistency": True
            },
            "confidence": 0.8
        }

        # 添加具体的样式调整
        if "color" in issue.description.lower():
            resolution["parameters"]["color_grading"] = "match_previous"
        if "brightness" in issue.description.lower():
            resolution["parameters"]["exposure_matching"] = True

        return resolution

    def _calculate_intermediate_appearances(self, prev_state: Dict, curr_state: Dict,
                                            character_id: str) -> List[Dict[str, Any]]:
        """计算中间外观状态"""
        intermediates = []

        # 简化的线性插值
        prev_appearance = prev_state.get("appearance", {})
        curr_appearance = curr_state.get("appearance", {})

        # 生成3个中间状态
        for i in range(1, 4):
            intermediate = {}
            weight = i / 4.0

            # 插值每个属性
            for key in set(prev_appearance.keys()) | set(curr_appearance.keys()):
                prev_val = prev_appearance.get(key, 0)
                curr_val = curr_appearance.get(key, 0)

                if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                    intermediate[key] = prev_val * (1 - weight) + curr_val * weight

            intermediates.append({
                "frame_offset": i,
                "appearance": intermediate,
                "blend_weight": weight
            })

        return intermediates

    def get_resolution_summary(self) -> Dict[str, Any]:
        """获取解决摘要"""
        successful_fixes = [f for f in self.fix_history if f["resolution_applied"]]
        failed_fixes = [f for f in self.fix_history if not f["resolution_applied"]]

        return {
            "total_issues_attempted": len(self.fix_history),
            "successful_resolutions": len(successful_fixes),
            "failed_resolutions": len(failed_fixes),
            "success_rate": len(successful_fixes) / len(self.fix_history) if self.fix_history else 0,
            "average_confidence": np.mean([f.get("confidence", 0) for f in successful_fixes]) if successful_fixes else 0,
            "recent_resolutions": self.fix_history[-5:] if self.fix_history else []
        }
