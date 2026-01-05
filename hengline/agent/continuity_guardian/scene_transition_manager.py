"""
@FileName: scene_transition_manager.py
@Description: 
@Author: HengLine
@Time: 2026/1/5 15:47
"""
import math
from typing import List, Dict, Optional

from hengline.agent.continuity_guardian.model.continuity_guard_guardian import GuardianConfig
from hengline.agent.continuity_guardian.model.continuity_transition_guardian import TransitionInstruction


class SceneTransitionManager:
    """场景转场管理器"""

    def __init__(self, config: GuardianConfig):
        self.config = config
        self.transition_history: List[TransitionInstruction] = []
        self.transition_patterns: Dict[str, Dict] = {}

    def analyze_transition(self, from_scene: Dict, to_scene: Dict) -> TransitionInstruction:
        """分析场景转场"""
        transition = TransitionInstruction(
            from_scene=from_scene.get("scene_id", "unknown"),
            to_scene=to_scene.get("scene_id", "unknown")
        )

        # 分析转场类型
        transition.transition_type = self._determine_transition_type(from_scene, to_scene)

        # 计算时间间隔
        time_gap = self._calculate_time_gap(from_scene, to_scene)
        transition.temporal_gap = time_gap

        # 分析空间变化
        spatial_changes = self._analyze_spatial_changes(from_scene, to_scene)
        transition.spatial_changes = spatial_changes

        # 分析角色转场
        character_transitions = self._analyze_character_transitions(from_scene, to_scene)
        transition.character_transitions = character_transitions

        # 记录历史
        self.transition_history.append(transition)

        return transition

    def _determine_transition_type(self, from_scene: Dict, to_scene: Dict) -> str:
        """确定转场类型"""
        from_type = from_scene.get("scene_type", "general")
        to_type = to_scene.get("scene_type", "general")

        # 相同类型场景：硬切
        if from_type == to_type:
            return "cut"

        # 动作相关转场：匹配剪辑
        if "action" in from_type or "action" in to_type:
            return "match_cut"

        # 环境转场：淡入淡出
        if "environment" in from_type or "environment" in to_type:
            return "fade"

        # 时间转场：溶解
        if "time" in from_scene or "time" in to_scene:
            return "dissolve"

        # 默认：交叉溶解
        return "cross_dissolve"

    def _calculate_time_gap(self, from_scene: Dict, to_scene: Dict) -> Optional[float]:
        """计算时间间隔"""
        from_time = from_scene.get("time_data", {}).get("current_time")
        to_time = to_scene.get("time_data", {}).get("current_time")

        if from_time and to_time:
            return abs(to_time - from_time)

        return None

    def _analyze_spatial_changes(self, from_scene: Dict, to_scene: Dict) -> List[str]:
        """分析空间变化"""
        changes = []

        # 检查位置变化
        from_pos = from_scene.get("environment", {}).get("global_position", [0, 0, 0])
        to_pos = to_scene.get("environment", {}).get("global_position", [0, 0, 0])

        if from_pos != to_pos:
            changes.append(f"场景位置变化: {from_pos} -> {to_pos}")

        # 检查比例变化
        from_scale = from_scene.get("environment", {}).get("scale", 1.0)
        to_scale = to_scene.get("environment", {}).get("scale", 1.0)

        if abs(from_scale - to_scale) > 0.1:
            changes.append(f"场景比例变化: {from_scale:.2f} -> {to_scale:.2f}")

        return changes

    def _analyze_character_transitions(self, from_scene: Dict, to_scene: Dict) -> Dict[str, Dict]:
        """分析角色转场"""
        transitions = {}

        from_chars = {c["id"]: c for c in from_scene.get("characters", [])}
        to_chars = {c["id"]: c for c in to_scene.get("characters", [])}

        # 检查共同角色
        common_chars = set(from_chars.keys()) & set(to_chars.keys())

        for char_id in common_chars:
            from_char = from_chars[char_id]
            to_char = to_chars[char_id]

            changes = {}

            # 检查位置变化
            if "position" in from_char and "position" in to_char:
                from_pos = from_char["position"]
                to_pos = to_char["position"]
                if from_pos != to_pos:
                    changes["position_change"] = {
                        "from": from_pos,
                        "to": to_pos,
                        "distance": self._calculate_distance(from_pos, to_pos)
                    }

            # 检查外观变化
            if "appearance" in from_char and "appearance" in to_char:
                if from_char["appearance"] != to_char["appearance"]:
                    changes["appearance_change"] = True

            # 检查状态变化
            if "state" in from_char and "state" in to_char:
                if from_char["state"] != to_char["state"]:
                    changes["state_change"] = {
                        "from": from_char["state"],
                        "to": to_char["state"]
                    }

            if changes:
                transitions[char_id] = changes

        return transitions

    def _calculate_distance(self, pos1, pos2) -> float:
        """计算距离"""
        if isinstance(pos1, (list, tuple)) and isinstance(pos2, (list, tuple)):
            if len(pos1) >= 3 and len(pos2) >= 3:
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                dz = pos2[2] - pos1[2]
                return math.sqrt(dx * dx + dy * dy + dz * dz)
        return 0.0

    def validate_transition(self, transition: TransitionInstruction) -> List[Dict]:
        """验证转场合理性"""
        issues = []

        # 检查时间跳跃
        if transition.temporal_gap and transition.temporal_gap > 3600:  # 1小时
            issues.append({
                "type": "excessive_time_gap",
                "severity": "medium",
                "description": f"时间跳跃过大: {transition.temporal_gap:.0f}秒",
                "suggestion": "添加时间过渡或说明"
            })

        # 检查空间跳跃
        if any("场景位置变化" in change for change in transition.spatial_changes):
            if transition.transition_type == "cut":
                issues.append({
                    "type": "abrupt_spatial_change",
                    "severity": "low",
                    "description": "硬切伴随空间跳跃可能造成混淆",
                    "suggestion": "考虑使用淡入淡出或匹配剪辑"
                })

        return issues

    def generate_transition_advice(self, transition: TransitionInstruction) -> List[str]:
        """生成转场建议"""
        advice = []

        # 基于转场类型的建议
        if transition.transition_type == "cut":
            advice.append("硬切适合快速节奏的场景转换")
            advice.append("确保剪辑点在动作或对话的自然断点")

        elif transition.transition_type == "fade":
            advice.append("淡入淡出适合时间或场景的过渡")
            advice.append("控制淡入淡出的持续时间（通常1-2秒）")

        elif transition.transition_type == "dissolve":
            advice.append("溶解转场适合表现时间流逝或梦境")
            advice.append("避免过度使用溶解转场")

        # 基于角色变化的建议
        for char_id, changes in transition.character_transitions.items():
            if "appearance_change" in changes:
                advice.append(f"角色 {char_id} 外观变化，确保观众能识别")
            if "position_change" in changes:
                distance = changes["position_change"]["distance"]
                if distance > 10:
                    advice.append(f"角色 {char_id} 移动距离过大 ({distance:.1f}米)，可能需要解释")

        return advice
