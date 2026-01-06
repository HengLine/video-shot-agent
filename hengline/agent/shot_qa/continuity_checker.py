"""
@FileName: continuity_checker.py
@Description: 连续性检查器 - 检查片段间的连续性
@Author: HengLine
@Time: 2026/1/6 16:03
"""
from typing import List, Dict, Any, Optional

from .model.check_models import ContinuityCheckResult, CheckStatus
from .model.issue_models import IssueSeverity


class ContinuityChecker:
    """连续性检查器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.position_tolerance = self.config.get("position_tolerance", 0.3)
        self.appearance_tolerance = self.config.get("appearance_tolerance", 0.1)
        self.time_gap_tolerance = self.config.get("time_gap_tolerance", 0.5)

    def check_all_continuity(self, shots: List[Any],
                             anchored_timeline: Any) -> List[ContinuityCheckResult]:
        """检查所有连续性"""
        results = []

        # 按时间顺序检查相邻镜头
        sorted_shots = sorted(shots, key=lambda x: x.time_range[0])

        for i in range(len(sorted_shots) - 1):
            current_shot = sorted_shots[i]
            next_shot = sorted_shots[i + 1]

            # 1. 位置连续性检查
            position_result = self.check_position_continuity(
                current_shot, next_shot, anchored_timeline
            )
            if position_result:
                results.append(position_result)

            # 2. 外观连续性检查
            appearance_result = self.check_appearance_continuity(
                current_shot, next_shot, anchored_timeline
            )
            if appearance_result:
                results.append(appearance_result)

            # 3. 动作连续性检查
            action_result = self.check_action_continuity(
                current_shot, next_shot, anchored_timeline
            )
            if action_result:
                results.append(action_result)

            # 4. 时间连续性检查
            temporal_result = self.check_temporal_continuity(
                current_shot, next_shot, anchored_timeline
            )
            if temporal_result:
                results.append(temporal_result)

        return results

    def check_position_continuity(self, shot_a: Any, shot_b: Any,
                                  anchored_timeline: Any) -> Optional[ContinuityCheckResult]:
        """检查位置连续性"""

        # 提取角色位置信息
        positions_a = self._extract_character_positions(shot_a)
        positions_b = self._extract_character_positions(shot_b)

        common_characters = set(positions_a.keys()) & set(positions_b.keys())

        if not common_characters:
            return None

        discrepancies = []
        max_discrepancy = 0.0

        for character in common_characters:
            pos_a = positions_a[character]
            pos_b = positions_b[character]

            # 计算位置差异（简化版）
            discrepancy = self._calculate_position_discrepancy(pos_a, pos_b)

            if discrepancy > self.position_tolerance:
                discrepancies.append({
                    "character": character,
                    "discrepancy": discrepancy,
                    "position_a": pos_a,
                    "position_b": pos_b
                })
                max_discrepancy = max(max_discrepancy, discrepancy)

        if not discrepancies:
            return ContinuityCheckResult(
                check_id=f"position_continuity_{shot_a.shot_id}_to_{shot_b.shot_id}",
                check_name="位置连续性检查",
                check_description="检查角色位置在镜头间是否连续",
                status=CheckStatus.PASSED,
                severity=IssueSeverity.INFO,
                score=1.0,
                continuity_type="position",
                details={"common_characters": list(common_characters)},
                evidence=["所有角色位置连续"]
            )
        else:
            # 计算得分
            score = max(0.0, 1.0 - (max_discrepancy / 2.0))  # 假设最大差异为2.0

            return ContinuityCheckResult(
                check_id=f"position_continuity_{shot_a.shot_id}_to_{shot_b.shot_id}",
                check_name="位置连续性检查",
                check_description="检查角色位置在镜头间是否连续",
                status=CheckStatus.FAILED,
                severity=IssueSeverity.HIGH if max_discrepancy > 0.5 else IssueSeverity.MEDIUM,
                score=score,
                continuity_type="position",
                position_discrepancy=max_discrepancy,
                details={
                    "discrepancies": discrepancies,
                    "tolerance": self.position_tolerance
                },
                evidence=[f"{len(discrepancies)}个角色位置不连续"],
                affected_elements=list(common_characters),
                previous_segment_id=shot_a.segment_id,
                current_segment_id=shot_b.segment_id,
                suggested_transition="使用匹配剪辑或动作连续性剪辑",
                recommended_adjustment="调整角色起始位置以匹配前一镜头结束位置"
            )

    def _extract_character_positions(self, shot: Any) -> Dict[str, Dict[str, Any]]:
        """从镜头中提取角色位置信息"""
        positions = {}

        # 从提示词中提取位置信息
        prompt = shot.full_sora_prompt.lower()

        # 简单的位置关键词提取
        position_keywords = {
            "left": ["left side", "on the left", "to the left"],
            "right": ["right side", "on the right", "to the right"],
            "center": ["center", "middle", "centered"],
            "foreground": ["foreground", "in front", "closer"],
            "background": ["background", "behind", "further"]
        }

        # 这里可以扩展为更复杂的NLP解析
        # 目前返回简单的位置估计
        for char in ["lin ran", "li wei", "character"]:  # 示例角色名
            if char in prompt:
                positions[char] = {
                    "estimated_position": "center",  # 简化
                    "confidence": 0.5
                }

        return positions

    def _calculate_position_discrepancy(self, pos_a: Dict[str, Any],
                                        pos_b: Dict[str, Any]) -> float:
        """计算位置差异度"""
        # 简化计算：如果位置描述相同，差异为0，否则为1
        if pos_a.get("estimated_position") == pos_b.get("estimated_position"):
            return 0.0
        else:
            return 1.0

    def check_appearance_continuity(self, shot_a: Any, shot_b: Any,
                                    anchored_timeline: Any) -> Optional[ContinuityCheckResult]:
        """检查外观连续性"""

        # 提取外观信息
        appearance_a = self._extract_appearance_info(shot_a)
        appearance_b = self._extract_appearance_info(shot_b)

        common_characters = set(appearance_a.keys()) & set(appearance_b.keys())

        if not common_characters:
            return None

        changes = []

        for character in common_characters:
            app_a = appearance_a[character]
            app_b = appearance_b[character]

            # 比较外观属性
            if app_a.get("clothing") != app_b.get("clothing"):
                changes.append(f"{character}: 服装从'{app_a.get('clothing')}'变为'{app_b.get('clothing')}'")

            if app_a.get("hairstyle") != app_b.get("hairstyle"):
                changes.append(f"{character}: 发型从'{app_a.get('hairstyle')}'变为'{app_b.get('hairstyle')}'")

            if app_a.get("accessories") != app_b.get("accessories"):
                changes.append(f"{character}: 配饰变化")

        if not changes:
            return ContinuityCheckResult(
                check_id=f"appearance_continuity_{shot_a.shot_id}_to_{shot_b.shot_id}",
                check_name="外观连续性检查",
                check_description="检查角色外观在镜头间是否连续",
                status=CheckStatus.PASSED,
                severity=IssueSeverity.INFO,
                score=1.0,
                continuity_type="appearance",
                details={"common_characters": list(common_characters)},
                evidence=["所有角色外观连续"]
            )
        else:
            # 计算得分：每个变化扣0.2分
            penalty = min(1.0, len(changes) * 0.2)
            score = max(0.0, 1.0 - penalty)

            severity = IssueSeverity.HIGH if "服装" in " ".join(changes) else IssueSeverity.MEDIUM

            return ContinuityCheckResult(
                check_id=f"appearance_continuity_{shot_a.shot_id}_to_{shot_b.shot_id}",
                check_name="外观连续性检查",
                check_description="检查角色外观在镜头间是否连续",
                status=CheckStatus.FAILED,
                severity=severity,
                score=score,
                continuity_type="appearance",
                appearance_changes=changes,
                details={
                    "changes": changes,
                    "character_count": len(common_characters)
                },
                evidence=changes[:3],  # 最多显示3个变化
                affected_elements=list(common_characters),
                previous_segment_id=shot_a.segment_id,
                current_segment_id=shot_b.segment_id,
                recommended_adjustment="确保角色服装、发型、配饰在不同镜头间保持一致"
            )

    def _extract_appearance_info(self, shot: Any) -> Dict[str, Dict[str, Any]]:
        """从镜头中提取外观信息"""
        appearance = {}

        # 从提示词中提取外观信息
        prompt = shot.full_sora_prompt.lower()

        # 简单的外观关键词提取
        appearance_patterns = {
            "clothing": ["wearing", "clothing", "dressed in", "shirt", "dress"],
            "hairstyle": ["hair", "hairstyle", "hairdo"],
            "accessories": ["glasses", "watch", "jewelry", "hat"]
        }

        # 这里可以扩展为更复杂的NLP解析
        # 目前返回简单的外观估计
        for char in ["lin ran", "li wei", "character"]:
            if char in prompt:
                appearance[char] = {
                    "clothing": "unknown",
                    "hairstyle": "unknown",
                    "accessories": []
                }

        return appearance

    def check_action_continuity(self, shot_a: Any, shot_b: Any,
                                anchored_timeline: Any) -> Optional[ContinuityCheckResult]:
        """检查动作连续性"""

        # 提取动作信息
        actions_a = self._extract_actions(shot_a)
        actions_b = self._extract_actions(shot_b)

        # 检查动作序列是否连贯
        discontinuity = None

        # 简单检查：如果shot_a以某个动作结束，shot_b应该以该动作继续或以自然过渡开始
        last_action_a = actions_a[-1] if actions_a else None
        first_action_b = actions_b[0] if actions_b else None

        if last_action_a and first_action_b:
            # 检查动作是否连贯
            if not self._are_actions_continuous(last_action_a, first_action_b):
                discontinuity = f"动作从'{last_action_a}'到'{first_action_b}'不连贯"

        if not discontinuity:
            return ContinuityCheckResult(
                check_id=f"action_continuity_{shot_a.shot_id}_to_{shot_b.shot_id}",
                check_name="动作连续性检查",
                check_description="检查角色动作在镜头间是否连续",
                status=CheckStatus.PASSED,
                severity=IssueSeverity.INFO,
                score=1.0,
                continuity_type="action",
                details={
                    "last_action_a": last_action_a,
                    "first_action_b": first_action_b
                },
                evidence=["动作序列连贯"]
            )
        else:
            return ContinuityCheckResult(
                check_id=f"action_continuity_{shot_a.shot_id}_to_{shot_b.shot_id}",
                check_name="动作连续性检查",
                check_description="检查角色动作在镜头间是否连续",
                status=CheckStatus.FAILED,
                severity=IssueSeverity.MEDIUM,
                score=0.6,
                continuity_type="action",
                action_discontinuity=discontinuity,
                details={
                    "last_action_a": last_action_a,
                    "first_action_b": first_action_b
                },
                evidence=[discontinuity],
                previous_segment_id=shot_a.segment_id,
                current_segment_id=shot_b.segment_id,
                suggested_transition="使用动作匹配剪辑或重叠动作",
                recommended_adjustment="调整动作起始帧以匹配前一镜头结束动作"
            )

    def _extract_actions(self, shot: Any) -> List[str]:
        """从镜头中提取动作信息"""
        actions = []
        prompt = shot.full_sora_prompt.lower()

        # 动作关键词
        action_keywords = [
            "sitting", "standing", "walking", "running", "talking",
            "looking", "holding", "smiling", "turning", "gesturing"
        ]

        for action in action_keywords:
            if action in prompt:
                actions.append(action)

        return actions

    def _are_actions_continuous(self, action_a: str, action_b: str) -> bool:
        """检查两个动作是否连贯"""
        # 定义动作连贯性规则
        continuous_pairs = [
            ("sitting", "standing"),
            ("standing", "walking"),
            ("walking", "running"),
            ("looking", "turning"),
            ("holding", "gesturing")
        ]

        # 如果动作相同或存在连贯对，则认为是连贯的
        return action_a == action_b or (action_a, action_b) in continuous_pairs

    def check_temporal_continuity(self, shot_a: Any, shot_b: Any,
                                  anchored_timeline: Any) -> Optional[ContinuityCheckResult]:
        """检查时间连续性"""

        end_time_a = shot_a.time_range[1]
        start_time_b = shot_b.time_range[0]

        time_gap = start_time_b - end_time_a

        if time_gap <= self.time_gap_tolerance:
            return ContinuityCheckResult(
                check_id=f"temporal_continuity_{shot_a.shot_id}_to_{shot_b.shot_id}",
                check_name="时间连续性检查",
                check_description="检查镜头间时间是否连续",
                status=CheckStatus.PASSED,
                severity=IssueSeverity.INFO,
                score=1.0,
                continuity_type="time",
                details={
                    "end_time_a": end_time_a,
                    "start_time_b": start_time_b,
                    "time_gap": time_gap
                },
                evidence=["时间连续，无显著间隙"]
            )
        else:
            score = max(0.0, 1.0 - (time_gap / 2.0))  # 假设最大允许间隙为2秒

            return ContinuityCheckResult(
                check_id=f"temporal_continuity_{shot_a.shot_id}_to_{shot_b.shot_id}",
                check_name="时间连续性检查",
                check_description="检查镜头间时间是否连续",
                status=CheckStatus.FAILED,
                severity=IssueSeverity.MEDIUM if time_gap < 1.0 else IssueSeverity.HIGH,
                score=score,
                continuity_type="time",
                time_gap=time_gap,
                details={
                    "end_time_a": end_time_a,
                    "start_time_b": start_time_b,
                    "time_gap": time_gap,
                    "tolerance": self.time_gap_tolerance
                },
                evidence=[f"时间间隙: {time_gap:.2f}秒"],
                previous_segment_id=shot_a.segment_id,
                current_segment_id=shot_b.segment_id,
                suggested_transition="使用淡入淡出或溶解过渡",
                recommended_adjustment="调整时间轴以消除间隙，或使用时间过渡效果"
            )

    def calculate_continuity_score(self, results: List[ContinuityCheckResult]) -> Dict[str, float]:
        """计算连续性评分"""
        if not results:
            return {
                "overall": 1.0,
                "position": 1.0,
                "appearance": 1.0,
                "action": 1.0,
                "temporal": 1.0
            }

        scores = {
            "position": [],
            "appearance": [],
            "action": [],
            "temporal": []
        }

        for result in results:
            if isinstance(result, ContinuityCheckResult):
                if result.continuity_type == "position":
                    scores["position"].append(result.score)
                elif result.continuity_type == "appearance":
                    scores["appearance"].append(result.score)
                elif result.continuity_type == "action":
                    scores["action"].append(result.score)
                elif result.continuity_type == "time":
                    scores["temporal"].append(result.score)

        # 计算平均分（如果没有检查项，默认1.0）
        avg_scores = {}
        for key, value_list in scores.items():
            if value_list:
                avg_scores[key] = sum(value_list) / len(value_list)
            else:
                avg_scores[key] = 1.0

        # 计算总分（加权平均）
        weights = {
            "position": 0.3,
            "appearance": 0.3,
            "action": 0.2,
            "temporal": 0.2
        }

        overall_score = sum(avg_scores[key] * weights[key] for key in avg_scores)

        avg_scores["overall"] = overall_score
        return avg_scores
