"""
@FileName: visual_validator.py
@Description: 
@Author: HengLine
@Time: 2026/1/5 22:48
"""
from typing import List

from hengline.agent.continuity_guardian.continuity_guardian_model import SegmentState, AnchoredSegment
from hengline.agent.continuity_guardian.model.continuity_guardian_report import ContinuityIssue


class VisualConsistencyValidator:
    """视觉一致性验证器"""

    def validate_across_segments(self, anchored_segments: List[AnchoredSegment]) -> List[ContinuityIssue]:
        """跨片段验证视觉一致性，如角色服装、环境灯光是否突兀变化"""
        issues = []

        # 按时间顺序两两检查相邻片段
        for i in range(len(anchored_segments) - 1):
            seg_end = anchored_segments[i].end_state
            seg_start_next = anchored_segments[i + 1].start_state

            # 检查角色外观一致性
            issues.extend(self._check_character_consistency(seg_end, seg_start_next))
            # 检查环境连续性（如白天突然变黑夜）
            issues.extend(self._check_environment_continuity(seg_end, seg_start_next))
            # 检查道具的连贯性（如一个杯子突然消失）
            issues.extend(self._check_prop_continuity(seg_end, seg_start_next))

        return issues

    def _check_character_consistency(self, state_a: SegmentState, state_b: SegmentState) -> List[ContinuityIssue]:
        issues = []
        for char_name in state_a.character_states:
            if char_name in state_b.character_states:
                char_a = state_a.character_states[char_name]
                char_b = state_b.character_states[char_name]

                # 检查关键外观属性
                if char_a.appearance.clothing != char_b.appearance.clothing:
                    issues.append(ContinuityIssue(
                        type="服装不一致",
                        severity="高",
                        description=f"角色'{char_name}'的服装在片段间发生改变",
                        location=f"从 {state_a.timestamp}s 到 {state_b.timestamp}s"
                    ))
                # 可继续检查发型、妆容、配饰等...
        return issues