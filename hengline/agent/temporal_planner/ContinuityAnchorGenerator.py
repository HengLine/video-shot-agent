"""
@FileName: ContinuityAnchorGenerator.py
@Description: 连续性锚点生成
@Author: HengLine
@Time: 2025/12/20 21:29
"""
from typing import List, Optional

from hengline.agent.temporal_planner.temporal_planner_model import ContinuityAnchor, TimeSegment


class ContinuityAnchorGenerator:
    """
    连续性锚点生成器

    为每个5秒片段生成视觉和逻辑锚点
    确保片段间的平滑过渡
    """

    def generate_anchors(self, segments: List[TimeSegment]) -> List[ContinuityAnchor]:
        """
        为所有分片生成连续性锚点
        """
        anchors = []

        for i in range(len(segments)):
            current = segments[i]

            # 1. 与前一片段的衔接锚点
            if i > 0:
                previous = segments[i - 1]
                visual_anchor = self._create_visual_match_anchor(previous, current)
                if visual_anchor:
                    anchors.append(visual_anchor)

            # 2. 与后一片段的准备锚点
            if i < len(segments) - 1:
                next_seg = segments[i + 1]
                transition_anchor = self._create_transition_anchor(current, next_seg)
                if transition_anchor:
                    anchors.append(transition_anchor)

            # 3. 片段内部的关键帧锚点
            internal_anchors = self._create_internal_anchors(current)
            anchors.extend(internal_anchors)

        return anchors

    def _create_visual_match_anchor(self, prev_segment: TimeSegment,
                                    curr_segment: TimeSegment) -> Optional[ContinuityAnchor]:
        """创建视觉匹配锚点"""
        # 提取需要保持一致的视觉元素
        common_elements = set(prev_segment.key_elements) & set(curr_segment.key_elements)

        if not common_elements:
            return None

        # 创建锚点描述
        anchor_description = f"保持以下元素一致: {', '.join(common_elements)}"

        # 提取具体的视觉匹配要求
        visual_requirements = []
        for element in common_elements:
            if "位置" in element or "姿势" in element:
                visual_requirements.append(f"{element} 应与前一片段保持一致")

        if visual_requirements:
            anchor_description += f"。具体要求: {'; '.join(visual_requirements)}"

        return ContinuityAnchor(
            anchor_id=f"anchor_{prev_segment.segment_id}_to_{curr_segment.segment_id}",
            type="visual_match",
            from_segment=prev_segment.segment_id,
            to_segment=curr_segment.segment_id,
            description=anchor_description,
            priority=8,  # 高优先级
            required_elements=list(common_elements)
        )

    def _create_transition_anchor(self, curr_segment: TimeSegment,
                                  next_segment: TimeSegment) -> ContinuityAnchor:
        """创建过渡锚点"""
        # 分析过渡类型
        transition_type = self._determine_transition_type(curr_segment, next_segment)

        # 创建过渡指令
        if transition_type == "smooth":
            description = f"平滑过渡到 {next_segment.segment_id}"
            priority = 6
        elif transition_type == "hard_cut":
            description = f"直接切换到 {next_segment.segment_id}"
            priority = 5
        elif transition_type == "match_cut":
            description = f"匹配剪辑到 {next_segment.segment_id} (保持视觉元素的一致性)"
            priority = 9
        else:
            description = f"过渡到 {next_segment.segment_id}"
            priority = 5

        return ContinuityAnchor(
            anchor_id=f"trans_{curr_segment.segment_id}_to_{next_segment.segment_id}",
            type="transition",
            from_segment=curr_segment.segment_id,
            to_segment=next_segment.segment_id,
            description=description,
            priority=priority,
            transition_type=transition_type
        )

    def _create_internal_anchors(self, segment: TimeSegment) -> List[ContinuityAnchor]:
        """创建片段内部的关键帧锚点"""
        anchors = []

        # 基于时长创建关键帧点
        duration = segment.duration
        if duration >= 3:
            # 中点关键帧
            mid_anchor = ContinuityAnchor(
                anchor_id=f"{segment.segment_id}_mid",
                type="keyframe",
                from_segment=segment.segment_id,
                to_segment=segment.segment_id,
                description=f"片段中点应突出: {self._extract_midpoint_content(segment)}",
                priority=4,
                timestamp=duration / 2
            )
            anchors.append(mid_anchor)

        return anchors