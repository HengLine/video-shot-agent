"""
@FileName: splitter_validator.py
@Description: 分片验证器
@Author: HengLine
@Time: 2026/1/16 23:56
"""

from typing import List, Dict, Any

from hengline.agent.temporal_planner.temporal_planner_model import TimelinePlan, TimeSegment, ContinuityAnchor


class SplitterValidator:
    """验证分片结果的合理性"""

    def __init__(self):
        self.min_segment_duration = 2.0
        self.max_segment_duration = 5.5
        self.max_consecutive_short = 2
        self.time_overlap_tolerance = 0.05  # 5ms容差
        self.visual_consistency_threshold = 0.7  # 视觉一致性阈值

    def validate_timeline(self, plan: TimelinePlan) -> Dict[str, Any]:
        """验证整个时间线规划"""

        issues = []
        warnings = []

        # 1. 验证片段时长
        for segment in plan.timeline_segments:
            duration_issues = self._validate_segment_duration(segment)
            issues.extend(duration_issues)

        # 2. 验证片段间时长关系
        duration_pattern_issues = self._validate_duration_patterns(plan.timeline_segments)
        issues.extend(duration_pattern_issues)

        # 3. 验证叙事连贯性
        continuity_issues = self._validate_narrative_continuity(plan)
        issues.extend(continuity_issues)

        # 4. 验证视觉一致性
        visual_issues = self._validate_visual_consistency(plan)
        issues.extend(visual_issues)

        # 5. 检查时间重叠
        overlap_issues = self._check_time_overlaps(plan.timeline_segments)
        issues.extend(overlap_issues)

        # 6. 验证锚点合理性
        anchor_issues = self._validate_continuity_anchors(plan.continuity_anchors, plan.timeline_segments)
        issues.extend(anchor_issues)

        # 7. 检查节奏合理性
        pacing_issues = self._validate_pacing_pattern(plan.pacing_analysis, plan.timeline_segments)
        issues.extend(pacing_issues)

        # 生成验证报告
        validation_report = {
            "total_issues": len(issues),
            "critical_issues": len([i for i in issues if i.get("severity") == "critical"]),
            "issues": issues,
            "warnings": warnings,
            "is_valid": len([i for i in issues if i.get("severity") == "critical"]) == 0,
            "segment_count": len(plan.timeline_segments),
            "total_duration": plan.total_duration,
            "segment_duration_stats": self._calculate_duration_stats(plan.timeline_segments)
        }

        return validation_report

    def _validate_segment_duration(self, segment: TimeSegment) -> List[Dict[str, Any]]:
        """验证片段时长"""
        issues = []

        if segment.duration < self.min_segment_duration:
            issues.append({
                "type": "duration_too_short",
                "severity": "warning",
                "segment_id": segment.segment_id,
                "duration": segment.duration,
                "message": f"片段时长({segment.duration:.1f}s)过短，可能影响叙事完整性",
                "suggestion": "考虑与前一片段合并或调整元素分配"
            })

        if segment.duration > self.max_segment_duration:
            issues.append({
                "type": "duration_too_long",
                "severity": "critical",
                "segment_id": segment.segment_id,
                "duration": segment.duration,
                "message": f"片段时长({segment.duration:.1f}s)超过5.5秒限制",
                "suggestion": "需要重新分割该片段，保持5秒左右时长"
            })

        # 检查是否接近理想时长
        if abs(segment.duration - 5.0) > 1.0:
            issues.append({
                "type": "duration_suboptimal",
                "severity": "info",
                "segment_id": segment.segment_id,
                "duration": segment.duration,
                "message": f"片段时长({segment.duration:.1f}s)偏离5秒标准",
                "suggestion": "理想时长应在4.5-5.5秒之间"
            })

        return issues

    def _validate_duration_patterns(self, segments: List[TimeSegment]) -> List[Dict[str, Any]]:
        """验证时长模式"""
        issues = []

        # 检查连续短片段
        consecutive_short = 0
        for i, segment in enumerate(segments):
            if segment.duration < 3.0:
                consecutive_short += 1
                if consecutive_short > self.max_consecutive_short:
                    issues.append({
                        "type": "consecutive_short_segments",
                        "severity": "warning",
                        "segment_ids": [segments[j].segment_id for j in range(i - consecutive_short + 1, i + 1)],
                        "message": f"连续{consecutive_short}个短片段（<3s），可能导致节奏过快",
                        "suggestion": "考虑合并部分短片段"
                    })
            else:
                consecutive_short = 0

        # 检查时长分布
        durations = [s.duration for s in segments]
        if len(durations) > 5:
            avg_duration = sum(durations) / len(durations)
            duration_variance = sum((d - avg_duration) ** 2 for d in durations) / len(durations)

            if duration_variance > 1.5:
                issues.append({
                    "type": "inconsistent_durations",
                    "severity": "warning",
                    "variance": duration_variance,
                    "message": f"片段时长差异较大（方差：{duration_variance:.2f}），可能影响观看节奏",
                    "suggestion": "调整分割点，使片段时长更均匀"
                })

        return issues

    def _validate_narrative_continuity(self, plan: TimelinePlan) -> List[Dict[str, Any]]:
        """验证叙事连贯性"""
        issues = []
        segments = plan.timeline_segments

        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]

            # 1. 检查叙事弧的连贯性
            if current.narrative_arc and next_seg.narrative_arc:
                continuity_score = self._calculate_narrative_continuity(
                    current.narrative_arc, next_seg.narrative_arc
                )

                if continuity_score < 0.4:
                    issues.append({
                        "type": "narrative_discontinuity",
                        "severity": "warning",
                        "from_segment": current.segment_id,
                        "to_segment": next_seg.segment_id,
                        "continuity_score": continuity_score,
                        "message": f"叙事弧不连贯：'{current.narrative_arc}' → '{next_seg.narrative_arc}'",
                        "suggestion": "检查是否需要调整叙事弧或添加过渡元素"
                    })

            # 2. 检查对话连续性
            dialogue_continuity = self._check_dialogue_continuity(current, next_seg, plan)
            if dialogue_continuity.get("needs_attention"):
                issues.append({
                    "type": "dialogue_continuity_issue",
                    "severity": dialogue_continuity.get("severity", "warning"),
                    "from_segment": current.segment_id,
                    "to_segment": next_seg.segment_id,
                    "message": dialogue_continuity.get("message"),
                    "suggestion": dialogue_continuity.get("suggestion", "确保对话自然衔接")
                })

            # 3. 检查动作序列合理性
            action_continuity = self._check_action_continuity(current, next_seg, plan)
            if action_continuity.get("issues"):
                issues.extend(action_continuity["issues"])

        return issues

    def _validate_visual_consistency(self, plan: TimelinePlan) -> List[Dict[str, Any]]:
        """验证视觉一致性"""
        issues = []
        segments = plan.timeline_segments

        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]

            # 1. 检查视觉标签一致性
            current_tags = current.visual_consistency_tags
            next_tags = next_seg.visual_consistency_tags

            common_tags = current_tags.intersection(next_tags)
            all_tags = current_tags.union(next_tags)

            if all_tags:  # 避免除以零
                consistency_score = len(common_tags) / len(all_tags)

                if consistency_score < self.visual_consistency_threshold:
                    lost_tags = current_tags - next_tags
                    new_tags = next_tags - current_tags

                    issues.append({
                        "type": "visual_inconsistency",
                        "severity": "warning",
                        "from_segment": current.segment_id,
                        "to_segment": next_seg.segment_id,
                        "consistency_score": consistency_score,
                        "lost_tags": list(lost_tags),
                        "new_tags": list(new_tags),
                        "message": f"视觉一致性较低（{consistency_score:.2f}），丢失标签：{lost_tags}，新增标签：{new_tags}",
                        "suggestion": "考虑是否合理，如果不合理需要调整分片或添加视觉匹配锚点"
                    })

            # 2. 检查场景突变
            scene_change_score = self._detect_scene_change(current, next_seg)
            if scene_change_score > 0.8:
                issues.append({
                    "type": "abrupt_scene_change",
                    "severity": "warning",
                    "from_segment": current.segment_id,
                    "to_segment": next_seg.segment_id,
                    "change_score": scene_change_score,
                    "message": "检测到可能突兀的场景变化",
                    "suggestion": "检查是否需要添加过渡或调整分片位置"
                })

            # 3. 检查角色位置连续性
            character_position_issues = self._check_character_position_continuity(current, next_seg, plan)
            if character_position_issues:
                issues.extend(character_position_issues)

        return issues

    def _check_time_overlaps(self, segments: List[TimeSegment]) -> List[Dict[str, Any]]:
        """检查时间重叠"""
        issues = []

        # 按开始时间排序
        sorted_segments = sorted(segments, key=lambda s: s.start_time)

        for i in range(len(sorted_segments) - 1):
            current = sorted_segments[i]
            next_seg = sorted_segments[i + 1]

            # 检查重叠
            if current.end_time > next_seg.start_time + self.time_overlap_tolerance:
                overlap_duration = current.end_time - next_seg.start_time

                issues.append({
                    "type": "time_overlap",
                    "severity": "critical",
                    "segment1": current.segment_id,
                    "segment2": next_seg.segment_id,
                    "overlap_duration": overlap_duration,
                    "message": f"时间重叠：{current.segment_id}与{next_seg.segment_id}重叠{overlap_duration:.2f}秒",
                    "suggestion": "调整时间范围，消除重叠"
                })

            # 检查间隙过大
            gap = next_seg.start_time - current.end_time
            if gap > 0.5:  # 大于0.5秒的间隙
                issues.append({
                    "type": "time_gap",
                    "severity": "warning",
                    "segment1": current.segment_id,
                    "segment2": next_seg.segment_id,
                    "gap_duration": gap,
                    "message": f"时间间隙：{current.segment_id}与{next_seg.segment_id}之间有{gap:.2f}秒间隙",
                    "suggestion": "检查是否有缺失内容或需要调整时间范围"
                })

        # 检查整体时间线完整性
        if sorted_segments:
            first_segment = sorted_segments[0]
            last_segment = sorted_segments[-1]

            if abs(first_segment.start_time - 0.0) > 0.01:
                issues.append({
                    "type": "timeline_not_start_at_zero",
                    "severity": "info",
                    "actual_start": first_segment.start_time,
                    "message": f"时间线未从0秒开始，实际开始时间：{first_segment.start_time:.2f}s",
                    "suggestion": "如有必要，调整起始时间"
                })

            # 检查是否有时间跳跃
            total_coverage = sum(seg.duration for seg in sorted_segments)
            timeline_span = last_segment.end_time - first_segment.start_time

            if abs(total_coverage - timeline_span) > 0.1:
                issues.append({
                    "type": "timeline_coverage_issue",
                    "severity": "info",
                    "total_coverage": total_coverage,
                    "timeline_span": timeline_span,
                    "message": f"片段总时长({total_coverage:.2f}s)与时间线跨度({timeline_span:.2f}s)不一致",
                    "suggestion": "检查时间分配是否合理"
                })

        return issues

    def _validate_continuity_anchors(
            self,
            anchors: List[ContinuityAnchor],
            segments: List[TimeSegment]
    ) -> List[Dict[str, Any]]:
        """验证连贯性锚点的合理性"""
        issues = []

        segment_ids = {s.segment_id for s in segments}

        for anchor in anchors:
            # 1. 检查锚点引用的片段是否存在
            if anchor.from_segment not in segment_ids:
                issues.append({
                    "type": "anchor_reference_invalid",
                    "severity": "critical",
                    "anchor_id": anchor.anchor_id,
                    "missing_segment": anchor.from_segment,
                    "message": f"锚点{anchor.anchor_id}引用的片段{anchor.from_segment}不存在",
                    "suggestion": "修复片段引用或重新生成锚点"
                })

            if anchor.to_segment not in segment_ids:
                issues.append({
                    "type": "anchor_reference_invalid",
                    "severity": "critical",
                    "anchor_id": anchor.anchor_id,
                    "missing_segment": anchor.to_segment,
                    "message": f"锚点{anchor.anchor_id}引用的片段{anchor.to_segment}不存在",
                    "suggestion": "修复片段引用或重新生成锚点"
                })

            # 2. 检查锚点类型是否合理
            if anchor.anchor_type not in ["visual_match", "transition", "keyframe", "character_state"]:
                issues.append({
                    "type": "invalid_anchor_type",
                    "severity": "warning",
                    "anchor_id": anchor.anchor_id,
                    "anchor_type": anchor.anchor_type,
                    "message": f"锚点类型'{anchor.anchor_type}'不被支持",
                    "suggestion": "使用支持的锚点类型：visual_match, transition, keyframe, character_state"
                })

            # 3. 检查优先级范围
            if not (1.0 <= anchor.priority <= 10.0):
                issues.append({
                    "type": "invalid_anchor_priority",
                    "severity": "warning",
                    "anchor_id": anchor.anchor_id,
                    "priority": anchor.priority,
                    "message": f"锚点优先级{anchor.priority}超出1-10范围",
                    "suggestion": "将优先级调整到1-10之间"
                })

        # 4. 检查锚点覆盖范围
        segment_anchor_counts = {}
        for anchor in anchors:
            segment_anchor_counts[anchor.from_segment] = segment_anchor_counts.get(anchor.from_segment, 0) + 1
            segment_anchor_counts[anchor.to_segment] = segment_anchor_counts.get(anchor.to_segment, 0) + 1

        for segment_id, count in segment_anchor_counts.items():
            if count > 10:
                issues.append({
                    "type": "excessive_anchors",
                    "severity": "warning",
                    "segment_id": segment_id,
                    "anchor_count": count,
                    "message": f"片段{segment_id}有过多锚点({count}个)，可能限制生成灵活性",
                    "suggestion": "考虑合并或简化部分锚点"
                })
            elif count == 0 and segment_id in segment_ids:
                issues.append({
                    "type": "no_anchors",
                    "severity": "info",
                    "segment_id": segment_id,
                    "message": f"片段{segment_id}没有锚点约束",
                    "suggestion": "如有必要，添加适当的连贯性锚点"
                })

        return issues

    def _validate_pacing_pattern(
            self,
            pacing_analysis: Any,
            segments: List[TimeSegment]
    ) -> List[Dict[str, Any]]:
        """验证节奏模式的合理性"""
        issues = []

        if not hasattr(pacing_analysis, 'intensity_curve'):
            return issues

        intensity_curve = pacing_analysis.intensity_curve

        # 检查强度曲线的合理性
        if len(intensity_curve) != len(segments):
            issues.append({
                "type": "pacing_curve_mismatch",
                "severity": "warning",
                "curve_length": len(intensity_curve),
                "segment_count": len(segments),
                "message": "节奏强度曲线长度与片段数量不匹配",
                "suggestion": "重新计算节奏分析"
            })

        # 检查强度突变
        for i in range(1, len(intensity_curve)):
            intensity_change = abs(intensity_curve[i] - intensity_curve[i - 1])
            if intensity_change > 1.5:
                issues.append({
                    "type": "abrupt_intensity_change",
                    "severity": "warning",
                    "segment_index": i,
                    "intensity_change": intensity_change,
                    "message": f"检测到节奏强度突变（变化值：{intensity_change:.2f}）",
                    "suggestion": "检查是否需要平滑过渡或重新分片"
                })

        # 检查是否有过长的低强度或高强度序列
        low_intensity_streak = 0
        high_intensity_streak = 0
        for i, intensity in enumerate(intensity_curve):
            if intensity < 0.8:
                low_intensity_streak += 1
                high_intensity_streak = 0
            elif intensity > 1.8:
                high_intensity_streak += 1
                low_intensity_streak = 0
            else:
                low_intensity_streak = 0
                high_intensity_streak = 0

            if low_intensity_streak >= 4:
                issues.append({
                    "type": "prolonged_low_intensity",
                    "severity": "info",
                    "segment_range": f"{i - low_intensity_streak + 1}-{i}",
                    "streak_length": low_intensity_streak,
                    "message": f"检测到连续{low_intensity_streak}个低强度片段",
                    "suggestion": "考虑调整节奏或重新分片以增加变化"
                })

            if high_intensity_streak >= 3:
                issues.append({
                    "type": "prolonged_high_intensity",
                    "severity": "warning",
                    "segment_range": f"{i - high_intensity_streak + 1}-{i}",
                    "streak_length": high_intensity_streak,
                    "message": f"检测到连续{high_intensity_streak}个高强度片段，可能导致观众疲劳",
                    "suggestion": "考虑插入休息点或调整强度分布"
                })

        return issues

    def _calculate_narrative_continuity(self, arc1: str, arc2: str) -> float:
        """计算两个叙事弧的连贯性分数"""
        if not arc1 or not arc2:
            return 0.0

        # 简单的关键词匹配
        keywords1 = set(arc1.lower().split())
        keywords2 = set(arc2.lower().split())

        if not keywords1 or not keywords2:
            return 0.0

        common_keywords = keywords1.intersection(keywords2)
        return len(common_keywords) / max(len(keywords1), len(keywords2))

    def _check_dialogue_continuity(
            self,
            current: TimeSegment,
            next_seg: TimeSegment,
            plan: TimelinePlan
    ) -> Dict[str, Any]:
        """检查对话连续性"""
        # 获取两个片段的对话元素
        current_dialogues = []
        next_dialogues = []

        for element_id in current.element_coverage:
            if element_id.startswith("dialogue_"):
                # 这里需要访问plan.duration_estimations或存储的原始数据
                current_dialogues.append(element_id)

        for element_id in next_seg.element_coverage:
            if element_id.startswith("dialogue_"):
                next_dialogues.append(element_id)

        if not current_dialogues and not next_dialogues:
            return {"needs_attention": False}

        # 检查对话中断
        if current_dialogues and not next_dialogues:
            # 前一片段有对话，后一片段没有
            return {
                "needs_attention": True,
                "severity": "info",
                "message": f"对话在片段{current.segment_id}结束",
                "suggestion": "检查是否为有意中断"
            }

        if not current_dialogues and next_dialogues:
            # 新对话开始
            return {
                "needs_attention": True,
                "severity": "info",
                "message": f"新对话在片段{next_seg.segment_id}开始",
                "suggestion": "确保对话开始自然"
            }

        # 两个片段都有对话，检查对话者连续性
        # 这里需要更多上下文信息，简化处理
        return {"needs_attention": False}

    def _check_action_continuity(
            self,
            current: TimeSegment,
            next_seg: TimeSegment,
            plan: TimelinePlan
    ) -> Dict[str, Any]:
        """检查动作序列连续性"""
        issues = []

        # 简化的动作连续性检查
        # 实际实现需要分析具体的动作序列

        return {"issues": issues, "needs_attention": len(issues) > 0}

    def _detect_scene_change(self, current: TimeSegment, next_seg: TimeSegment) -> float:
        """检测场景变化程度"""
        current_tags = current.visual_consistency_tags
        next_tags = next_seg.visual_consistency_tags

        if not current_tags or not next_tags:
            return 0.0

        # 提取场景相关标签
        location_tags1 = {t for t in current_tags if t.startswith("location_")}
        location_tags2 = {t for t in next_tags if t.startswith("location_")}

        if not location_tags1 or not location_tags2:
            return 0.0

        # 检查位置是否相同
        if location_tags1 != location_tags2:
            return 0.9  # 位置不同，高概率场景变化

        # 检查时间/天气变化
        time_tags1 = {t for t in current_tags if t.startswith("time_") or t.startswith("weather_")}
        time_tags2 = {t for t in next_tags if t.startswith("time_") or t.startswith("weather_")}

        if time_tags1 != time_tags2:
            return 0.7  # 时间/天气变化，中等概率场景变化

        return 0.1  # 低概率场景变化

    def _check_character_position_continuity(
            self,
            current: TimeSegment,
            next_seg: TimeSegment,
            plan: TimelinePlan
    ) -> List[Dict[str, Any]]:
        """检查角色位置连续性"""
        issues = []

        # 简化的位置连续性检查
        # 实际实现需要跟踪角色在场景中的位置

        return issues

    def _calculate_duration_stats(self, segments: List[TimeSegment]) -> Dict[str, Any]:
        """计算时长统计信息"""
        if not segments:
            return {}

        durations = [s.duration for s in segments]

        return {
            "min": min(durations),
            "max": max(durations),
            "average": sum(durations) / len(durations),
            "median": sorted(durations)[len(durations) // 2],
            "std_dev": self._calculate_std_dev(durations),
            "ideal_segments": len([d for d in durations if 4.5 <= d <= 5.5]),
            "short_segments": len([d for d in durations if d < 3.0]),
            "long_segments": len([d for d in durations if d > 5.5])
        }

    def _calculate_std_dev(self, durations: List[float]) -> float:
        """计算标准差"""
        if len(durations) < 2:
            return 0.0

        mean = sum(durations) / len(durations)
        variance = sum((d - mean) ** 2 for d in durations) / len(durations)
        return variance ** 0.5
