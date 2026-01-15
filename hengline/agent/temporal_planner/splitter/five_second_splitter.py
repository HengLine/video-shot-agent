"""
@FileName: five_second_splitter.py
@Description: 
@Author: HengLine
@Time: 2026/1/15 0:00
"""
import math
from typing import List, Dict, Any

from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.splitter.element_boundary_analyzer import ElementBoundaryAnalyzer
from hengline.agent.temporal_planner.splitter.splitter_config import SplitterConfig
from hengline.agent.temporal_planner.splitter.splitter_model import SplitDecision, SplitResult, SegmentSplitResult
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, ElementType, TimeSegment, ContainedElement


class FiveSecondSplitter:
    """5秒分片器核心类"""

    def __init__(self, config: SplitterConfig = None):
        self.config = config or SplitterConfig()
        self.boundary_analyzer = ElementBoundaryAnalyzer()

    def split_into_segments(self,
                            estimations: Dict[str, DurationEstimation],
                            element_order: List[str],
                            original_data: UnifiedScript = None) -> SegmentSplitResult:
        """
        将估算后的元素切分为5秒片段

        Args:
            estimations: 时长估算字典，key为元素ID
            element_order: 元素顺序列表
            original_data: 原始元素数据（可选）

        Returns:
            SegmentSplitResult: 分片结果
        """
        print(f"开始5秒分片处理，共{len(element_order)}个元素")

        # 准备数据
        segments = []
        split_decisions = []
        current_segment = self._create_empty_segment(len(segments))
        current_time = 0.0
        segment_start_time = 0.0

        # 处理每个元素
        for element_id in element_order:
            if element_id not in estimations:
                print(f"警告：元素 {element_id} 不在估算结果中，跳过")
                continue

            element = estimations[element_id]
            element_duration = element.estimated_duration

            # 检查是否需要开始新片段
            current_segment_duration = current_time - segment_start_time
            remaining_capacity = self.config.target_segment_duration - current_segment_duration

            # 如果当前片段已满，结束它
            if current_segment_duration >= self.config.target_segment_duration - self.config.segment_tolerance:
                self._finalize_segment(current_segment, segment_start_time, current_time)
                segments.append(current_segment)

                # 开始新片段
                segment_start_time = current_time
                current_segment = self._create_empty_segment(len(segments))
                current_segment_duration = 0.0
                remaining_capacity = self.config.target_segment_duration

            # 处理当前元素
            if element_duration <= remaining_capacity:
                # 元素能完整放入当前片段
                self._add_complete_element(current_segment, element, current_time,
                                           element_duration, original_data)
                current_time += element_duration

            else:
                # 元素需要切割
                print(f"需要切割元素: {element_id} (时长: {element_duration:.2f}秒)")

                # 决定切割策略
                if not element.can_be_split or element.split_priority <= 2:
                    # 优先不切割的元素，开始新片段
                    if current_segment_duration > 0:
                        # 先完成当前片段
                        self._finalize_segment(current_segment, segment_start_time, current_time)
                        segments.append(current_segment)

                    # 在新片段中开始这个元素
                    segment_start_time = current_time
                    current_segment = self._create_empty_segment(len(segments))

                    if element_duration > self.config.max_segment_duration:
                        # 超长元素，必须切割
                        print(f"超长元素 {element_id} 必须切割")
                        self._split_large_element(current_segment, element, current_time,
                                                  remaining_capacity, split_decisions, original_data)
                        current_time += element_duration
                    else:
                        # 可以完整放入新片段
                        self._add_complete_element(current_segment, element, current_time,
                                                   element_duration, original_data)
                        current_time += element_duration

                else:
                    # 可以切割的元素
                    split_result = self._split_element(current_segment, element, current_time,
                                                       remaining_capacity, split_decisions, original_data)

                    # 完成当前片段
                    self._finalize_segment(current_segment, segment_start_time,
                                           segment_start_time + self.config.target_segment_duration)
                    segments.append(current_segment)

                    # 创建新片段继续处理剩余部分
                    current_time = segment_start_time + self.config.target_segment_duration
                    segment_start_time = current_time
                    current_segment = self._create_empty_segment(len(segments))

                    # 添加剩余部分到新片段
                    if split_result.remaining_duration > 0:
                        self._add_partial_element(current_segment, element, current_time,
                                                  split_result.remaining_duration, "continue",
                                                  split_result.split_point, original_data)
                        current_time += split_result.remaining_duration

        # 处理最后一个片段
        if current_segment.contained_elements:
            final_duration = current_time - segment_start_time

            # 确保最后一个片段不要太短
            if final_duration < self.config.min_segment_duration and len(segments) > 0:
                # 合并到上一个片段
                last_segment = segments[-1]
                self._merge_segment(last_segment, current_segment, segment_start_time)
            else:
                self._finalize_segment(current_segment, segment_start_time, current_time)
                segments.append(current_segment)

        # 计算质量和统计信息
        statistics = self._calculate_statistics(segments, estimations, element_order)
        quality_scores = self._evaluate_segments_quality(segments, split_decisions)

        # 生成最终结果
        result = SegmentSplitResult(
            segments=segments,
            split_decisions=split_decisions,
            statistics=statistics,
            overall_quality_score=quality_scores["overall"],
            pacing_consistency_score=quality_scores["pacing_consistency"],
            continuity_score=quality_scores["continuity"]
        )

        print(f"分片完成：生成 {len(segments)} 个片段，总体质量 {quality_scores['overall']:.2f}")
        return result

    def _create_empty_segment(self, segment_index: int) -> TimeSegment:
        """创建空片段"""
        return TimeSegment(
            segment_id=f"seg_{segment_index + 1:03d}",
            time_range=(0.0, 0.0),
            duration=0.0,
            visual_summary="",
            contained_elements=[],
            start_anchor={},
            end_anchor={},
            continuity_requirements=[],
            pacing_score=5.0
        )

    def _add_complete_element(self, segment: TimeSegment, element: DurationEstimation,
                              global_start_time: float, duration: float,
                              original_data: Dict[str, Any]) -> None:
        """添加完整元素到片段"""
        start_offset = global_start_time - segment.time_range[0] if segment.time_range[0] > 0 else global_start_time

        contained_elem = ContainedElement(
            element_id=element.element_id,
            element_type=element.element_type,
            start_offset=start_offset,
            duration=duration,
            is_partial=False,
            element_data=original_data.get(element.element_id, {}) if original_data else {}
        )

        segment.contained_elements.append(contained_elem)

        # 更新片段状态
        if element.character_states:
            segment.end_anchor.setdefault("character_states", {}).update(element.character_states)
        if element.prop_states:
            segment.end_anchor.setdefault("prop_states", {}).update(element.prop_states)

        # 更新视觉摘要
        self._update_visual_summary(segment, element, "complete")

    def _add_partial_element(self, segment: TimeSegment, element: DurationEstimation,
                             global_start_time: float, duration: float, partial_type: str,
                             split_point: float = None, original_data: Dict[str, Any] = None) -> ContainedElement:
        """添加部分元素到片段"""
        start_offset = global_start_time - segment.time_range[0] if segment.time_range[0] > 0 else global_start_time

        contained_elem = ContainedElement(
            element_id=element.element_id,
            element_type=element.element_type,
            start_offset=start_offset,
            duration=duration,
            is_partial=True,
            partial_type=partial_type,
            element_data=original_data.get(element.element_id, {}) if original_data else {}
        )

        segment.contained_elements.append(contained_elem)

        # 更新视觉摘要
        self._update_visual_summary(segment, element, f"partial_{partial_type}")

        return contained_elem

    def _split_element(self, segment: TimeSegment, element: DurationEstimation,
                       global_start_time: float, remaining_capacity: float,
                       split_decisions: List[SplitDecision],
                       original_data: Dict[str, Any]) -> 'SplitResult':
        """切割元素并添加到当前片段"""

        # 寻找最佳切割点
        best_split_point = self._find_best_split_point(element, remaining_capacity, original_data)

        # 计算各部分时长
        part1_duration = element.estimated_duration * best_split_point
        part2_duration = element.estimated_duration - part1_duration

        # 确保第一部分能放入剩余容量
        if part1_duration > remaining_capacity:
            # 调整切割点
            best_split_point = remaining_capacity / element.estimated_duration
            part1_duration = remaining_capacity
            part2_duration = element.estimated_duration - part1_duration

        # 添加第一部分到当前片段
        self._add_partial_element(segment, element, global_start_time,
                                  part1_duration, "start", best_split_point, original_data)

        # 记录切割决策
        split_decision = SplitDecision(
            element_id=element.element_id,
            split_point=best_split_point,
            reason=self._get_split_reason(element, best_split_point),
            quality_score=self.boundary_analyzer.assess_split_quality(
                best_split_point, element.element_type,
                original_data.get(element.element_id, {}) if original_data else {}
            ),
            is_natural_boundary=best_split_point in self.boundary_analyzer.find_natural_boundaries(
                original_data.get(element.element_id, {}).get("description", "") if original_data else "",
                element.element_type
            )
        )
        split_decisions.append(split_decision)

        # 返回切割结果
        return SplitResult(
            split_point=best_split_point,
            part1_duration=part1_duration,
            remaining_duration=part2_duration
        )

    def _split_large_element(self, segment: TimeSegment, element: DurationEstimation,
                             global_start_time: float, remaining_capacity: float,
                             split_decisions: List[SplitDecision],
                             original_data: Dict[str, Any]) -> None:
        """处理超长元素（必须切割）"""
        total_duration = element.estimated_duration
        current_position = 0.0

        # 将超长元素切割为多个5秒片段
        while current_position < total_duration:
            # 计算当前片段能容纳的部分
            segment_remaining = self.config.target_segment_duration - segment.duration
            part_duration = min(segment_remaining, total_duration - current_position)

            if part_duration <= 0:
                break

            # 计算切割点
            split_point = current_position / total_duration

            # 添加部分元素
            partial_type = "start" if current_position == 0 else "middle"
            self._add_partial_element(segment, element, global_start_time + current_position,
                                      part_duration, partial_type, split_point, original_data)

            current_position += part_duration

            # 如果当前片段满了，结束它
            if abs(segment.duration - self.config.target_segment_duration) < self.config.segment_tolerance:
                self._finalize_segment(segment, segment.time_range[0],
                                       segment.time_range[0] + segment.duration)
                # 创建新片段继续
                new_segment = self._create_empty_segment(len(self.segments) if hasattr(self, 'segments') else 0)
                segment = new_segment

        # 记录切割决策
        split_decision = SplitDecision(
            element_id=element.element_id,
            split_point=1.0,  # 表示多次切割
            reason=f"超长元素强制切割为{math.ceil(total_duration / 5)}段",
            quality_score=0.6  # 强制切割质量较低
        )
        split_decisions.append(split_decision)

    def _find_best_split_point(self, element: DurationEstimation,
                               remaining_capacity: float,
                               original_data: Dict[str, Any]) -> float:
        """寻找最佳切割点"""
        element_duration = element.estimated_duration

        # 如果元素可以完整放入下一个片段，尽量不切割
        if element_duration <= self.config.target_segment_duration and element.split_priority <= 3:
            return 0.0  # 表示不切割

        # 寻找自然边界
        natural_boundaries = []
        if original_data and element.element_id in original_data:
            elem_data = original_data[element.element_id]
            description = elem_data.get("description", "")
            natural_boundaries = self.boundary_analyzer.find_natural_boundaries(
                description, element.element_type
            )

        # 优先选择自然边界
        for boundary in natural_boundaries:
            part_duration = element_duration * boundary
            if part_duration <= remaining_capacity:
                return boundary

        # 如果没有合适的自然边界，选择能填满剩余容量的点
        target_split = remaining_capacity / element_duration

        # 确保切割点不在首尾太近的位置
        if target_split < 0.2:
            target_split = 0.2
        elif target_split > 0.8:
            target_split = 0.8

        return target_split

    def _update_visual_summary(self, segment: TimeSegment, element: DurationEstimation,
                               addition_type: str) -> None:
        """更新视觉摘要"""
        elem_type_str = element.element_type.value

        if addition_type == "complete":
            summary_part = f"[{elem_type_str}:{element.element_id}]"
        elif addition_type.startswith("partial"):
            summary_part = f"[部分{elem_type_str}:{element.element_id}]"
        else:
            summary_part = f"[{elem_type_str}:{element.element_id}]"

        if not segment.visual_summary:
            segment.visual_summary = summary_part
        else:
            segment.visual_summary += " → " + summary_part

    def _finalize_segment(self, segment: TimeSegment, start_time: float, end_time: float) -> None:
        """完成片段的最终设置"""
        segment.time_range = (start_time, end_time)
        segment.duration = end_time - start_time

        # 计算片段质量
        segment.pacing_score = self._calculate_segment_pacing_score(segment)
        segment.completeness_score = self._calculate_segment_completeness_score(segment)
        segment.split_quality = self._calculate_segment_split_quality(segment)

        # 设置片段类型
        segment.segment_type = self._determine_segment_type(segment)

        # 生成视觉摘要（如果为空）
        if not segment.visual_summary and segment.contained_elements:
            element_types = [elem.element_type.value for elem in segment.contained_elements]
            type_counts = {}
            for t in element_types:
                type_counts[t] = type_counts.get(t, 0) + 1

            summary_parts = []
            for elem_type, count in type_counts.items():
                if count == 1:
                    summary_parts.append(elem_type)
                else:
                    summary_parts.append(f"{elem_type}x{count}")

            segment.visual_summary = " + ".join(summary_parts)

        # 为后续智能体生成建议
        self._generate_suggestions_for_segment(segment)

    def _generate_suggestions_for_segment(self, segment: TimeSegment) -> None:
        """为片段生成智能体建议"""
        # 镜头类型建议
        if any(elem.element_type == ElementType.DIALOGUE for elem in segment.contained_elements):
            segment.shot_type_suggestion = "medium_close_up"
            segment.camera_movement = "static"
        elif any(elem.element_type == ElementType.ACTION for elem in segment.contained_elements):
            segment.shot_type_suggestion = "medium_shot"
            segment.camera_movement = "slight_pan"
        elif any(elem.element_type == ElementType.SCENE for elem in segment.contained_elements):
            segment.shot_type_suggestion = "wide_shot"
            segment.camera_movement = "slow_pan"

        # 灯光建议
        emotional_elements = [e for e in segment.contained_elements
                              if getattr(e.element_data, 'emotional_weight', 1.0) > 1.5]
        if emotional_elements:
            segment.lighting_suggestion = "dramatic_side_lighting"
        else:
            segment.lighting_suggestion = "natural_lighting"

        # 焦点元素
        for elem in segment.contained_elements[:2]:  # 取前两个元素作为焦点
            if elem.element_type == ElementType.DIALOGUE:
                segment.focus_elements.append("speaker_facial_expression")
            elif elem.element_type == ElementType.ACTION:
                segment.focus_elements.append("key_action_movement")

    def _merge_segment(self, target_segment: TimeSegment, source_segment: TimeSegment,
                       source_start_time: float) -> None:
        """合并两个片段"""
        # 调整源片段元素的开始时间偏移
        time_diff = target_segment.end_time - source_start_time

        for elem in source_segment.contained_elements:
            elem.start_offset += time_diff
            target_segment.contained_elements.append(elem)

        # 更新视觉摘要
        if source_segment.visual_summary:
            if target_segment.visual_summary:
                target_segment.visual_summary += " + " + source_segment.visual_summary
            else:
                target_segment.visual_summary = source_segment.visual_summary

        # 更新结束时间和约束
        target_segment.time_range = (target_segment.start_time,
                                     target_segment.end_time + source_segment.duration)
        target_segment.duration = target_segment.end_time - target_segment.start_time

        # 合并约束
        if source_segment.end_anchor:
            target_segment.end_anchor.update(source_segment.end_anchor)

    def _calculate_statistics(self, segments: List[TimeSegment],
                              estimations: Dict[str, DurationEstimation],
                              element_order: List[str]) -> Dict[str, Any]:
        """计算分片统计信息"""
        total_duration = sum(seg.duration for seg in segments)
        total_elements = sum(len(seg.contained_elements) for seg in segments)

        # 计算切割比例
        partial_elements = 0
        for seg in segments:
            for elem in seg.contained_elements:
                if elem.is_partial:
                    partial_elements += 1

        partial_ratio = partial_elements / total_elements if total_elements > 0 else 0

        # 计算时长分布
        durations = [seg.duration for seg in segments]
        avg_duration = sum(durations) / len(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0

        # 计算元素类型分布
        type_distribution = {}
        for seg in segments:
            for elem in seg.contained_elements:
                elem_type = elem.element_type.value
                type_distribution[elem_type] = type_distribution.get(elem_type, 0) + 1

        return {
            "total_segments": len(segments),
            "total_duration": total_duration,
            "total_elements": total_elements,
            "partial_elements": partial_elements,
            "partial_ratio": partial_ratio,
            "duration_stats": {
                "average": avg_duration,
                "min": min_duration,
                "max": max_duration,
                "deviation": max_duration - min_duration
            },
            "element_type_distribution": type_distribution,
            "segments_per_minute": len(segments) / (total_duration / 60) if total_duration > 0 else 0
        }

    def _evaluate_segments_quality(self, segments: List[TimeSegment],
                                   split_decisions: List[SplitDecision]) -> Dict[str, float]:
        """评估片段质量"""
        if not segments:
            return {"overall": 0.0, "pacing_consistency": 0.0, "continuity": 0.0}

        # 1. 时长一致性评分
        durations = [seg.duration for seg in segments]
        avg_duration = sum(durations) / len(durations)

        duration_deviations = [abs(d - self.config.target_segment_duration) for d in durations]
        avg_deviation = sum(duration_deviations) / len(duration_deviations)

        # 标准化到0-1（0最好，1最差）
        duration_score = 1.0 - min(avg_deviation / self.config.target_segment_duration, 1.0)

        # 2. 切割质量评分
        if split_decisions:
            split_scores = [dec.quality_score for dec in split_decisions]
            avg_split_score = sum(split_scores) / len(split_scores)
        else:
            avg_split_score = 1.0  # 没有切割就是最好的

        # 3. 节奏一致性评分
        pacing_scores = [seg.pacing_score for seg in segments]
        pacing_variance = sum((score - 5.0) ** 2 for score in pacing_scores) / len(pacing_scores)
        pacing_consistency = 1.0 - min(pacing_variance / 25.0, 1.0)  # 5.0是基准分

        # 4. 视觉连贯性评分
        continuity_score = 1.0
        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]

            # 检查是否有不自然的切割
            if any(elem.is_partial and elem.partial_type == "end" for elem in current.contained_elements):
                if not any(elem.is_partial and elem.partial_type == "start" for elem in next_seg.contained_elements):
                    continuity_score *= 0.9  # 轻微扣分

        # 综合评分
        overall_score = (
                duration_score * self.config.weights["duration_deviation"] +
                avg_split_score * self.config.weights["split_quality"] +
                pacing_consistency * self.config.weights["pacing_consistency"] +
                continuity_score * self.config.weights["visual_coherence"]
        )

        return {
            "overall": overall_score,
            "pacing_consistency": pacing_consistency,
            "continuity": continuity_score
        }

    def _calculate_segment_pacing_score(self, segment: TimeSegment) -> float:
        """计算片段节奏得分（0-10）"""
        score = 5.0  # 基础分

        # 基于元素类型调整
        for elem in segment.contained_elements:
            if elem.element_type == ElementType.ACTION:
                score += 1.0
            elif elem.element_type == ElementType.DIALOGUE:
                score += 0.5
            elif elem.element_type == ElementType.SILENCE:
                score += 1.5  # 沉默增加节奏张力

        # 基于时长调整
        duration_ratio = segment.duration / self.config.target_segment_duration
        if duration_ratio < 0.8:
            score += 1.0  # 较短片段节奏更快
        elif duration_ratio > 1.2:
            score -= 1.0  # 较长片段节奏更慢

        return min(max(score, 0.0), 10.0)

    def _calculate_segment_completeness_score(self, segment: TimeSegment) -> float:
        """计算片段完整性得分（0-1）"""
        total_elements = len(segment.contained_elements)
        if total_elements == 0:
            return 0.0

        complete_elements = sum(1 for elem in segment.contained_elements if not elem.is_partial)

        # 如果有部分元素，检查是否是自然的开始/结束
        if any(elem.is_partial for elem in segment.contained_elements):
            # 检查片段是否以完整元素开始和结束
            first_elem = segment.contained_elements[0]
            last_elem = segment.contained_elements[-1]

            starts_with_complete = not first_elem.is_partial or first_elem.partial_type == "start"
            ends_with_complete = not last_elem.is_partial or last_elem.partial_type == "end"

            if starts_with_complete and ends_with_complete:
                return 0.8  # 自然的部分
            else:
                return 0.5  # 不自然的部分
        else:
            return 1.0  # 所有元素都完整

    def _calculate_segment_split_quality(self, segment: TimeSegment) -> float:
        """计算片段切割质量得分（0-1）"""
        if not any(elem.is_partial for elem in segment.contained_elements):
            return 1.0  # 没有切割就是最好的

        # 检查切割的合理性
        score = 1.0

        for elem in segment.contained_elements:
            if elem.is_partial:
                if elem.element_type == ElementType.DIALOGUE:
                    score *= 0.7  # 对话切割惩罚
                elif elem.element_type == ElementType.ACTION:
                    score *= 0.9  # 动作切割轻微惩罚

                # 检查是否是自然边界
                if elem.partial_type in ["start", "end"]:
                    score *= 1.1  # 在边界切割较好

        return min(max(score, 0.0), 1.0)

    def _determine_segment_type(self, segment: TimeSegment) -> str:
        """确定片段类型"""
        element_types = [elem.element_type for elem in segment.contained_elements]

        # 检查是否是情感高潮
        emotional_elements = [e for e in segment.contained_elements
                              if getattr(e.element_data, 'emotional_weight', 1.0) > 1.5]
        if len(emotional_elements) >= 2:
            return "climax"

        # 检查是否是转场
        if any(elem.is_partial and elem.partial_type == "end" for elem in segment.contained_elements):
            if any(elem.is_partial and elem.partial_type == "start" for elem in segment.contained_elements):
                return "transition"

        # 检查是否是场景建立
        if any(elem.element_type == ElementType.SCENE for elem in segment.contained_elements):
            scene_elements = [e for e in segment.contained_elements if e.element_type == ElementType.SCENE]
            if len(scene_elements) >= 2:
                return "setup"

        # 默认类型
        return "normal"

    def _get_split_reason(self, element: DurationEstimation, split_point: float) -> str:
        """获取切割原因描述"""
        reasons = []

        if element.split_priority <= 3:
            reasons.append("高优先级元素")

        if split_point < 0.3:
            reasons.append("在前部切割")
        elif split_point > 0.7:
            reasons.append("在后部切割")
        else:
            reasons.append("在中部切割")

        if element.element_type == ElementType.DIALOGUE:
            reasons.append("对话元素")
        elif element.element_type == ElementType.ACTION:
            reasons.append("动作元素")

        return " | ".join(reasons)

    def _string_to_element_type(self, type_str: str) -> ElementType:
        """字符串转换为 ElementType 枚举"""
        type_map = {
            "scene": ElementType.SCENE,
            "dialogue": ElementType.DIALOGUE,
            "action": ElementType.ACTION,
            "silence": ElementType.SILENCE,
            "transition": ElementType.TRANSITION
        }
        return type_map.get(type_str.lower(), ElementType.UNKNOWN)

    def _calculate_emotional_weight(self, element_data: Dict) -> float:
        """根据元素数据计算情感权重"""
        weight = 1.0

        if element_data.get("type") == "dialogue":
            emotion = element_data.get("emotion", "")
            if "微颤" in emotion or "哽咽" in emotion:
                weight = 1.8
            elif "紧张" in emotion or "激动" in emotion:
                weight = 1.5

            content = element_data.get("content", "")
            if "？" in content or "！" in content or "……" in content:
                weight *= 1.2

        elif element_data.get("type") == "scene":
            mood = element_data.get("mood", "")
            if "紧张" in mood or "压抑" in mood or "孤独" in mood:
                weight = 1.3

        return weight

    def _calculate_visual_complexity(self, element_data: Dict) -> float:
        """计算视觉复杂度"""
        complexity = 1.0

        if element_data.get("type") == "scene":
            key_visuals = element_data.get("key_visuals", [])
            complexity = 1.0 + len(key_visuals) * 0.2

        elif element_data.get("type") == "action":
            description = element_data.get("description", "")
            # 根据描述长度估算复杂度
            word_count = len(description.split())
            complexity = 1.0 + word_count * 0.05

        return min(complexity, 3.0)

    def _determine_splittable(self, element_type: ElementType, element_data: Dict) -> bool:
        """确定元素是否可切割"""
        # 对话和沉默通常不切割
        if element_type in [ElementType.DIALOGUE, ElementType.SILENCE]:
            return False

        # 检查是否有特殊标记
        if element_data.get("type") == "action":
            description = element_data.get("description", "")
            # 一些特殊动作可能不应该切割
            if "瞬间" in description or "突然" in description:
                return False

        return True

    def _get_split_priority(self, element_type: ElementType) -> int:
        """获取切割优先级"""
        priority_map = {
            ElementType.SILENCE: 1,
            ElementType.DIALOGUE: 2,
            ElementType.TRANSITION: 3,
            ElementType.ACTION: 4,
            ElementType.SCENE: 5,
            ElementType.UNKNOWN: 6
        }
        return priority_map.get(element_type, 5)

segment_splitter = FiveSecondSplitter()