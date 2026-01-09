"""
@FileName: time_analyzer.py
@Description: 
@Author: HengLine
@Time: 2026/1/9 18:39
"""
from typing import Dict, Any, List

from hengline.agent.temporal_planner.temporal_planner_model import TimelinePlan, TimeSegment, DurationEstimation


class TimelineAnalyzer:
    """时序分析器"""

    def analyze_timeline_structure(self, timeline_plan: TimelinePlan) -> Dict[str, Any]:
        """分析时序结构"""
        analysis = {
            "segment_distribution": self._analyze_segment_distribution(timeline_plan.timeline_segments),
            "duration_analysis": self._analyze_durations(timeline_plan.duration_estimations),
            "complexity_profile": self._analyze_complexity_profile(timeline_plan.timeline_segments),
            "continuity_patterns": self._identify_continuity_patterns(timeline_plan)
        }

        return analysis

    def _analyze_segment_distribution(self, segments: List[TimeSegment]) -> Dict[str, Any]:
        """分析片段分布"""
        if not segments:
            return {"total_segments": 0, "time_span": 0}

        # 计算时间跨度
        start_times = [s.start_time for s in segments]
        end_times = [s.end_time for s in segments]

        time_span = max(end_times) - min(start_times)

        # 计算平均片段时长
        segment_durations = [s.end_time - s.start_time for s in segments]
        avg_duration = sum(segment_durations) / len(segment_durations)

        return {
            "total_segments": len(segments),
            "time_span": time_span,
            "avg_segment_duration": avg_duration,
            "min_segment_duration": min(segment_durations),
            "max_segment_duration": max(segment_durations)
        }

    def _analyze_durations(self, duration_estimations: Dict[str, DurationEstimation]) -> Dict[str, Any]:
        """分析时长估算"""
        if not duration_estimations:
            return {"total_duration": 0, "confidence_score": 0}

        total_duration = sum(de.estimated_duration for de in duration_estimations.values())
        avg_confidence = sum(de.confidence for de in duration_estimations.values()) / len(duration_estimations)

        # 按类型分组
        type_durations = {}
        for de in duration_estimations.values():
            type_ = de.element_type
            if type_ not in type_durations:
                type_durations[type_] = 0
            type_durations[type_] += de.estimated_duration

        return {
            "total_duration": total_duration,
            "confidence_score": avg_confidence,
            "type_distribution": type_durations,
            "element_count": len(duration_estimations)
        }

    def _analyze_complexity_profile(self, segments: List[TimeSegment]) -> Dict[str, Any]:
        """分析复杂度分布"""
        complexity_counts = {"low": 0, "medium": 0, "high": 0}

        for segment in segments:
            complexity = segment.complexity_level.lower()
            if complexity in complexity_counts:
                complexity_counts[complexity] += 1

        total_segments = len(segments)
        complexity_percentages = {
            level: count / total_segments * 100
            for level, count in complexity_counts.items()
        }

        return {
            "counts": complexity_counts,
            "percentages": complexity_percentages,
            "dominant_complexity": max(complexity_counts, key=complexity_counts.get)
        }

    def _identify_continuity_patterns(self, timeline_plan: TimelinePlan) -> List[Dict[str, Any]]:
        """识别连续性模式"""
        patterns = []

        # 检查时间连续性模式
        segments = timeline_plan.timeline_segments
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]

            # 检查是否有明显的时间跳跃
            time_gap = curr.start_time - prev.end_time
            if abs(time_gap) > 5.0:
                patterns.append({
                    "pattern_type": "time_jump",
                    "position": i,
                    "gap_size": time_gap,
                    "description": f"片段 {prev.segment_id} 到 {curr.segment_id} 有时间跳跃"
                })

        # 检查复杂度变化模式
        complexity_changes = []
        for i in range(1, len(segments)):
            prev_complexity = segments[i - 1].complexity_level
            curr_complexity = segments[i].complexity_level

            if prev_complexity != curr_complexity:
                complexity_changes.append({
                    "from": prev_complexity,
                    "to": curr_complexity,
                    "position": i
                })

        if complexity_changes:
            patterns.append({
                "pattern_type": "complexity_transitions",
                "changes": complexity_changes,
                "description": f"检测到 {len(complexity_changes)} 个复杂度变化"
            })

        return patterns
