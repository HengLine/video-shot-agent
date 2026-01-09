"""
@FileName: anchor_manager.py
@Description: 
@Author: HengLine
@Time: 2026/1/9 18:41
"""
from typing import Any, Dict, List

from hengline.agent.temporal_planner.temporal_planner_model import ContinuityAnchor


class AnchorManager:
    """锚点管理器"""

    def analyze_continuity_anchors(self, anchors: List[ContinuityAnchor]) -> Dict[str, Any]:
        """分析连续性锚点"""
        analysis = {
            "anchor_count": len(anchors),
            "anchor_distribution": self._analyze_anchor_distribution(anchors),
            "importance_analysis": self._analyze_anchor_importance(anchors),
            "type_analysis": self._analyze_anchor_types(anchors),
            "coverage_assessment": self._assess_anchor_coverage(anchors)
        }

        return analysis

    def _analyze_anchor_distribution(self, anchors: List[ContinuityAnchor]) -> Dict[str, Any]:
        """分析锚点分布"""
        if not anchors:
            return {"time_range": 0, "avg_spacing": 0}

        timestamps = [a.timestamp for a in anchors]
        time_range = max(timestamps) - min(timestamps)

        # 计算锚点间距
        spacings = []
        sorted_timestamps = sorted(timestamps)
        for i in range(1, len(sorted_timestamps)):
            spacing = sorted_timestamps[i] - sorted_timestamps[i - 1]
            spacings.append(spacing)

        avg_spacing = sum(spacings) / len(spacings) if spacings else 0

        return {
            "time_range": time_range,
            "avg_spacing": avg_spacing,
            "min_spacing": min(spacings) if spacings else 0,
            "max_spacing": max(spacings) if spacings else 0
        }

    def _analyze_anchor_importance(self, anchors: List[ContinuityAnchor]) -> Dict[str, Any]:
        """分析锚点重要性"""
        importances = [a.importance for a in anchors]

        return {
            "avg_importance": sum(importances) / len(importances) if importances else 0,
            "min_importance": min(importances) if importances else 0,
            "max_importance": max(importances) if importances else 0,
            "high_importance_count": len([i for i in importances if i > 0.7]),
            "medium_importance_count": len([i for i in importances if 0.4 <= i <= 0.7]),
            "low_importance_count": len([i for i in importances if i < 0.4])
        }

    def _analyze_anchor_types(self, anchors: List[ContinuityAnchor]) -> Dict[str, Any]:
        """分析锚点类型"""
        type_counts = {}

        for anchor in anchors:
            type_ = anchor.anchor_type
            if type_ not in type_counts:
                type_counts[type_] = 0
            type_counts[type_] += 1

        return {
            "type_distribution": type_counts,
            "most_common_type": max(type_counts, key=type_counts.get) if type_counts else None
        }

    def _assess_anchor_coverage(self, anchors: List[ContinuityAnchor]) -> Dict[str, Any]:
        """评估锚点覆盖度"""
        if not anchors:
            return {"coverage_score": 0, "gap_assessment": "poor"}

        timestamps = sorted([a.timestamp for a in anchors])
        time_range = timestamps[-1] - timestamps[0]

        # 计算覆盖率
        if len(timestamps) < 2:
            return {"coverage_score": 0.3, "gap_assessment": "sparse"}

        avg_spacing = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)

        # 根据平均间距评估覆盖度
        if avg_spacing > 30:
            coverage_score = 0.3
            gap_assessment = "sparse"
        elif avg_spacing > 15:
            coverage_score = 0.5
            gap_assessment = "moderate"
        elif avg_spacing > 5:
            coverage_score = 0.7
            gap_assessment = "good"
        else:
            coverage_score = 0.9
            gap_assessment = "excellent"

        return {
            "coverage_score": coverage_score,
            "gap_assessment": gap_assessment,
            "suggested_additional_anchors": self._suggest_additional_anchors(timestamps, time_range)
        }

    def _suggest_additional_anchors(self, timestamps: List[float],
                                    time_range: float) -> List[Dict[str, Any]]:
        """建议额外锚点"""
        suggestions = []

        if len(timestamps) < 2:
            return suggestions

        # 找到最大的时间间隔
        max_gap = 0
        max_gap_start = 0

        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            if gap > max_gap:
                max_gap = gap
                max_gap_start = timestamps[i - 1]

        # 如果最大间隔太大，建议添加锚点
        if max_gap > time_range * 0.3:  # 超过总时长的30%
            suggestions.append({
                "position": max_gap_start + max_gap / 2,
                "reason": f"填补大间隔 ({max_gap:.1f}秒)",
                "priority": "high"
            })

        return suggestions
