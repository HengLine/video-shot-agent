"""
@FileName: pacing_analyzer.py
@Description: 节奏分析模型
@Author: HengLine
@Time: 2026/1/16 0:53
"""
from typing import Dict, List, Any

from hengline.agent.temporal_planner.temporal_planner_model import TimeSegment, DurationEstimation, ElementType, PacingProfile


class PacingAnalyzer:
    """分析整体节奏，为后续智能体提供节奏指导"""

    def analyze(self, segments: List[TimeSegment],
                estimations: Dict[str, DurationEstimation]) -> PacingProfile:
        """分析节奏特征"""

        intensities = []
        emotion_points = []

        for segment in segments:
            # 计算片段强度
            intensity = self._calculate_segment_intensity(segment, estimations)
            intensities.append(intensity)

            # 情感高点检测
            emotion_level = self._detect_emotion_level(segment, estimations)
            if emotion_level > 1.5:
                emotion_points.append({
                    "segment": segment.segment_id,
                    "time": segment.time_range[0],
                    "emotion_level": emotion_level
                })

        # 节奏类型识别
        pace_type = self._classify_pacing(intensities)

        # 关键转折点
        turning_points = self._find_turning_points(intensities, segments)

          {
            "pace_type": pace_type,
            "intensity_curve": intensities,
            "emotion_peaks": emotion_points,
            "turning_points": turning_points,
            "avg_intensity": sum(intensities) / len(intensities) if intensities else 0,
            "intensity_variance": self._calculate_variance(intensities),
            "recommended_pacing": self._generate_pacing_recommendation(pace_type, intensities)
        }

        return PacingProfile(


        )

    def _calculate_segment_intensity(self, segment: TimeSegment,
                                     estimations: Dict[str, DurationEstimation]) -> float:
        """计算片段节奏强度"""
        intensity = 0.0

        for elem_ref in segment.contained_elements:
            elem_id = elem_ref.element_id
            if elem_id in estimations:
                elem = estimations[elem_id]

                # 不同类型贡献不同强度
                if elem.element_type == ElementType.DIALOGUE:
                    intensity += 0.7 * elem.emotional_weight
                elif elem.element_type == ElementType.ACTION:
                    intensity += 1.2 * elem.visual_complexity
                elif elem.element_type == ElementType.SCENE:
                    intensity += 0.4
                elif elem.element_type == ElementType.SILENCE:
                    intensity += 0.9 * elem.emotional_weight  # 沉默也有强度

        # 归一化到0-3范围
        return min(intensity, 3.0)

    def _detect_emotion_level(self, segment: TimeSegment,
                              estimations: Dict[str, DurationEstimation]) -> float:
        """检测情感水平"""
        emotion_score = 0.0

        for elem_ref in segment.contained_elements:
            elem_id = elem_ref.element_id
            if elem_id in estimations:
                elem = estimations[elem_id]
                emotion_score += elem.emotional_weight - 1.0  # 基础1.0不计入

        return emotion_score

    def _classify_pacing(self, intensities: List[float]) -> str:
        """分类节奏类型"""
        if len(intensities) < 3:
            return "平缓"

        # 计算波动性
        variance = self._calculate_variance(intensities)

        if variance > 0.5:
            return "起伏波动型"
        elif max(intensities) > 2.0:
            return "紧张累积型"
        else:
            return "平稳叙述型"

    def _find_turning_points(self, intensities: List[float],
                             segments: List[TimeSegment]) -> List[Dict]:
        """找到节奏转折点"""
        turning_points = []

        for i in range(1, len(intensities) - 1):
            prev, curr, next_ = intensities[i - 1], intensities[i], intensities[i + 1]

            # 峰值点
            if curr > prev and curr > next_ and curr > 1.5:
                turning_points.append({
                    "type": "peak",
                    "segment": segments[i].segment_id,
                    "time": segments[i].time_range[0],
                    "intensity": curr
                })

            # 谷值点（情感低点或休息点）
            elif curr < prev and curr < next_ and curr < 1.0:
                turning_points.append({
                    "type": "valley",
                    "segment": segments[i].segment_id,
                    "time": segments[i].time_range[0],
                    "intensity": curr
                })

        return turning_points

    def _calculate_variance(self, values: List[float]) -> float:
        """计算方差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _generate_pacing_recommendation(self, pace_type: str,
                                        intensities: List[float]) -> str:
        """生成节奏建议"""
        recommendations = {
            "紧张累积型": "建议保持紧张感逐步升级，在峰值片段使用特写和快速剪辑",
            "起伏波动型": "建议利用波动创造节奏感，高潮与平静交替",
            "平稳叙述型": "建议保持平稳节奏，注重细节和情感积累"
        }

        return recommendations.get(pace_type, "建议保持自然叙事节奏")
