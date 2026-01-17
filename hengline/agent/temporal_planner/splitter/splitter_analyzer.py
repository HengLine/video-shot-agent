"""
@FileName: splitter_analyzer.py
@Description: 节奏分析器
@Author: HengLine
@Time: 2026/1/16 23:50
"""

from typing import List, Dict, Any

from hengline.agent.temporal_planner.temporal_planner_model import ElementType, PacingProfile, ScriptElement


class RhythmAnalyzer:
    """分析剧本的节奏模式"""

    def __init__(self):
        self.intensity_weights = {
            ElementType.ACTION: 1.8,
            ElementType.DIALOGUE: 1.2,
            ElementType.SCENE: 0.5
        }

        self.emotion_weights = {
            "紧张": 2.0,
            "激动": 1.8,
            "愤怒": 1.7,
            "悲伤": 1.3,
            "平静": 0.7,
            "沉思": 0.6,
            "微颤": 1.4,
            "哽咽": 1.5,
            "低声": 0.8,
            "沙哑": 0.9,
            "沉重": 1.2
        }

        self.action_type_weights = {
            "physiological": 1.3,  # 生理反应
            "facial": 1.4,  # 面部表情
            "gesture": 1.2,  # 手势
            "posture": 1.1,  # 姿势
            "interaction": 1.5,  # 交互
            "gaze": 1.0,  # 注视
            "device_alert": 1.6,  # 设备提醒
            "prop_fall": 1.7  # 道具掉落
        }

    def analyze_pacing(self, elements: List[ScriptElement]) -> PacingProfile:
        """分析整体节奏"""

        # 1. 计算每个元素的强度
        element_intensities = []
        for element in elements:
            intensity = self._calculate_element_intensity(element)
            element_intensities.append(intensity)

        # 2. 识别节奏模式
        pace_type = self._identify_pace_type(element_intensities)

        # 3. 检测峰值和谷值
        peaks = self._find_intensity_peaks(element_intensities)
        valleys = self._find_intensity_valleys(element_intensities)

        # 4. 计算统计指标
        stats = self._compute_statistics(elements, element_intensities)

        # 5. 生成建议
        suggestions = self._generate_pacing_suggestions(
            pace_type, peaks, valleys, stats, len(elements)
        )

        return PacingProfile(
            pace_type=pace_type,
            intensity_curve=element_intensities,
            peak_segments=peaks,
            rest_points=valleys,
            avg_dialogue_density=stats["dialogue_density"],
            action_intensity=stats["action_intensity"],
            scene_stability=stats["scene_stability"],
            pacing_suggestions=suggestions
        )

    def _calculate_element_intensity(self, element: ScriptElement) -> float:
        """计算单个元素的节奏强度"""
        base_intensity = self.intensity_weights.get(element.element_type, 1.0)

        # 考虑元素类型特定的强度因素
        if element.element_type == ElementType.DIALOGUE:
            dialogue = element.original_data
            emotion_weight = self.emotion_weights.get(dialogue.emotion, 1.0)

            # 基于内容长度和情感强度
            content_words = len(dialogue.content.split())
            content_density = min(content_words / 10.0, 2.0)  # 每10词为基准

            # 考虑沉默对话的特殊处理
            if dialogue.type == "silence":
                # 沉默时刻通常有情感重量
                silence_weight = 1.2 if dialogue.emotion in ["哽咽", "沉重"] else 0.8
                return base_intensity * emotion_weight * silence_weight

            return base_intensity * emotion_weight * content_density

        elif element.element_type == ElementType.ACTION:
            action = element.original_data
            type_weight = self.action_type_weights.get(action.type, 1.0)

            # 动作描述的复杂度
            desc_words = len(action.description.split())
            complexity = min(desc_words / 5.0, 1.5)  # 每5词为基准

            # 交互动作额外权重
            interaction_bonus = 0.2 if action.target else 0.0

            return base_intensity * type_weight * complexity + interaction_bonus

        elif element.element_type == ElementType.SCENE:
            scene = element.original_data
            # 场景强度基于情绪和视觉复杂度
            mood_weight = 1.0
            if hasattr(scene, 'mood'):
                if "紧张" in scene.mood or "压抑" in scene.mood:
                    mood_weight = 1.3
                elif "孤独" in scene.mood or "悲伤" in scene.mood:
                    mood_weight = 1.1
                elif "平静" in scene.mood:
                    mood_weight = 0.8

            # 视觉元素数量影响
            visual_complexity = len(scene.key_visuals) / 3.0 if hasattr(scene, 'key_visuals') else 1.0

            return base_intensity * mood_weight * min(visual_complexity, 2.0)

        return base_intensity

    def _identify_pace_type(self, intensities: List[float]) -> str:
        """识别节奏类型"""
        if len(intensities) < 3:
            return "neutral"

        # 计算统计指标
        variance = self._calculate_variance(intensities)
        trend = self._calculate_trend(intensities)
        avg_intensity = sum(intensities) / len(intensities)
        max_intensity = max(intensities) if intensities else 0

        # 基于规则的节奏类型识别
        if variance > 0.6 and trend > 0.4:
            return "rising_tension"  # 上升紧张型
        elif variance > 0.7 and trend < -0.3:
            return "falling_resolution"  # 下降解决型
        elif variance < 0.3 and avg_intensity < 1.0:
            return "calm_reflective"  # 平静反思型
        elif variance > 0.5 and max_intensity > 2.0 and len([i for i in intensities if i > 1.5]) > len(intensities) * 0.4:
            return "rapid_exchange"  # 快速交替型
        elif variance < 0.4 and 1.0 <= avg_intensity <= 1.8:
            return "moderate_narrative"  # 中等叙事型
        elif variance > 0.8:
            return "highly_variable"  # 高度变化型
        else:
            return "balanced"  # 平衡型

    def _calculate_variance(self, intensities: List[float]) -> float:
        """计算强度序列的方差（标准化）"""
        if len(intensities) < 2:
            return 0.0

        mean = sum(intensities) / len(intensities)

        # 计算方差
        variance = sum((x - mean) ** 2 for x in intensities) / len(intensities)

        # 标准化到0-1范围（基于经验阈值）
        # 在剧本强度分析中，方差大于1.0通常表示高度变化
        normalized_variance = min(variance / 1.0, 1.0)

        return normalized_variance

    def _calculate_trend(self, intensities: List[float]) -> float:
        """计算强度序列的趋势（-1到1）"""
        if len(intensities) < 3:
            return 0.0

        # 使用简单的线性回归计算趋势
        n = len(intensities)
        x = list(range(n))

        # 计算回归系数
        sum_x = sum(x)
        sum_y = sum(intensities)
        sum_xy = sum(x[i] * intensities[i] for i in range(n))
        sum_x2 = sum(x_i ** 2 for x_i in x)

        # 避免除以零
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # 标准化趋势值：考虑到通常的趋势强度
        # 斜率大于0.1表示明显上升趋势，小于-0.1表示明显下降趋势
        normalized_trend = max(min(slope / 0.1, 1.0), -1.0)

        return normalized_trend

    def _find_intensity_peaks(self, intensities: List[float]) -> List[int]:
        """找到强度峰值点"""
        peaks = []

        if len(intensities) < 3:
            return peaks

        # 使用滑动窗口检测局部峰值
        for i in range(1, len(intensities) - 1):
            # 当前点高于相邻两点
            if intensities[i] > intensities[i - 1] and intensities[i] > intensities[i + 1]:
                # 峰值必须显著高于平均水平
                if intensities[i] > 1.5:  # 显著峰值阈值
                    peaks.append(i)

        # 处理边界情况：第一个和最后一个元素可能是峰值
        if len(intensities) >= 2:
            if intensities[0] > intensities[1] and intensities[0] > 1.5:
                peaks.insert(0, 0)
            if intensities[-1] > intensities[-2] and intensities[-1] > 1.5:
                peaks.append(len(intensities) - 1)

        # 去重并排序
        peaks = sorted(set(peaks))

        return peaks

    def _find_intensity_valleys(self, intensities: List[float]) -> List[int]:
        """找到强度谷值点（休息点）"""
        valleys = []

        if len(intensities) < 3:
            return valleys

        # 计算平均强度
        avg_intensity = sum(intensities) / len(intensities)

        # 检测局部谷值
        for i in range(1, len(intensities) - 1):
            # 当前点低于相邻两点
            if intensities[i] < intensities[i - 1] and intensities[i] < intensities[i + 1]:
                # 谷值应该相对较低，作为潜在的休息点
                if intensities[i] < avg_intensity * 0.7:  # 显著低于平均值
                    valleys.append(i)

        # 处理边界情况
        if len(intensities) >= 2:
            if intensities[0] < intensities[1] and intensities[0] < avg_intensity * 0.7:
                valleys.insert(0, 0)
            if intensities[-1] < intensities[-2] and intensities[-1] < avg_intensity * 0.7:
                valleys.append(len(intensities) - 1)

        # 如果没有找到明显的谷值，选择相对较低的点
        if not valleys and len(intensities) > 0:
            # 找到最低的几个点
            sorted_indices = sorted(range(len(intensities)), key=lambda i: intensities[i])
            valleys = sorted_indices[:min(3, len(intensities) // 3)]  # 取最低的1/3

        # 去重并排序
        valleys = sorted(set(valleys))

        return valleys

    def _compute_statistics(self, elements: List[ScriptElement], intensities: List[float]) -> Dict[str, float]:
        """计算节奏统计指标"""
        total_elements = len(elements)

        if total_elements == 0:
            return {
                "dialogue_density": 0.0,
                "action_intensity": 0.0,
                "scene_stability": 0.0,
                "intensity_variance": 0.0,
                "intensity_range": 0.0
            }

        # 元素类型统计
        dialogue_count = sum(1 for e in elements if e.element_type == ElementType.DIALOGUE)
        action_count = sum(1 for e in elements if e.element_type == ElementType.ACTION)
        scene_count = sum(1 for e in elements if e.element_type == ElementType.SCENE)

        # 对话密度（考虑对话时长权重）
        dialogue_duration = sum(
            e.estimated_duration.estimated_duration
            for e in elements if e.element_type == ElementType.DIALOGUE
        )
        total_duration = sum(e.estimated_duration.estimated_duration for e in elements)

        dialogue_density = dialogue_duration / total_duration if total_duration > 0 else 0

        # 动作强度（加权）
        action_intensity_sum = sum(
            intensities[i] for i, e in enumerate(elements)
            if e.element_type == ElementType.ACTION
        )
        action_intensity = action_intensity_sum / action_count if action_count > 0 else 0

        # 场景稳定性（场景间强度变化）
        scene_indices = [i for i, e in enumerate(elements) if e.element_type == ElementType.SCENE]
        scene_stability = 0.0

        if len(scene_indices) > 1:
            scene_intensities = [intensities[i] for i in scene_indices]
            # 场景间强度变化越小，稳定性越高
            intensity_changes = [
                abs(scene_intensities[i] - scene_intensities[i - 1])
                for i in range(1, len(scene_intensities))
            ]
            avg_change = sum(intensity_changes) / len(intensity_changes) if intensity_changes else 0
            # 标准化稳定性：变化越小，稳定性越高（0-1）
            scene_stability = max(0, 1 - min(avg_change, 1))

        # 强度统计
        intensity_variance = self._calculate_variance(intensities)
        intensity_range = max(intensities) - min(intensities) if intensities else 0

        return {
            "dialogue_density": dialogue_density,
            "action_intensity": action_intensity,
            "scene_stability": scene_stability,
            "intensity_variance": intensity_variance,
            "intensity_range": intensity_range
        }

    def _generate_pacing_suggestions(
            self,
            pace_type: str,
            peaks: List[int],
            valleys: List[int],
            stats: Dict[str, float],
            total_elements: int
    ) -> List[str]:
        """生成节奏建议"""
        suggestions = []

        # 基于节奏类型的建议
        if pace_type == "rising_tension":
            suggestions.append("节奏呈上升紧张趋势，建议保持紧张感的累积和释放")
            if peaks:
                suggestions.append(f"主要紧张峰值出现在元素索引：{peaks}")

        elif pace_type == "falling_resolution":
            suggestions.append("节奏呈下降解决趋势，适合情感收尾或问题解决")

        elif pace_type == "calm_reflective":
            suggestions.append("节奏平静反思，适合情感深度表达和内心戏")
            if stats["intensity_variance"] < 0.2:
                suggestions.append("节奏变化较小，可考虑添加一些情绪波动")

        elif pace_type == "rapid_exchange":
            suggestions.append("节奏快速交替，适合对话交锋或紧张情节")
            if stats["dialogue_density"] > 0.7:
                suggestions.append("对话密度很高，可适当插入动作或反应镜头")

        elif pace_type == "highly_variable":
            suggestions.append("节奏变化较大，注意情感过渡的自然性")

        # 基于统计指标的建议
        if stats["dialogue_density"] > 0.8:
            suggestions.append("对话占比过高（>80%），建议增加视觉动作元素")
        elif stats["dialogue_density"] < 0.2:
            suggestions.append("对话占比较低（<20%），可能需要更多对话推进剧情")

        if stats["action_intensity"] > 2.0:
            suggestions.append("动作强度较高，注意动作节奏的呼吸感")

        if stats["scene_stability"] < 0.3:
            suggestions.append("场景间情绪跳跃较大，可能需要更多过渡")

        # 基于峰值/谷值分布的建议
        if peaks:
            peak_spacing = self._analyze_peak_spacing(peaks, total_elements)
            if peak_spacing["min_spacing"] < 3:
                suggestions.append("紧张峰值过于密集，考虑分散以增强冲击力")
            if peak_spacing["avg_spacing"] > 10:
                suggestions.append("紧张峰值间隔较大，可考虑增加中间张力")

        if valleys:
            if len(valleys) < 2 and total_elements > 10:
                suggestions.append("休息点较少，考虑在紧张场景后添加情感缓冲")

        # 节奏平衡建议
        if len(peaks) == 0 and total_elements > 8:
            suggestions.append("缺乏明显的情感峰值，可考虑增加高潮点")

        return suggestions

    def _analyze_peak_spacing(self, peaks: List[int], total_elements: int) -> Dict[str, Any]:
        """分析峰值间距"""
        if len(peaks) < 2:
            return {"min_spacing": total_elements, "avg_spacing": total_elements}

        spacings = []
        for i in range(1, len(peaks)):
            spacing = peaks[i] - peaks[i - 1]
            spacings.append(spacing)

        return {
            "min_spacing": min(spacings) if spacings else 0,
            "avg_spacing": sum(spacings) / len(spacings) if spacings else 0,
            "max_spacing": max(spacings) if spacings else 0
        }

    def calculate_segment_intensity(self, elements_in_segment: List[ScriptElement]) -> float:
        """计算片段的综合强度（用于分片器）"""
        if not elements_in_segment:
            return 0.5  # 默认中等强度

        intensities = [self._calculate_element_intensity(e) for e in elements_in_segment]

        # 使用加权平均，考虑元素时长
        total_duration = sum(
            e.estimated_duration.estimated_duration
            for e in elements_in_segment
        )

        if total_duration == 0:
            return sum(intensities) / len(intensities)

        weighted_sum = sum(
            intensities[i] * elements_in_segment[i].estimated_duration.estimated_duration
            for i in range(len(elements_in_segment))
        )

        return weighted_sum / total_duration

    def get_pacing_adjustment_factor(self, segment_intensity: float, overall_pace_type: str) -> float:
        """根据整体节奏类型调整片段强度"""
        # 返回调整因子，用于分片时考虑节奏需求
        if overall_pace_type == "rising_tension":
            # 上升趋势中，后期片段应保持或增加强度
            return 1.0 + (segment_intensity * 0.1)
        elif overall_pace_type == "calm_reflective":
            # 平静节奏中，避免过强的片段
            return max(0.7, min(1.0, 1.5 - segment_intensity * 0.3))
        elif overall_pace_type == "rapid_exchange":
            # 快速交替中，强度变化可以更大
            return 1.2 if segment_intensity > 1.5 else 0.9
        else:
            return 1.0
