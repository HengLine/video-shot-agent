# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: LLM + 规则约束实现的时序规划（负责将剧本按5秒粒度切分，估算动作时长，使用LangChain实现状态记忆）
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from datetime import datetime
from typing import List, Dict, Any

from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.base_temporal_planner import BaseTemporalPlanner
from hengline.agent.temporal_planner.estimator.timeline_planner_factory import TimelinePlannerFactory
from hengline.agent.temporal_planner.splitter.five_second_splitter import segment_splitter
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, TimeSegment, TimelinePlan, PacingAnalysis, ContinuityAnchor, ElementType
from hengline.logger import error, debug, info, warning
from hengline.prompts.temporal_planner_prompt import PromptConfig


class LLMTemporalPlanner(BaseTemporalPlanner):
    """ LLM 时长估算 """

    def __init__(self, llm_client, config: PromptConfig = None):
        """初始化时序规划智能体"""
        self.llm = llm_client
        self.config = config or PromptConfig()
        self.factory = TimelinePlannerFactory
        # 性能跟踪
        self.processing_stats = {
            "start_time": 0,
            "end_time": 0,
            "element_estimation_time": 0,
            "segmentation_time": 0,
            "analysis_time": 0
        }

        # 缓存
        self.element_cache = {}
        self.segment_cache = {}

    def plan_timeline(self, script_data: UnifiedScript) -> TimelinePlan | None:
        """
        规划剧本的时序分段

        Args:
            script_data: 结构化的剧本

        Returns:
            分段计划列表
        """
        self.processing_stats["start_time"] = datetime.now().second
        try:
            # 1. 估算所有元素的时长
            debug("步骤1: 估算元素时长")
            start_est = datetime.now().second
            estimations = self.factory.estimate_script(script_data)
            self.processing_stats["element_estimation_time"] = (datetime.now().second - start_est)

            # 2. 组织估算结果
            debug("步骤2: 组织估算结果")
            organized = self._organize_estimations(estimations, script_data)

            # 3. 生成5秒片段
            debug("步骤3: 生成5秒片段")
            start_seg = datetime.now().second
            segments = segment_splitter.split_into_segments(estimations, organized, script_data)
            self.processing_stats["segmentation_time"] = (datetime.now().second - start_seg)
            debug(f"自适应分片完成: {len(segments)} 个片段")

            # 4. 分析节奏
            debug("步骤4: 分析节奏")
            start_ana = datetime.now().second
            pacing_analysis = self._multidimensional_pacing_analysis(segments, estimations)
            self.processing_stats["analysis_time"] = (datetime.now().second - start_ana)

            # 5. 生成连续性锚点
            debug("步骤5: 生成连续性锚点")
            continuity_anchors = self._generate_continuity_anchors(segments, estimations)
            debug(f"连续性锚点生成: {len(continuity_anchors)} 个锚点")

            # 6. 生成最终计划
            debug("步骤6: 生成最终计划")
            final_plan = self._create_final_plan(
                segments=segments,
                estimations=estimations,
                pacing_analysis=pacing_analysis,
                continuity_anchors=continuity_anchors,
                script_data=script_data
            )

            # 7. 显示错误摘要
            error_summary = self.factory.get_error_summary()
            if error_summary["estimators_with_errors"] > 0:
                warning(f"\n发现错误: {error_summary['estimators_with_errors']}个估算器有错误")

            info(f"时序规划完成！ 片段数: {len(segments)}")

            return final_plan

        except Exception as e:
            error(f"处理失败: {str(e)}")
            raise

    def _multidimensional_pacing_analysis(self, segments: List[TimeSegment],
                                          estimations: Dict[str, DurationEstimation]) -> PacingAnalysis | None:
        """多维度节奏分析"""
        if not segments:
            return None

        # 1. 强度分析
        intensities = []
        for segment in segments:
            intensity = self._calculate_segment_intensity(segment, estimations)
            intensities.append(intensity)

        # 2. 情感分析
        emotional_arc = self._analyze_emotional_arc(segments, estimations)

        # 3. 视觉复杂度分析
        visual_complexity = self._analyze_visual_complexity(segments, estimations)

        # 4. 节奏模式识别
        pacing_profile = self._identify_pacing_pattern(intensities, emotional_arc)

        # 5. 关键点检测
        key_points = self._detect_key_points(segments, intensities, estimations)

        # 6. 节奏建议
        recommendations = self._generate_pacing_recommendations(
            pacing_profile, intensities, emotional_arc
        )

        return PacingAnalysis(
            pacing_profile=pacing_profile,
            intensity_curve=intensities,
            emotional_arc=emotional_arc,
            visual_complexity=visual_complexity,
            key_points=key_points,
            statistics={
                "avg_intensity": round(sum(intensities) / len(intensities), 2) if intensities else 0,
                "max_intensity": round(max(intensities), 2) if intensities else 0,
                "min_intensity": round(min(intensities), 2) if intensities else 0,
                "intensity_variance": round(self._calculate_variance(intensities), 3) if len(intensities) > 1 else 0
            },
            recommendations=recommendations,
            analysis_notes=self._generate_analysis_notes(pacing_profile, key_points)
        )

    def _calculate_segment_intensity(self, segment: TimeSegment,
                                     estimations: Dict[str, DurationEstimation]) -> float:
        """计算片段强度（优化版）"""
        intensity = 0.0

        for elem in segment.contained_elements:
            elem_id = elem.element_id
            if elem_id in estimations:
                est = estimations[elem_id]

                # 基础强度贡献
                base_contributions = {
                    "scene": 0.5,
                    "dialogue": 0.8,
                    "silence": 1.2,
                    "action": 1.0
                }

                base = base_contributions.get(est.element_type.value, 0.5)

                # 情感权重调整
                emotional_factor = est.emotional_weight

                # 视觉复杂度调整
                visual_factor = est.visual_complexity if hasattr(est, 'visual_complexity') else 1.0

                # 时长权重（较长的元素通常更重要）
                duration_weight = min(elem.duration / 3.0, 1.5)

                intensity += base * emotional_factor * visual_factor * duration_weight

        # 归一化到0-3范围，并考虑片段密度
        segment_density = len(segment.contained_elements) / 5.0  # 每5秒的元素数量
        density_factor = min(segment_density, 2.0) / 1.5  # 密度增加强度，但有限制

        return min(intensity * density_factor, 3.0)

    def _analyze_emotional_arc(self, segments: List[TimeSegment],
                               estimations: Dict[str, DurationEstimation]) -> List[Dict]:
        """分析情感弧线"""
        emotional_points = []

        for i, segment in enumerate(segments):
            segment_emotion = {
                "segment_id": segment.segment_id,
                "time": segment.time_range[0],
                "emotional_value": 0.0,
                "dominant_emotion": "neutral",
                "key_elements": []
            }

            # 计算情感值
            emotional_sum = 0
            count = 0

            for elem in segment.contained_elements:
                elem_id = elem.element_id
                if elem_id in estimations:
                    est = estimations[elem_id]
                    emotional_sum += est.emotional_weight
                    count += 1

                    # 记录高情感元素
                    if est.emotional_weight > 1.8:
                        segment_emotion["key_elements"].append({
                            "element_id": elem_id,
                            "type": est.element_type.value,
                            "emotional_weight": est.emotional_weight
                        })

            if count > 0:
                segment_emotion["emotional_value"] = round(emotional_sum / count, 2)

                # 确定主导情感
                if segment_emotion["emotional_value"] > 2.0:
                    segment_emotion["dominant_emotion"] = "intense"
                elif segment_emotion["emotional_value"] > 1.5:
                    segment_emotion["dominant_emotion"] = "emotional"
                elif segment_emotion["emotional_value"] > 1.0:
                    segment_emotion["dominant_emotion"] = "moderate"

            emotional_points.append(segment_emotion)

        return emotional_points

    def _analyze_visual_complexity(self, segments: List[TimeSegment],
                                   estimations: Dict[str, DurationEstimation]) -> List[float]:
        """分析视觉复杂度"""
        complexities = []

        for segment in segments:
            segment_complexity = 0.0
            count = 0

            for elem in segment.contained_elements:
                elem_id = elem.element_id
                if elem_id in estimations:
                    est = estimations[elem_id]
                    if hasattr(est, 'visual_complexity'):
                        segment_complexity += est.visual_complexity
                        count += 1

            avg_complexity = segment_complexity / count if count > 0 else 1.0
            complexities.append(round(avg_complexity, 2))

        return complexities

    def _identify_pacing_pattern(self, intensities: List[float],
                                 emotional_arc: List[Dict]) -> str:
        """识别节奏模式"""
        if len(intensities) < 3:
            return "简短片段"

        # 计算统计特征
        avg_intensity = sum(intensities) / len(intensities)
        variance = self._calculate_variance(intensities)
        max_intensity = max(intensities)
        min_intensity = min(intensities)

        # 情感变化特征
        emotional_changes = []
        for i in range(1, len(emotional_arc)):
            change = abs(emotional_arc[i]["emotional_value"] - emotional_arc[i - 1]["emotional_value"])
            emotional_changes.append(change)

        avg_emotional_change = sum(emotional_changes) / len(emotional_changes) if emotional_changes else 0

        # 模式识别
        if max_intensity > 2.5 and variance > 0.7:
            if avg_emotional_change > 0.4:
                return "强烈起伏情感剧"
            else:
                return "高潮迭起动作剧"
        elif variance > 0.4:
            if avg_emotional_change > 0.3:
                return "情感波动剧情"
            else:
                return "节奏变化剧情"
        elif avg_intensity > 1.8:
            return "持续紧张剧情"
        elif avg_intensity < 1.2:
            return "平缓抒情剧情"
        else:
            return "平衡叙述剧情"

    def _calculate_variance(self, values: List[float]) -> float:
        """计算方差"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return round(variance, 3)

    def _detect_key_points(self, segments: List[TimeSegment], intensities: List[float],
                           estimations: Dict[str, DurationEstimation]) -> List[Dict]:
        """检测关键点"""
        key_points = []

        # 强度峰值
        for i in range(1, len(intensities) - 1):
            if intensities[i] > intensities[i - 1] and intensities[i] > intensities[i + 1]:
                if intensities[i] > 1.8:  # 显著峰值
                    key_points.append({
                        "type": "intensity_peak",
                        "segment_id": segments[i].segment_id,
                        "time": segments[i].time_range[0],
                        "intensity": round(intensities[i], 2),
                        "description": "节奏高潮点"
                    })

        # 情感转折点
        emotional_values = []
        for elem in estimations.values():
            if hasattr(elem, 'emotional_weight'):
                emotional_values.append(elem.emotional_weight)

        if emotional_values:
            avg_emotion = sum(emotional_values) / len(emotional_values)
            for i, segment in enumerate(segments):
                segment_emotion = 0
                count = 0

                for elem in segment.contained_elements:
                    elem_id = elem.element_id
                    if elem_id in estimations:
                        est = estimations[elem_id]
                        segment_emotion += est.emotional_weight
                        count += 1

                if count > 0:
                    avg_segment_emotion = segment_emotion / count
                    if abs(avg_segment_emotion - avg_emotion) > 0.5:  # 显著偏离
                        key_points.append({
                            "type": "emotional_turning_point",
                            "segment_id": segment.segment_id,
                            "time": segment.time_range[0],
                            "emotion_level": round(avg_segment_emotion, 2),
                            "description": "情感转折点"
                        })

        # 按时间排序
        key_points.sort(key=lambda x: x["time"])

        return key_points

    def _generate_pacing_recommendations(self, profile: str, intensities: List[float],
                                         emotional_arc: List[Dict]) -> List[str]:
        """生成节奏建议"""
        recommendations = []

        profile_recommendations = {
            "强烈起伏情感剧": [
                "建议保持强烈的节奏对比，高潮点使用快速剪辑和特写",
                "情感低谷点可以适当延长，给观众情感缓冲时间",
                "注意高潮之间的情绪过渡要自然"
            ],
            "高潮迭起动作剧": [
                "动作高潮点之间要有足够的紧张感积累",
                "复杂动作需要足够的展示时间",
                "动作之间的过渡要流畅"
            ],
            "情感波动剧情": [
                "保持情感变化的自然流畅",
                "情感高点要有足够的表达时间",
                "情感转折点要清晰明确"
            ],
            "节奏变化剧情": [
                "节奏变化要有逻辑依据",
                "不同节奏段落间过渡要平滑",
                "保持整体节奏的协调性"
            ],
            "持续紧张剧情": [
                "持续紧张中要有微妙的变化避免单调",
                "可以通过细节和微表情增加层次",
                "紧张感的释放要有节奏"
            ],
            "平缓抒情剧情": [
                "保持平缓的节奏，注重情感细节",
                "可以通过视觉细节增加丰富度",
                "避免节奏过于拖沓"
            ]
        }

        recommendations.extend(profile_recommendations.get(profile, [
            "保持自然的叙事节奏",
            "注意节奏与情感的匹配"
        ]))

        # 基于强度曲线的具体建议
        if intensities:
            if max(intensities) < 1.5:
                recommendations.append("整体节奏较缓，考虑在某些关键点增加强度")
            elif min(intensities) > 1.8:
                recommendations.append("整体节奏较强，考虑在某些段落放松节奏")

        # 基于情感弧线的建议
        if emotional_arc:
            emotional_changes = []
            for i in range(1, len(emotional_arc)):
                change = abs(emotional_arc[i]["emotional_value"] - emotional_arc[i - 1]["emotional_value"])
                emotional_changes.append(change)

            if emotional_changes:
                avg_change = sum(emotional_changes) / len(emotional_changes)
                if avg_change > 0.4:
                    recommendations.append("情感变化较大，注意情感过渡的自然性")

        return recommendations

    def _generate_analysis_notes(self, profile: str, key_points: List[Dict]) -> str:
        """生成分析说明"""
        notes = []

        notes.append(f"节奏模式识别为'{profile}'")

        if key_points:
            intensity_peaks = [kp for kp in key_points if kp["type"] == "intensity_peak"]
            emotional_turns = [kp for kp in key_points if kp["type"] == "emotional_turning_point"]

            if intensity_peaks:
                notes.append(f"检测到{len(intensity_peaks)}个节奏高潮点")

            if emotional_turns:
                notes.append(f"检测到{len(emotional_turns)}个情感转折点")

        return "；".join(notes)

    def _generate_continuity_anchors(self, segments: List[TimeSegment],
                                     estimations: Dict[str, DurationEstimation]) -> List[ContinuityAnchor]:
        """生成智能连续性锚点"""
        anchors = []

        for i in range(len(segments) - 1):
            current_seg = segments[i]
            next_seg = segments[i + 1]

            # 分析过渡需求
            transition_needs = self._analyze_transition_needs(
                current_seg, next_seg, estimations
            )

            anchor = ContinuityAnchor(
                anchor_id=f"trans_{current_seg.segment_id}_to_{next_seg.segment_id}",
                type="segment_transition",
                from_segment=current_seg.segment_id,
                to_segment=next_seg.segment_id,
                time_point=current_seg.time_range[1],
                transition_type=transition_needs["type"],
                requirements=transition_needs["requirements"],
                visual_constraints=transition_needs["visual_constraints"],
                character_continuity=transition_needs["character_continuity"],
                priority=transition_needs["priority"],
                description=f"从{current_seg.segment_id}到{next_seg.segment_id}的过渡分析"
            )

            anchors.append(anchor)

        return anchors

    def _analyze_transition_needs(self, current_seg: TimeSegment, next_seg: TimeSegment,
                                  estimations: Dict[str, DurationEstimation]) -> Dict[str, Any]:
        """分析过渡需求"""
        # 分析片段类型
        current_types = set(elem.element_type for elem in current_seg.contained_elements)
        next_types = set(elem.element_type for elem in next_seg.contained_elements)

        # 默认值
        transition = {
            "type": "standard",
            "requirements": ["视觉风格保持一致", "环境光线连贯"],
            "visual_constraints": [],
            "character_continuity": [],
            "priority": "medium"
        }

        # 特殊过渡类型
        if ElementType.SILENCE in current_types and ElementType.DIALOGUE in next_types:
            transition["type"] = "silence_to_dialogue"
            transition["requirements"].append("沉默后的第一句话需要自然的情绪过渡")
            transition["priority"] = "high"

        elif ElementType.ACTION in current_types and ElementType.ACTION in next_types:
            # 连续动作
            transition["type"] = "action_sequence"
            transition["requirements"].append("动作序列必须流畅连贯")
            transition["visual_constraints"].append("动作轨迹必须自然")
            transition["priority"] = "high"

        elif ElementType.SCENE in current_types and ElementType.SCENE not in next_types:
            # 场景转换到非场景
            transition["type"] = "scene_exit"
            transition["requirements"].append("离开场景的过渡要自然")
            transition["visual_constraints"].append("镜头运动要连贯")

        # 分析角色状态连续性
        current_chars = set()
        next_chars = set()

        # 这里可以添加更复杂的角色状态分析
        # 暂时使用简单版本

        if current_chars & next_chars:  # 有共同角色
            transition["character_continuity"].append("共同角色的状态必须连贯")
            transition["priority"] = "high"

        return transition

    def _create_final_plan(self, segments: List[TimeSegment], estimations: Dict[str, DurationEstimation],
                           pacing_analysis: PacingAnalysis, continuity_anchors: List[ContinuityAnchor],
                           script_data: UnifiedScript) -> TimelinePlan:
        """创建最终计划"""
        total_duration = self._calculate_total_duration(segments)

        return TimelinePlan(
            timeline_segments=segments,
            duration_estimations=estimations,
            pacing_analysis=pacing_analysis,
            continuity_anchors=continuity_anchors,
            total_duration=total_duration,
            segments_count=len(segments),
            elements_count=len(estimations),
            estimations={k: v.to_dict() for k, v in estimations.items()},
            script_summary={
                "scenes_count": len(script_data.scenes),
                "dialogues_count": len(script_data.dialogues),
                "actions_count": len(script_data.actions)
            },
            processing_stats={
                **self.processing_stats,
                "end_time": datetime.now().isoformat(),
                "total_time": (datetime.now().second - self.processing_stats["start_time"])
            }
        )

    def _calculate_total_duration(self, segments: List[TimeSegment]) -> float:
        """计算总时长"""
        if not segments:
            return 0.0

        total = sum(segment.duration for segment in segments)
        return round(total, 2)
