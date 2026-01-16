# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: 时序规划智能体，负责将剧本按5秒粒度切分，估算动作时长，使用LangChain实现状态记忆
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any

from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.hybrid_temporal_planner import HybridTemporalPlanner
from hengline.agent.temporal_planner.splitter.pacing_analyzer import PacingAnalyzer
from hengline.agent.temporal_planner.splitter.segment_splitter import SegmentSplitter
from hengline.agent.temporal_planner.splitter.splitter_anchor import ContinuityAnchorGenerator
from hengline.agent.temporal_planner.temporal_planner_model import TimelinePlan, DurationEstimation, ElementType, TimeSegment, ContainedElement, PacingProfile, ContinuityAnchor
from hengline.logger import debug, error, info, warning
from utils.log_utils import print_log_exception


class TemporalPlannerAgent:
    """时序规划智能体

    输入：统一格式的剧本解析结果
    输出：精确的5秒时间分片方案

    核心任务：
    1. 为每个剧本元素（对话、动作、描述）估算合理时长
    2. 智能分割为5秒粒度的视频片段
    3. 确保时间分配的合理性和连贯性
    4. 标记关键时间节点和情绪转折点

    """

    def __init__(self, llm):
        """初始化时序规划智能体"""
        self.config = self._get_default_config(user_config=None)
        # 初始化各个组件
        self.temporal_planner = HybridTemporalPlanner(llm)

        # 初始化所有核心模块
        self.segment_splitter = SegmentSplitter()
        self.pacing_analyzer = PacingAnalyzer()
        self.anchor_generator = ContinuityAnchorGenerator()

        # 缓存和状态管理
        self.cache = {}
        self.plan_history = []

        # 统计信息
        self.stats = {
            "plans_generated": 0,
            "total_elements_processed": 0,
            "avg_segments_per_plan": 0.0,
            "avg_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }

    def _get_default_config(self, user_config: Optional[Dict]) -> Dict:
        """获取默认配置"""
        default_config = {
            # 模块配置
            "use_hybrid_estimator": True,  # 使用混合估算器
            "enable_caching": True,  # 启用缓存
            "cache_ttl_seconds": 3600,  # 缓存生存时间

            # 分割配置
            "segment_splitter_config": {
                "target_duration": 5.0,
                "min_duration": 4.5,
                "max_duration": 5.5,
                "preserve_complete_dialogues": True,
                "preserve_action_sequences": True,
                "min_narrative_score": 0.7,
                "segment_id_prefix": "seg"
            },

            # 节奏分析配置
            "pacing_analysis_config": {
                "enable_intensity_analysis": True,
                "enable_emotion_detection": True,
                "intensity_window_size": 3,
                "peak_detection_threshold": 0.8
            },

            # 连贯性配置
            "continuity_config": {
                "enable_character_tracking": True,
                "enable_prop_tracking": True,
                "enable_environment_tracking": True,
                "max_anchors_per_segment": 3
            },

            # 输出配置
            "output_config": {
                "include_detailed_stats": True,
                "generate_visual_summaries": True,
                "max_summary_length": 150,
                "export_formats": ["json", "summary"]
            },

            # 性能配置
            "performance_config": {
                "max_elements_per_batch": 50,
                "parallel_processing": False,
                "enable_progress_logging": True
            }
        }

        if user_config:
            # 深度合并配置
            self._deep_merge(default_config, user_config)

        return default_config

    def _deep_merge(self, base: Dict, update: Dict):
        """深度合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def plan_process(self, structured_script: UnifiedScript) -> TimelinePlan | None:
        """
        规划剧本的时序分段
        
        Args:
            structured_script: 结构化的剧本

        Returns:
            分段计划列表
        """
        debug("开始根据规则规划时序")
        start_time = datetime.now()

        try:
            # 时长估算字典，key为元素ID
            duration_estimations = self.temporal_planner.plan_timeline(structured_script)

            # 1. 验证输入数据
            validated_estimations = self._validate_estimations(duration_estimations)
            info(f"验证通过: {len(validated_estimations)} 个元素")

            # 2. 确定元素顺序（元素顺序列表）
            ordered_elements = self._determine_element_order(validated_estimations, element_order)

            # 3. 执行5秒分割
            segments = self._split_into_segments(validated_estimations, ordered_elements)
            info(f"分割完成: {len(segments)} 个片段")

            # 4. 分析节奏
            pacing_analysis = self._analyze_pacing(segments, validated_estimations)

            # 5. 生成连贯性锚点
            continuity_anchors = self._generate_continuity_anchors(segments)

            # 6. 计算总时长
            total_duration = self._calculate_total_duration(segments)

            # 7. 创建TimelinePlan
            timeline_plan = self._create_timeline_plan(
                segments=segments,
                estimations=validated_estimations,
                pacing_analysis=pacing_analysis,
                continuity_anchors=continuity_anchors,
                total_duration=total_duration
            )

            # 8. 更新统计信息
            self._update_statistics(segments, start_time)

            info(f"TimelinePlan创建成功: {len(segments)}片段, "
                 f"总时长: {total_duration:.2f}秒")

            return timeline_plan

        except Exception as e:
            print_log_exception()
            error(f"执行时序规划异常: {e}")
            return None

    def _organize_elements(self, estimations: Dict[str, Any],
                           script_data: UnifiedScript) -> List[Dict[str, Any]]:
        """组织元素为有序列表"""
        organized = []

        # 获取原始顺序（从script_data中）
        all_elements = []

        # 添加场景（按原始顺序）
        for scene in script_data.scenes:
            element_id = scene["scene_id"]
            if element_id in estimations:
                all_elements.append({
                    "id": element_id,
                    "type": "scene",
                    "data": scene,
                    "estimation": estimations[element_id],
                    "original_order": len(all_elements)  # 维护原始顺序
                })

        # 添加对话（按原始顺序）
        for dialogue in script_data.dialogues:
            element_id = dialogue["dialogue_id"]
            if element_id in estimations:
                # 检查是否为沉默
                elem_type = "silence" if (dialogue.get("type") == "silence" or
                                          not dialogue.get("content", "").strip()) else "dialogue"

                all_elements.append({
                    "id": element_id,
                    "type": elem_type,
                    "data": dialogue,
                    "estimation": estimations[element_id],
                    "original_order": len(all_elements)
                })

        # 添加动作（按原始顺序）
        for action in script_data.actions:
            element_id = action["action_id"]
            if element_id in estimations:
                all_elements.append({
                    "id": element_id,
                    "type": "action",
                    "data": action,
                    "estimation": estimations[element_id],
                    "original_order": len(all_elements)
                })

        # 按原始顺序排序
        all_elements.sort(key=lambda x: x["original_order"])

        return all_elements

    def _validate_estimations(self,
                              estimations: Dict[str, DurationEstimation]) -> Dict[str, DurationEstimation]:
        """验证时长估算数据"""
        validated = {}

        for element_id, estimation in estimations.items():
            if not isinstance(estimation, DurationEstimation):
                warning(f"元素 {element_id} 类型错误: {type(estimation)}")
                continue

            # 验证时长是否合理
            if estimation.estimated_duration <= 0:
                warning(f"元素 {element_id} 时长无效: {estimation.estimated_duration}")
                # 使用最小默认值
                estimation.estimated_duration = max(0.5, estimation.min_duration or 0.5)
                estimation.confidence = max(estimation.confidence - 0.2, 0.1)

            validated[element_id] = estimation

        if len(validated) == 0:
            raise ValueError("没有有效的时长估算数据")

        return validated

    def _determine_element_order(self,
                                 estimations: Dict[str, DurationEstimation],
                                 element_order: Optional[List[str]] = None) -> List[str]:
        """
        确定元素顺序

        如果提供了element_order，则使用它
        否则根据元素类型和ID排序
        """
        if element_order:
            # 验证提供的顺序是否包含所有元素
            missing_elements = set(estimations.keys()) - set(element_order)
            if missing_elements:
                warning(f"element_order中缺少元素: {missing_elements}")
                # 将缺少的元素添加到末尾
                element_order.extend(sorted(missing_elements))
            return element_order

        # 如果没有提供顺序，则按以下规则排序:
        # 1. 按元素类型: 场景 -> 对话 -> 动作 -> 沉默
        # 2. 同类型按ID排序
        type_priority = {
            ElementType.SCENE: 1,
            ElementType.DIALOGUE: 2,
            ElementType.ACTION: 3,
            ElementType.SILENCE: 4
        }

        def sort_key(element_id: str) -> Tuple[int, str]:
            est = estimations[element_id]
            return (type_priority.get(est.element_type, 99), element_id)

        return sorted(estimations.keys(), key=sort_key)

    def _split_into_segments(self,
                             estimations: Dict[str, DurationEstimation],
                             element_order: List[str]) -> List[TimeSegment]:
        """
        执行5秒分割

        调用之前实现的SegmentSplitter模块
        """
        try:
            # 调用之前实现的SegmentSplitter
            segments = self.segment_splitter.split_into_segments(
                estimations=estimations,
                element_order=element_order
            )

            # 验证分割结果
            validated_segments = []
            for segment in segments:
                validated_segment = self._validate_segment(segment)
                if validated_segment:
                    validated_segments.append(validated_segment)

            if len(validated_segments) == 0:
                raise ValueError("分割后没有有效的片段")

            return validated_segments

        except Exception as e:
            error(f"分割失败: {str(e)}")
            # 尝试使用简单的回退分割
            return self._fallback_split(estimations, element_order)

    def _validate_segment(self, segment: TimeSegment) -> Optional[TimeSegment]:
        """验证片段合理性"""
        # 检查时长
        if segment.duration <= 0:
            warning(f"片段 {segment.segment_id} 时长无效: {segment.duration}")
            return None

        # 检查时间范围
        start_time, end_time = segment.time_range
        if start_time < 0 or end_time < start_time:
            warning(f"片段 {segment.segment_id} 时间范围无效: {start_time}-{end_time}")
            return None

        # 检查包含的元素
        if not segment.contained_elements:
            warning(f"片段 {segment.segment_id} 没有包含元素")
            # 空片段可能是允许的（如过渡片段）

        return segment

    def _fallback_split(self,
                        estimations: Dict[str, DurationEstimation],
                        element_order: List[str]) -> List[TimeSegment]:
        """
        回退分割方法（当主要分割方法失败时）
        使用简单的顺序填充算法
        """
        warning("使用回退分割方法")

        segments = []
        current_segment_id = 1
        current_time = 0.0
        current_segment_elements = []
        current_segment_duration = 0.0

        target_duration = self.config["target_segment_duration"]
        min_duration = self.config["min_segment_duration"]
        max_duration = self.config["max_segment_duration"]

        for element_id in element_order:
            if element_id not in estimations:
                continue

            estimation = estimations[element_id]
            element_duration = estimation.estimated_duration

            # 检查是否开始新片段
            if current_segment_duration + element_duration > max_duration:
                # 完成当前片段
                if current_segment_elements:
                    segment = self._create_fallback_segment(
                        segment_id=current_segment_id,
                        start_time=current_time - current_segment_duration,
                        end_time=current_time,
                        elements=current_segment_elements,
                        estimations=estimations
                    )
                    segments.append(segment)
                    current_segment_id += 1

                # 重置当前片段
                current_segment_elements = []
                current_segment_duration = 0.0

            # 添加元素到当前片段
            current_segment_elements.append(ContainedElement(
                element_id=element_id,
                element_type=estimation.element_type,
                duration=element_duration,
                start_offset=current_segment_duration
            ))
            current_segment_duration += element_duration
            current_time += element_duration

            # 如果当前片段达到目标时长，完成它
            if current_segment_duration >= target_duration:
                segment = self._create_fallback_segment(
                    segment_id=current_segment_id,
                    start_time=current_time - current_segment_duration,
                    end_time=current_time,
                    elements=current_segment_elements,
                    estimations=estimations
                )
                segments.append(segment)
                current_segment_id += 1
                current_segment_elements = []
                current_segment_duration = 0.0

        # 处理最后一个片段
        if current_segment_elements:
            segment = self._create_fallback_segment(
                segment_id=current_segment_id,
                start_time=current_time - current_segment_duration,
                end_time=current_time,
                elements=current_segment_elements,
                estimations=estimations
            )
            segments.append(segment)

        return segments

    def _create_fallback_segment(self,
                                 segment_id: int,
                                 start_time: float,
                                 end_time: float,
                                 elements: List[ContainedElement],
                                 estimations: Dict[str, DurationEstimation]) -> TimeSegment:
        """创建回退片段"""
        # 生成视觉摘要
        visual_summary_parts = []
        for elem in elements[:3]:  # 最多取前3个元素
            element_id = elem.element_id
            if element_id in estimations:
                est = estimations[element_id]
                if est.element_type == ElementType.SCENE:
                    visual_summary_parts.append(f"场景:{element_id}")
                elif est.element_type == ElementType.DIALOGUE:
                    visual_summary_parts.append(f"对话:{element_id}")
                elif est.element_type == ElementType.ACTION:
                    visual_summary_parts.append(f"动作:{element_id}")
                elif est.element_type == ElementType.SILENCE:
                    visual_summary_parts.append(f"沉默:{element_id}")

        visual_summary = " | ".join(visual_summary_parts)
        if len(elements) > 3:
            visual_summary += f" 等{len(elements)}个元素"

        # 创建TimeSegment
        return TimeSegment(
            segment_id=f"{self.config['segment_id_prefix']}_{segment_id:03d}",
            time_range=(start_time, end_time),
            duration=end_time - start_time,
            visual_summary=visual_summary,
            contained_elements=elements,
            start_anchor={},
            end_anchor={},
            continuity_requirements=[],
            shot_type_suggestion="medium_shot",  # 默认中景
            lighting_suggestion="natural",
            focus_elements=[]
        )

    def _analyze_pacing(self,
                        segments: List[TimeSegment],
                        estimations: Dict[str, DurationEstimation]) -> Dict[str, Any]:
        """
        分析节奏

        调用之前实现的PacingAnalyzer模块
        """
        if not self.config["generate_pacing_analysis"]:
            return {}

        try:
            # 调用之前实现的 PacingAnalyzer
            pacing_profile = self.pacing_analyzer.analyze(segments, estimations)
            return self._convert_pacing_profile(pacing_profile)

        except Exception as e:
            warning(f"节奏分析失败: {str(e)}")
            return self._simple_pacing_analysis(segments, estimations)

    def _convert_pacing_profile(self, pacing_profile: PacingProfile) -> Dict[str, Any]:
        """转换PacingProfile为字典"""
        # 根据PacingAnalyzer的实际实现调整
        if hasattr(pacing_profile, 'to_dict'):
            return pacing_profile.to_dict()

        # 默认转换
        return {
            "pace_type": getattr(pacing_profile, 'pace_type', 'unknown'),
            "intensity_curve": getattr(pacing_profile, 'intensity_curve', []),
            "emotion_peaks": getattr(pacing_profile, 'emotion_peaks', []),
            "turning_points": getattr(pacing_profile, 'turning_points', []),
            "avg_intensity": getattr(pacing_profile, 'avg_intensity', 0.0),
            "intensity_variance": getattr(pacing_profile, 'intensity_variance', 0.0)
        }

    def _simple_pacing_analysis(self,
                                segments: List[TimeSegment],
                                estimations: Dict[str, DurationEstimation]) -> Dict[str, Any]:
        """简单的节奏分析（回退方法）"""
        if len(segments) < 2:
            return {"pace_type": "single_segment", "notes": "只有一个片段"}

        # 计算片段强度
        intensities = []
        for segment in segments:
            intensity = 0.0
            for elem in segment.contained_elements:
                element_id = elem.element_id
                if element_id in estimations:
                    est = estimations[element_id]
                    # 对话和沉默有情感权重
                    if est.element_type in [ElementType.DIALOGUE, ElementType.SILENCE]:
                        intensity += est.emotional_weight * 0.5
                    # 动作有视觉复杂度
                    elif est.element_type == ElementType.ACTION:
                        intensity += est.visual_complexity * 0.3

            intensities.append(min(intensity, 3.0))  # 限制在0-3范围

        # 判断节奏类型
        avg_intensity = sum(intensities) / len(intensities)
        if avg_intensity > 2.0:
            pace_type = "intense"
        elif avg_intensity > 1.0:
            pace_type = "moderate"
        else:
            pace_type = "calm"

        return {
            "pace_type": pace_type,
            "intensity_curve": intensities,
            "avg_intensity": avg_intensity,
            "max_intensity": max(intensities) if intensities else 0.0,
            "min_intensity": min(intensities) if intensities else 0.0,
            "notes": "简单节奏分析（回退方法）"
        }

    def _generate_continuity_anchors(self, segments: List[TimeSegment]) -> List[Dict[str, Any]]:
        """
        生成连贯性锚点

        调用之前实现的ContinuityAnchorGenerator模块
        """
        if not self.config["include_continuity_anchors"]:
            return []

        try:
            # 调用之前实现的ContinuityAnchorGenerator
            anchors = self.anchor_generator.generate(segments)
            return anchors

        except Exception as e:
            warning(f"连贯性锚点生成失败: {str(e)}")
            return self._simple_continuity_anchors(segments)

    def _simple_continuity_anchors(self, segments: List[TimeSegment]) -> List[Dict[str, Any]]:
        """简单的连贯性锚点（回退方法）"""
        anchors = []

        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]

            # 提取共同元素
            current_elements = {elem.element_id for elem in current.contained_elements}
            next_elements = {elem.element_id for elem in next_seg.contained_elements}
            common_elements = current_elements & next_elements

            if common_elements:
                anchor = {
                    "anchor_id": f"anchor_{current.segment_id}_to_{next_seg.segment_id}",
                    "type": "element_continuity",
                    "common_elements": list(common_elements),
                    "requirement": "保持共同元素的视觉和状态一致性",
                    "priority": "medium"
                }
                anchors.append(anchor)
            else:
                # 即使没有共同元素，也需要过渡
                anchor = {
                    "anchor_id": f"transition_{current.segment_id}_to_{next_seg.segment_id}",
                    "type": "transition",
                    "requirement": "确保场景/情绪的自然过渡",
                    "priority": "low"
                }
                anchors.append(anchor)

        return anchors

    def _calculate_total_duration(self, segments: List[TimeSegment]) -> float:
        """计算总时长"""
        if not segments:
            return 0.0

        # 取最后一个片段的结束时间
        last_segment = segments[-1]
        return last_segment.time_range[1]

    def _create_timeline_plan(self,
                              segments: List[TimeSegment],
                              estimations: Dict[str, DurationEstimation],
                              pacing_analysis: PacingProfile,
                              continuity_anchors: List[ContinuityAnchor],
                              total_duration: float) -> TimelinePlan:
        """
        创建TimelinePlan对象
        """
        # 确定主导情绪
        dominant_emotion = self._determine_dominant_emotion(estimations)

        # 确定关键转折点
        key_transition_points = self._find_key_transition_points(segments, pacing_analysis)

        # 确定全局视觉风格
        global_visual_style = self._determine_visual_style(estimations)

        # 创建TimelinePlan
        timeline_plan = TimelinePlan(
            timeline_segments=segments,
            duration_estimations=estimations,
            pacing_analysis=pacing_analysis,
            elements_count=len(estimations),
            continuity_anchors=continuity_anchors,
            total_duration=total_duration,
            segments_count=len(segments),
            global_visual_style=global_visual_style,
            dominant_emotion=dominant_emotion,
            key_transition_points=key_transition_points
        )

        return timeline_plan

    def _determine_dominant_emotion(self, estimations: Dict[str, DurationEstimation]) -> str:
        """确定主导情绪"""
        emotion_scores = {}

        for estimation in estimations.values():
            # 情感权重高的元素贡献更多
            weight = estimation.emotional_weight

            if estimation.element_type == ElementType.SCENE:
                # 场景的氛围
                scene_data = estimation.raw_data if hasattr(estimation, 'raw_data') else {}
                mood = scene_data.get("mood", "")
                if mood and mood != "未指定":
                    emotion_scores[mood] = emotion_scores.get(mood, 0) + weight

            elif estimation.element_type == ElementType.DIALOGUE:
                # 对话的情绪
                dialogue_data = estimation.raw_data if hasattr(estimation, 'raw_data') else {}
                emotion = dialogue_data.get("emotion", "")
                if emotion and emotion != "未指定":
                    # 简化情绪标签
                    if "微颤" in emotion or "哽咽" in emotion:
                        emotion_scores["悲伤/紧张"] = emotion_scores.get("悲伤/紧张", 0) + weight
                    elif "激动" in emotion:
                        emotion_scores["激动"] = emotion_scores.get("激动", 0) + weight
                    else:
                        emotion_scores[emotion] = emotion_scores.get(emotion, 0) + weight

            elif estimation.element_type == ElementType.SILENCE:
                # 沉默通常有情感
                emotion_scores["深沉"] = emotion_scores.get("深沉", 0) + weight * 1.5

        if emotion_scores:
            # 返回得分最高的情绪
            return max(emotion_scores.items(), key=lambda x: x[1])[0]

        return "neutral"

    def _find_key_transition_points(self,
                                    segments: List[TimeSegment],
                                    pacing_analysis: PacingProfile) -> List[float]:
        """找出关键转折点"""
        transition_points = []

        # 从节奏分析中获取转折点
        turning_points = pacing_analysis.turning_points
        for point in turning_points:
            if isinstance(point, dict) and "time" in point:
                transition_points.append(point["time"])
            elif isinstance(point, (int, float)):
                # 假设是时间值
                transition_points.append(float(point))

        # 如果没有从节奏分析中找到，则使用片段边界
        if not transition_points and len(segments) > 1:
            # 使用每个片段的开始时间（除了第一个）
            for segment in segments[1:]:
                transition_points.append(segment.time_range[0])

        # 去重并排序
        transition_points = sorted(set(transition_points))

        # 限制数量
        max_points = min(len(segments) // 2, 5)  # 最多5个或片段数的一半
        if len(transition_points) > max_points:
            # 选择时间间隔较均匀的点
            transition_points = transition_points[:max_points]

        return transition_points

    def _determine_visual_style(self, estimations: Dict[str, DurationEstimation]) -> str:
        """确定全局视觉风格"""
        style_scores = {}

        for estimation in estimations.values():
            visual_hints = getattr(estimation, 'visual_hints', {})

            # 检查灯光建议
            lighting = visual_hints.get("lighting_notes", "")
            if lighting:
                if "low_key" in lighting or "dark" in lighting:
                    style_scores["暗调/悬疑"] = style_scores.get("暗调/悬疑", 0) + 1
                elif "soft" in lighting or "warm" in lighting:
                    style_scores["柔和/温馨"] = style_scores.get("柔和/温馨", 0) + 1
                elif "dramatic" in lighting or "contrast" in lighting:
                    style_scores["戏剧性/高对比"] = style_scores.get("戏剧性/高对比", 0) + 1

            # 检查颜色调色板
            color_palette = visual_hints.get("color_palette", "")
            if color_palette:
                if "冷色调" in color_palette or "蓝色" in color_palette:
                    style_scores["冷色调"] = style_scores.get("冷色调", 0) + 1
                elif "暖色调" in color_palette or "黄色" in color_palette:
                    style_scores["暖色调"] = style_scores.get("暖色调", 0) + 1

        if style_scores:
            # 返回得分最高的风格
            return max(style_scores.items(), key=lambda x: x[1])[0]

        # 默认根据场景类型判断
        for estimation in estimations.values():
            if estimation.element_type == ElementType.SCENE:
                scene_data = getattr(estimation, 'raw_data', {})
                mood = scene_data.get("mood", "")
                if "紧张" in mood or "压抑" in mood:
                    return "暗调/悬疑"
                elif "孤独" in mood or "悲伤" in mood:
                    return "冷色调/忧郁"

        return "写实/自然"

    def _update_statistics(self, segments: List[TimeSegment], start_time: datetime):
        """更新统计信息"""
        processing_time = (datetime.now() - start_time).total_seconds()

        # 计算片段统计
        segments_by_type = {}
        total_duration = 0.0

        for segment in segments:
            # 统计片段类型（基于包含的元素）
            segment_type = "mixed"
            element_types = set()

            for elem in segment.contained_elements:
                elem_type = elem.element_type
                element_types.add(elem_type)

            if len(element_types) == 1:
                segment_type = list(element_types)[0]
            elif "dialogue" in element_types or "silence" in element_types:
                segment_type = "dialogue_focused"
            elif "action" in element_types:
                segment_type = "action_focused"
            elif "scene" in element_types:
                segment_type = "scene_focused"

            segments_by_type[segment_type] = segments_by_type.get(segment_type, 0) + 1
            total_duration += segment.duration

        # 更新统计
        self.stats.update({
            "total_segments_created": len(segments),
            "avg_segment_duration": total_duration / len(segments) if segments else 0.0,
            "total_processing_time": processing_time,
            "segments_by_type": segments_by_type,
            "processing_speed": len(segments) / processing_time if processing_time > 0 else 0.0
        })
