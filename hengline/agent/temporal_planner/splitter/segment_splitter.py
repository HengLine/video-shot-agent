"""
@FileName: five_second_splitter.py
@Description: 
@Author: HengLine
@Time: 2026/1/15 0:00
"""
from typing import List, Dict, Any, Optional, Tuple

from hengline.agent.temporal_planner.splitter.splitter_analyzer import NarrativeAnalyzer
from hengline.agent.temporal_planner.splitter.splitter_tracker import StateTracker, VisualConsistencyChecker
from hengline.logger import info, warning, debug
from hengline.agent.temporal_planner.splitter.splitter_model import SplitDecision, SplitPriority
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, ElementType, TimeSegment


class SegmentSplitter:
    """全面的5秒分割器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = self._get_default_config(config)

        # 状态跟踪
        self.state_tracker = StateTracker()
        self.narrative_analyzer = NarrativeAnalyzer()
        self.visual_consistency_checker = VisualConsistencyChecker()

        # 统计信息
        self.stats = {
            "total_segments": 0,
            "elements_split": 0,
            "elements_preserved": 0,
            "narrative_score_avg": 0.0,
            "visual_consistency_score_avg": 0.0
        }

    def _get_default_config(self, user_config: Optional[Dict]) -> Dict:
        """获取配置"""
        default_config = {
            # 时长控制
            "target_duration": 5.0,  # 目标时长（秒）
            "min_duration": 4.5,  # 最小时长
            "max_duration": 5.5,  # 最大时长
            "transition_buffer": 0.3,  # 过渡缓冲时间

            # 分割策略
            "split_priorities": [
                SplitPriority.PRESERVE_DIALOGUE,
                SplitPriority.PRESERVE_EMOTIONAL_FLOW,
                SplitPriority.PRESERVE_ACTION_SEQUENCE,
                SplitPriority.PRESERVE_SCENE_CONTINUITY,
                SplitPriority.BALANCE_DURATION
            ],

            # 叙事连贯性
            "min_narrative_score": 0.7,  # 最小叙事连贯性分数
            "emotional_flow_threshold": 2.0,  # 情感流最小持续时间

            # 视觉一致性
            "character_appearance_check": True,  # 检查角色外观
            "scene_consistency_check": True,  # 检查场景一致性
            "camera_angle_consistency": True,  # 检查摄像机角度一致性

            # 对话处理
            "preserve_complete_dialogues": True,  # 保持对话完整
            "dialogue_min_duration": 1.0,  # 对话最小持续时间
            "dialogue_max_duration": 4.5,  # 对话最大持续时间（在5秒内）
            "pause_between_dialogues": 0.2,  # 对话间停顿

            # 动作处理
            "preserve_action_sequences": True,  # 保持动作序列完整
            "action_completion_threshold": 0.8,  # 动作完成度阈值

            # 场景处理
            "scene_establishing_duration": 2.0,  # 场景建立时长
            "scene_transition_duration": 1.5,  # 场景过渡时长

            # 技术参数
            "segment_id_prefix": "seg",
            "visual_summary_max_elements": 3,  # 视觉摘要最大元素数
            "max_elements_per_segment": 4  # 每个片段最大元素数
        }

        if user_config:
            default_config.update(user_config)

        return default_config

    def split_into_segments(self,
                            estimations: Dict[str, DurationEstimation],
                            element_order: List[str],
                            context: Optional[Dict] = None) -> List[TimeSegment]:
        """
        全面的5秒分割

        参数:
            estimations: 时长估算字典
            element_order: 元素顺序
            context: 额外的上下文信息

        返回:
            TimeSegment列表
        """
        info("开始全面的5秒分割...")

        # 验证输入
        validated_estimations = self._validate_estimations(estimations, element_order)

        # 分析叙事结构
        narrative_structure = self.narrative_analyzer.analyze_structure(
            validated_estimations, element_order
        )

        # 初始化状态
        self.state_tracker.reset()

        # 执行智能分割
        segments = self._intelligent_splitting(
            validated_estimations,
            element_order,
            narrative_structure,
            context or {}
        )

        # 验证和优化片段
        optimized_segments = self._optimize_segments(segments, estimations)

        # 计算统计
        self._calculate_statistics(optimized_segments)

        info(f"分割完成: {len(optimized_segments)} 个片段")
        info(f"叙事平均分: {self.stats['narrative_score_avg']:.2f}")
        info(f"视觉一致性平均分: {self.stats['visual_consistency_score_avg']:.2f}")

        return optimized_segments

    def _intelligent_splitting(self,
                               estimations: Dict[str, DurationEstimation],
                               element_order: List[str],
                               narrative_structure: Dict[str, Any],
                               context: Dict) -> List[TimeSegment]:
        """
            智能分割算法 - 返回TimeSegment列表
        参数:
            estimations: 时长估算字典
            element_order: 元素顺序
            narrative_structure: 叙事结构信息
            context: 额外的上下文信息
        结合:
            1. 时长估算
            2. 叙事连贯性
            3. 视觉一致性
            4. 元素类型特性
            5. 上下文信息
        进行智能分割决策
        逻辑:
            1. 遍历元素，尝试将其放入当前片段
            2. 根据分割决策（完整放入、分割、延迟）处理元素
            3. 在满足条件时结束当前片段，开始新片段
            4. 最终生成TimeSegment对象列表
            5. 返回TimeSegment对象列表
        """
        raw_segments = []  # 临时存储字典格式
        current_segment = self._create_empty_segment(len(raw_segments))
        current_time = 0.0
        segment_start_time = 0.0

        i = 0
        while i < len(element_order):
            element_id = element_order[i]
            if element_id not in estimations:
                i += 1
                continue

            element = estimations[element_id]
            element_duration = element.estimated_duration

            # 计算当前片段已用时
            segment_elapsed = current_time - segment_start_time
            remaining = self.config["target_duration"] - segment_elapsed

            # 获取分割决策
            decision = self._make_split_decision(
                element=element,
                element_duration=element_duration,
                remaining_time=remaining,
                current_segment=current_segment,
                next_element_id=element_order[i + 1] if i + 1 < len(element_order) else None,
                next_element=estimations.get(element_order[i + 1]) if i + 1 < len(element_order) else None,
                narrative_structure=narrative_structure
            )

            # 执行分割决策
            if decision.split_type == "complete":
                # 完整放入当前片段
                self._add_element_to_segment(
                    current_segment, element,
                    current_time - segment_start_time, element_duration
                )
                current_time += element_duration
                i += 1

                # 检查是否应结束当前片段
                if self._should_end_segment(current_segment, segment_elapsed + element_duration):
                    self._finalize_segment(current_segment, segment_start_time, current_time)
                    raw_segments.append(current_segment)

                    # 开始新片段
                    segment_start_time = current_time
                    current_segment = self._create_empty_segment(len(raw_segments))

            elif decision.split_type == "split":
                # 分割元素
                part1_duration = decision.split_point
                part2_duration = element_duration - part1_duration

                # 第一部分放入当前片段
                self._add_partial_element(
                    current_segment, element,
                    current_time - segment_start_time,
                    part1_duration, "start"
                )

                # 结束当前片段
                self._finalize_segment(
                    current_segment,
                    segment_start_time,
                    segment_start_time + segment_elapsed + part1_duration
                )
                raw_segments.append(current_segment)

                # 更新时间和开始新片段
                current_time = segment_start_time + segment_elapsed + part1_duration
                segment_start_time = current_time
                current_segment = self._create_empty_segment(len(raw_segments))

                # 第二部分放入新片段
                self._add_partial_element(
                    current_segment, element,
                    0, part2_duration, "continue"
                )
                current_time += part2_duration
                i += 1

            elif decision.split_type == "delay":
                # 延迟到下一个片段
                if current_segment["contained_elements"]:
                    # 结束当前片段
                    self._finalize_segment(current_segment, segment_start_time, current_time)
                    raw_segments.append(current_segment)

                    # 开始新片段
                    segment_start_time = current_time
                    current_segment = self._create_empty_segment(len(raw_segments))

                # 在新片段中放入元素
                self._add_element_to_segment(
                    current_segment, element,
                    0, element_duration
                )
                current_time += element_duration
                i += 1

            else:
                # 未知分割类型，使用默认处理
                warning(f"未知分割类型: {decision.split_type}, 使用默认处理")
                i += 1

        # 处理最后一个片段
        if current_segment["contained_elements"]:
            self._finalize_segment(current_segment, segment_start_time, current_time)
            raw_segments.append(current_segment)

        # 转换为TimeSegment对象
        return self._convert_raw_segments_to_time_segments(raw_segments)

    def _convert_raw_segments_to_time_segments(self, raw_segments: List[Dict]) -> List[TimeSegment]:
        """将原始片段字典转换为TimeSegment对象"""
        time_segments = []

        for i, raw_segment in enumerate(raw_segments):
            segment = TimeSegment(
                segment_id=f"{self.config['segment_id_prefix']}_{i + 1:03d}",
                time_range=(raw_segment["start_time"], raw_segment["end_time"]),
                duration=raw_segment["duration"],
                visual_summary=raw_segment.get("visual_summary", ""),
                contained_elements=raw_segment["contained_elements"],
                start_anchor=raw_segment.get("start_anchor", {}),
                end_anchor=raw_segment.get("end_anchor", {}),
                continuity_requirements=raw_segment.get("continuity_requirements", []),
                shot_type_suggestion=raw_segment.get("shot_type_suggestion", "medium_shot"),
                lighting_suggestion=raw_segment.get("lighting_suggestion", "natural"),
                focus_elements=raw_segment.get("focus_elements", [])
            )

            # 可选：添加额外的元数据
            if "narrative_score" in raw_segment:
                segment.narrative_score = raw_segment["narrative_score"]
            if "visual_consistency_score" in raw_segment:
                segment.visual_consistency_score = raw_segment["visual_consistency_score"]
            if "emotional_flow" in raw_segment:
                segment.emotional_flow = raw_segment["emotional_flow"]

            time_segments.append(segment)

        return time_segments

    def _make_split_decision(self,
                             element: DurationEstimation,
                             element_duration: float,
                             remaining_time: float,
                             current_segment: Dict,
                             next_element_id: Optional[str],
                             next_element: Optional[DurationEstimation],
                             narrative_structure: Dict) -> SplitDecision:
        """
        做出分割决策

        综合考虑:
        1. 时长限制
        2. 叙事连贯性
        3. 视觉一致性
        4. 元素类型特性
        """
        # 基础检查：元素是否能完整放入
        if element_duration <= remaining_time:
            # 可以完整放入，检查其他约束
            return self._evaluate_complete_placement(
                element, element_duration, remaining_time,
                current_segment, next_element, narrative_structure
            )
        else:
            # 需要分割或延迟
            return self._evaluate_split_or_delay(
                element, element_duration, remaining_time,
                current_segment, next_element, narrative_structure
            )

    def _evaluate_complete_placement(self,
                                     element: DurationEstimation,
                                     element_duration: float,
                                     remaining_time: float,
                                     current_segment: Dict,
                                     next_element: Optional[DurationEstimation],
                                     narrative_structure: Dict) -> SplitDecision:
        """评估完整放置的可行性"""
        element_type = element.element_type

        # 检查叙事连贯性
        narrative_score = self.narrative_analyzer.evaluate_coherence(
            current_segment, element, next_element
        )

        # 检查视觉一致性
        visual_score = self.visual_consistency_checker.evaluate_consistency(
            current_segment, element
        )

        # 检查特定类型约束
        type_constraints_ok = True
        reason = "可以完整放置"

        if element_type == ElementType.DIALOGUE:
            # 对话：检查是否会被截断
            if element_duration > self.config["dialogue_max_duration"]:
                type_constraints_ok = False
                reason = "对话过长，需要分割"
            elif element_duration < self.config["dialogue_min_duration"]:
                type_constraints_ok = False
                reason = "对话过短，可能与其他元素合并"

        elif element_type == ElementType.ACTION:
            # 动作：检查是否完整
            action_completeness = self._evaluate_action_completeness(element)
            if action_completeness < self.config["action_completion_threshold"]:
                type_constraints_ok = False
                reason = "动作不完整，需要完整显示"

        elif element_type == ElementType.SCENE:
            # 场景：检查建立时间
            if element_duration < self.config["scene_establishing_duration"]:
                type_constraints_ok = False
                reason = "场景建立时间不足"

        # 决策
        if (type_constraints_ok and
                narrative_score >= self.config["min_narrative_score"] and
                visual_score >= 0.7):

            return SplitDecision(
                element_id=element.element_id,
                split_point=element_duration,
                split_type="complete",
                reason=reason,
                narrative_score=narrative_score,
                visual_consistency_score=visual_score,
                quality_score=(narrative_score + visual_score) / 2
            )
        else:
            # 需要延迟到下一个片段
            return SplitDecision(
                element_id=element.element_id,
                split_point=0,
                split_type="delay",
                reason=f"{reason} (叙事分: {narrative_score:.2f}, 视觉分: {visual_score:.2f})",
                narrative_score=narrative_score,
                visual_consistency_score=visual_score,
                quality_score=(narrative_score + visual_score) / 2
            )

    def _evaluate_split_or_delay(self,
                                 element: DurationEstimation,
                                 element_duration: float,
                                 remaining_time: float,
                                 current_segment: Dict,
                                 next_element: Optional[DurationEstimation],
                                 narrative_structure: Dict) -> SplitDecision:
        """评估分割或延迟的决策"""
        info(f"评估元素 {element.element_id} 的分割或延迟决策")

        # 检查是否可以分割
        can_split = self._can_element_be_split(element, remaining_time)

        if can_split:
            # 找到最佳分割点
            split_point = self._find_best_split_point(element, remaining_time)

            # 评估分割后的连贯性
            narrative_score = self.narrative_analyzer.evaluate_split_coherence(
                current_segment, element, split_point, element_duration
            )

            visual_score = self.visual_consistency_checker.evaluate_split_consistency(
                current_segment, element, split_point
            )

            if (narrative_score >= self.config["min_narrative_score"] and
                    visual_score >= 0.6):
                return SplitDecision(
                    element_id=element.element_id,
                    split_point=split_point,
                    split_type="split",
                    reason=f"元素可分割，分割点: {split_point:.2f}秒",
                    narrative_score=narrative_score,
                    visual_consistency_score=visual_score,
                    quality_score=(narrative_score + visual_score) / 2
                )

        # 如果不能分割或分割效果不好，延迟到下一个片段
        return SplitDecision(
            element_id=element.element_id,
            split_point=0,
            split_type="delay",
            reason="元素不适合分割，延迟到下一个片段",
            narrative_score=0.5,  # 中等分数
            visual_consistency_score=0.7,
            quality_score=0.5
        )

    def _can_element_be_split(self, element: DurationEstimation, max_part_duration: float) -> bool:
        """检查元素是否可以被分割"""
        element_type = element.element_type
        debug(f"检查元素 {element.element_id} 是否可以分割，类型: {element_type}, 时长: {element.estimated_duration}")

        # 对话：通常不分割，除非特别长
        if element_type == ElementType.DIALOGUE:
            if not self.config["preserve_complete_dialogues"]:
                return element.estimated_duration > max_part_duration * 2
            return False

        # 沉默：不分割
        elif element_type == ElementType.SILENCE:
            return False

        # 场景：可以分割，但要有合理断点
        elif element_type == ElementType.SCENE:
            return element.estimated_duration > self.config["scene_establishing_duration"] * 2

        # 动作：可以分割，但要考虑动作完整性
        elif element_type == ElementType.ACTION:
            if not self.config["preserve_action_sequences"]:
                return False

            # 检查动作是否有多阶段
            action_phases = self._analyze_action_phases(element)
            return len(action_phases) > 1

        return True

    def _find_best_split_point(self, element: DurationEstimation, max_part_duration: float) -> float:
        """找到最佳分割点"""
        element_type = element.element_type
        total_duration = element.estimated_duration

        if element_type == ElementType.SCENE:
            # 场景：在关键视觉元素之间分割
            visual_elements = getattr(element, 'visual_hints', {}).get('key_visuals', [])
            if visual_elements:
                # 简单策略：在前半部分分割
                return min(max_part_duration, total_duration * 0.6)

        elif element_type == ElementType.ACTION:
            # 动作：在动作阶段之间分割
            action_phases = self._analyze_action_phases(element)
            for phase_duration in action_phases:
                if phase_duration <= max_part_duration:
                    return phase_duration

        # 默认：在最大部分时长处分割
        return min(max_part_duration, total_duration * 0.8)

    def _analyze_action_phases(self, element: DurationEstimation) -> List[float]:
        """分析动作阶段"""
        description = getattr(element, 'description', '')
        if not description:
            return [element.estimated_duration]

        # 简单分析：基于描述中的动词和连接词
        phases = []
        words = description.split()

        # 检测动作阶段分隔词
        separators = ['然后', '接着', '随后', '之后', '再', '又']

        current_phase_words = 0
        for word in words:
            current_phase_words += 1
            if word in separators and current_phase_words > 2:
                # 估算阶段时长（假设每词0.3秒）
                phase_duration = current_phase_words * 0.3
                phases.append(phase_duration)
                current_phase_words = 0

        # 最后一个阶段
        if current_phase_words > 0:
            phase_duration = current_phase_words * 0.3
            phases.append(phase_duration)

        # 如果没有检测到阶段，返回整个动作
        if not phases:
            phases = [element.estimated_duration]

        return phases

    def _evaluate_action_completeness(self, element: DurationEstimation) -> float:
        """评估动作完整性"""
        description = getattr(element, 'description', '')
        if not description:
            return 1.0

        # 检查是否有明确的起始和结束
        has_start = any(word in description for word in ['开始', '起身', '拿起', '走向', '转向'])
        has_end = any(word in description for word in ['完成', '结束', '放下', '坐下', '停止'])

        if has_start and has_end:
            return 1.0
        elif has_start or has_end:
            return 0.7
        else:
            return 0.5

    def _should_end_segment(self, segment: Dict, current_duration: float) -> bool:
        """判断是否应该结束当前片段"""
        # 1. 达到目标时长
        if current_duration >= self.config["target_duration"]:
            return True

        # 2. 超过最大元素数
        if len(segment["contained_elements"]) >= self.config["max_elements_per_segment"]:
            return True

        # 3. 达到自然断点（如场景结束、情感高潮后）
        last_element = segment["contained_elements"][-1] if segment["contained_elements"] else None
        if last_element:
            element_id = last_element["id"]
            # 这里可以根据元素类型判断是否为自然断点
            # 例如：场景结束、重要对话结束等

        return False

    def _create_empty_segment(self, index: int) -> Dict:
        """创建空片段"""
        return {
            "index": index,
            "contained_elements": [],
            "visual_summary_parts": [],
            "character_states": {},
            "prop_states": {},
            "scene_states": {},
            "camera_angles": [],
            "emotional_flow": [],
            "start_time": 0.0,
            "end_time": 0.0,
            "narrative_score": 0.0,
            "visual_consistency_score": 0.0
        }

    def _add_element_to_segment(self, segment: Dict, element: DurationEstimation,
                                start_offset: float, duration: float):
        """添加完整元素到片段"""
        segment["contained_elements"].append({
            "id": element.element_id,
            "type": element.element_type.value,
            "start_offset": start_offset,
            "duration": duration,
            "is_partial": False
        })

        # 更新状态跟踪
        self.state_tracker.update(element, duration)

        # 收集视觉摘要信息
        self._add_to_visual_summary(segment, element)

        # 更新状态信息
        self._update_segment_states(segment, element)

    def _add_partial_element(self, segment: Dict, element: DurationEstimation,
                             start_offset: float, duration: float, part_type: str):
        """添加部分元素到片段"""
        segment["contained_elements"].append({
            "id": element.element_id,
            "type": element.element_type.value,
            "start_offset": start_offset,
            "duration": duration,
            "is_partial": True,
            "part_type": part_type  # "start", "continue", "end"
        })

        # 更新状态跟踪
        self.state_tracker.update_partial(element, duration, part_type)

        # 收集视觉摘要信息
        self._add_to_visual_summary(segment, element, is_partial=True)

        # 更新状态信息
        self._update_segment_states(segment, element)

    def _add_to_visual_summary(self, segment: Dict, element: DurationEstimation,
                               is_partial: bool = False):
        """添加到视觉摘要"""
        max_elements = self.config["visual_summary_max_elements"]

        if len(segment["visual_summary_parts"]) < max_elements:
            element_type = element.element_type.value
            element_id = element.element_id

            if is_partial:
                summary = f"{element_type}:{element_id}(部分)"
            else:
                summary = f"{element_type}:{element_id}"

            segment["visual_summary_parts"].append(summary)

    def _update_segment_states(self, segment: Dict, element: DurationEstimation):
        """更新片段状态信息"""
        # 角色状态
        if hasattr(element, 'character_states') and element.character_states:
            segment["character_states"].update(element.character_states)

        # 道具状态
        if hasattr(element, 'prop_states') and element.prop_states:
            segment["prop_states"].update(element.prop_states)

        # 场景状态
        if element.element_type == ElementType.SCENE:
            scene_data = getattr(element, 'raw_data', {})
            if scene_data:
                segment["scene_states"] = {
                    "location": scene_data.get("location", ""),
                    "time_of_day": scene_data.get("time_of_day", ""),
                    "weather": scene_data.get("weather", "")
                }

        # 摄像机角度
        visual_hints = getattr(element, 'visual_hints', {})
        shot_types = visual_hints.get("suggested_shot_types", [])
        if shot_types:
            segment["camera_angles"].extend(shot_types)

        # 情感流
        emotional_weight = getattr(element, 'emotional_weight', 1.0)
        segment["emotional_flow"].append({
            "element_id": element.element_id,
            "emotional_weight": emotional_weight
        })

    def _finalize_segment(self, segment: Dict, start_time: float, end_time: float):
        """完成片段的最终设置"""
        segment["start_time"] = start_time
        segment["end_time"] = end_time
        segment["duration"] = end_time - start_time

        # 生成视觉摘要
        if segment["visual_summary_parts"]:
            visual_summary = " | ".join(segment["visual_summary_parts"])
            if len(segment["contained_elements"]) > len(segment["visual_summary_parts"]):
                visual_summary += f" 等{len(segment['contained_elements'])}个元素"
            segment["visual_summary"] = visual_summary
        else:
            segment["visual_summary"] = "过渡片段"

        # 计算叙事和视觉评分
        segment["narrative_score"] = self._calculate_segment_narrative_score(segment)
        segment["visual_consistency_score"] = self._calculate_segment_visual_score(segment)

        # 为下一个智能体生成锚点
        segment["start_anchor"] = self._generate_start_anchor(segment)
        segment["end_anchor"] = self._generate_end_anchor(segment)
        segment["continuity_requirements"] = self._generate_continuity_requirements(segment)

        # 生成技术建议
        shot_type, lighting = self._suggest_technical_parameters(segment)
        segment["shot_type_suggestion"] = shot_type
        segment["lighting_suggestion"] = lighting
        segment["focus_elements"] = self._extract_focus_elements(segment)

    def _calculate_segment_narrative_score(self, segment: Dict) -> float:
        """计算片段叙事评分"""
        elements = segment["contained_elements"]
        if not elements:
            return 0.5  # 过渡片段中等分数

        # 检查元素类型组合
        element_types = set()
        for elem in elements:
            element_types.add(elem["type"])

        # 单一类型通常叙事更连贯
        if len(element_types) == 1:
            type_coherence = 0.9
        elif len(element_types) <= 2:
            type_coherence = 0.7
        else:
            type_coherence = 0.5

        # 检查情感流连续性
        emotional_flow = segment.get("emotional_flow", [])
        emotional_continuity = 0.8 if len(emotional_flow) > 0 else 0.5

        # 检查是否有部分元素
        has_partial = any(elem.get("is_partial", False) for elem in elements)
        if has_partial:
            partial_penalty = 0.2
        else:
            partial_penalty = 0.0

        # 综合评分
        narrative_score = (type_coherence * 0.4 +
                           emotional_continuity * 0.3 +
                           (1.0 - partial_penalty) * 0.3)

        return min(max(narrative_score, 0.1), 1.0)

    def _calculate_segment_visual_score(self, segment: Dict) -> float:
        """计算片段视觉一致性评分"""
        score = 1.0

        # 检查摄像机角度一致性
        camera_angles = segment.get("camera_angles", [])
        if camera_angles:
            unique_angles = len(set(camera_angles))
            if unique_angles > 2:
                score *= 0.8  # 角度太多可能不连贯

        # 检查场景状态一致性
        scene_states = segment.get("scene_states", {})
        if scene_states:
            # 场景状态明确有助于一致性
            if all(scene_states.values()):
                score *= 1.1  # 奖励

        # 检查是否有部分元素
        elements = segment.get("contained_elements", [])
        has_partial = any(elem.get("is_partial", False) for elem in elements)
        if has_partial:
            score *= 0.9  # 部分元素可能影响视觉连续性

        return min(max(score, 0.1), 1.0)

    def _generate_start_anchor(self, segment: Dict) -> Dict:
        """生成开始锚点"""
        anchor = {
            "type": "start_state",
            "constraints": [],
            "visual_match_requirements": []
        }

        # 角色状态约束
        character_states = segment.get("character_states", {})
        for char_name, state in character_states.items():
            anchor["constraints"].append({
                "element": f"character:{char_name}",
                "state": state,
                "priority": "high"
            })

        # 场景状态约束
        scene_states = segment.get("scene_states", {})
        if scene_states:
            anchor["constraints"].append({
                "element": "scene",
                "state": scene_states,
                "priority": "high"
            })

        # 摄像机角度约束
        camera_angles = segment.get("camera_angles", [])
        if camera_angles:
            anchor["constraints"].append({
                "element": "camera",
                "angle": camera_angles[0],  # 使用第一个角度
                "priority": "medium"
            })

        return anchor

    def _generate_end_anchor(self, segment: Dict) -> Dict:
        """生成结束锚点"""
        anchor = {
            "type": "end_state",
            "constraints": [],
            "continuity_hooks": []
        }

        # 基于最后一个元素生成过渡提示
        elements = segment.get("contained_elements", [])
        if elements:
            last_element = elements[-1]

            # 根据元素类型生成不同的过渡提示
            if last_element["type"] == "dialogue":
                anchor["continuity_hooks"].append("保持对话情绪连贯")
                anchor["continuity_hooks"].append("角色视线方向自然")
            elif last_element["type"] == "action":
                anchor["continuity_hooks"].append("动作完成状态保持")
                anchor["continuity_hooks"].append("身体姿势自然过渡")
            elif last_element["type"] == "scene":
                anchor["continuity_hooks"].append("场景氛围延续")

        # 通用过渡提示
        anchor["continuity_hooks"].append("灯光色调自然过渡")
        anchor["continuity_hooks"].append("摄像机运动流畅")

        return anchor

    def _generate_continuity_requirements(self, segment: Dict) -> List[str]:
        """生成连续性要求"""
        requirements = []

        # 角色外观一致性
        character_states = segment.get("character_states", {})
        if character_states:
            requirements.append("角色服装、发型、妆容必须一致")

        # 场景一致性
        scene_states = segment.get("scene_states", {})
        if scene_states:
            requirements.append("场景布置、灯光、色调必须一致")

        # 时间连续性
        elements = segment.get("contained_elements", [])
        if len(elements) > 1:
            requirements.append("动作和对话的时间逻辑必须连贯")

        return requirements

    def _suggest_technical_parameters(self, segment: Dict) -> Tuple[str, str]:
        """建议技术参数"""
        # 默认值
        shot_type = "medium_shot"
        lighting = "natural"

        # 分析元素类型
        elements = segment.get("contained_elements", [])
        element_types = [elem["type"] for elem in elements]

        if "dialogue" in element_types:
            shot_type = "medium_close_up"
            lighting = "three_point"
        elif "silence" in element_types:
            shot_type = "close_up"
            lighting = "soft_key"
        elif "action" in element_types:
            shot_type = "medium_shot"
            lighting = "dramatic"

        # 检查情感强度
        emotional_flow = segment.get("emotional_flow", [])
        if emotional_flow:
            avg_emotional_weight = sum(
                flow["emotional_weight"] for flow in emotional_flow
            ) / len(emotional_flow)

            if avg_emotional_weight > 1.5:
                shot_type = "close_up"
                lighting = "high_contrast"

        return shot_type, lighting

    def _extract_focus_elements(self, segment: Dict) -> List[str]:
        """提取焦点元素"""
        focus_elements = []

        # 根据元素类型添加焦点
        elements = segment.get("contained_elements", [])
        for elem in elements:
            if elem["type"] == "dialogue":
                focus_elements.append("speaker_face")
            elif elem["type"] == "action":
                focus_elements.append("main_action")
            elif elem["type"] == "scene":
                focus_elements.append("key_visual_element")

        # 去重
        unique_focus = []
        for focus in focus_elements:
            if focus not in unique_focus:
                unique_focus.append(focus)

        # 限制数量
        return unique_focus[:2]

    def _optimize_segments(self, segments: List[TimeSegment],
                           estimations: Dict[str, DurationEstimation]) -> List[TimeSegment]:
        """优化片段"""
        optimized_segments = []

        for i, raw_segment in enumerate(segments):
            raw_segment.segment_id = f"{self.config['segment_id_prefix']}_{i + 1:03d}"
            optimized_segments.append(raw_segment)

        return optimized_segments

    def _calculate_statistics(self, segments: List[TimeSegment]):
        """计算统计信息"""
        if not segments:
            return

        total_narrative_score = 0.0
        total_visual_score = 0.0

        for segment in segments:
            # 这里假设segment有narrative_score和visual_consistency_score属性
            # 实际实现中可能需要调整
            total_narrative_score += getattr(segment, 'narrative_score', 0.5)
            total_visual_score += getattr(segment, 'visual_consistency_score', 0.5)

        self.stats.update({
            "total_segments": len(segments),
            "narrative_score_avg": total_narrative_score / len(segments),
            "visual_consistency_score_avg": total_visual_score / len(segments)
        })

    def _validate_estimations(self, estimations: Dict[str, DurationEstimation],
                              element_order: List[str]) -> Dict[str, DurationEstimation]:
        """验证时长估算"""
        validated = {}

        for element_id in element_order:
            if element_id not in estimations:
                warning(f"元素 {element_id} 不在estimations中")
                continue

            estimation = estimations[element_id]

            # 验证时长
            if estimation.estimated_duration <= 0:
                warning(f"元素 {element_id} 时长无效: {estimation.estimated_duration}")
                # 设置最小默认值
                estimation.estimated_duration = max(0.5, getattr(estimation, 'min_duration', 0.5))

            validated[element_id] = estimation

        return validated
