# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: LLM + 规则约束实现的时序规划（负责将剧本按5秒粒度切分，估算动作时长，使用LangChain实现状态记忆）
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.base_temporal_planner import TemporalPlanner
from hengline.agent.temporal_planner.estimator.ai_duration_estimator import AIDurationEstimator
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, TimeSegment, ElementType, TimelinePlan, PacingAnalysis, ContinuityAnchor
from hengline.logger import error, debug, info


class LLMTemporalPlanner(TemporalPlanner):
    """ LLM 时长估算 """

    def __init__(self, llm_client):
        """初始化时序规划智能体"""
        self.llm = llm_client
        self.config = {}
        self.ai_estimator = AIDurationEstimator(llm_client)
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

        # 配置
        self.max_segment_duration = self.config.get("max_segment_duration", 5.0)
        self.min_segment_duration = self.config.get("min_segment_duration", 2.0)
        self.enable_batch_processing = self.config.get("enable_batch_processing", True)

    def plan_timeline(self, script_data: UnifiedScript) -> TimelinePlan | None:
        """
        规划剧本的时序分段

        Args:
            script_data: 结构化的剧本

        Returns:
            分段计划列表
        """
        """处理剧本的完整流程（优化版）"""
        self.processing_stats["start_time"] = datetime.now()

        debug("开始优化版AI时序规划...")

        try:
            # 1. 预处理和顺序提取
            element_order = self._preprocess_script(script_data)
            debug(f"预处理完成: {len(element_order)} 个元素")

            # 2. 智能时长估算（支持批量）
            start_est = datetime.now()
            estimations = self._estimate_durations_intelligently(element_order, script_data)
            self.processing_stats["element_estimation_time"] = (datetime.now() - start_est).total_seconds()

            debug(f"时长估算完成: {len(estimations)} 个元素")

            # 3. 自适应分片
            start_seg = datetime.now()
            segments = self._adaptive_segmentation(estimations, element_order)
            self.processing_stats["segmentation_time"] = (datetime.now() - start_seg).total_seconds()

            debug(f"自适应分片完成: {len(segments)} 个片段")

            # 4. 多维度节奏分析
            start_ana = datetime.now()
            pacing_analysis = self._multidimensional_pacing_analysis(segments, estimations)
            self.processing_stats["analysis_time"] = (datetime.now() - start_ana).total_seconds()

            debug(f"节奏分析完成: {pacing_analysis['pacing_profile']}")

            # 5. 智能连续性锚点生成
            continuity_anchors = self._generate_continuity_anchors(segments, estimations)

            debug(f"连续性锚点生成: {len(continuity_anchors)} 个锚点")

            # 6. 创建最终规划
            total_duration = self._calculate_total_duration(segments)

            plan = TimelinePlan(
                timeline_segments=segments,
                duration_estimations=estimations,
                pacing_analysis=pacing_analysis,
                continuity_anchors=continuity_anchors,
                total_duration=total_duration,
                segments_count=len(segments),
                pacing_profile=pacing_analysis.pacing_profile,
                processing_stats={
                    **self.processing_stats,
                    "end_time": datetime.now().isoformat(),
                    "total_time": (datetime.now() - self.processing_stats["start_time"]).total_seconds()
                }
            )

            info(f"时序规划完成！总时长: {total_duration:.1f}秒, 片段数: {len(segments)}")

            return plan

        except Exception as e:
            error(f"处理失败: {str(e)}")
            raise

    def _preprocess_script(self, script_data: UnifiedScript) -> List[Dict]:
        """预处理剧本数据（优化版）"""
        element_order = []

        # 场景
        for scene in script_data.scenes:
            element_order.append({
                "id": scene.scene_id,
                "type": ElementType.SCENE,
                "time_offset": scene.start_time,
                "duration": scene.duration,
                "data": scene,
                "priority": 1  # 场景通常是高优先级
            })

        # 对话（包括沉默）
        for dialogue in script_data.dialogues:
            is_silence = dialogue.type == "silence" or not dialogue.content.strip()
            element_order.append({
                "id": dialogue.dialogue_id,
                "type": ElementType.SILENCE if is_silence else ElementType.DIALOGUE,
                "time_offset": dialogue.time_offset,
                "duration": dialogue.duration,
                "data": dialogue,
                "priority": 2 if is_silence else 3  # 沉默优先级高于普通对话
            })

        # 动作
        for action in script_data.actions:
            # 判断动作重要性
            importance = self._assess_action_importance(action.description)

            element_order.append({
                "id": action.action_id,
                "type": ElementType.ACTION,
                "time_offset": action.time_offset,
                "duration": action.duration,
                "data": action,
                "priority": importance,
                "complexity": self._assess_action_complexity(action.description)
            })

        # 按时间偏移排序，相同时间按优先级排序
        element_order.sort(key=lambda x: (x["time_offset"], -x.get("priority", 0)))

        return element_order

    def _assess_action_importance(self, description: str) -> int:
        """评估动作重要性"""
        important_keywords = ["突然", "猛地", "瞬间", "泪水", "震惊", "关键", "重要"]
        moderate_keywords = ["缓缓", "轻轻", "慢慢", "注视", "看向"]

        desc_lower = description.lower()

        if any(keyword in desc_lower for keyword in important_keywords):
            return 1  # 高优先级
        elif any(keyword in desc_lower for keyword in moderate_keywords):
            return 3  # 中优先级
        else:
            return 4  # 低优先级

    def _assess_action_complexity(self, description: str) -> float:
        """评估动作复杂度（0-1）"""
        words = description.split()

        # 多部分动作
        action_parts = ["然后", "接着", "同时", "一边", "又"]
        part_count = sum(1 for part in action_parts if part in description)

        # 精细动作
        fine_movements = ["指尖", "眼神", "嘴角", "眉梢", "喉头"]
        fine_count = sum(1 for movement in fine_movements if movement in description)

        # 计算公式
        complexity = min(
            0.3 * len(words) / 10 +  # 长度因子
            0.4 * min(part_count / 3, 1) +  # 多部分因子
            0.3 * min(fine_count / 2, 1),  # 精细度因子
            1.0
        )

        return round(complexity, 2)

    def _estimate_durations_intelligently(self, element_order: List[Dict],
                                          script_data: UnifiedScript) -> Dict[str, DurationEstimation]:
        """智能时长估算（支持批量）"""
        estimations = {}

        # 分组处理：按类型和上下文分组
        element_groups = self._group_elements_by_context(element_order)

        for group_type, group_elements in element_groups.items():
            debug(f"  处理 {group_type} 组: {len(group_elements)} 个元素")

            if self.enable_batch_processing and len(group_elements) > 1:
                # 批量处理
                batch_results = self._batch_estimate_durations(group_elements, group_type)
                estimations.update(batch_results)
            else:
                # 逐个处理
                for element_info in group_elements:
                    estimation = self._estimate_single_duration(element_info)
                    if estimation:
                        estimations[estimation.element_id] = estimation

        return estimations

    def _group_elements_by_context(self, element_order: List[Dict]) -> Dict[str, List[Dict]]:
        """按上下文分组元素"""
        groups = {
            "scenes": [],
            "dialogues": [],
            "silences": [],
            "important_actions": [],
            "regular_actions": []
        }

        for element in element_order:
            elem_type = element["type"]

            if elem_type == ElementType.SCENE:
                groups["scenes"].append(element)
            elif elem_type == ElementType.DIALOGUE:
                groups["dialogues"].append(element)
            elif elem_type == ElementType.SILENCE:
                groups["silences"].append(element)
            elif elem_type == ElementType.ACTION:
                if element.get("priority", 4) <= 2:  # 重要动作
                    groups["important_actions"].append(element)
                else:
                    groups["regular_actions"].append(element)

        # 移除空组
        return {k: v for k, v in groups.items() if v}

    def _batch_estimate_durations(self, elements: List[Dict], group_type: str) -> Dict[str, DurationEstimation]:
        """批量估算时长（优化版）"""

        # 这里可以调用支持批量处理的AI接口
        # self.ai_estimator.batch_estimate(elements, group_type)
        # 暂时先逐个处理，返回结果
        estimations = {}

        for element in elements:
            estimation = self._estimate_single_duration(element)
            if estimation:
                estimations[estimation.element_id] = estimation

        return estimations

    def _estimate_single_duration(self, element_info: Dict) -> Optional[DurationEstimation]:
        """估算单个元素时长"""
        element_id = element_info["id"]

        # 检查缓存
        cache_key = f"{element_info['type']}_{element_id}"
        if cache_key in self.element_cache:
            return self.element_cache[cache_key]

        try:
            element_data = element_info["data"]

            if element_info["type"] == "scene":
                estimation = self.ai_estimator.estimate_scene_duration(element_data)
            elif element_info["type"] == "dialogue":
                estimation = self.ai_estimator.estimate_dialogue_duration(element_data)
            elif element_info["type"] == "silence":
                estimation = self.ai_estimator.estimate_silence_duration(element_data)
            elif element_info["type"] == "action":
                estimation = self.ai_estimator.estimate_action_duration(element_data)
            else:
                return None

            # 验证和调整估算
            validated = self._validate_and_adjust_estimation(estimation, element_info)

            # 缓存结果
            self.element_cache[cache_key] = validated

            return validated

        except Exception as e:
            print(f"    警告: {element_id} 估算失败: {str(e)}")
            # 返回降级估算
            return self._create_fallback_estimation(element_info)

    def _validate_and_adjust_estimation(self, estimation: DurationEstimation,
                                        element_info: Dict) -> DurationEstimation:
        """验证和调整估算结果"""
        original_duration = element_info.get("duration", 0)
        elem_type = element_info["type"]

        # 验证时长合理性
        min_max_ranges = {
            "scene": (1.5, 15.0),
            "dialogue": (0.8, 8.0),
            "silence": (0.5, 10.0),
            "action": (0.3, 12.0)
        }

        min_dur, max_dur = min_max_ranges.get(elem_type, (0.5, 10.0))
        ai_duration = estimation.ai_estimated_duration

        # 调整超出范围的时长
        if ai_duration < min_dur:
            ai_duration = min_dur
            estimation.confidence = min(estimation.confidence, 0.6)
        elif ai_duration > max_dur:
            ai_duration = max_dur
            estimation.confidence = min(estimation.confidence, 0.7)

        # 更新估算时长
        estimation.ai_estimated_duration = round(ai_duration, 2)

        # 对于重要元素，如果AI估算与原始值差异太大，可能需要特别处理
        if elem_type == "silence" and original_duration > 0:
            # 沉默通常需要足够的时间
            if ai_duration < original_duration * 0.7:
                estimation.ai_estimated_duration = max(ai_duration, original_duration * 0.8)

        return estimation

    def _create_fallback_estimation(self, element_info: Dict) -> DurationEstimation:
        """创建降级估算"""
        elem_type_map = {
            "scene": ElementType.SCENE,
            "dialogue": ElementType.DIALOGUE,
            "silence": ElementType.SILENCE,
            "action": ElementType.ACTION
        }

        original_duration = element_info.get("duration", 2.0)

        # 基于类型和复杂度的降级估算
        if element_info["type"] == "silence":
            fallback_duration = max(original_duration, 2.5)
        elif element_info["type"] == "action":
            complexity = element_info.get("complexity", 0.5)
            fallback_duration = original_duration * (0.8 + complexity * 0.4)
        else:
            fallback_duration = original_duration

        return DurationEstimation(
            element_id=element_info["id"],
            element_type=elem_type_map.get(element_info["type"], ElementType.SCENE),
            original_duration=original_duration,
            estimated_duration=round(fallback_duration, 2),
            confidence=0.4,
            reasoning_breakdown={},
            visual_hints={},
            key_factors=["降级估算"],
            pacing_notes=f"AI估算失败，使用基于{element_info['type']}的降级值",
            emotional_weight=1,
            visual_complexity=1,
            character_states={},
            prop_states={},
            estimated_at=datetime.now().isoformat()
        )

    def _adaptive_segmentation(self, estimations: Dict[str, DurationEstimation],
                               element_order: List[Dict]) -> List[TimeSegment]:
        """自适应分片算法"""
        segments = []
        current_segment = {
            "elements": [],
            "duration": 0.0,
            "start_time": 0.0,
            "types": set()
        }

        segment_index = 1

        for element_info in element_order:
            element_id = element_info["id"]

            if element_id not in estimations:
                continue

            estimation = estimations[element_id]
            element_duration = estimation.estimated_duration

            # 检查分片策略
            should_split = self._should_split_segment(
                current_segment, element_info, estimation, element_duration
            )

            if should_split and current_segment["elements"]:
                # 创建新片段
                segment = self._create_optimized_segment(
                    segment_index, current_segment, estimations
                )
                segments.append(segment)
                segment_index += 1

                # 重置当前片段
                current_segment = {
                    "elements": [],
                    "duration": 0.0,
                    "start_time": current_segment["start_time"] + current_segment["duration"],
                    "types": set()
                }

            # 添加元素到当前片段
            current_segment["elements"].append({
                "element_id": element_id,
                "type": estimation.element_type.value,
                "duration": element_duration,
                "start_in_segment": current_segment["duration"],
                "data": element_info
            })

            current_segment["duration"] += element_duration
            current_segment["types"].add(estimation.element_type.value)

        # 处理最后一个片段
        if current_segment["elements"]:
            segment = self._create_optimized_segment(
                segment_index, current_segment, estimations
            )
            segments.append(segment)

        return segments

    def _should_split_segment(self, current_segment: Dict, element_info: Dict,
                              estimation: DurationEstimation, element_duration: float) -> bool:
        """判断是否应该分片"""
        # 1. 超过最大时长
        if current_segment["duration"] + element_duration > self.max_segment_duration:
            return True

        # 2. 当前片段已经接近5秒，新元素会明显超出
        if (current_segment["duration"] > self.max_segment_duration * 0.8 and
                element_duration > self.max_segment_duration * 0.3):
            return True

        # 3. 类型不匹配：沉默不应该和场景放在一起
        if (estimation.element_type.value == "silence" and
                "scene" in current_segment["types"] and
                current_segment["duration"] > self.min_segment_duration):
            return True

        # 4. 关键转折点：重要对话或动作应该在新片段开始
        if (element_info.get("priority", 4) <= 2 and  # 高优先级
                current_segment["duration"] > self.min_segment_duration):
            return True

        # 5. 情感变化：高情感权重的元素可能需要新片段
        if (estimation.emotional_weight > 1.8 and
                current_segment["duration"] > self.min_segment_duration * 1.5):
            return True

        return False

    def _create_optimized_segment(self, index: int, segment_data: Dict,
                                  estimations: Dict[str, DurationEstimation]) -> TimeSegment:
        """创建优化的时间片段"""
        segment_id = f"seg_{index:03d}"
        start_time = segment_data["start_time"]
        duration = segment_data["duration"]
        end_time = start_time + duration

        # 生成优化的摘要
        visual_summary = self._generate_optimized_summary(segment_data, estimations)

        # 智能锚点生成
        start_anchor, end_anchor = self._generate_intelligent_anchors(segment_data, estimations)

        # 优化的镜头建议
        shot_suggestions = self._generate_optimized_shot_suggestions(segment_data, estimations)

        # 连续性要求
        continuity_reqs = self._extract_optimized_continuity(segment_data, estimations)

        # 准备包含的元素信息
        contained_elements = []
        for elem in segment_data["elements"]:
            contained_elements.append({
                "element_id": elem["element_id"],
                "element_type": elem["type"],
                "duration": elem["duration"],
                "start_in_segment": elem["start_in_segment"]
            })

        return TimeSegment(
            segment_id=segment_id,
            time_range=(round(start_time, 2), round(end_time, 2)),
            duration=round(duration, 2),
            visual_summary=visual_summary,
            contained_elements=contained_elements,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
            continuity_requirements=continuity_reqs,
            shot_type_suggestion=shot_suggestions["shot_type"],
            lighting_suggestion=shot_suggestions["lighting"],
            focus_elements=shot_suggestions["focus"]
        )

    def _generate_optimized_summary(self, segment_data: Dict,
                                    estimations: Dict[str, DurationEstimation]) -> str:
        """生成优化的视觉摘要"""
        elements = segment_data["elements"]

        if not elements:
            return "空片段"

        # 分析片段特征
        features = {
            "has_dialogue": False,
            "has_silence": False,
            "has_action": False,
            "has_scene": False,
            "emotional_intensity": 0,
            "key_elements": []
        }

        for elem in elements:
            elem_id = elem["element_id"]
            if elem_id in estimations:
                est = estimations[elem_id]

                if est.element_type.value == "dialogue":
                    features["has_dialogue"] = True
                elif est.element_type.value == "silence":
                    features["has_silence"] = True
                    features["emotional_intensity"] += est.emotional_weight
                elif est.element_type.value == "action":
                    features["has_action"] = True
                elif est.element_type.value == "scene":
                    features["has_scene"] = True

                # 收集关键元素
                if est.emotional_weight > 1.8 or (hasattr(est, 'visual_complexity') and est.visual_complexity > 1.5):
                    features["key_elements"].append(est.element_type.value)

        # 构建摘要
        summary_parts = []

        if features["has_scene"]:
            summary_parts.append("场景")
        if features["has_dialogue"]:
            summary_parts.append("对话")
        if features["has_silence"]:
            summary_parts.append("沉默")
        if features["has_action"]:
            summary_parts.append("动作")

        # 添加情感强度指示
        if features["emotional_intensity"] > 2.5:
            summary_parts.append("情感高潮")
        elif features["emotional_intensity"] > 1.5:
            summary_parts.append("情感表达")

        # 添加关键元素指示
        if features["key_elements"]:
            unique_keys = list(set(features["key_elements"]))
            if len(unique_keys) == 1:
                summary_parts.append(f"关键{unique_keys[0]}")

        return " | ".join(summary_parts) if summary_parts else "混合内容"

    def _generate_intelligent_anchors(self, segment_data: Dict,
                                      estimations: Dict[str, DurationEstimation]) -> Tuple[Dict, Dict]:
        """生成智能连续性锚点"""
        start_anchor = {
            "type": "segment_start",
            "constraints": [],
            "character_states": {},
            "prop_states": {},
            "environment_state": {}
        }

        end_anchor = {
            "type": "segment_end",
            "constraints": [],
            "character_states": {},
            "prop_states": {},
            "environment_state": {},
            "transition_hints": []
        }

        # 分析片段中的状态变化
        character_state_changes = {}
        prop_state_changes = {}

        for elem in segment_data["elements"]:
            elem_id = elem["element_id"]
            if elem_id in estimations:
                est = estimations[elem_id]

                # 收集状态变化
                for char, state in est.character_states.items():
                    character_state_changes[char] = state
                for prop, state in est.prop_states.items():
                    prop_state_changes[prop] = state

        # 设置开始状态（基于第一个元素）
        if segment_data["elements"]:
            first_elem = segment_data["elements"][0]
            first_id = first_elem["element_id"]
            if first_id in estimations:
                first_est = estimations[first_id]

                # 如果是场景，设置环境状态
                if first_est.element_type == ElementType.SCENE:
                    start_anchor["environment_state"] = {
                        "location": first_est.visual_hints.get("location", "未知"),
                        "lighting": first_est.visual_hints.get("lighting_notes", "自然光"),
                        "mood": first_est.key_factors[0] if first_est.key_factors else "中性"
                    }

        # 设置结束状态
        end_anchor["character_states"] = character_state_changes
        end_anchor["prop_states"] = prop_state_changes

        # 添加约束
        if character_state_changes:
            start_anchor["constraints"].append("角色初始状态必须准确")
            end_anchor["constraints"].append("角色状态变化必须连贯自然")

        if prop_state_changes:
            start_anchor["constraints"].append("道具初始位置必须准确")
            end_anchor["constraints"].append("道具状态变化必须合理")

        # 添加过渡提示
        if segment_data["elements"]:
            last_elem = segment_data["elements"][-1]
            last_id = last_elem["element_id"]
            if last_id in estimations:
                last_est = estimations[last_id]

                if last_est.element_type == ElementType.SILENCE:
                    end_anchor["transition_hints"].append("沉默后的情绪需要自然过渡")
                elif last_est.element_type == ElementType.DIALOGUE:
                    end_anchor["transition_hints"].append("对话后的反应时间很重要")
                elif last_est.element_type == ElementType.ACTION:
                    if hasattr(last_est, 'visual_complexity') and last_est.visual_complexity > 1.5:
                        end_anchor["transition_hints"].append("复杂动作后需要缓冲时间")

        return start_anchor, end_anchor

    def _generate_optimized_shot_suggestions(self, segment_data: Dict,
                                             estimations: Dict[str, DurationEstimation]) -> Dict:
        """生成优化的镜头建议"""
        elements = segment_data["elements"]

        # 默认值
        suggestions = {
            "shot_type": "medium_shot",
            "lighting": "natural",
            "focus": []
        }

        if not elements:
            return suggestions

        # 分析元素组合
        element_types = [elem["type"] for elem in elements]

        # 判断主导元素类型
        type_counts = {}
        for elem_type in element_types:
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1

        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "mixed"

        # 根据主导类型选择镜头
        shot_rules = {
            "silence": {"shot": "close_up", "lighting": "soft_dramatic", "focus": ["面部微表情"]},
            "dialogue": {"shot": "medium_close_up", "lighting": "natural_with_emphasis", "focus": ["说话者表情"]},
            "action": {"shot": "medium_shot", "lighting": "dynamic", "focus": ["动作轨迹"]},
            "scene": {"shot": "wide_shot", "lighting": "atmospheric", "focus": ["环境细节"]}
        }

        if dominant_type in shot_rules:
            base_suggestion = shot_rules[dominant_type]
            suggestions.update(base_suggestion)

        # 特殊组合处理
        if "silence" in element_types and "dialogue" in element_types:
            # 对话后的沉默
            suggestions["shot_type"] = "slow_zoom_to_close_up"
            suggestions["lighting"] = "gradual_dim"
            suggestions["focus"] = ["面部表情变化", "眼神焦点转移"]

        # 添加基于情感强度的调整
        emotional_intensity = 0
        for elem in elements:
            elem_id = elem["element_id"]
            if elem_id in estimations:
                est = estimations[elem_id]
                emotional_intensity += est.emotional_weight

        if emotional_intensity > 3.0:
            suggestions["lighting"] = "high_contrast_dramatic"
            suggestions["focus"].append("情感表达细节")

        # 去重焦点元素
        suggestions["focus"] = list(set(suggestions["focus"]))[:2]

        return suggestions

    def _extract_optimized_continuity(self, segment_data: Dict,
                                      estimations: Dict[str, DurationEstimation]) -> List[str]:
        """提取优化的连续性要求"""
        requirements = []

        # 检查状态变化
        has_state_changes = False
        for elem in segment_data["elements"]:
            elem_id = elem["element_id"]
            if elem_id in estimations:
                est = estimations[elem_id]
                if est.character_states or est.prop_states:
                    has_state_changes = True
                    break

        if has_state_changes:
            requirements.append("状态变化必须保持视觉和逻辑连贯性")

        # 检查情感连续性
        emotional_elements = []
        for elem in segment_data["elements"]:
            elem_id = elem["element_id"]
            if elem_id in estimations:
                est = estimations[elem_id]
                if est.emotional_weight > 1.5:
                    emotional_elements.append(est.element_type.value)

        if len(set(emotional_elements)) > 1:
            requirements.append("不同情感元素间的过渡必须自然")

        # 检查视觉复杂度
        complex_visuals = False
        for elem in segment_data["elements"]:
            elem_id = elem["element_id"]
            if elem_id in estimations:
                est = estimations[elem_id]
                if (hasattr(est, 'visual_complexity') and est.visual_complexity > 1.8) or \
                        est.element_type.value == "scene":
                    complex_visuals = True
                    break

        if complex_visuals:
            requirements.append("复杂视觉元素必须保持一致性")

        return requirements

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
            elem_id = elem["element_id"]
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
                duration_weight = min(elem["duration"] / 3.0, 1.5)

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
                elem_id = elem["element_id"]
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
                elem_id = elem["element_id"]
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
                    elem_id = elem["element_id"]
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
        current_types = set(elem["element_type"] for elem in current_seg.contained_elements)
        next_types = set(elem["element_type"] for elem in next_seg.contained_elements)

        # 默认值
        transition = {
            "type": "standard",
            "requirements": ["视觉风格保持一致", "环境光线连贯"],
            "visual_constraints": [],
            "character_continuity": [],
            "priority": "medium"
        }

        # 特殊过渡类型
        if "silence" in current_types and "dialogue" in next_types:
            transition["type"] = "silence_to_dialogue"
            transition["requirements"].append("沉默后的第一句话需要自然的情绪过渡")
            transition["priority"] = "high"

        elif "action" in current_types and "action" in next_types:
            # 连续动作
            transition["type"] = "action_sequence"
            transition["requirements"].append("动作序列必须流畅连贯")
            transition["visual_constraints"].append("动作轨迹必须自然")
            transition["priority"] = "high"

        elif "scene" in current_types and "scene" not in next_types:
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

    def _calculate_total_duration(self, segments: List[TimeSegment]) -> float:
        """计算总时长"""
        if not segments:
            return 0.0

        total = sum(segment.duration for segment in segments)
        return round(total, 2)
