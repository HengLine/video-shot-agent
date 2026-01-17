"""
@FileName: segment_splitter.py
@Description:  5秒分片器
@Author: HengLine
@Time: 2026/1/16 23:52
"""

from typing import List, Dict, Set, Any

from hengline.agent.temporal_planner.temporal_planner_model import TimeSegment, DurationEstimation, ElementType, ScriptElement


class SegmentSplitter:
    """智能5秒分片器"""

    def __init__(self, segment_duration: float = 5):
        self.segment_duration = segment_duration
        self.min_segment_fill = 2.0  # 最小片段填充时长
        self.max_split_tolerance = 0.3  # 分割容忍度

    def split_into_segments(
            self,
            elements: List[ScriptElement],
            duration_estimations: Dict[str, DurationEstimation]
    ) -> List[TimeSegment]:
        """将元素序列分割为5秒片段"""

        segments = []
        current_segment_elements = []
        current_time = 0.0
        current_duration = 0.0
        segment_id = 1

        for element in elements:
            element_duration = element.estimated_duration.estimated_duration

            # 检查是否可以放入当前片段
            if current_duration + element_duration <= self.segment_duration + self.max_split_tolerance:
                # 可以完整放入
                current_segment_elements.append(element)
                current_duration += element_duration
            else:
                # 需要分割或开始新片段
                if element.element_type == ElementType.DIALOGUE:
                    # 对话不分割，开始新片段
                    if current_duration >= self.min_segment_fill:
                        # 保存当前片段
                        segment = self._create_segment(
                            segment_id, current_segment_elements,
                            current_time, current_duration
                        )
                        segments.append(segment)
                        current_time += current_duration
                        segment_id += 1

                    # 新片段从当前元素开始
                    current_segment_elements = [element]
                    current_duration = element_duration
                else:
                    # 可以分割的元素（场景/动作）
                    remaining_capacity = self.segment_duration - current_duration

                    if remaining_capacity > self.min_segment_fill:
                        # 分割元素
                        split_result = self._split_element(element, remaining_capacity)

                        # 前部分放入当前片段
                        current_segment_elements.append(split_result["first_part"])
                        current_duration = self.segment_duration

                        # 完成当前片段
                        segment = self._create_segment(
                            segment_id, current_segment_elements,
                            current_time, current_duration
                        )
                        segments.append(segment)
                        current_time += current_duration
                        segment_id += 1

                        # 新片段从后部分开始
                        current_segment_elements = [split_result["second_part"]]
                        current_duration = split_result["second_part"].estimated_duration.estimated_duration
                    else:
                        # 直接开始新片段
                        if current_duration >= self.min_segment_fill:
                            segment = self._create_segment(
                                segment_id, current_segment_elements,
                                current_time, current_duration
                            )
                            segments.append(segment)
                            current_time += current_duration
                            segment_id += 1

                        current_segment_elements = [element]
                        current_duration = element_duration

            # 如果当前片段正好达到目标时长
            if abs(current_duration - self.segment_duration) < 0.1:
                segment = self._create_segment(
                    segment_id, current_segment_elements,
                    current_time, current_duration
                )
                segments.append(segment)
                current_time += current_duration
                segment_id += 1
                current_segment_elements = []
                current_duration = 0.0

        # 处理最后一个片段
        if current_segment_elements:
            segment = self._create_segment(
                segment_id, current_segment_elements,
                current_time, current_duration
            )
            segments.append(segment)

        return segments

    def _create_segment(
            self,
            segment_id: int,
            elements: List[ScriptElement],
            start_time: float,
            duration: float
    ) -> TimeSegment:
        """创建时间片段"""

        segment_id_str = f"segment_{segment_id:03d}"

        # 生成视觉内容描述
        visual_content = self._generate_visual_content(elements)

        # 生成叙事弧
        narrative_arc = self._generate_narrative_arc(elements)

        # 收集元素ID
        element_ids = [e.element_id for e in elements]

        # 提取视觉一致性标签
        visual_tags = self._extract_visual_tags(elements)

        return TimeSegment(
            segment_id=segment_id_str,
            time_range=(start_time, start_time + duration),
            duration=duration,
            visual_content=visual_content,
            element_coverage=element_ids,
            narrative_arc=narrative_arc,
            visual_consistency_tags=visual_tags
        )

    def _split_element(self, element: ScriptElement, available_time: float) -> Dict[str, Any]:
        """分割元素为两部分"""
        # 这里是简化的分割逻辑
        # 实际实现需要根据元素类型进行智能分割
        return {
            "first_part": element,
            "second_part": element,
            "split_point": available_time
        }

    def _generate_visual_content(self, elements: List[ScriptElement]) -> str:
        """生成片段的视觉内容描述"""
        descriptions = []

        for element in elements:
            if element.element_type == ElementType.SCENE:
                scene = element.original_data
                descriptions.append(f"场景：{scene.location}，{scene.description[:50]}...")
            elif element.element_type == ElementType.ACTION:
                action = element.original_data
                descriptions.append(f"动作：{action.actor}{action.description}")
            elif element.element_type == ElementType.DIALOGUE:
                dialogue = element.original_data
                descriptions.append(f"对话：{dialogue.speaker}：{dialogue.content}")

        return " | ".join(descriptions[:3])  # 只取前三个描述

    def _generate_narrative_arc(self, elements: List[ScriptElement]) -> str:
        """为片段生成叙事弧描述"""

        if not elements:
            return "空片段"

        # 1. 分析元素组合
        element_analysis = self._analyze_element_combination(elements)

        # 2. 提取关键叙事特征
        narrative_features = self._extract_narrative_features(elements)

        # 3. 确定主导叙事类型
        dominant_type = self._determine_dominant_narrative_type(element_analysis, narrative_features)

        # 4. 生成叙事弧描述
        narrative_arc = self._compose_narrative_arc_description(
            elements, dominant_type, element_analysis, narrative_features
        )

        return narrative_arc

    def _analyze_element_combination(self, elements: List[ScriptElement]) -> Dict[str, Any]:
        """分析元素组合模式"""

        analysis = {
            "element_counts": {
                ElementType.SCENE: 0,
                ElementType.DIALOGUE: 0,
                ElementType.ACTION: 0
            },
            "primary_type": None,
            "type_diversity": 0,
            "sequential_pattern": "",
            "emotional_tone": "neutral",
            "action_density": 0.0
        }

        # 统计元素类型
        for element in elements:
            element_type = element.element_type
            analysis["element_counts"][element_type] = analysis["element_counts"].get(element_type, 0) + 1

        # 确定主要类型
        total_elements = len(elements)
        for elem_type, count in analysis["element_counts"].items():
            if count > total_elements * 0.4:  # 占比超过40%为主导类型
                analysis["primary_type"] = elem_type
                break

        # 计算类型多样性（0-1之间）
        unique_types = sum(1 for count in analysis["element_counts"].values() if count > 0)
        analysis["type_diversity"] = unique_types / 3.0

        # 分析序列模式
        if len(elements) >= 2:
            seq_pattern = []
            for i in range(len(elements) - 1):
                type1 = elements[i].element_type
                type2 = elements[i + 1].element_type
                if type1 == type2:
                    seq_pattern.append("repeat")
                elif type1 == ElementType.SCENE and type2 == ElementType.DIALOGUE:
                    seq_pattern.append("scene_to_dialogue")
                elif type1 == ElementType.DIALOGUE and type2 == ElementType.ACTION:
                    seq_pattern.append("dialogue_to_action")
                else:
                    seq_pattern.append("other")

            # 找出最常见的模式
            if seq_pattern:
                from collections import Counter
                most_common = Counter(seq_pattern).most_common(1)[0][0]
                analysis["sequential_pattern"] = most_common

        # 计算动作密度
        action_count = analysis["element_counts"][ElementType.ACTION]
        analysis["action_density"] = action_count / total_elements if total_elements > 0 else 0

        return analysis

    def _extract_narrative_features(self, elements: List[ScriptElement]) -> Dict[str, Any]:
        """提取叙事特征"""

        features = {
            "has_dialogue": False,
            "has_action": False,
            "has_scene_change": False,
            "emotional_intensity": 0.0,
            "character_interaction": False,
            "key_moments": [],
            "time_progression": "real_time",
            "spatial_focus": "single"
        }

        emotional_weights = {
            "紧张": 0.9, "激动": 0.8, "愤怒": 0.85, "悲伤": 0.7,
            "平静": 0.3, "微颤": 0.6, "哽咽": 0.75, "低声": 0.4
        }

        action_intensity_weights = {
            "physiological": 0.7,  # 生理反应
            "facial": 0.8,  # 面部表情
            "gesture": 0.6,  # 手势
            "posture": 0.5,  # 姿势
            "interaction": 0.9,  # 交互
            "gaze": 0.4,  # 注视
            "device_alert": 0.7,  # 设备提醒
            "prop_fall": 0.8  # 道具掉落
        }

        # 检查特征
        for element in elements:
            element_type = element.element_type

            if element_type == ElementType.DIALOGUE:
                features["has_dialogue"] = True
                dialogue = element.original_data
                emotion = dialogue.emotion
                features["emotional_intensity"] = max(
                    features["emotional_intensity"],
                    emotional_weights.get(emotion, 0.5)
                )

                # 检查是否有角色交互
                if dialogue.speaker and dialogue.target and dialogue.speaker != dialogue.target:
                    features["character_interaction"] = True

            elif element_type == ElementType.ACTION:
                features["has_action"] = True
                action = element.original_data

                # 添加动作强度
                intensity = action_intensity_weights.get(action.type, 0.5)
                features["emotional_intensity"] = max(features["emotional_intensity"], intensity)

                # 识别关键动作
                if action.type in ["facial", "physiological", "interaction", "prop_fall"]:
                    key_moment = {
                        "type": action.type,
                        "description": action.description,
                        "actor": action.actor,
                        "significance": "high" if action.type in ["physiological", "interaction"] else "medium"
                    }
                    features["key_moments"].append(key_moment)

            elif element_type == ElementType.SCENE:
                scene = element.original_data
                # 检查场景变化
                if features.get("last_scene_id") and features["last_scene_id"] != scene.scene_id:
                    features["has_scene_change"] = True
                features["last_scene_id"] = scene.scene_id

        # 限制情感强度在0-1之间
        features["emotional_intensity"] = min(features["emotional_intensity"], 1.0)

        return features

    def _determine_dominant_narrative_type(
            self,
            element_analysis: Dict[str, Any],
            narrative_features: Dict[str, Any]
    ) -> str:
        """确定主导的叙事类型"""

        element_counts = element_analysis["element_counts"]
        total_elements = sum(element_counts.values())

        if total_elements == 0:
            return "neutral"

        # 计算各类型占比
        scene_ratio = element_counts[ElementType.SCENE] / total_elements
        dialogue_ratio = element_counts[ElementType.DIALOGUE] / total_elements
        action_ratio = element_counts[ElementType.ACTION] / total_elements

        # 根据占比和特征确定类型
        if dialogue_ratio >= 0.5:
            if narrative_features["emotional_intensity"] > 0.7:
                return "emotional_dialogue"
            elif narrative_features["character_interaction"]:
                return "interactive_dialogue"
            else:
                return "monologue_or_narration"

        elif action_ratio >= 0.5:
            if narrative_features["emotional_intensity"] > 0.7:
                return "emotional_action"
            elif any(moment["significance"] == "high" for moment in narrative_features.get("key_moments", [])):
                return "significant_action"
            else:
                return "routine_action"

        elif scene_ratio >= 0.5:
            if narrative_features.get("has_scene_change", False):
                return "scene_transition"
            else:
                return "scene_establishment"

        # 混合类型判断
        else:
            if element_analysis["type_diversity"] > 0.7:
                return "complex_interaction"
            elif narrative_features["emotional_intensity"] > 0.6:
                return "emotional_development"
            elif element_analysis["action_density"] > 0.4:
                return "action_sequence"
            else:
                return "balanced_narrative"

    def _compose_narrative_arc_description(
            self,
            elements: List[ScriptElement],
            dominant_type: str,
            element_analysis: Dict[str, Any],
            narrative_features: Dict[str, Any]
    ) -> str:
        """组合生成叙事弧描述"""

        # 基础描述模板
        type_descriptions = {
            "emotional_dialogue": "情感对话",
            "interactive_dialogue": "角色互动对话",
            "monologue_or_narration": "独白或叙述",
            "emotional_action": "情感驱动动作",
            "significant_action": "关键动作",
            "routine_action": "日常动作",
            "scene_transition": "场景过渡",
            "scene_establishment": "场景建立",
            "complex_interaction": "复杂交互",
            "emotional_development": "情感发展",
            "action_sequence": "动作序列",
            "balanced_narrative": "平衡叙事",
            "neutral": "叙事片段"
        }

        # 根据特征添加修饰语
        modifiers = []

        # 情感强度修饰
        if narrative_features.get("emotional_intensity", 0) > 0.8:
            modifiers.append("强烈情感的")
        elif narrative_features.get("emotional_intensity", 0) > 0.6:
            modifiers.append("情感丰富的")

        # 动作密度修饰
        if element_analysis.get("action_density", 0) > 0.5:
            modifiers.append("动作密集的")

        # 对话存在修饰
        if narrative_features.get("has_dialogue", False):
            modifiers.append("含有对话的")

        # 关键时刻修饰
        key_moments = narrative_features.get("key_moments", [])
        if any(m["significance"] == "high" for m in key_moments):
            modifiers.append("包含关键时刻的")

        # 构建最终描述
        base_description = type_descriptions.get(dominant_type, "叙事片段")

        if modifiers:
            # 去重并限制修饰语数量
            unique_modifiers = list(dict.fromkeys(modifiers))[:2]
            modifiers_str = "".join(unique_modifiers)
            return f"{modifiers_str}{base_description}"
        else:
            return base_description

    def _extract_visual_tags(self, elements: List[ScriptElement]) -> Set[str]:
        """提取视觉一致性标签"""
        tags = set()

        for element in elements:
            if element.element_type == ElementType.SCENE:
                scene = element.original_data
                tags.add(f"location_{scene.location}")
                if hasattr(scene, 'time_of_day'):
                    tags.add(f"time_{scene.time_of_day}")
                if hasattr(scene, 'weather'):
                    tags.add(f"weather_{scene.weather}")

        return tags
