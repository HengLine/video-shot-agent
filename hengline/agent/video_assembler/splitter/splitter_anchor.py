"""
@FileName: splitter_anchor.py
@Description: 连贯性锚点生成器
@Author: HengLine
@Time: 2026/1/16 23:55
"""

from typing import List, Dict, Set, Any

from hengline.agent.temporal_planner.temporal_planner_model import TimeSegment, ContinuityAnchor, ElementType, ScriptElement


class AnchorGenerator:
    """生成片段间的连贯性锚点"""

    def __init__(self):
        self.anchor_id_counter = 1

    def generate_anchors(
            self,
            segments: List[TimeSegment],
            all_elements: List[ScriptElement]
    ) -> List[ContinuityAnchor]:
        """为所有片段生成连贯性锚点"""

        anchors = []

        for i in range(len(segments) - 1):
            current_segment = segments[i]
            next_segment = segments[i + 1]

            # 生成各类锚点
            visual_anchors = self._generate_visual_match_anchors(
                current_segment, next_segment, all_elements
            )
            anchors.extend(visual_anchors)

            transition_anchors = self._generate_transition_anchors(
                current_segment, next_segment, all_elements
            )
            anchors.extend(transition_anchors)

            character_anchors = self._generate_character_state_anchors(
                current_segment, next_segment, all_elements
            )
            anchors.extend(character_anchors)

            keyframe_anchors = self._generate_keyframe_anchors(
                current_segment, next_segment, all_elements
            )
            anchors.extend(keyframe_anchors)

        return anchors

    def _generate_visual_match_anchors(
            self,
            current_segment: TimeSegment,
            next_segment: TimeSegment,
            all_elements: List[ScriptElement]
    ) -> List[ContinuityAnchor]:
        """生成视觉匹配锚点"""
        anchors = []

        # 1. 检查环境一致性
        current_env_tags = current_segment.visual_consistency_tags
        next_env_tags = next_segment.visual_consistency_tags
        common_env_tags = current_env_tags.intersection(next_env_tags)

        if common_env_tags:
            anchor_id = f"anchor_visual_{self.anchor_id_counter:03d}"
            self.anchor_id_counter += 1

            anchor = ContinuityAnchor(
                anchor_id=anchor_id,
                anchor_type="visual_match",
                priority=9.0,
                from_segment=current_segment.segment_id,
                to_segment=next_segment.segment_id,
                temporal_constraint=f"t={current_segment.end_time}:{next_segment.start_time}",
                description=f"环境一致性匹配：{', '.join(sorted(common_env_tags))}",
                sora_prompt=self._create_visual_match_prompt(common_env_tags),
                visual_reference=None,
                verification_method="visual_comparison",
                mandatory=True,
                visual_constraints={
                    "required_tags": list(common_env_tags),
                    "consistency_level": "high",
                    "allowed_variations": ["lighting_intensity", "camera_angle"]
                }
            )
            anchors.append(anchor)

        return anchors

    def _generate_transition_anchors(
            self,
            current_segment: TimeSegment,
            next_segment: TimeSegment,
            all_elements: List[ScriptElement]
    ) -> List[ContinuityAnchor]:
        """生成过渡锚点"""
        anchors = []

        # 分析两个片段的叙事关系
        narrative_relationship = self._analyze_narrative_relationship(
            current_segment, next_segment, all_elements
        )

        # 根据关系确定过渡类型
        transition_type = self._determine_transition_type(
            current_segment, next_segment, narrative_relationship
        )

        anchor_id = f"anchor_transition_{self.anchor_id_counter:03d}"
        self.anchor_id_counter += 1

        # 提取关键元素用于过渡描述
        key_elements = self._extract_key_elements_for_transition(
            current_segment, next_segment, all_elements
        )

        anchor = ContinuityAnchor(
            anchor_id=anchor_id,
            anchor_type="transition",
            priority=7.0,
            from_segment=current_segment.segment_id,
            to_segment=next_segment.segment_id,
            temporal_constraint=f"transition at {current_segment.end_time}",
            description=f"从'{current_segment.narrative_arc}'过渡到'{next_segment.narrative_arc}'",
            sora_prompt=self._create_transition_prompt(
                transition_type, current_segment, next_segment, key_elements
            ),
            verification_method="temporal_flow",
            mandatory=True,
            transition_type=transition_type,
            requirements={
                "smoothness_level": "high" if narrative_relationship == "continuous" else "medium",
                "motion_flow": "continuous" if transition_type == "dissolve" else "sharp",
                "audio_transition": "seamless" if transition_type in ["dissolve", "fade"] else "cut"
            },
            prohibited_elements=["abrupt_cut", "jump_cut"] if transition_type == "dissolve" else []
        )
        anchors.append(anchor)

        return anchors

    def _generate_character_state_anchors(
            self,
            current_segment: TimeSegment,
            next_segment: TimeSegment,
            all_elements: List[ScriptElement]
    ) -> List[ContinuityAnchor]:
        """生成角色状态锚点"""
        anchors = []

        # 获取两个片段中的角色及其状态
        current_characters = self._extract_character_states(current_segment, all_elements)
        next_characters = self._extract_character_states(next_segment, all_elements)

        # 为共同角色生成状态锚点
        for char_name, current_state in current_characters.items():
            if char_name in next_characters:
                next_state = next_characters[char_name]

                # 检查状态是否需要连续性约束
                if self._needs_continuity_constraint(current_state, next_state):
                    anchor_id = f"anchor_char_{self.anchor_id_counter:03d}"
                    self.anchor_id_counter += 1

                    anchor = ContinuityAnchor(
                        anchor_id=anchor_id,
                        anchor_type="character_state",
                        priority=8.5,
                        from_segment=current_segment.segment_id,
                        to_segment=next_segment.segment_id,
                        temporal_constraint=f"character {char_name} continuity",
                        description=f"角色'{char_name}'状态连续性：{current_state['summary']} → {next_state['summary']}",
                        sora_prompt=self._create_character_continuity_prompt(char_name, current_state, next_state),
                        verification_method="character_tracking",
                        mandatory=True,
                        state_change={
                            "character_name": char_name,
                            "from_state": current_state,
                            "to_state": next_state,
                            "allowed_changes": ["micro_expressions", "breathing_pattern"],
                            "prohibited_changes": ["clothing", "hairstyle", "position_jump"]
                        },
                        character_continuity={
                            "appearance_consistency": "strict",
                            "emotion_transition": "smooth",
                            "position_continuity": "required",
                            "gaze_direction": "maintained_if_speaking"
                        }
                    )
                    anchors.append(anchor)

        return anchors

    def _generate_keyframe_anchors(
            self,
            current_segment: TimeSegment,
            next_segment: TimeSegment,
            all_elements: List[ScriptElement]
    ) -> List[ContinuityAnchor]:
        """生成关键帧锚点"""
        anchors = []

        # 在当前片段的结尾处生成关键帧锚点
        keyframe_time = current_segment.end_time

        # 提取关键帧视觉元素
        keyframe_elements = self._extract_keyframe_elements(current_segment, all_elements)

        if keyframe_elements:
            anchor_id = f"anchor_keyframe_{self.anchor_id_counter:03d}"
            self.anchor_id_counter += 1

            anchor = ContinuityAnchor(
                anchor_id=anchor_id,
                anchor_type="keyframe",
                priority=6.0,
                from_segment=current_segment.segment_id,
                to_segment=next_segment.segment_id,
                temporal_constraint=f"keyframe at {keyframe_time}s",
                description=f"关键帧：{current_segment.segment_id}的结束状态",
                sora_prompt=self._create_keyframe_prompt(keyframe_elements),
                verification_method="frame_analysis",
                mandatory=False,  # 关键帧通常不是强制的
                timestamp=keyframe_time,
                visual_constraints={
                    "key_elements": keyframe_elements,
                    "composition_importance": "high",
                    "reference_for_next": True
                },
                requirements={
                    "frame_accuracy": "medium",
                    "visual_clarity": "high",
                    "serves_as_bridge": True
                }
            )
            anchors.append(anchor)

        return anchors

    def _analyze_narrative_relationship(
            self,
            current_segment: TimeSegment,
            next_segment: TimeSegment,
            all_elements: List[ScriptElement]
    ) -> str:
        """分析两个片段间的叙事关系"""
        # 获取两个片段的元素类型分布
        current_types = self._get_segment_element_types(current_segment, all_elements)
        next_types = self._get_segment_element_types(next_segment, all_elements)

        # 简单的叙事关系判断逻辑
        if current_segment.narrative_arc == next_segment.narrative_arc:
            return "continuous"

        # 检查是否有对话延续
        current_dialogue_chars = self._get_dialogue_characters(current_segment, all_elements)
        next_dialogue_chars = self._get_dialogue_characters(next_segment, all_elements)

        if current_dialogue_chars and next_dialogue_chars:
            if any(char in next_dialogue_chars for char in current_dialogue_chars):
                return "dialogue_continuation"

        # 检查场景变化
        current_env_tags = current_segment.visual_consistency_tags
        next_env_tags = next_segment.visual_consistency_tags

        if len(current_env_tags.intersection(next_env_tags)) < 2:
            return "scene_change"

        return "moderate_transition"

    def _determine_transition_type(
            self,
            current_segment: TimeSegment,
            next_segment: TimeSegment,
            narrative_relationship: str
    ) -> str:
        """确定过渡类型"""
        transition_map = {
            "continuous": "dissolve",
            "dialogue_continuation": "cut",
            "scene_change": "fade",
            "moderate_transition": "cut",
            "time_jump": "fade",
            "emotional_shift": "dissolve"
        }

        return transition_map.get(narrative_relationship, "cut")

    def _extract_character_states(
            self,
            segment: TimeSegment,
            all_elements: List[ScriptElement]
    ) -> Dict[str, Dict[str, Any]]:
        """提取片段中角色的状态信息"""
        character_states = {}

        for element_id in segment.element_coverage:
            element = next((e for e in all_elements if e.element_id == element_id), None)
            if not element:
                continue

            if element.element_type == ElementType.DIALOGUE:
                dialogue = element.original_data
                if dialogue.speaker not in character_states:
                    character_states[dialogue.speaker] = {
                        "is_speaking": True,
                        "emotion": dialogue.emotion,
                        "summary": f"说：'{dialogue.content[:20]}...'",
                        "action_context": "dialogue"
                    }
                else:
                    character_states[dialogue.speaker]["is_speaking"] = True
                    character_states[dialogue.speaker]["emotion"] = dialogue.emotion

            elif element.element_type == ElementType.ACTION:
                action = element.original_data
                if action.actor and not action.actor.startswith("prop_"):
                    if action.actor not in character_states:
                        character_states[action.actor] = {
                            "is_speaking": False,
                            "action": action.description,
                            "action_type": action.type,
                            "summary": f"动作：{action.description}",
                            "action_context": action.type
                        }
                    else:
                        # 更新动作信息
                        character_states[action.actor]["action"] = action.description
                        character_states[action.actor]["summary"] = f"动作：{action.description}"

        return character_states

    def _needs_continuity_constraint(
            self,
            current_state: Dict[str, Any],
            next_state: Dict[str, Any]
    ) -> bool:
        """判断是否需要连续性约束"""
        # 如果角色在对话中，需要连续性
        if current_state.get("is_speaking") or next_state.get("is_speaking"):
            return True

        # 如果有显著动作变化，需要连续性
        if current_state.get("action_context") == "physiological" and next_state.get("action_context") == "physiological":
            return True

        # 情绪变化显著时也需要连续性
        current_emotion = current_state.get("emotion", "")
        next_emotion = next_state.get("emotion", "")
        if current_emotion and next_emotion and current_emotion != next_emotion:
            emotion_intensity = {"紧张": 3, "激动": 3, "悲伤": 2, "平静": 1, "微颤": 2, "哽咽": 3}
            if abs(emotion_intensity.get(current_emotion, 1) - emotion_intensity.get(next_emotion, 1)) >= 2:
                return True

        return False

    def _extract_keyframe_elements(
            self,
            segment: TimeSegment,
            all_elements: List[ScriptElement]
    ) -> List[Dict[str, Any]]:
        """提取关键帧元素"""
        key_elements = []

        # 获取片段的最后几个元素作为关键帧参考
        relevant_elements = segment.element_coverage[-3:] if len(segment.element_coverage) > 3 else segment.element_coverage

        for element_id in relevant_elements:
            element = next((e for e in all_elements if e.element_id == element_id), None)
            if element:
                if element.element_type == ElementType.ACTION:
                    action = element.original_data
                    key_elements.append({
                        "type": "action",
                        "actor": action.actor,
                        "description": action.description,
                        "significance": "high" if action.type in ["facial", "physiological", "gesture"] else "medium"
                    })
                elif element.element_type == ElementType.DIALOGUE:
                    dialogue = element.original_data
                    key_elements.append({
                        "type": "dialogue",
                        "speaker": dialogue.speaker,
                        "content": dialogue.content[:30],
                        "emotion": dialogue.emotion
                    })

        return key_elements

    def _create_visual_match_prompt(self, tags: Set[str]) -> str:
        """创建视觉匹配的Sora提示"""
        tag_descriptions = {
            "location_城市公寓客厅": "城市公寓客厅内景",
            "time_夜晚": "夜间灯光氛围",
            "weather_大雨滂沱": "窗外大雨效果"
        }

        descriptions = [tag_descriptions.get(tag, tag) for tag in sorted(tags)]
        return f"保持视觉一致性：{', '.join(descriptions)}。"

    def _create_transition_prompt(
            self,
            transition_type: str,
            current_segment: TimeSegment,
            next_segment: TimeSegment,
            key_elements: List[Dict[str, Any]]
    ) -> str:
        """创建过渡的Sora提示"""
        transition_prompts = {
            "cut": "直接切换到下一镜头",
            "dissolve": "淡入淡出过渡到下一场景",
            "fade": "渐隐渐现过渡",
            "match": "动作匹配剪辑"
        }

        base_prompt = transition_prompts.get(transition_type, "平滑过渡")

        # 添加关键元素信息
        if key_elements:
            element_desc = "，".join([f"{e.get('actor', e.get('speaker', ''))}的{e.get('description', e.get('content', ''))[:20]}"
                                     for e in key_elements[:2]])
            return f"{base_prompt}，保持{element_desc}的连贯性。"

        return base_prompt

    def _create_character_continuity_prompt(
            self,
            character_name: str,
            current_state: Dict[str, Any],
            next_state: Dict[str, Any]
    ) -> str:
        """创建角色连续性的Sora提示"""
        prompt_parts = [f"保持角色'{character_name}'的连续性"]

        # 添加情绪连续性
        if current_state.get("emotion") and next_state.get("emotion"):
            if current_state["emotion"] != next_state["emotion"]:
                prompt_parts.append(f"情绪从{current_state['emotion']}过渡到{next_state['emotion']}")
            else:
                prompt_parts.append(f"保持{current_state['emotion']}情绪")

        # 添加动作连续性
        if current_state.get("action") and next_state.get("action"):
            prompt_parts.append(f"动作从'{current_state['action']}'自然过渡")

        return "，".join(prompt_parts) + "。"

    def _create_keyframe_prompt(self, keyframe_elements: List[Dict[str, Any]]) -> str:
        """创建关键帧的Sora提示"""
        if not keyframe_elements:
            return "保持画面构图和氛围的一致性。"

        descriptions = []
        for element in keyframe_elements:
            if element["type"] == "action":
                desc = f"{element['actor']}的{element['description']}"
            else:
                desc = f"{element['speaker']}说：'{element['content']}'"
            descriptions.append(desc)

        return f"关键帧参考：{'，'.join(descriptions[:2])}。"

    def _get_segment_element_types(
            self,
            segment: TimeSegment,
            all_elements: List[ScriptElement]
    ) -> Dict[ElementType, int]:
        """获取片段的元素类型分布"""
        type_count = {ElementType.SCENE: 0, ElementType.DIALOGUE: 0, ElementType.ACTION: 0}

        for element_id in segment.element_coverage:
            element = next((e for e in all_elements if e.element_id == element_id), None)
            if element:
                type_count[element.element_type] = type_count.get(element.element_type, 0) + 1

        return type_count

    def _get_dialogue_characters(
            self,
            segment: TimeSegment,
            all_elements: List[ScriptElement]
    ) -> Set[str]:
        """获取片段中的对话角色"""
        characters = set()

        for element_id in segment.element_coverage:
            element = next((e for e in all_elements if e.element_id == element_id), None)
            if element and element.element_type == ElementType.DIALOGUE:
                dialogue = element.original_data
                characters.add(dialogue.speaker)

        return characters

    def _extract_key_elements_for_transition(
            self,
            current_segment: TimeSegment,
            next_segment: TimeSegment,
            all_elements: List[ScriptElement]
    ) -> List[Dict[str, Any]]:
        """提取用于过渡描述的关键元素"""
        key_elements = []

        # 获取两个片段的最后一个和第一个重要元素
        segments_to_check = [current_segment, next_segment]

        for segment in segments_to_check:
            if segment.element_coverage:
                element_id = segment.element_coverage[-1] if segment == current_segment else segment.element_coverage[0]
                element = next((e for e in all_elements if e.element_id == element_id), None)

                if element:
                    if element.element_type == ElementType.DIALOGUE:
                        dialogue = element.original_data
                        key_elements.append({
                            "type": "dialogue",
                            "speaker": dialogue.speaker,
                            "content": dialogue.content[:20],
                            "position": "ending" if segment == current_segment else "beginning"
                        })
                    elif element.element_type == ElementType.ACTION:
                        action = element.original_data
                        key_elements.append({
                            "type": "action",
                            "actor": action.actor,
                            "description": action.description,
                            "position": "ending" if segment == current_segment else "beginning"
                        })

        return key_elements
