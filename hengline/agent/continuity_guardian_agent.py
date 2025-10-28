# -*- coding: utf-8 -*-
"""
@FileName: continuity_guardian_agent.py
@Description: 连续性守护智能体，负责跟踪角色状态，生成/验证连续性锚点
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from typing import Dict, List, Any, Optional

from hengline.logger import debug, warning, info


class ContinuityGuardianAgent:
    """连续性守护智能体"""

    def __init__(self):
        """初始化连续性守护智能体"""
        # 角色状态记忆
        self.character_states = {}

        # 默认角色外观
        self.default_appearances = {
            "pose": "standing",
            "position": "center of frame",
            "emotion": "neutral",
            "gaze_direction": "forward",
            "holding": "nothing"
        }

    def generate_continuity_constraints(self,
                                        segment: Dict[str, Any],
                                        prev_continuity_state: Optional[Dict[str, Any]] = None,
                                        scene_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate continuity constraints
        
        Args:
            segment: Current segment
            prev_continuity_state: Previous segment's continuity state
            scene_context: Scene context
            
        Returns:
            Continuity constraints
        """
        info(f"Generating continuity constraints, segment ID: {segment.get('id')}")

        # 初始化结果
        continuity_constraints = {
            "characters": {},
            "scene": scene_context or {},
            "camera": {}
        }

        # 获取分段中的角色
        actions = segment.get("actions", [])
        character_names = set()
        phone_characters = set()  # 存储电话那头的角色
        
        for action in actions:
            character_name = action.get("character")
            if character_name:
                if "phone caller" in character_name.lower() or "off-screen" in character_name:
                    phone_characters.add(character_name)
                else:
                    character_names.add(character_name)

        # 如果有上一段的状态，加载它
        if prev_continuity_state:
            self._load_prev_state(prev_continuity_state)

        # 如果有上一段状态，使用它作为约束
        if prev_continuity_state:
            # 处理列表类型的连续性状态（从extract_continuity_anchor返回）
            if isinstance(prev_continuity_state, list):
                for state in prev_continuity_state:
                    character_name = state.get("character_name")
                    # 确保这个角色在当前分段中
                    if character_name and character_name in character_names:
                        constraints = self._generate_character_constraints(character_name, state)
                        continuity_constraints["characters"][character_name] = constraints
            # 向后兼容：处理字典类型的连续性状态
            elif isinstance(prev_continuity_state, dict):
                for character_name, state in prev_continuity_state.items():
                    # 确保这个角色在当前分段中
                    if character_name in character_names:
                        constraints = self._generate_character_constraints(character_name, state)
                        continuity_constraints["characters"][character_name] = constraints
        else:
            # 为每个角色生成初始约束
            for character_name in character_names:
                # 从第一个动作中提取初始状态
                initial_action = next((a for a in actions if a.get("character") == character_name), None)
                initial_state = self._get_character_state(character_name)  # 使用默认状态
                
                # 生成约束
                constraints = self._generate_character_constraints(character_name, initial_state)
                continuity_constraints["characters"][character_name] = constraints

                # 保存初始状态到记忆
                self.character_states[character_name] = initial_state

        # 更新每个角色的状态
        for character_name in character_names:
            # 获取角色在本分段中的所有动作
            character_actions = [a for a in actions if a.get("character") == character_name]
            
            # 如果角色已经有状态，更新它
            if character_name in self.character_states:
                character_state = self.character_states[character_name]
                updated_state = self._update_character_state(character_state, character_actions)
            else:
                # 如果没有状态，使用默认状态
                updated_state = self._get_character_state(character_name)

            # 生成约束
            constraints = self._generate_character_constraints(character_name, updated_state)
            continuity_constraints["characters"][character_name] = constraints

            # 更新记忆中的状态
            self.character_states[character_name] = updated_state
        
        # 为电话那头的角色添加特殊约束
        for phone_character in phone_characters:
            phone_constraints = {
                "must_start_with_pose": "standing",
                "must_start_with_position": "off-screen",  # Ensure position is off-screen
                "must_start_with_emotion": "unknown",
                "must_start_with_gaze": "forward",
                "must_start_with_holding": "nothing",
                "character_description": f"{phone_character}, exists only through voice"
            }
            continuity_constraints["characters"][phone_character] = phone_constraints

        # 生成场景和相机约束
        continuity_constraints["camera"] = self._generate_camera_constraints(segment)

        debug(f"Continuity constraints generated: {continuity_constraints}")
        return continuity_constraints

    def extract_continuity_anchor(self,
                                  segment: Dict[str, Any],
                                  generated_shot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract continuity anchors from generated shot
        
        Args:
            segment: Segment information
            generated_shot: Generated shot
            
        Returns:
            List of continuity anchors
        """
        debug(f"Extracting continuity anchors, shot ID: {generated_shot.get('shot_id')}")

        anchors = []

        # 获取所有角色（包括final_state中的）
        all_characters = set()
        if "final_state" in generated_shot:
            for state in generated_shot["final_state"]:
                if state.get("character_name"):
                    all_characters.add(state.get("character_name"))
        
        # 如果没有从final_state获取到角色，尝试从initial_state获取
        if not all_characters and "initial_state" in generated_shot:
            for state in generated_shot["initial_state"]:
                if state.get("character_name"):
                    all_characters.add(state.get("character_name"))
        
        # 如果还是没有，使用characters_in_frame
        if not all_characters:
            all_characters = set(generated_shot.get("characters_in_frame", []))

        for character_name in all_characters:
            # 构建锚点
            anchor = {
                "character_name": character_name,
                "pose": "unknown",
                "position": "unknown",
                "gaze_direction": "unknown",
                "emotion": "unknown",
                "holding": "unknown"
            }

            # 从final_state提取信息
            if "final_state" in generated_shot:
                for state in generated_shot["final_state"]:
                    if state.get("character_name") == character_name:
                        # For phone caller character, ensure position is off-screen
                        position = state.get("position", "unknown")
                        if "phone caller" in character_name.lower() or "off-screen" in character_name:
                            position = "off-screen"
                        
                        anchor.update({
                            "pose": state.get("pose", "unknown"),
                            "position": position,
                            "gaze_direction": state.get("gaze_direction", "unknown"),
                            "emotion": state.get("emotion", "unknown"),
                            "holding": state.get("holding", "unknown")
                        })
                        break

            # 如果没有final_state，尝试从continuity_anchor提取
            elif "continuity_anchor" in generated_shot:
                for existing_anchor in generated_shot["continuity_anchor"]:
                    if existing_anchor.get("character_name") == character_name:
                        anchor.update(existing_anchor)
                        break
            # 从initial_state提取（如果final_state中没有）
            if anchor["pose"] == "unknown" and "initial_state" in generated_shot:
                for state in generated_shot["initial_state"]:
                    if state.get("character_name") == character_name:
                        # 对于电话那头的角色，确保位置是off-screen
                        position = state.get("position", "unknown")
                        if "电话那头" in character_name or "off-screen" in character_name:
                            position = "off-screen"
                        
                        anchor.update({
                            "pose": state.get("pose", "unknown"),
                            "position": position,
                            "gaze_direction": state.get("gaze_direction", "unknown"),
                            "emotion": state.get("emotion", "unknown"),
                            "holding": state.get("holding", "unknown")
                        })
                        break

            # Special handling for phone caller character
            if "phone caller" in character_name.lower() or "off-screen" in character_name:
                anchor["position"] = "off-screen"
                anchor["pose"] = "off-screen"

            anchors.append(anchor)

        debug(f"Continuity anchors extracted: {anchors}")
        return anchors

    def verify_continuity(self,
                          prev_anchor: List[Dict[str, Any]],
                          current_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify continuity and check for inconsistencies
        
        Args:
            prev_anchor: Previous segment's continuity anchors
            current_constraints: Current segment's continuity constraints
            
        Returns:
            Verification results and correction suggestions
        """
        debug("Verifying continuity")

        issues = []
        suggestions = []

        # 创建角色锚点映射
        prev_anchor_map = {a["character_name"]: a for a in prev_anchor}

        # 检查每个角色的连续性
        for character_name, constraints in current_constraints["characters"].items():
            if character_name in prev_anchor_map:
                prev_state = prev_anchor_map[character_name]

                # 检查姿势连续性
                if prev_state.get("pose") != constraints.get("must_start_with_pose"):
                    issues.append(f"Character {character_name} pose discontinuity")
                    suggestions.append(f"Correct {character_name}'s initial pose to: {prev_state.get('pose')}")

                # 检查位置连续性
                if prev_state.get("position") != constraints.get("must_start_with_position"):
                    issues.append(f"Character {character_name} position discontinuity")
                    suggestions.append(f"Correct {character_name}'s initial position to: {prev_state.get('position')}")

                # 检查情绪连续性
                if prev_state.get("emotion") != constraints.get("must_start_with_emotion"):
                    # 情绪可以有变化，但应该是合理的过渡
                    if not self._is_emotion_transition_valid(prev_state.get("emotion"), constraints.get("must_start_with_emotion")):
                        issues.append(f"Character {character_name} emotion transition unreasonable")
                    suggestions.append(f"Suggest adding emotion transition")

        result = {
            "is_continuous": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions
        }

        if issues:
            warning(f"Continuity verification found issues: {issues}")
        else:
            debug("Continuity verification passed")

        return result

    def _load_prev_state(self, prev_continuity_state: List[Dict[str, Any]]):
        """Load previous segment's continuity state"""
        for state in prev_continuity_state:
            character_name = state.get("character_name")
            if character_name:
                self.character_states[character_name] = state

    def _extract_characters(self, segment: Dict[str, Any]) -> List[str]:
        """Extract all characters from segment"""
        characters = set()
        for action in segment.get("actions", []):
            if "character" in action:
                characters.add(action["character"])
        return list(characters)

    def _get_character_state(self, character_name: str) -> Dict[str, Any]:
        """Get current state of character"""
        if character_name in self.character_states:
            return self.character_states[character_name].copy()
        else:
            # 返回默认状态
            return {
                "character_name": character_name,
                **self.default_appearances
            }

    def _update_character_state(self, state: Dict[str, Any], actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update character state based on actions"""
        updated_state = state.copy()

        for action in actions:
            # 更新动作相关状态
            if "action" in action:
                action_text = action["action"]

                # 更新姿势
                if "sitting" in action_text:
                    updated_state["pose"] = "sitting"
                elif "standing" in action_text:
                    updated_state["pose"] = "standing"
                elif "lying" in action_text:
                    updated_state["pose"] = "lying"
                elif "look down" in action_text:
                    updated_state["gaze_direction"] = "downward"
                elif "look up" in action_text:
                    updated_state["gaze_direction"] = "upward"
                elif "look at" in action_text or "see" in action_text:
                    updated_state["gaze_direction"] = "toward object"

                # 更新位置
                if "window" in action_text:
                    updated_state["position"] = "by window"
                elif "door" in action_text:
                    updated_state["position"] = "near entrance"
                elif "table" in action_text:
                    updated_state["position"] = "at table"

                # 更新手持物品
                if "phone" in action_text or "mobile" in action_text:
                    updated_state["holding"] = "smartphone"
                elif "coffee" in action_text:
                    updated_state["holding"] = "coffee cup"

            # 更新情绪
            if "emotion" in action:
                updated_state["emotion"] = action["emotion"]

        return updated_state

    def _generate_character_constraints(self, character_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate continuity constraints for character"""
        return {
            "must_start_with_pose": state.get("pose", "unknown"),
            "must_start_with_position": state.get("position", "unknown"),
            "must_start_with_emotion": state.get("emotion", "unknown"),
            "must_start_with_gaze": state.get("gaze_direction", "unknown"),
            "must_start_with_holding": state.get("holding", "unknown"),
            "character_description": self._generate_character_description(character_name, state)
        }

    def _generate_character_description(self, character_name: str, state: Dict[str, Any]) -> str:
        """Generate character description"""
        # Can generate more detailed character description as needed
        # Temporarily using simple description template
        return f"{character_name}, {state.get('pose')}, {state.get('emotion')}"

    def _generate_camera_constraints(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate camera constraints"""
        # Simple camera constraint logic
        num_actions = len(segment.get("actions", []))

        if num_actions == 1:
            # 单个动作，使用中景或特写
            shot_type = "medium shot"
        else:
            # 多个动作，使用中景或全景
            shot_type = "medium shot"

        return {
            "recommended_shot_type": shot_type,
            "recommended_angle": "eye-level",
            "must_maintain_consistency": True
        }

    def _is_emotion_transition_valid(self, prev_emotion: str, current_emotion: str) -> bool:
        """Check if emotion transition is valid"""
        # Define valid emotion transition pairs
        valid_transitions = {
            "calm": ["surprised", "attentive", "thinking", "smiling"],
            "surprised": ["shocked", "fearful", "confused", "calm"],
            "shocked": ["fearful", "sad", "angry", "calm"],
            "angry": ["aggressive", "calm", "sad"],
            "sad": ["crying", "calm", "accepting"],
            "happy": ["laughing", "calm", "excited"],
            "nervous": ["anxious", "fearful", "calm"],
            "fearful": ["running", "shocked", "calm"],
        }

        # 如果当前情绪在合理过渡列表中，或者没有定义过渡规则，则认为有效
        if prev_emotion in valid_transitions:
            return current_emotion in valid_transitions[prev_emotion]

        return True
