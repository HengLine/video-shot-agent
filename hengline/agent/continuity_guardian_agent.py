# -*- coding: utf-8 -*-
"""
@FileName: continuity_guardian_agent.py
@Description: 连续性守护智能体，负责跟踪角色状态，生成/验证连续性锚点
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import time
from typing import Dict, List, Any, Optional
import os

from hengline.config.continuity_guardian_config import ContinuityGuardianConfig
from hengline.logger import debug, warning, info
from hengline.tools.langchain_memory_tool import LangChainMemoryTool


class ContinuityGuardianAgent:
    """连续性守护智能体"""

    def __init__(self):
        """初始化连续性守护智能体"""
        # 初始化配置管理器
        self.config_manager = ContinuityGuardianConfig()
        # 角色状态记忆
        self.character_states = self.config_manager.character_states
        # 加载连续性守护智能体配置
        self.config = self.config_manager.config
        # 初始化LangChain记忆工具（替代原有的向量记忆+状态机）
        self.memory_tool = LangChainMemoryTool()

    def generate_continuity_constraints(self,
                                        segment: Dict[str, Any],
                                        prev_continuity_state: Optional[Dict[str, Any]] = None,
                                        scene_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成连续性约束
        
        Args:
            segment: 当前分段
            prev_continuity_state: 上一段的连续性状态
            scene_context: 场景上下文
            
        Returns:
            连续性约束
        """
        info(f"生成连续性约束, 段落ID: {segment.get('id')}")

        # 初始化结果
        continuity_constraints = {
            "characters": {},
            "scene": scene_context or {},
            "camera": {},
            "metadata": {"generated_at": os.path.getmtime(__file__)}
        }

        # 获取分段中的角色
        actions = segment.get("actions", [])
        character_names = set()
        phone_characters = set()  # 存储电话那头的角色

        for action in actions:
            character_name = action.get("character")
            if character_name:
                if "电话那头" in character_name or "off-screen" in character_name:
                    phone_characters.add(character_name)
                else:
                    character_names.add(character_name)

        # 如果有上一段的状态，加载它
        if prev_continuity_state:
            self.config_manager.load_prev_state(prev_continuity_state)

        # 从场景上下文获取角色外观信息
        character_appearances = {}
        if scene_context and "characters" in scene_context:
            for character in scene_context["characters"]:
                character_name = character.get("name")
                if character_name and "appearance" in character:
                    character_appearances[character_name] = character["appearance"]

        # 为每个角色设置外观信息
        for character_name, appearance in character_appearances.items():
            self.config_manager.set_character_appearance(character_name, appearance)

        # 为每个角色生成初始约束
        for character_name in character_names:
            # 从LangChain记忆中获取状态
            current_state = {
                "character": character_name,
                "segment": segment.get("id", "unknown")
            }
            
            # 存储当前状态到LangChain记忆
            self.memory_tool.store_state(current_state, "连续性守护智能体当前状态")
            
            # 获取状态转换建议
            suggestions = self.memory_tool.get_state_transition_suggestions(current_state)
            
            # 使用配置管理器生成约束
            initial_state = self.config_manager.get_character_state(character_name)
            constraints = self.config_manager._generate_character_constraints(character_name, initial_state)
            continuity_constraints["characters"][character_name] = constraints
            
            # 保存初始状态到记忆
            self.config_manager.character_states[character_name] = initial_state

        # 更新每个角色的状态
        for character_name in character_names:
            # 获取角色在本分段中的所有动作
            character_actions = [a for a in actions if a.get("character") == character_name]

            # 如果角色已经有状态，更新它
            if character_name in self.config_manager.character_states:
                character_state = self.config_manager.character_states[character_name]
                updated_state = self._update_character_state(character_state, character_actions)
                # 使用LangChain记忆工具存储状态
                self.memory_tool.store_state(updated_state, f"角色 {character_name} 更新状态")
            else:
                # 如果没有状态，使用默认状态
                updated_state = self.config_manager.get_character_state(character_name)

            # 生成约束
            constraints = self.config_manager._generate_character_constraints(character_name, updated_state)
            continuity_constraints["characters"][character_name] = constraints

            # 更新记忆中的状态
            self.config_manager.character_states[character_name] = updated_state

        # 为电话那头的角色添加特殊约束
        phone_rules = self.config.get('character_special_rules', {}).get('phone_characters', {})
        for phone_character in phone_characters:
            phone_constraints = {
                "must_start_with_pose": phone_rules.get('default_pose', "off-screen"),
                "must_start_with_position": phone_rules.get('default_position', "off-screen"),
                "must_start_with_emotion": phone_rules.get('default_emotion', "unknown"),
                "must_start_with_gaze": phone_rules.get('default_gaze', "前方"),
                "must_start_with_holding": phone_rules.get('default_holding', "无"),
                "character_description": f"{phone_character}, 仅通过声音存在"
            }
            continuity_constraints["characters"][phone_character] = phone_constraints

        # 生成场景和相机约束
        continuity_constraints["camera"] = self._generate_camera_constraints(segment)

        debug(f"生成的连续性约束: {continuity_constraints}")
        return continuity_constraints

    def extract_continuity_anchor(self,
                                  segment: Dict[str, Any],
                                  generated_shot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从生成的镜头中提取连续性锚点
        
        Args:
            segment: 分段信息
            generated_shot: 生成的镜头
            
        Returns:
            连续性锚点列表
        """
        debug(f"提取连续性锚点, 镜头ID: {generated_shot.get('shot_id')}")

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
                "holding": "unknown",
                "metadata": {"timestamp": os.path.getmtime(__file__)}
            }

            # 从final_state提取信息
            if "final_state" in generated_shot:
                for state in generated_shot["final_state"]:
                    if state.get("character_name") == character_name:
                        # 对于电话那头的角色，确保位置是off-screen
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

            # 对电话角色的特殊处理
            if "电话那头" in character_name or "off-screen" in character_name:
                anchor["position"] = "off-screen"
                anchor["pose"] = "off-screen"
            
            # 存储到LangChain记忆
            anchor_state = {
                "character": character_name,
                "pose": anchor["pose"],
                "position": anchor["position"],
                "emotion": anchor["emotion"],
                "gaze_direction": anchor["gaze_direction"],
                "holding": anchor["holding"]
            }
            self.memory_tool.store_state(anchor_state, "连续性锚点")

            anchors.append(anchor)

        debug(f"连续性锚点提取完成: {anchors}")
        return anchors

    def verify_continuity(self,
                          prev_anchor: List[Dict[str, Any]],
                          current_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证连续性并检查不一致性
        
        Args:
            prev_anchor: 上一段的连续性锚点
            current_constraints: 当前分段的连续性约束
            
        Returns:
            验证结果和修正建议
        """
        debug("验证连续性")

        issues = []
        suggestions = []
        vector_similarity_results = {}
        state_transition_results = []

        # 创建角色锚点映射
        prev_anchor_map = {a["character_name"]: a for a in prev_anchor}

        # 检查每个角色的连续性
        for character_name, constraints in current_constraints["characters"].items():
            if character_name in prev_anchor_map:
                prev_state = prev_anchor_map[character_name]
                
                # 构建当前期望状态
                current_state = {
                    "pose": constraints.get("must_start_with_pose", "unknown"),
                    "position": constraints.get("must_start_with_position", "unknown"),
                    "emotion": constraints.get("must_start_with_emotion", "unknown"),
                    "gaze_direction": constraints.get("must_start_with_gaze", "unknown"),
                    "holding": constraints.get("must_start_with_holding", "unknown")
                }
                
                # 使用LangChain记忆工具检索相似状态
                prev_state_dict = {
                    "character": character_name,
                    "pose": prev_state.get("pose", "unknown"),
                    "position": prev_state.get("position", "unknown"),
                    "emotion": prev_state.get("emotion", "unknown"),
                    "gaze_direction": prev_state.get("gaze_direction", "unknown"),
                    "holding": prev_state.get("holding", "unknown")
                }
                
                # 使用记忆工具获取状态转换建议
                suggestions = self.memory_tool.get_state_transition_suggestions(prev_state_dict)
                
                # 简化验证逻辑，只检查字段是否完全匹配
                is_valid = True
                invalid_fields = []
                
                # 比较每个字段
                for field in ["pose", "position", "emotion", "gaze_direction", "holding"]:
                    if prev_state_dict[field] != current_state[field]:
                        invalid_fields.append(field)
                        is_valid = False
                        field_name = self._get_field_name(field)
                        issues.append(f"角色 {character_name} 的 {field_name} 不连续: {prev_state_dict[field]} -> {current_state[field]}")
                        suggestions.append(f"建议将角色 {character_name} 的 {field_name} 修改为: {prev_state_dict[field]}")
                
                state_transition_results.append({
                    "character": character_name,
                    "is_valid": is_valid,
                    "invalid_fields": invalid_fields,
                    "transition_cost": 0  # 简化处理，不计算转换成本
                })

                # 检查姿势连续性
                if prev_state.get("pose") != constraints.get("must_start_with_pose"):
                    issues.append(f"角色 {character_name} 姿势不连续")
                    suggestions.append(f"修正 {character_name} 的初始姿势为: {prev_state.get('pose')}")

                # 检查位置连续性
                if prev_state.get("position") != constraints.get("must_start_with_position"):
                    issues.append(f"角色 {character_name} 位置不连续")
                    suggestions.append(f"修正 {character_name} 的初始位置为: {prev_state.get('position')}")

                # 检查情绪连续性
                if prev_state.get("emotion") != constraints.get("must_start_with_emotion"):
                    # 情绪可以有变化，但应该是合理的过渡
                    if not self._is_emotion_transition_valid(prev_state.get("emotion"), constraints.get("must_start_with_emotion")):
                        issues.append(f"角色 {character_name} 情绪过渡不合理")
                    suggestions.append(f"建议添加情绪过渡描述")

        result = {
            "is_continuous": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
            "vector_similarity": vector_similarity_results,
            "state_transitions": state_transition_results
        }

        if issues:
            warning(f"连续性验证发现问题: {issues}")
        else:
            debug("连续性验证通过")

        return result

    def _update_character_state(self, state: Dict[str, Any], actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """根据动作更新角色状态"""
        # 首先创建期望状态
        desired_state = state.copy()

        for action in actions:
            # 更新动作相关状态
            if "action" in action:
                action_text = action["action"]

                # 更新姿势
                if "坐" in action_text:
                    desired_state["pose"] = "坐"
                elif "站" in action_text or "站立" in action_text:
                    desired_state["pose"] = "站"
                elif "躺" in action_text or "躺着" in action_text:
                    desired_state["pose"] = "躺"
                elif "低头" in action_text or "向下看" in action_text:
                    desired_state["gaze_direction"] = "下"
                elif "抬头" in action_text or "向上看" in action_text:
                    desired_state["gaze_direction"] = "上"
                elif "看向" in action_text or "看着" in action_text:
                    desired_state["gaze_direction"] = "向物体"

                # 更新位置
                if "窗户" in action_text:
                    desired_state["position"] = "在窗户旁"
                elif "门" in action_text:
                    desired_state["position"] = "在门口"
                elif "桌子" in action_text:
                    desired_state["position"] = "在桌子上"

                # 更新手持物品
                if "手机" in action_text or "电话" in action_text:
                    desired_state["holding"] = "智能手机"
                elif "咖啡" in action_text:
                    desired_state["holding"] = "咖啡杯"

            # 更新情绪
            if "emotion" in action:
                desired_state["emotion"] = action["emotion"]
        
        # 简化处理，直接使用期望状态
        updated_state = desired_state

        return updated_state

    def _generate_camera_constraints(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """生成相机约束"""
        # 从配置加载相机约束规则
        camera_rules = self.config.get('camera_constraints', {})

        # 简单的相机约束逻辑
        num_actions = len(segment.get("actions", []))

        if num_actions == 1:
            # 单个动作
            shot_type = camera_rules.get('single_action_shot_type', "中景")
        else:
            # 多个动作
            shot_type = camera_rules.get('multi_action_shot_type', "中景")

        return {
            "recommended_shot_type": shot_type,
            "recommended_angle": camera_rules.get('default_angle', "平视角度"),
            "must_maintain_consistency": camera_rules.get('maintain_consistency', True)
        }
        
    def _get_field_name(self, field: str) -> str:
        """获取字段的中文名称"""
        field_names = {
            "pose": "姿势",
            "position": "位置",
            "emotion": "情绪",
            "gaze_direction": "视线方向",
            "holding": "手持物品"
        }
        return field_names.get(field, field)

    def _is_emotion_transition_valid(self, prev_emotion: str, current_emotion: str) -> bool:
        """检查情绪过渡是否合理"""
        # 简化版：相同情绪总是有效
        if prev_emotion == current_emotion:
            return True

        # 获取情绪类别
        prev_category = self.config_manager.emotion_categories.get(prev_emotion, "中性")
        curr_category = self.config_manager.emotion_categories.get(current_emotion, "中性")

        # 从配置加载情绪过渡规则
        valid_transitions = self.config.get('emotion_transition_rules', {
            "正面": ["正面", "中性"],
            "负面": ["负面", "中性"],
            "中性": ["正面", "负面", "中性"]
        })

        # 检查是否是有效的过渡
        return curr_category in valid_transitions.get(prev_category, [])

