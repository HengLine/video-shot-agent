# -*- coding: utf-8 -*-
"""
@FileName: shot_generator_agent.py
@Description: 分镜生成智能体，负责生成符合AI视频模型要求的提示词
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from typing import Dict, List, Any

# LLMChain在langchain 1.0+中已更改，我们将直接使用模型和提示词
from langchain_core.prompts import ChatPromptTemplate
from openai import AuthenticationError, APIError

from hengline.logger import debug, error, warning
from hengline.prompts.prompts_manager import prompt_manager
from utils.log_utils import print_log_exception


class ShotGeneratorAgent:
    """分镜生成智能体"""

    def __init__(self, llm=None):
        """
        初始化分镜生成智能体
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        # 分镜生成提示词模板 - 从YAML加载或使用默认
        self.shot_generation_template = prompt_manager.get_shot_generator_prompt()

    def generate_shot(self,
                      segment: Dict[str, Any],
                      continuity_constraints: Dict[str, Any],
                      scene_context: Dict[str, Any],
                      style: str = "realistic",
                      shot_id: int = 1) -> Dict[str, Any]:
        """
        生成单个分镜，使用YAML配置的提示词模板，增强错误处理和字段验证
        
        Args:
            segment: 分段信息
            continuity_constraints: 连续性约束
            scene_context: 场景上下文
            style: 视频风格
            shot_id: 分镜ID
            
        Returns:
            分镜对象
        """
        debug(f"生成分镜，ID: {shot_id}")

        # 准备输入数据
        actions_text = self._format_actions_text(segment.get("actions", []))
        continuity_constraints_text = self._format_continuity_constraints(continuity_constraints)

        # 构建提示词输入，确保所有变量与YAML模板匹配
        prompt_input = {
            "location": scene_context.get("location", "未知位置"),
            "time": scene_context.get("time", "未知时间"),
            "atmosphere": scene_context.get("atmosphere", "未知氛围"),
            "actions_text": actions_text,
            "continuity_constraints_text": continuity_constraints_text,
            "style": style,
            "shot_id": shot_id
        }

        try:
            if self.llm:
                debug("使用LLM和YAML配置的提示词模板生成分镜")
                try:
                    # 直接使用已初始化的ChatPromptTemplate对象
                    if isinstance(self.shot_generation_template, ChatPromptTemplate):
                        current_template = self.shot_generation_template
                    else:
                        # 如果是字符串，则创建模板
                        current_template = ChatPromptTemplate.from_template(self.shot_generation_template)

                    # 使用LLM生成
                    chain = current_template | self.llm
                    response = chain.invoke(prompt_input)
                    debug(f"原始LLM响应: {response[:200]}...")  # 记录部分原始响应用于调试
                    # 确保获取到content
                    if hasattr(response, 'content'):
                        response = response.content

                    # 解析响应
                    from hengline.tools import parse_json_response
                    shot_data = parse_json_response(response)
                    debug(f"成功解析LLM响应，生成了包含{len(shot_data)}个字段的分镜数据")
                except Exception as jde:
                    error(f"LLM响应JSON解析失败: {str(jde)}")
                    # 回退到规则生成
                    debug("JSON解析失败，回退到规则生成分镜")
                    shot_data = self._generate_shot_with_rules(segment, continuity_constraints, scene_context, style, shot_id)
                except AuthenticationError | ConnectionError | APIError as llm_e:
                    # 检查是否是API密钥错误
                    if "API key" in str(llm_e) or "401" in str(llm_e):
                        error(f"LLM生成分镜失败：API密钥错误或权限不足: {str(llm_e)}")
                    else:
                        error(f"LLM生成分镜失败: {str(llm_e)}")
                    # 回退到规则生成
                    debug("回退到规则生成分镜")
                    shot_data = self._generate_shot_with_rules(segment, continuity_constraints, scene_context, style, shot_id)
            else:
                # 如果没有LLM，使用规则生成
                debug("使用规则生成分镜")
                shot_data = self._generate_shot_with_rules(segment, continuity_constraints, scene_context, style, shot_id)

            # 计算时间信息
            start_time = (shot_id - 1) * 5
            end_time = shot_id * 5
            duration = 5

            # 构建完整的分镜对象，确保包含所有必要字段
            shot = {
                # 基础信息
                "shot_id": str(shot_id),  # 确保是字符串类型
                "time_range_sec": [(shot_id - 1) * 5, shot_id * 5],
                "scene_context": scene_context,
                "start_time": start_time,  # 添加必要的时间字段
                "end_time": end_time,
                "duration": duration,

                # 描述字段
                "ai_prompt": shot_data.get("ai_prompt", "Default AI prompt"),
                "description": shot_data.get("description", "默认描述"),

                # 相机信息
                "camera": shot_data.get("camera", {
                    "shot_type": "medium shot",
                    "angle": "eye-level",
                    "movement": "static"
                }),
                "camera_angle": "medium_shot",  # 添加必要的相机角度字段

                # 角色相关
                "characters_in_frame": self._extract_characters_in_frame(shot_data),
                "characters": self._extract_characters_in_frame(shot_data),  # 添加必要的角色字段
                "dialogue": "",  # 添加对话字段

                # 状态信息
                "initial_state": shot_data.get("initial_state", []),
                "final_state": shot_data.get("final_state", []),
                "continuity_anchor": self._generate_continuity_anchor(shot_data),
                "continuity_anchors": [],  # 添加必要的连续性锚点字段
                # 确保final_continuity_state字段为字典类型
                "final_continuity_state": {}
            }

            debug(f"分镜生成完成: {shot.get('description', '')[:100]}...")
            return shot

        except Exception as e:
            print_log_exception()
            error(f"分镜生成失败: {str(e)}")
            # 返回默认分镜
            default_shot = self._get_default_shot(segment, scene_context, style, shot_id)
            # 确保默认分镜中也包含final_continuity_state字段
            if "final_continuity_state" not in default_shot:
                default_shot["final_continuity_state"] = {}
            return default_shot

    def _format_actions_text(self, actions: List[Dict[str, Any]]) -> str:
        """格式化动作文本，确保动作序列合理"""
        lines = []
        # 确保动作按顺序排序
        sorted_actions = sorted(actions, key=lambda x: x.get("order", 0))

        # 按角色分组，优先处理主要角色
        main_actions = []
        phone_actions = []

        for action in sorted_actions:
            if "dialogue" in action:
                if "电话那头" in action.get('character', '') or "off-screen" in action.get('character', ''):
                    phone_actions.append(action)
                else:
                    main_actions.append(action)
            else:
                if "电话那头" in action.get('character', '') or "off-screen" in action.get('character', ''):
                    phone_actions.append(action)
                else:
                    main_actions.append(action)

        # 先添加主要角色的动作
        for idx, action in enumerate(main_actions, 1):
            if "dialogue" in action:
                lines.append(f"{idx}. {action['character']}（{action['emotion']}）：{action['dialogue']}")
            else:
                lines.append(f"{idx}. {action['character']} {action.get('action', '')}（{action.get('emotion', '平静')}）")

        # 再添加电话那头角色的动作
        for idx, action in enumerate(phone_actions, len(main_actions) + 1):
            if "dialogue" in action:
                lines.append(f"{idx}. {action['character']}（{action['emotion']}）[声音]: {action['dialogue']}")
            else:
                lines.append(f"{idx}. {action['character']} {action.get('action', '')}[声音]")

        return "\n".join(lines)

    def _format_continuity_constraints(self, constraints: Dict[str, Any]) -> str:
        """
        格式化连续性约束
        
        Args:
            constraints: 连续性约束字典
            
        Returns:
            格式化后的约束文本
        """
        lines = []

        # 添加角色约束
        characters = constraints.get("characters", {})

        # 确保characters是字典类型
        if isinstance(characters, dict):
            # 先处理主要角色（不在电话那头的）
            main_characters = {k: v for k, v in characters.items() if "电话那头" not in k and "off-screen" not in k}
            # 处理电话那头的角色
            phone_characters = {k: v for k, v in characters.items() if "电话那头" in k or "off-screen" in k}

            # 添加主要角色约束
            for character_name, char_constraints in main_characters.items():
                lines.append(f"角色 {character_name} 的约束：")
                for key, value in char_constraints.items():
                    if key.startswith("must_start_with_"):
                        constraint_name = key.replace("must_start_with_", "")
                        lines.append(f"  - 必须以 {constraint_name}: {value} 开始")
                    elif key == "character_description":
                        lines.append(f"  - 描述: {value}")

            # 添加电话那头角色的特殊约束
            for character_name, char_constraints in phone_characters.items():
                lines.append(f"角色 {character_name} 的约束（不在画面中）：")
                lines.append(f"  - 位置: off-screen")
                lines.append(f"  - 仅通过声音参与场景")
        elif isinstance(characters, list):
            # 如果是列表，进行适当处理
            debug(f"连续性约束中的characters是列表类型，包含{len(characters)}个元素")
            for idx, character_info in enumerate(characters):
                character_name = character_info.get("character_name", f"角色{idx + 1}")
                lines.append(f"角色 {character_name} 的约束：")
                if isinstance(character_info, dict):
                    for key, value in character_info.items():
                        if key != "character_name":
                            lines.append(f"  - {key}: {value}")
        else:
            debug(f"连续性约束中的characters类型未知: {type(characters).__name__}")

        # 添加相机约束
        if "camera" in constraints:
            lines.append("相机约束：")
            camera_constraints = constraints["camera"]
            # 确保camera_constraints是字典类型
            if isinstance(camera_constraints, dict):
                if "recommended_shot_type" in camera_constraints:
                    lines.append(f"  - 推荐镜头类型: {camera_constraints['recommended_shot_type']}")
                if "recommended_angle" in camera_constraints:
                    lines.append(f"  - 推荐角度: {camera_constraints['recommended_angle']}")
            elif isinstance(camera_constraints, list):
                debug(f"连续性约束中的camera是列表类型，包含{len(camera_constraints)}个元素")
            else:
                debug(f"连续性约束中的camera类型未知: {type(camera_constraints).__name__}")

        return "\n".join(lines)

    def _generate_shot_with_rules(self,
                                  segment: Dict[str, Any],
                                  continuity_constraints: Dict[str, Any],
                                  scene_context: Dict[str, Any],
                                  style: str,
                                  shot_id: int) -> Dict[str, Any]:
        """使用规则生成分镜（当LLM不可用时）"""
        actions = segment.get("actions", [])
        all_characters = list(continuity_constraints.get("characters", {}).keys())

        # 分离主要角色和电话那头的角色
        main_characters = [c for c in all_characters if "电话那头" not in c and "off-screen" not in c]
        phone_characters = [c for c in all_characters if "电话那头" in c or "off-screen" in c]

        # 生成中文描述
        description = f"场景：{scene_context.get('location', '')}，{scene_context.get('time', '')}，{scene_context.get('atmosphere', '')}。"

        # 按角色分组动作
        main_actions = [a for a in actions if "character" in a and a["character"] in main_characters]
        phone_actions = [a for a in actions if "character" in a and a["character"] in phone_characters]

        # 先添加主要角色的动作描述
        for action in main_actions:
            if "dialogue" in action:
                description += f"{action['character']}（{action['emotion']}）说：{action['dialogue']}。"
            else:
                description += f"{action['character']} {action.get('action', '')}。"

        # 再添加电话那头角色的动作描述
        for action in phone_actions:
            if "dialogue" in action:
                description += f"{action['character']}（画外音）说：{action['dialogue']}。"
            else:
                description += f"{action['character']}（画外音）{action.get('action', '')}。"

        # 生成英文提示词
        style_prefix = self._get_style_prefix(style)
        ai_prompt = f"{style_prefix} A scene in {scene_context.get('location', 'a place')} at {scene_context.get('time', 'some time')}. "

        # 添加主要角色的英文描述
        for action in main_actions:
            character = action['character']
            if "dialogue" in action:
                emotion = action.get('emotion', 'neutral')
                dialogue = action['dialogue']
                ai_prompt += f"A person named {character} says '{dialogue}' with {emotion} expression. "
            else:
                action_desc = action.get('action', 'does something')
                emotion = action.get('emotion', 'neutral')
                ai_prompt += f"A person named {character} {action_desc} with {emotion} expression. "

        # 添加电话那头角色的英文描述（标记为off-screen）
        for action in phone_actions:
            character = action['character']
            if "dialogue" in action:
                emotion = action.get('emotion', 'neutral')
                dialogue = action['dialogue']
                ai_prompt += f"{character} (off-screen voice) says '{dialogue}' with {emotion} expression. "
            else:
                action_desc = action.get('action', 'does something')
                ai_prompt += f"{character} (off-screen) {action_desc}. "

        # 生成相机信息
        camera = {
            "shot_type": "medium shot",
            "angle": "eye-level",
            "movement": "static"
        }

        # 生成初始状态和结束状态
        initial_state = []
        final_state = []

        # 处理主要角色的状态
        for character in main_characters:
            char_constraints = continuity_constraints["characters"][character]

            initial_state.append({
                "character_name": character,
                "pose": char_constraints.get("must_start_with_pose", "standing"),
                "position": char_constraints.get("must_start_with_position", "center"),
                "holding": char_constraints.get("must_start_with_holding", "nothing"),
                "emotion": char_constraints.get("must_start_with_emotion", "neutral"),
                "appearance": f"A person named {character}"
            })

            # 根据动作更新结束状态
            final_pose = char_constraints.get("must_start_with_pose", "standing")
            final_position = char_constraints.get("must_start_with_position", "center")
            final_emotion = char_constraints.get("must_start_with_emotion", "neutral")
            final_holding = char_constraints.get("must_start_with_holding", "nothing")

            # 分析角色动作来更新状态
            character_actions = [a for a in actions if "character" in a and a["character"] == character]
            for action in character_actions:
                action_text = action.get('action', '')
                if "dialogue" in action:
                    final_emotion = action.get('emotion', final_emotion)
                if "坐下" in action_text or "坐" == action_text:
                    final_pose = "sitting"
                elif "站" in action_text or "站立" == action_text:
                    final_pose = "standing"
                elif "走" in action_text or "移动" == action_text:
                    # 简单的位置切换逻辑
                    if final_position == "left":
                        final_position = "center"
                    elif final_position == "center":
                        final_position = "right"
                elif "打电话" in action_text or "接电话" in action_text:
                    final_holding = "phone"
                    final_emotion = "serious"
                elif "看" in action_text or "注视" in action_text:
                    # 视线方向根据描述调整
                    final_emotion = "focused"

            final_state.append({
                "character_name": character,
                "pose": final_pose,
                "position": final_position,
                "gaze_direction": "forward" if "看" not in ''.join([a.get('action', '') for a in character_actions]) else "side",
                "emotion": final_emotion,
                "holding": final_holding
            })

        # 处理电话那头角色的特殊状态
        for character in phone_characters:
            # 电话那头的角色状态保持一致且位置为off-screen
            phone_state = {
                "character_name": character,
                "pose": "off-screen",
                "position": "off-screen",
                "holding": "unknown",
                "emotion": "unknown",
                "appearance": f"{character} (off-screen)"
            }

            initial_state.append(phone_state)

            # 结束状态基本相同，只可能更新情绪
            final_phone_state = phone_state.copy()
            character_actions = [a for a in actions if "character" in a and a["character"] == character]
            if character_actions and "dialogue" in character_actions[0]:
                final_phone_state["emotion"] = character_actions[0].get('emotion', 'unknown')

            final_state.append(final_phone_state)

        return {
            "description": description,
            "ai_prompt": ai_prompt,
            "camera": camera,
            "initial_state": initial_state,
            "final_state": final_state
        }

    def _get_style_prefix(self, style: str) -> str:
        """获取风格前缀"""
        style_mapping = {
            "realistic": "Realistic, high detail, natural lighting,",
            "anime": "Anime style, colorful, expressive, 2D animation,",
            "cinematic": "Cinematic, professional lighting, shallow depth of field,",
            "cartoon": "Cartoon style, exaggerated features, vibrant colors,"
        }
        return style_mapping.get(style, "Detailed, realistic,")

    def _extract_characters_in_frame(self, shot_data: Dict[str, Any]) -> List[str]:
        """提取画面中的角色，排除电话那头和off-screen的角色"""
        characters = set()

        # 从初始状态提取，但排除电话那头和off-screen的角色
        for state in shot_data.get("initial_state", []):
            if "character_name" in state:
                character_name = state["character_name"]
                # 检查角色是否在画面中（不是电话那头或off-screen）
                is_in_frame = True
                if "position" in state and state["position"] == "off-screen":
                    is_in_frame = False
                if "电话那头" in character_name or "off-screen" in character_name:
                    is_in_frame = False

                if is_in_frame:
                    characters.add(character_name)

        # 如果没有提取到角色，尝试从动作中提取
        if not characters and "scene_context" in shot_data:
            # 这是一个回退机制
            return ["默认角色"]

        return list(characters)

    def _generate_continuity_anchor(self, shot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成连续性锚点，确保锚点数据完整一致"""
        anchors = []

        # 获取所有角色（从final_state和initial_state）
        all_characters = set()
        for state in shot_data.get("final_state", []):
            if "character_name" in state:
                all_characters.add(state["character_name"])
        for state in shot_data.get("initial_state", []):
            if "character_name" in state:
                all_characters.add(state["character_name"])

        # 从结束状态生成锚点
        for character_name in all_characters:
            # 查找该角色的final_state
            character_state = None
            for state in shot_data.get("final_state", []):
                if state.get("character_name") == character_name:
                    character_state = state
                    break

            # 如果没有找到，使用initial_state
            if not character_state:
                for state in shot_data.get("initial_state", []):
                    if state.get("character_name") == character_name:
                        character_state = state
                        break

            # 创建锚点
            anchor = {
                "character_name": character_name,
                "pose": "unknown",
                "position": "unknown",
                "gaze_direction": "unknown",
                "emotion": "unknown",
                "holding": "unknown"
            }

            # 如果找到状态，更新锚点
            if character_state:
                anchor.update({
                    "pose": character_state.get("pose", "unknown"),
                    "position": character_state.get("position", "unknown"),
                    "gaze_direction": character_state.get("gaze_direction", "unknown"),
                    "emotion": character_state.get("emotion", "unknown"),
                    "holding": character_state.get("holding", "unknown")
                })

                # 确保电话那头角色的位置正确
                if "电话那头" in character_name or "off-screen" in character_name:
                    anchor["position"] = "off-screen"
                    anchor["pose"] = "off-screen"

            anchors.append(anchor)

        return anchors

    def _get_default_shot(self,
                          segment: Dict[str, Any],
                          scene_context: Dict[str, Any],
                          style: str,
                          shot_id: int) -> Dict[str, Any]:
        """获取默认分镜（当生成失败时），确保包含所有必要字段以满足Pydantic验证"""
        # 从segment和scene_context中提取有用信息
        actions = segment.get("actions", [])
        location = scene_context.get("location", "未知位置")
        time = scene_context.get("time", "未知时间")
        atmosphere = scene_context.get("atmosphere", "未知氛围")

        # 提取角色信息
        characters = []
        dialogue = ""
        for action in actions:
            character_name = action.get("character", "角色")
            if character_name not in characters:
                characters.append(character_name)
            if "dialogue" in action:
                dialogue += f"{character_name}: {action['dialogue']}\n"

        if not characters:
            characters = ["默认角色"]

        # 计算时间信息
        start_time = (shot_id - 1) * 5
        end_time = shot_id * 5
        duration = end_time - start_time

        # 生成完整的默认分镜，确保包含所有Pydantic验证所需字段
        return {
            # 基础信息
            "shot_id": str(shot_id),  # 确保是字符串类型
            "time_range_sec": [(shot_id - 1) * 5, shot_id * 5],
            "scene_context": scene_context,
            "start_time": start_time,  # 添加必要的时间字段
            "end_time": end_time,
            "duration": duration,

            # 描述字段
            "ai_prompt": f"Default shot of {location} at {time}",
            "description": f"场景：{location}，{time}，{atmosphere}。分镜生成失败，使用默认描述。",

            # 相机信息
            "camera": {
                "shot_type": "medium shot",
                "angle": "eye-level",
                "movement": "static"
            },
            "camera_angle": "medium_shot",  # 添加必要的相机角度字段

            # 角色相关
            "characters_in_frame": characters,
            "characters": characters,  # 添加必要的角色字段
            "dialogue": dialogue.strip(),  # 添加对话字段

            # 状态信息
            "initial_state": [],
            "final_state": [],
            "continuity_anchor": [],
            "continuity_anchors": [],  # 添加必要的连续性锚点字段
            "final_continuity_state": {}  # 确保包含final_continuity_state字段，为字典类型
        }
