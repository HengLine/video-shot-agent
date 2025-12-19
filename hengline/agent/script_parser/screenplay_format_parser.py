# -*- coding: utf-8 -*-
"""
@FileName: script_parser_agent.py
@Description: LLM 剧本解析功能，通过LLM 将中文剧本转换为结构化动作序列
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import json
from typing import Dict, Any, Optional

from hengline.logger import debug
from hengline.prompts.prompts_manager import prompt_manager
from hengline.tools.json_parser_tool import parse_json_response
from .base_script_parser import ScriptParser
from .script_extractor.script_action_extractor import ScreenplayActionParser
from .script_extractor.script_character_extractor import ScreenplayCharacterParser
from .script_extractor.script_dialogue_extractor import ScreenplayDialogueParser
from .script_extractor.script_scene_segmenter import SceneSluglineParser


class ScreenplayFormatParser(ScriptParser):
    """
    标准剧本格式解析器

    专门处理好莱坞/专业剧本格式：
        - INT./EXT. 场景标题
        - 角色名全大写
        - 对话缩进
        - 动作描述
        - 转场标记

    支持本地规则解析和LLM增强解析
    """

    def __init__(self, llm_client):
        # 初始化各个处理器
        super().__init__(llm_client)
        self.scene_segmenter = SceneSluglineParser()
        self.character_extractor = ScreenplayCharacterParser()
        self.dialogue_extractor = ScreenplayDialogueParser()
        self.action_extractor = ScreenplayActionParser()

        self.action_counter = 0  # 用于生成唯一动作ID

    def _extract_with_llm(self, script_text: str) -> Optional[Dict[str, Any]]:
        """
        使用LLM直接解析剧本
        
        Args:
            script_text: 原始剧本文本
            
        Returns:
            解析结果或None
        """
        parser_prompt = prompt_manager.get_script_parser_prompt("screenplay_format_parser")

        # 构建完整提示词
        prompt = parser_prompt.format(script_text=script_text)

        # 调用LLM
        response = self._call_llm_with_retry(prompt)
        debug(f"LLM直接解析结果:\n {response}")

        # 解析LLM响应
        parsed_result = parse_json_response(response)

        return parsed_result

    def _parse_ai_response(self, response: str) -> Dict:
        """解析AI响应"""
        try:
            result = json.loads(response)

            # 验证必要字段
            required_fields = ["scenes", "characters", "dialogues", "actions", "metadata"]
            for field in required_fields:
                if field not in result:
                    result[field] = [] if field != "metadata" else {}

            # 确保每个场景有ID
            for i, scene in enumerate(result["scenes"]):
                if "scene_id" not in scene:
                    scene["scene_id"] = f"scene_{i + 1:03d}"

            # 确保对话和动作有scene_ref
            for dialogue in result["dialogues"]:
                if "scene_ref" not in dialogue and result["scenes"]:
                    dialogue["scene_ref"] = result["scenes"][0]["scene_id"]

            for action in result["actions"]:
                if "scene_ref" not in action and result["scenes"]:
                    action["scene_ref"] = result["scenes"][0]["scene_id"]

            return result

        except json.JSONDecodeError as e:
            print(f"AI响应JSON解析失败: {e}")
            raise Exception("无法解析AI响应")

    def _post_process_ai_result(self, ai_result: Dict) -> Dict:
        """后处理AI结果"""
        # 添加缺失的字段
        for scene in ai_result["scenes"]:
            if "characters" not in scene:
                scene["characters"] = []
            if "dialogue_ids" not in scene:
                scene["dialogue_ids"] = []
            if "action_ids" not in scene:
                scene["action_ids"] = []

        # 构建对话和动作ID映射
        for i, dialogue in enumerate(ai_result["dialogues"]):
            if "dialogue_id" not in dialogue:
                dialogue["dialogue_id"] = f"dialogue_{i + 1:03d}"

            # 添加到对应场景
            for scene in ai_result["scenes"]:
                if scene["scene_id"] == dialogue.get("scene_ref", ""):
                    scene["dialogue_ids"].append(dialogue["dialogue_id"])
                    break

        for i, action in enumerate(ai_result["actions"]):
            if "action_id" not in action:
                action["action_id"] = f"action_{i + 1:03d}"

            # 添加到对应场景
            for scene in ai_result["scenes"]:
                if scene["scene_id"] == action.get("scene_ref", ""):
                    scene["action_ids"].append(action["action_id"])
                    break

        return ai_result

    def extract_with_local(self, script_text: str) -> Optional[Dict[str, Any]]:
        """本地规则解析剧本"""
        # 1. 文本预处理
        cleaned_text = self.text_processor.preprocess_text(script_text)

        lines = cleaned_text.split('\n')

        result = {
            "scenes": [],  # 与自然语言解析器一致
            "characters": [],  # 与自然语言解析器一致
            "dialogues": [],  # 与自然语言解析器一致
            "actions": [],  # 与自然语言解析器一致
        }

        current_scene = None
        current_character = None
        dialogue_counter = 0
        action_counter = 0

        for line_num, line in enumerate(lines):
            line = line.rstrip()

            # 1. 解析场景标题
            scene_match = self.scene_segmenter.parse_slugline(line)
            if scene_match:
                if current_scene:
                    result["scenes"].append(current_scene)

                # 创建新场景（格式与自然语言解析器一致）
                current_scene = {
                    "scene_id": f"scene_{len(result['scenes']) + 1:03d}",
                    "description": "",
                    "location": scene_match["location"],
                    "time_of_day": scene_match["time_of_day"],
                    "mood": "",
                    "characters": [],
                    "dialogue_ids": [],
                    "action_ids": []
                }
                current_character = None
                continue

            # 2. 解析角色名
            character_match = self.character_extractor.parse_character_line(line, current_scene)
            if character_match:
                current_character = character_match["name"]

                # 添加到角色列表（格式与自然语言解析器一致）
                char_exists = any(char["name"] == current_character for char in result["characters"])
                if not char_exists:
                    result["characters"].append({
                        "name": current_character,
                        "age": None,
                        "gender": character_match.get("gender", "未知"),
                        "role_hint": character_match.get("role_hint", "角色"),
                        "description": ""
                    })

                # 添加到当前场景
                if current_scene and current_character not in current_scene["characters"]:
                    current_scene["characters"].append(current_character)
                continue

            # 3. 解析对话
            if current_character and self.dialogue_extractor.is_dialogue_line(line):
                dialogue_counter += 1
                dialogue_data = {
                    "dialogue_id": f"dialogue_{dialogue_counter:03d}",
                    "speaker": current_character,
                    "text": self.dialogue_extractor.clean_dialogue_text(line.strip()),
                    "emotion": self.dialogue_extractor.infer_emotion(line),
                    "scene_ref": current_scene["scene_id"] if current_scene else ""
                }

                result["dialogues"].append(dialogue_data)

                # 添加到当前场景
                if current_scene:
                    current_scene["dialogue_ids"].append(dialogue_data["dialogue_id"])
                continue

            # 4. 解析动作描述
            if line.strip() and current_scene and not current_character:
                action_counter += 1
                action_data = {
                    "action_id": f"action_{action_counter:03d}",
                    "type": "action",
                    "actor": "某人",
                    "description": line.strip(),
                    "intensity": 2,
                    "scene_ref": current_scene["scene_id"]
                }

                result["actions"].append(action_data)
                current_scene["action_ids"].append(action_data["action_id"])

                # 添加到场景描述
                if current_scene["description"]:
                    current_scene["description"] += " " + line.strip()
                else:
                    current_scene["description"] = line.strip()

        # 添加最后一个场景
        if current_scene:
            result["scenes"].append(current_scene)

        # 添加必需的描述字段（与自然语言解析器一致）
        result["descriptions"] = [
            scene.get("description", "")
            for scene in result["scenes"]
            if scene.get("description")
        ]

        # 添加解析置信度（与自然语言解析器一致）
        result["parsing_confidence"] = self.calculate_confidence(result["scenes"], result["characters"], result["dialogues"], result["actions"])

        return result
