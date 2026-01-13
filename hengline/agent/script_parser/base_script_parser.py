# -*- coding: utf-8 -*-
"""
@FileName: script_parser_agent.py
@Description: LLM 剧本解析基类，包含复杂度评估和路由决策逻辑
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from abc import abstractmethod
from typing import Dict, Any, List

from hengline.agent.base_agent import BaseAgent
from hengline.agent.script_parser.script_parser_models import UnifiedScript, Scene, Character, Dialogue, Action, Meta, Relationship
from hengline.agent.workflow.workflow_models import ScriptType
from hengline.tools.script_assessor_tool import ComplexityAssessor


class ScriptParser(BaseAgent):
    """优化版剧本解析智能体"""

    def __post_init__(self):
        """
        初始化剧本解析智能体
        """
        self.complexity_assessor = ComplexityAssessor()

    @abstractmethod
    def process(self, script_text: Any, script_format: ScriptType | UnifiedScript) -> UnifiedScript:
        """处理输入数据（子类实现）"""
        raise NotImplementedError("子类必须实现process方法")

    def _create_unified_script(self, original_text: str,
                               format_type: ScriptType,
                               parsed_data: Dict[str, Any]) -> UnifiedScript:
        """从解析数据创建UnifiedScript对象"""

        # 创建各个元素列表
        scenes = [Scene(**scene) for scene in parsed_data.get("scenes", [])]
        characters = [Character(**char) for char in parsed_data.get("characters", [])]
        dialogues = [Dialogue(**dialogue) for dialogue in parsed_data.get("dialogues", [])]
        actions = [Action(**action) for action in parsed_data.get("actions", [])]
        relationships = [Relationship(**relationship) for relationship in parsed_data.get("relationships", [])]
        # props = [Prop(**prop) for prop in parsed_data.get("props", [])]

        # calculate_confidence = self.calculate_confidence(scenes, characters, dialogues, actions)
        # info(f"AI 解析完成！评分: {calculate_confidence.get('overall'):.2f}/1.0")

        return UnifiedScript(
            _meta=parsed_data["_meta"] if "_meta" in parsed_data else Meta(),
            scenes=scenes,
            characters=characters,
            dialogues=dialogues,
            actions=actions,
            relationships=relationships,
            original_text=original_text,
            warnings=[],
            completeness_score=0,
            parsing_confidence={}
        )

    def calculate_confidence(self, scenes: List[Scene], characters: List[Character], dialogues: List[Dialogue], actions: List[Action]) -> Dict:
        """
        计算解析置信度

        """
        if not scenes:
            return {
                "overall": 0.3,
                "scene_detection": 0.1,
                "character_recognition": 0.3,
                "dialogue_extraction": 0.3,
                "action_extraction": 0.3,
                "note": "未检测到场景"
            }

        # 权重配置
        weights = {
            "scene_detection": 0.3,
            "character_recognition": 0.3,
            "dialogue_extraction": 0.2,
            "action_extraction": 0.2
        }

        # 1. 场景检测置信度
        scene_count = len(scenes)
        scene_confidence = min(1.0, scene_count / 10.0)  # 最多10个场景得满分

        # 2. 角色识别置信度
        character_count = len(characters)
        # 检查角色信息完整性
        char_info_score = sum(
            1 for char in characters
            if char.gender and char.gender != "未知"
        ) / max(character_count, 1)

        character_confidence = min(1.0, character_count / 15.0) * 0.7 + char_info_score * 0.3

        # 3. 对话提取置信度
        dialogue_count = len(dialogues)
        # 检查对话信息完整性
        dialogue_info_score = sum(
            1 for dialogue in dialogues
            if dialogue.speaker and dialogue.speaker != "未知"
        ) / max(dialogue_count, 1)

        dialogue_confidence = min(1.0, dialogue_count / 20.0) * 0.6 + dialogue_info_score * 0.4

        # 4. 动作提取置信度
        action_count = len(actions)
        # 检查动作信息完整性
        action_info_score = sum(
            1 for action in actions
            if action.actor and action.actor != "未知"
        ) / max(action_count, 1)

        action_confidence = min(1.0, action_count / 30.0) * 0.5 + action_info_score * 0.5

        # 5. 综合置信度
        overall_confidence = (
                scene_confidence * weights["scene_detection"] +
                character_confidence * weights["character_recognition"] +
                dialogue_confidence * weights["dialogue_extraction"] +
                action_confidence * weights["action_extraction"]
        )

        return {
            "overall": round(overall_confidence, 2),
            "scene_detection": round(scene_confidence, 2),
            "character_recognition": round(character_confidence, 2),
            "dialogue_extraction": round(dialogue_confidence, 2),
            "action_extraction": round(action_confidence, 2),
            "metrics": {
                "scene_count": scene_count,
                "character_count": character_count,
                "dialogue_count": dialogue_count,
                "action_count": action_count
            }
        }
