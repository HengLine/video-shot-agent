# -*- coding: utf-8 -*-
"""
@FileName: script_parser_agent.py
@Description: 剧本解析智能体，将整段中文剧本转换为结构化动作序列
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from typing import Any, Dict

from hengline.logger import debug, error, warning
from .script_parser.ai_storyboard_parser import AIStoryboardParser
from .script_parser.base_script_parser import TypeDetector
from .script_parser.natural_language_parser import NaturalLanguageParser
from .script_parser.screenplay_format_parser import ScreenplayFormatParser
from .script_parser.script_parser_model import UnifiedScript
from .script_parser.structured_scene_parser import StructuredSceneParser
from .workflow_models import ScriptType
from ..tools.script_validation_tool import BasicScriptValidator


class ScriptParserAgent:
    """优化版剧本解析智能体"""

    def __init__(self, llm):
        """
        初始化剧本解析智能体
        
        Args:
            llm: 语言模型实例（推荐GPT-4o）
        """

        self.validator = BasicScriptValidator()
        self.type_detector = TypeDetector()

        self.parsers = {
            ScriptType.NATURAL_LANGUAGE: NaturalLanguageParser(llm),
            ScriptType.AI_STORYBOARD: AIStoryboardParser(llm),
            ScriptType.STRUCTURED_SCENE: StructuredSceneParser(llm),
            ScriptType.SCREENPLAY_FORMAT: ScreenplayFormatParser(llm)
        }

    def process(self, script_text: str) -> UnifiedScript | None:
        """
        优化版剧本解析函数
        将整段中文剧本转换为结构化动作序列
        
        Args:
            script_text: 原始剧本文本

        Returns:
            结构化的剧本动作序列
        """
        debug(f"开始解析剧本: {script_text[:100]}...")

        # 1. 类型检测
        script_type = self.type_detector.detect(script_text)

        # 2. 选择对应解析器
        script_parser = self.parsers[script_type]

        # 3. 执行解析
        unified_script, complexity_score = script_parser.convert_script_format(script_text, script_type)

        # 4. 基础验证
        is_valid, issues = self.validator.validate(unified_script)

        # 添加验证结果到元数据
        if not hasattr(unified_script, 'metadata'):
            unified_script.metadata = {}

        unified_script.metadata.update({
            "validation": {
                "is_valid": is_valid,
                "issue_count": len(issues),
                "issues_by_severity": {
                    "error": len([i for i in issues if i["severity"] == "error"]),
                    "warning": len([i for i in issues if i["severity"] == "warning"]),
                    "info": len([i for i in issues if i["severity"] == "info"])
                },
                "complexity_score": complexity_score
            }
        })

        # 7. 如果有严重错误，记录但继续处理
        if not is_valid:
            warning(f"警告: 发现验证错误，但继续处理")
            for issue in issues:
                if issue["severity"] == "error":
                    error(f"  错误: {issue['message']}")

        return unified_script

    def _validate_result(self, result: Dict) -> Dict:
        """
        验证和清理AI解析结果
        """
        validated = {
            "scenes": [],
            "characters": [],
            "dialogues": [],
            "actions": [],
            "parsing_confidence": result.get("parsing_confidence", {"overall": 0.5})
        }

        # 验证场景
        for scene in result.get("scenes", []):
            if isinstance(scene, dict):
                validated["scenes"].append({
                    "scene_id": scene.get("scene_id", ""),
                    "description": scene.get("description", ""),
                    "location": scene.get("location", "未知地点"),
                    "time_of_day": scene.get("time_of_day", "未知时间"),
                    "mood": scene.get("mood", ""),
                    "characters": scene.get("characters", []),
                    "content": scene.get("content", "")
                })

        # 验证角色
        for char in result.get("characters", []):
            if isinstance(char, dict) and char.get("name"):
                validated["characters"].append({
                    "name": char["name"],
                    "age": char.get("age", "未知"),
                    "gender": char.get("gender", "未知"),
                    "role_hint": char.get("role_hint", "角色"),
                    "description": char.get("description", "")
                })

        # 验证对话
        for dialogue in result.get("dialogues", []):
            if isinstance(dialogue, dict) and dialogue.get("speaker") and dialogue.get("text"):
                validated["dialogues"].append({
                    "speaker": dialogue["speaker"],
                    "text": dialogue["text"],
                    "emotion": dialogue.get("emotion", "平静"),
                    "scene_ref": dialogue.get("scene_ref", "")
                })

        # 验证动作
        for action in result.get("actions", []):
            if isinstance(action, dict) and action.get("description"):
                validated["actions"].append({
                    "type": action.get("type", "unknown"),
                    "actor": action.get("actor", "某人"),
                    "description": action["description"],
                    "intensity": action.get("intensity", 2),
                    "scene_ref": action.get("scene_ref", "")
                })

        return validated