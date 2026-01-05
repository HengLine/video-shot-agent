# -*- coding: utf-8 -*-
"""
@FileName: script_parser_agent.py
@Description: LLM 剧本解析基类，包含复杂度评估和路由决策逻辑
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import hashlib
import re
import time
from typing import Dict, Any, Optional, List

from hengline.agent.workflow_models import ScriptType, ParserType
from hengline.logger import error
from hengline.tools.script_assessor_tool import ComplexityAssessor
from utils.log_utils import print_log_exception
from .script_extractor.script_action_extractor import action_extractor
from .script_extractor.script_character_extractor import character_extractor
from .script_extractor.script_emotion_extractor import emotion_extractor
from .script_parser_model import UnifiedScript, Action, Dialogue, Character, Scene
from .script_extractor.script_text_processor import TextProcessor


class ScriptParser:
    """优化版剧本解析智能体"""

    def __init__(self, llm_client):
        """
        初始化剧本解析智能体

        Args:
            llm_client: 语言模型实例（推荐GPT-4o）
        """
        self.llm = llm_client
        self.complexity_assessor = ComplexityAssessor()
        """
            设计原则：
            1. 简单格式 → 本地解析（速度快、成本低）
            2. 复杂/模糊格式 → AI解析（准确性高）
            3. 可配置路由策略
        """
        self.routing_rules = {
            "max_lines_for_local": 20,  # 少于50行用本地
            "complexity_threshold": 0.7,  # 复杂度阈值
            "confidence_threshold": 0.8,  # 置信度阈值
            "fallback_to_ai": True,  # 本地失败时回退到AI
            "preferred_parser": ParserType.LLM_PARSER  # 优先解析器
        }

        self.parser_config = {
            "min_scene_length": 10,  # 最小场景长度（字符）
            "max_scene_length": 1000,  # 最大场景长度
            "confidence_threshold": 0.6,  # 置信度阈值
            "enable_description_extraction": True,
            "enable_character_deduplication": True,
            "strict_mode": False  # 严格模式（更保守的解析）
        }
        # 中文处理工具
        self.text_processor = TextProcessor()

    def convert_script_format(self, script_text: str, script_type: ScriptType) -> tuple[UnifiedScript, float] | None:
        """
        优化版剧本解析函数
        将整段中文剧本转换为结构化动作序列
        
        Args:
            script_text: 原始剧本文本

        Returns:
            结构化的剧本动作序列
        """
        should_use_ai = False
        try:
            # 1. 复杂度评估
            complexity_score = self.complexity_assessor.assess_complexity(script_text)

            # 2. 路由决策
            should_use_ai = self._should_use_ai(script_text, complexity_score, script_type)

            # 初始化结果结构
            if should_use_ai and should_use_ai == True:
                # 使用AI解析
                raw_data = self._extract_with_llm(script_text)
            else:
                # 使用本地解析
                raw_data = self._extract_with_local(script_text)

            # 转换为统一格式
            return self._convert_to_unified_format(raw_data, script_type), complexity_score

        except Exception as e:
            print_log_exception()
            error(f"剧本解析失败: {str(e)}")
            return UnifiedScript(
                script_type=script_type.value,
                script_hash=hashlib.md5(script_text.encode()).hexdigest()[:16],
                parser_type=ParserType.LLM_PARSER.value if should_use_ai else ParserType.RULE_PARSER.value,
                scenes=[],
                characters=[],
                dialogues=[],
                actions=[],
                descriptions=[],
                parsing_confidence={
                    "character_parsing": 0.0,
                    "dialogue_parsing": 0.0,
                    "action_parsing": 0.0
                }
            ), 0.0

    def _extract_with_llm(self, script_text: str) -> Optional[Dict[str, Any]]:
        """ 需要实现将剧本解析结果转换为JSON字符串的功能
        """
        pass

    def _extract_with_local(self, script_text: str) -> Optional[Dict[str, Any]]:
        """ 本地解析剧本"""
        pass

    def _should_use_ai(self, text: str, complexity: float, script_type: ScriptType) -> bool:
        """
        决策是否使用AI解析器
        """
        # 优先使用AI解析器
        if self.routing_rules["preferred_parser"] == ParserType.LLM_PARSER:
            return True

        # 规则1：行数太多（可能复杂）
        line_count = text.count('\n') + 1
        if line_count > self.routing_rules["max_lines_for_local"]:
            return True

        # 规则2：复杂度超过阈值
        if complexity > self.routing_rules["complexity_threshold"]:
            return True

        # 规则3：特定格式强制使用AI
        if script_type == ScriptType.NATURAL_LANGUAGE and complexity > 0.2:
            # 自然语言特别适合AI理解
            return True

        if script_type == ScriptType.AI_STORYBOARD:
            # AI分镜剧本，已经是结构化格式
            return False

        # 规则4：尝试本地解析，如果置信度低则用AI
        try:
            local_result = self._extract_with_local(text)
            confidence = local_result.get('parsing_confidence', {}).get('overall', 0.0) if local_result else 0.0
            if float(confidence) < float(self.routing_rules["confidence_threshold"]):
                return True

        except Exception as e:
            # 本地解析失败，回退到AI
            print_log_exception()
            error(f"本地解析失败，回退到AI: {e}")
            return True

        return False

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> Any | None:
        """调用LLM，支持重试"""
        for attempt in range(max_retries):
            try:
                # response = self.llm.chat_complete(
                #     messages=[
                #         {"role": "system", "content": "你是一个专业的影视剧本解析分镜师，精通标准剧本格式，输出严格的JSON格式。"},
                #         {"role": "user", "content": prompt}
                #     ],
                #     temperature=0.1,
                #     response_format={"type": "json_object"}
                # )
                response = self.llm.invoke(prompt)
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"LLM调用失败: {e}")
                time.sleep(1)

    def _convert_to_unified_format(self, raw_data: Dict, script_type: ScriptType) -> UnifiedScript:
        """
        将原始解析数据转换为统一格式
        """
        # 构建场景
        scenes = []
        for i, raw_scene in enumerate(raw_data.get("scenes", [])):
            scene = Scene(
                scene_id=f"scene_{i + 1:03d}",
                order_index=i,
                location=raw_scene.get("location", "未知地点"),
                time_of_day=raw_scene.get("time_of_day", ""),
                mood=emotion_extractor.infer_mood(raw_scene),
                summary=raw_scene.get("description", ""),
                character_refs=raw_scene.get("characters", []),
                dialogue_refs=raw_scene.get("dialogue_ids", []),
                action_refs=raw_scene.get("action_ids", [])
            )
            scenes.append(scene)

        # 构建角色
        characters = []
        for raw_char in raw_data.get("characters", []):
            name = raw_char.get("name", "")
            char = Character(
                name=name,
                age=character_extractor.infer_age(name, raw_char),
                gender=character_extractor.infer_gender(name, raw_char),
                role_hint=raw_char.get("role", ""),
                description=raw_char.get("description", "")
            )
            characters.append(char)

        # 构建对话（不估算时长！）
        dialogues = []
        for i, raw_dialogue in enumerate(raw_data.get("dialogues", [])):
            dialogue = Dialogue(
                dialogue_id=f"dialogue_{i + 1:03d}",
                speaker=raw_dialogue.get("speaker", ""),
                text=raw_dialogue.get("text", ""),
                emotion=emotion_extractor.detect_emotion(raw_dialogue.get("text", "")),
                parenthetical=raw_dialogue.get("parenthetical", ""),
                scene_ref=raw_dialogue.get("scene_ref", "")
            )
            dialogues.append(dialogue)

        # 构建动作（不估算时长！）
        actions = []
        for i, raw_action in enumerate(raw_data.get("actions", [])):
            category, action_type = action_extractor.get_action_category_type(raw_action)
            action = Action(
                action_id=f"action_{i + 1:03d}",
                type=action_type,
                actor=raw_action.get("actor", ""),
                target=raw_action.get("target", ""),
                description=raw_action.get("description", ""),
                intensity=action_extractor.analyze_intensity(raw_action),
                scene_ref=raw_action.get("scene_ref", ""),
                category=category
            )
            actions.append(action)

        return UnifiedScript(
            script_type=script_type.value,
            script_hash=hashlib.md5(str(raw_data).encode()).hexdigest()[:16],
            parser_type=raw_data.get("parser_type", ParserType.LLM_PARSER.value),
            scenes=scenes,
            characters=characters,
            dialogues=dialogues,
            actions=actions,
            descriptions=raw_data.get("descriptions", []),
            parsing_confidence={
                "character_parsing": raw_data.get("confidence", {}).get("characters", 0.8),
                "dialogue_parsing": raw_data.get("confidence", {}).get("dialogues", 0.8),
                "action_parsing": raw_data.get("confidence", {}).get("actions", 0.7)
            }
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


class TypeDetector:
    """剧本类型检测器"""

    def detect(self, text: str) -> ScriptType:
        """检测剧本类型"""

        # 检测AI分镜剧本格式
        if self._is_ai_storyboard(text):
            return ScriptType.AI_STORYBOARD

        # 检测标准剧本格式
        if self._is_screenplay_format(text):
            return ScriptType.SCREENPLAY_FORMAT

        # 检测结构化场景格式
        if self._is_structured_scene(text):
            return ScriptType.STRUCTURED_SCENE

        # 默认为自然语言
        return ScriptType.NATURAL_LANGUAGE

    def _is_ai_storyboard(self, text: str) -> bool:
        """检测AI分镜剧本"""
        patterns = [
            r'^\s*\[\d+-\d+\](秒|s)?\s*[\n\r]',  # [0-5秒]
            r'画面[：:]\s*.+[\n\r]声音[：:]',  # 画面: ... 声音:
            r'镜头[：:]\s*.+[\n\r]动作[：:]',  # 镜头: ... 动作:
            r'^\s*【\d+-\d+秒】',  # 【0-5秒】
            r'^\s*\d+\.\s*\[\d+秒\]',  # 1. [5秒]
        ]
        return any(re.search(pattern, text, re.MULTILINE) for pattern in patterns)

    def _is_screenplay_format(self, text: str) -> bool:
        """检测标准剧本格式"""
        patterns = [
            r'^(INT\.|EXT\.|INT/EXT\.)\s+.+',  # INT. 或 EXT.
            r'^\s+[A-Z][A-Z\s]+\s*$[\n\r]^\s+.+',  # 角色名全大写
            r'^[\s]*[A-Z].+:$',  # 场景标题
            r'^FADE IN:',  # 淡入
            r'^CUT TO:',  # 切至
        ]
        lines = text.strip().split('\n')
        for pattern in patterns:
            for line in lines[:10]:  # 检查前10行
                if re.match(pattern, line.strip()):
                    return True
        return False

    def _is_structured_scene(self, text: str) -> bool:
        """
        检测是否为结构化分场格式
        """
        lines = text.strip().split('\n')

        # 检查结构化标记
        patterns = [
            r'^场景[：:]',
            r'^第[一二三四五六七八九十\d]+场',
            r'^地点[：:]',
            r'^时间[：:]',
            r'^人物[：:]',
            r'^角色[：:]',
            r'^内容[：:]',
            r'^##\s+场景',
            r'^【场景',
        ]

        # 检查前10行
        for i in range(min(10, len(lines))):
            line = lines[i].strip()
            for pattern in patterns:
                if re.match(pattern, line):
                    return True

        # 检查是否有明显的结构化分隔
        field_count = 0
        for line in lines[:20]:
            if re.search(r'[：:]\s*.+', line):
                field_count += 1

        return field_count >= 3  # 如果有3个以上的字段标记，认为是结构化格式
        # return any(re.search(pattern, text, re.MULTILINE) for pattern in patterns)
