# -*- coding: utf-8 -*-
"""
@FileName: script_parser_agent.py
@Description: LLM 剧本解析功能，结构化分场剧本
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import re
from typing import Dict, Any, Optional, List

from hengline.agent.workflow_models import ScriptType
from hengline.logger import debug
from hengline.prompts.prompts_manager import prompt_manager
from hengline.tools.json_parser_tool import parse_json_response
from .base_script_parser import ScriptParser
from .script_extractor.script_action_extractor import action_extractor
from .script_extractor.script_character_extractor import character_extractor
from .script_extractor.script_dialogue_extractor import dialogue_extractor
from .script_extractor.script_scene_segmenter import StructuredSceneSegmenter
from .script_parser_model import Character


class StructuredSceneParser(ScriptParser):
    """
        结构化分场剧本解析器

        支持两种模式：
        1. 本地解析：基于规则的快速解析
        2. AI解析：基于LLM的智能解析

        输出格式与自然语言解析器保持一致
        """

    def __init__(self, llm_client):
        # 初始化各个处理器
        super().__init__(llm_client)
        self.scene_segmenter = StructuredSceneSegmenter()

    def _extract_with_llm(self, script_text: str) -> Optional[Dict[str, Any]]:
        """
        使用LLM直接解析剧本
        
        Args:
            script_text: 原始剧本文本
            
        Returns:
            解析结果或None
        """
        parser_prompt = prompt_manager.get_script_parser_prompt("structured_scene_parser")

        # 1. 构建提示词
        prompt = parser_prompt.format(script_text=script_text)

        # 2. 调用LLM
        response = self._call_llm_with_retry(prompt)
        debug(f"{ScriptType.STRUCTURED_SCENE.value} 类型的 LLM 解析结果:\n {response}")

        # 3. 解析响应
        result = parse_json_response(response)

        # 4. 验证结果
        # validated_result = self._validate_result(result)

        return result

    def extract_with_local(self, script_text: str) -> Optional[Dict[str, Any]]:
        """本地规则解析剧本"""
        # 1. 文本预处理
        cleaned_text = self.text_processor.clean_text(script_text)

        # 1. 分割成结构化场景
        structured_scenes = self.scene_segmenter.segment(cleaned_text)

        # 2. 解析每个场景的结构化字段
        scenes_data = []
        all_characters = []
        all_dialogues = []
        all_actions = []

        for scene_idx, scene_text in enumerate(structured_scenes):
            # 2.1 解析结构化字段
            scene_base = self.scene_segmenter.parse(scene_text)

            # 2.2 构建场景基础信息
            scene_base.scene_id = f"scene_{scene_idx + 1:03d}"

            # 2.3 提取场景内容（如果有）
            if scene_base.summary:
                # 复用通用组件解析内容
                dialogues = dialogue_extractor.extract(scene_base.description, scene_base.scene_id)
                actions = action_extractor.extract(scene_base.description, scene_base.scene_id)

                # 更新ID列表
                scene_base.dialogue_refs = [d.dialogue_id for d in dialogues]
                scene_base.action_refs = [a.action_id for a in actions]

                all_dialogues.extend(dialogues)
                all_actions.extend(actions)

            # 2.4 处理角色信息
            if scene_base.character_refs:
                # 从结构化字段中提取角色
                structured_chars = self._extract_characters_from_field(
                    scene_base.character_refs
                )
                all_characters.extend(structured_chars)

            scenes_data.append(scene_base)

        # 3. 构建最终结果（与自然语言解析器格式一致）
        result = {
            "format_type": "structured_scene",
            "scenes": scenes_data,
            "characters": character_extractor.deduplicate_characters(all_characters),
            "dialogues": all_dialogues,
            "actions": all_actions,
            "descriptions": [s.description for s in scenes_data if s.description],
            "parsing_confidence": self._calculate_confidence(structured_scenes)
        }

        return result

    def _extract_characters_from_field(self, character_string: List[str]) -> List[Character]:
        """
        从结构化角色字段提取角色信息

        复用自然语言的角色提取器，但做适配
        """
        # 将角色列表字符串转换为类似自然语言的文本
        # 例如："张三、李四、王五" -> "角色有张三、李四和王五。"
        character_string = '、'.join(character_string)
        fake_text = f"角色有{character_string}。"

        # 复用角色提取器
        characters = character_extractor.extract(fake_text)

        return characters

    def _calculate_confidence(self, scenes: List[str]) -> Dict:
        """
        计算解析置信度

        基于结构化的完整度评估
        """
        if not scenes:
            return {"overall": 0.0}

        # 评估每个场景的结构化完整度
        scene_scores = []
        for scene_text in scenes:
            score = self._evaluate_scene_structure(scene_text)
            scene_scores.append(score)

        avg_score = sum(scene_scores) / len(scene_scores)

        return {
            "scene_detection": avg_score,
            "character_recognition": avg_score * 0.9,
            "dialogue_extraction": avg_score * 0.8,
            "action_extraction": avg_score * 0.7,
            "overall": avg_score
        }

    def _evaluate_scene_structure(self, scene_text: str) -> float:
        """
        评估场景结构化完整度
        """
        lines = scene_text.split('\n')

        # 计算结构化字段数量
        structured_field_count = 0
        for line in lines:
            if re.search(r'[^：:]{1,10}[：:].+', line):
                structured_field_count += 1

        # 计算置信度
        field_score = min(1.0, structured_field_count / 5)  # 最多5个字段

        # 内容长度评分
        content_length = len(scene_text)
        length_score = min(1.0, content_length / 100)  # 100字符为满分

        return (field_score * 0.6 + length_score * 0.4)
