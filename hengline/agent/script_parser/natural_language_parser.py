# -*- coding: utf-8 -*-
"""
@FileName: script_parser_agent.py
@Description: LLM 剧本解析功能，通过LLM 将自然语言剧本转换为结构化动作序列
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import re
from typing import Dict, Any, Optional, List

from hengline.logger import debug
from hengline.prompts.prompts_manager import prompt_manager
from hengline.tools.json_parser_tool import parse_json_response
from .base_script_parser import ScriptParser
#
from .script_extractor.script_action_extractor import action_extractor
from .script_extractor.script_character_extractor import character_extractor
from .script_extractor.script_dialogue_extractor import dialogue_extractor
from .script_extractor.script_environment_extractor import environment_extractor
from .script_extractor.script_scene_segmenter import NaturalLanguageSceneSegmenter
from ..workflow_models import ParserType, ScriptType


class NaturalLanguageParser(ScriptParser):

    def __init__(self, llm_client):
        # 初始化各个处理器
        super().__init__(llm_client)
        self.scene_segmenter = NaturalLanguageSceneSegmenter()

    def _extract_with_llm(self, script_text: str) -> Optional[Dict[str, Any]]:
        """
        使用LLM直接解析剧本
        
        Args:
            script_text: 原始剧本文本
            
        Returns:
            解析结果或None
        """
        parser_prompt = prompt_manager.get_script_parser_prompt("natural_language")

        # 构建完整提示词
        prompt_format = parser_prompt.format(script_text=script_text)
        # 调用LLM
        response = self._call_llm_with_retry(prompt_format)
        debug(f"{ScriptType.NATURAL_LANGUAGE.value} 类型的 LLM 解析结果:\n {response}")

        # 解析LLM响应
        parsed_result = parse_json_response(response)

        return parsed_result

    def _extract_with_local(self, script_text: str) -> Optional[Dict[str, Any]]:
        """
            自然语言剧本本地解析器

            特点：
            1. 纯Python实现，无外部依赖
            2. 基于规则和正则表达式
            3. 处理中文剧本场景
            4. 模块化设计，便于扩展
        """
        # 1. 文本预处理
        cleaned_text = self.text_processor.clean_text(script_text)

        # 2. 场景分割
        scenes = self.scene_segmenter.segment(cleaned_text)

        # 3. 逐场景解析
        parsed_scenes = []
        all_characters = []
        all_dialogues = []
        all_actions = []

        for scene_idx, scene_text in enumerate(scenes):
            scene_id = f"scene_{scene_idx + 1:03d}"

            # 3.1 提取场景基本信息
            scene_info = self.scene_segmenter.extract_scene_info(scene_text, scene_id)

            # 3.2 提取角色
            characters = character_extractor.extract(scene_text)
            scene_info.character_refs = [c.name for c in characters]
            all_characters.extend(characters)

            # 3.3 提取对话
            dialogues = dialogue_extractor.extract(scene_text, scene_id)
            scene_info.dialogue_refs = [d.dialogue_id for d in dialogues]
            all_dialogues.extend(dialogues)

            # 3.4 提取动作
            actions = action_extractor.extract(scene_text, scene_id)
            scene_info.action_refs = [a.action_id for a in actions]
            all_actions.extend(actions)

            # 3.5 提取环境信息
            environment = environment_extractor.extract(scene_text)
            scene_info.summary = environment

            parsed_scenes.append(scene_info)

        # 4. 去重和合并角色
        unique_characters = character_extractor.deduplicate_characters(all_characters)

        # 5. 构建结果
        result = {
            "parser_type": ParserType.RULE_PARSER.value,
            "scenes": parsed_scenes,
            "characters": unique_characters,
            "dialogues": all_dialogues,
            "actions": all_actions,
            "descriptions": self._extract_descriptions(cleaned_text),
            "parsing_confidence": self.calculate_confidence(parsed_scenes, all_characters, all_dialogues, all_actions)
        }

        return result

    def _extract_descriptions(self, text: str) -> List[Dict]:
        """
        提取描述性段落

        描述性段落特征：
        1. 不包含对话
        2. 不包含明显动作
        3. 主要是环境、外貌、心理描写
        """
        descriptions = []

        # 按段落分割
        paragraphs = text.split('\n')

        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # 判断是否是描述性段落
            if self._is_descriptive_paragraph(paragraph):
                desc_id = f"desc_{para_idx + 1:03d}"
                description_type = self._classify_description_type(paragraph)

                description = {
                    "description_id": desc_id,
                    "text": paragraph,
                    "type": description_type,
                    "length": len(paragraph),
                    "key_elements": self._extract_key_elements(paragraph)
                }

                descriptions.append(description)

        return descriptions

    def _is_descriptive_paragraph(self, paragraph: str) -> bool:
        """
        判断是否是描述性段落
        """
        # 规则1：排除对话为主的段落
        if self._is_dialogue_paragraph(paragraph):
            return False

        # 规则2：排除动作为主的段落
        if self._is_action_paragraph(paragraph):
            return False

        # 规则3：检查描述性关键词
        descriptive_keywords = [
            # 环境描写
            '天空', '阳光', '月光', '雨', '雪', '风', '云',
            '房间', '客厅', '卧室', '厨房', '办公室', '街道',
            '灯光', '阴影', '颜色', '形状', '大小',

            # 外貌描写
            '穿着', '衣服', '发型', '眼睛', '鼻子', '嘴巴',
            '身高', '身材', '容貌', '表情', '气质',

            # 心理描写
            '心里', '内心', '感觉', '觉得', '认为', '想起',
            '回忆', '思绪', '情感', '情绪', '心情',

            # 状态描写
            '静静地', '默默地', '悄然', '悄然无声',
            '弥漫着', '散发着', '充满着', '洋溢着'
        ]

        keyword_count = sum(1 for keyword in descriptive_keywords if keyword in paragraph)
        if keyword_count >= 2:
            return True

        # 规则4：检查句子结构和长度
        sentences = re.split(r'[。！？]', paragraph)
        if len(sentences) <= 2:
            # 段落较短，检查是否为描述
            avg_length = sum(len(s) for s in sentences) / max(len(sentences), 1)
            if avg_length > 15:  # 句子较长，可能是描述
                return True

        return False

    def _is_dialogue_paragraph(self, paragraph: str) -> bool:
        """判断是否是对话段落"""
        dialogue_indicators = [
            r'["「]',  # 引号
            r'[：:]',  # 冒号
            r'说[：:]', r'道[：:]', r'喊道[：:]',  # 说、道、喊道
            r'[?!？!]{2,}',  # 多个问号或感叹号
        ]

        for pattern in dialogue_indicators:
            if re.search(pattern, paragraph):
                return True

        return False

    def _is_action_paragraph(self, paragraph: str) -> bool:
        """判断是否是动作段落"""
        action_verbs = [
            '走', '跑', '坐', '站', '躺', '看', '听', '说',
            '拿', '放', '开', '关', '推', '拉', '打', '抱',
            '笑', '哭', '喊', '叫', '跳', '爬', '转', '抬'
        ]

        verb_count = sum(1 for verb in action_verbs if verb in paragraph)

        # 如果包含3个以上动作动词，很可能是动作段落
        if verb_count >= 3:
            return True

        # 检查连续动作描述
        action_patterns = [
            r'[\u4e00-\u9fa5]{1,4}[\u4e00-\u9fa5]{1,4}[\u4e00-\u9fa5]{1,4}，',  # 多个动作并列
            r'然后', r'接着', r'随后', r'于是',  # 动作连接词
        ]

        for pattern in action_patterns:
            if re.search(pattern, paragraph):
                return True

        return False

    def _classify_description_type(self, paragraph: str) -> str:
        """分类描述类型"""
        # 环境描写关键词
        environment_keywords = [
            '窗外', '室内', '户外', '天气', '气候', '季节',
            '建筑', '家具', '装饰', '布置', '环境', '氛围',
            '光线', '阴影', '颜色', '声音', '气味'
        ]

        # 外貌描写关键词
        appearance_keywords = [
            '穿着', '打扮', '服饰', '发型', '发色', '眼睛',
            '鼻子', '嘴巴', '耳朵', '脸庞', '皮肤', '身材',
            '身高', '体型', '姿态', '表情', '神情', '气质'
        ]

        # 心理描写关键词
        psychological_keywords = [
            '心想', '内心', '心里', '思绪', '回忆', '想起',
            '感到', '觉得', '认为', '希望', '担心', '害怕',
            '期待', '失望', '开心', '悲伤', '愤怒', '紧张'
        ]

        # 统计关键词出现次数
        env_count = sum(1 for kw in environment_keywords if kw in paragraph)
        appear_count = sum(1 for kw in appearance_keywords if kw in paragraph)
        psycho_count = sum(1 for kw in psychological_keywords if kw in paragraph)

        # 判断类型
        if env_count > max(appear_count, psycho_count):
            return "环境描写"
        elif appear_count > max(env_count, psycho_count):
            return "外貌描写"
        elif psycho_count > max(env_count, appear_count):
            return "心理描写"
        else:
            # 检查是否有时间/地点信息
            if any(kw in paragraph for kw in ['在', '从', '到', '里', '中', '内']):
                return "场景描写"
            else:
                return "一般描写"

    def _extract_key_elements(self, paragraph: str) -> List[str]:
        """提取关键元素"""
        elements = []

        # 提取名词性短语（简单实现）
        # 中文名词通常为2-4个字
        noun_patterns = [
            r'([\u4e00-\u9fa5]{2,4}的[\u4e00-\u9fa5]{2,4})',  # XX的XX
            r'([\u4e00-\u9fa5]{2,4}[\u4e00-\u9fa5]{2,4})',  # 双字词组合
        ]

        for pattern in noun_patterns:
            matches = re.findall(pattern, paragraph)
            for match in matches:
                # 过滤常见非关键元素
                if not self._is_common_word(match):
                    elements.append(match)

        # 去重并限制数量
        unique_elements = list(set(elements))
        return unique_elements[:10]  # 最多返回10个关键元素

    def _is_common_word(self, word: str) -> bool:
        """判断是否是常见词（非关键元素）"""
        common_words = [
            '的', '了', '在', '是', '有', '和', '与', '或',
            '这个', '那个', '这些', '那些', '一种', '一些',
            '非常', '十分', '特别', '极其', '相当'
        ]

        return any(common in word for common in common_words)
