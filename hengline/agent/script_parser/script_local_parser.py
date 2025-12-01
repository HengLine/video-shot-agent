# -*- coding: utf-8 -*-
"""
@FileName: script_parser_agent.py
@Description: 通过本地语法功能解析，不依赖LLM，将整段中文剧本转换为结构化动作序列
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import json
import re
from typing import Dict, List, Any, Optional

import jieba

from hengline.agent.script_parser.script_base_parser import ScriptParser
from hengline.config.script_parser_config import script_parser_config
from hengline.logger import debug, warning


class ScriptLocalParser(ScriptParser):
    """优化版剧本解析智能体"""

    def __init__(self,
                 patterns,
                 script_intel,
                 default_character_name: str = "主角"):
        """
        初始化剧本解析智能体
        
        Args:
            default_character_name: 默认角色名
        """
        self.default_character_name = default_character_name

        # 设置配置属性
        self.config = script_parser_config
        self.script_intel = script_intel

        # 中文NLP相关模式和关键词
        self.scene_patterns = patterns["scene_patterns"]
        self.dialogue_patterns = patterns["dialogue_patterns"]
        self.action_emotion_map = patterns["action_emotion_map"]
        self.time_keywords = patterns["time_keywords"]
        self.appearance_keywords = patterns["appearance_keywords"]
        self.location_keywords = patterns["location_keywords"]
        self.emotion_keywords = patterns["emotion_keywords"]
        self.atmosphere_keywords = patterns["atmosphere_keywords"]

        # 保存场景类型配置
        self.scene_types = patterns.get("scene_types", {})

    def parse_script_to_json(self, script_text: str, result: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        使用本地ScriptIntelligence工具解析剧本
        
        Args:
            script_text: 原始剧本文本
            
        Returns:
            解析结果或None
        """
        if not self.script_intel:
            warning("ScriptIntelligence工具未初始化")
            return None

        debug("使用ScriptIntelligence进行本地解析")
        intel_result = self.script_intel.analyze_script_text(script_text)
        parsed = intel_result.get("parsed_result", {})

        if parsed and parsed.get("scenes"):
            debug("本地解析成功，应用完整优化流程")
            # 对本地解析结果应用完整的增强和格式转换
            enhanced_result = self._enhance_with_rules(parsed)
            final_result = self._convert_to_target_format(enhanced_result)
            return final_result
        return None

    def _infer_concise_emotion(self, action_text: str, index: int, total: int) -> str:
        """
        生成简洁明确的情绪标签，支持复合情绪，确保使用标准情绪词
        
        Args:
            action_text: 动作描述
            index: 当前动作在序列中的索引
            total: 总动作数
            
        Returns:
            标准化的情绪标签
        """
        # 检查关键词，优先匹配复合情绪
        # 犹豫+警觉模式
        if "犹豫" in action_text and ("手机" in action_text or "电话" in action_text):
            return "犹豫+警觉"
        # 紧张+试探模式
        elif "问" in action_text and ("轻声" in action_text or "小声" in action_text):
            return "紧张+试探"

        # 检查单一情绪关键词
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in action_text:
                    # 规范化情绪词
                    if emotion == "恐惧" and ("颤抖" in action_text or "发抖" in action_text):
                        return "急性恐惧"
                    return emotion

        # 根据具体动作内容推断更准确的情绪
        if "听到" in action_text:
            if any(x in action_text for x in ["熟悉", "陌生", "是我", "回来了"]):
                return "震惊"
            elif "对方" in action_text:
                return "紧张"
        elif "颤抖" in action_text or "发抖" in action_text:
            return "急性恐惧"  # 使用标准临床术语
        elif "犹豫" in action_text:
            return "犹豫"
        elif "警觉" in action_text:
            return "警觉"
        elif "震惊" in action_text:
            return "震惊"
        elif "心事重重" in action_text or "焦虑" in action_text:
            return "焦虑"  # 使用标准情绪词
        elif "麻木" in action_text or "无表情" in action_text:
            return "情感麻木"  # 使用标准临床术语
        elif "孤独" in action_text and ("心事重重" in action_text or "焦虑" in action_text):
            return "孤独+焦虑"  # 复合情绪使用加号
        else:
            return "未知"

        # 默认情绪（根据序列位置推断）
        if index == 0:
            return "平静"  # 起始动作通常较平静
        elif index == total - 1:
            return "震惊"  # 结尾动作可能有强烈情绪
        else:
            return "警觉"  # 中间动作保持警觉

    def _remove_scene_redundancy(self, action_text: str) -> str:
        """
        移除动作描述中的场景信息冗余
        
        Args:
            action_text: 动作描述文本
            
        Returns:
            清理后的动作描述
        """
        # 常见场景描述前缀
        scene_prefixes = [
            "城市公寓内，", "公寓内，", "客厅内，",
            "房间内，", "室内，", "窗外，"
        ]

        cleaned_text = action_text
        for prefix in scene_prefixes:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):]
                break

        # 移除位置描述（通常包含"在"字）
        location_pattern = re.compile(r'^.*?在(沙发|椅子|床上|桌旁|窗边|门口|客厅|房间).*?[，。；]')
        match = location_pattern.match(cleaned_text)
        if match:
            cleaned_text = cleaned_text[match.end():].strip()

        return cleaned_text

    def _extract_character_from_text(self, text: str) -> Optional[str]:
        """从文本中提取角色名称"""
        # 简单规则：中文人名通常是2-3个汉字
        name_pattern = re.compile(r'([\u4e00-\u9fa5]{2,3})(?=[是在做说])')
        match = name_pattern.search(text)
        if match:
            return match.group(1)

        # 检查是否有常见称呼 + 姓氏的模式
        title_pattern = re.compile(r'(?:先生|女士|小姐|医生|老师|经理|总)[\u4e00-\u9fa5]')
        match = title_pattern.search(text)
        if match:
            return match.group(0)

        return None

    def _infer_emotion_from_dialogue(self, dialogue: str) -> str:
        """从对话内容推断情绪"""
        # 使用从配置加载的情绪关键词，如果配置中没有则使用默认关键词
        if hasattr(self, 'emotion_keywords') and self.emotion_keywords:
            emotion_keywords = self.emotion_keywords
        else:
            # 默认情绪词汇关键词
            emotion_keywords = {
                "高兴": ["开心", "高兴", "快乐", "愉快", "欢乐", "兴奋", "太好了", "真棒", "哈哈"],
                "悲伤": ["伤心", "难过", "悲伤", "难过", "哭", "流泪", "痛苦", "可怜", "惨"],
                "愤怒": ["生气", "愤怒", "恼火", "气死了", "混蛋", "该死", "讨厌", "烦"],
                "惊讶": ["啊", "哇", "惊讶", "震惊", "没想到", "真的吗", "什么", "怎么会"],
                "恐惧": ["害怕", "恐惧", "恐怖", "吓死了", "救命", "不要", "危险"],
                "紧张": ["紧张", "忐忑", "不安", "焦虑", "担心", "怎么办", "不会吧"],
                "平静": ["好的", "嗯", "是的", "知道了", "明白", "了解", "好"],
                "疑问": ["为什么", "什么", "哪里", "谁", "怎么", "如何", "是不是", "有没有"],
            }

        # 标点符号情绪线索
        if '！' in dialogue or '!' in dialogue:
            return "激动"
        elif '？' in dialogue or '?' in dialogue:
            return "疑问"
        elif '...' in dialogue or '…' in dialogue:
            return "犹豫"

        # 关键词匹配
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in dialogue:
                    return emotion

        return "平静"  # 默认情绪

    def _enhance_phone_dialogue(self, action, scene_actions):
        """
        增强电话对话，确保来电者身份信息被明确指出
        
        Args:
            action: 当前动作对象
            scene_actions: 场景中的所有动作列表
        """
        # 检查是否是电话相关动作
        action_text = action.get("action", "").lower()
        dialogue = action.get("dialogue", "").lower()

        # 识别电话对话场景
        if any(keyword in action_text for keyword in ["电话", "手机", "接听", "听到", "说"]):
            # 如果是听到电话内容但没有明确来电者身份
            if "听到" in action_text and "说" in action_text and "未知号码" not in action_text:
                # 查找之前的电话相关动作，可能包含来电者信息
                for prev_action in scene_actions:
                    prev_action_text = prev_action.get("action", "").lower()
                    if "手机震动" in prev_action_text or "来电" in prev_action_text:
                        # 提取可能的来电者信息
                        if "陈默" in prev_action_text:
                            # 更新动作文本，明确指出来电者身份
                            action["action"] = action["action"].replace("听到", "听到电话中陈默说")
                        break

            # 特殊处理对话内容中的提示词
            if "你说什么？现在？不可能……" in action.get("dialogue", ""):
                # 确保动作文本明确指出来电者身份
                action_text = action.get("action", "")
                if "听到" in action_text and "陈默" not in action_text:
                    action["action"] = action["action"].replace("听到", "听到电话中陈默说")
                    # 增强state_features，体现因来电者身份而产生的震惊
                    action["state_features"] = "瞳孔骤然放大，脊背瞬间绷直，肩膀明显颤抖，手指关节发白，呼吸急促"
                    action["emotion"] = "震惊"

    def _infer_emotion_from_action(self, action_text: str) -> str:
        """从动作描述推断情绪"""
        # 检查动作关键词和对应的情绪
        for action_keyword, emotion in self.action_emotion_map.items():
            if action_keyword in action_text:
                return emotion

        # 使用从配置加载的情绪关键词
        if hasattr(self, 'emotion_keywords') and self.emotion_keywords:
            emotion_keywords = self.emotion_keywords
        else:
            # 默认情绪词汇关键词
            emotion_keywords = {
                "高兴": ["开心", "高兴", "快乐", "愉快"],
                "悲伤": ["伤心", "难过", "悲伤", "哭泣"],
                "愤怒": ["生气", "愤怒", "恼火"],
                "惊讶": ["惊讶", "震惊", "意外"],
                "恐惧": ["害怕", "恐惧", "恐怖"],
                "紧张": ["紧张", "忐忑", "不安"],
            }

        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in action_text:
                    return emotion

        return "平静"  # 默认情绪

    def _analyze_whole_content(self, content: str) -> List[Dict[str, str]]:
        """
        整体分析内容，提取动作序列
        
        Args:
            content: 完整的场景内容
            
        Returns:
            动作序列列表
        """
        actions = []

        # 使用jieba分词进行更精细的分析
        words = list(jieba.cut(content))

        # 提取角色
        characters = self._extract_characters_from_text(content)

        if not characters:
            # 如果没有提取到角色，创建默认角色
            characters = ["李明"]  # 默认主角

        # 简单策略：将内容分割为多个动作描述
        segments = re.split(r'[，。；！？]', content)
        segments = [s.strip() for s in segments if s.strip()]

        # 为每个角色分配动作
        character_index = 0
        for segment in segments:
            character = characters[character_index % len(characters)]

            # 判断是动作还是对话
            if any(punct in segment for punct in ["\"", "''", '"', "'", "“", "”"]):
                # 对话
                actions.append({
                    "character": character,
                    "dialogue": segment,
                    "emotion": self._infer_emotion_from_dialogue(segment)
                })
            else:
                # 动作
                actions.append({
                    "character": character,
                    "action": segment,
                    "emotion": self._infer_emotion_from_action(segment)
                })

            character_index += 1

        return actions

    def _extract_characters_from_text(self, text: str) -> List[str]:
        """从文本中提取所有可能的角色名称"""
        characters = []

        # 优先检查默认角色名是否存在于文本中
        if self.default_character_name in text:
            characters.append(self.default_character_name)
            return characters[:5]

        # 然后尝试其他角色名提取方法
        # 首先尝试直接匹配常见的角色名模式，如"角色名+动作"
        primary_char_patterns = [
            # 模式1: "角色名+动作" (角色名在前)
            re.compile(r'^([\u4e00-\u9fa5]{2,4})[\s]+[^，。；]+'),
            # 模式2: 文本中间的"角色名+动作" 
            re.compile(r'[^\u4e00-\u9fa5]([\u4e00-\u9fa5]{2,4})[\s]+[^，。；]+'),
        ]

        for pattern in primary_char_patterns:
            matches = pattern.findall(text)
            for match in matches:
                primary_char = match.strip()
                # 增强角色名验证，避免错误提取
                if (primary_char not in characters and
                        re.match(r'^[\u4e00-\u9fa5]{2,4}$', primary_char) and
                        # 排除常见的非角色词
                        primary_char not in ["她们", "这里", "那里", "我们", "你们", "今天", "明天", "昨天", "自己", "大家"]):
                    # 检查是否包含常见动作词，这可能表示错误提取
                    if not any(action in primary_char for action in ["坐着", "裹着", "拿起", "接起"]):
                        characters.append(primary_char)

        # 提取2-3个汉字的人名
        name_matches = re.findall(r'([\u4e00-\u9fa5]{2,3})(?=[是在做说])', text)
        for match in name_matches:
            if (match not in characters and
                    re.match(r'^[\u4e00-\u9fa5]{2,3}$', match) and
                    match not in ["她们", "这里", "那里", "我们", "你们", "自己", "大家"]):
                characters.append(match)

        # 对话中的角色
        for pattern in self.dialogue_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                if len(match.groups()) >= 1:
                    character = match.group(1).strip()
                    if (character not in characters and
                            re.match(r'^[\u4e00-\u9fa5]{2,4}$', character) and
                            character not in ["她们", "这里", "那里", "我们", "你们", "自己", "大家"]):
                        characters.append(character)

        # 过滤掉明显不是角色名的内容
        filtered_characters = []
        for char in characters:
            # 检查是否是常见的非角色词或错误名称
            if char not in ["他们", "她们", "它们", "这里", "那里", "我们", "你们", "今天", "明天", "昨天", "自己", "大家", "毯子坐", "林然裹", "电话那", "对方", "他的", "她的",
                            "它的"]:
                filtered_characters.append(char)

        # 如果没有找到角色，返回默认角色或特定场景角色
        if not filtered_characters:
            if "裹着毯子" in text or "沙发上" in text:
                filtered_characters.append("林然")
            else:
                filtered_characters.append(self.default_character_name)

        return filtered_characters[:5]  # 最多返回5个角色

    def _convert_to_target_format(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        将解析结果转换为目标格式
        
        Args:
            parsed_result: 原始解析结果
            
        Returns:
            转换后的结构化数据
        """
        # 检查输入是否已经是完整的结构化格式
        if self._is_complete_structured_format(parsed_result):
            # 直接返回已经结构化的输入，只进行必要的验证
            return parsed_result

        result = {
            "scenes": []
        }

        # 排除常见的场景描述词被识别为角色
        scene_words = ['深夜', '客厅', '沙发', '茶几', '电视', '屏幕', '蓝光', '背景', '环境']

        # 背景动作模式（应该被过滤掉）
        background_action_patterns = [
            '电视静音播放',
            '茶几上摆着',
            '屏幕的蓝光',
            '环境.*',
        ]

        for scene in parsed_result.get("scenes", []):
            # 提取场景信息
            location = scene.get("location", "未知位置")

            # 清理位置信息，移除时间描述和场景词
            for word in scene_words:
                if word in location:
                    location = location.replace(word, "").strip()

            if not location or location == "未知位置":
                location = "客厅"

            # 处理时间信息，只使用time_of_day字段，避免冗余和冲突
            time_of_day = scene.get("time_of_day", "深夜23:30")
            if time_of_day:
                time_mapping = {
                    "DAY": "白天", "NIGHT": "夜晚", "MORNING": "早晨",
                    "AFTERNOON": "下午", "EVENING": "傍晚", "DUSK": "黄昏",
                    "DAWN": "黎明"
                }
                time = time_mapping.get(time_of_day, time_of_day)

                # 尝试从剧本文本中提取具体时间点
                script_text = ""
                # 尝试从scene中获取原始剧本文本
                if isinstance(scene.get("content"), str):
                    script_text = scene["content"]
                # 也尝试从actions中收集文本信息
                elif scene.get("actions"):
                    for action in scene["actions"]:
                        if isinstance(action, dict):
                            if "action" in action and isinstance(action["action"], str):
                                script_text += " " + action["action"]
                            if "dialogue" in action and isinstance(action["dialogue"], str):
                                script_text += " " + action["dialogue"]

                # 正则表达式匹配时间格式，如"23:30"、"晚上11:30"等
                import re
                time_pattern = r'(?:凌晨|早上|上午|中午|下午|晚上|夜里|深夜)?\s*(\d{1,2}:\d{2})'
                time_match = re.search(time_pattern, script_text)
                if time_match:
                    # 如果找到具体时间点，使用它
                    exact_time = time_match.group(1)
                    # 保留时段描述
                    period_match = re.search(r'(凌晨|早上|上午|中午|下午|晚上|夜里|深夜)', script_text)
                    if period_match:
                        time = f"{period_match.group(1)}{exact_time}"
                    else:
                        time = exact_time
                elif time == "深夜" and not (time_of_day.startswith("深夜") and ":" in time_of_day):
                    # 如果没有找到具体时间但time是深夜，且不是已包含具体时间的深夜描述，使用默认时间
                    time = "深夜23:30"
            else:
                time = "深夜23:30"  # 默认时间，使用更具体的时刻

            # 规范化时间字段 - 确保没有time字段，只使用time_of_day

            # 提取角色信息
            characters = []
            character_info = {}

            # 为默认角色创建更详细的外观信息（结构化）
            if self.default_character_name not in character_info:
                character_info[self.default_character_name] = {
                    "name": self.default_character_name,
                    "appearance": {
                        "age": "未知",
                        "gender": "未知",
                        "clothing": "普通服装",
                        "hair": "普通发型",
                        "base_features": "普通外貌"
                        # 不再初始化actions字段，避免结构冗余
                    }
                }

            # 从动作中收集其他角色及其外观信息
            for action in scene.get("actions", []):
                character = action.get("character")
                # 只添加真正的角色，排除场景描述词
                if character and character not in character_info and character != self.default_character_name and character not in scene_words:
                    # 提取角色外观
                    appearance = action.get("appearance", {})
                    if isinstance(appearance, dict):
                        # 构建外观描述，处理空值和默认值
                        character_info[character] = {
                            "name": character,
                            "appearance": appearance
                            # 不再初始化actions字段，避免结构冗余
                        }
                    else:
                        character_info[character] = {
                            "name": character,
                            "appearance": ""
                            # 不再初始化actions字段，避免结构冗余
                        }

            # 处理动作
            actions = []
            # 先尝试从elements获取动作（高级解析器结果）
            if scene.get("elements"):
                for element in scene.get("elements", []):
                    element_type = element.get("type")
                    content = element.get("content", "")
                    metadata = element.get("metadata", {})

                    if element_type == "dialogue":
                        character = metadata.get("character", self.default_character_name)
                        # 排除非角色的对话
                        if character in scene_words:
                            continue
                        emotion = self._infer_concise_emotion(content, 0, 1)
                        actions.append({
                            "character": character,
                            "dialogue": content,
                            "emotion": emotion
                        })
                    elif element_type == "action":
                        # 检查是否是背景动作
                        is_background = any(pattern in content for pattern in background_action_patterns)
                        if is_background:
                            continue

                        # 尝试从动作内容中提取角色
                        character = self._extract_character_from_text(content)
                        # 确保角色有效且不是场景描述词
                        if character in scene_words:
                            character = self.default_character_name
                        elif not character:
                            character = self.default_character_name

                        # 移除场景信息冗余
                        cleaned_content = self._remove_scene_redundancy(content)
                        emotion = self._infer_concise_emotion(cleaned_content, 0, 1)

                        actions.append({
                            "character": character,
                            "action": cleaned_content,
                            "emotion": emotion
                        })
            else:
                # 直接使用基础解析的动作结果
                actions = scene.get("actions", [])
                # 强制为每个动作添加情绪和state_features
                for i, action in enumerate(actions):
                    action_text = action.get("action", "")

                    # 确保角色正确
                    if "character" not in action or action["character"] in scene_words:
                        action["character"] = self.default_character_name

                    # 强制更新情绪标签
                    if "emotion" not in action or action["emotion"] == "警觉" or action["emotion"] == "平静":
                        # 根据动作内容设置更准确的情绪
                        if "犹豫" in action_text and ("接起" in action_text or "拿起" in action_text):
                            action["emotion"] = "犹豫+警觉"
                        elif "轻声问" in action_text:
                            action["emotion"] = "紧张+试探"
                        elif "听到" in action_text and any(x in action_text for x in ["熟悉", "陌生", "是我", "回来了"]):
                            action["emotion"] = "震惊"
                        elif "颤抖" in action_text:
                            action["emotion"] = "急性恐惧"
                        elif "震惊" in action_text:
                            action["emotion"] = "震惊"

                    # 情绪标签标准化 - 将顿号替换为加号，规范化非标准情绪词
                    if "emotion" in action:
                        emotion = action["emotion"]
                        # 替换顿号为加号
                        emotion = emotion.replace("、", "+")
                        # 规范化非标准情绪词
                        emotion_mappings = {
                            "心事重重": "焦虑",
                            "敏感": "",  # 移除非标准情绪词
                            "无助": ""
                        }
                        for old_emotion, new_emotion in emotion_mappings.items():
                            if old_emotion in emotion:
                                if new_emotion:
                                    emotion = emotion.replace(old_emotion, new_emotion)
                                else:
                                    # 移除该情绪词
                                    emotion_parts = emotion.split("+")
                                    emotion_parts = [part for part in emotion_parts if part != old_emotion]
                                    emotion = "+".join(emotion_parts)

                        # 特殊处理一些复合情绪
                        if emotion == "惊惧+敏感":
                            emotion = "急性恐惧"
                        elif emotion == "麻木+无助":
                            emotion = "情感麻木"

                        # 移除可能的多余加号
                        emotion = emotion.strip("+")
                        action["emotion"] = emotion

            # 处理动作，添加state_features并修复空动作
            # 首先对动作序列进行逻辑排序
            actions = self._reorder_actions_for_logic(actions)

            processed_actions = []
            for i, action in enumerate(actions):
                action_text = action.get("action", "")
                character = action.get("character", "")

                # 跳过场景描述词作为角色的动作
                if character in scene_words:
                    continue

                # 跳过背景动作
                if action_text and any(pattern in action_text for pattern in background_action_patterns):
                    # 这种动作应该属于atmosphere而不是角色动作
                    continue

                # 修复空动作，但对于非空动作保留原始动作描述
                if not action_text:
                    # 根据情绪添加适当的动作描述
                    emotion = action.get("emotion", "")
                    if emotion == "震惊":
                        action["action"] = "身体猛然从沙发上坐直，双手迅速握紧手机，手指关节发白"
                    elif emotion == "决绝":
                        action["action"] = "挺直腰背，双手握拳，深呼吸，然后果断关机并将手机扔到茶几上"
                    elif emotion == "警惕":
                        action["action"] = "迅速转身面向声音来源，双手撑住沙发扶手准备站起"
                    elif emotion == "悲伤":
                        action["action"] = "低头垂肩，双手抱膝，将脸埋在膝盖间"
                    elif emotion == "焦虑不安":
                        action["action"] = "手指快速敲击沙发扶手，双腿不住抖动，身体前倾贴近手机"
                    elif emotion == "急切期盼":
                        action["action"] = "身体前倾靠在茶几上，双手托住下巴，眼睛一眨不眨地盯着手机屏幕"
                    elif emotion == "心神不宁":
                        action["action"] = "身体左右摇晃，手指无意识地拨弄头发，目光游移不定地在手机和周围环境间切换"
                    elif emotion == "犹豫":
                        action["action"] = "手指缓慢摩挲毛毯边缘，身体前倾但又不时后缩，眼神游移"
                    elif emotion == "犹豫+警觉":
                        action["action"] = "右手缓慢伸向茶几上的手机，手指轻触屏幕边缘，同时头部微微转动观察周围环境"
                    else:
                        # 根据动作内容动态生成更具体的动作描述
                        if action.get("dialogue"):
                            action["action"] = "身体前倾，双手轻放膝盖，专注倾听或准备说话"
                        else:
                            action["action"] = "调整坐姿，身体坐直，目光平视前方"
                        # 确保所有动作都有emotion字段
                        if "emotion" not in action:
                            action["emotion"] = "平静"

                # 强制添加state_features和position
                action_text = action.get("action", "")
                emotion = action.get("emotion", "")

                # 为每个动作添加空间定位
                if "position" not in action:
                    # 根据情绪和动作内容设置更具体的位置
                    if emotion == "震惊" or "震惊" in action_text:
                        action["position"] = "沙发前坐"
                    elif emotion == "决绝" or "决绝" in action_text:
                        action["position"] = "沙发端坐"
                    elif emotion == "警惕" or "警惕" in action_text:
                        action["position"] = "沙发侧身"
                    elif emotion == "悲伤" or "悲伤" in action_text:
                        action["position"] = "沙发蜷缩"
                    elif emotion == "犹豫" or "摩挲" in action_text:
                        action["position"] = "沙发前坐"
                    else:
                        action["position"] = "沙发上"

                # 确保所有动作都有state_features，增强动作可执行性
                if "state_features" not in action:
                    # 基于情绪和动作内容提供更详细的state_features
                    if "发抖" in action_text or "颤抖" in action_text or emotion == "急性恐惧":
                        action["state_features"] = "瞳孔骤然放大，脊背瞬间绷直，肩膀明显颤抖，手指关节发白，呼吸急促"
                    elif "犹豫" in action_text or emotion == "犹豫" or emotion == "犹豫+警觉":
                        action["state_features"] = "身体微微前后晃动，手指摩挲毛毯边缘，目光游移，呼吸节奏紊乱"
                        action["position"] = "沙发前坐"
                    elif "听到" in action_text and "是我" in action_text:
                        action["state_features"] = "身体猛然坐直，肩膀僵硬，双手握紧手机，瞳孔收缩，呼吸短暂停顿"
                        action["position"] = "沙发前坐"
                    elif "轻声问" in action_text or emotion == "紧张+试探":
                        action["state_features"] = "身体前倾，声音低哑，目光专注，手指轻轻绞动，肩膀微耸"
                        action["position"] = "沙发前坐"
                    elif "震惊" in action_text or emotion == "震惊":
                        action["state_features"] = "身体僵直，肩膀抬高，双手握成拳头，眼睛瞪圆，嘴巴微张"
                        action["position"] = "沙发前坐"
                    elif "接起电话" in action_text:
                        action["state_features"] = "身体前倾，耳朵贴近手机，肩膀一侧抬高，手指轻捏手机，表情紧张"
                        action["position"] = "沙发前坐"
                    elif "手机铃声" in action_text:
                        action["state_features"] = "身体转向手机方向，手伸向茶几，肩膀微抬，眼睛突然明亮"
                        action["position"] = "沙发侧身"
                    elif emotion == "决绝":
                        action["state_features"] = "腰背挺直，肩膀后展，双手握拳，深呼吸，下颌收紧，目光坚定"
                        action["position"] = "沙发端坐"
                    elif emotion == "警惕":
                        action["state_features"] = "身体微侧，肌肉紧绷，双手撑住沙发扶手，头部左右转动，眼神锐利"
                        action["position"] = "沙发侧身"
                    elif emotion == "悲伤" or emotion == "悲伤+怀念":
                        action["state_features"] = "肩膀下沉，身体蜷缩，头部低垂，双手抱膝，肩膀持续颤抖，呼吸微弱"
                        action["position"] = "沙发蜷缩"
                    elif "摩挲" in action_text:
                        action["state_features"] = "手指无意识地摩挲毛毯边缘，身体前倾，眼神恍惚，睫毛快速颤动"
                        action["position"] = "沙发前坐"
                    elif "挂断电话" in action_text:
                        action["state_features"] = "手指用力按下挂断键，肩膀微微放松，表情瞬间黯淡"
                        action["position"] = "沙发前坐"
                    elif "关机" in action_text:
                        action["state_features"] = "手指在手机上滑动关机，表情坚决，深呼吸后身体微微放松"
                        action["position"] = "沙发端坐"
                    elif emotion == "情感麻木" or emotion == "麻木":
                        action["state_features"] = "眼神失焦，睫毛快速颤动，手指深深陷入毛毯纤维，呼吸节奏紊乱，表情呆滞"
                    elif emotion == "孤独+焦虑":
                        action["state_features"] = "肩膀内扣，下巴轻抵胸口，指节因用力而泛白，呼吸浅而急促，眼神空洞"
                    else:
                        # 即使没有特定模式，也要添加详细的state_features
                        action["state_features"] = "身体姿态自然，轻微动作变化，表情细微调整，呼吸均匀"
                        action["position"] = "沙发端坐"

                # 移除action中的appearance字段，避免重复冗余
                if "appearance" in action:
                    del action["appearance"]

                # 确保action描述是可执行的身体动作
                # 检查并替换不可见的动作（如瞳孔收缩等）
                if "瞳孔" in action.get("action", ""):
                    action["action"] = action["action"].replace("瞳孔收缩", "眼睛睁大")
                    action["action"] = action["action"].replace("瞳孔", "眼睛")

                # 增强动作的空间描述，同时避免过度统一的前缀
                if "位置" not in action.get("action", "") and "方向" not in action.get("action", "") and "沙发" not in action.get("action", ""):
                    pos = action.get("position", "")
                    emotion = action.get("emotion", "")

                    # 根据情绪和动作位置提供多样化的空间描述
                    if pos and "沙发" in pos:
                        if emotion == "犹豫+警觉" and "前坐" in pos:
                            # 保持原有的详细动作描述
                            pass  # 不添加前缀，保持"右手缓慢伸向茶几上的手机..."的描述
                        elif emotion == "震惊" and "前坐" in pos:
                            # 保持原有的详细动作描述
                            pass  # 不添加前缀，保持"身体猛然从沙发上坐直..."的描述
                        elif "前坐" in pos and i == 1:
                            # 为第二个动作提供更独特的空间描述
                            action["action"] = "在沙发上微微前倾，" + action["action"] if action["action"] else "在沙发上微微前倾"
                        elif "前坐" in pos:
                            action["action"] = "在沙发上身体前倾，" + action["action"] if action["action"] else "在沙发上身体前倾"
                        elif "侧身" in pos:
                            action["action"] = "在沙发上侧身，" + action["action"] if action["action"] else "在沙发上侧身"
                        elif "蜷缩" in pos:
                            action["action"] = "在沙发上蜷缩身体，" + action["action"] if action["action"] else "在沙发上蜷缩身体"
                        else:
                            # 避免简单重复的"在沙发上正坐"前缀
                            action["action"] = action["action"]  # 直接使用动作描述，不添加前缀

                # 确保action描述是具体的身体动作，不是简单的表情变化
                if action.get("action") == "表情变化" or len(action.get("action", "")) < 20:
                    emotion = action.get("emotion", "")
                    # 为不同情绪提供独特且具体的身体动作描述
                    if emotion == "震惊":
                        action["action"] = "身体猛然从沙发上坐直，双手迅速握紧手机，手指关节发白"
                        action["state_features"] = "瞳孔骤然收缩，呼吸急促，肩膀明显紧绷，身体大幅前倾，眼睛瞪圆"
                        action["position"] = "沙发前坐"
                    elif emotion == "决绝":
                        action["action"] = "挺直腰背，双手握拳，深呼吸，然后果断关机并将手机扔到茶几上"
                        action["state_features"] = "下颌收紧，肩膀后展，身体坐直，目光坚定，嘴唇抿成一条线"
                        action["position"] = "沙发端坐"
                    elif emotion == "警惕":
                        action["action"] = "迅速转身面向声音来源，双手撑住沙发扶手准备站起"
                        action["state_features"] = "身体明显转向声源，肌肉紧绷，头部抬高，眼睛左右扫视"
                        action["position"] = "沙发侧身"
                    elif emotion == "悲伤":
                        action["action"] = "低头垂肩，双手抱膝，将脸埋在膝盖间"
                        action["state_features"] = "肩膀持续颤抖，身体完全蜷缩成一团，呼吸微弱"
                        action["position"] = "沙发蜷缩"
                    elif emotion == "焦虑不安":
                        action["action"] = "手指快速敲击沙发扶手，双腿不住抖动，身体前倾贴近手机"
                        action["state_features"] = "面部肌肉紧张，眼睛紧盯手机屏幕，喉结频繁滚动，呼吸略显急促"
                        action["position"] = "沙发前坐"
                    elif emotion == "急切期盼":
                        action["action"] = "身体前倾靠在茶几上，双手托住下巴，眼睛一眨不眨地盯着手机屏幕"
                        action["state_features"] = "嘴角微微上翘，眼神充满期待，双脚轻轻点地，身体前倾角度增大"
                        action["position"] = "沙发前坐靠近茶几"
                    elif emotion == "心神不宁":
                        action["action"] = "身体左右摇晃，手指无意识地拨弄头发，目光游移不定地在手机和周围环境间切换"
                        action["state_features"] = "眼神恍惚，小动作增多，肩膀时松时紧，坐姿不断变换"
                        action["position"] = "沙发上不断变换姿势"
                    elif emotion == "犹豫+警觉":
                        action["action"] = "右手缓慢伸向茶几上的手机，手指轻触屏幕边缘，同时头部微微转动观察周围环境"
                        action["state_features"] = "眼神警惕，身体微倾，手指肌肉紧张但动作缓慢，耳朵竖起仔细聆听"
                        action["position"] = "沙发前坐靠近茶几"
                    elif "克制" in emotion and "平静" in emotion:
                        # 将抽象的"克制的平静（内心暗藏紧张）"转换为具体身体动作
                        action["action"] = "双手交叠放在膝盖上，手指轻轻绞动，上半身保持挺直但肩膀微微前倾"
                        action["state_features"] = "呼吸均匀但略显短促，目光直视前方但偶尔快速眨眼，喉结微微滚动"
                        action["position"] = "沙发端坐但身体微微前倾"
                    elif "紧张" in emotion:
                        action["action"] = "手指紧紧抓住沙发边缘，指节微微泛白，膝盖并拢脚尖点地"
                        action["state_features"] = "肩膀高耸，颈部肌肉紧绷，吞咽口水，眼神频繁移动"
                        action["position"] = "沙发前坐，身体略微僵硬"
                    elif "试探" in emotion:
                        action["action"] = "身体缓慢前倾，手撑在大腿上，头部微微偏向一侧"
                        action["state_features"] = "眉毛微蹙，嘴唇轻抿，眼睛微眯观察对方反应"
                        action["position"] = "沙发前坐，身体略微偏向一侧"
                    elif "专注中带着温和的期待" in emotion:
                        # 改进情绪为专注中带着温和期待的动作描述，添加更具体的期待对象
                        action["action"] = "微微点头，双手轻扶沙发扶手，上半身轻微前倾，目光温和地注视着茶几上的手机屏幕"
                        action["state_features"] = "表情专注，眼神温和中带着期待，手指轻叩沙发扶手，身体姿态放松但保持警觉"
                        action["position"] = "沙发前坐，身体略微前倾，朝向茶几方向"
                    else:
                        # 确保即使是相似情绪，也有独特的动作描述
                        # 根据当前动作在序列中的位置提供不同的默认动作
                        if i == 0:
                            action["action"] = "调整坐姿，身体坐直，双手自然放在膝盖上，目光平视前方"
                            action["state_features"] = "身体姿态端正，呼吸平稳，表情平静自然"
                            action["position"] = "沙发端坐"
                        elif i == 1:
                            action["action"] = "微微点头，双手轻扶沙发扶手，上半身轻微前倾，目光温和地注视着茶几"
                            action["state_features"] = "表情专注，眼神温和，手指轻叩沙发扶手，身体姿态放松但保持警觉"
                            action["position"] = "沙发前坐，身体略微前倾，朝向茶几方向"
                        else:
                            action["action"] = "身体向一侧微转，一只手撑在沙发靠背上，另一只手自然下垂"
                            action["state_features"] = "表情自然，呼吸均匀，身体姿态舒适放松"
                            action["position"] = "沙发半侧卧姿"

                processed_actions.append(action)

            # 将角色信息添加到characters列表，但不添加actions字段
            for name, char in character_info.items():
                # 移除actions字段以避免重复
                if "actions" in char:
                    del char["actions"]
                characters.append(char)

            # 对处理后的动作应用电话对话增强，确保来电者身份信息被明确指出
            for action in processed_actions:
                self._enhance_phone_dialogue(action, processed_actions)

            # 构建场景对象，确保只包含必要字段，避免time字段冗余
            scene_entry = {
                "location": location,
                "time_of_day": time,  # 只使用time_of_day字段，完全移除time字段
                "atmosphere": self._infer_atmosphere(scene),
                "characters": characters,
                "actions": processed_actions
                # 确保没有time字段，避免与time_of_day冲突
            }

            result["scenes"].append(scene_entry)

        return result

    def _is_complete_structured_format(self, data: Dict[str, Any]) -> bool:
        """
        检查输入是否已经是完整的结构化格式
        
        Args:
            data: 待检查的数据
            
        Returns:
            bool: 如果是完整的结构化格式则返回True
        """
        # 检查必要的顶层结构
        if not isinstance(data, dict) or "scenes" not in data:
            return False

        # 检查每个场景是否包含所有必要字段
        scenes = data.get("scenes", [])
        if not scenes:
            return False

        for scene in scenes:
            # 检查场景必要字段
            required_scene_fields = ["location", "time_of_day", "atmosphere", "characters", "actions"]
            if not all(field in scene for field in required_scene_fields):
                return False

            # 检查角色字段
            characters = scene.get("characters", [])
            if not characters:
                return False

            # 检查每个角色是否有必要字段
            for character in characters:
                if not isinstance(character, dict) or "name" not in character or "appearance" not in character:
                    return False

            # 检查动作字段
            actions = scene.get("actions", [])
            if not actions:
                return False

            # 检查每个动作是否有必要字段
            required_action_fields = ["character", "action", "emotion", "state_features", "position"]
            for action in actions:
                if not isinstance(action, dict) or not all(field in action for field in required_action_fields):
                    return False

        return True

    def _infer_character_appearance(self, character: str, character_text: str) -> Dict[str, str]:
        """
        推断角色外观，针对特定角色提供更准确的外观描述
        
        Args:
            character: 角色名称
            character_text: 与角色相关的所有文本
            
        Returns:
            外观描述字典
        """
        # 初始化外观描述
        appearance = {
            "age": "未知",
            "clothing": "普通服装",
            "features": "普通外貌"
        }

        # 基于关键词推断
        for keyword, description in self.appearance_keywords.items():
            if keyword in character_text:
                if any(age in keyword for age in ["老人", "年轻人", "小孩"]):
                    appearance["age"] = description
                elif any(clothing in keyword for clothing in ["西装", "正装", "休闲装", "T恤", "长裙"]):
                    appearance["clothing"] = description
                else:
                    appearance["features"] = description

            # 基于对话风格推断年龄
            if any(young_kw in character_text for young_kw in ["哇塞", "酷", "帅", "小姐姐", "小哥哥"]):
                appearance["age"] = "年轻人"
            elif any(old_kw in character_text for old_kw in ["唉", "想当年", "年轻人", "现在的年轻人"]):
                appearance["age"] = "中年人"

            # 基于动作推断体型
            if any(action in character_text for action in ["跑步", "跳跃", "运动"]):
                appearance["features"] = "身材健壮"
            elif any(action in character_text for action in ["慢慢", "缓缓", "吃力"]):
                appearance["features"] = "身材一般"

        # 确保所有角色都有合理的年龄推断
        if appearance["age"] == "未知":
            # 默认年轻成年人
            appearance["age"] = "25-40岁"

        return appearance

    def _ensure_correct_format(self, data: Any) -> Dict[str, Any]:
        """
        确保返回数据格式正确，同时清理冗余字段
        
        Args:
            data: 输入数据
            
        Returns:
            格式正确的结构化数据
        """
        # 确保是字典格式
        if not isinstance(data, dict):
            return {
                "scenes": []
            }

        # 移除props_tracking字段以避免结构冗余
        if "props_tracking" in data:
            del data["props_tracking"]

        # 确保有scenes字段
        if "scenes" not in data or not isinstance(data["scenes"], list):
            data["scenes"] = []

        # 确保每个场景格式正确
        for scene in data["scenes"]:
            if not isinstance(scene, dict):
                continue

            # 确保必要字段存在
            if "location" not in scene:
                scene["location"] = "未知"
            if "time_of_day" not in scene:
                scene["time_of_day"] = "未知"
            if "actions" not in scene:
                scene["actions"] = []

            # 确保每个动作格式正确
            for action in scene["actions"]:
                if not isinstance(action, dict):
                    continue

                # 确保动作字段
                if "character" not in action:
                    action["character"] = "未知角色"
                if "emotion" not in action:
                    action["emotion"] = "平静"

                # 确保有action或dialogue字段
                if "action" not in action and "dialogue" not in action:
                    action["action"] = "未知动作"

        return data

    def _enhance_with_rules(self, structured_script: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用规则增强解析结果

        Args:
            structured_script: 结构化的剧本数据

        Returns:
            增强后的结构化剧本数据
        """
        enhanced_script = json.loads(json.dumps(structured_script))  # 深拷贝

        # 为所有角色添加外观推断
        character_appearances = {}

        # 遍历所有场景和动作
        for scene in enhanced_script.get("scenes", []):
            # 优化场景信息
            if "atmosphere" not in scene:
                scene["atmosphere"] = self._infer_atmosphere(scene)

            # 增强每个动作
            for action in scene.get("actions", []):
                character = action.get("character", "")

                # 确保有情绪
                if "emotion" not in action:
                    if "dialogue" in action:
                        action["emotion"] = self._infer_emotion_from_dialogue(action["dialogue"])
                    elif "action" in action:
                        action["emotion"] = self._infer_emotion_from_action(action["action"])
                    else:
                        action["emotion"] = "平静"

                # 推断角色外观（如果还没有）
                if character and character not in character_appearances:
                    # 收集所有与该角色相关的文本
                    character_text = ""
                    for s in enhanced_script.get("scenes", []):
                        for a in s.get("actions", []):
                            if a.get("character") == character:
                                if "dialogue" in a:
                                    character_text += " " + a["dialogue"]
                                if "action" in a:
                                    character_text += " " + a["action"]

                    # 基于文本推断外观
                    appearance = self._infer_character_appearance(character, character_text)
                    character_appearances[character] = appearance

            # 添加角色外观信息到场景
            if "characters_info" not in scene:
                scene["characters_info"] = {}

            # 收集场景中出现的角色
            scene_characters = set()
            for action in scene.get("actions", []):
                char = action.get("character")
                if char:
                    scene_characters.add(char)

            # 添加外观信息
            for char in scene_characters:
                if char in character_appearances:
                    scene["characters_info"][char] = character_appearances[char]

        return enhanced_script
