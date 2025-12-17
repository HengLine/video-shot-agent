# -*- coding: utf-8 -*-
"""
@FileName: script_parser_agent.py
@Description: 剧本解析智能体，将整段中文剧本转换为结构化动作序列
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional

from config.config import get_embedding_config
from hengline.config.script_parser_config import script_parser_config
from hengline.logger import debug, error, warning
from utils.log_utils import print_log_exception
from .script_parser.script_base_parser import ScriptParser
from .script_parser.script_llm_parser import ScriptLLMParser
from .script_parser.script_local_parser import ScriptLocalParser
from .script_parser.script_parser_fallback import ScriptParserFallback
from ..prompts.prompts_manager import prompt_manager
from ..tools import create_script_intelligence
from ..tools.json_parser_tool import parse_json_response


class ScriptParserAgent(ScriptParser):
    """优化版剧本解析智能体"""

    def __init__(self,
                 llm=None,
                 storage_dir: Optional[str] = None,
                 default_character_name: str = "主角"):
        """
        初始化剧本解析智能体
        
        Args:
            llm: 语言模型实例（推荐GPT-4o）
            storage_dir: 知识库存储目录
            default_character_name: 默认角色名
        """
        self.llm = llm
        # 从config获取嵌入模型配置
        self.embedding_config = get_embedding_config()

        # 初始化智能分析工具
        try:
            self.script_intel = create_script_intelligence(
                embedding_model_type=self.embedding_config["provider"],
                embedding_model_name=self.embedding_config["model"],
                embedding_model_config=self.embedding_config,
                storage_dir=storage_dir
            )
            debug("成功初始化ScriptIntelligence工具")
        except Exception as e:
            warning(f"初始化ScriptIntelligence失败，但将继续使用基础功能: {str(e)}")
            self.script_intel = None

        self.llm_parser = ScriptLLMParser(llm=llm, script_intel=self.script_intel)
        self.default_character_name = default_character_name

        # 设置配置文件路径
        self.config_path = str(Path(__file__).parent.parent / "config" / "script_parser_config.yaml")

        # 中文NLP相关模式和关键词
        patterns = script_parser_config.initialize_patterns(self.config_path)
        self.local_parser = ScriptLocalParser(patterns=patterns, script_intel=self.script_intel, default_character_name=default_character_name)
        self.parser_fallback = ScriptParserFallback(patterns=patterns, default_character_name=default_character_name)

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

    def parse_script(self, script_text: str, result: Dict[str, Any] = None) -> dict[str, Any] | None:
        """
        优化版剧本解析函数
        将整段中文剧本转换为结构化动作序列
        
        Args:
            script_text: 原始剧本文本

        Returns:
            结构化的剧本动作序列
        """
        debug(f"开始解析剧本: {script_text[:100]}...")

        try:
            # 初始化结果结构
            result = {"scenes": []}
            final_result = None

            # 1. 优先尝试LLM解析（如果配置了）
            if self.llm:
                final_result = self.llm_parser.parse_script(script_text)
                debug("使用LLM解析结果作为最终输出")

            # 2. 如果LLM解析失败或未配置，使用本地解析器
            if final_result is None:
                final_result = self.local_parser.parse_script(script_text)
                debug("使用本地解析结果作为最终输出")

            # 3. 回退到基础解析方法
            if final_result is None:
                debug("所有解析方法失败，回退到基础解析")
                self.parser_fallback.parse_script(script_text, result)
                final_result = self._ensure_correct_format(result)
                debug(f"基础解析完成，提取了 {len(final_result.get('scenes', []))} 个场景")

            # 4. 如果仍然没有场景，执行最后的兜底解析
            if not final_result.get("scenes"):
                debug("执行兜底解析逻辑")
                self.parser_fallback.parse_script(script_text, final_result)

            debug(f"剧本解析完成，提取了 {len(final_result.get('scenes', []))} 个场景")
            return final_result

        except Exception as e:
            print_log_exception()
            error(f"剧本解析失败: {str(e)}")
            # 返回最小化的结果结构
            return {"scenes": []}

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

    def enhance_with_llm(self, structured_script: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用LLM增强解析结果
        添加情绪识别和角色外观推断，同时确保画外音信息完整
        
        Args:
            structured_script: 结构化的剧本数据
            
        Returns:
            增强后的结构化剧本数据
        """
        if not self.llm:
            debug("未配置LLM，使用规则增强代替")
            return self._enhance_with_rules(structured_script)

        try:
            # 定义默认提示词模板 - 强调必须保留原始核心信息
            default_prompt = """
            请作为一个专业的中文剧本分析专家，对以下结构化剧本进行增强处理：
            
            【核心规则：绝对不能改变原始剧本的核心信息！】
            - 必须完全保留原始角色的姓名（如"林然"不能变成"林晓"）
            - 必须保留原始场景的地点（如"公寓客厅"不能变成"公寓卧室"）
            - 必须保留原始对话的内容，不能修改或创建新的对话
            - 必须保留原始动作的基本含义和顺序
            
            【增强任务】
            1. 为每个动作添加更细致的情绪标签
            2. 为每个角色丰富外观描述（年龄、穿着、外貌特征等）
            3. 细化场景氛围和环境细节
            4. 为动作添加微表情和身体语言描述
            5. 保持原始动作序列的顺序不变
            
            请严格按照输入的JSON结构返回增强后的结果，只在现有字段上添加或丰富内容，
            绝对不要修改原始剧本的核心信息。
            
            原始剧本：
            {script_json}
            """

            # 直接使用PromptManager获取提示词
            try:
                prompt_template = prompt_manager.get_prompt('enhance_script')
                debug("成功从PromptManager获取提示词模板")
            except Exception as e:
                debug(f"获取提示词失败，使用默认提示词: {str(e)}")
                prompt_template = default_prompt

            # 填充提示词模板
            filled_prompt = prompt_template.format(
                script_json=json.dumps(structured_script, ensure_ascii=False)
            )

            # 调用LLM
            debug("开始调用LLM增强剧本解析结果")
            response = self.llm.invoke(filled_prompt)

            # 初始化增强剧本变量
            enhanced_script = None

            # 尝试使用JSON响应解析器解析响应
            try:
                enhanced_script = parse_json_response(response)
                debug(f"LLM增强成功，返回增强后的剧本结构: {enhanced_script}")
            except Exception as e:
                # 记录错误并保持原有的异常处理流程
                error(f"JSON解析器处理失败: {str(e)}")
                raise

            # 确保返回格式正确
            if enhanced_script is not None:
                return self._ensure_correct_format(enhanced_script)
            else:
                raise json.JSONDecodeError("无法解析LLM响应为JSON格式", "", 0)
        except json.JSONDecodeError as e:
            print_log_exception()
            warning(f"LLM增强失败：响应不是有效的JSON格式: {str(e)}")
        except Exception as e:
            print_log_exception()
            # 检查是否是API密钥错误
            if "API key" in str(e) or "401" in str(e):
                warning(f"LLM增强失败：API密钥错误或权限不足: {str(e)}")
            else:
                warning(f"LLM增强失败，使用规则增强代替: {str(e)}")

        # 使用规则增强作为后备
        return self._enhance_with_rules(structured_script)

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
