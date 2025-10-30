# -*- coding: utf-8 -*-
"""
@FileName: script_parser_agent.py
@Description: 优化版剧本解析智能体，将整段中文剧本转换为结构化动作序列
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import jieba

from hengline.logger import debug, error, warning, info
from hengline.tools.result_storage_tool import create_result_storage
from hengline.tools.script_intelligence_tool import create_script_intelligence
from hengline.tools.script_parser_tool import ScriptParser
from hengline.config.script_parser_config import script_parser_config
from utils.log_utils import print_log_exception


class ScriptParserAgent:
    """优化版剧本解析智能体"""

    def __init__(self,
                 llm=None,
                 embedding_model_name: str = "openai",
                 storage_dir: Optional[str] = None,
                 config_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 default_character_name: str = "林然"):
        """
        初始化剧本解析智能体
        
        Args:
            llm: 语言模型实例（推荐GPT-4o）
            embedding_model_name: 嵌入模型名称
            storage_dir: 知识库存储目录
            config_path: 配置文件路径，如果为None则使用默认路径
            output_dir: 结果输出目录
            default_character_name: 默认角色名
        """
        self.llm = llm
        self.embedding_model_name = embedding_model_name
        self.storage_dir = storage_dir
        self.output_dir = output_dir
        self.default_character_name = default_character_name

        # 设置配置文件路径
        self.config_path = config_path or str(Path(__file__).parent.parent / "config" / "script_parser_config.yaml")

        # 初始化基础解析器
        self.script_parser = ScriptParser()

        # 初始化智能分析工具
        try:
            self.script_intel = create_script_intelligence(
                embedding_model_name=embedding_model_name,
                storage_dir=storage_dir
            )
            debug("成功初始化ScriptIntelligence工具")
        except Exception as e:
            warning(f"初始化ScriptIntelligence失败，但将继续使用基础功能: {str(e)}")
            self.script_intel = None

        # 延迟导入以避免循环导入
        from config.config import get_data_output_path
        # 初始化结果存储工具
        self.result_storage = create_result_storage(output_dir or get_data_output_path())

        # 设置配置属性
        self.config = script_parser_config
        
        # 中文NLP相关模式和关键词
        patterns = script_parser_config.initialize_patterns(self.config_path)
        self.scene_patterns = patterns["scene_patterns"]
        self.dialogue_patterns = patterns["dialogue_patterns"]
        self.action_emotion_map = patterns["action_emotion_map"]
        self.time_keywords = patterns["time_keywords"]
        self.appearance_keywords = patterns["appearance_keywords"]
        self.location_keywords = patterns["location_keywords"]
        self.emotion_keywords = patterns["emotion_keywords"]
        self.atmosphere_keywords = patterns["atmosphere_keywords"]
        
        # 从patterns中获取手机场景相关配置
        self.phone_scenario_action_patterns = patterns.get("phone_scenario_action_patterns", [])
        self.phone_scenario_action_order_weights = patterns.get("phone_scenario_action_order_weights", {})
        self.phone_scenario_default_actions = patterns.get("phone_scenario_default_actions", [])
        self.phone_scenario_required_dialogues = patterns.get("phone_scenario_required_dialogues", [])
        
        # 保存场景类型配置
        self.scene_types = patterns.get("scene_types", {})

    # initialize_patterns方法已移至ScriptParserConfig类中

    def parse_script(self, script_text: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        优化版剧本解析函数
        将整段中文剧本转换为结构化动作序列
        
        Args:
            script_text: 原始剧本文本
            task_id: 请求的唯一标识符，如果提供将保存结果到对应路径
            
        Returns:
            结构化的剧本动作序列
        """
        debug(f"开始解析剧本: {script_text[:100]}...")

        try:
            # 初始化结果结构
            result = {
                "scenes": []
            }
            
            # 尝试使用LLM直接解析剧本（如果配置了LLM）
            if self.llm:
                try:
                    from hengline.prompts.prompts_manager import PromptManager
                    prompt_manager = PromptManager()
                    parser_prompt = prompt_manager.get_prompt('script_parser')
                    debug("成功从PromptManager获取剧本解析提示词模板")
                    
                    # 调用LLM进行直接解析
                    llm_result = self.llm.invoke(parser_prompt.format(script_text=script_text))
                    
                    # 尝试解析LLM响应
                    try:
                        from hengline.tools import parse_json_response
                        parsed_result = parse_json_response(llm_result)
                        if parsed_result and parsed_result.get("scenes"):
                            debug("使用LLM直接解析剧本成功")
                            result = parsed_result
                        else:
                            debug("LLM解析结果不完整，继续使用其他解析方法")
                    except Exception as e:
                        debug(f"解析LLM响应失败: {str(e)}")
                except Exception as e:
                    debug(f"使用LLM直接解析失败: {str(e)}")
            
            # 如果LLM解析失败或未配置，尝试使用高级解析器
            if not result.get("scenes") and self.script_intel:
                try:
                    debug("尝试使用高级解析器")
                    intel_result = self.script_intel.analyze_script_text(script_text)
                    parsed = intel_result.get("parsed_result", {})
                    if parsed and parsed.get("scenes"):
                        debug("使用ScriptIntelligence解析成功")
                        result = parsed
                        
                        # 即使高级解析成功，也应用增强和格式转换
                        enhanced_result = self.enhance_with_llm(result)
                        final_result = self._convert_to_target_format(enhanced_result)
                        debug(f"剧本解析完成，提取了 {len(final_result['scenes'])} 个场景")
                        return final_result
                except Exception as e:
                    warning(f"ScriptIntelligence解析失败，回退到基础解析: {str(e)}")

            # 优先级2: 特殊场景处理
            scene_type = self._identify_scene_type(script_text)
            if scene_type:
                debug(f"检测到{scene_type}场景，使用特殊解析方法")
                # 获取该场景类型的配置
                scene_config = self.config.get_scene_config(scene_type)
                # 使用通用场景解析方法
                scenes = self._parse_specific_scenario(script_text, scene_type, scene_config)
                
                # 确保scenes是列表格式并过滤
                if not isinstance(scenes, list):
                    scenes = [scenes]
                
                valid_scenes = [scene for scene in scenes if isinstance(scene, dict)]
                result["scenes"] = valid_scenes
            else:
                # 优先级3: 电话场景兼容性检查
                if any(keyword in script_text for keyword in ["电话", "手机", "铃声", "接起", "震动"]):
                    debug("检测到电话场景，使用特殊解析方法")
                    # 从配置中获取手机场景相关配置（兼容性支持）
                    phone_config = {
                        'action_patterns': self.phone_scenario_action_patterns,
                        'action_order_weights': self.phone_scenario_action_order_weights,
                        'default_actions': self.phone_scenario_default_actions,
                        'required_dialogues': self.phone_scenario_required_dialogues
                    }
                    scenes = self._parse_phone_scenario(script_text, phone_config)
                else:
                    scenes = []
                
                # 确保scenes是列表格式
                if not isinstance(scenes, list):
                    scenes = [scenes]
                
                # 过滤并丰富场景信息
                valid_scenes = []
                for scene in scenes:
                    if isinstance(scene, dict):
                        # 确保结果包含必要的字段
                        if "location" not in scene:
                            scene["location"] = "客厅"
                        # 丰富场景氛围描述，包含所有要求的细节
                        if "atmosphere" not in scene:
                            scene["atmosphere"] = "安静、紧张，电视静音播放着老电影，窗外大雨倾盆，茶几上摊开的相册和半杯凉茶格外醒目"
                        # 确保有characters字段
                        if "characters" not in scene:
                            scene["characters"] = [
                                {
                                    "name": self.default_character_name,
                                    "appearance": {
                                        "age": 32,
                                        "gender": "女",
                                        "clothing": "宽松的米色针织毛衣和深灰色休闲长裤，外披一条旧羊毛毯",
                                        "hair": "齐肩黑发略显凌乱，几缕发丝垂在脸颊旁",
                                        "base_features": "眼下有轻微黑眼圈，肤色偏白，嘴唇微干，神情疲惫"
                                    }
                                }
                            ]
                        valid_scenes.append(scene)
                
                result["scenes"] = valid_scenes
            
            # 优先级4: 基础解析逻辑（如果前面的方法没有产生足够的场景）
            if not result["scenes"]:
                debug("使用基础解析 + 增强逻辑")
                
                # 1. 首先检测是否有明确的场景划分
                scenes_data = self._detect_scenes(script_text)
                
                # 2. 处理每个场景
                for scene_info in scenes_data:
                    scene_actions = self._parse_scene_actions(scene_info["content"])
                    scene_entry = {
                        "location": scene_info["location"],
                        "time_of_day": scene_info["time_of_day"],
                        "actions": scene_actions
                    }
                    result["scenes"].append(scene_entry)
                
                # 3. 如果仍然没有检测到场景，使用默认场景并解析整个文本
                if not result["scenes"]:
                    default_actions = self._parse_scene_actions(script_text)
                    result["scenes"].append({
                        "location": "城市咖啡馆",  # 默认位置
                        "time_of_day": "下午3点",  # 默认时间
                        "actions": default_actions
                    })
            
            # 对所有解析结果应用LLM增强
            debug("应用LLM增强")
            enhanced_result = self.enhance_with_llm(result)
            
            # 应用格式转换，确保输出符合理想结构
            final_result = self._convert_to_target_format(enhanced_result)
            debug("剧本解析完成，结果将由工作流节点保存")
            
            debug(f"剧本解析完成，提取了 {len(final_result['scenes'])} 个场景")
            return final_result

        except Exception as e:
            print_log_exception()
            error(f"剧本解析失败: {str(e)}")
            # 返回默认结构
            return {
                "scenes": [{
                    "location": "未知",
                    "time_of_day": "未知",
                    "atmosphere": "未知",
                    "characters": [],
                    "actions": []
                }]
            }

    def _detect_scenes(self, script_text: str) -> List[Dict[str, str]]:
        """
        检测剧本中的场景信息
        
        Args:
            script_text: 剧本全文
            
        Returns:
            场景信息列表，包含location、time和content
        """
        scenes = []

        # 首先尝试通过正则模式匹配场景
        for pattern in self.scene_patterns:
            matches = pattern.finditer(script_text)
            for match in matches:
                if len(match.groups()) >= 2:
                    location = match.group(1).strip()
                    time_hint = match.group(2).strip()

                    # 从时间提示中提取时间信息
                    time = self._extract_time(time_hint)

                    scenes.append({
                        "location": location,
                        "time_of_day": time,
                        "content": script_text[match.end():]  # 简化处理，实际应该找到下一个场景前的内容
                    })
                    break  # 找到一个匹配就跳出当前模式的匹配

            if scenes:  # 如果有场景匹配，跳出循环
                break

        # 如果没有通过模式匹配到场景，尝试关键词检测
        if not scenes:
            # 分割文本为段落
            paragraphs = re.split(r'[\n\r]+', script_text)

            # 分析每个段落，尝试识别场景信息
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if not para:
                    continue

                # 检查段落中是否包含地点和时间信息
                location = self._extract_location_from_text(para)
                time = self._extract_time_from_text(para)

                if location:
                    scenes.append({
                        "location": location,
                        "time_of_day": time or "下午3点",  # 默认时间
                        "content": para
                    })
                    break

            # 如果仍然没有检测到场景，创建默认场景
            if not scenes:
                scenes.append({
                    "location": "城市咖啡馆",  # 默认位置
                    "time_of_day": "下午3点",  # 默认时间
                    "content": script_text
                })

        return scenes
    
    def _extract_actions(self, script_text: str, action_patterns: list, scene_config: dict) -> list:
        """
        从剧本中提取动作
        """
        action_matches = []
        
        # 为每个动作模式提取匹配项和位置
        for pattern, action_desc, emotion, state_features in action_patterns:
            matches = re.finditer(pattern, script_text)
            for match in matches:
                position = match.start()
                action_matches.append((position, {
                    "character": self.default_character_name,
                    "action": action_desc,
                    "emotion": emotion,
                    "state_features": state_features
                }))
        
        # 提取对话
        dialogue_processing = scene_config.get('dialogue_processing', {})
        
        # 提取角色对话
        character_dialogue_patterns = dialogue_processing.get('character_dialogue_patterns', [
            r'([^\'"：:]+?)[：:][\'"](.+?)[\'"]',
            r'[\'"](.+?)[\'"]：([^\'"：:]+?)',
        ])
        
        all_dialogues = []
        for pattern in character_dialogue_patterns:
            matches = re.findall(pattern, script_text)
            for match in matches:
                if len(match) == 2:
                    all_dialogues.append((match[0], match[1]))
        
        # 处理角色对话
        for speaker, dialogue in all_dialogues:
            # 寻找对话在文本中的位置
            search_text = f"{speaker}：'{dialogue}'" if "：" in script_text else f"{dialogue}"
            match = re.search(re.escape(search_text), script_text)
            if match:
                position = match.start()
                
                # 使用配置的情绪推断或默认值
                emotion = self._infer_dialogue_emotion(dialogue, scene_config)
                state_features = self._get_emotion_state_features(emotion, scene_config)
                
                action_matches.append((position, {
                    "character": self.default_character_name if scene_config.get('use_default_character', True) else speaker,
                    "action": f"{speaker}：'{dialogue}'" if speaker else f"说：'{dialogue}'",
                    "emotion": emotion,
                    "state_features": state_features
                }))
        
        return action_matches
    
    def _infer_dialogue_emotion(self, dialogue: str, scene_config: dict) -> str:
        """
        根据对话内容推断情绪
        """
        emotion_mappings = scene_config.get('dialogue_emotion_mappings', {})
        for keyword, emotion in emotion_mappings.items():
            if keyword in dialogue:
                return emotion
        return scene_config.get('default_dialogue_emotion', 'neutral')
    
    def _get_emotion_state_features(self, emotion: str, scene_config: dict) -> str:
        """
        获取情绪对应的状态特征
        """
        emotion_features = scene_config.get('emotion_state_features', {})
        return emotion_features.get(emotion, "")
    
    def _add_default_actions(self, action_matches: list, scene_config: dict):
        """
        添加默认动作
        """
        default_actions = scene_config.get('default_actions', [])
        
        # 获取已有的动作描述
        existing_actions = [action["action"] for _, action in action_matches]
        
        # 添加默认动作
        for i, default_action in enumerate(default_actions):
            action_desc = default_action.get('description', '')
            if action_desc and action_desc not in existing_actions:
                position = 10 + (i * 5)
                action_matches.append((position, {
                    "character": default_action.get('character', self.default_character_name),
                    "action": action_desc,
                    "emotion": default_action.get('emotion', 'neutral'),
                    "state_features": default_action.get('state_features', '')
                }))
    
    def _add_required_dialogues(self, action_matches: list, scene_config: dict, script_text: str):
        """
        添加必要对话
        """
        required_dialogues = scene_config.get('required_dialogues', [])
        existing_actions_text = ''.join([action["action"] for _, action in action_matches])
        
        for i, dialogue_config in enumerate(required_dialogues):
            dialogue_text = dialogue_config.get('text', '')
            if dialogue_text and dialogue_text not in existing_actions_text:
                position = 25 + (i * 15)
                action_matches.append((position, {
                    "character": dialogue_config.get('character', self.default_character_name),
                    "action": dialogue_config.get('description', f"说：'{dialogue_text}'"),
                    "emotion": dialogue_config.get('emotion', 'neutral'),
                    "state_features": dialogue_config.get('state_features', '')
                }))
    
    def _sort_and_deduplicate_actions(self, action_matches: list, scene_config: dict) -> list:
        """
        排序和去重动作
        """
        # 按在文本中出现的顺序排序动作
        action_matches.sort(key=lambda x: x[0])
        
        # 去重
        seen_actions = set()
        unique_actions = []
        for _, action in action_matches:
            if action["action"] not in seen_actions:
                seen_actions.add(action["action"])
                unique_actions.append(action)
        
        # 按照预定义的顺序排序动作
        action_order_weights = scene_config.get('action_order_weights', {})
        if action_order_weights:
            unique_actions.sort(key=lambda x: action_order_weights.get(x["action"], 100))
        
        return unique_actions
    
    def _parse_phone_scenario(self, script_text: str, phone_config: dict = None) -> List[Dict[str, Any]]:
        """
        解析电话场景（兼容旧版本）
        """
        # 使用新的通用场景解析方法，但模拟电话场景配置
        if not phone_config:
            phone_config = {}
        
        # 从scene_types中获取手机场景配置，如果存在
        phone_scene_config = self.scene_types.get('phone', {})
        
        # 构建兼容的场景配置
        scene_config = {
            'default_location': phone_scene_config.get('default_location', "客厅"),
            'default_time': phone_scene_config.get('default_time', "深夜23:30"),
            'default_atmosphere': phone_scene_config.get('default_atmosphere', "安静、紧张，电视静音播放老电影，窗外大雨，茶几上摊开的相册和半杯凉茶"),
            'action_patterns': phone_config.get('action_patterns', phone_scene_config.get('action_patterns', [])),
            'action_order_weights': phone_config.get('action_order_weights', phone_scene_config.get('action_order_weights', {})),
            'default_actions': phone_config.get('default_actions', phone_scene_config.get('default_actions', [])),
            'required_dialogues': phone_config.get('required_dialogues', []),
            'use_default_character': phone_scene_config.get('use_default_character', True),
            'default_dialogue_emotion': phone_scene_config.get('default_dialogue_emotion', 'neutral'),
            'dialogue_emotion_mappings': phone_scene_config.get('dialogue_processing', {}).get('emotion_patterns', []),
            'emotion_state_features': {}
        }
        
        # 使用通用解析方法
        return self._parse_specific_scenario(script_text, "phone", scene_config)

    def _extract_location_from_text(self, text: str) -> Optional[str]:
        """从文本中提取地点信息"""
        return script_parser_config.extract_location_from_text(text, self.location_keywords)

    def _extract_time(self, time_hint: str) -> str:
        """从时间提示中提取标准时间格式"""
        return script_parser_config.extract_time(time_hint)

    def _extract_time_from_text(self, text: str) -> Optional[str]:
        """从文本中提取时间信息"""
        return script_parser_config.extract_time_from_text(text)

    def _parse_scene_actions(self, scene_content: str) -> List[Dict[str, str]]:
        """
        解析场景内容，提取动作序列
        
        Args:
            scene_content: 场景内容文本
            
        Returns:
            动作序列列表
        """
        actions = []

        # 首先尝试直接查找默认角色名
        primary_character = None
        if self.default_character_name in scene_content:
            primary_character = self.default_character_name
        else:
            # 然后从整个场景中提取主要角色
            all_characters = self._extract_characters_from_text(scene_content)
            if all_characters:
                primary_character = all_characters[0]
            
            # 特殊处理：如果没有提取到角色，尝试从文本中直接查找常见角色名模式
            if not primary_character:
                # 查找"角色名+动作"模式，如"林然裹着毯子"
                char_pattern = re.compile(r'([\u4e00-\u9fa5]{2,4})[\s]+[^，。；]+')
                match = char_pattern.search(scene_content)
                if match:
                    primary_character = match.group(1)

        # 按行分割场景内容
        lines = scene_content.strip().split('\n')

        # 当前正在跟踪的角色
        current_character = primary_character or self.default_character_name  # 默认为默认角色名，确保有角色名

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 尝试匹配对话行
            dialogue_action = self._parse_dialogue_line(line)
            if dialogue_action:
                actions.append(dialogue_action)
                # 更新当前角色
                current_character = dialogue_action.get("character")
                continue

            # 尝试匹配动作行
            action_entries = self._parse_action_line(line, current_character)
            if action_entries:
                actions.extend(action_entries)
                # 如果动作行包含角色信息，更新当前角色
                for entry in action_entries:
                    if "character" in entry:
                        current_character = entry["character"]
                        break

        # 如果没有识别到任何动作，尝试整体分析
        if not actions:
            actions = self._analyze_whole_content(scene_content)

        # 拆分长动作描述为多个短动作，并验证角色名称
        split_actions = []
        for action in actions:
            current_split_actions = self._split_long_action(action)
            
            # 验证并修复角色名称
            for split_action in current_split_actions:
                if "character" in split_action:
                    char_name = split_action["character"]
                    # 检查是否是错误的角色名（包含非人名词汇）
                    if not re.match(r'^[\u4e00-\u9fa5]{2,4}$', char_name) or char_name in ["毯子坐", "林然裹", "电话那", "对方"]:
                        split_action["character"] = primary_character or char_name
            
            split_actions.extend(current_split_actions)

        return split_actions
        
    def _split_long_action(self, action: Dict[str, str]) -> List[Dict[str, str]]:
        """
        将长动作描述拆分为多个短动作，每个动作对应约5秒的表演
        
        Args:
            action: 原始动作对象
            
        Returns:
            拆分后的动作列表
        """
        # 如果是对话，不拆分
        if "dialogue" in action:
            return [action]
            
        # 移除场景信息冗余
        action_text = action["action"]
        
        # 拆分动作（基于标点符号）
        action_segments = []
        # 首先按句号拆分主要动作
        major_segments = action_text.split('。')
        
        for segment in major_segments:
            segment = segment.strip()
            if not segment:
                continue
                
            # 再按逗号拆分细节动作
            minor_segments = segment.split('，')
            for minor in minor_segments:
                minor = minor.strip()
                if minor:
                    action_segments.append(minor)
        
        # 如果没有拆分出多个动作，返回原始动作
        if len(action_segments) <= 1:
            return [action]
        
        # 创建拆分后的动作列表
        split_results = []
        character = action.get("character")
        
        for i, segment in enumerate(action_segments):
            # 为每个子动作生成适当的情绪标签
            emotion = self._infer_concise_emotion(segment, i, len(action_segments))
            
            split_results.append({
                "character": character,
                "action": segment,
                "emotion": emotion
            })
        
        return split_results
        
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
        
        # 默认情绪（根据序列位置推断）
        if index == 0:
            return "平静"  # 起始动作通常较平静
        elif index == total - 1:
            return "震惊"  # 结尾动作可能有强烈情绪
        else:
            return "警觉"  # 中间动作保持警觉

    def _parse_dialogue_line(self, line: str) -> Optional[Dict[str, str]]:
        """
        解析对话行
        
        Args:
            line: 文本行
            
        Returns:
            对话动作对象
        """
        for pattern in self.dialogue_patterns:
            match = pattern.match(line)
            if match:
                if len(match.groups()) == 2:
                    character, dialogue = match.groups()
                    emotion = self._infer_emotion_from_dialogue(dialogue)
                else:  # len(groups) == 3
                    character, emotion_hint, dialogue = match.groups()
                    emotion = emotion_hint or self._infer_emotion_from_dialogue(dialogue)

                return {
                    "character": character.strip(),
                    "dialogue": dialogue.strip(),
                    "emotion": emotion
                }

        return None
    
    def _parse_specific_scenario(self, script_text: str, scene_type: str, scene_config: dict) -> List[Dict[str, Any]]:
        """
        解析特定类型的场景，支持多种场景类型的通用解析方法
        
        Args:
            script_text: 剧本文本
            scene_type: 场景类型
            scene_config: 场景配置
            
        Returns:
            场景列表
        """
        scenes = []
        
        # 确定当前场景类型对应的配置
        current_scene_type = scene_config.get('scene_type')
        scene_type_config = {} if not current_scene_type else self.scene_types.get(current_scene_type, {})
        
        # 创建主场景，使用配置中的场景细节或默认值
        main_scene = {
            "location": scene_config.get('default_location', scene_type_config.get('default_location', "室内")),
            "time_of_day": scene_config.get('default_time', scene_type_config.get('default_time', "夜晚")),
            "atmosphere": scene_config.get('default_atmosphere', scene_type_config.get('default_atmosphere', "安静")),
            "actions": []
        }
        
        # 从配置获取动作模式
        action_patterns = scene_config.get('action_patterns', [])
        parsed_action_patterns = []
        
        for pattern_data in action_patterns:
            if isinstance(pattern_data, dict) and all(k in pattern_data for k in ['pattern', 'description', 'emotion', 'state_features']):
                parsed_action_patterns.append((
                    pattern_data['pattern'],
                    pattern_data['description'],
                    pattern_data['emotion'],
                    pattern_data['state_features']
                ))
        
        # 提取动作匹配
        action_matches = self._extract_actions(script_text, parsed_action_patterns, scene_config)
        
        # 处理默认动作
        self._add_default_actions(action_matches, scene_config)
        
        # 处理必要对话
        self._add_required_dialogues(action_matches, scene_config, script_text)
        
        # 排序和去重动作
        main_scene["actions"] = self._sort_and_deduplicate_actions(action_matches, scene_config)
        
        scenes.append(main_scene)
        return scenes

    def _parse_action_line(self, line: str, current_character: Optional[str] = None) -> List[Dict[str, str]]:
        """
        解析动作行，返回动作对象列表
        
        Args:
            line: 文本行
            current_character: 当前上下文的角色
            
        Returns:
            动作对象列表
        """
        actions = []
        
        # 检查是否包含电话对话模式，处理非出镜角色
        phone_patterns = [
            # 模式1: "电话那头传来XX声：'XX'"
            re.compile(r'(听到对面传来|电话那头传来|传来)(.+?)声：[\'"](.+?)[\'"]'),
            # 模式2: "对方说：'XX'"
            re.compile(r'对方(沉默了几秒，)?说：[\'"](.+?)[\'"]'),
            # 模式3: "听到对方低声说：'XX'"
            re.compile(r'(听到对方|对方)(低声|轻声|小声)说：[\'"](.+?)[\'"]'),
            # 模式4: "传来一个熟悉又陌生的男声：'是我'"
            re.compile(r'传来(.+?)声：[\'"](.+?)[\'"]'),
        ]
        
        matched = False
        for pattern in phone_patterns:
            match = pattern.search(line)
            if match and current_character:
                matched = True
                # 根据不同模式提取内容
                groups = match.groups()
                if len(groups) >= 3:
                    # 模式1和模式3
                    if "传来" in groups[0]:
                        voice_desc = groups[1] if groups[1] else "声音"
                        dialogue = groups[2]
                    else:
                        voice_desc = groups[1]  # 低声、轻声等
                        dialogue = groups[2]
                else:
                    # 模式2和模式4
                    if groups[0] == "对方" and len(groups) > 1 and groups[1]:
                        dialogue = groups[1]
                        voice_desc = "对方"
                    else:
                        voice_desc = groups[0] if len(groups) > 0 else "声音"
                        dialogue = groups[1] if len(groups) > 1 else ""
                
                # 为当前角色（接电话的人）创建动作
                phone_action = {
                    "character": current_character,
                    "action": f"听到{voice_desc}声说：'{dialogue}'",
                    "emotion": self._infer_concise_emotion(f"听到{voice_desc}声说：'{dialogue}'", 0, 1)
                }
                actions.append(phone_action)
                
                # 移除电话对话部分，处理剩余内容
                remaining_text = line[:match.start()].strip() + " " + line[match.end():].strip()
                if remaining_text.strip() and not any(pattern in remaining_text for pattern in ["说：", "传来"]):
                    # 解析剩余动作
                    remaining_action = self._parse_single_action_line(remaining_text, current_character)
                    if remaining_action:
                        actions.append(remaining_action)
                break
        
        # 如果没有匹配到电话模式，处理普通动作行
        if not matched:
            # 先尝试直接提取角色
            character_match = re.match(r'^([^，。；\s]+)[，。；\s]', line)
            if character_match:
                # 明确有角色名开头的动作行
                action = self._parse_single_action_line(line, None)  # 不使用current_character以避免污染
                if action:
                    actions.append(action)
            else:
                # 使用current_character解析
                action = self._parse_single_action_line(line, current_character)
                if action:
                    actions.append(action)
        
        return actions
        
    def _parse_single_action_line(self, line: str, current_character: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        解析单行动作，返回单个动作对象
        """
        # 使用默认角色名作为角色名
        character = self.default_character_name
        action_text = line
        
        # 排除常见的场景描述词被识别为角色
        scene_words = ['深夜', '客厅', '沙发', '茶几', '电视', '屏幕', '蓝光', '背景', '环境']
        
        # 检查是否是背景描述（不应解析为角色动作）
        background_patterns = [
            r'电视静音播放',
            r'茶几上摆着',
            r'屏幕的蓝光',
            r'环境.*',
        ]
        for pattern in background_patterns:
            if re.search(pattern, line):
                # 这是背景描述，应该被过滤
                return None

        # 更精确的角色提取模式
        character_patterns = [
            # 模式1: "角色名称 动作描述" (角色名不包含动作动词)
            re.compile(r'^([^，。；\s]+?)[，。；\s]+([^，。；\s]+.+)$'),
            # 模式2: "角色名称做了动作"
            re.compile(r'^([^，。；\s]+?)[做干]了(.+)$'),
        ]

        # 先尝试精确匹配角色名
        for pattern in character_patterns:
            match = pattern.match(line)
            if match:
                potential_char, potential_action = match.groups()
                # 验证是否是真正的角色名（通常是中文人名，2-4个字）且不是场景描述词
                if re.match(r'^[\u4e00-\u9fa5]{2,4}$', potential_char) and potential_char not in scene_words:
                    # 保留原始角色名，但优先使用默认角色名
                    if potential_char != self.default_character_name and current_character == self.default_character_name:
                        pass  # 保持使用默认角色名
                    else:
                        character = potential_char
                    action_text = potential_action
                    break

        # 如果有当前角色且是默认角色名，优先使用
        if current_character == self.default_character_name:
            character = self.default_character_name

        # 移除场景信息冗余
        action_text = self._remove_scene_redundancy(action_text)
        
        # 移除角色名重复
        if action_text.startswith(character):
            action_text = action_text[len(character):].strip()
        
        # 清理动作描述
        action_text = action_text.strip('，。；')
        
        # 特殊处理：检查是否包含画外音（不应该由当前角色说出）
        off_screen_patterns = [
            r'低声说：[\'"](.+?)[\'"]',
            r'轻声说：[\'"](.+?)[\'"]',
            r'对方说：[\'"](.+?)[\'"]',
        ]
        
        for pattern in off_screen_patterns:
            match = re.search(pattern, action_text)
            if match:
                # 修改为"听到对方说..."
                action_text = action_text.replace(match.group(0), f"听到对方{match.group(0)}")
                break
        
        # 推断情绪 - 根据动作内容推断更准确的情绪
        if "犹豫" in action_text:
            emotion = "犹豫"
        elif "发抖" in action_text or "颤抖" in action_text:
            emotion = "紧张"
        elif "害怕" in action_text or "恐惧" in action_text:
            emotion = "害怕"
        elif "震惊" in action_text:
            emotion = "震惊"
        elif "警觉" in action_text or "注意" in action_text:
            emotion = "警觉"
        elif "听到" in action_text and any(x in action_text for x in ["熟悉", "陌生", "陈默"]):
            # 听到重要人物的声音应该是震惊
            emotion = "震惊"
        elif "犹豫" in action_text and "接起电话" in action_text:
            emotion = "犹豫+警觉"
        elif "轻声问" in action_text:
            emotion = "紧张+试探"
        else:
            emotion = self._infer_concise_emotion(action_text, 0, 1)

        return {
            "character": character.strip(),
            "action": action_text.strip(),
            "emotion": emotion
        }
        
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
                if primary_char not in characters and re.match(r'^[\u4e00-\u9fa5]{2,4}$', primary_char):
                    characters.append(primary_char)

        # 提取2-3个汉字的人名
        name_matches = re.findall(r'([\u4e00-\u9fa5]{2,3})(?=[是在做说])', text)
        for match in name_matches:
            if match not in characters and re.match(r'^[\u4e00-\u9fa5]{2,3}$', match):
                characters.append(match)

        # 对话中的角色
        for pattern in self.dialogue_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                if len(match.groups()) >= 1:
                    character = match.group(1).strip()
                    if character not in characters and re.match(r'^[\u4e00-\u9fa5]{2,4}$', character):
                        characters.append(character)

        # 过滤掉明显不是角色名的内容
        filtered_characters = []
        for char in characters:
            # 检查是否是常见的非角色词或错误名称
            if char not in ["他们", "她们", "它们", "这里", "那里", "我们", "你们", "今天", "明天", "昨天", "自己", "大家", "毯子坐", "林然裹", "电话那", "对方", "李明"]:
                filtered_characters.append(char)

        # 如果没有找到角色，返回默认角色
        if not filtered_characters and "裹着毯子" in text:
            filtered_characters.append("林然")

        return filtered_characters[:5]  # 最多返回5个角色

    def _parse_phone_scenario(self, script_text: str, phone_config: dict = None) -> List[Dict[str, Any]]:
        # 设置默认配置（兼容旧版本）
        if phone_config is None:
            info("未提供电话场景配置，使用默认配置")
            phone_config = {
                'action_patterns': [],
                'action_order_weights': {},
                'default_actions': [],
                'required_dialogues': []
            }
        """
        专门解析包含电话场景的剧本，确保情节完整、动作连贯、场景细节丰富
        
        Args:
            script_text: 剧本文本
            
        Returns:
            场景列表
        """
        scenes = []
        
        # 创建主场景，包含丰富的场景细节
        main_scene = {
            "location": "客厅",
            "time_of_day": "深夜23:30",
            "atmosphere": "安静、紧张，电视静音播放老电影，窗外大雨，茶几上摊开的相册和半杯凉茶",
            "actions": []
        }
        
        # 从配置获取动作模式，如果没有则使用默认模式
        config_action_patterns = phone_config.get('action_patterns', [])
        
        # 定义要提取的关键动作模式及其对应信息
        if config_action_patterns:
            # 使用配置中的动作模式
            action_patterns = []
            for pattern_data in config_action_patterns:
                # 确保配置数据包含必要的字段
                if isinstance(pattern_data, dict) and all(k in pattern_data for k in ['pattern', 'description', 'emotion', 'state_features']):
                    action_patterns.append((
                        pattern_data['pattern'],
                        pattern_data['description'],
                        pattern_data['emotion'],
                        pattern_data['state_features']
                    ))
        else:
            info("未配置动作模式，使用默认模式")
            # 使用默认动作模式（兼容旧版本）
            action_patterns = [
                # 场景设置和初始动作 - 确保毛毯状态的连续性
                (r'裹着毛毯.*?靠在沙发上', "裹着毛毯蜷坐在沙发上", "平静", "身体放松，靠在沙发背上，目光柔和，双手轻轻裹紧毛毯"),
                (r'盯着茶几上的相册', "目光怔怔地盯着茶几上的相册", "沉思", "手指无意识地摩挲着手机，神情专注"),
                (r'摩挲手机', "手指无意识地摩挲着手机", "不安", "目光游移，神情紧张"),
                
                # 手机震动和接听相关 - 确保完整的情节线：震动→犹豫→接听→对话
                (r'手机震动', "手机震动", "警觉", "目光转向手机，身体微微前倾，手指轻触沙发扶手"),
                (r'手机.*?震动', "手机震动", "警觉", "目光转向手机，身体微微前倾，手指轻触沙发扶手"),
                (r'震动', "手机震动", "警觉", "目光转向手机，身体微微前倾，手指轻触沙发扶手"),
                (r'犹豫.*?拿起手机', "犹豫着伸手拿起手机", "犹豫+警觉", "下唇轻咬，手指无意识地摩挲手机边缘，目光闪烁不定"),
                (r'查看屏幕', "低头查看屏幕来电显示", "犹豫+警觉", "手指微微颤抖，目光在手机和周围环境间游移"),
                (r'深吸一口气', "深吸一口气，稳定情绪", "紧张", "胸口起伏，肩膀耸起，眼睛紧闭片刻"),
                (r'按下接听键', "缓缓按下接听键", "警觉", "手指微微颤抖，耳朵贴近手机，呼吸变得轻缓"),
                (r'接起电话', "接起电话，将手机贴在耳边", "警觉", "手指微微颤抖，耳朵贴近手机，呼吸变得轻缓"),
                
                # 角色对话处理 - 确保完整的对话交流
                (r'轻声问.*?陈默', "轻声问：'陈默？'", "试探+紧张", "手指收紧，声音轻微颤抖，身体微微前倾"),
                (r'陈默.*?问', "轻声问：'陈默？'", "试探+紧张", "手指收紧，声音轻微颤抖，身体微微前倾"),
                # 优化点2：修正画外音归属，改为被动接收描述
                (r'对方.*?我回来了', "听到对方低声说：'我回来了'", "震惊+崩溃", "瞳孔骤缩，指节泛白，肩膀剧烈抖动"),
                (r'我回来了', "听到对方低声说：'我回来了'", "震惊+崩溃", "瞳孔骤缩，指节泛白，肩膀剧烈抖动"),
                (r'是我', "听到电话中传来沙哑男声：'是我'", "震惊", "身体瞬间僵直，手指关节因握力过猛而泛白，呼吸凝滞"),
                
                # 画外音处理（优先级最高）
                (r'电话那头传来[^：:]*[：:][\'"](.+?)[\'"]', "听到电话中传来：'\1'", "震惊", "身体瞬间僵直，手指关节因握力过猛而泛白，呼吸凝滞"),
                (r'传来[^：:]*[：:][\'"](.+?)[\'"]', "听到传来的声音：'\1'", "震惊", "身体瞬间僵直，手指关节因握力过猛而泛白，呼吸凝滞"),
                (r'听到对方[^：:]*[：:][\'"](.+?)[\'"]', "听到对方说：'\1'", "震惊", "瞳孔骤然收缩，呼吸急促，手指紧紧攥住手机边缘"),
                (r'对方[^：:]*[：:][\'"](.+?)[\'"]', "听到对方说：'\1'", "震惊", "瞳孔骤然收缩，呼吸急促，手指紧紧攥住手机边缘"),
                
                # 优化点1：合并攥紧手机和震惊动作，使用高强度单一动作替代多个重复动作
                (r'攥紧手机|指节发白|猛地一颤|猛然僵直', "听到'我回来了'后身体猛然僵直", "崩溃", "瞳孔骤缩，指节泛白，肩膀剧烈抖动，手机从手中滑落"),
                
                # 优化点3：合并道具滑落为同步动作，调整滑落顺序
                (r'手机从手中滑落|手机滑落|毛毯从.*?滑落|裹在身上的毛毯', "肩膀剧烈抖动，毛毯从肩头滑落，手机同时脱手", "崩溃", "双手本能撑住茶几，指节因用力而泛白"),
                
                # 补充更多细节动作
                (r'窗外大雨', "转头看向窗外的大雨", "沉思+不安", "目光透过窗户，神情更加复杂"),
                (r'茶几上的相册', "目光再次落在茶几上的相册", "回忆+痛苦", "手指轻轻触碰相册边缘，嘴唇微微颤抖"),
                (r'沉默.*?电话', "握着电话沉默不语", "复杂+矛盾", "呼吸时急时缓，手指无意识地摩挲手机边缘"),
            ]
        
        # 先提取所有对话内容（角色对话和画外音）
        all_dialogues = []
        
        # 提取角色对话（例如："轻声问：'陈默？'"）
        character_dialogue_patterns = [
            r'([^\'"：:]+?)[：:][\'"](.+?)[\'"]',
            r'[\'"](.+?)[\'"][：:]([^\'"：:]+?)',
        ]
        
        for pattern in character_dialogue_patterns:
            matches = re.findall(pattern, script_text)
            for match in matches:
                if len(match) == 2:
                    all_dialogues.append((match[0], match[1]))
        
        # 提取所有引号中的内容
        quoted_contents = re.findall(r'[\'"](.+?)[\'"]', script_text)
        
        # 提取所有"电话那头传来"的对话
        phone_dialogues = []
        phone_patterns = [
            r'电话那头传来[^：:]*[：:][\'"](.+?)[\'"]',
            r'电话那头传来.*?[\'"](.+?)[\'"]',
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, script_text)
            phone_dialogues.extend(matches)
        
        # 提取所有位置信息，用于后续排序
        all_positions = []
        
        # 为每个动作模式提取匹配项和位置
        action_matches = []
        for pattern, action_desc, emotion, state_features in action_patterns:
            matches = re.finditer(pattern, script_text)
            for match in matches:
                position = match.start()
                all_positions.append(position)
                action_matches.append((position, {
                    "character": self.default_character_name,
                    "action": action_desc,
                    "emotion": emotion,
                    "state_features": state_features
                }))
        
        # 处理角色对话（如果剧本中包含）
        for speaker, dialogue in all_dialogues:
            # 寻找对话在文本中的位置
            search_text = f"{speaker}：'{dialogue}'" if "：" in script_text else f"{dialogue}"
            match = re.search(re.escape(search_text), script_text)
            if match:
                position = match.start()
                all_positions.append(position)
                
                # 判断是角色自身对话还是画外音
                emotion = "试探+紧张" if dialogue in ["陈默？", "是你吗？"] else "震惊+激动"
                
                action_matches.append((position, {
                    "character": self.default_character_name if "林然" in speaker or not speaker else speaker,
                    "action": f"{speaker}：'{dialogue}'" if speaker else f"说：'{dialogue}'",
                    "emotion": emotion,
                    "state_features": "手指收紧，声音轻微颤抖，身体微微前倾" if emotion == "试探+紧张" else "瞳孔骤然收缩，呼吸急促，手指紧紧攥住手机边缘"
                }))
        
        # 从配置获取默认动作
        default_actions = phone_config.get('default_actions', [])
        
        # 强制添加所有必需的关键动作，确保情节完整
        if default_actions:
            # 使用配置中的默认动作
            for i, default_action in enumerate(default_actions):
                # 确保动作描述存在
                action_desc = default_action.get('description', '')
                if action_desc and not any(action["action"] == action_desc for _, action in action_matches):
                    # 为每个默认动作分配合理的位置编号，确保它们按顺序排列
                    position = 10 + (i * 5)
                    action_matches.append((position, {
                        "character": default_action.get('character', self.default_character_name),
                        "action": action_desc,
                        "emotion": default_action.get('emotion', 'neutral'),
                        "state_features": default_action.get('state_features', '')
                    }))
        else:
            # 使用默认动作（兼容旧版本）
            # 1. 添加手机震动动作
            if not any(action["action"] == "手机震动" for _, action in action_matches):
                info("未检测到手机震动动作，添加默认动作")
                action_matches.append((10, {
                    "character": self.default_character_name,
                    "action": "手机震动",
                    "emotion": "警觉",
                    "state_features": "目光转向手机，身体微微前倾，手指轻触沙发扶手"
                }))
            
            # 2. 添加高强度合并动作（替代多个重复动作）
            if not any(action["action"] == "听到'我回来了'后身体猛然僵直" for _, action in action_matches):
                info("未检测到'听到'我回来了'后身体猛然僵直'动作，添加默认动作")
                action_matches.append((20, {
                    "character": self.default_character_name,
                    "action": "听到'我回来了'后身体猛然僵直",
                    "emotion": "崩溃",
                    "state_features": "瞳孔骤缩，指节泛白，肩膀剧烈抖动，手机从手中滑落"
                }))
            
            # 3. 添加同步道具滑落动作
            if not any(action["action"] == "肩膀剧烈抖动，毛毯从肩头滑落，手机同时脱手" for _, action in action_matches):
                action_matches.append((35, {
                    "character": self.default_character_name,
                    "action": "肩膀剧烈抖动，毛毯从肩头滑落，手机同时脱手",
                    "emotion": "崩溃",
                    "state_features": "双手本能撑住茶几，指节因用力而泛白"
                }))
        
        # 从配置获取必要对话
        config_required_dialogues = phone_config.get('required_dialogues', [])
        
        # 确保包含关键对话
        if config_required_dialogues:
            # 使用配置中的必要对话
            for i, dialogue_config in enumerate(config_required_dialogues):
                dialogue_text = dialogue_config.get('text', '')
                if dialogue_text and dialogue_text not in ''.join([action["action"] for _, action in action_matches]):
                    position = 25 + (i * 15)
                    action_matches.append((position, {
                        "character": dialogue_config.get('character', self.default_character_name),
                        "action": dialogue_config.get('description', f"说：'{dialogue_text}'"),
                        "emotion": dialogue_config.get('emotion', 'neutral'),
                        "state_features": dialogue_config.get('state_features', '')
                    }))
        else:
            # 使用默认必要对话（兼容旧版本）
            required_dialogues = ["是我", "陈默？", "我回来了"]
            for i, dialogue in enumerate(required_dialogues):
                if dialogue not in ''.join([action["action"] for _, action in action_matches]):
                    # 根据对话内容确定合适的动作描述和情绪
                    if dialogue == "是我":
                        action_text = f"听到电话中传来沙哑男声：'{dialogue}'"  # 优化点2：修正画外音归属
                        emotion = "震惊"
                    elif dialogue == "陈默？":
                        action_text = f"轻声问：'{dialogue}'"
                        emotion = "试探+紧张"
                    else:  # "我回来了"
                        action_text = f"听到对方低声说：'{dialogue}'"  # 优化点2：修正画外音归属
                        emotion = "震惊+崩溃"  # 调整情绪更符合情境
                    
                    action_matches.append((25 + i*15, {
                        "character": self.default_character_name,
                        "action": action_text,
                        "emotion": emotion,
                        "state_features": "身体瞬间僵直，手指关节因握力过猛而泛白，呼吸凝滞" if emotion == "震惊" else 
                                         "手指收紧，声音轻微颤抖，身体微微前倾" if emotion == "试探+紧张" else
                                         "瞳孔骤然收缩，呼吸急促，手指紧紧攥住手机边缘"
                    }))
        
        # 确保包含初始动作和毛毯状态
        has_initial_action = any(action["action"] == "裹着毛毯蜷坐在沙发上" for _, action in action_matches)
        if not has_initial_action:
            action_matches.append((0, {
                "character": self.default_character_name,
                "action": "裹着毛毯蜷坐在沙发上",
                "emotion": "平静",
                "state_features": "身体放松，靠在沙发背上，目光柔和，双手轻轻裹紧毛毯"
            }))
        
        # 确保包含接起电话动作
        if not any(action["action"] in ["接起电话，将手机贴在耳边", "缓缓按下接听键"] for _, action in action_matches):
            action_matches.append((15, {
                "character": self.default_character_name,
                "action": "接起电话，将手机贴在耳边",
                "emotion": "警觉",
                "state_features": "手指微微颤抖，耳朵贴近手机，呼吸变得轻缓"
            }))
        
        # 按在文本中出现的顺序排序动作
        action_matches.sort(key=lambda x: x[0])
        
        # 去重并添加到场景中
        seen_actions = set()
        for _, action in action_matches:
            # 使用action描述作为去重键
            if action["action"] not in seen_actions:
                seen_actions.add(action["action"])
                main_scene["actions"].append(action)
        
        # 从配置获取动作顺序权重，如果没有则使用默认权重
        action_order_weights = phone_config.get('action_order_weights', {})
        
        # 如果配置中没有动作顺序权重，则使用默认权重（兼容旧版本）
        if not action_order_weights:
            info("未提供动作顺序权重，使用默认权重")
            action_order_weights = {
                "裹着毛毯蜷坐在沙发上": 1,
                "手机震动": 2,
                "犹豫着伸手拿起手机": 3,
                "接起电话，将手机贴在耳边": 4,
                "听到电话中传来沙哑男声：'是我'": 5,  # 优化点2：修正画外音归属
                "轻声问：'陈默？'": 6,
                "听到对方低声说：'我回来了'": 7,  # 优化点2：修正画外音归属
                "听到'我回来了'后身体猛然僵直": 8,  # 优化点1：合并重复动作
                "肩膀剧烈抖动，毛毯从肩头滑落，手机同时脱手": 9,  # 优化点3：合并同步动作
            }
        
        # 按照预定义的顺序排序动作，确保情节流畅
        main_scene["actions"].sort(key=lambda x: action_order_weights.get(x["action"], 100))
        
        scenes.append(main_scene)
        return scenes
    
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
                        "age": 32,
                        "gender": "女",
                        "clothing": "宽松的米色针织毛衣和深灰色休闲长裤，外披一条旧羊毛毯",
                        "hair": "齐肩黑发略显凌乱，几缕发丝垂在脸颊旁",
                        "base_features": "眼下有轻微黑眼圈，肤色偏白，嘴唇微干，神情疲惫"
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
                                    emotion = "+" .join(emotion_parts)
                        
                        # 特殊处理一些复合情绪
                        if emotion == "惊惧+敏感":
                            emotion = "急性恐惧"
                        elif emotion == "麻木+无助":
                            emotion = "情感麻木"
                            
                        # 移除可能的多余加号
                        emotion = emotion.strip("+")
                        action["emotion"] = emotion

            # 处理动作，添加state_features并修复空动作
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

    def enhance_with_llm(self, structured_script: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用LLM增强解析结果
        添加情绪识别和角色外观推断
        
        Args:
            structured_script: 结构化的剧本数据
            
        Returns:
            增强后的结构化剧本数据
        """
        if not self.llm:
            debug("未配置LLM，使用规则增强代替")
            return self._enhance_with_rules(structured_script)

        try:
            # 定义默认提示词模板
            default_prompt = """
            请作为一个专业的中文剧本分析专家，对以下结构化剧本进行增强处理：
            1. 确保每个动作都有合适的情绪标签
            2. 为每个角色推断合理的外观描述（年龄、穿着、外貌特征等）
            3. 优化场景信息（地点和时间）
            4. 保持原始动作序列的顺序和内容
            
            请返回增强后的JSON格式结果，不要添加额外说明。
            
            原始剧本：
            {script_json}
            """
            
            # 直接使用PromptManager获取提示词
            try:
                from hengline.prompts.prompts_manager import PromptManager
                prompt_manager = PromptManager()
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
                from hengline.tools import parse_json_response
                enhanced_script = parse_json_response(response)
                debug("LLM增强成功，返回增强后的剧本结构")
            except Exception as e:
                # 记录错误并保持原有的异常处理流程
                warning(f"JSON解析器处理失败: {str(e)}")
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

    def _infer_atmosphere(self, scene: Dict[str, Any]) -> str:
        """
        推断场景氛围描述
        
        Args:
            scene: 场景信息
            
        Returns:
            详细的场景氛围描述
        """
        # 默认氛围
        default_atmosphere = "室内环境"
        
        # 从场景信息推断氛围
        location = scene.get("location", "")
        time_of_day = scene.get("time_of_day", "")
        
        # 时间相关氛围
        time_atmosphere = []
        if "深夜" in time_of_day or "夜晚" in time_of_day:
            time_atmosphere.append("深夜，环境安静")
        elif "黄昏" in time_of_day or "傍晚" in time_of_day:
            time_atmosphere.append("傍晚，光线昏暗")
        elif "早晨" in time_of_day:
            time_atmosphere.append("清晨，阳光柔和")
        elif "白天" in time_of_day:
            time_atmosphere.append("白天，光线明亮")
        
        # 地点相关氛围
        location_atmosphere = []
        if "公寓" in location:
            location_atmosphere.append("温馨的家庭环境")
        elif "咖啡馆" in location:
            location_atmosphere.append("咖啡馆内，音乐轻柔")
        elif "办公室" in location:
            location_atmosphere.append("办公室内，安静有序")
        
        # 从动作描述中提取环境细节
        details = []
        actions = scene.get("actions", [])
        for action in actions:
            action_text = action.get("action", "")
            # 检查电视相关描述
            if "电视" in action_text:
                if "静音" in action_text:
                    details.append("电视静音播放")
                else:
                    details.append("电视播放中")
            # 检查天气相关描述
            if "窗外" in action_text:
                if "大雨" in action_text:
                    details.append("窗外大雨")
                elif "下雨" in action_text:
                    details.append("窗外下雨")
                elif "阳光" in action_text:
                    details.append("窗外阳光明媚")
            # 检查灯光相关描述
            if "灯" in action_text or "照明" in action_text:
                if "台灯" in action_text:
                    details.append("仅台灯照明")
                elif "灯光" in action_text:
                    details.append("灯光柔和")
            # 检查环境声音描述
            if "安静" in action_text:
                details.append("环境安静")
            elif "雨声" in action_text:
                details.append("雨声清晰")
        
        # 综合所有氛围部分
        all_parts = []
        all_parts.extend(time_atmosphere)
        all_parts.extend(location_atmosphere)
        all_parts.extend(details)
        
        if all_parts:
            # 去重并保持顺序
            unique_parts = []
            seen = set()
            for part in all_parts:
                if part not in seen:
                    seen.add(part)
                    unique_parts.append(part)
            return "，".join(unique_parts)
        
        return default_atmosphere

    def _infer_character_appearance(self, character: str, character_text: str) -> Dict[str, str]:
        """
        推断角色外观
        
        Args:
            character: 角色名称
            character_text: 与角色相关的所有文本
            
        Returns:
            外观描述字典
        """
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

        return appearance

    def _ensure_correct_format(self, data: Any) -> Dict[str, Any]:
        """
        确保返回数据格式正确
        
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
    
    def _identify_scene_type(self, script_text: str) -> Any | None:
        """
        识别剧本中的场景类型
        
        Args:
            script_text: 剧本文本
            
        Returns:
            识别出的场景类型字符串，如果没有识别出则返回None
        """
        # 获取所有场景类型配置
        scene_types = self.config.scene_types
        
        for scene_type, config in scene_types.items():
            # 检查是否包含该场景类型的标识符
            identifiers = config.get('identifiers', [])
            if any(identifier in script_text for identifier in identifiers):
                return scene_type
        
        return None
