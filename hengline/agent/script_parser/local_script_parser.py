"""
@FileName: local_script_parser.py
@Description: 
@Author: HengLine
@Time: 2026/1/10 22:33
"""
import re
from typing import Dict, Any

from hengline.agent.script_parser.base_script_parser import ScriptParser
from hengline.agent.script_parser.script_parser_models import UnifiedScript, Character, Prop, Scene, Dialogue
from hengline.agent.workflow.workflow_models import ScriptType


class LocalScriptParser(ScriptParser):

    def __init__(self):
        """
        初始化剧本解析智能体
        """
        # 本地规则：用于校验和补全AI解析结果
        self.local_rules = {
            "character_name_patterns": [
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # 英文名
                r"([\u4e00-\u9fa5]{2,4})",  # 中文名（2-4字）
                r"(角色\s*[：:]\s*([^\s，。]+))",
                r"([^\s，。]+)\s*(?:说|道|问|喊|叫|称)"
            ],
            "scene_location_patterns": [
                r"(?:在|位于|处于)\s*([^，。]+?)(?:的|里|内|上|中)",
                r"(?:场景|地点)\s*[：:]\s*([^，。]+)",
                r"(?:INT\.|EXT\.)\s*([^-\n]+)",  # 室内/室外
                r"(?:室内|室外|房间|客厅|卧室|办公室|街道|公园)(?:[^，。]*?)"
            ],
            "dialogue_patterns": [
                r"([^\s，。：:]+)\s*[：:]\s*[\"']?([^\"'\n]+?)[\"']?[。！？]",
                r"([^\s，。]+)\s*(?:说|道|问|喊|叫|称)[：:]\s*[\"']?([^\"'\n]+?)[\"']?",
                r"[\"']([^\"'\n]+?)[\"']\s*[，,]?\s*([^\s，。]+)\s*(?:说|道)"
            ],
            "action_patterns": [
                r"([^\s，。]+)\s*(?:走|跑|坐|站|拿|看|笑|哭|转身|点头|摇头)(?:[^，。]*?)",
                r"(?:然后|接着|随后)\s*([^\s，。]+)\s*(?:开始|继续|停止)(?:[^，。]*?)",
                r"([^\s，。]+)\s*(?:手持|拿着|带着|使用)(?:[^，。]*?)"
            ]
        }

    def process(self, script_text: Any, unified_script: UnifiedScript) -> UnifiedScript:
        """
        应用本地规则进行校验和补全

        这是AI解析后的质量保证层
        """
        # 1. 校验角色名称一致性
        unified_script = self._validate_character_consistency(unified_script, script_text)

        # 2. 补全缺失的场景信息
        unified_script = self._complete_scene_info(unified_script, script_text)

        # 3. 提取AI可能遗漏的对话
        unified_script = self._extract_missing_dialogues(unified_script, script_text)

        # 4. 识别和补全道具
        unified_script = self._identify_props(unified_script, script_text)

        # 5. 连接相关元素（如对话和动作）
        unified_script = self._connect_related_elements(unified_script)

        return unified_script

    def _validate_character_consistency(self, script: UnifiedScript,
                                        original_text: str) -> UnifiedScript:
        """校验角色名称一致性"""

        # 从原始文本中提取所有可能的角色名
        extracted_names = set()
        for pattern in self.local_rules["character_name_patterns"]:
            matches = re.findall(pattern, original_text)
            for match in matches:
                if isinstance(match, tuple):
                    name = match[0] if match[0] else (match[1] if len(match) > 1 else "")
                else:
                    name = match
                if name and len(name) >= 2:  # 过滤太短的名字
                    extracted_names.add(name.strip())

        # 检查AI提取的角色名是否在文本中出现
        ai_character_names = {char.name for char in script.characters if char.name}

        # 找出AI可能遗漏的角色
        missing_in_ai = extracted_names - ai_character_names
        missing_in_text = ai_character_names - extracted_names

        # 如果有差异，添加到警告
        if missing_in_ai:
            script.warnings.append(f"AI可能遗漏了角色: {', '.join(missing_in_ai)}")

            # 为遗漏的角色创建基本Character对象
            for name in missing_in_ai:
                if len(name) <= 20:  # 避免过长的误匹配
                    script.characters.append(Character(
                        element_id=f"char_extracted_{len(script.characters)}",
                        element_type="character",
                        content=name,
                        name=name,
                        confidence=0.5  # 较低置信度
                    ))

        if missing_in_text:
            script.warnings.append(f"AI提取了文本中未明确出现的角色名: {', '.join(missing_in_text)}")

        return script

    def _complete_scene_info(self, script: UnifiedScript,
                             original_text: str) -> UnifiedScript:
        """补全缺失的场景信息"""

        # 如果AI没有提取到场景，尝试从文本中提取
        if not script.scenes:
            # 使用正则表达式提取可能的场景
            location_matches = []
            for pattern in self.local_rules["scene_location_patterns"]:
                matches = re.findall(pattern, original_text)
                location_matches.extend(matches)

            # 去重
            unique_locations = list(set([loc[0] if isinstance(loc, tuple) else loc
                                         for loc in location_matches]))

            # 为每个位置创建基本场景
            for i, location in enumerate(unique_locations[:3]):  # 最多3个场景
                script.scenes.append(Scene(
                    element_id=f"scene_extracted_{i + 1}",
                    element_type="scene",
                    content=f"发生在{location}的场景",
                    location=location,
                    confidence=0.6
                ))

        return script

    def _extract_missing_dialogues(self, script: UnifiedScript,
                                   original_text: str) -> UnifiedScript:
        """提取AI可能遗漏的对话"""

        # 统计AI提取的对话数量
        ai_dialogue_count = len(script.dialogues)

        # 使用正则表达式提取对话
        extracted_dialogues = []
        for pattern in self.local_rules["dialogue_patterns"]:
            matches = re.findall(pattern, original_text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    speaker, content = match[0], match[1]
                else:
                    continue

                if speaker and content and len(content) <= 100:  # 过滤过长的内容
                    extracted_dialogues.append({
                        "speaker": speaker.strip(),
                        "content": content.strip()
                    })

        # 如果AI提取的对话明显少于正则提取的，补充一些
        if extracted_dialogues and ai_dialogue_count < len(extracted_dialogues) * 0.5:
            script.warnings.append("AI可能遗漏了部分对话")

            # 添加提取的对话（避免重复）
            existing_dialogues = {(d.speaker, d.content) for d in script.dialogues}

            for i, dialogue in enumerate(extracted_dialogues[:10]):  # 最多补充10个
                key = (dialogue["speaker"], dialogue["content"])
                if key not in existing_dialogues:
                    script.dialogues.append(Dialogue(
                        element_id=f"dialogue_extracted_{len(script.dialogues)}",
                        element_type="dialogue",
                        content=dialogue["content"],
                        speaker=dialogue["speaker"],
                        confidence=0.7
                    ))

        return script

    def _identify_props(self, script: UnifiedScript,
                        original_text: str) -> UnifiedScript:
        """识别和补全道具"""

        # 常见的道具关键词
        prop_keywords = [
            "杯子", "咖啡杯", "茶杯", "手机", "书本", "钥匙", "包", "钱包",
            "眼镜", "手表", "戒指", "项链", "帽子", "外套", "雨伞", "文件夹",
            "电脑", "平板", "笔", "笔记本", "照片", "画", "花", "礼物"
        ]

        # 从文本中查找道具
        found_props = []
        for keyword in prop_keywords:
            if keyword in original_text:
                # 查找上下文
                context_pattern = f".{{0,30}}{keyword}.{{0,30}}"
                contexts = re.findall(context_pattern, original_text)

                for context in contexts[:2]:  # 取前两个上下文
                    # 尝试推断持有者
                    owner = None
                    for char in script.characters:
                        if char.name and char.name in context:
                            owner = char.name
                            break

                    found_props.append({
                        "name": keyword,
                        "context": context,
                        "owner": owner
                    })

        # 添加找到的道具到脚本中
        for prop_info in found_props:
            # 检查是否已存在
            existing_prop_names = {prop.content for prop in script.props}
            if prop_info["name"] not in existing_prop_names:
                script.props.append(Prop(
                    element_id=f"prop_extracted_{len(script.props)}",
                    element_type="prop",
                    content=prop_info["name"],
                    prop_type="日常物品",
                    location="场景中",
                    state="使用中",
                    owner=prop_info["owner"],
                    metadata={"context": prop_info["context"]},
                    confidence=0.8
                ))

        return script

    def _connect_related_elements(self, script: UnifiedScript) -> UnifiedScript:
        """连接相关元素（如对话和动作的时序关系）"""

        # 这里可以实现更复杂的元素关联逻辑
        # 例如：将对话和动作按时间顺序排列

        return script

    def _fallback_parse(self, text: str, format_type: ScriptType) -> Dict[str, Any]:
        """备用解析方法（当AI解析失败时使用）"""
        print("🔄 使用备用解析方法...")

        # 创建一个最基本的解析结构
        fallback_data = {
            "scenes": [{
                "element_id": "scene_fallback",
                "element_type": "scene",
                "content": "主要场景",
                "location": "未知地点",
                "confidence": 0.3
            }],
            "characters": [],
            "dialogues": [],
            "actions": [],
            "props": []
        }

        # 尝试提取一些基本信息
        lines = text.split('\n')

        # 提取可能的人名
        name_pattern = r'([\u4e00-\u9fa5]{2,3}|[A-Z][a-z]+\s+[A-Z][a-z]+)'
        potential_names = re.findall(name_pattern, text)

        for i, name in enumerate(set(potential_names[:5])):  # 最多5个名字
            fallback_data["characters"].append({
                "element_id": f"char_fallback_{i}",
                "element_type": "character",
                "content": name,
                "name": name,
                "confidence": 0.4
            })

        # 提取可能的对话
        for i, line in enumerate(lines):
            if ':' in line or '：' in line or '说' in line:
                parts = re.split(r'[:：]', line, 1)
                if len(parts) == 2:
                    speaker, content = parts[0].strip(), parts[1].strip()
                    if speaker and content and len(speaker) < 20:
                        fallback_data["dialogues"].append({
                            "element_id": f"dialogue_fallback_{i}",
                            "element_type": "dialogue",
                            "content": content,
                            "speaker": speaker,
                            "confidence": 0.5
                        })

        return fallback_data
