"""
@FileName: script_field_extractor.py
@Description: 
@Author: HengLine
@Time: 2025/12/19 23:15
"""
import re
from typing import Dict, Optional


class StoryboardFieldExtractor:
    """分镜字段提取器 - 分镜特有"""

    def __init__(self):
        # 字段名标准化
        self.field_mapping = {
            # 视觉相关
            "画面": "visual",
            "镜头": "visual",
            "场景": "visual",
            "图像": "visual",
            "视觉": "visual",
            "VIDEO": "visual",
            "VISUAL": "visual",

            # 音频相关
            "声音": "audio",
            "音频": "audio",
            "音效": "audio",
            "音乐": "audio",
            "对白": "audio",
            "AUDIO": "audio",
            "SOUND": "audio",

            # 动作相关
            "动作": "action",
            "行为": "action",
            "表演": "action",
            "ACTION": "action",

            # 摄像机相关
            "镜头运动": "camera",
            "摄像机": "camera",
            "拍摄": "camera",
            "CAMERA": "camera",

            # 灯光相关
            "灯光": "lighting",
            "照明": "lighting",
            "光影": "lighting",
            "LIGHTING": "lighting",

            # 其他
            "旁白": "narration",
            "字幕": "subtitle",
            "转场": "transition",
            "特效": "effect",
        }

    def extract_field(self, line: str) -> Optional[Dict]:
        """
        提取分镜字段

        格式：字段名：字段值
        """
        # 支持中文和英文冒号
        patterns = [
            r'^([^：:]{1,10})[：:]\s*(.+)$',  # 标准格式
            r'^([^：:]{1,10})[：:]\s*$',  # 只有字段名
        ]

        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                field_name = match.group(1).strip()
                field_value = match.group(2).strip() if len(match.groups()) > 1 else ""

                # 标准化字段名
                std_field_name = self.field_mapping.get(field_name, field_name.lower())

                return {
                    "original_name": field_name,
                    "field_name": std_field_name,
                    "field_value": field_value
                }

        return None

    @staticmethod
    def merge_fields_to_description(fields: Dict) -> str:
        """
        将多个字段合并为场景描述
        """
        description_parts = []

        # 视觉描述优先
        if "visual" in fields:
            description_parts.append(f"画面：{fields['visual']}")

        # 动作描述
        if "action" in fields:
            description_parts.append(f"动作：{fields['action']}")

        # 音频描述
        if "audio" in fields:
            description_parts.append(f"声音：{fields['audio']}")

        # 其他字段
        other_fields = ["camera", "lighting", "narration", "transition"]
        for field in other_fields:
            if field in fields:
                field_name_cn = StoryboardFieldExtractor._get_chinese_field_name(field)
                description_parts.append(f"{field_name_cn}：{fields[field]}")

        return "；".join(description_parts)

    @staticmethod
    def _get_chinese_field_name(field_name: str) -> str:
        """获取字段的中文名"""
        reverse_mapping = {
            "visual": "画面",
            "audio": "声音",
            "action": "动作",
            "camera": "镜头",
            "lighting": "灯光",
            "narration": "旁白",
            "transition": "转场"
        }
        return reverse_mapping.get(field_name, field_name)