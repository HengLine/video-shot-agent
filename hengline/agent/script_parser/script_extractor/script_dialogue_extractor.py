"""
@FileName: script_dialogue_extractor.py
@Description: 对话提取器
@Author: HengLine
@Time: 2025/12/19 0:16
"""
import re
from typing import List, Optional, Dict

from .script_emotion_extractor import emotion_extractor
from hengline.agent.script_parser.script_parser_model import Dialogue


class DialogueExtractor:
    """对话提取器"""

    def __init__(self):
        # 对话模式（从中文剧本中总结）
        self.dialogue_patterns = [
            # 模式1：说话者："对话"
            (r'([^"「」:：]{1,6})[：:]["「]([^"「」]+)["」]', 1, 2),

            # 模式2："对话"（说话者）
            (r'["「]([^"「」]+)["」]\s*（([^）]+)）', 2, 1),

            # 模式3：说话者说："对话"
            (r'([^"「」]{1,6})说[：:]?["「]([^"「」]+)["」]', 1, 2),

            # 模式4：说话者道："对话"
            (r'([^"「」]{1,6})道[：:]?["「]([^"「」]+)["」]', 1, 2),

            # 模式5：说话者喊道："对话"
            (r'([^"「」]{1,6})喊道[：:]?["「]([^"「」]+)["」]', 1, 2),

            # 模式6：说话者轻声说："对话"
            (r'([^"「」]{1,6})轻声说[：:]?["「]([^"「」]+)["」]', 1, 2),

            # 模式7：无说话者的直接对话
            (r'["「]([^"「」]+)["」]', None, 1),
        ]

    def extract(self, text: str, scene_id: str) -> List[Dialogue]:
        """
        提取对话
        """
        dialogues = []
        dialogue_counter = 1

        # 按句子分割
        sentences = re.split(r'[。！？]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 3:
                continue

            # 尝试匹配各种对话模式
            for pattern, speaker_group, dialogue_group in self.dialogue_patterns:
                match = re.search(pattern, sentence)
                if match:
                    if speaker_group is None:
                        speaker = "未知"
                        dialogue_text = match.group(dialogue_group)
                    else:
                        speaker = match.group(speaker_group).strip()
                        dialogue_text = match.group(dialogue_group).strip()

                    # 清理说话者名称
                    speaker = self._clean_speaker_name(speaker)

                    # 创建对话对象
                    dialogue_id = f"dialogue_{dialogue_counter:03d}"

                    dialogue = Dialogue(
                        dialogue_id=dialogue_id,
                        speaker=speaker,
                        text=dialogue_text,
                        emotion=emotion_extractor.detect_emotion(dialogue_text),
                        parenthetical="",
                        scene_ref=scene_id,
                    )

                    dialogues.append(dialogue)
                    dialogue_counter += 1
                    break  # 匹配成功后跳出

        return dialogues

    def _clean_speaker_name(self, speaker: str) -> str:
        """
        清理说话者名称
        """
        # 移除常见后缀
        suffixes = ['说', '道', '喊道', '轻声说', '低声说', '大声说',
                    '问道', '回答', '解释', '补充', '继续']

        for suffix in suffixes:
            if speaker.endswith(suffix):
                speaker = speaker[:-len(suffix)].strip()

        # 移除标点和空格
        speaker = re.sub(r'[：:]\s*$', '', speaker)
        speaker = speaker.strip()

        return speaker

dialogue_extractor = DialogueExtractor()


class ScreenplayDialogueParser:
    """对话解析器 - 标准格式特有"""

    def __init__(self):
        self.dialogue_counter = 0

    def is_dialogue_line(self, line: str) -> bool:
        """
        判断是否是对话行

        标准格式：在角色名之后，通常有缩进
        """
        line_stripped = line.strip()

        # 空行不是对话
        if not line_stripped:
            return False

        # 检查是否以场景标题、转场、角色名开头
        if (line_stripped.startswith(('INT.', 'EXT.', 'CUT TO', 'FADE')) or
                line_stripped.isupper() or
                re.match(r'^[\u4e00-\u9fa5]{2,4}$', line_stripped)):
            return False

        # 对话通常有缩进（但不是绝对）
        # 检查是否有常见的对话特征
        has_dialogue_features = (
                line_stripped.startswith('"') or  # 以引号开头
                line_stripped.endswith('"') or  # 以引号结尾
                '"' in line_stripped or  # 包含引号
                '（' in line_stripped or  # 中文括号
                '(' in line_stripped  # 英文括号（可能是表演提示）
        )

        # 或者行长度适中（10-200字符），不是动作描述
        reasonable_length = 10 <= len(line_stripped) <= 200

        return has_dialogue_features or reasonable_length

    def parse_dialogue(self, line: str, speaker: str, scene_ref: Optional[str]) -> Optional[Dict]:
        """解析对话内容"""
        line_stripped = line.strip()

        # 清理引号
        dialogue_text = self.clean_dialogue_text(line_stripped)

        if not dialogue_text:
            return None

        # 生成对话ID
        self.dialogue_counter += 1
        dialogue_id = f"dialogue_{self.dialogue_counter:03d}"

        # 推断情绪
        emotion = self.infer_emotion(dialogue_text)

        return {
            "dialogue_id": dialogue_id,
            "speaker": speaker,
            "text": dialogue_text,
            "emotion": emotion,
            "scene_ref": scene_ref if scene_ref else "",
            "original_line": line_stripped
        }

    def clean_dialogue_text(self, text: str) -> str:
        """清理对话文本"""
        # 移除开头和结尾的引号
        text = text.strip('"\'')

        # 移除括号内的表演提示（如果有）
        text = re.sub(r'\s*\([^)]+\)\s*$', '', text)  # 行尾的括号
        text = re.sub(r'^\s*\([^)]+\)\s*', '', text)  # 行首的括号

        # 标准化中文标点
        text = text.replace('...', '…')

        return text.strip()

    def infer_emotion(self, text: str) -> str:
        """推断对话情绪"""
        text_lower = text.lower()

        emotion_keywords = {
            "愤怒": ["damn", "hell", "idiot", "stupid", "愤怒", "生气", "混蛋"],
            "悲伤": ["sorry", "sad", "cry", "tear", "悲伤", "难过", "对不起"],
            "高兴": ["happy", "great", "wonderful", "love", "高兴", "开心", "喜欢"],
            "惊讶": ["what", "oh", "wow", "really", "惊讶", "吃惊", "真的吗"],
            "恐惧": ["afraid", "scared", "fear", "help", "害怕", "恐惧", "救命"],
            "疑问": ["?", "why", "how", "what if", "疑问", "为什么", "如何"],
        }

        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return emotion

        # 检查标点符号
        if text.endswith('!') or text.endswith('！'):
            return "激动"
        elif text.endswith('?') or text.endswith('？'):
            return "疑问"

        return "平静"