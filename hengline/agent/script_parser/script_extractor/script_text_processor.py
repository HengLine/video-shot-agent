"""
@FileName: script_text_extractor.py
@Description: 文本处理器
@Author: HengLine
@Time: 2025/12/19 0:13
"""
import re
from typing import List


class TextProcessor:
    """文本预处理工具"""

    def clean_text(self, text: str) -> str:
        """
        清理文本，标准化格式
        """
        # 1. 替换特殊字符
        replacements = {
            '\u3000': ' ',  # 全角空格
            '\xa0': ' ',  # 不间断空格
            '　': ' ',  # 中文全角空格
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # 2. 标准化标点
        text = self._normalize_punctuation(text)

        # 3. 移除多余空行和空格
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                # 移除行内多余空格
                line = re.sub(r'\s+', ' ', line)
                cleaned_lines.append(line)

        # 4. 重新组合
        return '\n'.join(cleaned_lines)

    def _normalize_punctuation(self, text: str) -> str:
        """标准化中文标点"""
        # 英文标点转中文标点
        mapping = {
            ',': '，',
            '.': '。',
            '!': '！',
            '?': '？',
            ';': '；',
            ':': '：',
            '(': '（',
            ')': '）',
            '<': '《',
            '>': '》',
        }

        for eng, chn in mapping.items():
            text = text.replace(eng, chn)

        return text

    def preprocess_text(self, text: str) -> str:
        """
        预处理剧本文本
        """
        lines = text.strip().split('\n')
        processed_lines = []

        for line in lines:
            line = line.rstrip()  # 只移除右侧空白

            # 保留标准格式的缩进（非常重要！）
            # 计算前置空格/制表符
            indent_match = re.match(r'^(\s*)', line)
            indent = indent_match.group(1) if indent_match else ""

            # 标准化字符但保留格式
            processed_line = self._standardize_line(line, indent)
            processed_lines.append(processed_line)

        return '\n'.join(processed_lines)

    def _standardize_line(self, line: str, indent: str) -> str:
        """标准化单行文本"""
        line_content = line[len(indent):] if indent else line

        # 标准化引号
        line_content = line_content.replace('"', '"').replace("'", "'")

        # 标准化省略号
        line_content = line_content.replace('...', '…').replace('。。。', '…')

        # 标准化破折号
        line_content = re.sub(r'--+', '—', line_content)

        return indent + line_content

    def split_sentences(self, text: str) -> List[str]:
        """
        中文分句
        """
        # 中文句子结束符
        delimiters = r'[。！？!?]'
        sentences = re.split(delimiters, text)

        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences
