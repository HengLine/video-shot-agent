"""
@FileName: script_time_extractor.py
@Description: 时间标记解析器 - 分镜特有
@Author: HengLine
@Time: 2025/12/19 23:11
"""
import re
from typing import Optional, Dict


class TimeSegmentExtractor:
    """时间标记解析器 - 分镜特有"""

    def parse_time_marker(self, line: str) -> Optional[Dict]:
        """
        解析时间标记

        处理格式：
        - [0-5秒]
        - [0-5s]
        - [0:00-0:05]
        - 【0-5秒】
        - 0-5秒：
        """
        patterns = [
            # 格式：[开始-结束秒]
            (r'^\[(\d+)-(\d+)\](秒|s)?', self._parse_bracket_format),

            # 格式：【开始-结束秒】
            (r'^【(\d+)-(\d+)】(秒|s)?', self._parse_bracket_format),

            # 格式：[时:分:秒-时:分:秒]
            (r'^\[(\d+:\d+(?::\d+)?)-(\d+:\d+(?::\d+)?)\]', self._parse_timestamp_format),

            # 格式：开始-结束秒：
            (r'^(\d+)-(\d+)(秒|s)[：:]', self._parse_colon_format),

            # 格式：开始秒-结束秒
            (r'^(\d+)秒-(\d+)秒', self._parse_seconds_format),
        ]

        for pattern, parser_func in patterns:
            match = re.match(pattern, line)
            if match:
                result = parser_func(match)
                result["original"] = line
                return result

        return None

    def _parse_bracket_format(self, match) -> Dict:
        """解析括号格式的时间"""
        start = int(match.group(1))
        end = int(match.group(2))

        return {
            "start": start,
            "end": end,
            "duration": end - start
        }

    def _parse_timestamp_format(self, match) -> Dict:
        """解析时间戳格式"""
        start_str = match.group(1)
        end_str = match.group(2)

        start_seconds = self._timestamp_to_seconds(start_str)
        end_seconds = self._timestamp_to_seconds(end_str)

        return {
            "start": start_seconds,
            "end": end_seconds,
            "duration": end_seconds - start_seconds
        }

    def _parse_colon_format(self, match) -> Dict:
        """解析冒号格式"""
        start = int(match.group(1))
        end = int(match.group(2))

        return {
            "start": start,
            "end": end,
            "duration": end - start
        }

    def _parse_seconds_format(self, match) -> Dict:
        """解析秒数格式"""
        start = int(match.group(1))
        end = int(match.group(2))

        return {
            "start": start,
            "end": end,
            "duration": end - start
        }

    def _timestamp_to_seconds(self, timestamp: str) -> int:
        """时间戳转秒数"""
        parts = timestamp.split(':')

        if len(parts) == 3:  # 时:分:秒
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:  # 分:秒
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        else:  # 只有秒
            return int(parts[0])
