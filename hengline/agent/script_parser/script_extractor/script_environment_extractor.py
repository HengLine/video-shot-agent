"""
@FileName: script_environment_extractor.py
@Description: 环境提取器
@Author: HengLine
@Time: 2025/12/19 0:18
"""
from typing import List, Dict


class EnvironmentExtractor:
    """环境信息提取器"""

    def extract(self, text: str) -> Dict:
        """
        提取环境信息
        """
        environment = {
            "weather": self._extract_weather(text),
            "lighting": self._extract_lighting(text),
            "soundscape": self._extract_sound(text),
            "key_objects": self._extract_objects(text),
            "atmosphere_adjectives": self._extract_atmosphere(text)
        }
        return environment

    def _extract_weather(self, text: str) -> str:
        """提取天气信息"""
        weather_keywords = {
            "晴天": ["晴天", "阳光", "太阳", "晴朗", "明媚"],
            "阴天": ["阴天", "阴沉", "多云", "乌云"],
            "雨天": ["雨", "下雨", "暴雨", "小雨", "雷雨", "雨滴"],
            "雪天": ["雪", "下雪", "雪花", "大雪", "冰雪"],
            "雾天": ["雾", "雾气", "雾霾", "朦胧"],
            "夜晚": ["夜晚", "黑夜", "星空", "月光"],
        }

        for weather, keywords in weather_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return weather

        return "一般天气"

    def _extract_lighting(self, text: str) -> str:
        """提取灯光信息"""
        lighting_keywords = {
            "明亮": ["明亮", "光亮", "亮堂", "耀眼", "灿烂"],
            "昏暗": ["昏暗", "阴暗", "黑暗", "黯淡", "微弱"],
            "温暖": ["温暖", "柔和", "温馨", "暖色", "橘黄"],
            "冷清": ["冷清", "冷色", "苍白", "惨白", "冷光"],
            "闪烁": ["闪烁", "闪动", "摇曳", "忽明忽暗"],
        }

        for lighting, keywords in lighting_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return lighting

        return "一般光线"

    def _extract_sound(self, text: str) -> List[str]:
        """提取声音信息"""
        sounds = []

        sound_keywords = [
            "安静", "寂静", "喧闹", "嘈杂", "音乐", "歌声",
            "雨声", "风声", "雷声", "笑声", "哭声", "脚步声",
            "敲门声", "电话声", "钟声", "警报声", "呼吸声"
        ]

        for keyword in sound_keywords:
            if keyword in text:
                sounds.append(keyword)

        return sounds

    def _extract_objects(self, text: str) -> List[str]:
        """提取关键物体"""
        objects = []

        # 常见场景物体
        common_objects = [
            "桌子", "椅子", "沙发", "床", "窗户", "门", "窗帘",
            "灯", "电视", "手机", "电脑", "书本", "杯子", "花瓶",
            "画", "照片", "钟表", "钥匙", "钱包", "包包", "衣服"
        ]

        for obj in common_objects:
            if obj in text:
                objects.append(obj)

        return objects

    def _extract_atmosphere(self, text: str) -> List[str]:
        """提取氛围形容词"""
        atmosphere = []

        atmosphere_keywords = [
            "温馨", "浪漫", "紧张", "恐怖", "神秘", "悲伤",
            "欢乐", "尴尬", "压抑", "轻松", "严肃", "活泼",
            "孤独", "热闹", "宁静", "混乱", "有序", "杂乱"
        ]

        for keyword in atmosphere_keywords:
            if keyword in text:
                atmosphere.append(keyword)

        return atmosphere

environment_extractor = EnvironmentExtractor()
