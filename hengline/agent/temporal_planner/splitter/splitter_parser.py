"""
@FileName: splitter_converter.py
@Description: 数据格式的转换
@Author: HengLine
@Time: 2026/1/15 16:19
"""

from typing import Dict, Optional


class ActionParser:
    """动作解析器"""

    def parse_action(self, action_description: str) -> Dict:
        """解析动作描述，提取关键信息"""
        result = {
            "type": self._classify_action_type(action_description),
            "primary_verb": self._extract_primary_verb(action_description),
            "target": self._extract_target(action_description),
            "intensity": self._estimate_intensity(action_description),
            "duration_hint": self._extract_duration_hint(action_description)
        }
        return result

    def _classify_action_type(self, description: str) -> str:
        """分类动作类型"""
        if any(word in description for word in ["裹着", "蜷在", "坐", "站", "躺"]):
            return "posture"
        elif any(word in description for word in ["盯着", "看向", "注视"]):
            return "gaze"
        elif any(word in description for word in ["按下", "拿起", "放下", "接听"]):
            return "interaction"
        elif any(word in description for word in ["手指收紧", "喉头滚动", "呼吸"]):
            return "physiological"
        elif any(word in description for word in ["瞳孔", "泪水", "微笑", "皱眉"]):
            return "facial"
        elif any(word in description for word in ["震动", "亮起", "显示"]):
            return "device_alert"
        elif any(word in description for word in ["滑落", "掉落", "移动"]):
            return "prop_movement"
        else:
            return "general"

    def _extract_primary_verb(self, description: str) -> str:
        """提取主要动词"""
        verbs = ["裹着", "蜷在", "盯着", "按下", "收紧", "滚动", "停滞", "坐直", "收缩", "滑落"]
        for verb in verbs:
            if verb in description:
                return verb
        return "动作"

    def _extract_target(self, description: str) -> str:
        """提取动作目标"""
        targets = {
            "手机": ["手机", "接听键", "屏幕"],
            "羊毛毯": ["毛毯", "毯子"],
            "沙发": ["沙发"],
            "茶几": ["茶几"],
            "茶水": ["茶", "水杯"],
            "相册": ["相册"]
        }

        for target, keywords in targets.items():
            for keyword in keywords:
                if keyword in description:
                    return target

        return ""

    def _estimate_intensity(self, description: str) -> float:
        """估算动作强度"""
        intensity_words = {
            "轻轻": 0.3,
            "缓慢": 0.4,
            "突然": 0.8,
            "猛烈": 0.9,
            "瞬间": 0.7,
            "紧紧": 0.8,
            "剧烈": 0.9
        }

        max_intensity = 0.5  # 默认中等强度
        for word, intensity in intensity_words.items():
            if word in description:
                max_intensity = max(max_intensity, intensity)

        return max_intensity

    def _extract_duration_hint(self, description: str) -> Optional[float]:
        """从描述中提取时长提示"""
        import re

        # 匹配时间描述
        patterns = [
            r"(\d+)秒",  # "三秒"
            r"(\d+\.?\d*)秒",  # "3秒"或"3.5秒"
            r"片刻",  # "片刻"
            r"一瞬",  # "一瞬"
            r"瞬间",  # "瞬间"
        ]

        for pattern in patterns:
            match = re.search(pattern, description)
            if match:
                if pattern == r"片刻":
                    return 2.0
                elif pattern in [r"一瞬", r"瞬间"]:
                    return 0.5
                else:
                    try:
                        return float(match.group(1))
                    except:
                        pass

        return None
