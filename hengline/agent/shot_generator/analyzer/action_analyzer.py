"""
@FileName: action_analyzer.py
@Description: 
@Author: HengLine
@Time: 2026/1/18 0:03
"""
from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class ActionIntensityAnalyzer:
    """动作强度分析器"""
    ACTION_INTENSITY_EXAMPLES = {
        # 1.0-1.3: 极低-低强度（细微动作）
        "LEVEL_1": {
            "range": (1.0, 1.3),
            "description": "日常、静态、细微动作",
            "examples": [
                ("呼吸", 1.0, "平静的呼吸"),
                ("眨眼", 1.05, "自然的眨眼"),
                ("微笑", 1.1, "淡淡的微笑"),
                ("注视", 1.15, "安静地注视"),
                ("思考", 1.2, "沉思的表情"),
                ("点头", 1.25, "轻微点头"),
                ("伸手", 1.3, "慢慢伸手")
            ],
            "shot_duration_factor": 1.2,  # 这类动作需要更多时间观察细节
            "recommended_shots": ["close_up", "extreme_close_up"]
        },

        # 1.3-1.7: 低-中等强度（日常动作）
        "LEVEL_2": {
            "range": (1.3, 1.7),
            "description": "日常活动、有目的动作",
            "examples": [
                ("行走", 1.35, "正常速度行走"),
                ("坐下", 1.4, "自然地坐下"),
                ("拿起", 1.45, "拿起杯子"),
                ("放下", 1.5, "放下书本"),
                ("转身", 1.55, "慢慢转身"),
                ("喝水", 1.6, "喝一口水"),
                ("写字", 1.65, "在纸上写字")
            ],
            "shot_duration_factor": 1.0,  # 标准时长
            "recommended_shots": ["medium", "two_shot"]
        },

        # 1.7-2.2: 中等-高强度（显著动作）
        "LEVEL_3": {
            "range": (1.7, 2.2),
            "description": "显著、有力、快速动作",
            "examples": [
                ("奔跑", 1.8, "快速奔跑"),
                ("跳跃", 1.9, "跳过障碍"),
                ("推门", 2.0, "用力推开门"),
                ("投掷", 2.1, "扔出球"),
                ("躲避", 2.15, "快速躲避"),
                ("舞蹈", 2.2, "热情的舞蹈")
            ],
            "shot_duration_factor": 0.9,  # 时间稍短，节奏更快
            "recommended_shots": ["action_wide", "medium", "tracking_shot"]
        },

        # 2.2-2.8: 高-极高强度（激烈动作）
        "LEVEL_4": {
            "range": (2.2, 2.8),
            "description": "激烈、冲突、高能量动作",
            "examples": [
                ("打斗", 2.3, "拳击对打"),
                ("摔倒", 2.4, "重重摔倒"),
                ("追逐", 2.5, "激烈追逐"),
                ("挣扎", 2.6, "拼命挣扎"),
                ("爆炸", 2.7, "爆炸闪避"),
                ("营救", 2.8, "危险营救")
            ],
            "shot_duration_factor": 0.8,  # 快速剪辑
            "recommended_shots": ["action_wide", "action_close", "handheld", "fast_cut"]
        },

        # 2.8-3.0: 极端强度（生死攸关）
        "LEVEL_5": {
            "range": (2.8, 3.0),
            "description": "极端、生死攸关、转变性动作",
            "examples": [
                ("死亡", 2.9, "角色死亡"),
                ("蜕变", 3.0, "最终蜕变"),
                ("牺牲", 3.0, "英雄牺牲"),
                ("觉醒", 3.0, "能力觉醒"),
                ("毁灭", 3.0, "毁灭性攻击")
            ],
            "shot_duration_factor": 1.5,  # 需要时间让观众消化
            "recommended_shots": ["extreme_close_up", "slow_motion", "dramatic_wide"]
        }
    }

    def analyze_intensity(self, action_description: str,
                          action_type: str,
                          context: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        多维度分析动作强度
        返回：(总强度, 各维度分数)
        """

        # 1. 物理强度维度 (0.0-1.0)
        physical_score = self._calculate_physical_intensity(action_type, action_description)

        # 2. 情感强度维度 (0.0-1.0)
        emotional_score = self._calculate_emotional_intensity(action_description, context)

        # 3. 叙事重要性维度 (0.0-1.0)
        narrative_score = self._calculate_narrative_importance(context)

        # 4. 速度/节奏维度 (0.0-1.0)
        speed_score = self._calculate_speed_intensity(action_description, action_type)

        # 5. 复杂度维度 (0.0-1.0)
        complexity_score = self._calculate_complexity(action_description)

        # 加权计算总强度
        weights = {
            "physical": 0.35,  # 物理强度最重要
            "emotional": 0.25,  # 情感强度其次
            "narrative": 0.20,  # 叙事重要性
            "speed": 0.15,  # 速度
            "complexity": 0.05  # 复杂度
        }

        total_score = (
                physical_score * weights["physical"] +
                emotional_score * weights["emotional"] +
                narrative_score * weights["narrative"] +
                speed_score * weights["speed"] +
                complexity_score * weights["complexity"]
        )

        # 转换为1.0-3.0范围
        final_intensity = 1.0 + (total_score * 2.0)  # 映射到1.0-3.0

        breakdown = {
            "physical": physical_score,
            "emotional": emotional_score,
            "narrative": narrative_score,
            "speed": speed_score,
            "complexity": complexity_score,
            "total_raw": total_score
        }

        return round(final_intensity, 2), breakdown

    def _calculate_physical_intensity(self, action_type: str, description: str) -> float:
        """计算物理强度"""
        # 动作类型基准强度
        type_intensities = {
            # 极低强度 (0.0-0.2)
            "breathe": 0.05, "blink": 0.05, "glance": 0.1, "smile": 0.15,

            # 低强度 (0.2-0.4)
            "sit": 0.2, "stand": 0.25, "walk": 0.3, "nod": 0.2, "point": 0.25,
            "pick_up": 0.3, "put_down": 0.25, "turn": 0.25,

            # 中等强度 (0.4-0.6)
            "run": 0.5, "jump": 0.55, "climb": 0.6, "push": 0.5, "pull": 0.5,
            "throw": 0.55, "catch": 0.45, "dance": 0.5,

            # 高强度 (0.6-0.8)
            "fight": 0.7, "punch": 0.75, "kick": 0.8, "tackle": 0.8,
            "fall": 0.65, "dodge": 0.7, "chase": 0.7, "escape": 0.75,

            # 极高强度 (0.8-1.0)
            "explosion": 0.95, "crash": 0.9, "death": 1.0, "rescue": 0.9,
            "transform": 0.85, "superpower": 0.95
        }

        base = type_intensities.get(action_type, 0.3)

        # 根据描述词调整
        intensity_boosters = {
            "缓慢": -0.2, "轻轻": -0.15, "小心翼翼": -0.1,
            "快速": +0.2, "猛烈": +0.3, "用力": +0.25,
            "全力": +0.35, "疯狂": +0.4, "拼命": +0.45
        }

        for word, boost in intensity_boosters.items():
            if word in description:
                base = min(1.0, max(0.0, base + boost))

        return base

    def _calculate_emotional_intensity(self, description: str, context: Dict) -> float:
        """计算情感强度"""
        base = 0.3  # 默认中性

        # 情感关键词检测
        emotional_keywords = {
            # 低情感强度 (0.2-0.4)
            "平静": 0.2, "微笑": 0.25, "注视": 0.3, "思考": 0.3,

            # 中等情感强度 (0.4-0.6)
            "激动": 0.5, "兴奋": 0.55, "紧张": 0.6, "担心": 0.5,
            "期待": 0.45, "决心": 0.55,

            # 高情感强度 (0.6-0.8)
            "愤怒": 0.7, "恐惧": 0.75, "悲伤": 0.65, "痛苦": 0.8,
            "绝望": 0.85, "狂喜": 0.7, "憎恨": 0.75,

            # 极高情感强度 (0.8-1.0)
            "崩溃": 0.9, "疯狂": 0.95, "歇斯底里": 0.95, "心碎": 0.9
        }

        # 检测描述中的情感词
        for keyword, score in emotional_keywords.items():
            if keyword in description:
                base = max(base, score)

        # 上下文情感加成
        context_mood = context.get("mood", "neutral")
        mood_multiplier = {
            "平静": 0.8, "中性": 1.0, "紧张": 1.3, "悲伤": 1.2,
            "愤怒": 1.4, "恐惧": 1.5, "喜悦": 1.1, "浪漫": 1.1
        }.get(context_mood, 1.0)

        return min(1.0, base * mood_multiplier)

    def _calculate_narrative_importance(self, context: Dict) -> float:
        """计算叙事重要性"""
        # 基于场景重要性
        scene_importance = context.get("scene_importance", 1.0)  # 1.0-3.0
        normalized = (scene_importance - 1.0) / 2.0  # 映射到0.0-1.0

        # 是否是关键情节转折点
        is_plot_twist = context.get("is_plot_twist", False)
        if is_plot_twist:
            normalized = max(normalized, 0.8)

        # 是否是角色关键时刻
        is_character_moment = context.get("is_character_moment", False)
        if is_character_moment:
            normalized = min(1.0, normalized + 0.3)

        return normalized

    def _calculate_speed_intensity(self, description: str, action_type: str) -> float:
        """计算速度强度"""
        # 动作类型的速度基准
        speed_baselines = {
            "sit": 0.1, "stand": 0.1, "walk": 0.3, "run": 0.7,
            "jump": 0.6, "fall": 0.8, "dodge": 0.9, "throw": 0.5,
            "catch": 0.4, "turn": 0.2, "glance": 0.05
        }

        base = speed_baselines.get(action_type, 0.3)

        # 速度修饰词
        speed_modifiers = {
            "缓慢": -0.3, "慢慢": -0.25, "轻轻": -0.2,
            "迅速": +0.3, "快速": +0.25, "突然": +0.35,
            "瞬间": +0.4, "疾速": +0.45, "闪电般": +0.5
        }

        for word, modifier in speed_modifiers.items():
            if word in description:
                base = min(1.0, max(0.0, base + modifier))

        return base

    def _calculate_complexity(self, description: str) -> float:
        """计算动作复杂度"""
        complexity = 0.2  # 基础复杂度

        # 复合动作检测
        if "同时" in description or "一边...一边" in description:
            complexity += 0.3

        # 精细动作
        if "精确" in description or "精准" in description or "细致" in description:
            complexity += 0.2

        # 协调性要求
        coordination_words = ["协调", "配合", "同步", "平衡"]
        for word in coordination_words:
            if word in description:
                complexity += 0.25

        # 动作步骤数量（简单启发式）
        step_indicators = ["先", "然后", "接着", "最后", "再"]
        step_count = sum(1 for indicator in step_indicators if indicator in description)
        complexity += min(0.3, step_count * 0.1)

        return min(1.0, complexity)

    def calculate_action_duration(self, action_description: str,
                                  action_type: str,
                                  intensity: float) -> float:
        """根据强度计算动作时长"""

        # 基础时长（基于动作类型）
        base_durations = {
            "gesture": 1.2,  # 手势
            "expression": 1.5,  # 表情
            "movement": 2.0,  # 移动
            "interaction": 2.5,  # 交互
            "physical": 3.0,  # 物理动作
        }

        base = base_durations.get(action_type, 2.0)

        # 强度对时长的影响（非线性关系）
        # 中等强度动作时间最短（节奏快）
        # 低强度和极高强度动作时间都较长
        if intensity < 1.5:
            # 低强度：需要时间观察细节
            duration = base * 1.3
        elif intensity < 2.0:
            # 中等强度：标准或稍快
            duration = base * 1.0
        elif intensity < 2.5:
            # 高强度：快速动作
            duration = base * 0.8
        else:
            # 极高强度：可能需要慢动作或延长
            if "死亡" in action_description or "牺牲" in action_description:
                duration = base * 2.0  # 重要时刻需要时间
            else:
                duration = base * 0.7  # 快速剪辑

        # 情感加成
        emotional_words = ["哭泣", "拥抱", "亲吻", "告白", "告别"]
        if any(word in action_description for word in emotional_words):
            duration *= 1.4

        return round(max(0.8, duration), 2)  # 最少0.8秒