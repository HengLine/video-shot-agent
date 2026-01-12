"""
@FileName: rule_base_estimator.py
@Description: 时长估算基类
@Author: HengLine
@Time: 2026/1/12 23:07
"""
from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
from abc import abstractmethod
import re

from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation

@dataclass
class EstimationContext:
    """估算上下文信息"""
    scene_type: str = "indoor"  # indoor/outdoor/special
    emotional_tone: str = "neutral"  # neutral/tense/emotional/relaxed
    character_count: int = 1
    location_complexity: float = 1.0  # 1-5 scale
    time_of_day: str = "day"  # day/night/dawn/dusk
    weather: str = "clear"  # clear/rain/snow/fog
    previous_pacing: str = "normal"  # fast/normal/slow
    overall_pacing_target: str = "normal"  # 整体节奏目标


class BaseDurationEstimator(ABC):
    """时长估算基类"""

    def __init__(self, config: Dict[str, Any] = None):
        """初始化估算器"""
        self.config = config or self._get_default_config()
        self.rules = self._initialize_rules()
        self.context = EstimationContext()

    def set_context(self, context: EstimationContext) -> None:
        """设置估算上下文"""
        self.context = context

    def update_context_from_script(self, script_data: UnifiedScript) -> None:
        """从剧本数据更新上下文"""
        if not script_data:
            return

        # 从场景数据提取上下文信息
        scenes = script_data.scenes
        if scenes:
            scene = scenes[0]  # 取第一个场景
            self.context.time_of_day = scene.time_of_day
            self.context.weather = scene.weather
            self.context.emotional_tone = scene.mood

            # 判断场景类型
            location = scene.location
            if "室外" in location or "户外" in location or "街道" in location:
                self.context.scene_type = "outdoor"
            elif "室内" in location or "房间" in location or "客厅" in location:
                self.context.scene_type = "indoor"
            else:
                self.context.scene_type = "special"

        # 角色数量
        characters = script_data.characters
        self.context.character_count = len(characters)

    @abstractmethod
    def _initialize_rules(self) -> Dict[str, Any]:
        """初始化规则库（子类必须实现）"""
        pass

    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置（子类必须实现）"""
        pass

    @abstractmethod
    def estimate(self, element_data: Dict[str, Any]) -> DurationEstimation:
        """估算单个元素（子类必须实现）"""
        pass

    def batch_estimate(self, elements_data: List[Dict[str, Any]]) -> Dict[str, DurationEstimation]:
        """批量估算多个元素"""
        estimations = {}
        for element_data in elements_data:
            estimation = self.estimate(element_data)
            estimations[estimation.element_id] = estimation
        return estimations

    # ============== 通用工具方法 ==============

    def _analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """分析文本复杂度"""
        if not text:
            return {"word_count": 0, "sentence_count": 0, "complexity_score": 0}

        # 分割单词（中文按字，英文按词）
        words = self._split_words(text)
        word_count = len(words)

        # 句子数量（按标点分割）
        sentences = re.split(r'[。！？；.!?;]', text)
        sentence_count = len([s for s in sentences if s.strip()])

        # 计算复杂度得分
        complexity_score = self._calculate_complexity_score(words, sentences)

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "complexity_score": complexity_score
        }

    def _split_words(self, text: str) -> List[str]:
        """分割单词（中英文混合处理）"""
        # 简单的分割逻辑，实际可能需要更复杂的处理
        # 移除标点，分割空白字符
        cleaned = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        words = [w for w in cleaned.split() if w]
        return words

    def _calculate_complexity_score(self, words: List[str], sentences: List[str]) -> float:
        """计算文本复杂度得分"""
        if not words:
            return 0.0

        # 基于词数和句子结构
        word_count = len(words)
        sentence_count = len(sentences)

        if sentence_count == 0:
            return 0.0

        # 平均句长
        avg_sentence_length = word_count / sentence_count

        # 复杂度得分：0-5范围
        score = min(avg_sentence_length / 5, 3.0)  # 句长因素
        score += min(word_count / 20, 2.0)  # 总词数因素

        return round(score, 2)

    def _detect_emotional_intensity(self, text: str, metadata: Dict[str, Any] = None) -> float:
        """检测情感强度"""
        intensity = 1.0  # 默认值

        # 从元数据获取情感信息
        if metadata:
            emotion = metadata.get("emotion", "")
            mood = metadata.get("mood", "")

            if any(word in emotion for word in ["紧张", "激动", "愤怒", "恐惧", "震惊"]):
                intensity = 1.8
            elif any(word in emotion for word in ["悲伤", "忧郁", "孤独", "压抑"]):
                intensity = 1.5
            elif any(word in emotion for word in ["喜悦", "兴奋", "轻松"]):
                intensity = 1.2
            elif any(word in emotion for word in ["平静", "中性"]):
                intensity = 1.0

        # 从文本中检测情感词汇
        emotional_words = {
            "强烈情感": ["爱", "恨", "死", "活", "痛苦", "快乐", "恐惧", "愤怒"],
            "中等情感": ["担心", "希望", "喜欢", "讨厌", "紧张", "放松"],
            "轻微情感": ["可能", "也许", "大概", "似乎", "好像"]
        }

        for level, words in emotional_words.items():
            for word in words:
                if word in text:
                    if level == "强烈情感":
                        intensity = max(intensity, 2.0)
                    elif level == "中等情感":
                        intensity = max(intensity, 1.5)

        return intensity

    def _calculate_confidence(self, element_data: Dict[str, Any],
                              analysis_results: Dict[str, Any]) -> float:
        """计算估算置信度"""
        confidence = 0.7  # 基础置信度

        # 数据完整性
        completeness_score = self._assess_data_completeness(element_data)
        confidence *= completeness_score

        # 分析结果的确定性
        if "complexity_score" in analysis_results:
            complexity = analysis_results["complexity_score"]
            # 中等复杂度的置信度最高，太简单或太复杂都会降低置信度
            if 0.5 <= complexity <= 2.0:
                confidence *= 1.1
            elif complexity > 3.0:
                confidence *= 0.8

        # 确保置信度在合理范围
        return round(max(0.3, min(confidence, 0.95)), 2)

    def _assess_data_completeness(self, element_data: Dict[str, Any]) -> float:
        """评估数据完整性"""
        required_fields = self._get_required_fields()

        if not required_fields:
            return 1.0

        present_count = 0
        for field in required_fields:
            if field in element_data and element_data[field]:
                present_count += 1

        return present_count / len(required_fields)

    def _get_required_fields(self) -> List[str]:
        """获取必需字段（子类可覆盖）"""
        return []  # 基类不要求特定字段

    def _apply_pacing_adjustment(self, base_duration: float) -> float:
        """应用节奏调整"""
        adjustment = 1.0

        pacing_target = self.context.overall_pacing_target
        previous_pacing = self.context.previous_pacing

        # 基于整体节奏目标的调整
        if pacing_target == "fast":
            adjustment *= 0.8  # 加快20%
        elif pacing_target == "slow":
            adjustment *= 1.2  # 减慢20%

        # 基于前序节奏的调整（避免突变）
        if previous_pacing == "fast" and pacing_target == "slow":
            adjustment *= 1.1  # 从快到慢需要更平缓的过渡
        elif previous_pacing == "slow" and pacing_target == "fast":
            adjustment *= 0.9  # 从慢到快可以稍快

        return base_duration * adjustment

    def _apply_context_adjustments(self, base_duration: float,
                                   element_data: Dict[str, Any]) -> float:
        """应用上下文调整因子"""
        adjusted = base_duration

        # 情绪基调调整
        emotional_adjustment = self._get_emotional_adjustment()
        adjusted *= emotional_adjustment

        # 时间调整
        time_adjustment = self._get_time_of_day_adjustment()
        adjusted *= time_adjustment

        # 天气调整
        weather_adjustment = self._get_weather_adjustment()
        adjusted *= weather_adjustment

        return adjusted

    def _get_emotional_adjustment(self) -> float:
        """获取情绪调整因子"""
        adjustments = {
            "tense": 0.9,  # 紧张场景节奏更快
            "emotional": 1.2,  # 情感场景需要更多时间
            "relaxed": 1.1,  # 放松场景节奏稍慢
            "neutral": 1.0
        }
        return adjustments.get(self.context.emotional_tone, 1.0)

    def _get_time_of_day_adjustment(self) -> float:
        """获取时间调整因子"""
        adjustments = {
            "night": 1.15,  # 夜晚场景通常更慢
            "dawn": 1.1,  # 黎明场景稍慢
            "dusk": 1.1,  # 黄昏场景稍慢
            "day": 1.0
        }
        return adjustments.get(self.context.time_of_day, 1.0)

    def _get_weather_adjustment(self) -> float:
        """获取天气调整因子"""
        adjustments = {
            "rain": 1.1,  # 雨景需要更多氛围时间
            "snow": 1.15,  # 雪景节奏更慢
            "fog": 1.2,  # 雾景节奏最慢
            "clear": 1.0
        }
        return adjustments.get(self.context.weather, 1.0)

    def _calculate_duration_range(self, base_duration: float,
                                  confidence: float) -> Tuple[float, float]:
        """计算时长范围（最小值、最大值）"""
        # 置信度越高，范围越小
        range_factor = 1.5 - (confidence * 0.5)  # 0.7-1.3之间

        min_duration = base_duration * (1.0 - (range_factor - 1.0) * 0.3)
        max_duration = base_duration * range_factor

        return round(min_duration, 2), round(max_duration, 2)
