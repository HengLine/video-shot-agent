"""
@FileName: rule_base_estimator.py
@Description: 时长估算基类
@Author: HengLine
@Time: 2026/1/12 23:07
"""
import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

from hengline.agent.script_parser.script_parser_models import UnifiedScript, Scene
from hengline.agent.temporal_planner.estimator.base_estimator import BaseDurationEstimator
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, ElementType
from hengline.config.keyword_config import get_keyword_config
from hengline.config.temporal_planner_config import get_planner_config


@dataclass
class ElementWithContext:
    element_id: str
    element_type: ElementType
    data: Any
    time_offset: float = 0.0


@dataclass
class EstimationContext:
    """估算上下文信息"""
    scene_type: str = "indoor"  # indoor/outdoor/special
    emotional_tone: str = "neutral"  # neutral/tense/emotional/relaxed
    character_count: int = 1
    location_complexity: float = 1.0  # 1-5 scale
    time_of_day: str = "day"  # day/night/dawn/dusk
    weather: str = "clear"  # clear/rain/snow/fog
    # 节奏信息
    previous_pacing: str = "normal"  # fast/normal/slow
    overall_pacing: str = "normal"

    # 语义特征
    semantic_density: float = 1.0
    visual_complexity: float = 1.0

    # 序列信息
    position_in_sequence: int = 0
    total_elements: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "scene_type": self.scene_type,
            "emotional_tone": self.emotional_tone,
            "character_count": self.character_count,
            "location_complexity": self.location_complexity,
            "previous_pacing": self.previous_pacing,
            "overall_pacing": self.overall_pacing,
            "semantic_density": self.semantic_density,
            "visual_complexity": self.visual_complexity,
            "position_in_sequence": self.position_in_sequence,
            "total_elements": self.total_elements
        }


class BaseRuleDurationEstimator(BaseDurationEstimator):
    """时长估算基类"""

    def __init__(self):
        """初始化估算器"""
        super().__init__()
        self.context = EstimationContext()
        self.keyword_config = get_keyword_config()
        self.planner_config = get_planner_config()
        self.rules = self._initialize_rules()

    def set_context(self, context: EstimationContext) -> None:
        """设置估算上下文"""
        self.context = context

    def update_context_from_script(self, script_data: UnifiedScript) -> None:
        """从剧本数据更新上下文"""
        if not script_data:
            return

        scenes = script_data.scenes
        if scenes:
            self._update_context_from_scene(scenes[0])

        self.context.character_count = len(script_data.characters)

    def _update_context_from_scene(self, scene: Scene) -> None:
        """从场景数据更新上下文"""
        # 时间
        self.context.time_of_day = self._normalize_time_of_day(scene.time_of_day)

        # 天气
        self.context.weather = self._normalize_weather(scene.weather)

        # 情绪基调
        self.context.emotional_tone = self._normalize_emotional_tone(scene.mood)

        # 场景类型
        self.context.scene_type = self._determine_scene_type_from_location(scene.location)

    def _normalize_time_of_day(self, time_str: str) -> str:
        """规范化时间描述"""
        time_map = {
            "夜晚": "night", "晚上": "night", "深夜": "night", "night": "night",
            "黎明": "dawn", "拂晓": "dawn", "清晨": "dawn", "dawn": "dawn",
            "黄昏": "dusk", "傍晚": "dusk", "dusk": "dusk",
            "白天": "day", "上午": "day", "下午": "day", "中午": "day", "day": "day"
        }
        return time_map.get(time_str, "day")

    def _normalize_weather(self, weather_str: str) -> str:
        """规范化天气描述"""
        weather_map = {
            "大雨": "rain", "小雨": "rain", "雨": "rain", "下雨": "rain", "rain": "rain",
            "雪": "snow", "下雪": "snow", "大雪": "snow", "snow": "snow",
            "雾": "fog", "大雾": "fog", "雾天": "fog", "fog": "fog",
            "晴朗": "clear", "晴天": "clear", "阳光": "clear", "clear": "clear",
            "多云": "clear", "阴天": "clear", "cloudy": "clear"
        }
        return weather_map.get(weather_str, "clear")

    def _normalize_emotional_tone(self, mood_str: str) -> str:
        """规范化情绪基调"""
        # 简单映射，实际可以更复杂
        if any(word in mood_str for word in ["紧张", "激动", "愤怒", "恐惧"]):
            return "tense"
        elif any(word in mood_str for word in ["悲伤", "忧郁", "孤独", "压抑"]):
            return "emotional"
        elif any(word in mood_str for word in ["喜悦", "兴奋", "轻松"]):
            return "relaxed"
        return "neutral"

    def _determine_scene_type_from_location(self, location: str) -> str:
        """根据地点确定场景类型"""
        if any(word in location for word in ["室外", "户外", "街道", "公园", "山", "森林"]):
            return "outdoor"
        elif any(word in location for word in ["室内", "房间", "客厅", "卧室", "办公室", "咖啡厅"]):
            return "indoor"
        return "special"

    @abstractmethod
    def _initialize_rules(self) -> Dict[str, Any]:
        """初始化规则库（子类必须实现）"""
        pass

    @abstractmethod
    def estimate(self, element_data: Any, context: Dict = None) -> DurationEstimation:
        """估算单个元素（子类必须实现）"""
        pass

    def batch_estimate(self, elements_data: List[Any], context: Dict = None) -> Dict[str, DurationEstimation]:
        """批量估算多个元素"""
        estimations = {}
        for element_data in elements_data:
            estimation = self.estimate(element_data, context)
            estimations[estimation.element_id] = estimation
        return estimations

    # ============== 通用工具方法（从配置读取） ==============
    def _analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """分析文本复杂度"""
        if not text:
            return {"word_count": 0, "sentence_count": 0, "complexity_score": 0}

        # 获取分割模式
        text_config = self.planner_config.get_base_config("text_analysis", {})
        word_split_pattern = text_config.get("word_split_pattern", "[^\w\s\u4e00-\u9fff]")
        sentence_split_pattern = text_config.get("sentence_split_pattern", "[。！？；.!?;]")

        # 分割单词
        cleaned = re.sub(word_split_pattern, ' ', text)
        words = [w for w in cleaned.split() if w]
        word_count = len(words)

        # 句子数量
        sentences = re.split(sentence_split_pattern, text)
        sentence_count = len([s for s in sentences if s.strip()])

        # 计算复杂度得分
        complexity_score = self._calculate_complexity_score(word_count, sentence_count, 0, "")

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "complexity_score": complexity_score
        }

    def _calculate_complexity_score(self, word_count: int, component_count: int,
                                    fine_motor_count: int, emotion_intensity: str) -> float:
        """计算复杂度得分"""
        score = 0.0

        # 词数贡献
        score += min(word_count / 10, 2.0)

        # 组成部分数量贡献
        score += min(component_count / 3, 1.5)

        # 精细动作贡献
        if fine_motor_count and fine_motor_count > 0:
            score += fine_motor_count * 0.3

        # 情感强度贡献
        if emotion_intensity and emotion_intensity != "":
            intensity_scores = {"mild": 0, "moderate": 0.5, "strong": 1.0, "dramatic": 1.5}
            score += intensity_scores.get(emotion_intensity, 0)

        return round(score, 2)

    def _detect_emotional_intensity(self, text: str, metadata: Dict[str, Any] = None) -> float:
        """检测情感强度"""
        intensity = 1.0  # 默认值

        # 从元数据获取情感信息
        if metadata:
            emotion = metadata.get("emotion", "")
            mood = metadata.get("mood", "")

            # 从配置获取情感词汇
            emotional_words_config = self.keyword_config.get_emotion_keywords()

            # 检查强烈情感
            strong_words = emotional_words_config.get("intense_emotion", [])
            if any(word in emotion for word in strong_words):
                intensity = max(intensity, 2.0)

            # 检查中等情感
            medium_words = emotional_words_config.get("moderate_emotion", [])
            if any(word in emotion for word in medium_words):
                intensity = max(intensity, 1.5)

        return intensity

    def _calculate_confidence(self, element_data: Any,
                              analysis_results: Dict[str, Any]) -> float:
        """计算估算置信度"""
        # 获取基础置信度配置
        min_confidence = self.planner_config.get_base_config("min_confidence", 0.3)
        max_confidence = self.planner_config.get_base_config("max_confidence", 0.95)
        default_confidence = self.planner_config.get_base_config("default_confidence", 0.7)

        confidence = default_confidence

        # 数据完整性
        completeness_score = self._assess_data_completeness(element_data)
        confidence *= completeness_score

        # 分析结果的确定性
        if "complexity_score" in analysis_results:
            complexity = analysis_results["complexity_score"]
            # 中等复杂度的置信度最高
            if 0.5 <= complexity <= 2.0:
                confidence *= 1.1
            elif complexity > 3.0:
                confidence *= 0.8

        # 确保置信度在合理范围
        return round(max(min_confidence, min(confidence, max_confidence)), 2)

    def _assess_data_completeness(self, element_data: Any) -> float:
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
        return []

    def _apply_pacing_adjustment(self, base_duration: float) -> float:
        """应用节奏调整"""
        adjustment = 1.0

        pacing_target = self.context.overall_pacing
        previous_pacing = self.context.previous_pacing

        # 基于整体节奏目标的调整
        if pacing_target == "fast":
            adjustment *= 0.8  # 加快20%
        elif pacing_target == "slow":
            adjustment *= 1.2  # 减慢20%

        # 基于前序节奏的调整
        if previous_pacing == "fast" and pacing_target == "slow":
            adjustment *= 1.1
        elif previous_pacing == "slow" and pacing_target == "fast":
            adjustment *= 0.9

        return base_duration * adjustment

    def _apply_context_adjustments(self, base_duration: float) -> float:
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
        # 这个应该由子类实现，因为不同元素类型有不同的调整规则
        return 1.0

    def _get_time_of_day_adjustment(self) -> float:
        """获取时间调整因子"""
        # 这个应该由子类实现
        return 1.0

    def _get_weather_adjustment(self) -> float:
        """获取天气调整因子"""
        # 这个应该由子类实现
        return 1.0

    def _calculate_duration_range(self, base_duration: float,
                                  confidence: float) -> Tuple[float, float]:
        """计算时长范围（最小值、最大值）"""
        # 置信度越高，范围越小
        range_factor = 1.5 - (confidence * 0.5)  # 0.7-1.3之间

        min_duration = round(base_duration * (1.0 - (range_factor - 1.0) * 0.3), 2)
        max_duration = round(base_duration * range_factor, 2)

        return round(min_duration, 2), round(max_duration, 2)

    def _get_keywords(self, category: str, subcategory: str = None) -> Any:
        """获取关键词配置"""
        keywords = self.keyword_config.get_other_keywords(category)

        if subcategory:
            return keywords.get(subcategory, {})

        return keywords
