"""
@FileName: temporal_planner_config.py
@Description: 时序规划智能体配置管理
@Author: HengLine
@Time: 2025/10/27 17:24
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

import yaml

from hengline.language_manage import Language, get_language_code
from hengline.logger import debug, warning, info


@dataclass
class DurationConfig:
    """时长估算配置"""

    # AI调整配置
    enable_ai_adjustment: bool = True
    ai_adjustment_threshold: float = 30.0  # 总时长超过30秒时使用AI调整
    ai_confidence_threshold: float = 0.7  # 置信度低于此值使用AI

    # 语速配置（字/秒）
    speech_rates: Dict[str, float] = field(default_factory=lambda: {
        "slow": 2.0,  # 悲伤、沉思、老人
        "normal": 3.0,  # 正常对话
        "fast": 4.0,  # 激动、争吵
        "yelling": 5.0  # 大喊
    })

    # 年龄影响语速因子
    age_speech_factors: Dict[str, float] = field(default_factory=lambda: {
        "child": 1.2,  # 儿童：语速快20%
        "teen": 1.1,  # 青少年：快10%
        "adult": 1.0,  # 成人：标准
        "elder": 0.7  # 老人：慢30%
    })

    # 情绪影响乘数
    emotion_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "平静": 1.0,
        "悲伤": 1.4,  # 悲伤时需要更多表达时间
        "愤怒": 1.2,  # 愤怒时可能停顿
        "恐惧": 1.3,  # 恐惧时可能结巴
        "喜悦": 0.9,  # 喜悦时可能语速快
        "激动": 0.8,  # 激动时语速快
        "疑问": 1.1,  # 疑问时可能有停顿
        "大喊": 0.7  # 大喊时语速极快
    })

    # 时长限制（秒）
    min_dialogue_duration: float = 1.0
    max_dialogue_duration: float = 15.0
    min_action_duration: float = 0.5
    max_action_duration: float = 10.0
    min_description_duration: float = 2.0
    max_description_duration: float = 8.0

    # 停顿时间配置
    pause_config: Dict[str, float] = field(default_factory=lambda: {
        "comma": 0.3,  # 逗号停顿
        "period": 0.8,  # 句号停顿
        "exclamation": 1.0,  # 感叹号停顿
        "question": 1.2,  # 问号停顿
        "ellipsis": 1.5  # 省略号停顿
    })

    # 反应时间配置（秒）
    reaction_times: Dict[str, float] = field(default_factory=lambda: {
        "平静": 0.5,
        "惊讶": 1.5,
        "震惊": 2.0,
        "思考": 1.8,
        "犹豫": 1.2
    })

    # 动作基础时长（秒）
    action_base_times: Dict[str, Any] = field(default_factory=lambda: {
        "walk": {"slow": 4.0, "normal": 2.5, "fast": 1.5},
        "run": {"slow": 3.0, "normal": 2.0, "fast": 1.0},
        "sit": 2.5,
        "stand": 2.0,
        "turn": 1.5,
        "look": {"glance": 1.0, "stare": 3.0, "scan": 2.0},
        "gesture": {"simple": 1.0, "complex": 2.5, "emphatic": 1.8},
        "facial": {"subtle": 1.5, "strong": 3.0, "change": 2.0},
        "touch": {"light": 1.2, "firm": 1.8, "grab": 2.2}
    })


@dataclass
class SegmentConfig:
    """分片配置"""

    # 目标时长配置
    target_segment_duration: float = 5.0
    min_segment_duration: float = 3.0
    max_segment_duration: float = 7.0
    max_segments_per_scene: int = 100
    segment_tolerance: float = 1.0  # 允许的偏差

    # 边界优化配置
    avoid_cutting_mid_dialogue: bool = True # 避免在对话中间切割
    avoid_cutting_mid_action: bool = True   # 避免在动作中间切割
    prefer_natural_breaks: bool = True  # 优先选择自然停顿点
    optimize_boundaries: bool = True    # 启用边界优化
    split_long_segments: bool = True    # 分割过长片段

    # 合并/分割策略
    min_content_for_segment: float = 2.0  # 片段最少内容时长
    max_content_for_segment: float = 6.0  # 片段最多内容时长

    # 连续性权重
    visual_continuity_weight: float = 0.6
    audio_continuity_weight: float = 0.4

    # 关键帧检测
    detect_keyframes: bool = True
    keyframe_threshold: float = 0.7

    # 情绪连贯性
    maintain_emotional_flow: bool = True
    emotion_transition_smoothness: float = 0.8


@dataclass
class TemporalPlanningConfig:
    """时序规划配置"""

    # 时长估算配置
    duration_config: DurationConfig = field(default_factory=lambda: DurationConfig(
        enable_ai_adjustment=True,
        ai_adjustment_threshold=30.0,  # 总时长超过30秒时使用AI调整

        # 语速配置（字/秒）
        speech_rates={
            "slow": 2.0,
            "normal": 3.0,
            "fast": 4.0,
            "yelling": 5.0
        },

        # 时长限制
        min_dialogue_duration=1.0,
        max_dialogue_duration=15.0,
        min_action_duration=0.5,
        max_action_duration=10.0
    ))

    # 分片配置
    segment_config: SegmentConfig = field(default_factory=lambda: SegmentConfig(
        target_segment_duration=5.0,
        min_segment_duration=3.0,
        max_segment_duration=7.0,

        # 边界优化
        avoid_cutting_mid_dialogue=True,
        avoid_cutting_mid_action=True,
        prefer_natural_breaks=True
    ))


@dataclass
class AIDurationConfig:
    """AI时长调整配置"""

    # 复杂度阈值
    complexity_threshold: float = 0.6  # 超过此分数使用AI调整

    # 复杂度权重
    complexity_weights: Dict[str, float] = field(default_factory=lambda: {
        "emotional": 0.35,  # 情感复杂度权重
        "interaction": 0.25,  # 交互复杂度权重
        "action": 0.15,  # 动作复杂度权重
        "dialogue": 0.15,  # 对话复杂度权重
        "confidence": 0.10  # 置信度权重
    })

    # 调整限制
    max_adjustment_ratio: float = 0.4  # 最大调整幅度 ±40%
    max_scene_adjustment_ratio: float = 0.2  # 场景总时长最大变化 ±20%

    # AI权重配置
    ai_weight: float = 0.6  # AI调整的基础权重
    min_ai_weight: float = 0.3  # 最小AI权重
    max_ai_weight: float = 0.8  # 最大AI权重

    # 时长限制
    min_dialogue_duration: float = 1.0
    max_dialogue_duration: float = 15.0
    min_action_duration: float = 0.5
    max_action_duration: float = 10.0

    # LLM配置
    llm_temperature: float = 0.1  # 低温度保证稳定性
    llm_max_tokens: int = 2000
    max_scenes_per_request: int = 3  # 每次请求最多处理3个场景

    # 缓存配置
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 缓存1小时
    force_refresh: bool = False  # 强制刷新缓存

    def _get_default_config(self):
        """获取默认配置"""
        return AIDurationConfig()


class TemporalPlannerConfig:
    """时序规划智能体配置类"""

    def __init__(self, language: Language = None):
        """初始化配置管理器
        
        Args:
            language: 语言枚举，默认使用系统设置的语言
        """
        # 设置当前语言
        if language:
            self._language = language.value
        else:
            self._language = get_language_code()

        # 设置配置文件路径
        self._set_config_path()

        self._config_data = {}
        self._base_actions = {}
        self._modifiers = {}
        self._dialogue_config = {}
        self._character_speed_factors = {}
        self._target_segment_duration = 5.0
        self._max_duration_deviation = 0.5
        self._min_action_duration = 0.4
        self._default_duration = 0.8

        # 加载配置
        self.load_configuration()

    def _set_config_path(self):
        """设置配置文件路径"""
        # 根据语言选择配置文件路径
        if self._language == Language.EN:
            self.config_path = Path(__file__).parent / "en" / "action_duration_config.yaml"
        else:
            self.config_path = Path(__file__).parent / "zh" / "action_duration_config.yaml"

    def load_configuration(self):
        """从配置文件加载动作时长数据"""
        if not self.config_path.exists():
            warning(f"配置文件不存在: {self.config_path}，使用默认配置")
            self._set_default_config()
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)

            # 加载基础动作时长
            self._base_actions = self._config_data.get('base_actions', {})

            # 加载修饰词修正系数
            self._modifiers = self._config_data.get('modifiers', {})

            # 加载对话时长配置
            self._dialogue_config = self._config_data.get('dialogue', {})

            # 加载角色速度因子
            self._character_speed_factors = self._config_data.get('character_speed_factors', {})

            # 加载分段策略配置
            segmentation_config = self._config_data.get('segmentation', {})
            self._target_segment_duration = segmentation_config.get('target_duration', 5.0)
            self._max_duration_deviation = segmentation_config.get('max_buffer', 0.5)
            self._min_action_duration = segmentation_config.get('min_action_duration', 0.4)

            # 默认动作时长
            self._default_duration = 0.8

            info(f"成功加载配置文件: {self.config_path}")
            debug(f"加载了 {len(self._base_actions)} 个基础动作配置")

        except Exception as e:
            warning(f"加载配置文件失败: {str(e)}，使用默认配置")
            self._set_default_config()

    def _set_default_config(self):
        """设置默认配置"""
        self._base_actions = {
            "站立": 0.5, "坐下": 1.3, "说话": 3.0, "移动": 1.0, "转身": 1.0,
            "抬手": 0.5, "点头": 0.4, "摇头": 0.5, "眨眼": 0.2, "思考": 2.0
        }
        self._modifiers = {
            "快速": 0.7, "慢慢": 1.7, "轻轻": 0.9, "重重": 1.3,
            "缓缓": 1.5, "突然": 0.8, "持续": 1.8
        }
        self._dialogue_config = {
            "base_per_char": 0.35,
            "min_duration": 1.5,
            "max_duration": 6.0,
            "emotion_multipliers": {"默认": 1.0, "愤怒": 1.2, "悲伤": 1.4, "兴奋": 0.9}
        }
        self._character_speed_factors = {"default": 1.0, "老人": 1.5, "儿童": 0.8}
        self._target_segment_duration = 5.0
        self._max_duration_deviation = 0.5
        self._min_action_duration = 0.4
        self._default_duration = 0.8

    # Getter methods
    @property
    def base_actions(self) -> Dict[str, float]:
        """获取基础动作时长"""
        return self._base_actions

    @property
    def modifiers(self) -> Dict[str, float]:
        """获取修饰词修正系数"""
        return self._modifiers

    @property
    def dialogue_config(self) -> Dict[str, Any]:
        """获取对话时长配置"""
        return self._dialogue_config

    @property
    def character_speed_factors(self) -> Dict[str, float]:
        """获取角色速度因子"""
        return self._character_speed_factors

    @property
    def target_segment_duration(self) -> float:
        """获取目标分段时长"""
        return self._target_segment_duration

    @property
    def max_duration_deviation(self) -> float:
        """获取最大时长偏差"""
        return self._max_duration_deviation

    @property
    def min_action_duration(self) -> float:
        """获取最小动作时长"""
        return self._min_action_duration

    @property
    def default_duration(self) -> float:
        """获取默认动作时长"""
        return self._default_duration

    # Setter methods
    @target_segment_duration.setter
    def target_segment_duration(self, value: float):
        """设置目标分段时长"""
        if value > 0:
            self._target_segment_duration = value
            debug(f"目标分段时长已设置为: {value}秒")

    @max_duration_deviation.setter
    def max_duration_deviation(self, value: float):
        """设置最大时长偏差"""
        if value >= 0:
            self._max_duration_deviation = value
            debug(f"最大时长偏差已设置为: {value}秒")

    def set_language(self, language: Language):
        """设置语言并重新加载配置
        
        Args:
            language: 语言枚举
        """
        if language.value != self._language:
            self._language = language.value
            self._set_config_path()
            self.load_configuration()
            debug(f"时序规划配置语言已切换为: {self._language}")

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "base_actions_count": len(self._base_actions),
            "modifiers_count": len(self._modifiers),
            "target_segment_duration": self._target_segment_duration,
            "max_duration_deviation": self._max_duration_deviation,
            "min_action_duration": self._min_action_duration,
            "language": self._language
        }


# 创建全局配置实例
planner_config = TemporalPlannerConfig()


def get_planner_config(language: Language = None) -> TemporalPlannerConfig:
    """
    获取时序规划配置实例
    
    Args:
        language: 语言枚举，默认使用系统设置的语言
        
    Returns:
        TemporalPlannerConfig: 配置实例
    """
    global planner_config

    if language and language.value != planner_config._language:
        planner_config.set_language(language)

    return planner_config


def reload_configuration() -> TemporalPlannerConfig:
    """
    重新加载配置文件
    """
    global planner_config
    planner_config.load_configuration()
    debug("配置已重新加载")
    return planner_config
