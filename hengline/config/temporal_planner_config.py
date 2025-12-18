"""
@FileName: temporal_planner_config.py
@Description: 时序规划智能体配置管理
@Author: HengLine
@Time: 2025/10/27 17:24
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from hengline.logger import debug, warning, info
from hengline.language_manage import Language, get_language_code


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
        if self._language == Language.EN.value:
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

