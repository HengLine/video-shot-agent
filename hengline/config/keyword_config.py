# -*- coding: utf-8 -*-
"""
@FileName: keyword_config.py
@Description: 统一关键词配置加载器
@Author: HengLine
@Time: 2025/12/18
"""
import os
import yaml
from typing import Dict, List, Any

from hengline.logger import debug, warning, info, error

from hengline.language_manage import Language, get_language_code, set_language


class KeywordConfig:
    """统一关键词配置加载器"""
    
    def __init__(self, config_path: str = None, language: Language = None):
        """
        初始化关键词配置加载器
        
        Args:
            config_path: 配置文件路径，默认使用hengline/config/keyword_config.yaml
            language: 默认语言，使用Language枚举
        """
        self._config_data = {}
        self._pose_mapping = {}
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 处理语言参数
        if language:
            self._language = language.value
        else:
            # 使用默认语言
            self._language = get_language_code()
        
        # 根据语言选择配置文件路径
        if config_path:
            self._config_file = config_path
        else:
            if self._language == Language.EN.value:
                self._config_file = os.path.join(current_dir, 'en', 'keyword_config.yaml')
            else:
                # 默认使用中文配置
                self._config_file = os.path.join(current_dir, 'zh', 'keyword_config.yaml')
        
        # 加载配置
        self.load_configuration()
    
    def load_configuration(self):
        """
        加载配置文件
        """
        try:
            with open(self._config_file, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f) or {}
                
            # 更新姿态映射
            self._pose_mapping = self._config_data.get('pose_keywords', {})
            
            info(f"成功加载关键词配置文件: {self._config_file}")
        except FileNotFoundError:
            error(f"关键词配置文件不存在: {self._config_file}")
            self._config_data = {}
            self._pose_mapping = {}
        except Exception as e:
            error(f"加载关键词配置文件时发生错误: {str(e)}")
            self._config_data = {}
            self._pose_mapping = {}

    def get_pose_keywords(self, language: Language = None) -> Dict[str, Any]:
        """
        获取姿态相关关键词
        
        Args:
            language: 语言，使用Language枚举，默认使用当前设置的语言
            
        Returns:
            Dict: 姿态关键词配置
        """
        return self._get_keywords('pose_keywords', language)
    
    def get_action_keywords(self, language: Language = None) -> Dict[str, Any]:
        """
        获取动作识别相关关键词
        
        Args:
            language: 语言，使用Language枚举，默认使用当前设置的语言
            
        Returns:
            Dict: 动作识别关键词配置
        """
        return self._get_keywords('action_keywords', language)
    
    def get_emotion_keywords(self, language: Language = None) -> Dict[str, Any]:
        """
        获取情绪相关关键词
        
        Args:
            language: 语言，使用Language枚举，默认使用当前设置的语言
            
        Returns:
            Dict: 情绪关键词配置
        """
        return self._get_keywords('emotion_keywords', language)
    
    def get_scene_keywords(self, language: Language = None) -> Dict[str, Any]:
        """
        获取场景相关关键词
        
        Args:
            language: 语言，使用Language枚举，默认使用当前设置的语言
            
        Returns:
            Dict: 场景关键词配置
        """
        return self._get_keywords('scene_keywords', language)
    
    def get_character_keywords(self, language: Language = None) -> Dict[str, Any]:
        """
        获取角色相关关键词
        
        Args:
            language: 语言，使用Language枚举，默认使用当前设置的语言
            
        Returns:
            Dict: 角色关键词配置
        """
        return self._get_keywords('character_keywords', language)
    
    def get_dialogue_keywords(self, language: Language = None) -> Dict[str, Any]:
        """
        获取对话相关关键词
        
        Args:
            language: 语言，使用Language枚举，默认使用当前设置的语言
            
        Returns:
            Dict: 对话关键词配置
        """
        return self._get_keywords('dialogue_keywords', language)
    
    def get_position_keywords(self, language: Language = None) -> Dict[str, Any]:
        """
        获取位置相关关键词
        
        Args:
            language: 语言，使用Language枚举，默认使用当前设置的语言
            
        Returns:
            Dict: 位置关键词配置
        """
        return self._get_keywords('position_keywords', language)
    
    def get_prop_keywords(self, language: Language = None) -> Dict[str, Any]:
        """
        获取道具相关关键词
        
        Args:
            language: 语言，使用Language枚举，默认使用当前设置的语言
            
        Returns:
            Dict: 道具关键词配置
        """
        return self._get_keywords('prop_keywords', language)
    
    def get_action_split_keywords(self, language: Language = None) -> Dict[str, Any]:
        """
        获取动作拆分相关关键词
        
        Args:
            language: 语言，使用Language枚举，默认使用当前设置的语言
            
        Returns:
            Dict: 动作拆分关键词配置
        """
        return self._get_keywords('action_split_keywords', language)
    
    def get_gaze_keywords(self, language: Language = None) -> Dict[str, Any]:
        """
        获取视线方向相关关键词
        
        Args:
            language: 语言，使用Language枚举，默认使用当前设置的语言
            
        Returns:
            Dict: 视线方向关键词配置
        """
        return self._get_keywords('gaze_keywords', language)
    
    def get_state_keywords(self, language: Language = None) -> Dict[str, Any]:
        """
        获取状态特征相关关键词
        
        Args:
            language: 语言，使用Language枚举，默认使用当前设置的语言
            
        Returns:
            Dict: 状态特征关键词配置
        """
        return self._get_keywords('state_keywords', language)
    
    def _get_keywords(self, category: str, language: Language = None) -> Dict[str, Any]:
        """
        内部方法：获取指定分类和语言的关键词
        
        Args:
            category: 关键词分类
            language: 语言，使用Language枚举，默认使用当前设置的语言
            
        Returns:
            Dict: 关键词配置
        """
        # 如果指定了语言并且与当前语言不同，则临时加载该语言的配置
        if language and language.value != self._language:
            temp_config = {}  # 临时配置
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 根据指定的语言选择配置文件路径
            temp_config_path = os.path.join(current_dir, language.value, 'keyword_config.yaml')
            
            try:
                with open(temp_config_path, 'r', encoding='utf-8') as f:
                    temp_config = yaml.safe_load(f) or {}
            except Exception as e:
                error(f"加载临时关键词配置文件时发生错误: {str(e)}")
                # 返回空字典或当前语言的配置
                return self._config_data.get(category, {})
            
            return temp_config.get(category, {})
        
        # 如果没有指定语言或语言与当前语言相同，则返回当前配置
        return self._config_data.get(category, {})
    
    def get_pose_mapping(self, language: Language = None) -> Dict[str, str]:
        """
        获取姿态标准化映射
        
        Args:
            language: 语言，使用Language枚举，默认使用当前设置的语言
            
        Returns:
            Dict: 姿态标准化映射
        """
        # 如果指定了语言并且与当前语言不同，则临时加载该语言的配置
        if language and language.value != self._language:
            temp_config = {}  # 临时配置
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 根据指定的语言选择配置文件路径
            temp_config_path = os.path.join(current_dir, language.value, 'keyword_config.yaml')
            
            try:
                with open(temp_config_path, 'r', encoding='utf-8') as f:
                    temp_config = yaml.safe_load(f) or {}
                    return temp_config.get('pose_keywords', {})
            except Exception as e:
                error(f"加载临时关键词配置文件时发生错误: {str(e)}")
                # 返回当前语言的配置
                return self._pose_mapping
        
        # 如果没有指定语言或语言与当前语言相同，则返回当前配置
        return self._pose_mapping
    
    def reload_configuration(self):
        """
        重新加载配置文件
        """
        self.load_configuration()
        debug("关键词配置已重新加载")
    
    def set_language(self, language: Language):
        """
        设置语言
        
        Args:
            language: 语言，使用Language枚举
        """
        if language.value != self._language:
            self._language = language.value
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 更新配置文件路径
            if self._language == Language.EN.value:
                self._config_file = os.path.join(current_dir, 'en', 'keyword_config.yaml')
            else:
                # 默认使用中文配置
                self._config_file = os.path.join(current_dir, 'zh', 'keyword_config.yaml')
            
            # 重新加载配置
            self.load_configuration()


# 创建全局配置实例
keyword_config = KeywordConfig()


def get_keyword_config() -> KeywordConfig:
    """
    获取关键词配置实例
    
    Returns:
        KeywordConfig: 配置实例
    """
    return keyword_config


def reload_keyword_configuration():
    """
    重新加载关键词配置文件
    """
    global keyword_config
    keyword_config.load_configuration()
    debug("关键词配置已重新加载")

