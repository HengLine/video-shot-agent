"""
@FileName: continuity_guardian_config.py
@Description: 
@Author: HengLine
@Time: 2025/11/3 21:31
"""
import os
from typing import Dict, List, Any, Optional

import yaml

from hengline.logger import debug, warning


class ContinuityGuardianConfig:
    def __init__(self):
        """初始化连续性守护智能体"""
        # 角色状态记忆
        self.character_states = {}

        # 加载连续性守护智能体配置
        self.config = self._load_config()

        # 从配置中加载默认角色外观
        self.default_appearances = self.config.get('default_appearances', {})

        # 构建情绪映射表
        self._build_emotion_mapping()

    def _load_config(self) -> Dict[str, Any]:
        """加载连续性守护智能体配置"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config', 'continuity_guardian_config.yaml'
        )

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            debug(f"成功加载连续性守护智能体配置: {config_path}")
            return config
        except Exception as e:
            warning(f"加载连续性守护智能体配置失败: {e}")
            # 返回默认配置
            return {
                'default_appearances': {
                    "pose": "站立",
                    "position": "画面中央",
                    "emotion": "平静",
                    "gaze_direction": "前方",
                    "holding": "无"
                },
                'emotion_mapping': {},
                'emotion_transition_rules': {}
            }


    def _build_emotion_mapping(self):
        """从配置构建情绪映射表"""
        self.emotion_categories = {}

        # 从配置加载情绪映射
        emotion_mapping = self.config.get('emotion_mapping', {})

        # 构建情绪到类别的映射（配置文件中已直接使用中文类别）
        for category, emotions in emotion_mapping.items():
            for emotion in emotions:
                self.emotion_categories[emotion] = category

        debug(f"构建的情绪映射表: {self.emotion_categories}")

    def load_prev_state(self, prev_continuity_state: Optional[Dict[str, Any]]):
        """加载上一段的连续性状态"""
        for state in prev_continuity_state:
            character_name = state.get("character_name")
            if character_name:
                self.character_states[character_name] = state

    def extract_characters(self, segment: Dict[str, Any]) -> List[str]:
        """提取段落中的所有角色"""
        characters = set()
        for action in segment.get("actions", []):
            if "character" in action:
                characters.add(action["character"])
        return list(characters)

    def get_character_state(self, character_name: str) -> Dict[str, Any]:
        """获取角色的当前状态"""
        if character_name in self.character_states:
            return self.character_states[character_name].copy()
        else:
            # 返回默认状态
            return {
                "character_name": character_name,
                **self.default_appearances
            }

    
    def _generate_character_constraints(self, character_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成角色连续性约束"""
        return {
            "must_start_with_pose": state.get("pose", "unknown"),
            "must_start_with_position": state.get("position", "unknown"),
            "must_start_with_emotion": state.get("emotion", "unknown"),
            "must_start_with_gaze": state.get("gaze_direction", "unknown"),
            "must_start_with_holding": state.get("holding", "unknown"),
            "character_description": self._generate_character_description(character_name, state)
        }

    def _generate_character_description(self, character_name: str, state: Dict[str, Any]) -> str:
        """生成角色描述"""
        # 可以根据需要生成更详细的角色描述
        # 暂时使用简单的描述模板
        return f"{character_name}, {state.get('pose')}, {state.get('emotion')}"
