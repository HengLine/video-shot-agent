"""
@FileName: continuity_transition_guardian.py
@Description: 关键帧与转场
@Author: HengLine
@Time: 2026/1/4 17:36
"""
from typing import Dict, List, Any, Optional

from .continuity_state_guardian import CharacterState, PropState, EnvironmentState


class KeyframeAnchor:
    """关键帧锚点"""

    def __init__(self, scene_id: str, timestamp: float):
        self.scene_id = scene_id
        self.timestamp = timestamp
        self.characters: Dict[str, CharacterState] = {}
        self.props: Dict[str, PropState] = {}
        self.environment: Optional[EnvironmentState] = None
        self.continuity_checks: List[Dict] = []

    def add_character_state(self, character: CharacterState):
        """添加角色状态"""
        self.characters[character.character_id] = character

    def add_prop_state(self, prop: PropState):
        """添加道具状态"""
        self.props[prop.prop_id] = prop


class TransitionInstruction:
    """转场指令"""

    def __init__(self, from_scene: str, to_scene: str):
        self.from_scene = from_scene
        self.to_scene = to_scene
        self.transition_type: str = "cut"  # cut/fade/dissolve等
        from typing import Optional
        self.temporal_gap: Optional[float] = None  # 时间间隔
        self.spatial_changes: List[str] = []  # 空间变化描述
        self.character_transitions: Dict[str, Dict] = {}  # 角色转场

    def add_character_transition(self, character_id: str, changes: Dict[str, Any]):
        """添加角色转场信息"""
        self.character_transitions[character_id] = changes
