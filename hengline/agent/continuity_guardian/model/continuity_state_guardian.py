"""
@FileName: continuity_state_guardian.py
@Description: 状态守护模块
@Author: HengLine
@Time: 2026/1/4 17:32
"""
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


class CharacterState:
    """角色状态管理"""

    def __init__(self, character_id: str, name: str):
        self.character_id = character_id
        self.name = name
        self.appearance: Dict[str, Any] = {}  # 外貌特征
        self.outfit: Dict[str, Any] = {}  # 服装装扮
        self.emotional_state: str = "neutral"  # 情绪状态
        self.physical_state: Dict[str, Any] = {}  # 物理状态（受伤、疲惫等）
        self.inventory: List[str] = []  # 携带物品
        self.position: Optional[Tuple[float, float, float]] = None  # 位置坐标
        self.orientation: float = 0.0  # 朝向角度
        self.interactions: List[str] = []  # 最近交互记录
        self.timestamp: datetime = datetime.now()

    def update_appearance(self, attributes: Dict[str, Any]):
        """更新外貌特征"""
        self.appearance.update(attributes)
        self.timestamp = datetime.now()

    def change_outfit(self, new_outfit: Dict[str, Any]):
        """更换服装"""
        self.outfit = new_outfit.copy()
        self.timestamp = datetime.now()

    def add_to_inventory(self, item_id: str):
        """添加物品到库存"""
        if item_id not in self.inventory:
            self.inventory.append(item_id)

    def remove_from_inventory(self, item_id: str):
        """从库存移除物品"""
        if item_id in self.inventory:
            self.inventory.remove(item_id)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "character_id": self.character_id,
            "name": self.name,
            "appearance": self.appearance,
            "outfit": self.outfit,
            "emotional_state": self.emotional_state,
            "physical_state": self.physical_state,
            "inventory": self.inventory,
            "position": self.position,
            "orientation": self.orientation,
            "interactions": self.interactions,
            "timestamp": self.timestamp.isoformat()
        }


class EnvironmentState:
    """环境状态管理"""

    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.time_of_day: str = "day"  # 时间：day/night/dawn/dusk
        self.weather: str = "clear"  # 天气
        self.lighting: Dict[str, Any] = {}  # 光照条件
        self.ambient_sounds: List[str] = []  # 环境音效
        self.active_effects: List[str] = []  # 活跃效果
        self.prop_positions: Dict[str, Tuple] = {}  # 场景内道具位置
        self.character_positions: Dict[str, Tuple] = {}  # 角色位置

    def change_time(self, new_time: str):
        """改变时间"""
        self.time_of_day = new_time

    def change_weather(self, new_weather: str):
        """改变天气"""
        self.weather = new_weather

    def add_effect(self, effect_id: str):
        """添加环境效果"""
        if effect_id not in self.active_effects:
            self.active_effects.append(effect_id)


class PropState:
    """道具状态管理"""

    def __init__(self, prop_id: str, name: str):
        self.prop_id = prop_id
        self.name = name
        self.position: Optional[Tuple[float, float, float]] = None
        self.orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.state: str = "default"  # 道具状态（破碎、打开、关闭等）
        self.owner: Optional[str] = None  # 当前持有者
        self.interaction_history: List[Dict] = []  # 交互历史
        self.physical_condition: Dict[str, Any] = {}  # 物理条件

    def update_position(self, position: Tuple[float, float, float]):
        """更新位置"""
        self.position = position

    def change_state(self, new_state: str):
        """改变状态"""
        self.state = new_state

    def record_interaction(self, character_id: str, action: str):
        """记录交互"""
        self.interaction_history.append({
            "character_id": character_id,
            "action": action,
            "timestamp": datetime.now()
        })

