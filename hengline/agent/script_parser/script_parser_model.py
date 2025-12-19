"""
@FileName: script_parser_model.py
@Description: 剧本解析模型定义模块
@Author: HengLine
@Time: 2025/12/18 22:35
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Scene:
    """场景的纯内容表示（不含时间）"""
    scene_id: str
    order_index: int  # 逻辑顺序，不是时间顺序

    # 内容要素
    location: str  # 地点
    time_of_day: str  # 时间（白天/夜晚）
    mood: str  # 氛围
    summary: Optional[Any]  # 场景摘要

    # 包含的元素引用
    character_refs: List[str]  # 出现的角色
    dialogue_refs: List[str]  # 包含的对话
    action_refs: List[str]  # 包含的动作
    description: str = ""  # 场景描述文本

    # 裁剪的时间范围
    start_time: Optional[float] = 0  # 开始时间（秒）
    end_time: Optional[float] = 5  # 结束时间（秒）
    duration: Optional[float] = 5  # 持续时间（秒）


@dataclass
class Dialogue:
    """对话的纯文本表示"""
    dialogue_id: str
    speaker: str  # 说话者
    text: str  # 对话内容
    emotion: str  # 情绪标签（从文本推断）
    parenthetical: str  # 表演提示（如果有）
    scene_ref: str  # 所属场景


@dataclass
class Action:
    """动作的语义描述"""
    action_id: str
    type: str  # 动作类型：physical, verbal等
    category: str  # 动作类别：movement, gesture, facial等
    actor: str  # 执行者
    target: str  # 目标（如果有）
    description: str  # 动作描述文本
    intensity: int  # 强度 1-5
    scene_ref: str  # 所属场景


@dataclass
class Character:
    """角色定义"""
    name: str
    age: int
    gender: str
    role_hint: str
    description: str
    appearance: Dict[str, str] = field(default_factory=dict)
    current_state: Dict[str, Any] = field(default_factory=dict)
    continuity_id: str = ""  # 用于跨片段跟踪


@dataclass
class UnifiedScript:
    """统一格式的剧本表示"""

    # 元数据
    script_type: str  # 原始格式类型
    script_hash: str  # 原始文本哈希
    parser_type: str  # 使用的解析器类型

    # 结构化内容
    scenes: List[Scene]  # 场景序列（按逻辑顺序）
    characters: List[Character]  # 角色信息
    dialogues: List[Dialogue]  # 对话内容
    actions: List[Action]  # 动作描述
    descriptions: List[str]  # 环境描述

    # 解析质量
    parsing_confidence: Dict[str, float]
