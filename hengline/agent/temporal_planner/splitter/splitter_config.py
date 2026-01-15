"""
@FileName: splitter_config.py
@Description: 
@Author: HengLine
@Time: 2026/1/14 23:58
"""
from hengline.agent.temporal_planner.temporal_planner_model import ElementType


class SplitterConfig:
    """分片器配置"""

    def __init__(self):
        # 分片参数
        self.target_segment_duration = 5.0
        self.segment_tolerance = 0.2  # 容忍度（秒）
        self.min_segment_duration = 4.0
        self.max_segment_duration = 6.0

        # 切割策略
        self.preserve_dialogues = True  # 优先保持对话完整
        self.preserve_emotional_moments = True  # 优先保持情感时刻完整
        self.avoid_splitting_key_actions = True  # 避免切割关键动作

        # 切割优先级（数字越小优先级越高）
        self.split_priority_map = {
            ElementType.SILENCE: 1,  # 沉默最优先不切割
            ElementType.DIALOGUE: 2,  # 对话次优先不切割
            ElementType.ACTION: 4,  # 动作可以切割
            ElementType.SCENE: 5,  # 场景较容易切割
            ElementType.TRANSITION: 3,  # 转场中等
            ElementType.UNKNOWN: 6  # 未知类型
        }

        # 视觉连贯性规则
        self.min_shot_duration = 1.0  # 最小镜头持续时间
        self.max_shot_duration = 8.0  # 最大镜头持续时间

        # 质量评估权重
        self.weights = {
            "duration_deviation": 0.3,  # 时长偏差权重
            "split_quality": 0.4,  # 切割质量权重
            "pacing_consistency": 0.2,  # 节奏一致性权重
            "visual_coherence": 0.1  # 视觉连贯性权重
        }