"""
@FileName: splitter_tracker.py
@Description: 
@Author: HengLine
@Time: 2026/1/16 0:54
"""
from typing import Dict

from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, ElementType


class StateTracker:
    """状态跟踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置状态"""
        self.character_states = {}  # 角色状态
        self.prop_states = {}  # 道具状态
        self.scene_state = {}  # 场景状态
        self.camera_state = None  # 摄像机状态
        self.time_elapsed = 0.0  # 已过时间

    def update(self, element: DurationEstimation, duration: float):
        """更新状态"""
        self.time_elapsed += duration

        # 更新角色状态
        if hasattr(element, 'character_states'):
            self.character_states.update(element.character_states)

        # 更新道具状态
        if hasattr(element, 'prop_states'):
            self.prop_states.update(element.prop_states)

        # 更新场景状态（如果是场景元素）
        if element.element_type == ElementType.SCENE:
            scene_data = getattr(element, 'raw_data', {})
            self.scene_state = {
                "location": scene_data.get("location", ""),
                "time_of_day": scene_data.get("time_of_day", ""),
                "weather": scene_data.get("weather", "")
            }

    def update_partial(self, element: DurationEstimation, duration: float, part_type: str):
        """更新部分元素状态"""
        self.time_elapsed += duration

        if part_type == "end":
            # 元素结束，更新最终状态
            self.update(element, 0)  # 只更新状态，不增加时间


class VisualConsistencyChecker:
    """视觉一致性检查器"""

    def evaluate_consistency(self, current_segment: Dict,
                             next_element: DurationEstimation) -> float:
        """评估视觉一致性"""
        score = 1.0

        # 检查场景一致性
        segment_scene = current_segment.get("scene_states", {})
        if segment_scene and next_element.element_type == ElementType.SCENE:
            next_scene_data = getattr(next_element, 'raw_data', {})
            next_scene = {
                "location": next_scene_data.get("location", ""),
                "time_of_day": next_scene_data.get("time_of_day", "")
            }

            # 检查场景是否变化
            if segment_scene.get("location") != next_scene.get("location"):
                score *= 0.7  # 场景变化

        # 检查摄像机角度
        segment_angles = current_segment.get("camera_angles", [])
        next_visual_hints = getattr(next_element, 'visual_hints', {})
        next_angles = next_visual_hints.get("suggested_shot_types", [])

        if segment_angles and next_angles:
            # 检查角度转换的合理性
            last_angle = segment_angles[-1]
            first_next_angle = next_angles[0]

            # 角度转换矩阵（简化）
            angle_transitions = {
                ("close_up", "medium_shot"): 0.9,
                ("medium_shot", "close_up"): 0.8,
                ("medium_shot", "wide_shot"): 0.7,
                ("wide_shot", "medium_shot"): 0.8
            }

            transition_score = angle_transitions.get(
                (last_angle, first_next_angle), 0.6
            )
            score = score * 0.6 + transition_score * 0.4

        return score

    def evaluate_split_consistency(self, current_segment: Dict,
                                   element: DurationEstimation,
                                   split_point: float) -> float:
        """评估分割的视觉一致性"""
        score = 0.7  # 分割本身会影响一致性

        # 元素类型考虑
        if element.element_type == ElementType.SCENE:
            # 场景分割需要特别注意视觉连续性
            score *= 0.8

        # 检查是否有视觉提示可以帮助分割
        visual_hints = getattr(element, 'visual_hints', {})
        if visual_hints.get("key_visuals"):
            # 有关键视觉元素可以帮助定位分割点
            score *= 1.1

        return min(max(score, 0.1), 1.0)
