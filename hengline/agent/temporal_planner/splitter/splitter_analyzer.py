"""
@FileName: splitter_analyzer.py
@Description: 叙事分析器
@Author: HengLine
@Time: 2026/1/15 23:57
"""
from typing import Dict, Optional, Any, List

from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, ElementType


class NarrativeAnalyzer:
    """叙事分析器"""

    def analyze_structure(self, estimations: Dict[str, DurationEstimation],
                          element_order: List[str]) -> Dict[str, Any]:
        """分析叙事结构"""
        structure = {
            "emotional_arc": [],  # 情感弧线
            "key_moments": [],  # 关键时刻
            "scene_boundaries": [],  # 场景边界
            "dialogue_blocks": [],  # 对话块
            "action_sequences": []  # 动作序列
        }

        current_emotion = "neutral"
        current_scene = None
        dialogue_block = []
        action_sequence = []

        for element_id in element_order:
            if element_id not in estimations:
                continue

            element = estimations[element_id]

            # 跟踪情感变化
            emotional_weight = getattr(element, 'emotional_weight', 1.0)
            if emotional_weight > 1.5:
                structure["emotional_arc"].append({
                    "element_id": element_id,
                    "emotion": "high",
                    "time": sum(e.estimated_duration for e in estimations.values()
                                if list(estimations.keys()).index(e.element_id) <
                                list(estimations.keys()).index(element_id))
                })

            # 检测场景边界
            if element.element_type == ElementType.SCENE:
                if current_scene != element_id:
                    structure["scene_boundaries"].append(element_id)
                    current_scene = element_id

            # 检测对话块
            if element.element_type in [ElementType.DIALOGUE, ElementType.SILENCE]:
                dialogue_block.append(element_id)
            elif dialogue_block:
                structure["dialogue_blocks"].append(dialogue_block.copy())
                dialogue_block = []

            # 检测动作序列
            if element.element_type == ElementType.ACTION:
                action_sequence.append(element_id)
            elif action_sequence:
                structure["action_sequences"].append(action_sequence.copy())
                action_sequence = []

        # 处理最后的块
        if dialogue_block:
            structure["dialogue_blocks"].append(dialogue_block)
        if action_sequence:
            structure["action_sequences"].append(action_sequence)

        return structure

    def evaluate_coherence(self, current_segment: Dict,
                           next_element: DurationEstimation,
                           following_element: Optional[DurationEstimation]) -> float:
        """评估连贯性"""
        score = 1.0

        # 检查元素类型转换
        segment_elements = current_segment.get("contained_elements", [])
        if segment_elements:
            last_element_type = segment_elements[-1]["type"]
            next_element_type = next_element.element_type.value

            # 类型转换的合理性
            type_transitions = {
                ("scene", "dialogue"): 0.9,
                ("scene", "action"): 0.8,
                ("dialogue", "action"): 0.7,
                ("dialogue", "silence"): 0.9,
                ("action", "dialogue"): 0.6,
                ("silence", "dialogue"): 0.8
            }

            transition_score = type_transitions.get(
                (last_element_type, next_element_type), 0.5
            )
            score = score * 0.3 + transition_score * 0.7

        # 检查情感连续性
        segment_emotional_flow = current_segment.get("emotional_flow", [])
        next_emotional_weight = getattr(next_element, 'emotional_weight', 1.0)

        if segment_emotional_flow:
            last_emotional = segment_emotional_flow[-1]["emotional_weight"]
            emotional_change = abs(next_emotional_weight - last_emotional)

            if emotional_change > 1.0:
                score *= 0.8  # 情感变化过大

        return score

    def evaluate_split_coherence(self, current_segment: Dict,
                                 element: DurationEstimation,
                                 split_point: float,
                                 total_duration: float) -> float:
        """评估分割后的连贯性"""
        score = 0.6  # 基础分（分割本身会影响连贯性）

        # 分割点的合理性
        if split_point < total_duration * 0.3:
            score *= 0.8  # 分割点太靠前
        elif split_point > total_duration * 0.7:
            score *= 0.9  # 分割点太靠后
        else:
            score *= 1.1  # 分割点适中

        # 元素类型的考虑
        if element.element_type == ElementType.ACTION:
            # 动作在阶段之间分割较好
            score *= 1.1
        elif element.element_type == ElementType.SCENE:
            # 场景分割可能影响建立感
            score *= 0.9

        return min(max(score, 0.1), 1.0)

