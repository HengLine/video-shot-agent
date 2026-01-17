"""
@FileName: splitter_builder.py
@Description: 元素顺序构建器
@Author: HengLine
@Time: 2026/1/16 23:47
"""

from typing import List, Dict

from hengline.agent.script_parser.script_parser_models import Scene, Dialogue, Action
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, ElementType, ScriptElement


class ElementSequenceBuilder:
    """构建剧本元素的正确时序顺序"""

    def __init__(self):
        self.time_tolerance = 0.5  # 时间容差（秒）

    def build_sequence(
            self,
            scenes: List[Scene],
            dialogues: List[Dialogue],
            actions: List[Action],
            duration_estimations: Dict[str, DurationEstimation]
    ) -> List[ScriptElement]:
        """构建包含所有元素的时序序列"""

        # 1. 转换所有元素为ScriptElement
        script_elements = []

        # 场景元素
        for scene in scenes:
            element = ScriptElement(
                element_id=f"{scene.scene_id}",
                element_type=ElementType.SCENE,
                original_data=scene,
                estimated_duration=duration_estimations.get(f"{scene.scene_id}"),
                order_priority=10  # 场景有最高优先级
            )
            script_elements.append(element)

        # 对话元素
        for dialogue in dialogues:
            element = ScriptElement(
                element_id=f"{dialogue.dialogue_id}",
                element_type=ElementType.DIALOGUE,
                original_data=dialogue,
                estimated_duration=duration_estimations.get(f"{dialogue.dialogue_id}"),
                order_priority=5
            )
            script_elements.append(element)

        # 动作元素
        for action in actions:
            element = ScriptElement(
                element_id=f"{action.action_id}",
                element_type=ElementType.ACTION,
                original_data=action,
                estimated_duration=duration_estimations.get(f"{action.action_id}"),
                order_priority=3
            )
            script_elements.append(element)

        # 2. 基于time_offset排序
        script_elements.sort(key=lambda x: self._get_element_time_offset(x))

        # 3. 构建依赖关系
        self._build_dependencies(script_elements)

        return script_elements

    def _get_element_time_offset(self, element: ScriptElement) -> float:
        """获取元素的时间偏移"""
        if element.element_type == ElementType.SCENE:
            return element.original_data.start_time
        elif element.element_type == ElementType.DIALOGUE:
            return element.original_data.time_offset
        elif element.element_type == ElementType.ACTION:
            return element.original_data.time_offset
        return 0.0

    def _build_dependencies(self, elements: List[ScriptElement]):
        """构建元素间的依赖关系"""
        for i, element in enumerate(elements):
            if i > 0:
                prev_element = elements[i - 1]

                # 如果当前元素紧接着前一个元素（考虑容差）
                current_time = self._get_element_time_offset(element)
                prev_end_time = self._get_element_time_offset(prev_element)
                prev_duration = prev_element.estimated_duration.estimated_duration if prev_element.estimated_duration else 0

                if abs(current_time - (prev_end_time + prev_duration)) < self.time_tolerance:
                    element.dependencies.append(prev_element.element_id)
