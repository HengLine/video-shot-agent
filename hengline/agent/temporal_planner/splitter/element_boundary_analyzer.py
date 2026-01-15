"""
@FileName: element_boundary_analyzer.py
@Description: 
@Author: HengLine
@Time: 2026/1/14 23:59
"""
from typing import Dict, List

from hengline.agent.temporal_planner.temporal_planner_model import ElementType


class ElementBoundaryAnalyzer:
    """元素边界分析器"""

    @staticmethod
    def find_natural_boundaries(description: str, element_type: ElementType) -> List[float]:
        """寻找自然边界点（在元素内的相对位置，0-1）"""
        boundaries = []

        if element_type == ElementType.DIALOGUE:
            # 对话的自然边界：句子结束、疑问、停顿
            if "。" in description or "！" in description or "？" in description:
                boundaries.append(0.95)  # 接近结尾

            # 对话中的逗号可能是小边界
            if "，" in description:
                boundaries.append(0.5)

        elif element_type == ElementType.ACTION:
            # 动作的自然边界：动作完成点
            action_completion_words = ["完成", "结束", "之后", "然后", "接着"]
            for word in action_completion_words:
                if word in description:
                    boundaries.append(0.8)
                    break

            # 复合动作的中间点
            if "然后" in description or "接着" in description:
                boundaries.append(0.5)

        elif element_type == ElementType.SCENE:
            # 场景的自然边界：视觉焦点转移
            if "然后" in description or "接着" in description:
                boundaries.append(0.5)

        return boundaries

    @staticmethod
    def assess_split_quality(split_point: float, element_type: ElementType,
                             element_data: Dict) -> float:
        """评估切割点质量（0-1，越高越好）"""
        quality = 1.0

        # 避免在元素极短部分切割
        if split_point < 0.1 or split_point > 0.9:
            quality *= 0.6

        # 对话切割惩罚
        if element_type == ElementType.DIALOGUE:
            quality *= 0.7

            # 检查是否在语义完整处切割
            description = element_data.get("description", "")
            if split_point > 0.5 and "？" in description:
                # 在疑问句后切割较好
                quality *= 1.2

        # 动作切割评估
        elif element_type == ElementType.ACTION:
            # 在动作完成点切割较好
            completion_words = ["完成", "结束", "之后"]
            description = element_data.get("description", "")
            has_completion = any(word in description for word in completion_words)

            if has_completion and split_point > 0.7:
                quality *= 1.3

        return min(max(quality, 0.0), 1.0)