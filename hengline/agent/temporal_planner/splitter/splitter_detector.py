"""
@FileName: splitter_detector.py
@Description: 
@Author: HengLine
@Time: 2026/1/16 23:05
"""
from typing import Dict, List


class KeyframeDetector:
    """关键帧检测器"""

    def detect_keyframes(self, seg1: Dict, seg2: Dict) -> List[Dict]:
        """检测需要视觉匹配的关键帧"""
        keyframes = []

        # 检测片段结束/开始的关键帧
        # 这里可以添加更复杂的检测逻辑
        if self._is_visual_transition(seg1, seg2):
            keyframes.append({
                "id": f"transition_{seg1['segment_id']}_{seg2['segment_id']}",
                "description": "片段过渡视觉匹配",
                "sora_prompt": "平滑过渡，保持视觉连续性",
                "needs_match": True,
                "mandatory": True
            })

        return keyframes

    def _is_visual_transition(self, seg1: Dict, seg2: Dict) -> bool:
        """判断是否是重要的视觉过渡"""
        # 简化实现：总是假设需要过渡匹配
        return True
