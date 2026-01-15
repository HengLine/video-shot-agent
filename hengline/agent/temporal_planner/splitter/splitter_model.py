"""
@FileName: splitter_model.py
@Description: 
@Author: HengLine
@Time: 2026/1/15 0:03
"""
from dataclasses import dataclass
from typing import Dict, Any, List

from hengline.agent.temporal_planner.temporal_planner_model import TimeSegment


@dataclass
class SplitResult:
    """切割结果"""
    split_point: float
    part1_duration: float
    remaining_duration: float


@dataclass
class SplitDecision:
    """切割决策"""
    element_id: str
    split_point: float  # 切割点（在元素内的相对时间）
    reason: str  # 切割原因
    quality_score: float = 0.0  # 切割质量得分

    # 切割点特征
    is_natural_boundary: bool = False  # 是否为自然边界
    visual_continuity: bool = True  # 是否保持视觉连贯性
    emotional_continuity: bool = True  # 是否保持情感连贯性


@dataclass
class SegmentSplitResult:
    """分片结果"""
    segments: List[TimeSegment]  # 生成的5秒片段
    split_decisions: List[SplitDecision]  # 切割决策记录
    statistics: Dict[str, Any]  # 统计信息

    # 质量指标
    overall_quality_score: float = 0.0
    pacing_consistency_score: float = 0.0
    continuity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "segments": [seg.to_dict() for seg in self.segments],
            "split_decisions": [
                {
                    "element_id": dec.element_id,
                    "split_point": dec.split_point,
                    "reason": dec.reason,
                    "quality_score": dec.quality_score
                }
                for dec in self.split_decisions
            ],
            "statistics": self.statistics,
            "quality_scores": {
                "overall": self.overall_quality_score,
                "pacing_consistency": self.pacing_consistency_score,
                "continuity": self.continuity_score
            }
        }
