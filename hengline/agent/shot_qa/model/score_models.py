"""
@FileName: score_models.py
@Description: 评分相关数据模型
@Author: HengLine
@Time: 2026/1/6 15:59
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class QualityScores:
    """质量评分"""

    # 连续性评分
    continuity_score: float = 1.0
    position_consistency_score: float = 1.0
    appearance_consistency_score: float = 1.0
    temporal_consistency_score: float = 1.0

    # 约束满足评分
    constraint_satisfaction_score: float = 1.0
    critical_constraint_score: float = 1.0
    overall_constraint_score: float = 1.0

    # 视觉质量评分
    visual_quality_score: float = 1.0
    composition_score: float = 1.0
    lighting_score: float = 1.0
    color_score: float = 1.0
    style_consistency_score: float = 1.0

    # 技术质量评分
    technical_quality_score: float = 1.0
    prompt_quality_score: float = 1.0
    camera_quality_score: float = 1.0
    feasibility_score: float = 1.0

    # 整体评分
    overall_quality_score: float = 1.0
    weighted_quality_score: float = 1.0

    # 置信度
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "continuity": {
                "overall": self.continuity_score,
                "position": self.position_consistency_score,
                "appearance": self.appearance_consistency_score,
                "temporal": self.temporal_consistency_score
            },
            "constraints": {
                "overall": self.constraint_satisfaction_score,
                "critical": self.critical_constraint_score,
                "weighted": self.overall_constraint_score
            },
            "visual": {
                "overall": self.visual_quality_score,
                "composition": self.composition_score,
                "lighting": self.lighting_score,
                "color": self.color_score,
                "style": self.style_consistency_score
            },
            "technical": {
                "overall": self.technical_quality_score,
                "prompt": self.prompt_quality_score,
                "camera": self.camera_quality_score,
                "feasibility": self.feasibility_score
            },
            "overall": {
                "score": self.overall_quality_score,
                "weighted": self.weighted_quality_score
            }
        }


@dataclass
class ScoreWeighting:
    """评分权重配置"""

    # 连续性权重
    continuity_weight: float = 0.25
    position_weight: float = 0.4
    appearance_weight: float = 0.3
    temporal_weight: float = 0.3

    # 约束权重
    constraint_weight: float = 0.30
    critical_constraint_weight: float = 0.5
    high_constraint_weight: float = 0.3
    medium_constraint_weight: float = 0.15
    low_constraint_weight: float = 0.05

    # 视觉质量权重
    visual_weight: float = 0.25
    composition_weight: float = 0.3
    lighting_weight: float = 0.25
    color_weight: float = 0.25
    style_weight: float = 0.2

    # 技术质量权重
    technical_weight: float = 0.20
    prompt_weight: float = 0.4
    camera_weight: float = 0.3
    feasibility_weight: float = 0.3
