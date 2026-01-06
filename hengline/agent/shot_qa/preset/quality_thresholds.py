"""
@FileName: quality_thresholds.py
@Description: 质量阈值配置
@Author: HengLine
@Time: 2026/1/6 16:02
"""
from hengline.agent.shot_qa.model.review_models import QualityThresholds

# 不同严格度级别的阈值配置
QUALITY_THRESHOLD_PRESETS = {
    "strict": QualityThresholds(
        # 连续性阈值
        continuity_score_threshold=0.9,
        position_consistency_threshold=0.8,
        appearance_consistency_threshold=0.95,

        # 约束满足阈值
        constraint_satisfaction_threshold=0.95,
        critical_constraint_threshold=1.0,

        # 视觉质量阈值
        visual_appeal_threshold=0.8,
        style_consistency_threshold=0.85,
        color_consistency_threshold=0.8,

        # 技术质量阈值
        prompt_quality_threshold=0.7,
        technical_feasibility_threshold=0.85,

        # 整体质量阈值
        overall_quality_threshold=0.85,
        approval_threshold=0.9
    ),

    "balanced": QualityThresholds(
        # 连续性阈值
        continuity_score_threshold=0.8,
        position_consistency_threshold=0.7,
        appearance_consistency_threshold=0.9,

        # 约束满足阈值
        constraint_satisfaction_threshold=0.9,
        critical_constraint_threshold=1.0,

        # 视觉质量阈值
        visual_appeal_threshold=0.7,
        style_consistency_threshold=0.8,
        color_consistency_threshold=0.75,

        # 技术质量阈值
        prompt_quality_threshold=0.6,
        technical_feasibility_threshold=0.8,

        # 整体质量阈值
        overall_quality_threshold=0.75,
        approval_threshold=0.8
    ),

    "lenient": QualityThresholds(
        # 连续性阈值
        continuity_score_threshold=0.7,
        position_consistency_threshold=0.6,
        appearance_consistency_threshold=0.8,

        # 约束满足阈值
        constraint_satisfaction_threshold=0.8,
        critical_constraint_threshold=0.9,  # 允许关键约束90%满足

        # 视觉质量阈值
        visual_appeal_threshold=0.6,
        style_consistency_threshold=0.7,
        color_consistency_threshold=0.65,

        # 技术质量阈值
        prompt_quality_threshold=0.5,
        technical_feasibility_threshold=0.7,

        # 整体质量阈值
        overall_quality_threshold=0.65,
        approval_threshold=0.7
    )
}


def get_thresholds_preset(preset_name: str = "balanced") -> QualityThresholds:
    """获取阈值预设"""
    return QUALITY_THRESHOLD_PRESETS.get(preset_name, QUALITY_THRESHOLD_PRESETS["balanced"])
