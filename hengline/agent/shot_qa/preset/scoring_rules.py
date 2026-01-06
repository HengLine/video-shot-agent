"""
@FileName: scoring_rules.py
@Description: 评分规则配置
@Author: HengLine
@Time: 2026/1/6 16:02
"""

# 连续性评分规则
CONTINUITY_SCORING_RULES = {
    "position_consistency": {
        "weight": 0.3,
        "scoring_method": "distance_based",
        "perfect_score_threshold": 0.1,  # 位置差异小于0.1得满分
        "zero_score_threshold": 1.0,  # 位置差异大于1.0得0分
        "penalty_per_unit": 0.2  # 每单位差异扣0.2分
    },

    "appearance_consistency": {
        "weight": 0.3,
        "scoring_method": "binary_with_tolerance",
        "tolerance": 1,  # 允许1个外观元素变化
        "perfect_score": 1.0,
        "penalty_per_violation": 0.3
    },

    "temporal_consistency": {
        "weight": 0.2,
        "scoring_method": "time_gap_based",
        "perfect_gap": 0.0,
        "acceptable_gap": 0.5,
        "unacceptable_gap": 2.0
    },

    "action_continuity": {
        "weight": 0.2,
        "scoring_method": "sequence_based",
        "perfect_sequence": 1.0,
        "broken_sequence": 0.0
    }
}

# 约束满足评分规则
CONSTRAINT_SCORING_RULES = {
    "critical_constraints": {
        "weight": 0.4,
        "scoring_method": "binary",  # 必须100%满足
        "perfect_score": 1.0,
        "failure_score": 0.0
    },

    "high_priority_constraints": {
        "weight": 0.3,
        "scoring_method": "percentage",
        "tolerance": 0.1,  # 允许10%偏差
        "perfect_score": 1.0
    },

    "medium_priority_constraints": {
        "weight": 0.2,
        "scoring_method": "percentage",
        "tolerance": 0.2,  # 允许20%偏差
        "perfect_score": 1.0
    },

    "low_priority_constraints": {
        "weight": 0.1,
        "scoring_method": "percentage",
        "tolerance": 0.3,  # 允许30%偏差
        "perfect_score": 1.0
    }
}

# 视觉质量评分规则
VISUAL_QUALITY_SCORING_RULES = {
    "composition": {
        "weight": 0.3,
        "factors": ["rule_of_thirds", "balance", "framing", "depth"],
        "points_per_factor": 0.25,
        "penalty_per_issue": 0.1
    },

    "lighting": {
        "weight": 0.25,
        "factors": ["consistency", "direction", "quality", "mood"],
        "points_per_factor": 0.25,
        "penalty_per_issue": 0.1
    },

    "color": {
        "weight": 0.25,
        "factors": ["harmony", "consistency", "palette", "grading"],
        "points_per_factor": 0.25,
        "penalty_per_issue": 0.1
    },

    "style": {
        "weight": 0.2,
        "factors": ["consistency", "appropriateness", "execution"],
        "points_per_factor": 0.33,
        "penalty_per_issue": 0.15
    }
}

# 技术质量评分规则
TECHNICAL_SCORING_RULES = {
    "prompt_quality": {
        "weight": 0.4,
        "factors": ["clarity", "completeness", "specificity", "length"],
        "points_per_factor": 0.25,
        "penalty_per_issue": 0.05
    },

    "camera_parameters": {
        "weight": 0.3,
        "factors": ["validity", "consistency", "appropriateness", "feasibility"],
        "points_per_factor": 0.25,
        "penalty_per_issue": 0.1
    },

    "generation_feasibility": {
        "weight": 0.3,
        "factors": ["physical_possibility", "complexity", "ai_capability"],
        "points_per_factor": 0.33,
        "penalty_per_issue": 0.15
    }
}

# 整体评分权重
OVERALL_SCORING_WEIGHTS = {
    "continuity": 0.25,
    "constraints": 0.30,
    "visual_quality": 0.25,
    "technical_quality": 0.20
}


def calculate_weighted_score(scores: dict, weights: dict) -> float:
    """计算加权得分"""
    total_score = 0.0
    total_weight = 0.0

    for key, score in scores.items():
        if key in weights:
            weight = weights[key]
            total_score += score * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return total_score / total_weight
