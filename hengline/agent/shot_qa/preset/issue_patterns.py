"""
@FileName: issue_patterns.py
@Description: 问题模式配置
@Author: HengLine
@Time: 2026/1/6 16:02
"""

# 常见问题模式
ISSUE_PATTERNS = {
    "character_jump": {
        "pattern": "character position changes abruptly",
        "severity": "high",
        "detection_method": "position_analysis",
        "suggested_fix": "add transitional movement or adjust position gradually"
    },

    "prop_discontinuity": {
        "pattern": "prop appears/disappears without reason",
        "severity": "medium",
        "detection_method": "prop_tracking",
        "suggested_fix": "show prop being picked up/put down, or keep consistent"
    },

    "lighting_inconsistency": {
        "pattern": "lighting changes abruptly between shots",
        "severity": "medium",
        "detection_method": "lighting_analysis",
        "suggested_fix": "maintain consistent lighting direction and intensity"
    },

    "camera_angle_mismatch": {
        "pattern": "camera angles don't match eyeline or perspective",
        "severity": "low",
        "detection_method": "camera_analysis",
        "suggested_fix": "adjust camera angles to maintain consistent perspective"
    },

    "action_timing_issue": {
        "pattern": "actions don't flow naturally in time",
        "severity": "high",
        "detection_method": "temporal_analysis",
        "suggested_fix": "adjust timing or add transitional actions"
    }
}

# 修复模式
FIX_PATTERNS = {
    "add_transition": {
        "description": "添加过渡效果",
        "applicable_issues": ["character_jump", "lighting_inconsistency"],
        "implementation": "use dissolve, fade, or match cut",
        "effectiveness": 0.8
    },

    "adjust_timing": {
        "description": "调整时间",
        "applicable_issues": ["action_timing_issue"],
        "implementation": "extend or shorten action duration",
        "effectiveness": 0.7
    },

    "reposition_element": {
        "description": "重新定位元素",
        "applicable_issues": ["character_jump", "camera_angle_mismatch"],
        "implementation": "adjust position/orientation to match previous shot",
        "effectiveness": 0.9
    },

    "add_explanation": {
        "description": "添加解释性元素",
        "applicable_issues": ["prop_discontinuity"],
        "implementation": "show character picking up or putting down the prop",
        "effectiveness": 0.85
    }
}