"""
@FileName: fix_models.py
@Description: 修复建议模型
@Author: HengLine
@Time: 2026/1/6 16:10
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, List, Dict


class FixType(Enum):
    AUTO = "auto"  # 可自动修复
    MANUAL = "manual"  # 需要手动修复
    CONFIG = "config"  # 配置调整
    REPLACE = "replace"  # 需要重新生成


@dataclass
class FixBase:
    """修复基类"""

    fix_id: str
    fix_type: FixType
    target_issue_id: str
    description: str

    # 修复内容
    fix_content: Optional[Any] = None
    fix_instructions: List[str] = field(default_factory=list)

    # 效果评估
    expected_effectiveness: float = 0.8  # 0-1
    confidence_score: float = 0.7  # 0-1

    # 复杂度
    complexity: str = "low"  # "low", "medium", "high"
    estimated_time_minutes: int = 5


@dataclass
class AutoFixSuggestion(FixBase):
    """自动修复建议"""

    # 自动修复特性
    can_be_auto_applied: bool = True
    auto_apply_confidence: float = 0.8  # 自动应用的置信度

    # 修复代码
    fix_code: Optional[str] = None
    fix_parameters: Dict[str, Any] = field(default_factory=dict)

    # 验证
    requires_verification: bool = True
    verification_check: Optional[str] = None


@dataclass
class ManualFixRecommendation(FixBase):
    """手动修复建议"""

    # 手动修复特性
    fix_steps: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    skill_level_required: str = "intermediate"  # "beginner", "intermediate", "expert"

    # 示例
    examples: List[Dict[str, Any]] = field(default_factory=list)
    before_after_comparison: Optional[Dict[str, str]] = None

    # 风险
    risks: List[str] = field(default_factory=list)
    risk_mitigation: List[str] = field(default_factory=list)
