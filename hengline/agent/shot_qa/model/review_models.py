"""
@FileName: review_models.py
@Description: 审查相关数据模型
@Author: HengLine
@Time: 2026/1/6 15:59
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any

from .check_models import ContinuityCheckResult, ConstraintCheckResult, VisualQualityCheckResult, TechnicalCheckResult
from .fix_models import ManualFixRecommendation, AutoFixSuggestion
from .issue_models import CriticalIssue, IssueWarning, IssueSuggestion
from .score_models import QualityScores
from ...continuity_guardian.model.continuity_guardian_report import AnchoredTimeline
from ...shot_generator.model.shot_models import SoraReadyShots


class ReviewDecision(Enum):
    APPROVED = "approved"  # 通过，可直接使用
    APPROVED_WITH_ISSUES = "approved_with_issues"  # 通过但有问题
    NEEDS_REVISION = "needs_revision"  # 需要修订
    REJECTED = "rejected"  # 拒绝，需要重新生成


@dataclass
class ReviewConfig:
    """审查配置"""

    # 检查项配置
    enable_continuity_checks: bool = True
    enable_constraint_checks: bool = True
    enable_visual_quality_checks: bool = True
    enable_technical_checks: bool = True

    # 严格度配置
    strictness_level: str = "balanced"  # "strict", "balanced", "lenient"

    # 阈值配置
    auto_fix_threshold: float = 0.7  # 自动修复建议阈值
    rejection_threshold: float = 0.3  # 拒绝阈值

    # 报告配置
    generate_detailed_report: bool = True
    include_suggestions: bool = True
    include_auto_fixes: bool = True


@dataclass
class QualityThresholds:
    """质量阈值配置"""

    # 连续性阈值
    continuity_score_threshold: float = 0.8
    position_consistency_threshold: float = 0.7
    appearance_consistency_threshold: float = 0.9

    # 约束满足阈值
    constraint_satisfaction_threshold: float = 0.9
    critical_constraint_threshold: float = 1.0  # 关键约束必须100%满足

    # 视觉质量阈值
    visual_appeal_threshold: float = 0.7
    style_consistency_threshold: float = 0.8
    color_consistency_threshold: float = 0.75

    # 技术质量阈值
    prompt_quality_threshold: float = 0.6
    technical_feasibility_threshold: float = 0.8

    # 整体质量阈值
    overall_quality_threshold: float = 0.75
    approval_threshold: float = 0.8


@dataclass
class QualityReviewInput:
    """质量审查输入"""

    # 来自智能体4的生成结果
    sora_shots: SoraReadyShots

    # 来自智能体3的原始约束（用于交叉验证）
    anchored_timeline: AnchoredTimeline

    # 审查配置
    review_config: ReviewConfig = field(default_factory=ReviewConfig)

    # 质量阈值
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)


# ==================== 最终决策模型 ====================

@dataclass
class OverallAssessment:
    """总体评估"""

    decision: ReviewDecision
    decision_reason: str

    # 质量总结
    quality_summary: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)

    # 风险评估
    risk_level: str = "low"  # "low", "medium", "high"
    risk_factors: List[str] = field(default_factory=list)

    # 使用建议
    usage_recommendation: Optional[str] = None
    limitations: List[str] = field(default_factory=list)


@dataclass
class FinalDecision:
    """最终决策"""

    decision: ReviewDecision
    decision_date: datetime = field(default_factory=datetime.now)

    # 决策依据
    primary_reasons: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)

    # 质量指标
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0

    # 阈值检查
    meets_quality_thresholds: bool = True
    meets_continuity_thresholds: bool = True
    meets_constraint_thresholds: bool = True

    # 下一步行动
    next_action: str = "proceed"  # "proceed", "revise", "regenerate"
    action_deadline: Optional[datetime] = None


@dataclass
class NextStep:
    """下一步行动"""

    step_id: str
    step_type: str  # "fix", "review", "approve", "generate"

    # 行动详情
    description: str
    action_items: List[str] = field(default_factory=list)

    # 责任分配
    responsible_party: str = "user"  # "user", "system", "both"
    estimated_effort: str = "low"  # "low", "medium", "high"

    # 依赖关系
    dependencies: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    # 状态
    is_required: bool = True
    is_completed: bool = False
    completion_date: Optional[datetime] = None


# ==================== 报告元数据 ====================

@dataclass
class ReportMetadata:
    """报告元数据"""

    report_id: str
    report_version: str = "1.0.0"

    # 生成信息
    generated_at: datetime = field(default_factory=datetime.now)
    generation_duration_seconds: float = 0.0

    # 审查范围
    reviewed_shots_count: int = 0
    reviewed_constraints_count: int = 0
    performed_checks_count: int = 0

    # 系统信息
    reviewer_version: str = "1.0.0"
    configuration_used: str = "default"

    # 质量保证
    review_confidence: float = 0.8  # 0-1
    verification_status: str = "pending"  # "pending", "verified", "unverified"

    metadata: Dict[str, Any] = field(default_factory=dict)

    # 跟踪信息
    parent_review_id: Optional[str] = None
    related_reviews: List[str] = field(default_factory=list)


# ==================== 输出模型 ====================

@dataclass
class QualityReviewOutput:
    """质量审查输出"""

    # 核心评估结果
    overall_assessment: OverallAssessment
    final_decision: FinalDecision

    # 详细检查结果
    continuity_checks: List[ContinuityCheckResult]
    constraint_checks: List[ConstraintCheckResult]
    visual_quality_checks: List[VisualQualityCheckResult]
    technical_checks: List[TechnicalCheckResult]

    # 问题报告
    critical_issues: List[CriticalIssue]
    warnings: List[IssueWarning]
    suggestions: List[IssueSuggestion]

    # 修复建议
    auto_fix_suggestions: List[AutoFixSuggestion]
    manual_fix_recommendations: List[ManualFixRecommendation]

    # 质量评分
    quality_scores: QualityScores

    # 下一步行动
    next_steps: List[NextStep]

    # 报告元数据
    report_metadata: ReportMetadata

    # 原始数据引用
    input_references: Dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        """转换为摘要字典"""
        return {
            "decision": self.final_decision.decision.value,
            "overall_score": self.quality_scores.overall_quality_score,
            "critical_issues": len(self.critical_issues),
            "warnings": len(self.warnings),
            "suggestions": len(self.suggestions),
            "passed_checks": self.final_decision.passed_checks,
            "failed_checks": self.final_decision.failed_checks,
            "next_action": self.final_decision.next_action,
            "generated_at": self.report_metadata.generated_at.isoformat()
        }

    def get_approval_status(self) -> Dict[str, Any]:
        """获取批准状态"""
        return {
            "is_approved": self.final_decision.decision in [ReviewDecision.APPROVED, ReviewDecision.APPROVED_WITH_ISSUES],
            "requires_revision": self.final_decision.decision == ReviewDecision.NEEDS_REVISION,
            "requires_regeneration": self.final_decision.decision == ReviewDecision.REJECTED,
            "can_proceed_with_issues": len(self.critical_issues) == 0 and len(self.warnings) > 0,
            "blocking_issues": [issue.title for issue in self.critical_issues]
        }
