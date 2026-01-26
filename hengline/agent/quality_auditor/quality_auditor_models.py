"""
@FileName: quality_auditor_models.py
@Description: 质量审核模型
@Author: HengLine
@Time: 2026/1/19 22:58
"""
from datetime import datetime
from typing import List, Optional, Any, Dict, Literal

from pydantic import Field, BaseModel


class RuleViolation(BaseModel):
    """规则违反记录"""

    rule_id: str = Field(..., description="规则ID")
    rule_name: str = Field(..., description="规则名称")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        default="medium",
        description="严重程度"
    )

    # 违规详情
    fragment_id: Optional[str] = Field(
        default=None,
        description="涉及片段ID（如有）"
    )
    description: str = Field(..., description="违规描述")
    expected_value: Optional[Any] = Field(
        default=None,
        description="期望值"
    )
    actual_value: Optional[Any] = Field(
        default=None,
        description="实际值"
    )

    # 修复建议
    fix_suggestion: Optional[str] = Field(
        default=None,
        description="修复建议"
    )
    auto_fixable: bool = Field(
        default=False,
        description="是否可自动修复"
    )


class LLMCoherenceAssessment(BaseModel):
    """LLM连贯性评估"""

    # 评估维度
    visual_coherence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="视觉连贯性评分"
    )
    narrative_flow: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="叙事流畅性评分"
    )
    emotional_consistency: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="情绪一致性评分"
    )

    # 问题检测
    detected_issues: List[str] = Field(
        default_factory=list,
        description="检测到的问题列表"
    )

    # 整体评估
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="整体评分（0-10）"
    )
    assessment_summary: str = Field(
        default="",
        description="评估总结"
    )

    # 置信度
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="评估置信度"
    )


class ContinuityAudit(BaseModel):
    """连续性审计结果"""

    # 连续性检查
    character_continuity: Dict[str, Any] = Field(
        default_factory=dict,
        description="角色连续性检查结果"
    )
    prop_continuity: Dict[str, Any] = Field(
        default_factory=dict,
        description="道具连续性检查结果"
    )
    scene_continuity: Dict[str, Any] = Field(
        default_factory=dict,
        description="场景连续性检查结果"
    )

    # 时间连续性
    temporal_continuity: Dict[str, Any] = Field(
        default_factory=dict,
        description="时间连续性检查结果"
    )

    # 问题汇总
    continuity_issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="连续性问题列表"
    )

    # 连续性评分
    continuity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="整体连续性评分"
    )


class QualityAuditReport(BaseModel):
    """质量审查报告 - 阶段5输出"""

    # 报告元数据
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "audit_id": "audit_001",
            "audited_at": datetime.now().isoformat(),
            "audit_version": "v1.0",
            "audited_project": "proj_001"
        },
        description="审计报告元数据"
    )

    # 总体状态
    overall_status: Literal["passed", "needs_revision", "failed"] = Field(
        default="passed",
        description="总体审查状态"
    )

    # 检查结果
    rule_check_results: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total_rules_checked": 0,
            "passed_rules": 0,
            "violations": [],
            "warnings": []
        },
        description="规则检查结果"
    )

    # LLM评估
    llm_assessment: LLMCoherenceAssessment = Field(
        default_factory=LLMCoherenceAssessment,
        description="LLM连贯性评估结果"
    )

    # 连续性审计
    continuity_audit: ContinuityAudit = Field(
        default_factory=ContinuityAudit,
        description="连续性审计结果"
    )

    # 质量指标
    quality_metrics: Dict[str, Any] = Field(
        default_factory=lambda: {
            "overall_quality_score": 0.0,
            "temporal_quality": 0.0,
            "visual_quality": 0.0,
            "narrative_quality": 0.0,
            "technical_quality": 0.0
        },
        description="质量指标评分"
    )

    # 问题汇总
    issue_summary: Dict[str, Any] = Field(
        default_factory=lambda: {
            "critical_issues": [],
            "major_issues": [],
            "minor_issues": [],
            "suggestions": []
        },
        description="问题汇总"
    )

    # 修复建议
    repair_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="修复建议列表"
    )

    # 自动化修复
    auto_fixes_applied: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="已应用的自动修复"
    )

    # 审查结论
    conclusion: Dict[str, Any] = Field(
        default_factory=lambda: {
            "summary": "审查通过",
            "next_steps": ["可以开始视频生成"],
            "risk_level": "low"
        },
        description="审查结论和建议"
    )
