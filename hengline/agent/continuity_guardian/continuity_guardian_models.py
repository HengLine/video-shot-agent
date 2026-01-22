"""
@FileName: continuity_guardian_models.py
@Description: 连续性模型
@Author: HengLine
@Time: 2026/1/18 14:26
"""
from datetime import datetime
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field

from hengline.agent.base_models import BaseMetadata, RiskLevel, DifficultyLevel


class ContinuityIssue(BaseModel):
    """连续性问题"""
    check_id: str = Field(..., description="检查ID")
    type: str = Field(..., description="问题类型")
    shots_involved: List[str] = Field(..., min_length=1, description="涉及镜头ID")
    description: str = Field(..., description="问题描述")
    status: str = Field(default="detected", description="状态：detected/resolved/ignored")
    severity: RiskLevel = Field(..., description="严重程度")
    suggestion: str = Field(..., description="解决建议")
    auto_fixable: bool = Field(default=False, description="是否可自动修复")
    fix_action: Optional[str] = Field(None, description="修复动作")
    priority: int = Field(default=5, ge=1, le=10, description="优先级（1-10）")


class OptimizationSuggestion(BaseModel):
    """优化建议"""
    suggestion_id: str = Field(..., description="建议ID")
    type: str = Field(..., description="建议类型：拆分/情感/节奏等")
    target_shot: str = Field(..., description="目标镜头ID")
    current_state: Dict[str, Any] = Field(..., description="当前状态")
    issue: str = Field(..., description="存在的问题")
    recommendation: str = Field(..., description="推荐方案")
    expected_improvement: Dict[str, Any] = Field(..., description="预期改善")
    implementation_difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM, description="实现难度")


class ShotAdjustment(BaseModel):
    """镜头调整"""
    shot_id: str = Field(..., description="镜头ID")
    adjustment_type: str = Field(..., description="调整类型：duration/split/pose等")
    adjustment: str = Field(..., description="调整内容")
    original_value: Optional[Any] = Field(None, description="原始值")
    new_value: Optional[Any] = Field(None, description="新值")
    reason: str = Field(..., description="调整原因")
    impact_assessment: Optional[Dict[str, Any]] = Field(None, description="影响评估")


class ContinuityPreCheckModel(BaseMetadata):
    """
    连续性预审模型 - 第三阶段输出
    对镜头拆分进行宏观连续性检查
    """
    checked_model: str = Field(..., description="被检查的模型ID")

    # 总体评估
    overall_assessment: Dict[str, Any] = Field(
        ...,
        description="总体评估，包含评分、风险等级、建议等"
    )

    # 详细检查结果
    temporal_checks: List[ContinuityIssue] = Field(
        default_factory=list,
        description="时间连续性检查"
    )
    spatial_checks: List[ContinuityIssue] = Field(
        default_factory=list,
        description="空间连续性检查"
    )
    character_checks: List[ContinuityIssue] = Field(
        default_factory=list,
        description="角色连续性检查"
    )
    visual_checks: List[ContinuityIssue] = Field(
        default_factory=list,
        description="视觉连续性检查"
    )

    # 优化建议
    optimization_suggestions: List[OptimizationSuggestion] = Field(
        default_factory=list,
        description="优化建议列表"
    )

    # 审批与调整
    approved_with_changes: bool = Field(default=False, description="是否批准（需调整）")
    approved_as_is: bool = Field(default=False, description="是否原样批准")
    requires_manual_review: bool = Field(default=False, description="是否需要人工审核")

    # 需要执行的调整
    required_adjustments: List[ShotAdjustment] = Field(
        default_factory=list,
        description="需要执行的调整"
    )

    # 风险评估（新增）
    risk_assessment: Dict[str, Any] = Field(
        default_factory=dict,
        description="风险评估详情"
    )

    # 元数据
    check_timestamp: datetime = Field(default_factory=datetime.now, description="检查时间")
    checker_version: str = Field(default="1.0", description="检查器版本")

    class Config:
        schema_extra = {
            "example": {
                "id": "PRECHECK_001",
                "project_id": "PROJ_001",
                "checked_model": "BREAKDOWN_001",
                "overall_assessment": {
                    "continuity_score": 0.85,
                    "risk_level": "medium",
                    "recommendation": "需要调整拆分点"
                },
                "approved_with_changes": True,
                "required_adjustments": []
            }
        }
