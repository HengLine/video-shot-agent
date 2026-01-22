"""
@FileName: quality_auditor_models.py
@Description: 质量审核模型
@Author: HengLine
@Time: 2026/1/19 22:58
"""
from datetime import datetime
from typing import List, Optional, Any, Dict

from pydantic import Field, BaseModel

from hengline.agent.base_models import AIPlatform, GenerationStatus, BaseMetadata


class QualityMetric(BaseModel):
    """质量指标"""
    metric_name: str = Field(..., description="指标名称")
    value: float = Field(..., ge=0.0, le=1.0, description="指标值")
    threshold: float = Field(..., ge=0.0, le=1.0, description="阈值")
    weight: float = Field(default=1.0, ge=0.0, description="权重")
    description: Optional[str] = Field(None, description="指标描述")


class TechnicalQualityCheck(BaseModel):
    """技术质量检查"""
    video_stability_score: Optional[float] = Field(None, description="视频稳定性评分")
    artifact_detection_score: Optional[float] = Field(None, description="伪影检测评分")
    color_consistency_score: Optional[float] = Field(None, description="色彩一致性评分")
    resolution_check: Optional[bool] = Field(None, description="分辨率检查")
    frame_rate_check: Optional[bool] = Field(None, description="帧率检查")
    issues_detected: List[str] = Field(default_factory=list, description="检测到的问题")
    passed: Optional[bool] = Field(None, description="是否通过")


class ContentComplianceCheck(BaseModel):
    """内容符合度检查"""
    action_match_score: float = Field(..., ge=0.0, le=1.0, description="动作匹配度")
    character_match_score: float = Field(..., ge=0.0, le=1.0, description="角色匹配度")
    environment_match_score: float = Field(..., ge=0.0, le=1.0, description="环境匹配度")
    emotion_match_score: float = Field(..., ge=0.0, le=1.0, description="情感匹配度")
    overall_compliance: float = Field(..., ge=0.0, le=1.0, description="总体符合度")
    mismatch_details: List[str] = Field(default_factory=list, description="不匹配详情")
    passed_threshold: bool = Field(..., description="是否通过阈值")


class AestheticQualityCheck(BaseModel):
    """审美质量检查"""
    pace_consistency_score: Optional[float] = Field(None, description="节奏一致性评分")
    emotional_arc_score: Optional[float] = Field(None, description="情感弧线评分")
    visual_rhythm_score: Optional[float] = Field(None, description="视觉节奏评分")
    composition_score: Optional[float] = Field(None, description="构图评分")
    lighting_aesthetic_score: Optional[float] = Field(None, description="灯光审美评分")
    subjective_rating: Optional[float] = Field(None, description="主观评分")
    reviewer_notes: Optional[str] = Field(None, description="评审备注")


class GenerationResult(BaseModel):
    """生成结果"""
    fragment_id: str = Field(..., description="片段ID")
    generation_id: str = Field(..., description="生成ID")
    platform: AIPlatform = Field(..., description="生成平台")

    # 状态信息
    status: GenerationStatus = Field(..., description="生成状态")
    start_time: datetime = Field(..., description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    duration_seconds: Optional[float] = Field(None, description="生成时长")

    # 输出信息
    output_path: Optional[str] = Field(None, description="输出路径")
    thumbnail_path: Optional[str] = Field(None, description="缩略图路径")
    metadata_path: Optional[str] = Field(None, description="元数据路径")

    # 质量检查
    technical_quality: Optional[TechnicalQualityCheck] = Field(None, description="技术质量")
    content_compliance: Optional[ContentComplianceCheck] = Field(None, description="内容符合度")
    aesthetic_quality: Optional[AestheticQualityCheck] = Field(None, description="审美质量")

    # 参数信息
    parameters_used: Dict[str, Any] = Field(default_factory=dict, description="使用参数")
    prompt_used: Optional[str] = Field(None, description="使用的提示词")
    seed_used: Optional[int] = Field(None, description="使用的种子")

    # 错误信息
    error_message: Optional[str] = Field(None, description="错误信息")
    error_type: Optional[str] = Field(None, description="错误类型")
    retry_count: int = Field(default=0, description="重试次数")


class RegenerationDecision(BaseModel):
    """重生成决策"""
    fragment_id: str = Field(..., description="片段ID")
    decision_id: str = Field(..., description="决策ID")

    # 决策信息
    decision: str = Field(..., description="决策：regenerate/accept/adjust")
    reason: str = Field(..., description="决策原因")
    priority: str = Field(default="medium", description="优先级")

    # 重生成参数
    new_parameters: Dict[str, Any] = Field(default_factory=dict, description="新参数")
    prompt_adjustments: Optional[List[str]] = Field(None, description="提示词调整")
    platform_switch: Optional[AIPlatform] = Field(None, description="平台切换")

    # 质量要求
    quality_requirements: Dict[str, float] = Field(
        default_factory=dict,
        description="质量要求"
    )

    # 执行状态
    executed: bool = Field(default=False, description="是否已执行")
    execution_result: Optional[str] = Field(None, description="执行结果")


class QualityReportModel(BaseMetadata):
    """
    质量报告模型 - 第六阶段输出
    对生成结果进行质量评估并提供重生成决策
    """
    generation_source: str = Field(..., description="来源生成ID或提示词ID")

    # 生成结果
    generation_results: List[GenerationResult] = Field(
        ...,
        min_length=1,
        description="生成结果列表"
    )

    # 质量总结
    overall_quality_score: float = Field(..., ge=0.0, le=1.0, description="总体质量评分")
    technical_quality_score: Optional[float] = Field(None, description="技术质量评分")
    content_quality_score: Optional[float] = Field(None, description="内容质量评分")
    aesthetic_quality_score: Optional[float] = Field(None, description="审美质量评分")

    # 问题分析
    critical_issues: List[str] = Field(default_factory=list, description="严重问题")
    moderate_issues: List[str] = Field(default_factory=list, description="中等问题")
    minor_issues: List[str] = Field(default_factory=list, description="轻微问题")

    # 连续性评估
    continuity_assessment: Dict[str, Any] = Field(
        default_factory=dict,
        description="连续性评估"
    )

    # 重生成决策
    regeneration_decisions: List[RegenerationDecision] = Field(
        default_factory=list,
        description="重生成决策"
    )

    # 推荐操作
    recommended_actions: List[str] = Field(default_factory=list, description="推荐操作")

    # 通过状态
    passed_technical: bool = Field(default=False, description="通过技术检查")
    passed_content: bool = Field(default=False, description="通过内容检查")
    passed_aesthetic: bool = Field(default=False, description="通过审美检查")
    overall_passed: bool = Field(default=False, description="总体通过")

    class Config:
        schema_extra = {
            "example": {
                "id": "QUALITY_001",
                "project_id": "PROJ_001",
                "generation_source": "PROMPTS_001",
                "overall_quality_score": 0.78,
                "generation_results": [],
                "critical_issues": ["角色不一致", "运动不自然"],
                "regeneration_decisions": [],
                "overall_passed": False
            }
        }
