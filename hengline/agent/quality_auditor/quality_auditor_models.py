"""
@FileName: quality_auditor_models.py
@Description: 质量审核模型
@Author: HengLine
@Time: 2026/1/19 22:58
"""
from datetime import datetime
from typing import List, Optional, Any, Dict, Literal

from pydantic import Field, BaseModel


class BasicViolation(BaseModel):
    """MVP违规记录"""
    rule_id: str = Field(..., description="规则ID")
    rule_name: str = Field(..., description="规则名称")
    description: str = Field(..., description="违规描述")
    severity: Literal["info", "warning", "error"] = Field(
        default="warning",
        description="严重程度"
    )
    fragment_id: Optional[str] = Field(
        default=None,
        description="涉及的片段ID"
    )
    suggestion: Optional[str] = Field(
        default=None,
        description="改进建议"
    )


class QualityAuditReport(BaseModel):
    """MVP质量审查报告"""

    # 元数据
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "audited_at": datetime.now().isoformat(),
            "version": "mvp_1.0",
            "auditor_type": "basic"
        }
    )

    # 项目信息
    project_info: Dict[str, Any] = Field(
        default_factory=lambda: {
            "title": "",
            "fragment_count": 0,
            "total_duration": 0.0
        }
    )

    # 审查状态
    status: Literal["passed", "needs_review", "failed"] = Field(
        default="passed",
        description="整体审查状态"
    )

    # 检查明细
    checks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="执行的检查项目"
    )

    # 违规记录
    violations: List[BasicViolation] = Field(
        default_factory=list,
        description="发现的违规问题"
    )

    # 统计数据
    stats: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total_checks": 0,
            "passed_checks": 0,
            "warnings": 0,
            "errors": 0,
            "fragments_checked": 0
        }
    )

    # 简单建议
    suggestions: List[str] = Field(
        default_factory=lambda: [
            "检查所有片段时长是否≤5秒",
            "确保没有空提示词"
        ]
    )

    # 最终结论
    conclusion: str = Field(
        default="审查通过，可以开始视频生成",
        description="审查结论"
    )
