"""
@FileName: video_assembler_models.py
@Description: 视频组装合成模型
@Author: HengLine
@Time: 2026/1/19 23:02
"""
from datetime import datetime
from typing import Optional, List, Any, Dict

from pydantic import Field, BaseModel


class FragmentContinuity(BaseModel):
    """片段连续性约束"""

    # 必须保持的元素
    mandatory_elements: List[str] = Field(
        default_factory=list,
        description="必须出现在画面中的元素：角色、道具等"
    )

    # 状态约束
    character_constraints: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="角色状态约束，key为角色名"
    )

    prop_constraints: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="道具状态约束，key为道具名"
    )

    scene_constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="场景状态约束"
    )

    # 视觉一致性
    visual_consistency: Dict[str, Any] = Field(
        default_factory=lambda: {
            "lighting_consistency": "high",
            "color_palette": "consistent",
            "style_continuity": "required"
        },
        description="视觉一致性要求"
    )


class FragmentTransition(BaseModel):
    """片段间转场信息"""
    from_fragment_id: Optional[str] = Field(
        default=None,
        description="前一个片段ID"
    )
    to_fragment_id: Optional[str] = Field(
        default=None,
        description="后一个片段ID"
    )

    transition_type: str = Field(
        default="cut",
        description="转场类型：cut/fade/dissolve/wipe/match"
    )
    duration: float = Field(
        default=0.0,
        description="转场持续时间（秒）"
    )

    # 连续性要求
    continuity_requirements: Dict[str, Any] = Field(
        default_factory=lambda: {
            "position_continuity": "required",
            "action_continuity": "required",
            "timing_continuity": "required"
        },
        description="转场连续性要求"
    )


class VideoFragment(BaseModel):
    """视频片段 - 阶段3输出"""

    id: str = Field(..., description="片段唯一ID，格式：frag_001_001")

    # 引用信息
    shot_id: str = Field(..., description="所属镜头ID")
    element_ids: List[str] = Field(
        default_factory=list,
        description="包含的剧本元素ID列表"
    )

    # 时间信息
    start_time: float = Field(default=0.0, description="全局开始时间")
    end_time: float = Field(default=0.0, description="全局结束时间")
    duration: float = Field(
        default=0.0,
        ge=0.5,
        le=5.2,
        description="片段时长，限制在0.5-5.2秒之间"
    )

    # 内容信息
    content_description: str = Field(
        default="",
        description="片段内容描述（用于Prompt生成）"
    )
    key_visual: Optional[str] = Field(
        default=None,
        description="关键视觉元素描述"
    )

    # 连续性信息
    continuity: FragmentContinuity = Field(
        default_factory=FragmentContinuity,
        description="连续性约束"
    )

    # 转场信息
    transition: FragmentTransition = Field(
        default_factory=FragmentTransition,
        description="转场设置"
    )

    # 技术参数
    technical_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "min_duration": 2.0,
            "max_duration": 5.0,
            "preferred_duration": 4.0
        },
        description="技术参数要求"
    )

    # 质量标记
    quality_flags: Dict[str, Any] = Field(
        default_factory=lambda: {
            "is_optimal_length": False,
            "has_continuity_issues": False,
            "needs_special_handling": False
        },
        description="质量标记"
    )


class FragmentSequence(BaseModel):
    """片段序列 - 阶段3输出"""

    # 元数据
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "generated_at": datetime.now().isoformat(),
            "split_strategy": "adaptive_5s",
            "max_duration": 5.0
        },
        description="片段序列元数据"
    )

    # 核心数据
    fragments: List[VideoFragment] = Field(
        default_factory=list,
        description="视频片段列表，按时间顺序排列"
    )

    # 引用信息
    source_shots: Dict[str, Any] = Field(
        default_factory=dict,
        description="源镜头序列的引用信息"
    )

    # 分段时间线
    timeline: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="分段时间线，用于可视化"
    )

    # 统计数据
    stats: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total_fragments": 0,
            "total_duration": 0.0,
            "avg_fragment_duration": 0.0,
            "optimal_fragments": 0,  # 长度在4-5秒的片段数
            "short_fragments": 0,  # 长度<2秒的片段数
            "merged_fragments": 0  # 合并产生的片段数
        },
        description="片段序列统计数据"
    )

    # 连续性分析
    continuity_analysis: Dict[str, Any] = Field(
        default_factory=lambda: {
            "continuity_score": 0.0,
            "break_points": [],  # 连续性断点
            "critical_transitions": []  # 关键转场
        },
        description="连续性分析结果"
    )
