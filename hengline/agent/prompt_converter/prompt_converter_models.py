"""
@FileName: prompt_converter_models.py
@Description: 模型
@Author: HengLine
@Time: 2026/1/18 14:25
"""
from datetime import datetime
from typing import Dict, Any, List

from pydantic import Field, BaseModel


class PromptConstraint(BaseModel):
    """Prompt中的约束条件"""

    # 视觉约束
    visual_constraints: List[str] = Field(
        default_factory=list,
        description="视觉约束列表，如：角色服装、道具位置等"
    )

    # 连续性约束
    continuity_constraints: List[str] = Field(
        default_factory=list,
        description="连续性约束，确保与前后片段一致"
    )

    # 风格约束
    style_constraints: Dict[str, Any] = Field(
        default_factory=lambda: {
            "overall_style": "cinematic",
            "color_palette": "consistent",
            "lighting_style": "dramatic"
        },
        description="风格约束"
    )

    # 技术约束
    technical_constraints: Dict[str, Any] = Field(
        default_factory=lambda: {
            "motion_blur": "natural",
            "frame_consistency": "high",
            "aspect_ratio": "16:9"
        },
        description="技术约束"
    )


class ModelSpecificParams(BaseModel):
    """模型特定参数"""

    # 通用参数
    model_name: str = Field(default="runway_gen2", description="目标AI模型名称")

    # Sora特定参数
    sora_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "style_preset": "cinematic",
            "motion_consistency": "high"
        },
        description="Sora模型特定参数"
    )

    # Runway特定参数
    runway_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "interpolate_frames": True,
            "motion_brush": True,
            "style_preset": "cinematic"
        },
        description="Runway模型特定参数"
    )

    # Pika特定参数
    pika_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "style_preset": "cinematic",
            "aspect_ratio": "16:9"
        },
        description="Pika模型特定参数"
    )


class AIVideoPrompt(BaseModel):
    """AI视频生成提示词 - 阶段4输出"""

    fragment_id: str = Field(..., description="对应的片段ID")

    # 核心提示词
    prompt_text: str = Field(
        default="",
        description="主提示词文本，包含视觉描述和约束"
    )

    # 负面提示词
    negative_prompt: str = Field(
        default="blurry, distorted, low quality, cartoonish, bad anatomy",
        description="负面提示词，避免的内容"
    )

    # 约束条件
    constraints: PromptConstraint = Field(
        default_factory=PromptConstraint,
        description="约束条件集合"
    )

    # 技术参数
    technical_parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "duration": 4.0,
            "fps": 24,
            "resolution": "1024x576",
            "seed": None,
            "cfg_scale": 7.5,
            "steps": 50
        },
        description="技术参数配置"
    )

    # 模型特定参数
    model_params: ModelSpecificParams = Field(
        default_factory=ModelSpecificParams,
        description="模型特定参数配置"
    )

    # 提示词质量评估
    quality_metrics: Dict[str, Any] = Field(
        default_factory=lambda: {
            "clarity_score": 0.0,
            "specificity_score": 0.0,
            "consistency_score": 0.0,
            "estimated_success_rate": 0.0
        },
        description="提示词质量评估指标"
    )

    # 生成信息
    generation_info: Dict[str, Any] = Field(
        default_factory=lambda: {
            "prompt_version": "v1.0",
            "template_used": "cinematic_standard",
            "llm_optimized": True,
            "optimization_iterations": 1
        },
        description="提示词生成信息"
    )


class AIVideoInstructions(BaseModel):
    """AI视频生成指令集 - 阶段4最终输出"""

    # 项目元数据
    project_info: Dict[str, Any] = Field(
        default_factory=lambda: {
            "project_id": "proj_001",
            "project_name": "AI Video Project",
            "created_at": datetime.now().isoformat(),
            "total_duration": 0.0,
            "target_model": "runway_gen2"
        },
        description="项目信息"
    )

    # 片段指令集
    fragments: List[AIVideoPrompt] = Field(
        default_factory=list,
        description="按时间顺序排列的片段指令"
    )

    # 全局设置
    global_settings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "style_consistency": True,
            "color_consistency": True,
            "character_consistency": True,
            "seed_behavior": "incremental"  # incremental/fixed/random
        },
        description="全局生成设置"
    )

    # 输出配置
    output_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "output_format": "mp4",
            "codec": "h264",
            "bitrate": "5000k",
            "audio_settings": {
                "include_audio": False,
                "sample_rate": 44100
            }
        },
        description="输出文件配置"
    )

    # 执行计划
    execution_plan: Dict[str, Any] = Field(
        default_factory=lambda: {
            "generation_order": "sequential",  # sequential/parallel
            "batch_size": 1,
            "quality_checkpoints": [],  # 质量检查点
            "fallback_strategy": "retry_with_adjusted_params"
        },
        description="执行计划和策略"
    )

    # 资源预估
    resource_estimation: Dict[str, Any] = Field(
        default_factory=lambda: {
            "estimated_tokens": 0,
            "estimated_time": 0.0,
            "estimated_cost": 0.0,
            "storage_required": "0 MB"
        },
        description="资源消耗预估"
    )
