"""
@FileName: prompt_converter_models.py
@Description: 模型
@Author: HengLine
@Time: 2026/1/18 14:25
"""
from typing import Dict, Any, List, Optional

from pydantic import Field, BaseModel

from hengline.agent.base_models import AIPlatform, BaseMetadata


class LayeredPrompt(BaseModel):
    """分层提示词"""
    layer_1_core_action: str = Field(..., description="核心动作层")
    layer_2_character_details: str = Field(..., description="角色细节层")
    layer_3_environment: str = Field(..., description="环境层")
    layer_4_cinematic_style: str = Field(..., description="电影风格层")
    layer_5_technical_specs: str = Field(..., description="技术规格层")
    layer_6_continuity_notes: str = Field(..., description="连续性备注层")
    combined_prompt: Optional[str] = Field(None, description="组合后的完整提示词")


class MotionDescription(BaseModel):
    """运动描述"""
    traditional: str = Field(..., description="传统描述")
    optimized: str = Field(..., description="AI优化描述")
    physics_based: Optional[str] = Field(None, description="基于物理的描述")
    motion_keywords: List[str] = Field(default_factory=list, description="运动关键词")
    avoid_keywords: List[str] = Field(default_factory=list, description="避免的关键词")


class NegativePromptEnhanced(BaseModel):
    """增强的负面提示词"""
    general_issues: str = Field(..., description="通用问题")
    motion_specific: str = Field(..., description="运动特定问题")
    consistency_issues: str = Field(..., description="一致性问题")
    physics_issues: str = Field(..., description="物理问题")
    platform_specific: Optional[str] = Field(None, description="平台特定问题")


class PlatformSpecificConfig(BaseModel):
    """平台特定配置"""
    platform: AIPlatform = Field(..., description="AI平台")
    motion_brush_instructions: Optional[str] = Field(None, description="运动笔刷指令")
    camera_motion_prompt: Optional[str] = Field(None, description="摄像机运动提示")
    seed_strategy: Optional[str] = Field(None, description="种子策略")
    character_reference: Optional[str] = Field(None, description="角色引用方式")
    motion_weight: Optional[float] = Field(None, description="运动权重")
    style_weight: Optional[float] = Field(None, description="风格权重")
    init_image_required: Optional[bool] = Field(None, description="是否需要初始图像")


class GenerationParameters(BaseModel):
    """生成参数"""
    motion_intensity: float = Field(default=5.0, ge=0.0, le=10.0, description="运动强度")
    style_fidelity: float = Field(default=0.8, ge=0.0, le=1.0, description="风格保真度")
    consistency_strength: float = Field(default=0.9, ge=0.0, le=1.0, description="一致性强度")
    temporal_coherence: float = Field(default=0.85, ge=0.0, le=1.0, description="时间连贯性")
    guidance_scale: Optional[float] = Field(None, description="引导尺度")
    seed: Optional[int] = Field(None, description="随机种子")
    steps: Optional[int] = Field(None, description="生成步数")


class RegenerationStrategy(BaseModel):
    """重生成策略"""
    trigger_condition: str = Field(..., description="触发条件")
    action: str = Field(..., description="执行动作")
    parameters: Dict[str, Any] = Field(..., description="参数")
    priority: str = Field(default="medium", description="优先级")
    fallback_action: Optional[str] = Field(None, description="后备动作")


class PromptItem(BaseModel):
    """提示词项"""
    fragment_id: str = Field(..., description="对应片段ID")

    # 分层提示词
    layered_prompt_structure: LayeredPrompt = Field(..., description="分层提示词结构")

    # 运动描述
    motion_description_optimized: MotionDescription = Field(..., description="优化运动描述")

    # 负面提示词
    negative_prompt_enhanced: NegativePromptEnhanced = Field(..., description="增强负面提示")

    # 平台配置
    platform_config: Dict[AIPlatform, PlatformSpecificConfig] = Field(
        default_factory=dict,
        description="各平台配置"
    )

    # 生成参数
    generation_parameters_tuned: Dict[AIPlatform, GenerationParameters] = Field(
        default_factory=dict,
        description="各平台调优参数"
    )

    # 连续性指令
    continuity_instructions: List[str] = Field(
        default_factory=list,
        description="连续性指令列表"
    )

    # 重生成策略
    regeneration_strategies: List[RegenerationStrategy] = Field(
        default_factory=list,
        description="重生成策略"
    )

    # 质量目标
    quality_targets: Dict[str, float] = Field(
        default_factory=dict,
        description="质量目标评分"
    )

    # 引用信息
    references: Optional[Dict[str, str]] = Field(
        None,
        description="引用信息（图像、视频等）"
    )


class ConsistencyStrategy(BaseModel):
    """一致性策略"""
    character_consistency: str = Field(..., description="角色一致性策略")
    environment_consistency: str = Field(..., description="环境一致性策略")
    lighting_consistency: str = Field(..., description="光照一致性策略")
    motion_consistency: str = Field(..., description="运动一致性策略")
    style_transfer_method: Optional[str] = Field(None, description="风格迁移方法")


class GenerationSequence(BaseModel):
    """生成序列"""
    order: List[str] = Field(..., description="生成顺序")
    batch_groups: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="批次分组"
    )
    dependencies: Optional[Dict[str, List[str]]] = Field(
        None,
        description="依赖关系"
    )
    priority_order: Optional[List[str]] = Field(None, description="优先级顺序")


class AIPromptModel(BaseMetadata):
    """
    AI提示词模型 - 第五阶段输出
    为每个片段生成优化的AI提示词和参数
    """
    fragments_source: str = Field(..., description="来源片段模型ID")
    continuity_source: Optional[str] = Field(None, description="连续性详细指令ID")

    # 目标平台
    primary_platform: AIPlatform = Field(..., description="主要目标平台")
    secondary_platforms: List[AIPlatform] = Field(default_factory=list, description="次要平台")

    # 平台通用配置
    platform_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="平台通用配置"
    )

    # 提示词数据
    fragment_prompts: List[PromptItem] = Field(
        ...,
        min_length=1,
        description="片段提示词列表"
    )

    # 一致性策略
    consistency_strategy: ConsistencyStrategy = Field(
        ...,
        description="一致性策略"
    )

    # 生成序列规划
    generation_sequence: GenerationSequence = Field(
        ...,
        description="生成序列规划"
    )

    # 质量检查点
    quality_checkpoints: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="质量检查点"
    )

    # 资源引用
    resource_references: Optional[Dict[str, str]] = Field(
        None,
        description="资源引用（角色表、环境参考等）"
    )

    # 元数据
    prompt_engine_version: str = Field(default="1.0", description="提示词引擎版本")
    optimization_level: str = Field(default="standard", description="优化级别")

    class Config:
        schema_extra = {
            "example": {
                "id": "PROMPTS_001",
                "project_id": "PROJ_001",
                "fragments_source": "FRAGMENTS_001",
                "primary_platform": "runway_gen2",
                "secondary_platforms": ["pika_labs"],
                "fragment_prompts": [],
                "consistency_strategy": {
                    "character_consistency": "use_reference_images",
                    "environment_consistency": "maintain_setup"
                },
                "generation_sequence": {
                    "order": ["FRAG_001", "FRAG_002"],
                    "batch_groups": []
                }
            }
        }
