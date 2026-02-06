"""
@FileName: prompt_converter_models.py
@Description: 模型
@Author: HengLine
@Github: https://github.com/HengLine/video-shot-agent
@Time: 2026/1/18 14:25
"""
from datetime import datetime
from typing import Dict, Any, List, Optional

from pydantic import Field, BaseModel


class AIVideoPrompt(BaseModel):
    """MVP AI视频提示词模型"""
    fragment_id: str = Field(..., description="对应的片段ID")

    # 核心提示词
    prompt: str = Field(..., description="正向提示词文本")
    negative_prompt: str = Field(
        default="blurry, distorted, low quality, cartoonish, bad anatomy",
        description="负面提示词"
    )

    # 基本技术参数
    duration: float = Field(
        ...,
        ge=0.5,
        description="视频时长（秒）"
    )

    # 模型选择
    model: str = Field(
        default="runway_gen2",
        description="AI视频模型：runway_gen2/sora/pika（MVP先用runway）"
    )

    # 简化的风格提示
    style: Optional[str] = Field(
        default=None,
        description="风格提示：cinematic/realistic/anime/等"
    )

    # 扩展标记
    requires_special_attention: bool = Field(
        default=False,
        description="需要特殊处理的标记"
    )


class AIVideoInstructions(BaseModel):
    """MVP AI视频指令集输出"""

    # 元数据
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "generated_at": datetime.now().isoformat(),
            "version": "mvp_1.0",
            "target_model": "runway_gen2"
        }
    )

    # 项目信息
    project_info: Dict[str, Any] = Field(
        default_factory=lambda: {
            "title": "",
            "total_fragments": 0,
            "total_duration": 0.0,
            "source_fragments": []  # 原始片段ID列表
        }
    )

    # 核心指令
    fragments: List[AIVideoPrompt] = Field(
        default_factory=list,
        description="片段提示词列表"
    )

    # 极简全局设置
    global_settings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "style_consistency": True,
            "use_common_negative_prompt": True
        }
    )

    # 简化的执行建议
    execution_suggestions: List[str] = Field(
        default_factory=lambda: [
            "按顺序生成片段",
            "保持相同种子值以获得一致性",
            "生成后检查片段衔接"
        ]
    )
