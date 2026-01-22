"""
@FileName: video_assembler_models.py
@Description: 视频组装合成模型
@Author: HengLine
@Time: 2026/1/19 23:02
"""
from typing import Optional, List, Any, Dict

from pydantic import Field, BaseModel

from hengline.agent.base_models import BaseMetadata


class VideoSegment(BaseModel):
    """视频片段"""
    fragment_id: str = Field(..., description="片段ID")
    video_path: str = Field(..., description="视频文件路径")
    duration: float = Field(..., gt=0.0, description="时长（秒）")
    start_time_in_final: Optional[float] = Field(None, description="在最终视频中的开始时间")
    transitions: List[Dict[str, Any]] = Field(default_factory=list, description="转场效果")
    audio_track: Optional[str] = Field(None, description="音频轨道信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="片段元数据")


class AudioTrack(BaseModel):
    """音频轨道"""
    track_id: str = Field(..., description="轨道ID")
    type: str = Field(..., description="类型：dialogue/music/sfx/ambient")
    source_path: str = Field(..., description="源文件路径")
    start_time: float = Field(..., ge=0.0, description="开始时间")
    duration: float = Field(..., gt=0.0, description="时长")
    volume: float = Field(default=1.0, ge=0.0, le=2.0, description="音量")
    fade_in: Optional[float] = Field(None, description="淡入时长")
    fade_out: Optional[float] = Field(None, description="淡出时长")
    effects: Optional[List[str]] = Field(None, description="音效")


class FinalVideoSpec(BaseModel):
    """最终视频规格"""
    container: str = Field(default="mp4", description="容器格式")
    codec: str = Field(default="h264", description="视频编码")
    audio_codec: str = Field(default="aac", description="音频编码")
    resolution: str = Field(default="1920x1080", description="分辨率")
    frame_rate: int = Field(default=30, description="帧率")
    bitrate: str = Field(default="15Mbps", description="比特率")
    aspect_ratio: str = Field(default="16:9", description="宽高比")
    color_space: str = Field(default="rec709", description="色彩空间")


class AssemblyInstruction(BaseModel):
    """合成指令"""
    segment_order: List[str] = Field(..., description="片段顺序")
    transitions: List[Dict[str, Any]] = Field(default_factory=list, description="转场设置")
    color_grading: Optional[Dict[str, Any]] = Field(None, description="色彩校正")
    audio_mix: List[AudioTrack] = Field(default_factory=list, description="音频混音")
    overlays: Optional[List[Dict[str, Any]]] = Field(None, description="叠加层")
    subtitles: Optional[List[Dict[str, Any]]] = Field(None, description="字幕")
    metadata_embedding: Dict[str, Any] = Field(default_factory=dict, description="嵌入元数据")


class FinalVideoModel(BaseMetadata):
    """
    最终视频模型 - 最终阶段输出
    包含所有合成指令和最终输出规格
    """
    quality_report_source: Optional[str] = Field(None, description="来源质量报告ID")
    generation_results_source: List[str] = Field(
        default_factory=list,
        description="来源生成结果ID列表"
    )

    # 视频片段
    video_segments: List[VideoSegment] = Field(
        ...,
        min_length=1,
        description="视频片段列表"
    )

    # 合成指令
    assembly_instructions: AssemblyInstruction = Field(
        ...,
        description="合成指令"
    )

    # 输出规格
    final_spec: FinalVideoSpec = Field(default_factory=FinalVideoSpec, description="最终规格")

    # 质量控制
    final_quality_check: Optional[Dict[str, Any]] = Field(
        None,
        description="最终质量控制检查"
    )

    # 输出路径
    output_directory: str = Field(..., description="输出目录")
    final_video_path: Optional[str] = Field(None, description="最终视频路径")
    preview_path: Optional[str] = Field(None, description="预览路径")

    # 状态
    assembly_status: str = Field(default="pending", description="合成状态")
    assembly_progress: float = Field(default=0.0, ge=0.0, le=1.0, description="合成进度")

    # 元数据
    total_duration: float = Field(..., gt=0.0, description="总时长")
    total_segments: int = Field(..., ge=1, description="总片段数")
    assembly_version: str = Field(default="1.0", description="合成器版本")

    class Config:
        schema_extra = {
            "example": {
                "id": "FINAL_001",
                "project_id": "PROJ_001",
                "video_segments": [],
                "assembly_instructions": {
                    "segment_order": ["FRAG_001", "FRAG_002"],
                    "transitions": [],
                    "audio_mix": []
                },
                "final_spec": {
                    "container": "mp4",
                    "resolution": "1920x1080",
                    "frame_rate": 30
                },
                "total_duration": 45.5,
                "total_segments": 12,
                "assembly_status": "pending"
            }
        }
