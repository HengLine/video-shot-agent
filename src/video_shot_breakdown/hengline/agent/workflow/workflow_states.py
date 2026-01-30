# -*- coding: utf-8 -*-
"""
@FileName: workflow_states.py
@Description: 分镜生成工作流的状态定义
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import uuid
from typing import Dict, List, Optional, Any

from pydantic import BaseModel

from video_shot_breakdown.hengline.agent.prompt_converter.prompt_converter_models import AIVideoInstructions
from video_shot_breakdown.hengline.agent.quality_auditor.quality_auditor_models import QualityAuditReport
from video_shot_breakdown.hengline.agent.script_parser.script_parser_models import ParsedScript
from video_shot_breakdown.hengline.agent.shot_segmenter.shot_segmenter_models import ShotSequence
from video_shot_breakdown.hengline.agent.video_splitter.video_splitter_models import FragmentSequence
from video_shot_breakdown.hengline.agent.workflow.workflow_models import AgentStage
from video_shot_breakdown.hengline.hengline_config import HengLineConfig


class InputState(BaseModel):
    """工作流输入状态"""
    raw_script: str  # 原始剧本文本
    user_config: HengLineConfig = {}  # 用户配置（模型选择、风格偏好等）
    task_id: str = str(uuid.uuid4())  # 唯一标识符

class ScriptParsingState(BaseModel):
    """剧本解析相关状态"""
    parsed_script: ParsedScript = None  # 结构化剧本
    parse_errors: List[str] = []  # 解析错误信息
    parse_warnings: List[str] = []  # 解析警告信息


class ShotGeneratorState(BaseModel):
    """分镜生成相关状态"""
    shot_sequence: ShotSequence = None  # 镜头序列
    current_shot_index: int = None  # 当前处理的镜头索引
    shot_errors: Dict[str, List] = None  # 按镜头存储的错误


class VideoSegmenterState(BaseModel):
    """视频拆分相关状态"""
    fragment_sequence: FragmentSequence = None  # AI视频片段序列
    fragment_quality_scores: Dict[str, float] = None  # 片段质量评分


class PromptConverterState(BaseModel):
    """指令转换相关状态"""
    instructions: AIVideoInstructions = None  # AI生成指令
    prompt_templates_used: List[str] = None  # 使用的Prompt模板


class QualityAuditorState(BaseModel):
    """质量审查相关状态"""
    audit_report: Optional[QualityAuditReport]  = None # 质量审查报告
    audit_failures: List[str] = []  # 审查失败项
    audit_warnings: List[str] = []  # 审查警告项


class OutputState(BaseModel):
    """工作流输出状态"""
    final_output: Optional[Dict] = None  # 最终输出结果
    execution_plan: Optional[Dict] = None  # 执行计划说明
    error: Optional[str] = None  # 错误信息


class WorkflowState(InputState, ScriptParsingState, ShotGeneratorState,
                    VideoSegmenterState, PromptConverterState, QualityAuditorState, OutputState):
    """
    完整的分镜生成工作流状态
    通过继承多个特定功能的状态类来组合，实现高内聚低耦合
    """
    # === 工作流控制 ===
    current_stage: AgentStage = AgentStage.START  # 当前处理阶段
    retry_count: int = 0  # 重试计数
    max_retries: int = 3  # 最大重试次数
    should_abort: bool = False  # 是否中止流程
    error_messages: List[str] = []  # 累计错误信息

    # === 连续性管理 ===
    continuity_state: Dict = {}  # 当前连续性状态
    continuity_issues: List[Dict] = []  # 连续性问题列表
    continuity_anchors: Dict = {}  # 连续性锚点映射

    # 人工决策
    needs_human_review: bool = False
    human_feedback: Dict[str, Any] = {}
