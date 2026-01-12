# -*- coding: utf-8 -*-
"""
@FileName: workflow_states.py
@Description: 分镜生成工作流的状态定义
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from typing import Dict, List, Any, Optional, TypedDict

from hengline.agent.continuity_guardian.model.continuity_guardian_report import AnchoredTimeline
from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.shot_generator.model.shot_models import SoraShot
from hengline.agent.temporal_planner.temporal_planner_model import TimeSegment, TimelinePlan
from hengline.agent.workflow.workflow_models import VideoStyle


class InputState(TypedDict):
    """工作流输入状态"""
    script_text: str  # 原始剧本文本
    style: VideoStyle  # 视频风格："realistic"（逼真）、"anime"（动漫）、"cinematic"（电影）、"cartoon"（卡通）
    prev_continuity_state: Optional[Dict[str, Any]]  # 上一段的连续性状态
    duration_per_shot: int  # 每段时长
    task_id: str  # 唯一标识符


class ScriptParsingState(TypedDict):
    """剧本解析相关状态"""
    structured_script: UnifiedScript  # 结构化剧本
    title: str  # 剧本标题
    memory: dict  # 自定义 memory 字段


class TimelinePlanningState(TypedDict):
    """时间线规划相关状态"""
    segments: Optional[List[TimelinePlan]]  # 时间线分段
    current_segment_index: int  # 当前处理的分段索引


class ShotGenerationState(TypedDict):
    """分镜生成相关状态"""
    shots: List[SoraShot]  # 已生成的分镜列表
    current_continuity_state: Optional[AnchoredTimeline]  # 当前连续性状态
    retry_count: int  # 重试次数
    max_retries: int  # 最大重试次数
    current_segment: Optional[TimeSegment]  # 当前处理的分段
    current_shot: Optional[SoraShot]  # 当前生成的分镜


class ReviewState(TypedDict):
    """审查相关状态"""
    qa_results: List[Dict[str, Any]]  # 单个分镜审查结果列表
    sequence_qa: Optional[Dict[str, Any]]  # 分镜序列审查结果


class OutputState(TypedDict):
    """工作流输出状态"""
    result: Optional[Dict[str, Any]]  # 最终结果
    error: Optional[str]  # 错误信息


class StoryboardWorkflowState(InputState, ScriptParsingState, TimelinePlanningState,
                              ShotGenerationState, ReviewState, OutputState):
    """
    完整的分镜生成工作流状态
    通过继承多个特定功能的状态类来组合，实现高内聚低耦合
    """
    pass
