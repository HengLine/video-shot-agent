# -*- coding: utf-8 -*-
"""
@FileName: workflow_states.py
@Description: 分镜生成工作流的状态定义
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from typing import Dict, List, Optional, TypedDict, Any

from hengline.agent.script_parser.script_parser_models import ParsedScript


class InputState(TypedDict):
    """工作流输入状态"""
    raw_script: str  # 原始剧本文本
    user_config: Dict  # 用户配置（模型选择、风格偏好等）
    duration_per_shot: int  # 每段时长
    task_id: str  # 唯一标识符


class ScriptParsingState(TypedDict):
    """剧本解析相关状态"""
    parsed_script: ParsedScript  # 结构化剧本
    parse_errors: List[str]  # 解析错误信息
    parse_warnings: List[str]  # 解析警告信息


class ShotGeneratorState(TypedDict):
    """分镜生成相关状态"""
    shots: List[Dict]  # 镜头序列
    current_shot_index: int  # 当前处理的镜头索引
    shot_errors: Dict[str, List]  # 按镜头存储的错误


class VideoSegmenterState(TypedDict):
    """视频拆分相关状态"""
    fragments: List[Dict]  # AI视频片段序列
    fragment_quality_scores: Dict[str, float]  # 片段质量评分


class PromptConverterState(TypedDict):
    """指令转换相关状态"""
    ai_instructions: List[Dict]  # AI生成指令
    prompt_templates_used: List[str]  # 使用的Prompt模板


class QualityAuditorState(TypedDict):
    """质量审查相关状态"""
    audit_report: Optional[Dict]  # 质量审查报告
    audit_failures: List[str]  # 审查失败项
    audit_warnings: List[str]  # 审查警告项


class OutputState(TypedDict):
    """工作流输出状态"""
    final_output: Optional[Dict]  # 最终输出结果
    execution_plan: Optional[Dict]  # 执行计划说明
    error: Optional[str]  # 错误信息


class WorkflowState(InputState, ScriptParsingState, ShotGeneratorState,
                              VideoSegmenterState, PromptConverterState, QualityAuditorState, OutputState):
    """
    完整的分镜生成工作流状态
    通过继承多个特定功能的状态类来组合，实现高内聚低耦合
    """
    # === 工作流控制 ===
    current_stage: str  # 当前处理阶段
    retry_count: int  # 重试计数
    max_retries: int  # 最大重试次数
    should_abort: bool  # 是否中止流程
    error_messages: List[str]  # 累计错误信息

    # === 连续性管理 ===
    continuity_state: Dict  # 当前连续性状态
    continuity_issues: List[Dict]  # 连续性问题列表
    continuity_anchors: Dict  # 连续性锚点映射

    # 人工决策
    human_feedback: Dict[str, Any]
