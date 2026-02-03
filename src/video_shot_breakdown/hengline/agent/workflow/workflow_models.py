"""
@FileName: workflow_models.py
@Description: 工作流模型定义文件，包含工作流状态和条件的枚举类
@Author: HengLine
@Time: 2026/1/27 19:12
"""
from enum import Enum, unique


@unique
class AgentStage(str, Enum):
    """智能体工作流阶段枚举类，定义工作流的不同阶段状态"""
    START = "loaded"  # 开始状态
    INIT = "initialized"  # 初始化状态
    PARSER = "parsed"  # 剧本解析完成
    SEGMENTER = "segmented"  # 分镜拆分完成
    SPLITTER = "split"  # 片段分隔完成
    CONVERTER = "converted"  # 指令转换完成
    AUDITOR = "audited"  # 质量审查完成
    CONTINUITY = "continuity_check"  # 连续性检查完成
    ERROR = "error_handling"  # 错误处理中
    HUMAN = "human_intervention"  # 人工干预中
    END = "completed"  # 工作流完成


class ConditionalEdges(str, Enum):
    """工作流条件分支枚举类，定义工作流中的条件分支"""
    PARSE_SCRIPT = "parse_script"  # 剧本解析分支
    SEGMENT_SHOT = "segment_shot"  # 分镜拆分分支
    SPLIT_VIDEO = "split_video"  # 片段分隔分支
    CONVERT_PROMPT = "convert_prompt"  # 指令转换分支
    AUDIT_QUALITY = "audit_quality"  # 质量审查分支


@unique
class DecisionState(str, Enum):
    """决策函数状态枚举类，定义决策函数返回的状态"""
    SUCCESS = "success"  # 操作成功
    CRITICAL_FAILURE = "critical_failure"  # 严重失败，需要终止流程
    NEEDS_ADJUSTMENT = "needs_adjustment"  # 需要调整
    RETRY = "retry"  # 需要重试
    NEEDS_HUMAN = "needs_human"  # 需要人工干预
    CONTINUE = "continue"  # 继续下一步
    RECOVERABLE = "recoverable"  # 可恢复的错误
    ABORT = "abort"  # 中止流程


@unique
class PipelineState(str, Enum):
    """工作流管道状态枚举类，定义工作流管道中的状态"""
    CONTINUITY_CHECK = "continuity_check"  # 连续性检查
    HUMAN_INTERVENTION = "human_intervention"  # 人工干预
    GENERATE_PROMPTS = "generate_prompts"  # 生成指令
    SPLIT_SHOTS = "split_shots"  # 拆分镜头
    FRAGMENT_FOR_AI = "fragment_for_ai"  # AI分段
    QUALITY_AUDIT = "quality_audit"  # 质量审查
    GENERATE_OUTPUT = "generate_output"  # 生成输出
    ERROR_HANDLER = "error_handler"  # 错误处理


@unique
class PipelineNode(str, Enum):
    """工作流管道节点枚举类，定义工作流管道中的节点"""
    PARSE_SCRIPT = "parse_script"  # 剧本解析节点
    SEGMENT_SHOT = "segment_shot"  # 分镜拆分节点
    SPLIT_VIDEO = "split_video"  # 片段分隔节点
    CONVERT_PROMPT = "convert_prompt"  # 指令转换节点
    AUDIT_QUALITY = "audit_quality"  # 质量审查节点
    CONTINUITY_CHECK = "continuity_check"  # 连续性检查节点
    HUMAN_INTERVENTION = "human_intervention"  # 人工干预节点
    ERROR_HANDLER = "error_handler"  # 错误处理节点
    GENERATE_OUTPUT = "generate_output"  # 生成输出节点
    END = "end"  # 结束节点