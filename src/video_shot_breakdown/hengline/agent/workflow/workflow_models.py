"""
@FileName: workflow_models.py
@Description: 
@Author: HengLine
@Time: 2026/1/27 19:12
"""
from enum import Enum, unique


@unique
class AgentStage(str, Enum):
    START = "loaded"  # 开始
    INIT = "initialized"  # 开始
    PARSER = "parsed"  # 智能体1：剧本解析
    SEGMENTER = "segmented"  # 智能体2：分镜拆分
    SPLITTER = "split"  # 智能体3：片段分隔
    CONVERTER = "converted"  # 智能体4：指令转换
    AUDITOR = "audited"  # 智能体5：质量审查
    CONTINUITY = "continuity_check"  # 结束
    ERROR = "error_handling"  # 结束
    HUMAN = "human_intervention"  # 结束
    END = "completed"  # 结束


class ConditionalEdges(str, Enum):
    PARSE_SCRIPT = "parse_script"
    SEGMENT_SHOT = "segment_shot"
    SPLIT_VIDEO = "split_video"
    CONVERT_PROMPT = "convert_prompt"
    AUDIT_QUALITY = "audit_quality"

