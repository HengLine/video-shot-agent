"""
@FileName: workflow_models.py
@Description:  工作流模型定义模块
@Author: HengLine
@Time: 2025/11/30 19:12
"""
from enum import Enum, unique


@unique
class AgentType(Enum):
    PARSER = "parser"  # 智能体1：剧本解析
    PLANNER = "planner"  # 智能体2：时序规划
    CONTINUITY = "continuity"  # 智能体3：连贯性
    VISUAL = "visual"  # 智能体4：视觉生成
    REVIEWER = "reviewer"  # 智能体5：质量审查


@unique
class ScriptType(Enum):
    """剧本格式类型"""
    NATURAL_LANGUAGE = "natural_language"  # 自然语言描述
    AI_STORYBOARD = "ai_storyboard"  # AI分镜脚本
    STRUCTURED_SCENE = "structured_scene"  # 结构化场景描述
    STANDARD_SCRIPT = "standard_script"  # 标准剧本格式
    DIALOGUE_ONLY = "dialogue_only"  # 纯对话
    MIXED_FORMAT = "mixed_format"  # 混合格式


@unique
class ParserType(Enum):
    LLM_PARSER = "llm_parser"  # LLM 解析器
    RULE_PARSER = "rule_parser"  # 本地规则解析器


@unique
class VideoStyle(Enum):
    # 逼真
    REALISTIC = 'realistic'
    # 动漫
    ANIME = 'anime'
    # 电影
    CINEMATIC = 'cinematic'
    # 卡通
    CARTOON = 'cartoon'