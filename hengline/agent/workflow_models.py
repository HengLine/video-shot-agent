"""
@FileName: workflow_models.py
@Description:  工作流模型定义模块
@Author: HengLine
@Time: 2025/11/30 19:12
"""
from enum import Enum, unique


@unique
class ScriptType(Enum):
    NATURAL_LANGUAGE = "natural_language"  # 自然语言描述
    STRUCTURED_SCENE = "structured_scene"  # 结构化分场剧本
    AI_STORYBOARD = "ai_storyboard"  # AI生成的分镜剧本
    SCREENPLAY_FORMAT = "screenplay_format"  # 标准剧本格式


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
