#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HengLine 工具模块
提供LlamaIndex集成和剧本智能分析功能
"""

# 导出JSON响应解析器
from hengline.tools.json_parser_tool import (
    JsonResponseParser,
    json_parser,
    parse_json_response,
    extract_json_from_markdown
)
# LlamaIndex 核心功能
from .llama_index_loader import DocumentLoader, DirectoryLoader
from .llama_index_retriever import DocumentRetriever
from .llama_index_tool import create_vector_store
# 结果存储功能
from .result_storage_tool import (
    ResultStorage,
    create_result_storage,
    save_script_parser_result,
    load_script_parser_result
)
# 剧本智能分析功能
from .script_intelligence_tool import (
    ScriptIntelligence,
    create_script_intelligence,
    analyze_script,
    search_script
)
# 剧本知识库功能
from .script_knowledge_tool import (
    ScriptKnowledgeBase,
    create_script_knowledge_base
)
# 剧本解析功能
from .script_parser_tool import (
    ScriptParserTool,
    parse_script_to_documents,
    parse_script_file_to_documents,
    Scene,
    Character,
    SceneElement
)

__all__ = [
    # LlamaIndex 核心功能
    "DocumentLoader",
    "DirectoryLoader",
    "DocumentRetriever",
    "create_vector_store",

    # 剧本解析
    "ScriptParserTool",
    "parse_script_to_documents",
    "parse_script_file_to_documents",
    "Scene",
    "Character",
    "SceneElement",

    # 剧本知识库
    "ScriptKnowledgeBase",
    "create_script_knowledge_base",

    # 剧本智能分析
    "ScriptIntelligence",
    "create_script_intelligence",
    "analyze_script",
    "search_script",

    # 结果存储
    "ResultStorage",
    "create_result_storage",
    "save_script_parser_result",
    "load_script_parser_result",
    
    # JSON响应解析
    "JsonResponseParser",
    "json_parser",
    "parse_json_response",
    "extract_json_from_markdown"
]
