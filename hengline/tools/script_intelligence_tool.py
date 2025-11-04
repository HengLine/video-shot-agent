#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
剧本智能分析综合工具模块
整合剧本解析、知识库管理、智能检索等功能
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional, Any

from hengline.tools.script_parser_tool import (
    ScriptParser,
    parse_script_to_documents,
    parse_script_file_to_documents,
    Scene, Character, SceneElement
)
from llama_index.core.retrievers import BaseRetriever

from hengline.logger import debug, info, error, warning
from hengline.client.embedding_client import get_embedding_client
from hengline.tools.script_knowledge_tool import (
    create_script_knowledge_base
)


class ScriptIntelligence:
    """
    剧本智能分析主类
    提供剧本解析、知识库构建、智能检索、统计分析等一站式功能
    """

    def __init__(self,
                 embedding_model_type: str = "openai",
                 embedding_model_name: Optional[str] = None,
                 embedding_model_config: Optional[Dict[str, Any]] = None,
                 storage_dir: Optional[str] = None,
                 chunk_size: int = 512,
                 chunk_overlap: int = 20):
        """
        初始化剧本智能分析工具
        
        Args:
            embedding_model_type: 嵌入模型类型 (openai, huggingface, ollama)
            embedding_model_name: 嵌入模型名称（具体模型标识）
            embedding_model_config: 嵌入模型配置参数
            storage_dir: 知识库存储目录
            chunk_size: 文本分块大小
            chunk_overlap: 文本块重叠大小
        """
        self.embedding_model_type = embedding_model_type
        self.embedding_model_name = embedding_model_name
        self.embedding_model_config = embedding_model_config or {}
        self.storage_dir = storage_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化组件
        self.parser = ScriptParser()
        self.embedding_model = None
        self.knowledge_base = None

        # 初始化嵌入模型
        try:
            # 准备参数
            model_kwargs = {}
            if embedding_model_name:
                model_kwargs["model"] = embedding_model_name

            # 合并配置
            merged_config = {**self.embedding_model_config, **model_kwargs}

            self.embedding_model = get_embedding_client(
                model_type=embedding_model_type,
                model_name=embedding_model_name,
                **merged_config
            )
            debug(f"初始化嵌入模型成功: {embedding_model_type}{f'/{embedding_model_name}' if embedding_model_name else ''}")
        except Exception as e:
            warning(f"初始化嵌入模型失败，将在需要时重新尝试: {str(e)}")

        # 初始化知识库
        self.knowledge_base = create_script_knowledge_base(
            embedding_model=self.embedding_model,
            storage_dir=storage_dir
        )

        debug("剧本智能分析工具初始化完成")

    def analyze_script_text(self, script_text: str, script_id: str = None) -> Dict[str, Any]:
        """
        分析剧本文本
        
        Args:
            script_text: 剧本文本
            script_id: 剧本唯一标识
            
        Returns:
            分析结果
        """
        try:
            if script_id is None:
                script_id = f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            debug(f"开始分析剧本文本: {script_id}")

            # 解析剧本
            parsed_result, documents = parse_script_to_documents(script_text)

            # 添加到知识库
            kb_result = self.knowledge_base.add_script_text(script_text, script_id)

            # 生成分析报告
            analysis = self._generate_script_analysis(parsed_result)

            result = {
                "script_id": script_id,
                "parsed_result": parsed_result,
                "kb_result": kb_result,
                "analysis": analysis,
                "document_count": len(documents),
                "timestamp": datetime.now().isoformat()
            }

            info(f"剧本分析完成: {script_id}")
            return result

        except Exception as e:
            error(f"分析剧本文本失败: {str(e)}")
            raise

    def analyze_script_file(self, file_path: str, script_id: str = None) -> Dict[str, Any]:
        """
        分析剧本文件
        
        Args:
            file_path: 剧本文件路径
            script_id: 剧本唯一标识
            
        Returns:
            分析结果
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"剧本文件不存在: {file_path}")

            if script_id is None:
                # 从文件名生成ID
                base_name = os.path.basename(file_path)
                script_id = f"script_{os.path.splitext(base_name)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            debug(f"开始分析剧本文件: {file_path} as {script_id}")

            # 解析剧本
            parsed_result, documents = parse_script_file_to_documents(file_path)

            # 添加到知识库
            kb_result = self.knowledge_base.add_script_file(file_path, script_id)

            # 生成分析报告
            analysis = self._generate_script_analysis(parsed_result)

            result = {
                "script_id": script_id,
                "file_path": file_path,
                "parsed_result": parsed_result,
                "kb_result": kb_result,
                "analysis": analysis,
                "document_count": len(documents),
                "timestamp": datetime.now().isoformat()
            }

            info(f"剧本文件分析完成: {file_path}")
            return result

        except Exception as e:
            error(f"分析剧本文件失败: {str(e)}")
            raise

    def analyze_script_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        分析目录中的所有剧本
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归处理子目录
            
        Returns:
            分析结果
        """
        try:
            if not os.path.exists(directory_path):
                raise FileNotFoundError(f"目录不存在: {directory_path}")

            debug(f"开始分析剧本目录: {directory_path}, 递归: {recursive}")

            # 添加目录到知识库
            kb_result = self.knowledge_base.add_script_directory(directory_path, recursive)

            # 生成综合分析报告
            analysis = self._generate_directory_analysis(kb_result)

            result = {
                "directory_path": directory_path,
                "kb_result": kb_result,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }

            info(f"剧本目录分析完成: {directory_path}")
            return result

        except Exception as e:
            error(f"分析剧本目录失败: {str(e)}")
            raise

    def search(self, query: str, search_type: str = "similarity",
               top_k: int = 5, use_rerank: bool = False) -> Dict[str, Any]:
        """
        搜索剧本内容
        
        Args:
            query: 搜索查询
            search_type: 搜索类型 (similarity, mmr)
            top_k: 返回结果数量
            use_rerank: 是否使用重排序
            
        Returns:
            搜索结果
        """
        try:
            debug(f"执行搜索: {query}, 类型: {search_type}, top_k: {top_k}")

            # 使用知识库进行搜索
            results = self.knowledge_base.query(
                query_text=query,
                search_type=search_type,
                similarity_top_k=top_k,
                use_rerank=use_rerank
            )

            # 增强搜索结果
            enhanced_results = self._enhance_search_results(results)

            info(f"搜索完成: {query}, 找到{len(results['results'])}个结果")
            return enhanced_results

        except Exception as e:
            error(f"搜索失败: {str(e)}")
            raise

    def get_scene_info(self, scene_number: int) -> Optional[Dict[str, Any]]:
        """
        获取场景信息
        
        Args:
            scene_number: 场景编号
            
        Returns:
            场景信息
        """
        return self.knowledge_base.query_scene(scene_number)

    def get_character_info(self, character_name: str) -> Optional[Dict[str, Any]]:
        """
        获取角色信息
        
        Args:
            character_name: 角色名称
            
        Returns:
            角色信息
        """
        return self.knowledge_base.query_character(character_name)

    def get_script_statistics(self) -> Dict[str, Any]:
        """
        获取剧本统计信息
        
        Returns:
            统计信息
        """
        return self.knowledge_base.get_statistics()

    def create_custom_retriever(self, config: Dict[str, Any]) -> BaseRetriever:
        """
        创建自定义检索器
        
        Args:
            config: 检索器配置
            
        Returns:
            检索器实例
        """
        try:
            search_type = config.get("search_type", "similarity")
            top_k = config.get("top_k", 5)
            use_rerank = config.get("use_rerank", False)
            rerank_model = config.get("rerank_model", "BAAI/bge-reranker-large")

            retriever = self.knowledge_base.create_retriever(
                search_type=search_type,
                similarity_top_k=top_k,
                use_rerank=use_rerank,
                rerank_model=rerank_model
            )

            debug(f"创建自定义检索器成功")
            return retriever

        except Exception as e:
            error(f"创建自定义检索器失败: {str(e)}")
            raise

    def export_knowledge_base(self, export_dir: str) -> Dict[str, Any]:
        """
        导出知识库
        
        Args:
            export_dir: 导出目录
            
        Returns:
            导出结果
        """
        try:
            os.makedirs(export_dir, exist_ok=True)

            # 导出统计信息
            stats_path = os.path.join(export_dir, "statistics.json")
            stats = self.get_script_statistics()
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            # 导出解析结果
            parsed_dir = os.path.join(export_dir, "parsed_results")
            os.makedirs(parsed_dir, exist_ok=True)

            for script_id, parsed_result in self.knowledge_base.parsed_results.items():
                parsed_path = os.path.join(parsed_dir, f"{script_id}.json")
                with open(parsed_path, 'w', encoding='utf-8') as f:
                    json.dump(parsed_result, f, ensure_ascii=False, indent=2)

            # 导出存储的向量库（如果存在）
            if self.storage_dir and os.path.exists(self.storage_dir):
                import shutil
                vector_dir = os.path.join(export_dir, "vector_store")
                shutil.copytree(self.storage_dir, vector_dir, dirs_exist_ok=True)

            export_info = {
                "export_dir": export_dir,
                "script_count": stats["script_count"],
                "exported_parsed_results": len(self.knowledge_base.parsed_results),
                "timestamp": datetime.now().isoformat()
            }

            info(f"知识库导出成功: {export_dir}")
            return export_info

        except Exception as e:
            error(f"导出知识库失败: {str(e)}")
            raise

    def clear_all(self):
        """
        清空所有数据
        """
        try:
            self.knowledge_base.clear()
            info("已清空所有数据")
        except Exception as e:
            error(f"清空数据失败: {str(e)}")
            raise

    def _generate_script_analysis(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成剧本分析报告
        
        Args:
            parsed_result: 解析结果
            
        Returns:
            分析报告
        """
        scenes = parsed_result.get("scenes", [])
        characters = parsed_result.get("characters", {})
        elements = parsed_result.get("elements", [])

        # 场景分析
        scene_types = {}
        time_distribution = {}

        for scene in scenes:
            # 统计场景类型
            location_type = "UNKNOWN"
            if scene.get("location", "").startswith("INT"):
                location_type = "室内"
            elif scene.get("location", "").startswith("EXT"):
                location_type = "室外"
            elif scene.get("location", "").startswith("I/E"):
                location_type = "室内/室外"

            scene_types[location_type] = scene_types.get(location_type, 0) + 1

            # 统计时间分布
            time_of_day = scene.get("time_of_day", "未知")
            time_distribution[time_of_day] = time_distribution.get(time_of_day, 0) + 1

        # 角色分析
        character_dialogue_counts = {}
        for char_name, char_info in characters.items():
            character_dialogue_counts[char_name] = char_info.get("dialogue_count", 0)

        # 按对话次数排序角色
        top_characters = sorted(
            character_dialogue_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # 元素类型统计
        element_types = {}
        for element in elements:
            elem_type = element.get("type", "unknown")
            element_types[elem_type] = element_types.get(elem_type, 0) + 1

        # 计算平均场景长度
        total_lines = sum(scene.get("end_line", 0) - scene.get("start_line", 0) + 1 for scene in scenes)
        avg_scene_lines = total_lines / len(scenes) if scenes else 0

        analysis = {
            "overview": {
                "total_scenes": len(scenes),
                "total_characters": len(characters),
                "total_elements": len(elements),
                "total_lines": parsed_result.get("total_lines", 0),
                "avg_scene_lines": round(avg_scene_lines, 2)
            },
            "scene_analysis": {
                "scene_types": scene_types,
                "time_distribution": time_distribution
            },
            "character_analysis": {
                "top_characters_by_dialogue": [
                    {"name": name, "dialogue_count": count}
                    for name, count in top_characters
                ],
                "character_count": len(characters)
            },
            "element_analysis": {
                "element_types": element_types
            }
        }

        return analysis

    def _generate_directory_analysis(self, kb_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成目录分析报告
        
        Args:
            kb_result: 知识库添加结果
            
        Returns:
            分析报告
        """
        added_scripts = kb_result.get("added_scripts", [])

        # 计算总体统计
        total_scenes = sum(script.get("scene_count", 0) for script in added_scripts)
        total_characters = sum(script.get("character_count", 0) for script in added_scripts)
        total_documents = sum(script.get("document_count", 0) for script in added_scripts)

        # 分析单个剧本统计
        script_stats = []
        for script in added_scripts:
            script_stat = {
                "script_id": script.get("script_id"),
                "file_path": script.get("file_path"),
                "scene_count": script.get("scene_count"),
                "character_count": script.get("character_count"),
                "document_count": script.get("document_count")
            }
            script_stats.append(script_stat)

        analysis = {
            "overview": {
                "total_scripts": len(added_scripts),
                "total_scenes": total_scenes,
                "total_characters": total_characters,
                "total_documents": total_documents,
                "avg_scenes_per_script": round(total_scenes / len(added_scripts), 2) if added_scripts else 0,
                "avg_characters_per_script": round(total_characters / len(added_scripts), 2) if added_scripts else 0
            },
            "script_details": script_stats
        }

        return analysis

    def _enhance_search_results(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        增强搜索结果
        
        Args:
            search_results: 原始搜索结果
            
        Returns:
            增强后的搜索结果
        """
        enhanced_results = search_results.copy()

        # 对每个结果添加类型标签和上下文增强
        for result in enhanced_results.get("results", []):
            metadata = result.get("metadata", {})

            # 添加结果类型标签
            if metadata.get("type") == "scene":
                result["result_type"] = "场景"
                result["display_title"] = f"场景 {metadata.get('scene_number', '?')}: {metadata.get('scene_heading', '')}"
            elif metadata.get("type") == "character":
                result["result_type"] = "角色"
                result["display_title"] = f"角色: {metadata.get('character_name', '')}"
            else:
                result["result_type"] = "内容"
                result["display_title"] = result["text"][:50] + "..." if len(result["text"]) > 50 else result["text"]

            # 添加相关性评分的描述
            score = result.get("score", 0)
            if score > 0.8:
                result["relevance"] = "极高相关"
            elif score > 0.6:
                result["relevance"] = "高度相关"
            elif score > 0.4:
                result["relevance"] = "中度相关"
            else:
                result["relevance"] = "低相关"

        return enhanced_results


def create_script_intelligence(
        embedding_model_type: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        embedding_model_config: Optional[Dict[str, Any]] = None,
        storage_dir: Optional[str] = None
) -> ScriptIntelligence:
    """
    创建剧本智能分析实例
    
    Args:
        embedding_model_type: 嵌入模型类型 (openai, huggingface, ollama)，如果为None则从配置中读取
        embedding_model_name: 嵌入模型名称（具体模型标识）
        embedding_model_config: 嵌入模型配置
        storage_dir: 存储目录
        
    Returns:
        剧本智能分析实例
    """
    # 导入配置函数
    from config.config import get_embedding_config

    # 处理向后兼容情况：如果只提供了embedding_model_name且它是模型类型
    # （openai, huggingface, ollama），则将其视为model_type
    embedding_config = get_embedding_config()

    # 如果embedding_model_type为None，则从配置中读取
    if embedding_model_type is None:
        embedding_model_type = embedding_config.get("provider", "openai")
        debug(f"从配置中读取嵌入模型类型: {embedding_model_type}")

    embedding_model_config = embedding_model_config or embedding_config

    return ScriptIntelligence(
        embedding_model_type=embedding_model_type,
        embedding_model_name=embedding_model_name,
        embedding_model_config=embedding_model_config,
        storage_dir=storage_dir
    )


# 便捷函数
def analyze_script(script_path_or_text: str, is_file: bool = True, **kwargs) -> Dict[str, Any]:
    """
    便捷的剧本分析函数
    
    Args:
        script_path_or_text: 剧本文件路径或文本内容
        is_file: 是否是文件路径
        **kwargs: 其他参数，支持embedding_model_type、embedding_model_name等
        
    Returns:
        分析结果
    """
    intelligence = create_script_intelligence(**kwargs)

    if is_file:
        return intelligence.analyze_script_file(script_path_or_text)
    else:
        return intelligence.analyze_script_text(script_path_or_text)


def search_script(query: str, storage_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    便捷的剧本搜索函数
    
    Args:
        query: 搜索查询
        storage_dir: 知识库存储目录
        **kwargs: 其他搜索参数
        
    Returns:
        搜索结果
    """
    intelligence = create_script_intelligence(storage_dir=storage_dir)
    return intelligence.search(query, **kwargs)
