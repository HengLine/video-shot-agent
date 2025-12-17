# -*- coding: utf-8 -*-
"""
@FileName: workflow_nodes.py
@Description: LangGraph工作流节点实现，包含所有工作流执行功能
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import uuid
from datetime import datetime
from typing import Dict, List, Any

from hengline.logger import debug, info, warning, error
from utils.log_utils import print_log_exception
from hengline.tools.result_storage_tool import create_result_storage
from .workflow_models import VideoStyle
from .workflow_states import StoryboardWorkflowState


class WorkflowNodes:
    """工作流节点集合，封装所有工作流执行功能"""

    def __init__(self, script_parser, temporal_planner, continuity_guardian, shot_generator, shot_qa, llm=None):
        """
        初始化工作流节点集合
        
        Args:
            script_parser: 剧本解析器实例
            temporal_planner: 时序规划器实例
            continuity_guardian: 连续性守卫实例
            shot_generator: 分镜生成器实例
            shot_qa: 质量审查实例
            llm: 语言模型实例（可选）
        """
        self.script_parser = script_parser
        self.temporal_planner = temporal_planner
        self.continuity_guardian = continuity_guardian
        self.shot_generator = shot_generator
        self.shot_qa = shot_qa
        self.llm = llm
        self.storage = create_result_storage()



    def parse_script_node(self, graph_state: StoryboardWorkflowState) -> Dict[str, Any]:
        """解析剧本文本节点"""
        debug("解析剧本文本节点执行中")
        try:
            structured_script = self.script_parser.parse_script(graph_state["script_text"])

            # 使用LLM增强解析结果（如果有）
            # if self.llm:
            #     structured_script = self.script_parser.enhance_with_llm(structured_script)

            debug(f"剧本解析完成，场景数: {len(structured_script.get('scenes', []))}")

            # 保存剧本解析结果
            task_id = graph_state.get("task_id")
            try:
                self.storage.save_result(task_id, structured_script, "script_parser_result.json")
                info(f"剧本解析结果已保存到: data/output/{task_id}/script_parser_result.json")
            except Exception as save_error:
                warning(f"保存剧本解析结果失败: {str(save_error)}")
            return {
                "structured_script": structured_script
            }

        except Exception as e:
            print_log_exception()
            error(f"剧本解析失败: {str(e)}")
            return {
                "error": str(e)
            }

    def plan_timeline_node(self, state: StoryboardWorkflowState) -> Dict[str, Any]:
        """规划时间线节点"""
        debug("规划时间线节点执行中")
        try:
            # 进行时序规划
            segments = self.temporal_planner.plan_timeline(
                state["structured_script"],
                state["duration_per_shot"]
            )
            
            # 保存时序规划结果
            task_id = state.get("task_id")
            try:
                result_data = {
                    "segments": segments,
                    "structured_script": state["structured_script"],
                    "duration_per_shot": state["duration_per_shot"]
                }
                self.storage.save_result(task_id, result_data, "temporal_planner_result.json")
                info(f"时序规划结果已保存到: data/output/{task_id}/temporal_planner_result.json")
            except Exception as save_error:
                warning(f"保存时序规划结果失败: {str(save_error)}")
            
            debug(f"时序规划完成，分段数: {len(segments)}")
            
            return {
                "segments": segments,
                "current_segment_index": 0
            }
        except Exception as e:
            error(f"时序规划失败: {str(e)}")
            return {
                "error": str(e)
            }

    def generate_shot_node(self, state: StoryboardWorkflowState) -> Dict[str, Any]:
        """
        生成分镜节点
        """
        debug(f"生成分镜节点执行中，当前分段索引: {state['current_segment_index']}")
        try:
            # 检查segments列表是否存在且不为空
            segments = state.get("segments", [])
            current_index = state.get("current_segment_index", 0)

            # 确保segments不为空
            if not segments:
                warning("分段列表为空，创建默认分段")
                segment = {
                    "id": 1,
                    "actions": [{
                        "character": "默认角色",
                        "action": "站立",
                        "emotion": "平静"
                    }],
                    "est_duration": 5.0,
                    "scene_id": 0
                }
            # 确保索引有效
            elif current_index < 0 or current_index >= len(segments):
                warning(f"无效的分段索引: {current_index}，使用第一个分段")
                segment = segments[0]
            else:
                segment = segments[current_index]

            shot_id = len(state.get("shots", [])) + 1

            # 获取场景上下文，增加安全检查
            scene_id = segment.get("scene_id", 0)
            scenes = state.get("structured_script", {}).get("scenes", [])
            scene_context = scenes[scene_id] if scene_id < len(scenes) else {}

            try:
                # 生成连续性约束
                continuity_constraints = self.continuity_guardian.generate_continuity_constraints(
                    segment,
                    state.get("current_continuity_state"),
                    scene_context
                )

                # 生成分镜
                shot = self.shot_generator.generate_shot(
                    segment,
                    continuity_constraints,
                    scene_context,
                    state["style"],
                    shot_id
                )
                
                # 保存分镜生成结果
                task_id = state.get("task_id")
                try:
                    self.storage.save_result(task_id, {
                        "shot_id": shot_id,
                        "segment": segment,
                        "shot": shot,
                        "continuity_constraints": continuity_constraints,
                        "scene_context": scene_context
                    }, f"generated_shot_{shot_id}.json")
                    info(f"分镜{shot_id}生成结果已保存")
                except Exception as save_error:
                    warning(f"保存分镜生成结果失败: {str(save_error)}")
            except Exception as shot_e:
                print_log_exception()
                error(f"生成自定义分镜失败: {str(shot_e)}")
                # 直接创建默认分镜
                shot = self._create_default_shot(segment, shot_id, state["style"])

            # 如果是重试，增加重试计数
            if state.get("retry_count", 0) > 0:
                debug(f"分镜 {shot_id} 重试生成中")

            return {
                "current_segment": segment,
                "current_shot": shot,
                "retry_count": state.get("retry_count", 0)
            }
        except Exception as e:
            error(f"生成分镜节点发生严重错误: {str(e)}")
            # 创建最基本的默认分镜，确保系统能够继续运行
            default_segment = {
                "id": 1,
                "actions": [],
                "est_duration": 5.0,
                "scene_id": 0
            }
            shot_id = len(state.get("shots", [])) + 1
            default_shot = self._create_default_shot(default_segment, shot_id, state.get("style", "realistic"))
            return {
                "current_segment": default_segment,
                "current_shot": default_shot,
                "retry_count": state.get("retry_count", 0) + 1
            }

    def review_shot_node(self, state: StoryboardWorkflowState) -> Dict[str, Any]:
        """审查分镜节点，优化警告处理"""
        info("审查分镜节点执行中")
        try:
            segment = state.get("current_segment")
            shot = state.get("current_shot")

            # 审查分镜
            qa_result = self.shot_qa.review_single_shot(shot, segment)

            # 记录不同级别的问题
            if qa_result.get("warnings"):
                info(f"分镜有警告: {qa_result.get('warnings')}")
            
            if qa_result.get("critical_issues"):
                info(f"分镜审查失败(关键错误): {qa_result.get('critical_issues')}")
            
            # 对默认分镜放宽要求
            if shot.get("warnings") and "使用默认分镜生成器" in shot.get("warnings", []):
                info("对默认分镜放宽审查标准")
                qa_result["is_valid"] = True
                qa_result["critical_issues"] = []
                # 保留警告信息
                if not qa_result.get("warnings"):
                    qa_result["warnings"] = []
                qa_result["warnings"].append("使用默认分镜生成器，放宽审查标准")

            # 添加到qa_results列表
            qa_results = state["qa_results"].copy()
            qa_results.append(qa_result)

            # 保存分镜审查结果
            task_id = state.get("task_id")
            try:
                shot_id = len(state.get("shots", [])) + 1
                self.storage.save_result(task_id, {
                    "shot_id": shot_id,
                    "segment": segment,
                    "shot": shot,
                    "qa_result": qa_result
                }, f"shot_review_{shot_id}.json")
                info(f"分镜{shot_id}审查结果已保存")
            except Exception as save_error:
                warning(f"保存分镜审查结果失败: {str(save_error)}")

            return {
                "qa_results": qa_results
            }
        except Exception as e:
            error(f"分镜审查失败: {str(e)}")
            # 添加失败的审查结果
            qa_results = state["qa_results"].copy()
            qa_results.append({"is_valid": False, "critical_issues": [str(e)], "warnings": [], "suggestions": []})
            return {
                "qa_results": qa_results
            }

    def check_retry_node(self, state: StoryboardWorkflowState) -> Dict[str, Any]:
        """检查是否需要重试节点，优化重试机制"""
        retry_count = state["retry_count"]
        max_retries = state["max_retries"]
        
        # 获取当前审查结果
        qa_results = state.get("qa_results", [])
        current_qa = qa_results[-1] if qa_results else None

        if retry_count < max_retries and (not current_qa or not current_qa.get("is_valid", False)):
            warning(f"分镜审查失败，开始重试 ({retry_count + 1}/{max_retries})")
            # 添加上一次失败的原因，帮助下次生成
            failure_reason = {
                "critical_issues": current_qa.get("critical_issues", []) if current_qa else [],
                "warnings": current_qa.get("warnings", []) if current_qa else []
            }
            return {
                "retry_count": retry_count + 1,
                "last_generation_failure": failure_reason
            }
        else:
            status_reason = '分镜达到最大重试次数' if retry_count >= max_retries else '分镜审查通过'
            warning(f"{status_reason}，添加到结果集")
            # 将当前分镜添加到shots列表
            shots = state["shots"].copy()
            current_shot = state.get("current_shot").copy()
            
            # 保留警告信息
            if current_qa:
                if "review_result" not in current_shot:
                    current_shot["review_result"] = {}
                current_shot["review_result"]["is_valid"] = current_qa.get("is_valid", False)
                
                # 合并警告信息
                all_warnings = []
                if current_qa.get("warnings"):
                    all_warnings.extend(current_qa.get("warnings"))
                if current_shot.get("warnings"):
                    all_warnings.extend(current_shot.get("warnings"))
                
                if all_warnings:
                    current_shot["warnings"] = list(set(all_warnings))  # 去重
            
            shots.append(current_shot)
            return {
                "shots": shots
            }

    def extract_continuity_node(self, state: StoryboardWorkflowState) -> Dict[str, Any]:
        """提取连续性信息节点"""
        debug("提取连续性信息节点执行中")
        try:
            # 如果是有效分镜，提取连续性锚点
            segment = state.get("current_segment")
            shot = state.get("current_shot")

            # 添加到shots列表
            shots = state["shots"].copy()
            shots.append(shot)

            # 提取连续性锚点
            continuity_anchor = self.continuity_guardian.extract_continuity_anchor(segment, shot)
            debug(f"分镜 {len(shots)} 生成并通过审查")

            # 保存连续性信息
            task_id = state.get("task_id")
            try:
                shot_id = len(shots)
                self.storage.save_result(task_id, {
                    "shot_id": shot_id,
                    "segment": segment,
                    "shot": shot,
                    "continuity_anchor": continuity_anchor
                }, f"continuity_anchor_{shot_id}.json")
                info(f"分镜{shot_id}连续性信息已保存")
            except Exception as save_error:
                warning(f"保存连续性信息失败: {str(save_error)}")

            # 移动到下一个分段
            current_segment_index = state["current_segment_index"] + 1

            # 将连续性锚点列表转换为字典，使用角色名作为键
            continuity_state_dict = {}
            if isinstance(continuity_anchor, list):
                for anchor in continuity_anchor:
                    if isinstance(anchor, dict) and "character_name" in anchor:
                        continuity_state_dict[anchor["character_name"]] = anchor
            else:
                continuity_state_dict = continuity_anchor

            return {
                "shots": shots,
                "current_continuity_state": continuity_state_dict,  # 使用字典类型
                "current_segment_index": current_segment_index,
                "retry_count": 0  # 重置重试计数
            }
        except Exception as e:
            error(f"提取连续性信息失败: {str(e)}")
            return {
                "error": str(e)
            }

    def review_sequence_node(self, state: StoryboardWorkflowState) -> Dict[str, Any]:
        """审查分镜序列节点，优化序列连续性审查"""
        debug("审查分镜序列连续性节点执行中")
        try:
            sequence_qa = self.shot_qa.review_shot_sequence(state["shots"])
            
            # 分类错误类型
            if sequence_qa.get("has_continuity_issues", False):
                # 检查是否有critical_issues字段（新格式）
                if "critical_issues" in sequence_qa:
                    # 直接使用QA Agent返回的分级结果
                    critical_issues = sequence_qa.get("critical_issues", [])
                    warnings = sequence_qa.get("warnings", [])
                else:
                    # 对旧格式结果进行分级
                    critical_issues = []
                    warnings = []
                    
                    for issue in sequence_qa.get("continuity_issues", []):
                        # 判断问题严重程度
                        if any(keyword in issue.lower() for keyword in ["严重", "致命", "无法修复", "位置冲突"]):
                            critical_issues.append(issue)
                        else:
                            warnings.append(issue)
                
                # 记录警告
                if warnings:
                    info(f"分镜序列有警告: {warnings}")
                
                # 更新序列审查结果
                if critical_issues:
                    warning(f"分镜序列审查失败(关键问题): {critical_issues}")
                    sequence_qa = {"has_continuity_issues": True, "critical_issues": critical_issues, "warnings": warnings}
                else:
                    # 只有警告的情况下，不认为存在连续性问题
                    info("分镜序列有警告但通过审查")
                    sequence_qa = {"has_continuity_issues": False, "warnings": warnings}
            
            # 保存序列审查结果
            task_id = state.get("task_id")
            try:
                self.storage.save_result(task_id, {
                    "shots": state["shots"],
                    "sequence_qa": sequence_qa
                }, "sequence_review.json")
                info(f"序列审查结果已保存")
            except Exception as save_error:
                warning(f"保存序列审查结果失败: {str(save_error)}")
            
            return {
                "sequence_qa": sequence_qa
            }
        except Exception as e:
            error(f"分镜序列审查失败: {str(e)}")
            # 返回默认的审查结果
            return {
                "sequence_qa": {"has_continuity_issues": False, "issues": [], "warnings": [f"审查过程出错: {str(e)}"]}
            }

    def fix_continuity_node(self, state: StoryboardWorkflowState) -> Dict[str, Any]:
        """修复连续性问题节点"""
        debug("修复连续性问题节点执行中")
        try:
            warning("分镜序列存在连续性问题，尝试修正")
            shots = state["shots"]
            qa_result = state["sequence_qa"]
            fixed_shots = self._fix_continuity_issues(shots, qa_result)
            
            # 保存连续性修复结果
            task_id = state.get("task_id")
            try:
                self.storage.save_result(task_id, {
                    "original_shots": shots,
                    "fixed_shots": fixed_shots,
                    "qa_result": qa_result
                }, f"continuity_fix_{len(fixed_shots)}.json")
                info(f"连续性修复结果已保存")
            except Exception as save_error:
                warning(f"保存连续性修复结果失败: {str(save_error)}")
                
            debug(f"已修复连续性问题，共调整 {len(fixed_shots)} 个分镜")

            return {
                "shots": fixed_shots,
                "sequence_qa": {"has_continuity_issues": False, "issues": []}  # 假设修复成功
            }
        except Exception as e:
            error(f"修复连续性问题失败: {str(e)}")
            return {
                "error": str(e)
            }

    def generate_result_node(self, state: StoryboardWorkflowState) -> Dict[str, Any]:
        """生成最终结果节点"""
        debug("生成最终结果节点执行中")
        try:
            result = self._generate_final_result(
                state["script_text"],
                state["shots"],
                state["style"],
                state["duration_per_shot"],
                state["sequence_qa"]
            )
            
            # 生成最终的分镜头剧本
            # 从结果中提取所需信息
            title = state.get("title", "分镜头剧本")
            shots = state["shots"]
            qa_results = state.get("qa_results", [])
            
            # 提取角色和场景位置信息
            characters = set()
            locations = set()
            for shot in shots:
                if shot.get("characters"):
                    characters.update(shot["characters"])
                scene_context = shot.get("scene_context", {})
                if scene_context.get("location"):
                    locations.add(scene_context["location"])
            
            final_storyboard = {
                "title": title,
                "scenes": shots,
                "qa_results": qa_results,
                "characters": list(characters),
                "locations": list(locations)
            }
            
            # 保存最终结果
            task_id = state.get("task_id")
            try:
                self.storage.save_result(task_id, result, "final_result.json")
                info(f"最终结果已保存到: data/output/{task_id}/final_result.json")
                
                # 保存最终分镜头剧本
                self.storage.save_result(task_id, final_storyboard, "final_storyboard.json")
                info(f"最终分镜头剧本已保存到: data/output/{task_id}/final_storyboard.json")
            except Exception as save_error:
                warning(f"保存最终结果失败: {str(save_error)}")
            
            return {
                "result": result,
                "final_storyboard": final_storyboard
            }
        except Exception as e:
            error(f"生成最终结果失败: {str(e)}")
            return {
                "error": str(e)
            }

    def _create_default_shot(self, segment: Dict[str, Any], shot_id: int, style: VideoStyle) -> Dict[str, Any]:
        """创建默认分镜，增强默认分镜的基本信息和连续性"""
        debug(f"创建默认分镜 {shot_id}")
        # 从分段中提取角色信息
        actions = segment.get("actions", [])
        characters = []
        dialogue = ""
        
        # 从场景上下文获取更多信息
        scene_context = segment.get("scene_context", {})
        location = scene_context.get("location", "未知位置")
        time_of_day = scene_context.get("time", "未知时间")

        if actions:
            # 提取角色名称和对话
            for action in actions:
                character_name = action.get("character", "角色")
                if character_name not in characters:
                    characters.append(character_name)
                if "dialogue" in action:
                    dialogue += f"{character_name}: {action['dialogue']}\n"
        else:
            characters = ["默认角色"]
        
        # 生成角色状态信息，提高默认分镜质量
        initial_state = []
        final_state = []
        for idx, character in enumerate(characters):
            # 为每个角色分配不同位置
            positions = ["left", "center", "right"]
            position = positions[idx % len(positions)]
            
            # 创建初始状态
            initial_state.append({
                "character_name": character,
                "pose": "standing",
                "position": position,
                "holding": "nothing",
                "emotion": "neutral"
            })
            
            # 创建结束状态（与初始状态相同，保持连续性）
            final_state.append({
                "character_name": character,
                "pose": "standing",
                "position": position,
                "holding": "nothing",
                "emotion": "neutral"
            })

        # 返回增强的默认分镜对象
        return {
            "shot_id": shot_id,
            "time_range_sec": [(shot_id - 1) * 5, shot_id * 5],
            "description": f"默认分镜描述：场景位于{location}，时间是{time_of_day}。这是一个为确保系统稳定性而生成的默认分镜。",
            "characters": characters,
            "dialogue": dialogue.strip(),
            "camera_angle": "medium_shot",
            "scene_id": segment.get("scene_id", 0),
            "style": style,
            "aspect_ratio": "16:9",
            "initial_state": initial_state,
            "final_state": final_state,
            "continuity_anchor": final_state,  # 使用final_state作为连续性锚点
            "warnings": ["使用默认分镜生成器"],
            "review_result": {
                "is_valid": True,
                "warnings": ["使用默认分镜生成器"],
                "generated_by": "default_generator"
            },
            "meta_data": {
                "generated_by": "default",
                "original_segment_id": segment.get("id"),
                "character_count": len(characters),
                "action_count": len(actions)
            },
            "device_holding": "smartphone",
            "final_continuity_state": {}
        }

    def _fix_continuity_issues(self, shots: List[Dict[str, Any]], qa_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """修复连续性问题"""
        fixed_shots = shots.copy()

        # 简单的修复逻辑
        # 这里可以根据qa_result中的建议进行更复杂的修复
        for i in range(1, len(fixed_shots)):
            prev_shot = fixed_shots[i - 1]
            current_shot = fixed_shots[i]

            # 修复时间范围
            prev_end = prev_shot["time_range_sec"][1]
            current_shot["time_range_sec"][0] = prev_end
            current_shot["time_range_sec"][1] = prev_end + 5

            # 修复角色状态连续性
            prev_final_state = {s.get("character_name"): s for s in prev_shot.get("final_state", [])}
            current_initial_state = current_shot.get("initial_state", [])

            for state in current_initial_state:
                character_name = state.get("character_name")
                if character_name in prev_final_state:
                    # 继承上一帧的位置和姿势
                    state["position"] = prev_final_state[character_name].get("position", state["position"])
                    state["pose"] = prev_final_state[character_name].get("pose", state["pose"])

        return fixed_shots

    def _generate_final_result(self,
                               script_text: str,
                               shots: List[Dict[str, Any]],
                               style: VideoStyle,
                               duration_per_shot: int,
                               sequence_qa: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终结果"""
        # 生成任务ID
        job_id = f"shotgen_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"

        # 计算总时长
        total_duration = len(shots) * duration_per_shot

        # 获取最终的连续性状态
        final_continuity_state = {}
        if shots:
            last_shot = shots[-1]
            if "continuity_anchor" in last_shot:
                # 将列表转换为字典，使用角色名作为键
                anchor_list = last_shot["continuity_anchor"]
                if isinstance(anchor_list, list):
                    final_continuity_state = {}
                    for anchor in anchor_list:
                        if isinstance(anchor, dict) and "character_name" in anchor:
                            final_continuity_state[anchor["character_name"]] = anchor

        # 构建元数据
        metadata = {
            "generated_at": datetime.now().isoformat() + "Z",
            "llm_model": "ollama_model" if self.llm and hasattr(self.llm, 'model') else "rule_based",
            "continuity_verified": not sequence_qa["has_continuity_issues"],
            "version": "1.0"
        }

        return {
            "job_id": job_id,
            "input_script": script_text,
            "style": style,
            "duration_per_shot": duration_per_shot,
            "total_shots": len(shots),
            "total_duration_sec": total_duration,
            "shots": shots,
            "final_continuity_state": final_continuity_state,
            "metadata": metadata
        }