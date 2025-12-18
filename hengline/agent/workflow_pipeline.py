# -*- coding: utf-8 -*-
"""
@FileName: multi_agent_pipeline.py
@Description: 多智能体协作流程，负责协调各个智能体完成端到端的分镜生成
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import uuid
from typing import Dict, List, Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from hengline.logger import debug, info, error
from .continuity_guardian_agent import ContinuityGuardianAgent
from .shot_qa_agent import QAAgent
from .script_parser_agent import ScriptParserAgent
from .shot_generator_agent import ShotGeneratorAgent
from .temporal_planner_agent import TemporalPlannerAgent
from .workflow_models import VideoStyle
from .workflow_nodes import WorkflowNodes
from .workflow_states import StoryboardWorkflowState


class MultiAgentPipeline:
    """多智能体协作流程"""

    def __init__(self, llm=None):
        """
        初始化多智能体流程
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        # 先初始化memory，确保在_init_workflow中可以使用
        self.memory = MemorySaver()
        self._init_agents()
        self.workflow = self._init_workflow()

    def _init_agents(self):
        """初始化各个智能体"""
        debug("初始化智能体组件")

        self.script_parser = ScriptParserAgent(llm=self.llm)
        self.temporal_planner = TemporalPlannerAgent()
        self.continuity_guardian = ContinuityGuardianAgent()
        self.shot_generator = ShotGeneratorAgent(llm=self.llm)
        self.shot_qa = QAAgent(llm=self.llm)

        # 初始化工作流节点集合
        self.workflow_nodes = WorkflowNodes(
            script_parser=self.script_parser,
            temporal_planner=self.temporal_planner,
            continuity_guardian=self.continuity_guardian,
            shot_generator=self.shot_generator,
            shot_qa=self.shot_qa,
            llm=self.llm
        )

    def _init_workflow(self):
        """初始化基于LangGraph的工作流"""
        debug("初始化LangGraph工作流")

        # 创建状态图
        workflow = StateGraph(StoryboardWorkflowState)

        # 定义工作流节点
        workflow.add_node("parse_script", lambda graph_state: self.workflow_nodes.parse_script_node(graph_state))
        workflow.add_node("plan_timeline", lambda graph_state: self.workflow_nodes.plan_timeline_node(graph_state))
        workflow.add_node("generate_shot", lambda graph_state: self.workflow_nodes.generate_shot_node(graph_state))
        workflow.add_node("review_shot", lambda graph_state: self.workflow_nodes.review_shot_node(graph_state))
        workflow.add_node("extract_continuity", lambda graph_state: self.workflow_nodes.extract_continuity_node(graph_state))
        workflow.add_node("review_sequence", lambda graph_state: self.workflow_nodes.review_sequence_node(graph_state))
        workflow.add_node("fix_continuity", lambda graph_state: self.workflow_nodes.fix_continuity_node(graph_state))
        workflow.add_node("generate_result", lambda graph_state: self.workflow_nodes.generate_result_node(graph_state))

        # 定义条件边，根据解析结果是否有效继续流程（工作流执行流程）
        workflow.add_conditional_edges(
            "parse_script",
            lambda graph_state: "continue",  # 始终继续到下一步
            {"continue": "plan_timeline"}
        )

        workflow.add_conditional_edges(
            "plan_timeline",
            lambda graph_state: "continue",
            {"continue": "generate_shot"}
        )

        workflow.add_conditional_edges(
            "generate_shot",
            lambda graph_state: "continue",
            {"continue": "review_shot"}
        )

        workflow.add_conditional_edges(
            "review_shot",
            lambda graph_state: "valid" if graph_state["qa_results"][-1]["is_valid"] else "invalid",
            {"valid": "extract_continuity", "invalid": "check_retry"}
        )

        # 添加检查重试逻辑的条件边
        workflow.add_conditional_edges(
            "extract_continuity",
            lambda graph_state: "next_segment" if graph_state["current_segment_index"] < len(graph_state["segments"]) else "review_sequence",
            {"next_segment": "generate_shot", "review_sequence": "review_sequence"}
        )

        # 添加自定义检查重试节点
        workflow.add_node("check_retry", lambda graph_state: self.workflow_nodes.check_retry_node(graph_state))
        workflow.add_conditional_edges(
            "check_retry",
            lambda graph_state: "retry" if graph_state["retry_count"] < graph_state["max_retries"] else "use_current",
            {"retry": "generate_shot", "use_current": "extract_continuity"}
        )

        workflow.add_conditional_edges(
            "review_sequence",
            lambda graph_state: "fix" if graph_state["sequence_qa"]["has_continuity_issues"] else "done",
            {"fix": "fix_continuity", "done": "generate_result"}
        )

        workflow.add_conditional_edges(
            "fix_continuity",
            lambda graph_state: "continue",
            {"continue": "generate_result"}
        )

        # 设置入口点
        workflow.set_entry_point("parse_script")

        # 编译工作流
        return workflow.compile(checkpointer=self.memory)

    def run_pipeline(self,
                     script_text: str,
                     style: VideoStyle = VideoStyle.REALISTIC,
                     duration_per_shot: int = 5,
                     task_id: str = None,
                     prev_continuity_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        运行完整的分镜生成流程
        
        Args:
            script_text: 原始剧本文本
            style: 视频风格
            duration_per_shot: 每段时长
            prev_continuity_state: 上一段的连续性状态
            task_id: 任务ID
            
        Returns:
            完整的分镜结果
        """
        info("开始运行分镜生成流程")

        try:
            # 重置连续性守护智能体的状态，确保每次处理新剧本时都是全新的状态
            self.continuity_guardian.reset_state()
            
            # 当更换剧本时，忽略传入的 prev_continuity_state
            # 这样可以确保使用全新的状态处理新剧本
            prev_continuity_state = None
            
            # 创建初始状态
            initial_state: StoryboardWorkflowState = {
                "script_text": script_text,
                "style": style.value,
                "task_id": task_id,
                "duration_per_shot": duration_per_shot,
                "prev_continuity_state": prev_continuity_state,
                "structured_script": None,
                "segments": None,
                "shots": [],
                "current_continuity_state": prev_continuity_state,
                "current_segment_index": 0,
                "retry_count": 0,
                "max_retries": 2,
                "qa_results": [],
                "sequence_qa": None,
                "result": None,
                "error": None
            }

            # 使用LangGraph运行工作流
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            result = self.workflow.invoke(initial_state, config)

            # 返回最终结果
            if result.get("result"):
                info("分镜生成流程完成")
                return result["result"]
            else:
                error("工作流未生成有效结果")
                return {
                    "error": result.get("error", "未知错误"),
                    "status": "failed",
                    "shots": result.get("shots", []),
                    "final_continuity_state": result.get("current_continuity_state", prev_continuity_state),
                    "total_duration": len(result.get("shots", [])) * duration_per_shot
                }

        except Exception as e:
            error(f"分镜生成流程失败: {str(e)}")
            # 返回错误响应
            return {
                "error": str(e),
                "status": "failed",
                "shots": [],
                "final_continuity_state": prev_continuity_state,
                "total_duration": 0
            }

    def _fix_continuity_issues(self, shots: List[Dict[str, Any]], qa_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """修复连续性问题"""
        # 委托给workflow_nodes处理
        return self.workflow_nodes._fix_continuity_issues(shots, qa_result)

    def _generate_final_result(self,
                               script_text: str,
                               shots: List[Dict[str, Any]],
                               style: VideoStyle,
                               duration_per_shot: int,
                               sequence_qa: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终结果"""
        # 委托给workflow_nodes处理
        return self.workflow_nodes._generate_final_result(script_text, shots, style, duration_per_shot, sequence_qa)

    def create_langgraph_workflow(self):
        """创建基于LangGraph的工作流（可选）"""
        # 已经在初始化时创建了工作流
        return self.workflow

    def get_workflow_state(self, thread_id: str) -> Optional[StoryboardWorkflowState]:
        """
        获取指定线程的工作流状态
        
        Args:
            thread_id: 线程ID
            
        Returns:
            工作流状态，如果不存在则返回None
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            # 从checkpointer获取状态
            for state in self.memory.list(config):
                return state
            return None
        except Exception as e:
            error(f"获取工作流状态失败: {str(e)}")
            return None

    def continue_workflow(self, thread_id: str, additional_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        继续执行现有工作流
        
        Args:
            thread_id: 线程ID
            additional_input: 额外的输入数据
            
        Returns:
            工作流执行结果
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            # 获取最新状态
            current_state = self.get_workflow_state(thread_id)

            if not current_state:
                raise ValueError(f"未找到线程ID为 {thread_id} 的工作流状态")

            # 更新状态
            if additional_input:
                current_state.update(additional_input)

            # 继续执行工作流
            result = self.workflow.invoke(current_state, config)
            return result.get("result", result)
        except Exception as e:
            error(f"继续工作流失败: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }