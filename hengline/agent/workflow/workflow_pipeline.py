"""
@FileName: multi_agent_pipeline.py
@Description: 多智能体协作流程，负责协调各个智能体完成端到端的分镜生成
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from typing import List, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from hengline.agent.script_parser_agent import ScriptParserAgent
from hengline.logger import debug
from .workflow_decision import DecisionFunctions
from .workflow_models import AgentStage
from .workflow_nodes import WorkflowNodes
from .workflow_states import WorkflowState
from ..prompt_converter_agent import PromptConverterAgent
from ..quality_auditor_agent import QualityAuditorAgent
from ..shot_segmenter_agent import ShotSegmenterAgent
from ..video_splitter_agent import VideoSplitterAgent
from ...hengline_config import HengLineConfig


class MultiAgentPipeline:
    """多智能体协作流程"""

    def __init__(self, task_id, config: Optional[HengLineConfig]):
        """
        初始化多智能体流程
        
        Args:
            task_id: 任务ID
            config: 用户配置（LLM）
        """
        self.task_id = task_id
        self.memory = MemorySaver()  # 状态记忆器
        self.config = config or HengLineConfig()
        self.llm = self.config.get_llm_by_config()
        self._init_agents()
        self.workflow = self._build_workflow()

    def _init_agents(self):
        """初始化各个智能体"""
        debug("初始化智能体组件")

        self.script_parser = ScriptParserAgent(llm=self.llm, config=self.config)
        self.shot_segmenter = ShotSegmenterAgent(llm=self.llm, config=self.config)
        self.video_splitter = VideoSplitterAgent(llm=self.llm, config=self.config)
        self.prompt_converter = PromptConverterAgent(llm=self.llm, config=self.config)
        self.quality_auditor = QualityAuditorAgent(llm=self.llm, config=self.config)

        # 初始化工作流节点集合
        self.workflow_nodes = WorkflowNodes(
            script_parser=self.script_parser,
            shot_segmenter=self.shot_segmenter,
            video_splitter=self.video_splitter,
            prompt_converter=self.prompt_converter,
            quality_auditor=self.quality_auditor,
            llm=self.llm
        )
        self.decision_funcs = DecisionFunctions()

    def _build_workflow(self):
        """初始化基于LangGraph的工作流"""
        debug("初始化LangGraph工作流")

        # 创建状态图
        workflow = StateGraph(WorkflowState)

        # ========== 定义所有工作流节点 ==========
        workflow.add_node("parse_script",
                          lambda graph_state: self.workflow_nodes.parse_script_node(graph_state))

        workflow.add_node("split_shots",
                          lambda graph_state: self.workflow_nodes.split_shots_node(graph_state))

        workflow.add_node("fragment_for_ai",
                          lambda graph_state: self.workflow_nodes.fragment_for_ai_node(graph_state))

        workflow.add_node("generate_prompts",
                          lambda graph_state: self.workflow_nodes.generate_prompts_node(graph_state))

        workflow.add_node("quality_audit",
                          lambda graph_state: self.workflow_nodes.quality_audit_node(graph_state))

        workflow.add_node("continuity_check",
                          lambda graph_state: self.workflow_nodes.continuity_check_node(graph_state))

        workflow.add_node("error_handler",
                          lambda graph_state: self.workflow_nodes.error_handler_node(graph_state))

        workflow.add_node("generate_output",
                          lambda graph_state: self.workflow_nodes.generate_output_node(graph_state))

        workflow.add_node("human_intervention",
                          lambda graph_state: self.workflow_nodes.human_intervention_node(graph_state))

        # ========== 定义工作流执行流程 ==========

        # 设置入口点
        workflow.set_entry_point("parse_script")

        # 剧本解析后的分支
        workflow.add_conditional_edges(
            "parse_script",
            lambda graph_state: self.decision_funcs.decide_after_parsing(graph_state),
            {
                "success": "split_shots",
                "critical_failure": "error_handler",
                "needs_human": "human_intervention"
            }
        )

        # 镜头拆分后的分支
        workflow.add_conditional_edges(
            "split_shots",
            lambda graph_state: self.decision_funcs.decide_after_splitting(graph_state),
            {
                "success": "fragment_for_ai",
                "retry": "split_shots",  # 重试当前节点
                "critical_failure": "error_handler"
            }
        )

        # AI分段后的分支
        workflow.add_conditional_edges(
            "fragment_for_ai",
            lambda graph_state: self.decision_funcs.decide_after_fragmenting(graph_state),
            {
                "success": "generate_prompts",
                "needs_adjustment": "split_shots",  # 返回调整镜头
                "critical_failure": "error_handler"
            }
        )

        # Prompt生成后的分支（总是进入质量审查）
        workflow.add_conditional_edges(
            "generate_prompts",
            lambda graph_state: "quality_audit",
            {"quality_audit": "quality_audit"}
        )

        # 质量审查后的分支
        workflow.add_conditional_edges(
            "quality_audit",
            lambda graph_state: self.decision_funcs.decide_after_audit(graph_state),
            {
                "continuity_check": "continuity_check",
                "generate_prompts": "generate_prompts",  # 重新生成Prompt
                "split_shots": "split_shots",  # 重新拆分镜头
                "human_intervention": "human_intervention"
            }
        )

        # 连续性检查后的分支
        workflow.add_conditional_edges(
            "continuity_check",
            lambda graph_state: self.decision_funcs.decide_after_continuity(graph_state),
            {
                "passed": "generate_output",
                "fixable_issues": "generate_prompts",  # 调整Prompt修复连续性
                "structural_issues": "fragment_for_ai",  # 调整分段
                "needs_human": "human_intervention"
            }
        )

        # 错误处理后的分支
        workflow.add_conditional_edges(
            "error_handler",
            lambda graph_state: self.decision_funcs.decide_after_error(graph_state),
            {
                "recoverable": "parse_script",  # 从开头重试
                "needs_human": "human_intervention",
                "abort": END
            }
        )

        # 人工干预后的分支
        workflow.add_conditional_edges(
            "human_intervention",
            lambda graph_state: self.decision_funcs.decide_after_human(graph_state),
            {
                "continue": "generate_output",  # 继续流程
                "retry": "parse_script",  # 重新开始
                "abort": END  # 中止流程
            }
        )

        # 结果生成后结束
        workflow.add_edge("generate_output", END)

        return workflow.compile(checkpointer=self.memory)

        # ========== 公开接口 ==========

    async def run_process(self, raw_script: str, config: HengLineConfig) -> Dict:
        """执行完整的工作流"""
        initial_state = WorkflowState(
            raw_script=raw_script,
            user_config=config or {},
            task_id=self.task_id,
            needs_human_review=False,
            human_feedback={},
            max_retries=config.max_retries,
            current_stage=AgentStage.INIT,
        )

        try:
            # 执行工作流
            final_state = await self.workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": f"process_{id(raw_script)}"}}
            )

            return {
                "success": final_state.final_output is not None,
                "data": final_state.final_output,
                "errors": final_state.error_messages,
                "processing_stats": {
                    "retry_count": final_state.retry_count,
                    "stages_completed": self._get_completed_stages(final_state)
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None
            }

    def _get_completed_stages(self, state: WorkflowState) -> List[AgentStage]:
        """获取已完成的阶段列表"""
        stages = []
        if state.parsed_script:
            stages.append(AgentStage.PARSER)
        if state.shot_sequence:
            stages.append(AgentStage.SEGMENTER)
        if state.fragment_sequence:
            stages.append(AgentStage.SPLITTER)
        if state.instructions:
            stages.append(AgentStage.CONVERTER)
        if state.audit_report:
            stages.append(AgentStage.AUDITOR)
        if state.continuity_issues is not None:
            stages.append(AgentStage.CONTINUITY)
        return stages
