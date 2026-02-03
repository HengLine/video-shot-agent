"""
@FileName: multi_agent_pipeline.py
@Description: 多智能体协作流程，负责协调各个智能体完成端到端的分镜生成
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from typing import List, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from video_shot_breakdown.hengline.agent.script_parser_agent import ScriptParserAgent
from video_shot_breakdown.logger import debug
from .workflow_decision import DecisionFunctions
from .workflow_models import AgentStage, PipelineState, PipelineNode
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
        workflow.add_node(PipelineNode.PARSE_SCRIPT,
                          lambda graph_state: self.workflow_nodes.parse_script_node(graph_state))

        workflow.add_node(PipelineNode.SEGMENT_SHOT,
                          lambda graph_state: self.workflow_nodes.split_shots_node(graph_state))

        workflow.add_node(PipelineNode.SPLIT_VIDEO,
                          lambda graph_state: self.workflow_nodes.fragment_for_ai_node(graph_state))

        workflow.add_node(PipelineNode.CONVERT_PROMPT,
                          lambda graph_state: self.workflow_nodes.generate_prompts_node(graph_state))

        workflow.add_node(PipelineNode.AUDIT_QUALITY,
                          lambda graph_state: self.workflow_nodes.quality_audit_node(graph_state))

        workflow.add_node(PipelineNode.CONTINUITY_CHECK,
                          lambda graph_state: self.workflow_nodes.continuity_check_node(graph_state))

        workflow.add_node(PipelineNode.ERROR_HANDLER,
                          lambda graph_state: self.workflow_nodes.error_handler_node(graph_state))

        workflow.add_node(PipelineNode.GENERATE_OUTPUT,
                          lambda graph_state: self.workflow_nodes.generate_output_node(graph_state))

        workflow.add_node(PipelineNode.HUMAN_INTERVENTION,
                          lambda graph_state: self.workflow_nodes.human_intervention_node(graph_state))

        # ========== 定义工作流执行流程 ==========

        # 设置入口点
        workflow.set_entry_point(PipelineNode.PARSE_SCRIPT)

        # 剧本解析后的分支
        workflow.add_conditional_edges(
            PipelineNode.PARSE_SCRIPT,
            lambda graph_state: self.decision_funcs.decide_after_parsing(graph_state),
            {
                PipelineState.SUCCESS: PipelineNode.SEGMENT_SHOT,
                PipelineState.CRITICAL_FAILURE: PipelineNode.ERROR_HANDLER,
                PipelineState.NEEDS_HUMAN: PipelineNode.HUMAN_INTERVENTION
            }
        )

        # 镜头拆分后的分支
        workflow.add_conditional_edges(
            PipelineNode.SEGMENT_SHOT,
            lambda graph_state: self.decision_funcs.decide_after_splitting(graph_state),
            {
                PipelineState.SUCCESS: PipelineNode.SPLIT_VIDEO,
                PipelineState.RETRY: PipelineNode.SEGMENT_SHOT,  # 重试当前节点
                PipelineState.CRITICAL_FAILURE: PipelineNode.ERROR_HANDLER
            }
        )

        # AI分段后的分支
        workflow.add_conditional_edges(
            PipelineNode.SPLIT_VIDEO,
            lambda graph_state: self.decision_funcs.decide_after_fragmenting(graph_state),
            {
                PipelineState.SUCCESS: PipelineNode.CONVERT_PROMPT,
                PipelineState.NEEDS_ADJUSTMENT: PipelineNode.SEGMENT_SHOT,  # 返回调整镜头
                PipelineState.CRITICAL_FAILURE: PipelineNode.ERROR_HANDLER
            }
        )

        # Prompt生成后的分支（总是进入质量审查）
        workflow.add_conditional_edges(
            PipelineNode.CONVERT_PROMPT,
            lambda graph_state: PipelineNode.AUDIT_QUALITY,
            {PipelineNode.AUDIT_QUALITY: PipelineNode.AUDIT_QUALITY}
        )

        # 质量审查后的分支
        workflow.add_conditional_edges(
            PipelineNode.AUDIT_QUALITY,
            lambda graph_state: self.decision_funcs.decide_after_audit(graph_state),
            {
                PipelineNode.CONTINUITY_CHECK: PipelineNode.CONTINUITY_CHECK,
                PipelineNode.CONVERT_PROMPT: PipelineNode.CONVERT_PROMPT,  # 重新生成Prompt
                PipelineNode.SEGMENT_SHOT: PipelineNode.SEGMENT_SHOT,  # 重新拆分镜头
                PipelineNode.HUMAN_INTERVENTION: PipelineNode.HUMAN_INTERVENTION
            }
        )

        # 连续性检查后的分支
        workflow.add_conditional_edges(
            PipelineNode.CONTINUITY_CHECK,
            lambda graph_state: self.decision_funcs.decide_after_continuity(graph_state),
            {
                PipelineState.SUCCESS: PipelineNode.GENERATE_OUTPUT,
                PipelineState.FIXABLE_ISSUES: PipelineNode.CONVERT_PROMPT,  # 调整Prompt修复连续性
                PipelineState.STRUCTURAL_ISSUES: PipelineNode.SPLIT_VIDEO,  # 调整分段
                PipelineState.NEEDS_HUMAN: PipelineNode.HUMAN_INTERVENTION
            }
        )

        # 错误处理后的分支
        workflow.add_conditional_edges(
            PipelineNode.ERROR_HANDLER,
            lambda graph_state: self.decision_funcs.decide_after_error(graph_state),
            {
                PipelineState.RECOVERABLE: PipelineNode.PARSE_SCRIPT,  # 从开头重试
                PipelineState.NEEDS_HUMAN: PipelineNode.HUMAN_INTERVENTION,
                PipelineState.ABORT: END
            }
        )

        # 人工干预后的分支
        workflow.add_conditional_edges(
            PipelineNode.HUMAN_INTERVENTION,
            lambda graph_state: self.decision_funcs.decide_after_human(graph_state),
            {
                PipelineState.CONTINUE: PipelineNode.GENERATE_OUTPUT,  # 继续流程
                PipelineState.RETRY: PipelineNode.PARSE_SCRIPT,  # 重新开始
                PipelineState.ABORT: END  # 中止流程
            }
        )

        # 结果生成后结束
        workflow.add_edge(PipelineNode.GENERATE_OUTPUT, END)

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
                PipelineState.SUCCESS: final_state.final_output is not None,
                "data": final_state.final_output,
                "errors": final_state.error_messages,
                "processing_stats": {
                    "retry_count": final_state.retry_count,
                    "stages_completed": self._get_completed_stages(final_state)
                }
            }

        except Exception as e:
            return {
                PipelineState.SUCCESS: False,
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
