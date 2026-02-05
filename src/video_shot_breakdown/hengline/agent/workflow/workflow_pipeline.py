"""
@FileName: multi_agent_pipeline.py
@Description: 多智能体协作流程，负责协调各个智能体完成端到端的分镜生成
@Author: HengLine
@Github: https://github.com/HengLine/video-shot-agent
@Time: 2025/10 - 2025/11
"""
from typing import List, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from video_shot_breakdown.hengline.agent.script_parser_agent import ScriptParserAgent
from video_shot_breakdown.logger import debug
from .workflow_decision import DecisionFunctions
from .workflow_models import AgentStage, PipelineNode, DecisionState
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

        # ========== 剧本解析后的分支 ==========
        workflow.add_conditional_edges(
            PipelineNode.PARSE_SCRIPT,  # 当前节点：剧本解析
            lambda graph_state: self.decision_funcs.decide_after_parsing(graph_state),  # 决策函数：解析后判断
            {
                # 解析成功，进入镜头拆分阶段
                DecisionState.SUCCESS: PipelineNode.SEGMENT_SHOT,  # 下一步：拆分镜头

                # 解析遇到严重错误（如数据损坏、系统错误），进入错误处理
                DecisionState.CRITICAL_FAILURE: PipelineNode.ERROR_HANDLER,  # 下一步：错误处理

                # 解析结果需要人工判断（如剧本格式模糊、内容歧义）
                DecisionState.REQUIRE_HUMAN: PipelineNode.HUMAN_INTERVENTION  # 下一步：人工干预
            }
        )

        # ========== 镜头拆分后的分支 ==========
        workflow.add_conditional_edges(
            PipelineNode.SEGMENT_SHOT,  # 当前节点：镜头拆分
            lambda graph_state: self.decision_funcs.decide_after_splitting(graph_state),  # 决策函数：拆分后判断
            {
                # 拆分成功，进入AI分段阶段
                DecisionState.SUCCESS: PipelineNode.SPLIT_VIDEO,  # 下一步：AI分段

                # 拆分需要重试（如过长镜头过多、临时问题）
                DecisionState.SHOULD_RETRY: PipelineNode.SEGMENT_SHOT,  # 下一步：重试当前节点（镜头拆分）

                # 拆分需要调整后重试（如参数不合理）
                DecisionState.RETRY_WITH_ADJUSTMENT: PipelineNode.SEGMENT_SHOT,  # 下一步：调整后重试当前节点

                # 拆分遇到严重错误，进入错误处理
                DecisionState.CRITICAL_FAILURE: PipelineNode.ERROR_HANDLER  # 下一步：错误处理
            }
        )

        # ========== AI分段后的分支 ==========
        workflow.add_conditional_edges(
            PipelineNode.SPLIT_VIDEO,  # 当前节点：AI分段（片段切割）
            lambda graph_state: self.decision_funcs.decide_after_fragmenting(graph_state),  # 决策函数：分段后判断
            {
                # 分段成功，进入提示词生成阶段
                DecisionState.SUCCESS: PipelineNode.CONVERT_PROMPT,  # 下一步：生成提示词

                # 分段需要调整（如个别片段时长不合理）
                DecisionState.NEEDS_ADJUSTMENT: PipelineNode.SPLIT_VIDEO,  # 下一步：重试当前节点（AI分段）

                # 分段需要修复（如多个片段超时、结构性问题）
                DecisionState.NEEDS_FIX: PipelineNode.SPLIT_VIDEO,  # 下一步：重新生成片段

                # 分段遇到严重错误，进入错误处理
                DecisionState.CRITICAL_FAILURE: PipelineNode.ERROR_HANDLER,  # 下一步：错误处理

                # 分段结果需要人工判断
                DecisionState.REQUIRE_HUMAN: PipelineNode.HUMAN_INTERVENTION  # 下一步：人工干预
            }
        )

        # ========== Prompt生成后的分支 ==========
        # 问题：这里lambda函数返回PipelineNode，应该返回DecisionState
        # 修正：使用决策函数decide_after_prompts
        workflow.add_conditional_edges(
            PipelineNode.CONVERT_PROMPT,  # 当前节点：提示词生成
            lambda graph_state: self.decision_funcs.decide_after_prompts(graph_state),  # 决策函数：生成后判断
            {
                # 生成成功，进入质量审查阶段
                DecisionState.SUCCESS: PipelineNode.AUDIT_QUALITY,  # 下一步：质量审查

                # 生成通过验证，进入质量审查阶段
                DecisionState.VALID: PipelineNode.AUDIT_QUALITY,  # 下一步：质量审查

                # 生成需要调整（如提示词质量不高、有空提示词）
                DecisionState.NEEDS_ADJUSTMENT: PipelineNode.CONVERT_PROMPT,  # 下一步：调整提示词
                DecisionState.NEEDS_OPTIMIZATION: PipelineNode.CONVERT_PROMPT,  # 下一步：调整提示词

                # 生成遇到严重错误，进入错误处理
                DecisionState.CRITICAL_FAILURE: PipelineNode.ERROR_HANDLER  # 下一步：错误处理
            }
        )

        # ========== 质量审查后的分支 ==========
        workflow.add_conditional_edges(
            PipelineNode.AUDIT_QUALITY,  # 当前节点：质量审查
            lambda graph_state: self.decision_funcs.decide_after_audit(graph_state),  # 决策函数：审查后判断
            {
                # 审查成功通过，进入连续性检查阶段
                DecisionState.SUCCESS: PipelineNode.CONTINUITY_CHECK,  # 下一步：连续性检查

                # 审查验证通过（有轻微问题），进入连续性检查阶段
                DecisionState.VALID: PipelineNode.CONTINUITY_CHECK,  # 下一步：连续性检查

                # 审查需要调整（如提示词需要优化）
                DecisionState.NEEDS_ADJUSTMENT: PipelineNode.CONVERT_PROMPT,  # 下一步：调整提示词

                # 审查需要修复（如片段问题需要重新生成）
                DecisionState.NEEDS_FIX: PipelineNode.SPLIT_VIDEO,  # 下一步：重新生成片段

                # 审查需要重试（如临时问题）
                DecisionState.SHOULD_RETRY: PipelineNode.CONVERT_PROMPT,  # 下一步：重试提示词生成

                # 审查需要调整后重试（如参数需要调整）
                DecisionState.RETRY_WITH_ADJUSTMENT: PipelineNode.CONVERT_PROMPT,  # 下一步：调整后重试

                # 审查需要人工判断（如发现严重不确定性问题）
                DecisionState.REQUIRE_HUMAN: PipelineNode.HUMAN_INTERVENTION,  # 下一步：人工干预

                # 审查失败（业务逻辑失败），进入错误处理
                DecisionState.FAILED: PipelineNode.ERROR_HANDLER,  # 下一步：错误处理

                # 审查遇到严重错误（系统错误），进入错误处理
                DecisionState.CRITICAL_FAILURE: PipelineNode.ERROR_HANDLER  # 下一步：错误处理
            }
        )

        # ========== 连续性检查后的分支 ==========
        workflow.add_conditional_edges(
            PipelineNode.CONTINUITY_CHECK,  # 当前节点：连续性检查
            lambda graph_state: self.decision_funcs.decide_after_continuity(graph_state),  # 决策函数：检查后判断
            {
                # 检查成功通过，进入生成输出阶段
                DecisionState.SUCCESS: PipelineNode.GENERATE_OUTPUT,  # 下一步：生成输出

                # 检查验证通过（有可接受问题），进入生成输出阶段
                DecisionState.VALID: PipelineNode.GENERATE_OUTPUT,  # 下一步：生成输出

                # 检查需要调整（如可修复的连续性问题）
                DecisionState.NEEDS_ADJUSTMENT: PipelineNode.CONVERT_PROMPT,  # 下一步：调整提示词

                # 检查需要修复（如结构性问题需要重新分段）
                DecisionState.NEEDS_FIX: PipelineNode.SPLIT_VIDEO,  # 下一步：调整AI分段

                # 检查需要优化（如连续性问题需要优化）
                DecisionState.NEEDS_OPTIMIZATION: PipelineNode.CONVERT_PROMPT,  # 下一步：调整提示词

                # 检查需要人工判断（如复杂的连续性问题）
                DecisionState.REQUIRE_HUMAN: PipelineNode.HUMAN_INTERVENTION,  # 下一步：人工干预

                # 检查遇到严重错误，进入错误处理
                DecisionState.CRITICAL_FAILURE: PipelineNode.ERROR_HANDLER  # 下一步：错误处理
            }
        )

        # ========== 错误处理后的分支 ==========
        # 问题：这里使用了PipelineState，应该使用DecisionState
        # 修正：使用DecisionState枚举值
        workflow.add_conditional_edges(
            PipelineNode.ERROR_HANDLER,  # 当前节点：错误处理
            lambda graph_state: self.decision_funcs.decide_after_error(graph_state),  # 决策函数：处理后判断
            {
                # 错误可恢复（如验证错误），重新开始流程
                DecisionState.VALID: PipelineNode.PARSE_SCRIPT,  # 下一步：重新开始（剧本解析）

                # 错误应该重试（如网络问题），重新开始流程
                DecisionState.SHOULD_RETRY: PipelineNode.PARSE_SCRIPT,  # 下一步：重新开始（剧本解析）

                # 错误需要人工处理（如多次重试失败）
                DecisionState.REQUIRE_HUMAN: PipelineNode.HUMAN_INTERVENTION,  # 下一步：人工干预

                # 错误需要中止流程（如用户取消、超时）
                DecisionState.ABORT_PROCESS: END  # 下一步：结束工作流
            }
        )

        # ========== 人工干预后的分支 ==========
        # 问题：这里使用了PipelineState，应该使用DecisionState
        # 修正：使用DecisionState枚举值
        workflow.add_conditional_edges(
            PipelineNode.HUMAN_INTERVENTION,  # 当前节点：人工干预
            lambda graph_state: self.decision_funcs.decide_after_human(graph_state),  # 决策函数：干预后判断
            {
                # 人工决定继续流程
                DecisionState.SUCCESS: PipelineNode.GENERATE_OUTPUT,
                DecisionState.VALID: PipelineNode.GENERATE_OUTPUT,

                # 人工要求重试
                DecisionState.SHOULD_RETRY: PipelineNode.PARSE_SCRIPT,

                DecisionState.NEEDS_ADJUSTMENT: PipelineNode.CONVERT_PROMPT,
                DecisionState.NEEDS_FIX: PipelineNode.SPLIT_VIDEO,
                DecisionState.NEEDS_OPTIMIZATION: PipelineNode.CONVERT_PROMPT,
                DecisionState.RETRY_WITH_ADJUSTMENT: PipelineNode.CONVERT_PROMPT,
                DecisionState.REQUIRE_HUMAN: PipelineNode.HUMAN_INTERVENTION,
                DecisionState.ABORT_PROCESS: END
            }
        )

        # ========== 结果生成后结束 ==========
        workflow.add_edge(PipelineNode.GENERATE_OUTPUT, END)  # 生成输出后直接结束

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
