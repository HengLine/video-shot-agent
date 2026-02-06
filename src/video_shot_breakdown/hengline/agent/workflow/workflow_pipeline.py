"""
@FileName: multi_agent_pipeline.py
@Description: 多智能体协作流程，负责协调各个智能体完成端到端的分镜生成
@Author: HengLine
@Github: https://github.com/HengLine/video-shot-agent
@Time: 2025/10 - 至今
"""
from typing import Dict, Optional, Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from video_shot_breakdown.hengline.agent.script_parser_agent import ScriptParserAgent
from video_shot_breakdown.logger import debug, error
from .workflow_decision import PipelineDecision
from .workflow_models import AgentStage, PipelineNode, PipelineState
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
        self.decision_funcs = PipelineDecision()

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

        # 添加循环检查节点
        workflow.add_node(PipelineNode.LOOP_CHECK,
                          lambda graph_state: self.workflow_nodes.loop_check_node(graph_state))

        # ========== 定义工作流执行流程 ==========

        # 设置入口点
        workflow.set_entry_point(PipelineNode.PARSE_SCRIPT)

        workflow.add_conditional_edges(
            PipelineNode.PARSE_SCRIPT,
            lambda graph_state: self.decision_funcs.decide_after_parsing(graph_state),
            {
                # 解析成功，进入镜头拆分阶段
                PipelineState.SUCCESS: PipelineNode.SEGMENT_SHOT,

                # 解析需要人工干预
                PipelineState.NEEDS_HUMAN: PipelineNode.HUMAN_INTERVENTION,

                # 解析失败
                PipelineState.FAILED: PipelineNode.ERROR_HANDLER
            }
        )

        # ========== 镜头拆分后的分支（保持原有逻辑，但能检测循环） ==========
        workflow.add_conditional_edges(
            PipelineNode.SEGMENT_SHOT,  # 当前节点：镜头拆分
            lambda graph_state: self.decision_funcs.decide_after_splitting(graph_state),  # 决策函数：拆分后判断
            {
                # 拆分成功，进入AI分段阶段（通过循环检查节点）
                PipelineState.SUCCESS: PipelineNode.LOOP_CHECK,  # 下一步：循环检查 -> AI分段

                # 拆分需要重试（如过长镜头过多、临时问题）
                PipelineState.RETRY: PipelineNode.SEGMENT_SHOT,  # 下一步：重试当前节点（镜头拆分）

                # 拆分需要修复/调整（如参数不合理）
                PipelineState.NEEDS_REPAIR: PipelineNode.SEGMENT_SHOT,  # 下一步：修复后重试当前节点

                # 拆分遇到严重错误，进入错误处理
                PipelineState.FAILED: PipelineNode.ERROR_HANDLER  # 下一步：错误处理
            }
        )

        # ========== AI分段后的分支 ==========
        workflow.add_conditional_edges(
            PipelineNode.SPLIT_VIDEO,  # 当前节点：AI分段（片段切割）
            lambda graph_state: self.decision_funcs.decide_after_fragmenting(graph_state),  # 决策函数：分段后判断
            {
                # 分段成功，进入提示词生成阶段（通过循环检查节点）
                PipelineState.SUCCESS: PipelineNode.LOOP_CHECK,  # 下一步：循环检查 -> 提示词生成

                # 分段需要修复/调整（如片段时长不合理）
                PipelineState.NEEDS_REPAIR: PipelineNode.SPLIT_VIDEO,  # 下一步：修复后重试当前节点（AI分段）

                # 分段需要重试（如临时问题）
                PipelineState.RETRY: PipelineNode.SPLIT_VIDEO,  # 下一步：重试当前节点

                # 分段遇到严重错误，进入错误处理
                PipelineState.FAILED: PipelineNode.ERROR_HANDLER,  # 下一步：错误处理

                # 分段结果需要人工判断
                PipelineState.NEEDS_HUMAN: PipelineNode.HUMAN_INTERVENTION  # 下一步：人工干预
            }
        )

        # ========== Prompt生成后的分支 ==========
        workflow.add_conditional_edges(
            PipelineNode.CONVERT_PROMPT,  # 当前节点：提示词生成
            lambda graph_state: self.decision_funcs.decide_after_prompts(graph_state),  # 决策函数：生成后判断
            {
                # 生成成功，进入质量审查阶段（通过循环检查节点）
                PipelineState.SUCCESS: PipelineNode.LOOP_CHECK,  # 下一步：循环检查 -> 质量审查

                # 生成验证通过（有小问题），进入质量审查阶段
                PipelineState.VALID: PipelineNode.LOOP_CHECK,  # 下一步：循环检查 -> 质量审查

                # 生成需要修复/调整（如提示词质量不高、有空提示词）
                PipelineState.NEEDS_REPAIR: PipelineNode.CONVERT_PROMPT,  # 下一步：修复提示词

                # 生成需要重试（如临时问题）
                PipelineState.RETRY: PipelineNode.CONVERT_PROMPT,  # 下一步：重试提示词生成

                # 生成遇到严重错误，进入错误处理
                PipelineState.FAILED: PipelineNode.ERROR_HANDLER  # 下一步：错误处理
            }
        )

        # ========== 循环检查节点的分支 ==========
        workflow.add_conditional_edges(
            PipelineNode.LOOP_CHECK,
            lambda graph_state: self.decision_funcs.decide_after_loop_check(graph_state),
            {
                # 根据阶段决定下一个节点
                PipelineNode.SEGMENT_SHOT: PipelineNode.SEGMENT_SHOT,  # 继续到镜头拆分
                PipelineNode.SPLIT_VIDEO: PipelineNode.SPLIT_VIDEO,  # 继续到AI分段
                PipelineNode.CONVERT_PROMPT: PipelineNode.CONVERT_PROMPT,  # 继续到提示词生成
                PipelineNode.AUDIT_QUALITY: PipelineNode.AUDIT_QUALITY,  # 继续到质量审查

                # 特殊状态处理
                PipelineState.FAILED: PipelineNode.ERROR_HANDLER,  # 循环超限，错误处理
                PipelineState.NEEDS_HUMAN: PipelineNode.HUMAN_INTERVENTION,  # 需要人工干预
            }
        )

        # ========== 质量审查后的分支 ==========
        workflow.add_conditional_edges(
            PipelineNode.AUDIT_QUALITY,  # 当前节点：质量审查
            lambda graph_state: self.decision_funcs.decide_after_audit(graph_state),  # 决策函数：审查后判断
            {
                # 审查成功通过，进入连续性检查阶段
                PipelineState.SUCCESS: PipelineNode.CONTINUITY_CHECK,  # 下一步：连续性检查

                # 审查验证通过（有轻微问题），进入连续性检查阶段
                PipelineState.VALID: PipelineNode.CONTINUITY_CHECK,  # 下一步：连续性检查

                # 审查需要修复/调整（如提示词需要优化）
                PipelineState.NEEDS_REPAIR: PipelineNode.CONVERT_PROMPT,  # 下一步：修复提示词

                # 审查需要重试（如临时问题）
                PipelineState.RETRY: PipelineNode.CONVERT_PROMPT,  # 下一步：重试提示词生成

                # 审查需要人工判断（如发现严重不确定性问题）
                PipelineState.NEEDS_HUMAN: PipelineNode.HUMAN_INTERVENTION,  # 下一步：人工干预

                # 审查失败（业务逻辑失败或系统错误），进入错误处理
                PipelineState.FAILED: PipelineNode.ERROR_HANDLER,  # 下一步：错误处理
            }
        )

        # ========== 连续性检查后的分支 ==========
        workflow.add_conditional_edges(
            PipelineNode.CONTINUITY_CHECK,  # 当前节点：连续性检查
            lambda graph_state: self.decision_funcs.decide_after_continuity(graph_state),  # 决策函数：检查后判断
            {
                # 检查成功通过，进入生成输出阶段
                PipelineState.SUCCESS: PipelineNode.GENERATE_OUTPUT,  # 下一步：生成输出

                # 检查验证通过（有可接受问题），进入生成输出阶段
                PipelineState.VALID: PipelineNode.GENERATE_OUTPUT,  # 下一步：生成输出

                # 检查需要修复/调整（如可修复的连续性问题）
                PipelineState.NEEDS_REPAIR: PipelineNode.CONVERT_PROMPT,  # 下一步：修复提示词

                # 检查需要人工判断（如复杂的连续性问题）
                PipelineState.NEEDS_HUMAN: PipelineNode.HUMAN_INTERVENTION,  # 下一步：人工干预

                # 检查遇到严重错误，进入错误处理
                PipelineState.FAILED: PipelineNode.ERROR_HANDLER  # 下一步：错误处理
            }
        )

        # ========== 错误处理后的分支 ==========
        workflow.add_conditional_edges(
            PipelineNode.ERROR_HANDLER,  # 当前节点：错误处理
            lambda graph_state: self.decision_funcs.decide_next_after_error(graph_state),  # 决策函数：处理后判断
            # {
            #     # 错误可恢复（如验证错误），根据错误来源决定重试节点
            #     PipelineState.VALID: self.decision_funcs.decide_retry_node_based_on_error_source,  # 根据错误来源决定
            #
            #     # 错误应该重试（如网络问题），根据错误来源决定重试节点
            #     PipelineState.RETRY: self.decision_funcs.decide_retry_node_based_on_error_source,  # 根据错误来源决定
            #
            #     # 错误需要修复/调整（如参数问题）
            #     PipelineState.NEEDS_REPAIR: PipelineNode.CONVERT_PROMPT,  # 根据错误来源决定
            #
            #     # 错误需要人工处理（如多次重试失败）
            #     PipelineState.NEEDS_HUMAN: PipelineNode.HUMAN_INTERVENTION,
            #
            #     # 错误需要中止流程（如用户取消、超时）
            #     PipelineState.ABORT: END
            # }
        )

        # ========== 人工干预后的分支 ==========
        workflow.add_conditional_edges(
            PipelineNode.HUMAN_INTERVENTION,  # 当前节点：人工干预
            lambda graph_state: self.decision_funcs.decide_after_human(graph_state),  # 决策函数：干预后判断
            {
                # 人工决定继续流程
                PipelineState.SUCCESS: PipelineNode.GENERATE_OUTPUT,
                PipelineState.VALID: PipelineNode.GENERATE_OUTPUT,

                # 人工要求重试
                PipelineState.RETRY: PipelineNode.PARSE_SCRIPT,

                # 人工要求修复/调整
                PipelineState.NEEDS_REPAIR: PipelineNode.CONVERT_PROMPT,

                # 人工需要进一步干预
                PipelineState.NEEDS_HUMAN: PipelineNode.HUMAN_INTERVENTION,

                # 人工决定中止流程
                PipelineState.ABORT: END
            }
        )

        # ========== 结果生成后结束 ==========
        workflow.add_edge(PipelineNode.GENERATE_OUTPUT, END)  # 生成输出后直接结束

        return workflow.compile(checkpointer=self.memory)

        # ========== 公开接口 ==========

    async def run_process(self, raw_script: str, config: HengLineConfig) -> Dict:
        """执行完整的工作流"""
        # 计算全局循环限制：所有节点最大循环次数之和
        total_node_max_loops = config.max_total_loops
        default_global_max_loops = total_node_max_loops * 2  # 2倍安全系数

        initial_state = WorkflowState(
            raw_script=raw_script,
            user_config=config or {},
            task_id=self.task_id,
            current_stage=AgentStage.INIT,
            # 镜头及片段参数
            max_shot_duration=config.max_shot_duration,
            min_shot_duration=config.min_shot_duration,
            max_fragment_duration=config.max_fragment_duration,
            min_fragment_duration=config.min_fragment_duration,
            max_prompt_length=config.max_prompt_length,
            min_prompt_length=config.min_prompt_length,
            # 节点循环控制
            # node_max_loops=node_max_loops,
            # 阶段重试控制
            # stage_max_retries=stage_max_retries,

            # 全局循环控制
            global_max_loops=getattr(config, 'global_max_loops', default_global_max_loops),
            global_loop_exceeded=config.global_loop_exceeded,

            # 其他
            loop_warning_issued=config.loop_warning_issued,
            current_node=PipelineNode.PARSE_SCRIPT,
        )

        try:
            # 执行工作流
            final_state = await self.workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": f"process_{id(raw_script)}"}}
            )

            success = False
            data = None
            errors = []
            processing_stats = {}

            # 检查 final_state 的类型
            if isinstance(final_state, dict):
                # 如果是字典，尝试从中提取
                debug("final_state 是字典类型")
                success = final_state.get("success", False)
                data = final_state.get("data")
                errors = final_state.get("errors", [])

                # 尝试提取状态信息
                state_obj = final_state.get("state")
                if state_obj:
                    processing_stats = self._get_completed_stages(state_obj)
            elif hasattr(final_state, 'final_output'):
                # 如果是 WorkflowState 对象
                debug("final_state 是 WorkflowState 对象")
                success = final_state.final_output is not None
                data = final_state.final_output
                errors = final_state.error_messages if hasattr(final_state, 'error_messages') else []
                processing_stats = self._get_completed_stages(final_state)
            else:
                # 未知类型
                debug(f"final_state 是未知类型: {type(final_state)}")
                # 尝试将其视为字典访问
                try:
                    success = getattr(final_state, 'success', False)
                    data = getattr(final_state, 'data', None)
                    errors = getattr(final_state, 'errors', [])
                    processing_stats = self._get_completed_stages(final_state)
                except:
                    # 如果失败，创建默认响应
                    success = False
                    data = None
                    errors = ["无法解析工作流结果"]
                    processing_stats = {"error": "unknown_state_type"}

            return {
                "success": success,
                "data": data,
                "errors": errors,
                "processing_stats": processing_stats,
                "task_id": self.task_id
            }

        except Exception as e:
            error(f"执行工作流时出错: {str(e)}")
            import traceback
            traceback.print_exc()

            return {
                "success": False,
                "error": str(e),
                "data": None,
                "processing_stats": {
                    "error": "workflow_exception",
                    "exception": str(e),
                    "task_id": self.task_id
                },
                "task_id": self.task_id
            }

    def _get_completed_stages(self, state: WorkflowState) -> dict[str, Any]:
        """获取已完成的阶段列表和统计信息（简化版）"""
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

        # 计算各节点的剩余循环次数
        node_loop_summary = {}
        for node, max_loops in state.node_max_loops.items():
            current_loops = state.node_current_loops.get(node, 0)
            exceeded = state.node_loop_exceeded.get(node, False)

            node_loop_summary[node.value] = {
                "current": current_loops,
                "max": max_loops,
                "remaining": max(0, max_loops - current_loops),
                "exceeded": exceeded
            }

        # 计算各阶段的剩余重试次数
        stage_retry_summary = {}
        for node, max_retries in state.stage_max_retries.items():
            current_retries = state.stage_current_retries.get(node, 0)

            stage_retry_summary[node.value] = {
                "current": current_retries,
                "max": max_retries,
                "remaining": max(0, max_retries - current_retries)
            }

        # 综合统计信息
        stage_info = {
            "completed_stages": stages,
            "stage_count": len(stages),

            # 节点循环统计
            "node_loops": node_loop_summary,
            "total_node_loops": sum(state.node_current_loops.values()),

            # 阶段重试统计
            "stage_retries": stage_retry_summary,
            "total_retries": state.total_retries,

            # 全局循环统计
            "global_loops": {
                "current": state.global_current_loops,
                "max": state.global_max_loops,
                "remaining": max(0, state.global_max_loops - state.global_current_loops),
                "exceeded": state.global_loop_exceeded
            },

            # 当前状态
            "current_node": state.current_node.value if state.current_node else None,
            "last_node": state.last_node.value if state.last_node else None,
            "has_errors": len(state.error_messages) > 0 if state.error_messages else False,
            "error_count": len(state.error_messages) if state.error_messages else 0,
        }

        return stage_info
