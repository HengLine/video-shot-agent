# -*- coding: utf-8 -*-
"""
@FileName: workflow_nodes.py
@Description: LangGraph工作流节点实现，包含所有工作流执行功能
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import datetime

from video_shot_breakdown.hengline.agent.workflow.workflow_models import AgentStage
from video_shot_breakdown.hengline.agent.workflow.workflow_states import WorkflowState
from video_shot_breakdown.hengline.logger import error, debug
from video_shot_breakdown.hengline.tools.result_storage_tool import create_result_storage
from video_shot_breakdown.utils.log_utils import print_log_exception


class WorkflowNodes:
    """工作流节点集合，封装所有工作流执行功能"""

    def __init__(self, script_parser, shot_segmenter, video_splitter, prompt_converter, quality_auditor, llm=None):
        """
        初始化工作流节点集合
        
        Args:
            script_parser: 剧本解析器实例
            shot_segmenter: 分镜生成器实例
            video_splitter: 视频分割
            prompt_converter: 提示词转换
            quality_auditor: 质量审查实例
            llm: 语言模型实例（可选）
        """
        self.script_parser = script_parser
        self.shot_segmenter = shot_segmenter
        self.video_splitter = video_splitter
        self.prompt_converter = prompt_converter
        self.quality_auditor = quality_auditor

        self.llm = llm
        self.storage = create_result_storage()

    def parse_script_node(self, state: WorkflowState) -> WorkflowState:
        """
        剧本解析节点
        功能：将原始剧本解析为结构化元素序列
        输入：raw_script
        输出：parsed_script (包含顺序保持的元素列表)
        """
        try:
            parsed_script = self.script_parser.parser_process(state.raw_script)
            debug(f"剧本解析完成，场景数: {len(parsed_script.scenes)}")

            # 保存剧本解析结果
            self.storage.save_obj_result(state.task_id, parsed_script, "script_parser_result.json")

            state.parsed_script = parsed_script
            state.current_stage = AgentStage.PARSER

        except Exception as e:
            print_log_exception()
            error(f"剧本解析节点异常: {str(e)}")
            state.error = str(e)
            state.error_messages.append(str(e))

        return state

    def split_shots_node(self, state: WorkflowState) -> WorkflowState:
        """
        镜头拆分节点
        功能：将结构化剧本拆分为视觉镜头
        输入：parsed_script
        输出：shots (带时间戳的镜头序列)
        """
        try:
            shot_sequence = self.shot_segmenter.shot_process(state.parsed_script)
            debug(f"分镜解析完成，镜头数: {len(shot_sequence.shots)}")

            # 保存剧本解析结果
            self.storage.save_obj_result(state.task_id, shot_sequence, "shot_segmenter_result.json")

            state.shot_sequence = shot_sequence
            state.current_stage = AgentStage.SEGMENTER

        except Exception as e:
            print_log_exception()
            error(f"分镜解析节点异常: {str(e)}")
            state.error = str(e)
            state.error_messages.append(str(e))

        return state

    def fragment_for_ai_node(self, state: WorkflowState) -> WorkflowState:
        """
        AI分段节点
        功能：将镜头按5秒限制切分为AI可处理的片段
        输入：shots
        输出：fragments (符合5秒限制的片段序列)
        """
        # 1. 检查镜头时长，>5秒的进行切分
        # 2. <2秒的考虑合并
        # 3. 在动作边界自然切分
        # 4. 生成片段级连续性锚点
        try:
            fragment_sequence = self.video_splitter.video_process(state.shot_sequence)
            debug(f"视频分段完成，视频片段数: {len(fragment_sequence.fragments)}")

            # 保存剧本解析结果
            self.storage.save_obj_result(state.task_id, fragment_sequence, "video_splitter_result.json")

            state.fragment_sequence = fragment_sequence
            state.current_stage = AgentStage.SPLITTER

        except Exception as e:
            print_log_exception()
            error(f"视频分段异常: {str(e)}")
            state.error = str(e)
            state.error_messages.append(str(e))

        return state

    def generate_prompts_node(self, state: WorkflowState) -> WorkflowState:
        """
        Prompt生成节点
        功能：为每个片段生成AI视频生成提示词
        输入：fragments
        输出：ai_instructions (包含Prompt和技术参数)
        """
        # 1. 选择AI视频模型模板
        # 2. 使用LLM优化视觉描述
        # 3. 嵌入连续性约束
        # 4. 生成技术参数

        try:
            instructions = self.prompt_converter.prompt_process(state.fragment_sequence)
            debug(f"片段指令转换完成，指令片段数: {len(instructions.fragments)}")

            # 保存剧本解析结果
            self.storage.save_obj_result(state.task_id, instructions, "prompt_converter_result.json")

            state.instructions = instructions
            state.current_stage = AgentStage.CONVERTER

        except Exception as e:
            print_log_exception()
            error(f"片段指令转换异常: {str(e)}")
            state.error = str(e)
            state.error_messages.append(str(e))

        return state

    def quality_audit_node(self, state: WorkflowState) -> WorkflowState:
        """
        质量审查节点
        功能：检查输出质量，包括时长、连贯性等
        输入：ai_instructions
        输出：audit_report (审查报告和建议)
        """
        # 1. 硬规则检查：时长≤5.2秒
        # 2. 连续性基础检查
        # 3. 使用LLM评估视觉连贯性
        # 4. 生成修正建议

        try:
            audit_report = self.quality_auditor.qa_process(state.instructions)
            debug(f"质量审查完成，违规记录数: {len(audit_report.violations)}")

            # 保存剧本解析结果
            self.storage.save_obj_result(state.task_id, audit_report, "quality_auditor_result.json")

            state.audit_report = audit_report
            state.current_stage = AgentStage.AUDITOR

        except Exception as e:
            print_log_exception()
            error(f"质量审查异常: {str(e)}")
            state.error = str(e)
            state.error_messages.append(str(e))

        return state

    def continuity_check_node(self, state: WorkflowState) -> WorkflowState:
        """
        连续性检查节点
        功能：检查跨片段的视觉连续性
        输入：ai_instructions, fragments
        输出：continuity_issues (连续性问题列表)
        """
        # TODO: 实现连续性检查逻辑
        # 1. 跟踪角色服装、道具状态
        # 2. 检查场景一致性
        # 3. 验证位置和动作连续性
        # 4. 标记不连续点
        state.continuity_issues = []
        state.current_stage = AgentStage.CONTINUITY

        return state

    def error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """
        错误处理节点
        功能：处理流程中的错误，决定重试或中止
        输入：error_messages, retry_count
        输出：更新状态，决定下一步
        """
        # 检查是否需要人工干预
        if state.retry_count >= state.max_retries:
            state.needs_human_review = True

        state.current_stage = AgentStage.ERROR

        return state

    def generate_output_node(self, state: WorkflowState) -> WorkflowState:
        """
        结果生成节点
        功能：组装最终输出结果
        输入：所有阶段的结果
        输出：final_output (完整处理结果)
        """
        state.current_stage = AgentStage.END

        fragments = state.instructions.fragments if state.instructions else []

        state.final_output = {
            "status": "completed",
            "metadata": {
                "processed_at": datetime.datetime.now().isoformat(),
                "total_fragments": len(fragments),
                "total_duration": sum(f.duration for f in fragments)
            },
            "fragments": fragments,
            "audit_report": state.audit_report,
            "continuity_report": {
                "issues": state.continuity_issues,
                "issue_count": len(state.continuity_issues)
            },
            "execution_instructions": [
                "1. 按顺序生成每个片段视频",
                "2. 使用相同的风格参数保持一致性",
                "3. 按fragment_id顺序拼接"
            ]
        }
        return state

    def human_intervention_node(self, state: WorkflowState) -> WorkflowState:
        """
        人工干预节点
        功能：暂停流程等待人工输入
        输入：需要人工决策的状态
        输出：人工处理后的状态
        """
        state.current_stage = AgentStage.HUMAN
        # 这里应该等待外部系统（如Web界面）提供反馈
        # 实际实现时可以通过回调或消息队列处理

        # 模拟人工反馈（实际应从外部获取）
        if state.human_feedback:
            # 应用人工修正
            pass

        return state
