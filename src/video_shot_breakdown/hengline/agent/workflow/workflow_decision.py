"""
@FileName: workflow_decision.py
@Description: 决策函数类 - 控制工作流分支逻辑
@Author: HengLine
@Github: https://github.com/HengLine/video-shot-agent
@Time: 2026/1/26 16:12
"""
from typing import Dict

from video_shot_breakdown.hengline.agent.quality_auditor.quality_auditor_models import AuditStatus
from video_shot_breakdown.hengline.agent.workflow.workflow_models import DecisionState
from video_shot_breakdown.logger import error, warning, info
from video_shot_breakdown.hengline.agent.workflow.workflow_states import WorkflowState


class DecisionFunctions:
    """决策函数类 - 控制工作流分支逻辑"""

    def decide_after_parsing(self, state: WorkflowState) -> DecisionState:
        """剧本解析后的决策"""
        parsed_script = state.parsed_script
        if not parsed_script:
            error("剧本解析，数据为空")
            return DecisionState.CRITICAL_FAILURE

        # 检查解析质量
        if not parsed_script.is_valid:
            error("剧本解析，数据有问题")
            return DecisionState.CRITICAL_FAILURE

        return DecisionState.SUCCESS

    def decide_after_splitting(self, state: WorkflowState) -> DecisionState:
        """镜头拆分后的决策"""
        shot_sequence = state.shot_sequence
        if not shot_sequence or len(shot_sequence.shots) < 1:
            error("镜头拆分，数据为空")
            return DecisionState.CRITICAL_FAILURE

        # 检查是否有过长镜头需要重试
        long_shots = [s for s in shot_sequence.shots if s.duration > 15]
        if long_shots and state.retry_count < state.max_retries:
            state.retry_count += 1
            warning("镜头拆分，有过长镜头需要重试")
            return DecisionState.RETRY

        return DecisionState.SUCCESS

    def decide_after_fragmenting(self, state: WorkflowState) -> DecisionState:
        """AI分段后的决策"""
        fragment_sequence = state.fragment_sequence
        if not fragment_sequence or len(fragment_sequence.fragments) < 1:
            error("AI分段后，数据为空")
            return DecisionState.CRITICAL_FAILURE

        # 检查时长合规性
        invalid_fragments = [f for f in fragment_sequence.fragments if f.duration > 5.2]
        if invalid_fragments:
            if len(invalid_fragments) <= 1:  # 只有1个片段有问题
                warning("AI分段后，片段有问题")
                return DecisionState.NEEDS_ADJUSTMENT
            else:
                error("AI分段后，不符合")
                return DecisionState.CRITICAL_FAILURE

        return DecisionState.SUCCESS

    def decide_after_prompts(self, state: WorkflowState) -> DecisionState:
        """Prompt生成后的决策"""
        instructions = state.instructions
        if not instructions or len(instructions.fragments) < 1:
            error("Prompt生成，不符合")
            return DecisionState.CRITICAL_FAILURE

        return DecisionState.CONTINUE  # 总是进入质量审查

    def decide_after_audit(self, state: WorkflowState) -> DecisionState:
        """
        质量审查后的决策

        Args:
            state: 工作流状态，包含审计报告

        Returns:
            DecisionState: 决策结果
        """
        # 获取审计报告
        report = state.audit_report

        # 如果没有报告，返回需要重试
        if not report:
            warning("审计报告为空，返回RETRY")
            return DecisionState.RETRY

        # 获取审计状态（支持字符串和枚举）
        status = report.status

        # 决策映射：AuditStatus -> DecisionState
        decision_map: Dict[AuditStatus, DecisionState] = {
            # 通过 -> 成功，继续下一步
            AuditStatus.PASSED: DecisionState.SUCCESS,

            # 轻微问题 -> 可恢复，继续流程
            AuditStatus.MINOR_ISSUES: DecisionState.RECOVERABLE,

            # 主要问题 -> 需要调整提示词
            AuditStatus.MAJOR_ISSUES: DecisionState.NEEDS_ADJUSTMENT,

            # 严重问题 -> 需要重新处理
            AuditStatus.CRITICAL_ISSUES: DecisionState.RETRY,

            # 需要人工干预
            AuditStatus.NEEDS_HUMAN: DecisionState.NEEDS_HUMAN,

            # 失败 -> 严重失败
            AuditStatus.FAILED: DecisionState.CRITICAL_FAILURE
        }

        # 获取决策结果
        decision = decision_map.get(status, DecisionState.RETRY)

        # 记录决策日志
        info(f"审计状态: {status.value} -> 决策: {decision.value}")

        return decision

    def decide_after_continuity(self, state: WorkflowState) -> DecisionState:
        """连续性检查后的决策"""
        issues = state.continuity_issues

        if not issues:
            return DecisionState.PASSED

        # 根据问题严重性决定
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        if critical_issues:
            return DecisionState.STRUCTURAL_ISSUES
        else:
            return DecisionState.FIXABLE_ISSUES

    def decide_after_error(self, state: WorkflowState) -> DecisionState:
        """错误处理后的决策"""
        if state.retry_count >= state.max_retries:
            return DecisionState.NEEDS_HUMAN
        return DecisionState.RECOVERABLE

    def decide_after_human(self, state: WorkflowState) -> DecisionState:
        """人工干预后的决策"""
        # 实际应该从state中读取人工决策
        human_decision = state.human_feedback.get("decision", DecisionState.CONTINUE)
        return human_decision