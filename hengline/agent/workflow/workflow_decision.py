"""
@FileName: workflow_decision.py
@Description: 决策函数类 - 控制工作流分支逻辑
@Author: HengLine
@Time: 2026/1/26 16:12
"""
from hengline.logger import error, warning
from hengline.agent.workflow.workflow_states import WorkflowState


class DecisionFunctions:
    """决策函数类 - 控制工作流分支逻辑"""

    def decide_after_parsing(self, state: WorkflowState) -> str:
        """剧本解析后的决策"""
        parsed_script = state.get("parsed_script")
        if not parsed_script:
            error("剧本解析，数据为空")
            return "critical_failure"

        # 检查解析质量
        if not parsed_script.is_valid:
            error("剧本解析，数据有问题")
            return "critical_failure"

        return "success"

    def decide_after_splitting(self, state: WorkflowState) -> str:
        """镜头拆分后的决策"""
        shot_sequence = state.get("shot_sequence")
        if not shot_sequence or len(shot_sequence.shots) < 1:
            error("镜头拆分，数据为空")
            return "critical_failure"

        # 检查是否有过长镜头需要重试
        long_shots = [s for s in shot_sequence.shots if s.duration > 15]
        if long_shots and state["retry_count"] < state["max_retries"]:
            state["retry_count"] += 1
            warning("镜头拆分，有过长镜头需要重试")
            return "retry"

        return "success"

    def decide_after_fragmenting(self, state: WorkflowState) -> str:
        """AI分段后的决策"""
        fragment_sequence = state.get("fragment_sequence")
        if not fragment_sequence or len(fragment_sequence.fragments) < 1:
            error("AI分段后，数据为空")
            return "critical_failure"

        # 检查时长合规性
        invalid_fragments = [f for f in fragment_sequence.fragments if f.duration > 5.2]
        if invalid_fragments:
            if len(invalid_fragments) <= 1:  # 只有1个片段有问题
                warning("AI分段后，片段有问题")
                return "needs_adjustment"
            else:
                error("AI分段后，不符合")
                return "critical_failure"

        return "success"

    def decide_after_prompts(self, state: WorkflowState) -> str:
        """Prompt生成后的决策"""
        instructions = state.get("instructions")
        if not instructions or len(instructions.fragments) < 1:
            error("Prompt生成，不符合")
            return "critical_failure"

        return "continue"  # 总是进入质量审查

    def decide_after_audit(self, state: WorkflowState) -> str:
        """质量审查后的决策"""
        report = state.get("audit_report", {})
        status = report.get("overall_status", "failed")

        decision_map = {
            "passed": "continuity_check",
            "minor_issues": "continuity_check",
            "major_issues": "generate_prompts",  # 重新生成Prompt
            "critical_issues": "split_shots",  # 重新拆分镜头
            "needs_human": "human_intervention"
        }

        return decision_map.get(status, "critical_issues")

    def decide_after_continuity(self, state: WorkflowState) -> str:
        """连续性检查后的决策"""
        issues = state.get("continuity_issues", [])

        if not issues:
            return "passed"

        # 根据问题严重性决定
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        if critical_issues:
            return "structural_issues"
        else:
            return "fixable_issues"

    def decide_after_error(self, state: WorkflowState) -> str:
        """错误处理后的决策"""
        if state["retry_count"] >= state["max_retries"]:
            return "needs_human"
        return "recoverable"

    def decide_after_human(self, state: WorkflowState) -> str:
        """人工干预后的决策"""
        # 实际应该从state中读取人工决策
        human_decision = state.get("human_feedback", {}).get("decision", "continue")
        return human_decision