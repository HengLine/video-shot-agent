"""
@FileName: workflow_decision.py
@Description: 决策函数类 - 控制工作流分支逻辑
@Author: HengLine
@Github: https://github.com/HengLine/video-shot-agent
@Time: 2026/1/26 16:12
"""
from video_shot_breakdown.hengline.agent.human_decision.human_decision_converter import HumanDecisionConverter
from video_shot_breakdown.hengline.agent.quality_auditor.quality_auditor_models import AuditStatus
from video_shot_breakdown.hengline.agent.workflow.workflow_models import DecisionState
from video_shot_breakdown.hengline.agent.workflow.workflow_states import WorkflowState
from video_shot_breakdown.logger import error, warning, info


class DecisionFunctions:
    """决策函数类 - 控制工作流分支逻辑

    职责说明：
    1. 每个函数接收 WorkflowState 作为输入
    2. 根据当前状态数据做出决策判断
    3. 返回 DecisionState 枚举值，指示下一步应该做什么
    4. 决策结果将决定工作流的下一个节点

    设计原则：
    - 保持函数纯净，不修改传入的状态对象
    - 明确的错误处理和日志记录
    - 合理的状态转换逻辑
    - 考虑重试机制和人工干预需求
    """
    def __init__(self):
        """初始化决策函数"""
        self.converter = HumanDecisionConverter()

    def decide_after_parsing(self, state: WorkflowState) -> DecisionState:
        """剧本解析后的决策

        决策逻辑：
        1. 检查解析结果是否存在
        2. 检查解析结果是否有效
        3. 根据检查结果返回相应决策状态

        可能的返回状态：
        - SUCCESS: 解析成功，可以继续下一步
        - REQUIRE_HUMAN: 解析数据有问题，需要人工判断
        - CRITICAL_FAILURE: 解析完全失败，需要错误处理

        Args:
            state: 包含解析结果的工作流状态

        Returns:
            DecisionState: 决策结果
        """
        parsed_script = state.parsed_script

        # 检查解析结果是否存在
        if not parsed_script:
            error("剧本解析，数据为空")
            return DecisionState.CRITICAL_FAILURE

        # 检查解析质量
        if not parsed_script.is_valid:
            error("剧本解析，数据有问题")
            # 数据无效时，需要人工判断是否可以继续
            return DecisionState.REQUIRE_HUMAN

        # 解析成功，可以继续下一步
        return DecisionState.SUCCESS

    def decide_after_splitting(self, state: WorkflowState) -> DecisionState:
        """镜头拆分后的决策

        决策逻辑：
        1. 检查拆分结果是否存在
        2. 检查是否有过长镜头
        3. 根据重试次数和问题严重性决定

        可能的返回状态：
        - SUCCESS: 拆分成功，可以继续下一步
        - SHOULD_RETRY: 有可重试的问题，重试当前节点
        - RETRY_WITH_ADJUSTMENT: 需要调整参数后重试
        - CRITICAL_FAILURE: 拆分完全失败

        Args:
            state: 包含镜头序列的工作流状态

        Returns:
            DecisionState: 决策结果
        """
        shot_sequence = state.shot_sequence

        # 检查拆分结果是否存在
        if not shot_sequence or len(shot_sequence.shots) < 1:
            error("镜头拆分，数据为空")
            return DecisionState.CRITICAL_FAILURE

        # 检查是否有过长镜头（超过15秒）
        long_shots = [s for s in shot_sequence.shots if s.duration > 15]

        if long_shots:
            # 还有重试机会时，可以重试
            if state.retry_count < state.max_retries:
                state.retry_count += 1
                warning(f"镜头拆分，发现{len(long_shots)}个过长镜头，需要重试")
                return DecisionState.SHOULD_RETRY
            else:
                # 超过重试次数，需要调整参数
                warning(f"镜头拆分，超过重试次数，需要调整参数")
                return DecisionState.RETRY_WITH_ADJUSTMENT

        # 拆分成功，可以继续下一步
        return DecisionState.SUCCESS

    def decide_after_fragmenting(self, state: WorkflowState) -> DecisionState:
        """AI分段后的决策

        决策逻辑：
        1. 检查分段结果是否存在
        2. 检查时长合规性（不超过5.2秒）
        3. 检查片段质量（不过短）
        4. 根据问题数量和类型决定

        可能的返回状态：
        - SUCCESS: 分段成功，可以继续下一步
        - NEEDS_ADJUSTMENT: 有少量问题，需要调整
        - NEEDS_FIX: 有多个问题，需要修复
        - NEEDS_OPTIMIZATION: 有过短片段，需要优化
        - CRITICAL_FAILURE: 分段完全失败

        Args:
            state: 包含片段序列的工作流状态

        Returns:
            DecisionState: 决策结果
        """
        fragment_sequence = state.fragment_sequence

        # 检查分段结果是否存在
        if not fragment_sequence or len(fragment_sequence.fragments) < 1:
            error("AI分段后，数据为空")
            return DecisionState.CRITICAL_FAILURE

        # 检查时长合规性：不能超过5.2秒
        invalid_fragments = [f for f in fragment_sequence.fragments if f.duration > 5.2]

        if invalid_fragments:
            if len(invalid_fragments) <= 1:
                # 只有1个片段有问题，需要调整
                warning("AI分段后，发现1个超长片段")
                return DecisionState.NEEDS_ADJUSTMENT
            elif len(invalid_fragments) <= 3:
                # 2-3个片段有问题，需要修复
                warning(f"AI分段后，发现{len(invalid_fragments)}个超长片段")
                return DecisionState.NEEDS_FIX
            else:
                # 多个片段有问题，严重失败
                error(f"AI分段后，发现{len(invalid_fragments)}个超长片段，不符合要求")
                return DecisionState.CRITICAL_FAILURE

        # 检查片段质量：不能过短（小于0.5秒）
        short_fragments = [f for f in fragment_sequence.fragments if f.duration < 0.5]
        if short_fragments:
            # 有过短片段，需要优化
            warning(f"发现{len(short_fragments)}个过短片段（<0.5秒）")
            return DecisionState.NEEDS_OPTIMIZATION

        # 分段成功，可以继续下一步
        return DecisionState.SUCCESS

    def decide_after_prompts(self, state: WorkflowState) -> DecisionState:
        """Prompt生成后的决策

        决策逻辑：
        1. 检查提示词结果是否存在
        2. 检查提示词是否为空
        3. 检查提示词长度是否合理
        4. 返回相应的决策状态

        可能的返回状态：
        - VALID: 提示词可用，可以继续质量审查
        - NEEDS_ADJUSTMENT: 有空提示词，需要调整
        - NEEDS_OPTIMIZATION: 有过长提示词，需要优化
        - CRITICAL_FAILURE: 提示词生成完全失败

        Args:
            state: 包含AI指令的工作流状态

        Returns:
            DecisionState: 决策结果
        """
        instructions = state.instructions

        # 检查提示词结果是否存在
        if not instructions or len(instructions.fragments) < 1:
            error("Prompt生成，数据为空")
            return DecisionState.CRITICAL_FAILURE

        # 检查是否有空提示词
        empty_prompts = [f for f in instructions.fragments if not f.prompt.strip()]
        if empty_prompts:
            warning(f"发现{len(empty_prompts)}个空提示词")
            return DecisionState.NEEDS_ADJUSTMENT

        # 检查提示词长度是否过长（超过300字符）
        long_prompts = [f for f in instructions.fragments if len(f.prompt) > 300]
        if long_prompts:
            warning(f"发现{len(long_prompts)}个过长提示词（>300字符）")
            return DecisionState.NEEDS_OPTIMIZATION

        # 提示词可用，进入质量审查
        return DecisionState.VALID

    def decide_after_audit(self, state: WorkflowState) -> DecisionState:
        """质量审查后的决策

        决策逻辑：
        1. 检查审计报告是否存在
        2. 根据审计状态映射到决策状态
        3. 记录审计决策日志

        映射关系：
        - PASSED -> SUCCESS: 审查通过
        - MINOR_ISSUES -> VALID: 有轻微问题，但可以继续
        - MODERATE_ISSUES -> NEEDS_ADJUSTMENT: 有中度问题，需要调整
        - MAJOR_ISSUES -> NEEDS_FIX: 有主要问题，需要修复
        - CRITICAL_ISSUES -> SHOULD_RETRY: 有严重问题，应该重试
        - NEEDS_HUMAN -> REQUIRE_HUMAN: 需要人工干预
        - FAILED -> FAILED: 审查失败

        Args:
            state: 包含审计报告的工作流状态

        Returns:
            DecisionState: 决策结果
        """
        report = state.audit_report

        # 检查审计报告是否存在
        if not report:
            warning("审计报告为空，返回重试")
            return DecisionState.SHOULD_RETRY

        # 审计状态到决策状态的映射
        decision_map = {
            AuditStatus.PASSED: DecisionState.SUCCESS,
            AuditStatus.MINOR_ISSUES: DecisionState.VALID,
            AuditStatus.MODERATE_ISSUES: DecisionState.NEEDS_ADJUSTMENT,  # -> CONVERT_PROMPT
            AuditStatus.MAJOR_ISSUES: DecisionState.NEEDS_FIX,  # -> SPLIT_VIDEO
            AuditStatus.CRITICAL_ISSUES: DecisionState.SHOULD_RETRY,  # -> CONVERT_PROMPT
            AuditStatus.NEEDS_HUMAN: DecisionState.REQUIRE_HUMAN,
            AuditStatus.FAILED: DecisionState.FAILED
        }

        # 获取决策结果
        decision = decision_map.get(report.status, DecisionState.SHOULD_RETRY)

        # 记录审计决策日志
        info(f"质量审查决策: 审计状态={report.status.value}, 决策={decision.value}")

        return decision

    def decide_after_continuity(self, state: WorkflowState) -> DecisionState:
        """连续性检查后的决策

        决策逻辑：
        1. 检查是否有连续性问题
        2. 根据问题严重性分类
        3. 返回相应的决策状态

        问题分类：
        - critical: 严重问题，需要修复
        - moderate: 中度问题，需要调整
        - minor: 轻微问题，需要优化
        - 其他: 验证通过

        可能的返回状态：
        - SUCCESS: 没有连续性问题
        - NEEDS_FIX: 有严重问题，需要修复
        - NEEDS_ADJUSTMENT: 有中度问题，需要调整
        - NEEDS_OPTIMIZATION: 有轻微问题，需要优化
        - VALID: 未知严重性，但验证通过

        Args:
            state: 包含连续性问题的的工作流状态

        Returns:
            DecisionState: 决策结果
        """
        issues = state.continuity_issues

        # 检查是否有连续性问题
        if not issues:
            # 没有连续性问题，可以直接生成输出
            return DecisionState.SUCCESS

        # 根据问题严重性分类
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        moderate_issues = [i for i in issues if i.get("severity") == "moderate"]
        minor_issues = [i for i in issues if i.get("severity") == "minor"]

        # 根据问题严重性返回相应决策
        if critical_issues:
            warning(f"发现{len(critical_issues)}个严重连续性问题")
            return DecisionState.NEEDS_FIX
        elif moderate_issues:
            warning(f"发现{len(moderate_issues)}个中度连续性问题")
            return DecisionState.NEEDS_ADJUSTMENT
        elif minor_issues:
            warning(f"发现{len(minor_issues)}个轻微连续性问题")
            return DecisionState.NEEDS_OPTIMIZATION
        else:
            # 未知严重性，但验证通过
            warning(f"发现{len(issues)}个连续性问题，但严重性未知")
            return DecisionState.VALID

    def decide_after_error(self, state: WorkflowState) -> DecisionState:
        """错误处理后的决策

        决策逻辑：
        1. 检查是否超过重试次数
        2. 分析错误类型
        3. 根据错误类型决定下一步

        错误类型分类：
        - 网络/超时错误: 应该重试
        - 验证/无效错误: 可恢复错误
        - 配置错误: 调整后重试
        - 其他错误: 普通失败

        可能的返回状态：
        - REQUIRE_HUMAN: 超过重试次数，需要人工干预
        - SHOULD_RETRY: 网络/超时错误，应该重试
        - VALID: 验证错误，可恢复
        - RETRY_WITH_ADJUSTMENT: 配置错误，调整后重试
        - FAILED: 其他错误，失败

        Args:
            state: 包含错误信息的工作流状态

        Returns:
            DecisionState: 决策结果
        """
        # 检查是否超过重试次数
        if state.retry_count >= state.max_retries:
            warning(f"错误处理: 超过最大重试次数({state.max_retries})")
            return DecisionState.REQUIRE_HUMAN

        # 分析最后一个错误
        last_error = state.error_messages[-1] if state.error_messages else ""

        # 根据错误类型决定下一步
        if "network" in last_error.lower() or "timeout" in last_error.lower():
            # 网络或超时错误，可以重试
            warning("错误处理: 网络或超时错误，建议重试")
            return DecisionState.SHOULD_RETRY
        elif "validation" in last_error.lower() or "invalid" in last_error.lower():
            # 验证错误，可恢复
            warning("错误处理: 验证错误，可恢复")
            return DecisionState.VALID
        elif "configuration" in last_error.lower():
            # 配置错误，需要调整后重试
            warning("错误处理: 配置错误，需要调整后重试")
            return DecisionState.RETRY_WITH_ADJUSTMENT
        else:
            # 其他错误，失败
            warning("错误处理: 其他错误，失败")
            return DecisionState.FAILED

    def decide_after_human(self, state: WorkflowState) -> DecisionState:
        """人工干预后的决策（简化版）

        流程：
        1. 从状态中提取输入
        2. 调用转换器进行转换
        3. 返回决策状态

        Args:
            state: 工作流状态

        Returns:
            DecisionState: 决策状态
        """
        # 从状态中获取人工输入
        human_feedback = state.human_feedback or {}
        raw_input = human_feedback.get("decision", "CONTINUE")
        is_timeout = human_feedback.get("timeout", False)

        # 创建转换上下文
        context = {
            "task_id": state.task_id,
            "current_stage": str(state.current_stage),
            "is_timeout": is_timeout,
            "retry_count": state.retry_count,
            "has_errors": len(state.error_messages) > 0 if state.error_messages else False,
        }

        info(f"开始决策处理，原始输入: {raw_input}")

        # 步骤1：标准化输入
        normalized_input = self.converter.normalize_input(raw_input)

        # 步骤2：转换为决策状态
        decision_state = self.converter.convert_to_decision_state(normalized_input, context)

        # 步骤3：验证决策合理性
        is_valid = self.converter.validate_decision(decision_state, context)

        if not is_valid:
            warning(f"决策验证失败: {decision_state.value}，使用默认继续")
            decision_state = DecisionState.SUCCESS

        # 获取决策描述
        description = self.converter.get_decision_description(decision_state)

        info(f"决策完成: {raw_input} -> {normalized_input} -> {decision_state.value} ({description})")

        return decision_state