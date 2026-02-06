"""
@FileName: human_decision_intervention.py
@Description: 人工干预工作流节点
@Author: HengLine
@Time: 2026/2/5 17:28
"""
import threading
import time

from video_shot_breakdown.hengline.agent.human_decision.human_decision_converter import HumanDecisionConverter
from video_shot_breakdown.hengline.agent.workflow.workflow_models import PipelineState
from video_shot_breakdown.hengline.agent.workflow.workflow_states import WorkflowState
from video_shot_breakdown.logger import info, warning, error


class HumanIntervention:
    """人工干预工作流节点

    职责：
    1. 显示当前状态信息给用户
    2. 从控制台获取人工输入（带3分钟超时）
    3. 将输入保存到状态中
    4. 返回更新后的状态

    注意：这个节点不决定下一步去哪，只收集人工输入
    """

    def __init__(self, timeout_seconds: int = 180):
        """
        初始化人工干预节点

        Args:
            timeout_seconds: 超时时间，默认3分钟
        """
        self.timeout_seconds = timeout_seconds
        self.input_received = False
        self.user_input = None
        self.timeout_occurred = False

        # 创建转换器用于显示
        self.converter = HumanDecisionConverter()

    def _get_user_input_with_timeout(self) -> str:
        """获取用户输入（带超时控制）"""
        print("\n" + "=" * 60)
        print("AI视频生成系统 - 人工干预节点")
        print("=" * 60)
        print("\n系统需要人工决策以继续处理")
        print(f"等待时间: {self.timeout_seconds}秒 ({self.timeout_seconds // 60}分钟)")

        # 启动输入线程
        input_thread = threading.Thread(target=self._input_thread, daemon=True)
        input_thread.start()

        # 启动超时监控
        timeout_thread = threading.Thread(target=self._timeout_monitor, daemon=True)
        timeout_thread.start()

        # 等待完成
        start_time = time.time()
        while not self.input_received and (time.time() - start_time) < self.timeout_seconds + 5:
            time.sleep(0.1)

        if self.timeout_occurred:
            warning("人工输入超时，使用默认继续决策")
            return "CONTINUE"
        elif self.user_input:
            info(f"接收到人工输入: {self.user_input}")
            return self.user_input
        else:
            warning("未接收到有效输入，使用默认继续决策")
            return "CONTINUE"

    def _input_thread(self):
        """输入采集线程"""
        try:
            print("\n请选择下一步操作:")
            print("  [1] CONTINUE  - 继续流程（默认）")
            print("  [2] APPROVE   - 批准通过")
            print("  [3] RETRY     - 重新开始")
            print("  [4] REPAIR    - 修复问题")
            print("  [5] REPAIR    - 修复问题")
            print("  [6] REPAIR    - 修复问题")
            print("  [7] ESCALATE  - 升级处理")
            print("  [8] ABORT     - 中止流程")
            print("\n输入选项编号 (1-8) 或输入选项名称: ", end="", flush=True)

            raw_input = input().strip()

            # 处理数字输入
            if raw_input.isdigit():
                num_map = {
                    "1": "CONTINUE",
                    "2": "APPROVE",
                    "3": "RETRY",
                    "4": "REPAIR",
                    "5": "REPAIR",
                    "6": "REPAIR",
                    "7": "ESCALATE",
                    "8": "ABORT"
                }
                self.user_input = num_map.get(raw_input, "CONTINUE")
            else:
                # 直接使用输入，转为大写
                self.user_input = raw_input.upper() if raw_input else "CONTINUE"

            self.input_received = True

        except Exception as e:
            error(f"输入线程异常: {str(e)}")
            self.user_input = "CONTINUE"
            self.input_received = True

    def _timeout_monitor(self):
        """超时监控线程"""
        start_time = time.time()

        while time.time() - start_time < self.timeout_seconds:
            if self.input_received:
                return

            # 显示剩余时间（每分钟一次）
            elapsed = int(time.time() - start_time)
            remaining = self.timeout_seconds - elapsed

            if remaining > 0 and remaining % 60 == 0:
                minutes = remaining // 60
                print(f"\n[提醒] 剩余等待时间: {minutes}分钟")

            time.sleep(1)

        # 超时处理
        if not self.input_received:
            print(f"\n等待{self.timeout_seconds}秒无响应，自动继续流程...")
            self.timeout_occurred = True
            self.user_input = "CONTINUE"
            self.input_received = True

    def _display_decision_options(self):
        """显示决策选项"""
        print("\n可选决策:")
        print("-" * 40)

        options = [
            ("1", "CONTINUE", "继续流程"),
            ("2", "APPROVE", "批准通过"),
            ("3", "RETRY", "重新开始"),
            ("4", "REPAIR", "修复问题"),
            ("5", "REPAIR", "修复问题"),
            ("6", "REPAIR", "修复问题"),
            ("7", "ESCALATE", "升级处理"),
            ("8", "ABORT", "中止流程"),
        ]

        for num, code, desc in options:
            # 获取对应的 PipelineState
            decision_state = self.converter.STANDARD_TO_STATE_MAP.get(code, PipelineState.SUCCESS)
            decision_desc = self.converter.get_decision_description(decision_state)
            input_desc = self.converter.get_standard_input_description(code)
            print(f"  [{num}] {code:10} - {input_desc:8} -> {decision_desc}")

        print("-" * 40)

    def _display_state_info(self, state: WorkflowState):
        """显示状态信息"""
        print("\n当前状态:")
        print("-" * 40)

        # 基础信息
        print(f"任务ID: {state.task_id}")
        print(f"当前阶段: {state.current_stage}")

        # 如果有错误
        if state.error_messages and len(state.error_messages) > 0:
            print(f"错误: {len(state.error_messages)}个")
            for i, err in enumerate(state.error_messages[-2:], 1):
                truncated = err[:60] + "..." if len(err) > 60 else err
                print(f"     {i}. {truncated}")

        # 重试信息
        if state.retry_count > 0:
            print(f"重试: {state.retry_count}/{state.max_retries}")

        # 质量审查
        if state.audit_report:
            status = state.audit_report.status.value
            print(f"质量: {status}")

        # 连续性问题
        if state.continuity_issues:
            print(f"🔗 连续性: {len(state.continuity_issues)}个问题")

        print("-" * 40)

    def _display_timeout_info(self):
        """显示超时信息"""
        minutes = self.timeout_seconds // 60
        seconds = self.timeout_seconds % 60

        print(f"\n超时设置: {minutes}分{seconds}秒")
        print("超时将自动选择: CONTINUE (继续流程)")
        print(f"超时后映射到: {self.converter.get_decision_description(PipelineState.SUCCESS)}")

    def __call__(self, graph_state: WorkflowState) -> WorkflowState:
        """执行人工干预节点"""
        # 清屏或分隔
        print("\n" + "=" * 60)
        print("AI视频生成 - 人工决策节点")
        print("=" * 60)

        # 显示状态信息
        self._display_state_info(graph_state)

        # 显示决策选项
        self._display_decision_options()

        # 显示超时信息
        self._display_timeout_info()

        print(f"\n等待输入 ({self.timeout_seconds}秒超时)...")

        # 获取输入（使用之前的超时逻辑）
        human_input = self._get_user_input_with_timeout()

        # 更新状态
        graph_state.human_feedback = {
            "decision": human_input,
            "timeout": self.timeout_occurred,
            "auto_decision": self.timeout_occurred,
            "timestamp": time.time(),
            "raw_input": human_input,
        }

        # 显示用户选择
        normalized = self.converter.normalize_input(human_input)
        decision_state = self.converter.convert_to_decision_state(normalized)
        decision_desc = self.converter.get_decision_description(decision_state)

        print(f"\n已选择: {human_input}")
        print(f"标准化: {normalized}")
        print(f"决策: {decision_state.value} ({decision_desc})")

        return graph_state
