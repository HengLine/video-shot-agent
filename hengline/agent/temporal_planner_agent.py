# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: 时序规划智能体，负责将剧本按5秒粒度切分，估算动作时长，使用LangChain实现状态记忆
@Author: HengLine
@Time: 2025/10 - 2025/11
"""

from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.llm_temporal_planner import LLMTemporalPlanner
from hengline.agent.temporal_planner.local_temporal_planner import LocalRuleTemporalPlanner
from hengline.agent.temporal_planner.temporal_planner_model import TimelinePlan
from hengline.logger import debug, error
from utils.log_utils import print_log_exception


class TemporalPlannerAgent:
    """时序规划智能体

    输入：统一格式的剧本解析结果
    输出：精确的5秒时间分片方案

    核心任务：
    1. 为每个剧本元素（对话、动作、描述）估算合理时长
    2. 智能分割为5秒粒度的视频片段
    3. 确保时间分配的合理性和连贯性
    4. 标记关键时间节点和情绪转折点

    """

    def __init__(self, llm):
        """初始化时序规划智能体"""
        # 初始化各个组件
        self.rule_planner = LocalRuleTemporalPlanner()
        self.llm_planner = LLMTemporalPlanner(llm)
        """
        estimate_strategy: 估算策略
                - "balanced": 平衡模式（默认）
                - "ai_intensive": AI密集型
                - "rule_based": 规则为主
                - "fast": 快速模式
        """
        self.estimate_strategy: str = "balanced"

    def plan_process(self, structured_script: UnifiedScript) -> TimelinePlan | None:
        """
        规划剧本的时序分段
        
        Args:
            structured_script: 结构化的剧本

        Returns:
            分段计划列表
        """
        debug("开始根据规则规划时序")
        try:


            return self.llm_planner.plan_timeline(structured_script)

        except Exception as e:
            print_log_exception()
            error(f"执行时序规划异常: {e}")
            return None


