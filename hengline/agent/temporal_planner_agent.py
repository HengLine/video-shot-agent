# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: 时序规划智能体，负责将剧本按5秒粒度切分，估算动作时长，使用LangChain实现状态记忆
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from typing import List, Dict, Any

from hengline.agent.temporal_planner.base_temporal_planner import TemporalPlanner
from hengline.agent.temporal_planner.local_temporal_planner import RuleTemporalPlanner
from hengline.logger import debug, error


class TemporalPlannerAgent(TemporalPlanner):
    """时序规划智能体"""

    def __init__(self):
        """初始化时序规划智能体"""
        # 初始化PromptManager，使用正确的提示词目录路径
        super().__init__()

        self.rule_planner = RuleTemporalPlanner()  # 预留用于未来可能的规则规划器集成
        debug(f"时序规划智能体初始化完成，加载了 {len(self.config.base_actions)} 个基础动作配置")

    def plan_timeline(self, structured_script: Dict[str, Any], target_duration: int = 5) -> List[Dict[str, Any]] | None:
        """
        规划剧本的时序分段
        
        Args:
            structured_script: 结构化的剧本
            target_duration: 目标分段时长（秒）
            
        Returns:
            分段计划列表
        """
        debug("开始根据规则规划时序")
        if target_duration:
            self.config.target_segment_duration = target_duration

        try:
            optimized_segments = self.rule_planner.plan_timeline(structured_script, target_duration)
            debug(f"时序规划完成，生成了 {len(optimized_segments)} 个分段")
            return optimized_segments

        except Exception as e:
            error(f"执行时序规划异常: {e}")
            return None
