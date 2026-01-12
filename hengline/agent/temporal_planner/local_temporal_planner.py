# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: 基于规则的启发式算法，实现时序规划（负责将剧本按5秒粒度切分，估算动作时长）
@Author: HengLine
@Time: 2025/10 - 2025/12
"""
from datetime import datetime

from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.base_temporal_planner import TemporalPlanner
from hengline.agent.temporal_planner.temporal_planner_model import TimelinePlan
from hengline.config.temporal_planner_config import get_planner_config, TemporalPlanningConfig
from hengline.tools.action_duration_tool import ActionDurationEstimator
from hengline.tools.langchain_memory_tool import LangChainMemoryTool


class LocalRuleTemporalPlanner(TemporalPlanner):
    """基于规则的动作合并算法
            规则优先级：
            1. 情感强烈变化点（如震惊）→ 必须独立镜头
            2. 对话前后 → 通常拆分
            3. 物理位置/视角变化 → 建议拆分
            4. 时长填充与合并
    """

    def __init__(self):
        """初始化时序规划智能体"""
        # 初始化各个组件
        # 获取配置实例
        self.config = get_planner_config()

        self.config = TemporalPlanningConfig()



        # self.rule_planner = RuleTemporalPlanner()  # 预留用于未来可能的规则规划器集成

        # 初始化动作时长估算器
        self.duration_estimator = ActionDurationEstimator()

        # 初始化LangChain记忆工具（替代原有的向量记忆+状态机）
        self.memory_tool = LangChainMemoryTool()

    def plan_timeline(self, structured_script: UnifiedScript) -> TimelinePlan | None:
        """
        规划剧本的时序分段
        
        Args:
            structured_script: 结构化的剧本

        Returns:
            分段计划列表
        """
        # 1. 估算每个元素的时长
        duration_estimations = self.duration_estimator.estimate_all(structured_script)

        # 2. 创建5秒分片
        timeline_segments = self.segmenter.segment_timeline(structured_script, duration_estimations)

        # 3. 生成连续性锚点
        continuity_anchors = self.anchor_generator.generate_anchors(timeline_segments)

        # 4. 分析节奏
        pacing_analysis = self.pacing_analyzer.analyze(timeline_segments, structured_script)

        # 5. 验证质量
        quality_metrics = self.quality_validator.validate(timeline_segments,
                                                          duration_estimations,
                                                          structured_script)
        # 6. 构建最终结果
        timeline_plan = TimelinePlan(
            timeline_segments=timeline_segments,
            duration_estimations=duration_estimations,
            continuity_anchors=continuity_anchors,
            pacing_analysis=pacing_analysis,
            quality_metrics=quality_metrics,
            metadata={
                "total_duration": sum(seg.duration for seg in timeline_segments),
                "segment_count": len(timeline_segments),
                "processing_timestamp": datetime.now().isoformat(),
                "agent_version": "1.0"
            }
        )

        return timeline_plan
