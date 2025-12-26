"""
@FileName: model_temporal_planner.py
@Description: 基于模型的时序规划智能体模块，通过时间标签预测模型，实现视频动作的时序规划
@Author: HengLine
@Time: 2025/12/1 13:41
"""

from hengline.agent.script_parser.script_parser_model import UnifiedScript
from hengline.agent.temporal_planner.base_temporal_planner import TemporalPlanner
from hengline.agent.temporal_planner.temporal_planner_model import TimelinePlan


class ModelTemporalPlanner(TemporalPlanner):
    """
    # 可以训练一个回归模型预测每个动作的时长
    # 或者分类模型预测“是否应该在此处切割镜头”
    """

    def plan_timeline(self, structured_script: UnifiedScript) -> TimelinePlan | None:
        pass
