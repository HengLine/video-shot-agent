# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: 基于规则的启发式算法，实现时序规划（负责将剧本按5秒粒度切分，估算动作时长）
@Author: HengLine
@Time: 2025/10 - 2025/12
"""
import time
from typing import Dict

from hengline.agent.script_parser2.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.base_temporal_planner import BaseTemporalPlanner
from hengline.agent.temporal_planner.estimator.estimator_factory import estimator_factory
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation
from hengline.logger import debug, error
from utils.log_utils import print_log_exception


class LocalRuleTemporalPlanner(BaseTemporalPlanner):
    """基于规则的动作合并算法
            规则优先级：
            1. 情感强烈变化点（如震惊）→ 必须独立镜头
            2. 对话前后 → 通常拆分
            3. 物理位置/视角变化 → 建议拆分
            4. 时长填充与合并
    """

    def __init__(self):
        """初始化时序规划智能体"""

        # 初始化估算器
        super().__init__()
        self.factory = estimator_factory

    def estimate_all_elements(self, script_data: UnifiedScript, context: Dict = None) -> Dict[str, DurationEstimation]:
        """
        使用 规则估算所有元素的时长 - 实现基类抽象方法
        """
        start_time = time.time()
        try:
            # 根据元素类型调用不同的AI估算方法
            estimation = self.factory.estimate_script_with_rules(script_data, context)

            # 记录处理时间
            processing_time = time.time() - start_time
            debug(f"LLM估算完成，用时: {processing_time:.2f}秒")

            return estimation

        except Exception as e:
            print_log_exception()
            error(f"LLM估算失败 {script_data.meta}: {str(e)}")
            return {}
