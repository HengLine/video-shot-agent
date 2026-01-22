# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: LLM + 规则约束实现的时序规划（负责将剧本按5秒粒度切分，估算动作时长，使用LangChain实现状态记忆）
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import time
from typing import Dict

from hengline.agent.script_parser2.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.base_temporal_planner import BaseTemporalPlanner
from hengline.agent.temporal_planner.estimator.estimator_factory import estimator_factory
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation
from hengline.logger import error, debug
from utils.log_utils import print_log_exception


class LLMTemporalPlanner(BaseTemporalPlanner):
    """ LLM 时长估算 """

    def __init__(self, llm):
        """初始化时序规划智能体"""
        super().__init__()
        self.llm = llm
        # 使用工厂方法创建估算器
        self.factory = estimator_factory

    def estimate_all_elements(self, script_data: UnifiedScript, context: Dict = None) -> Dict[str, DurationEstimation]:
        """
        使用LLM估算单个元素（实现抽象方法）
        """
        start_time = time.time()
        try:
            # 根据元素类型调用不同的AI估算方法
            estimation = self.factory.estimate_script_with_llm(self.llm, script_data, context)

            # 记录处理时间
            processing_time = time.time() - start_time
            debug(f"LLM估算完成，用时: {processing_time:.2f}秒")

            return estimation

        except Exception as e:
            print_log_exception()
            error(f"LLM估算失败 {script_data.meta}: {str(e)}")
            return {}
