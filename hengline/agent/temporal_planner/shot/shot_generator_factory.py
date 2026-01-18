"""
@FileName: shot_generator_factory.py
@Description: 
@Author: HengLine
@Time: 2026/1/17 22:16
"""
from typing import Dict

from hengline.agent.temporal_planner.shot.ai_shot_generator import AIEnhancedShotGenerator
from hengline.agent.temporal_planner.shot.base_shot_generator import BaseShotGenerator
from hengline.agent.temporal_planner.shot.rule_shot_generator import RuleShotGenerator
from hengline.agent.temporal_planner.temporal_planner_model import EstimationSource


class ShotGeneratorFactory:
    """分镜头生成器工厂"""

    @staticmethod
    def create_generator(generator_type: EstimationSource = EstimationSource.LOCAL_RULE,
                         config: Dict = None,
                         llm=None) -> BaseShotGenerator:
        """创建分镜头生成器"""
        config = config or {}

        if generator_type == EstimationSource.LOCAL_RULE:
            return RuleShotGenerator(config)
        elif generator_type == EstimationSource.AI_LLM:
            return AIEnhancedShotGenerator(config, llm)
        elif generator_type == EstimationSource.HYBRID:
            # 混合模式：默认使用规则基，需要时自动切换AI
            return AIEnhancedShotGenerator(config, llm)
        else:
            raise ValueError(f"未知的生成器类型: {generator_type}")
