"""
@FileName: shot_splitter_factory.py
@Description: 
@Author: HengLine
@Time: 2026/1/26 22:14
"""
from hengline.agent.base_models import AgentMode
from hengline.agent.shot_segmenter.base_shot_segmenter import BaseShotSegmenter
from hengline.agent.shot_segmenter.llm_shot_segmenter import LLMShotSegmenter
from hengline.agent.shot_segmenter.rule_shot_segmenter import RuleShotSegmenter


class ShotSegmenterFactory:
    """分镜拆分器工厂"""

    @staticmethod
    def create_segmenter(mode_type: AgentMode = AgentMode.LLM, **kwargs) -> BaseShotSegmenter:
        """创建分镜拆分器"""
        if mode_type == AgentMode.RULE:
            return RuleShotSegmenter(kwargs.get("config"))
        elif mode_type == AgentMode.LLM:
            llm_client = kwargs.get("llm_client")
            if not llm_client:
                raise ValueError("LLM拆分器需要llm_client参数")
            return LLMShotSegmenter(llm_client, kwargs.get("config"))
        else:
            raise ValueError(f"未知的拆分器类型: {mode_type}")

# 使用工厂
# segmenter = ShotSegmenterFactory.create_splitter(AgentMode.RULE)
# segmenter = ShotSegmenterFactory.create_splitter(AgentMode.LLM, llm_client=my_llm_client)