"""
@FileName: shot_splitter_factory.py
@Description: 
@Author: HengLine
@Time: 2026/1/26 22:14
"""
from hengline.agent.base_models import AgentMode
from hengline.agent.video_splitter.base_video_splitter import BaseVideoSplitter
from hengline.agent.video_splitter.llm_video_splitter import LLMVideoSplitter
from hengline.agent.video_splitter.rule_video_splitter import RuleVideoSplitter


class VideoSplitterFactory:
    """分镜拆分器工厂"""

    @staticmethod
    def create_splitter(mode_type: AgentMode = AgentMode.LLM, **kwargs) -> BaseVideoSplitter:
        """创建分镜拆分器"""
        if mode_type == AgentMode.RULE:
            return RuleVideoSplitter(kwargs.get("config"))
        elif mode_type == AgentMode.LLM:
            llm_client = kwargs.get("llm_client")
            if not llm_client:
                raise ValueError("LLM拆分器需要llm_client参数")
            return LLMVideoSplitter(llm_client, kwargs.get("config"))
        else:
            raise ValueError(f"未知的拆分器类型: {mode_type}")

# 使用工厂
# splitter = VideoSplitterFactory.create_splitter(AgentMode.RULE)
# splitter = VideoSplitterFactory.create_splitter(AgentMode.LLM, llm_client=my_llm_client)
