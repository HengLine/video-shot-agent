"""
@FileName: prompt_converter_agent.py
@Description: 提示词转换智能体
@Author: HengLine
@Time: 2026/1/18 14:23
"""
from typing import Optional

from hengline.agent.base_models import AgentMode
from hengline.agent.prompt_converter.prompt_converter_factory import PromptConverterFactory
from hengline.agent.prompt_converter.prompt_converter_models import AIVideoInstructions
from hengline.agent.video_splitter.video_splitter_models import FragmentSequence
from hengline.hengline_config import HengLineConfig
from hengline.logger import debug, error
from utils.log_utils import print_log_exception


class PromptConverterAgent:
    """提示指令转换器"""

    def __init__(self, llm, config: Optional[HengLineConfig]):
        """
        初始化分镜生成智能体

        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        self.config = config or {}
        if self.config.enable_llm:
            self.converter = PromptConverterFactory.create_converter(AgentMode.LLM, config, llm)
        else:
            self.converter = PromptConverterFactory.create_converter(AgentMode.RULE, config)

    def prompt_process(self, fragment_sequence: FragmentSequence) -> AIVideoInstructions | None:
        """ 视频片段 """
        debug("开始视频转换提示词")
        try:

            return self.converter.convert(fragment_sequence)

        except Exception as e:
            print_log_exception()
            error(f"视频转换提示词异常: {e}")
            return None
