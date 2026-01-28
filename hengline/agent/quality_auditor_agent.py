"""
@FileName: quality_auditor_agent.py
@Description: 质量审查器
@Author: HengLine
@Time: 2026/1/25 21:59
"""
from typing import Optional

from hengline.agent.base_models import AgentMode
from hengline.agent.prompt_converter.prompt_converter_models import AIVideoInstructions
from hengline.agent.quality_auditor.quality_auditor_factory import QualityAuditorFactory
from hengline.agent.quality_auditor.quality_auditor_models import QualityAuditReport
from hengline.hengline_config import HengLineConfig
from hengline.logger import debug, error
from utils.log_utils import print_log_exception


class QualityAuditorAgent:
    """质量审查器"""

    def __init__(self, llm, config: Optional[HengLineConfig]):
        """
        初始化分镜生成智能体

        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        self.config = config or {}
        if self.config.enable_llm:
            self.auditor = QualityAuditorFactory.create_auditor(AgentMode.LLM, config, llm)
        else:
            self.auditor = QualityAuditorFactory.create_auditor(AgentMode.RULE, config)

    def qa_process(self, instructions: AIVideoInstructions) -> QualityAuditReport | None:
        """ 视频片段 """
        debug("开始审查质量")
        try:
            return self.auditor.audit(instructions)

        except Exception as e:
            print_log_exception()
            error(f"质量审查异常: {e}")
            return None
