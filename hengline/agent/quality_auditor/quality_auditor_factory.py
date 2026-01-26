"""
@FileName: shot_splitter_factory.py
@Description: 
@Author: HengLine
@Time: 2026/1/26 22:14
"""
from hengline.agent.base_models import AgentMode
from hengline.agent.quality_auditor.base_quality_auditor import BaseQualityAuditor
from hengline.agent.quality_auditor.llm_quality_auditor import LLMQualityAuditor
from hengline.agent.quality_auditor.rule_quality_auditor import RuleQualityAuditor


class QualityAuditorFactory:
    """分镜拆分器工厂"""

    @staticmethod
    def create_auditor(mode_type: AgentMode = AgentMode.LLM, **kwargs) -> BaseQualityAuditor:
        """创建分镜拆分器"""
        if mode_type == AgentMode.RULE:
            return RuleQualityAuditor(kwargs.get("config"))
        elif mode_type == AgentMode.LLM:
            llm_client = kwargs.get("llm_client")
            if not llm_client:
                raise ValueError("LLM拆分器需要llm_client参数")
            return LLMQualityAuditor(llm_client, kwargs.get("config"))
        else:
            raise ValueError(f"未知的拆分器类型: {mode_type}")

# 使用工厂
# auditor = QualityAuditorFactory.create_auditor(AgentMode.RULE)
# auditor = QualityAuditorFactory.create_auditor(AgentMode.LLM, llm_client=llm)
