"""
@FileName: llm_quality_auditor.py
@Description: 
@Author: HengLine
@Time: 2026/1/27 0:00
"""
from typing import Dict, Any, Optional

from video_shot_breakdown.hengline.agent.base_agent import BaseAgent
from video_shot_breakdown.hengline.agent.prompt_converter.prompt_converter_models import AIVideoInstructions
from video_shot_breakdown.hengline.agent.quality_auditor.base_quality_auditor import BaseQualityAuditor
from video_shot_breakdown.hengline.agent.quality_auditor.quality_auditor_models import QualityAuditReport
from video_shot_breakdown.hengline.agent.quality_auditor.rule_quality_auditor import RuleQualityAuditor
from video_shot_breakdown.hengline.hengline_config import HengLineConfig
from video_shot_breakdown.logger import info, error


class LLMQualityAuditor(BaseQualityAuditor, BaseAgent):
    """基于LLM的质量审查器"""

    def __init__(self, llm_client, config: Optional[HengLineConfig]):
        super().__init__(config)
        self.llm_client = llm_client

    def audit(self, instructions: AIVideoInstructions) -> QualityAuditReport:
        """使用LLM进行质量审查"""
        info(f"使用LLM进行质量审查")

        # 先执行基本规则检查
        basic_auditor = RuleQualityAuditor(self.config)
        basic_report = basic_auditor.audit(instructions)

        # 然后使用LLM进行更深入的检查
        try:
            llm_result = self._call_llm_audit(instructions)
            self._merge_llm_result(basic_report, llm_result)
        except Exception as e:
            error(f"LLM审查失败: {str(e)}")
            # 只使用基本规则结果

        return self.post_process(basic_report)

    def _call_llm_audit(self, instructions: AIVideoInstructions) -> Dict[str, Any]:
        """调用LLM进行审查"""
        # 准备片段列表文本
        fragments_list = []
        for frag in instructions.fragments[:10]:  # 限制数量防止token超限
            fragments_list.append(f"- {frag.fragment_id}: {frag.prompt[:100]}... ({frag.duration}秒)")

        fragments_text = "\n".join(fragments_list)

        # 准备提示词
        user_prompt = self._get_prompt_template("quality_auditor").format(
            title=instructions.project_info.get("title", "未命名"),
            fragment_count=len(instructions.fragments),
            total_duration=instructions.project_info.get("total_duration", 0.0),
            fragments_list=fragments_text
        )

        # 调用LLM
        system_prompt = "你是一位资深电影导演和AI提示词专家，请对以下分镜内容进行全面审查。"

        # 调用LLM
        return self._call_llm_parse_with_retry(self.llm_client, system_prompt, user_prompt)

    def _merge_llm_result(self, report: QualityAuditReport, llm_result: Dict[str, Any]) -> None:
        """合并LLM审查结果"""
        # 添加LLM检查记录
        self._add_check(
            report,
            "LLM连贯性检查",
            llm_result.get("status", "needs_review"),
            llm_result.get("summary", "")
        )

        # 添加LLM发现的问题
        for issue in llm_result.get("issues", []):
            self._add_violation(
                report=report,
                rule_id="llm_coherence",
                rule_name="LLM连贯性检查",
                description=issue.get("description", ""),
                severity=issue.get("severity", "warning"),
                fragment_id=issue.get("fragment_id"),
                suggestion=issue.get("suggestion")
            )
