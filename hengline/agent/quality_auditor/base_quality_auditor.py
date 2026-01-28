"""
@FileName: base_quality_auditor.py
@Description: 
@Author: HengLine
@Time: 2026/1/27 0:00
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from hengline.agent.prompt_converter.prompt_converter_models import AIVideoInstructions
from hengline.agent.quality_auditor.quality_auditor_models import QualityAuditReport, BasicViolation
from hengline.hengline_config import HengLineConfig
from hengline.logger import info


class BaseQualityAuditor(ABC):
    """质量审查器抽象基类"""

    def __init__(self, config: Optional[HengLineConfig]):
        self.config = config
        self._initialize()

    def _initialize(self):
        """初始化审查器"""
        info(f"初始化质量审查器: {self.__class__.__name__}")

    @abstractmethod
    def audit(self, instructions: AIVideoInstructions) -> QualityAuditReport:
        """审查AI视频指令（抽象方法）"""
        pass

    def post_process(self, report: QualityAuditReport) -> QualityAuditReport:
        """后处理：填充统计数据等"""
        # 计算统计数据
        total_checks = len(report.checks)
        passed_checks = sum(1 for check in report.checks if check.get("status") == "passed")
        warnings = sum(1 for violation in report.violations if violation.severity == "warning")
        errors = sum(1 for violation in report.violations if violation.severity == "error")

        status: str = "passed"
        # 确定整体状态
        if errors > 0:
            status = "failed"
        elif warnings > 0:
            status = "needs_review"

        # 更新报告
        report.stats.update({
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "warnings": warnings,
            "errors": errors
        })

        report.status = status

        # 更新结论
        if status == "passed":
            report.conclusion = "审查通过，可以开始视频生成"
        elif status == "needs_review":
            report.conclusion = f"有{warnings}个警告需要检查，建议修复后重新审查"
        else:
            report.conclusion = f"有{errors}个错误需要修复，无法开始视频生成"

        info(f"审查完成: {passed_checks}/{total_checks}通过, {warnings}警告, {errors}错误")
        return report

    def _add_check(self, report: QualityAuditReport, check_name: str, status: str, details: str = "") -> None:
        """添加检查记录"""
        report.checks.append({
            "name": check_name,
            "status": status,
            "details": details,
            "checked_at": datetime.now().isoformat()
        })

    def _add_violation(self, report: QualityAuditReport, rule_id: str, rule_name: str,
                       description: str, severity: str = "warning",
                       fragment_id: Optional[str] = None, suggestion: Optional[str] = None) -> None:
        """添加违规记录"""
        violation = BasicViolation(
            rule_id=rule_id,
            rule_name=rule_name,
            description=description,
            severity=severity,
            fragment_id=fragment_id,
            suggestion=suggestion
        )
        report.violations.append(violation)
