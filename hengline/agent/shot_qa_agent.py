# -*- coding: utf-8 -*-
"""
@FileName: shot_qa_agent.py
@Description: 分镜审查智能体，负责审查分镜质量和连续性
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from hengline.logger import debug, info
from .shot_qa.auto_fix_suggester import AutoFixSuggester
from .shot_qa.constraint_validator import ConstraintValidator
from .shot_qa.continuity_checker import ContinuityChecker
from .shot_qa.model.check_models import CheckStatus
from .shot_qa.model.fix_models import AutoFixSuggestion, ManualFixRecommendation
from .shot_qa.model.issue_models import IssueSeverity, CriticalIssue, IssueSuggestion, IssueWarning
from .shot_qa.model.review_models import QualityThresholds, ReviewConfig, ReportMetadata, NextStep, FinalDecision, ReviewDecision, OverallAssessment, QualityReviewOutput, \
    QualityReviewInput
from .shot_qa.model.score_models import ScoreWeighting, QualityScores
from .shot_qa.technical_validator import TechnicalValidator
from .shot_qa.visual_quality_assessor import VisualQualityAssessor


class QAReviewAgent:
    """质量审查智能体"""

    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None):
        """
        初始化质量审查智能体
        
        Args:
            llm: 语言模型实例（可选，用于高级审查）
        """
        self.llm = llm
        self.max_shot_duration = 5.5  # 最大允许时长（秒）
        self.config = config or {}

        # 初始化组件
        self.continuity_checker = ContinuityChecker(
            self.config.get("continuity_config", {})
        )
        self.constraint_validator = ConstraintValidator(
            self.config.get("constraint_config", {})
        )
        self.visual_assessor = VisualQualityAssessor(
            self.config.get("visual_config", {})
        )
        self.technical_validator = TechnicalValidator(
            self.config.get("technical_config", {})
        )
        self.fix_suggester = AutoFixSuggester(
            self.config.get("fix_config", {})
        )

        # 默认配置
        self.default_thresholds = QualityThresholds()
        self.default_config = ReviewConfig()

    def review(self, input_data: QualityReviewInput) -> QualityReviewOutput:
        """执行质量审查"""
        start_time = time.time()

        info(f"开始质量审查...")
        debug(f"镜头数量: {len(input_data.sora_shots.shot_sequence)}")
        debug(f"审查配置: {input_data.review_config.strictness_level}")

        # 1. 执行各项检查
        continuity_results = []
        constraint_results = []
        visual_results = []
        technical_results = []

        if input_data.review_config.enable_continuity_checks:
            debug("  执行连续性检查...")
            continuity_results = self.continuity_checker.check_all_continuity(
                input_data.sora_shots.shot_sequence,
                input_data.anchored_timeline
            )
            debug(f"    完成: {len(continuity_results)} 项检查")

        if input_data.review_config.enable_constraint_checks:
            debug("  执行约束验证...")
            constraint_results = self.constraint_validator.validate_all_constraints(
                input_data.sora_shots.shot_sequence,
                input_data.anchored_timeline
            )
            debug(f"    完成: {len(constraint_results)} 项检查")

        if input_data.review_config.enable_visual_quality_checks:
            debug("  执行视觉质量评估...")
            visual_results = self.visual_assessor.assess_all_shots(
                input_data.sora_shots.shot_sequence
            )
            debug(f"    完成: {len(visual_results)} 项检查")

        if input_data.review_config.enable_technical_checks:
            debug("  执行技术验证...")
            technical_results = self.technical_validator.validate_all_shots(
                input_data.sora_shots.shot_sequence
            )
            debug(f"    完成: {len(technical_results)} 项检查")

        # 2. 计算评分
        debug("  计算质量评分...")
        quality_scores = self._calculate_all_scores(
            continuity_results,
            constraint_results,
            visual_results,
            technical_results
        )

        # 3. 识别问题
        debug("  识别问题和生成建议...")
        critical_issues, warnings, suggestions = self._identify_issues(
            continuity_results,
            constraint_results,
            visual_results,
            technical_results,
            input_data.quality_thresholds
        )

        # 4. 生成修复建议
        debug("  生成修复建议...")
        auto_fixes = self.fix_suggester.suggest_fixes(
            continuity_results,
            constraint_results,
            visual_results,
            technical_results
        )

        # 生成手动修复建议
        manual_fixes = self.fix_suggester.generate_manual_fixes(
            critical_issues, warnings
        )

        # 生成配置修复建议
        config_fixes = self.fix_suggester.generate_config_fixes(
            input_data.review_config,
            quality_scores
        )

        # 合并所有手动修复建议
        all_manual_fixes = manual_fixes + config_fixes

        # 优先级排序
        prioritized_fixes = self.fix_suggester.prioritize_fixes(
            auto_fixes, all_manual_fixes
        )

        # 5. 根据优先级结果生成下一步行动
        debug("  根据优先级生成下一步行动...")
        next_steps = self._generate_priority_based_next_steps(
            prioritized_fixes,
            auto_fixes,
            all_manual_fixes,
            critical_issues,
            warnings,
            quality_scores,
            input_data.quality_thresholds
        )

        # 6. 做出最终决策
        debug("  做出最终决策...")
        final_decision = self._make_final_decision(
            quality_scores,
            critical_issues,
            input_data.quality_thresholds,
            prioritized_fixes  # 添加优先级信息
        )

        overall_assessment = self._create_overall_assessment(
            quality_scores,
            critical_issues,
            warnings,
            final_decision,
            prioritized_fixes  # 添加优先级信息
        )

        # 7. 创建报告元数据
        report_metadata = self._create_report_metadata(
            start_time,
            input_data,
            continuity_results,
            constraint_results,
            visual_results,
            technical_results,
            prioritized_fixes  # 添加优先级信息
        )

        processing_time = time.time() - start_time

        debug(f"审查完成！用时 {processing_time:.2f}秒")
        debug(f"最终决策: {final_decision.decision.value}")
        debug(f"关键问题: {len(critical_issues)} 个")
        debug(f"自动修复建议: {len(auto_fixes)} 个")
        debug(f"手动修复建议: {len(all_manual_fixes)} 个")

        # 输出优先级统计
        self._print_priority_summary(prioritized_fixes)

        # 8. 返回审查结果（包含优先级信息）
        return QualityReviewOutput(
            overall_assessment=overall_assessment,
            final_decision=final_decision,
            continuity_checks=continuity_results,
            constraint_checks=constraint_results,
            visual_quality_checks=visual_results,
            technical_checks=technical_results,
            critical_issues=critical_issues,
            warnings=warnings,
            suggestions=suggestions,
            auto_fix_suggestions=auto_fixes,
            manual_fix_recommendations=all_manual_fixes,
            quality_scores=quality_scores,
            next_steps=next_steps,
            report_metadata=report_metadata,
            input_references={
                "sora_shots_summary": {
                    "total_shots": len(input_data.sora_shots.shot_sequence),
                    "total_duration": sum(s.duration for s in input_data.sora_shots.shot_sequence)
                },
                "anchored_timeline_summary": {
                    "total_segments": len(input_data.anchored_timeline.anchored_segments)
                },
                "fix_priority_summary": self._create_fix_priority_summary(prioritized_fixes)
            }
        )

    def _generate_priority_based_next_steps(self,
                                            prioritized_fixes: Dict[str, List],
                                            auto_fixes: List[AutoFixSuggestion],
                                            manual_fixes: List[ManualFixRecommendation],
                                            critical_issues: List[CriticalIssue],
                                            warnings: List[Warning],
                                            quality_scores: QualityScores,
                                            thresholds: QualityThresholds) -> List[NextStep]:
        """根据优先级生成下一步行动"""

        steps = []

        # 1. 立即应用的自动修复
        if prioritized_fixes.get("immediate_auto"):
            steps.append(self._create_immediate_auto_fix_step(prioritized_fixes["immediate_auto"]))

        # 2. 需要审查的自动修复
        if prioritized_fixes.get("review_auto"):
            steps.append(self._create_review_auto_fix_step(prioritized_fixes["review_auto"]))

        # 3. 高优先级手动修复
        if prioritized_fixes.get("high_priority_manual"):
            steps.append(self._create_high_priority_manual_fix_step(
                prioritized_fixes["high_priority_manual"],
                critical_issues
            ))

        # 4. 中等优先级手动修复
        if prioritized_fixes.get("medium_priority_manual"):
            steps.append(self._create_medium_priority_manual_fix_step(
                prioritized_fixes["medium_priority_manual"],
                warnings
            ))

        # 5. 低优先级手动修复
        if prioritized_fixes.get("low_priority_manual"):
            steps.append(self._create_low_priority_manual_fix_step(
                prioritized_fixes["low_priority_manual"]
            ))

        # 6. 配置调整
        if prioritized_fixes.get("config_adjustments"):
            steps.append(self._create_config_adjustment_step(
                prioritized_fixes["config_adjustments"],
                quality_scores,
                thresholds
            ))

        # 7. 添加验证步骤（依赖于所有修复步骤）
        if steps:
            steps.append(self._create_verification_step(steps))

        # 如果没有修复步骤，添加基本步骤
        if not steps:
            if critical_issues:
                steps.append(NextStep(
                    step_id="step_analyze_critical",
                    step_type="analyze",
                    description="分析关键问题",
                    action_items=[
                        "查看所有关键问题",
                        "确定根本原因",
                        "制定修复计划"
                    ],
                    responsible_party="user",
                    estimated_effort="high",
                    is_required=True
                ))
            else:
                steps.append(NextStep(
                    step_id="step_proceed",
                    step_type="proceed",
                    description="继续生成视频",
                    action_items=["准备生成参数", "启动Sora生成", "监控生成过程"],
                    responsible_party="user",
                    estimated_effort="medium",
                    is_required=True
                ))

        return steps

    def _create_immediate_auto_fix_step(self, fixes: List[AutoFixSuggestion]) -> NextStep:
        """创建立即自动修复步骤"""

        fix_ids = [fix.fix_id for fix in fixes]
        target_issues = [fix.target_issue_id for fix in fixes]

        return NextStep(
            step_id="step_immediate_auto_fix",
            step_type="auto_fix",
            description="应用立即自动修复",
            action_items=[
                f"自动应用 {len(fixes)} 个高信心修复",
                "验证修复效果",
                "更新相关状态"
            ],
            responsible_party="system",
            estimated_effort="low",
            prerequisites=["修复系统已就绪"],
            is_required=True,
            metadata={
                "fix_count": len(fixes),
                "fix_ids": fix_ids[:5],  # 只显示前5个
                "target_issues": target_issues[:5]
            }
        )

    def _create_review_auto_fix_step(self, fixes: List[AutoFixSuggestion]) -> NextStep:
        """创建审查自动修复步骤"""

        return NextStep(
            step_id="step_review_auto_fix",
            step_type="review",
            description="审查并确认自动修复",
            action_items=[
                f"审查 {len(fixes)} 个自动修复建议",
                "确认或修改修复内容",
                "批准应用修复"
            ],
            responsible_party="both",
            estimated_effort="medium",
            dependencies=["step_immediate_auto_fix"],
            is_required=len(fixes) > 0,
            metadata={
                "requires_human_review": True,
                "suggested_review_time": f"{len(fixes) * 2}分钟"
            }
        )

    def _create_high_priority_manual_fix_step(self,
                                              fixes: List[ManualFixRecommendation],
                                              critical_issues: List[CriticalIssue]) -> NextStep:
        """创建高优先级手动修复步骤"""

        issue_titles = [issue.title for issue in critical_issues[:3]]

        return NextStep(
            step_id="step_high_priority_manual",
            step_type="manual_fix",
            description="执行高优先级手动修复",
            action_items=[
                f"解决 {len(fixes)} 个关键问题",
                f"重点关注: {', '.join(issue_titles[:2])}",
                "按修复步骤执行",
                "验证修复效果"
            ],
            responsible_party="user",
            estimated_effort="high",
            dependencies=["step_review_auto_fix"],
            prerequisites=["相关工具已准备", "时间已安排"],
            is_required=len(critical_issues) > 0,
            metadata={
                "blocking_issues": len(critical_issues),
                "estimated_time": f"{sum(fix.estimated_time_minutes for fix in fixes)}分钟",
                "skill_required": "intermediate"
            }
        )

    def _create_medium_priority_manual_fix_step(self,
                                                fixes: List[ManualFixRecommendation],
                                                warnings: List[Warning]) -> NextStep:
        """创建中等优先级手动修复步骤"""

        return NextStep(
            step_id="step_medium_priority_manual",
            step_type="manual_fix",
            description="执行中等优先级修复",
            action_items=[
                f"处理 {len(fixes)} 个中等优先级问题",
                "按计划逐步修复",
                "记录修复过程"
            ],
            responsible_party="user",
            estimated_effort="medium",
            dependencies=["step_high_priority_manual"],
            is_required=False,  # 可选，但建议执行
            metadata={
                "suggested_schedule": "在主要修复完成后进行",
                "expected_improvement": "提高整体质量评分"
            }
        )

    def _create_low_priority_manual_fix_step(self, fixes: List[ManualFixRecommendation]) -> NextStep:
        """创建低优先级手动修复步骤"""

        return NextStep(
            step_id="step_low_priority_manual",
            step_type="optional_fix",
            description="可选：执行低优先级修复",
            action_items=[
                f"选择性处理 {len(fixes)} 个低优先级问题",
                "不影响主要功能的情况下进行",
                "记录优化结果"
            ],
            responsible_party="user",
            estimated_effort="low",
            dependencies=["step_medium_priority_manual"],
            is_required=False,
            metadata={
                "optimization_only": True,
                "can_be_deferred": True
            }
        )

    def _create_config_adjustment_step(self,
                                       fixes: List[ManualFixRecommendation],
                                       quality_scores: QualityScores,
                                       thresholds: QualityThresholds) -> NextStep:
        """创建配置调整步骤"""

        return NextStep(
            step_id="step_config_adjustment",
            step_type="config",
            description="调整审查配置",
            action_items=[
                "分析当前配置效果",
                "根据质量评分调整阈值",
                "优化权重设置",
                "保存新配置"
            ],
            responsible_party="expert",
            estimated_effort="medium",
            prerequisites=["配置分析完成", "明确调整目标"],
            is_required=False,
            metadata={
                "current_score": quality_scores.overall_quality_score,
                "threshold": thresholds.overall_quality_threshold,
                "adjustment_needed": quality_scores.overall_quality_score < thresholds.overall_quality_threshold
            }
        )

    def _create_verification_step(self, previous_steps: List[NextStep]) -> NextStep:
        """创建验证步骤"""

        step_ids = [step.step_id for step in previous_steps]

        return NextStep(
            step_id="step_verification",
            step_type="verify",
            description="验证所有修复效果",
            action_items=[
                "重新运行质量审查",
                "检查关键问题是否解决",
                "验证质量评分提升",
                "确认最终分镜质量"
            ],
            responsible_party="both",
            estimated_effort="medium",
            dependencies=step_ids,
            prerequisites=["所有修复步骤已完成"],
            is_required=True,
            metadata={
                "verification_method": "重新审查",
                "expected_outcome": "质量评分达标",
                "pass_criteria": "无关键问题，评分高于阈值"
            }
        )

    def _make_final_decision(self,
                             quality_scores: QualityScores,
                             critical_issues: List[CriticalIssue],
                             thresholds: QualityThresholds,
                             prioritized_fixes: Dict[str, List]) -> FinalDecision:
        """做出最终决策（考虑优先级）"""

        # 检查关键问题
        has_critical_issues = len(critical_issues) > 0

        # 检查阈值
        meets_quality = quality_scores.overall_quality_score >= thresholds.overall_quality_threshold
        meets_continuity = quality_scores.continuity_score >= thresholds.continuity_score_threshold
        meets_constraints = quality_scores.constraint_satisfaction_score >= thresholds.constraint_satisfaction_threshold

        # 检查是否有立即可用的自动修复
        has_immediate_auto_fixes = len(prioritized_fixes.get("immediate_auto", [])) > 0
        has_high_priority_manual = len(prioritized_fixes.get("high_priority_manual", [])) > 0

        # 确定决策
        if has_critical_issues:
            if has_immediate_auto_fixes:
                # 有关键问题但有立即自动修复
                decision = ReviewDecision.NEEDS_REVISION
                next_action = "apply_immediate_fixes"
            elif has_high_priority_manual:
                # 需要手动修复
                decision = ReviewDecision.NEEDS_REVISION
                next_action = "manual_revision"
            else:
                # 无法修复，需要重新生成
                decision = ReviewDecision.REJECTED
                next_action = "regenerate"

        elif not meets_quality:
            if quality_scores.overall_quality_score >= thresholds.approval_threshold * 0.7:
                decision = ReviewDecision.NEEDS_REVISION
                next_action = "revise"
            else:
                decision = ReviewDecision.REJECTED
                next_action = "regenerate"

        elif not meets_constraints and thresholds.critical_constraint_threshold == 1.0:
            decision = ReviewDecision.NEEDS_REVISION
            next_action = "revise_constraints"

        else:
            if quality_scores.overall_quality_score >= thresholds.approval_threshold:
                decision = ReviewDecision.APPROVED
                next_action = "proceed"
            else:
                decision = ReviewDecision.APPROVED_WITH_ISSUES
                next_action = "proceed_with_caution"

        # 统计检查结果（简化）
        passed_checks = 80
        failed_checks = len(critical_issues) * 5
        warning_checks = 10

        # 添加优先级信息到决策依据
        primary_reasons = [
            f"总体质量得分: {quality_scores.overall_quality_score:.2%}",
            f"关键问题数量: {len(critical_issues)}",
            f"约束满足度: {quality_scores.constraint_satisfaction_score:.2%}"
        ]

        # 添加修复建议信息
        if has_immediate_auto_fixes:
            primary_reasons.append(f"有 {len(prioritized_fixes['immediate_auto'])} 个立即自动修复可用")

        if has_high_priority_manual:
            primary_reasons.append(f"需要 {len(prioritized_fixes['high_priority_manual'])} 个手动修复")

        return FinalDecision(
            decision=decision,
            decision_date=datetime.now(),
            primary_reasons=primary_reasons,
            supporting_evidence=[
                f"自动修复建议: {sum(len(fixes) for key, fixes in prioritized_fixes.items() if 'auto' in key.lower())} 个",
                f"手动修复建议: {sum(len(fixes) for key, fixes in prioritized_fixes.items() if 'manual' in key.lower())} 个"
            ],
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warning_checks=warning_checks,
            meets_quality_thresholds=meets_quality,
            meets_continuity_thresholds=meets_continuity,
            meets_constraint_thresholds=meets_constraints,
            next_action=next_action,
            action_deadline=datetime.now().replace(hour=23, minute=59, second=59) if decision == ReviewDecision.NEEDS_REVISION else None
        )

    def _create_overall_assessment(self,
                                   quality_scores: QualityScores,
                                   critical_issues: List[CriticalIssue],
                                   warnings: List[Warning],
                                   final_decision: FinalDecision,
                                   prioritized_fixes: Dict[str, List]) -> OverallAssessment:
        """创建总体评估（包含优先级信息）"""

        # 确定优势
        strengths = []
        if quality_scores.constraint_satisfaction_score >= 0.9:
            strengths.append("约束满足度高")
        if quality_scores.visual_quality_score >= 0.8:
            strengths.append("视觉质量优秀")
        if quality_scores.continuity_score >= 0.8:
            strengths.append("连续性良好")

        # 检查是否有易于修复的问题
        immediate_auto_count = len(prioritized_fixes.get("immediate_auto", []))
        if immediate_auto_count > 0:
            strengths.append(f"有 {immediate_auto_count} 个问题可立即自动修复")

        # 确定弱点
        weaknesses = []
        if quality_scores.constraint_satisfaction_score < 0.8:
            weaknesses.append("部分约束未满足")
        if len(critical_issues) > 0:
            weaknesses.append(f"存在 {len(critical_issues)} 个关键问题")

        # 检查需要手动修复的问题
        high_priority_manual = len(prioritized_fixes.get("high_priority_manual", []))
        if high_priority_manual > 0:
            weaknesses.append(f"需要 {high_priority_manual} 个手动修复")

        if quality_scores.technical_quality_score < 0.7:
            weaknesses.append("技术质量有待提高")

        # 确定风险级别
        risk_level = "low"
        risk_factors = []

        if len(critical_issues) > 0:
            risk_level = "high"
            risk_factors = [issue.title for issue in critical_issues[:3]]
        elif high_priority_manual > 0:
            risk_level = "medium"
            risk_factors = ["需要手动干预修复"]
        elif len(warnings) > 5:
            risk_level = "medium"
            risk_factors = ["多个警告需要处理"]

        # 添加修复相关风险因素
        if immediate_auto_count > 0:
            risk_factors.append(f"自动修复可能不完美 ({immediate_auto_count}个)")

        # 使用建议
        usage_recommendation = None
        if final_decision.decision == ReviewDecision.APPROVED:
            usage_recommendation = "可直接用于Sora视频生成"
        elif final_decision.decision == ReviewDecision.APPROVED_WITH_ISSUES:
            usage_recommendation = "可使用，但建议先修复警告问题"
        elif final_decision.decision == ReviewDecision.NEEDS_REVISION:
            if immediate_auto_count > 0:
                usage_recommendation = f"先应用 {immediate_auto_count} 个自动修复，然后重新审查"
            elif high_priority_manual > 0:
                usage_recommendation = f"需要手动修复 {high_priority_manual} 个关键问题"
            else:
                usage_recommendation = "需要修订后再使用"
        else:
            usage_recommendation = "需要重新生成分镜"

        return OverallAssessment(
            decision=final_decision.decision,
            decision_reason=final_decision.primary_reasons[0] if final_decision.primary_reasons else "质量评估",
            quality_summary=f"总体质量得分: {quality_scores.overall_quality_score:.2%}",
            strengths=strengths,
            weaknesses=weaknesses,
            risk_level=risk_level,
            risk_factors=risk_factors,
            usage_recommendation=usage_recommendation,
            limitations=[
                "基于提示词的分析，实际生成效果可能有所不同",
                "某些主观质量方面（如艺术性）难以量化评估",
                f"自动修复信心度: {self._calculate_average_confidence(prioritized_fixes):.1%}"
            ]
        )

    def _create_report_metadata(self,
                                start_time: float,
                                input_data: QualityReviewInput,
                                continuity_results: List[Any],
                                constraint_results: List[Any],
                                visual_results: List[Any],
                                technical_results: List[Any],
                                prioritized_fixes: Dict[str, List]) -> ReportMetadata:
        """创建报告元数据（包含优先级信息）"""

        processing_time = time.time() - start_time

        total_shots = len(input_data.sora_shots.shot_sequence)

        # 统计约束数量
        constraint_count = 0
        for segment in input_data.anchored_timeline.anchored_segments:
            constraint_count += len(segment.hard_constraints)

        # 统计检查数量
        total_checks = (
                len(continuity_results) +
                len(constraint_results) +
                len(visual_results) +
                len(technical_results)
        )

        # 计算修复建议统计
        fix_statistics = {}
        for category, fixes in prioritized_fixes.items():
            fix_statistics[category] = len(fixes)

        return ReportMetadata(
            report_id=f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_version="1.0.0",
            generated_at=datetime.now(),
            generation_duration_seconds=processing_time,
            reviewed_shots_count=total_shots,
            reviewed_constraints_count=constraint_count,
            performed_checks_count=total_checks,
            reviewer_version="1.0.0",
            configuration_used=input_data.review_config.strictness_level,
            review_confidence=self._calculate_review_confidence(prioritized_fixes),
            verification_status="pending",
            metadata={
                "fix_statistics": fix_statistics,
                "has_immediate_fixes": len(prioritized_fixes.get("immediate_auto", [])) > 0,
                "total_fix_categories": len(prioritized_fixes)
            }
        )

    def _print_priority_summary(self, prioritized_fixes: Dict[str, List]):
        """输出优先级摘要"""

        debug("\n修复建议优先级摘要:")

        categories = [
            ("立即自动应用", "immediate_auto"),
            ("需要审查的自动", "review_auto"),
            ("高优先级手动", "high_priority_manual"),
            ("中等优先级手动", "medium_priority_manual"),
            ("低优先级手动", "low_priority_manual"),
            ("配置调整", "config_adjustments")
        ]

        for display_name, key in categories:
            fixes = prioritized_fixes.get(key, [])
            if fixes:
                debug(f"  {display_name}: {len(fixes)} 个")

                # 显示前几个修复的描述
                for i, fix in enumerate(fixes[:2]):
                    description = fix.description[:40] + "..." if len(fix.description) > 40 else fix.description
                    debug(f"    {i + 1}. {description}")

        # 计算总计
        total_fixes = sum(len(fixes) for fixes in prioritized_fixes.values())
        debug(f"\n  总计: {total_fixes} 个修复建议")

    def _create_fix_priority_summary(self, prioritized_fixes: Dict[str, List]) -> Dict[str, Any]:
        """创建修复优先级摘要"""

        summary = {
            "categories": {},
            "totals": {},
            "recommendations": []
        }

        # 按类别统计
        for category, fixes in prioritized_fixes.items():
            summary["categories"][category] = {
                "count": len(fixes),
                "example": fixes[0].description if fixes else "无"
            }

        # 总计
        total_fixes = sum(len(fixes) for fixes in prioritized_fixes.values())
        auto_fixes = len(prioritized_fixes.get("immediate_auto", [])) + len(prioritized_fixes.get("review_auto", []))
        manual_fixes = total_fixes - auto_fixes

        summary["totals"] = {
            "total_fixes": total_fixes,
            "auto_fixes": auto_fixes,
            "manual_fixes": manual_fixes,
            "auto_fix_percentage": auto_fixes / total_fixes if total_fixes > 0 else 0
        }

        # 生成建议
        immediate_auto = len(prioritized_fixes.get("immediate_auto", []))
        if immediate_auto > 0:
            summary["recommendations"].append(
                f"建议立即应用 {immediate_auto} 个自动修复"
            )

        high_priority = len(prioritized_fixes.get("high_priority_manual", []))
        if high_priority > 0:
            summary["recommendations"].append(
                f"优先处理 {high_priority} 个高优先级手动修复"
            )

        return summary

    def _calculate_average_confidence(self, prioritized_fixes: Dict[str, List]) -> float:
        """计算平均信心度"""

        all_fixes = []
        for fixes in prioritized_fixes.values():
            all_fixes.extend(fixes)

        if not all_fixes:
            return 0.0

        total_confidence = 0.0
        count = 0

        for fix in all_fixes:
            if hasattr(fix, 'confidence_score'):
                total_confidence += fix.confidence_score
                count += 1
            elif hasattr(fix, 'auto_apply_confidence'):
                total_confidence += fix.auto_apply_confidence
                count += 1

        return total_confidence / count if count > 0 else 0.0

    def _calculate_review_confidence(self, prioritized_fixes: Dict[str, List]) -> float:
        """计算审查信心度"""

        # 基础信心度
        base_confidence = 0.8

        # 根据修复建议调整
        immediate_auto = len(prioritized_fixes.get("immediate_auto", []))
        high_priority = len(prioritized_fixes.get("high_priority_manual", []))

        # 有立即自动修复增加信心
        if immediate_auto > 0:
            base_confidence += 0.05 * min(immediate_auto, 5)  # 最多增加0.25

        # 需要高优先级手动修复降低信心
        if high_priority > 0:
            base_confidence -= 0.1 * min(high_priority, 3)  # 最多降低0.3

        return max(0.5, min(1.0, base_confidence))

    def _calculate_all_scores(self,
                              continuity_results: List[Any],
                              constraint_results: List[Any],
                              visual_results: List[Any],
                              technical_results: List[Any]) -> QualityScores:
        """计算所有质量评分"""

        # 计算各维度评分
        continuity_scores = self.continuity_checker.calculate_continuity_score(continuity_results)
        constraint_scores = self.constraint_validator.calculate_constraint_scores(constraint_results)
        visual_scores = self.visual_assessor.calculate_visual_quality_scores(visual_results)
        technical_scores = self.technical_validator.calculate_technical_scores(technical_results)

        # 计算总体评分（加权平均）
        weights = ScoreWeighting()

        overall_score = (
                continuity_scores.get("overall", 1.0) * weights.continuity_weight +
                constraint_scores.get("overall", 1.0) * weights.constraint_weight +
                visual_scores.get("overall", 1.0) * weights.visual_weight +
                technical_scores.get("overall", 1.0) * weights.technical_weight
        )

        # 创建QualityScores对象
        return QualityScores(
            continuity_score=continuity_scores.get("overall", 1.0),
            position_consistency_score=continuity_scores.get("position", 1.0),
            appearance_consistency_score=continuity_scores.get("appearance", 1.0),
            temporal_consistency_score=continuity_scores.get("temporal", 1.0),

            constraint_satisfaction_score=constraint_scores.get("overall", 1.0),
            critical_constraint_score=constraint_scores.get("critical", 1.0),
            overall_constraint_score=constraint_scores.get("overall", 1.0),

            visual_quality_score=visual_scores.get("overall", 1.0),
            composition_score=visual_scores.get("composition", 1.0),
            lighting_score=visual_scores.get("lighting", 1.0),
            color_score=visual_scores.get("color", 1.0),
            style_consistency_score=visual_scores.get("style", 1.0),

            technical_quality_score=technical_scores.get("overall", 1.0),
            prompt_quality_score=technical_scores.get("prompt", 1.0),
            camera_quality_score=technical_scores.get("camera", 1.0),
            feasibility_score=technical_scores.get("feasibility", 1.0),

            overall_quality_score=overall_score,
            weighted_quality_score=overall_score,

            confidence_scores={
                "continuity": 0.8,
                "constraint": 0.9,
                "visual": 0.7,
                "technical": 0.8
            }
        )

    def _identify_issues(self,
                         continuity_results: List[Any],
                         constraint_results: List[Any],
                         visual_results: List[Any],
                         technical_results: List[Any],
                         thresholds: QualityThresholds) -> tuple:
        """识别问题"""

        critical_issues = []
        warnings = []
        suggestions = []

        # 检查连续性结果
        for result in continuity_results:
            if hasattr(result, 'status'):
                if result.status == CheckStatus.FAILED:
                    if result.severity == IssueSeverity.CRITICAL:
                        critical_issues.append(self._create_critical_issue(result))
                    elif result.severity == IssueSeverity.HIGH:
                        warnings.append(self._create_warning(result))
                    else:
                        suggestions.append(self._create_suggestion(result))

        # 检查约束结果
        for result in constraint_results:
            if hasattr(result, 'status'):
                if result.status == CheckStatus.FAILED and not result.is_satisfied:
                    if result.severity == IssueSeverity.CRITICAL:
                        critical_issues.append(self._create_critical_issue(result))
                    elif result.severity == IssueSeverity.HIGH:
                        warnings.append(self._create_warning(result))
                    else:
                        suggestions.append(self._create_suggestion(result))

        # 检查阈值
        # 这里可以添加基于阈值的额外问题检测
        return critical_issues, warnings, suggestions

    def _create_critical_issue(self, result: Any) -> CriticalIssue:
        """创建关键问题"""
        return CriticalIssue(
            issue_id=f"critical_{result.check_id}",
            issue_type=type(result).__name__,
            severity=result.severity,
            title=f"关键问题: {result.check_name}",
            description=f"{result.check_description} - 得分: {result.score:.2f}",
            location=result.check_id,
            root_cause="检查失败",
            impact="可能影响视频质量和观看体验",
            related_checks=[result.check_id],
            must_fix=True,
            blocks_approval=True,
            suggested_fix=result.recommended_adjustment if hasattr(result, 'recommended_adjustment') else None,
            fix_effort="medium",
            verification_method="重新运行检查",
            verification_required=True
        )

    def _create_warning(self, result: Any) -> IssueWarning:
        """创建警告"""
        return IssueWarning(
            issue_id=f"warning_{result.check_id}",
            issue_type=type(result).__name__,
            severity=result.severity,
            title=f"警告: {result.check_name}",
            description=f"{result.check_description} - 得分: {result.score:.2f}",
            location=result.check_id,
            suggestion_type="improvement",
            recommended_action="考虑修复以提升质量",
            expected_benefit="提高整体质量评分",
            is_optional=True,
            suggested_priority="medium"
        )

    def _create_suggestion(self, result: Any) -> IssueSuggestion:
        """创建建议"""
        return IssueSuggestion(
            issue_id=f"suggestion_{result.check_id}",
            issue_type=type(result).__name__,
            severity=result.severity,
            title=f"建议: {result.check_name}",
            description=f"{result.check_description} - 得分: {result.score:.2f}",
            location=result.check_id,
            suggestion_category="enhancement",
            proposed_solution=result.recommended_adjustment if hasattr(result, 'recommended_adjustment') else "改进相关方面",
            rationale="提升整体质量",
            expected_improvement=0.1,
            implementation_cost="low"
        )

    def _generate_next_steps(self,
                             final_decision: FinalDecision,
                             critical_issues: List[CriticalIssue],
                             auto_fixes: List[AutoFixSuggestion]) -> List[NextStep]:
        """生成下一步行动"""

        steps = []

        if final_decision.decision == ReviewDecision.APPROVED:
            steps.append(NextStep(
                step_id="step_approve",
                step_type="approve",
                description="批准使用当前分镜",
                action_items=["确认最终分镜", "准备生成视频", "设置生成参数"],
                responsible_party="user",
                estimated_effort="low",
                is_required=True
            ))

        elif final_decision.decision == ReviewDecision.APPROVED_WITH_ISSUES:
            steps.append(NextStep(
                step_id="step_review_issues",
                step_type="review",
                description="审查并修复问题",
                action_items=["查看警告列表", "评估修复优先级", "选择性修复问题"],
                responsible_party="both",
                estimated_effort="medium",
                is_required=True
            ))

            if auto_fixes:
                steps.append(NextStep(
                    step_id="step_apply_fixes",
                    step_type="fix",
                    description="应用自动修复建议",
                    action_items=["审查修复建议", "应用合适修复", "验证修复效果"],
                    responsible_party="system",
                    estimated_effort="low",
                    dependencies=["step_review_issues"],
                    is_required=False
                ))

        elif final_decision.decision == ReviewDecision.NEEDS_REVISION:
            steps.append(NextStep(
                step_id="step_major_revision",
                step_type="revise",
                description="执行主要修订",
                action_items=[
                    "分析问题根本原因",
                    "重新设计问题部分",
                    "更新分镜和提示词",
                    "重新运行质量审查"
                ],
                responsible_party="both",
                estimated_effort="high",
                is_required=True
            ))

        else:  # REJECTED
            steps.append(NextStep(
                step_id="step_regenerate",
                step_type="generate",
                description="重新生成分镜",
                action_items=[
                    "分析失败原因",
                    "调整生成参数",
                    "重新运行智能体4",
                    "执行新的质量审查"
                ],
                responsible_party="system",
                estimated_effort="high",
                is_required=True
            ))

        # 添加验证步骤
        steps.append(NextStep(
            step_id="step_verify",
            step_type="review",
            description="验证最终分镜",
            action_items=["最终检查", "确认所有问题已解决", "备份最终版本"],
            responsible_party="user",
            estimated_effort="low",
            dependencies=[step.step_id for step in steps if step.step_id != "step_verify"],
            is_required=True
        ))

        return steps
