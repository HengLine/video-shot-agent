"""
@FileName: auto_fix_suggester.py
@Description: 自动修复建议器 - 生成修复建议
@Author: HengLine
@Time: 2026/1/6 16:04
"""
import copy
from typing import List, Dict, Any, Optional

from .model.check_models import CheckStatus
from .model.fix_models import AutoFixSuggestion, FixType, ManualFixRecommendation
from .model.issue_models import CriticalIssue, IssueSeverity, IssueWarning


class AutoFixSuggester:
    """自动修复建议器"""
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.fix_templates = self._load_fix_templates()
        self.manual_fix_templates = self._load_manual_fix_templates()

    def _load_fix_templates(self) -> Dict[str, Any]:
        """加载自动修复模板"""
        return {
            "continuity_position": {
                "description": "修复位置连续性",
                "template": "调整{character}的位置，从{position_a}渐变到{position_b}",
                "parameters": ["character", "position_a", "position_b"],
                "confidence": 0.7,
                "fix_type": "auto"
            },
            "continuity_appearance": {
                "description": "修复外观连续性",
                "template": "确保{character}的{attribute}在不同镜头间保持一致：{value}",
                "parameters": ["character", "attribute", "value"],
                "confidence": 0.8,
                "fix_type": "auto"
            },
            "constraint_missing": {
                "description": "添加缺失约束",
                "template": "在提示词中添加：'{constraint}'",
                "parameters": ["constraint"],
                "confidence": 0.9,
                "fix_type": "auto"
            },
            "prompt_quality": {
                "description": "改进提示词质量",
                "template": "添加{missing_element}描述以提高提示词完整性",
                "parameters": ["missing_element"],
                "confidence": 0.6,
                "fix_type": "auto"
            },
            "camera_parameters": {
                "description": "调整相机参数",
                "template": "将{parameter}从{current_value}调整为{suggested_value}",
                "parameters": ["parameter", "current_value", "suggested_value"],
                "confidence": 0.7,
                "fix_type": "auto"
            }
        }

    def _load_manual_fix_templates(self) -> Dict[str, Any]:
        """加载手动修复模板"""
        return {
            "critical_continuity": {
                "description": "修复关键连续性错误",
                "steps": [
                    "分析前后镜头的不一致处",
                    "设计平滑的过渡方案",
                    "手动调整角色位置/外观/动作",
                    "使用过渡效果（溶解、淡入淡出等）平滑差异",
                    "验证修复后的连续性"
                ],
                "required_tools": ["视频编辑软件", "关键帧工具"],
                "skill_level": "intermediate",
                "examples": [
                    {
                        "before": "角色从房间左侧突然跳到右侧",
                        "after": "添加移动动作，显示角色走到右侧"
                    }
                ],
                "risks": ["过渡不自然", "时间轴错位"],
                "mitigation": ["预览效果", "逐步调整"]
            },
            "critical_constraint": {
                "description": "修复关键约束违反",
                "steps": [
                    "重新设计满足约束的场景",
                    "调整角色交互或道具使用",
                    "更新提示词以明确约束要求",
                    "生成新的镜头验证约束满足",
                    "进行约束专项测试"
                ],
                "required_tools": ["剧本编辑器", "约束检查工具"],
                "skill_level": "intermediate",
                "examples": [
                    {
                        "before": "角色服装不符合蓝色衬衫要求",
                        "after": "重新设计场景，确保角色始终穿着蓝色衬衫"
                    }
                ],
                "risks": ["场景不连贯", "剧情改变"],
                "mitigation": ["保持核心剧情", "渐进式修改"]
            },
            "visual_quality_issue": {
                "description": "修复视觉质量问题",
                "steps": [
                    "分析构图、灯光、色彩问题",
                    "重新设计视觉元素",
                    "调整相机角度和运动",
                    "优化色彩调色板和灯光设置",
                    "进行视觉质量验证"
                ],
                "required_tools": ["视觉设计工具", "色彩校正工具"],
                "skill_level": "expert",
                "examples": [
                    {
                        "before": "构图不平衡，主体偏左",
                        "after": "重新构图，使用三分法平衡画面"
                    }
                ],
                "risks": ["风格不一致", "过度调整"],
                "mitigation": ["参考风格指南", "小步修改"]
            },
            "technical_feasibility": {
                "description": "解决技术可行性问题",
                "steps": [
                    "分析Sora生成限制",
                    "简化复杂交互或动作",
                    "调整物理上不可能的描述",
                    "优化提示词以提高生成成功率",
                    "测试生成可行性"
                ],
                "required_tools": ["Sora提示词优化器", "可行性分析工具"],
                "skill_level": "intermediate",
                "examples": [
                    {
                        "before": "复杂多人同步舞蹈",
                        "after": "简化舞蹈动作，分镜头拍摄"
                    }
                ],
                "risks": ["效果打折", "创意损失"],
                "mitigation": ["保留核心创意", "分步实现"]
            },
            "complex_multi_issue": {
                "description": "修复复杂多问题组合",
                "steps": [
                    "优先级排序：先修复阻塞性问题",
                    "设计整体解决方案",
                    "分阶段实施修复",
                    "每阶段后进行验证",
                    "最终整体测试"
                ],
                "required_tools": ["项目管理工具", "问题跟踪系统"],
                "skill_level": "expert",
                "examples": [
                    {
                        "before": "多个连续性错误+约束违反",
                        "after": "系统性重设计场景，解决所有问题"
                    }
                ],
                "risks": ["修复冲突", "时间超支"],
                "mitigation": ["详细规划", "定期检查"]
            }
        }

    def suggest_fixes(self,
                      continuity_results: List[Any],
                      constraint_results: List[Any],
                      visual_results: List[Any],
                      technical_results: List[Any]) -> List[AutoFixSuggestion]:
        """生成自动修复建议"""
        fixes = []

        # 1. 连续性修复建议
        fixes.extend(self._suggest_continuity_fixes(continuity_results))

        # 2. 约束修复建议
        fixes.extend(self._suggest_constraint_fixes(constraint_results))

        # 3. 视觉质量修复建议
        fixes.extend(self._suggest_visual_fixes(visual_results))

        # 4. 技术质量修复建议
        fixes.extend(self._suggest_technical_fixes(technical_results))

        # 去重和排序
        fixes = self._deduplicate_fixes(fixes)
        fixes = sorted(fixes, key=lambda x: x.confidence_score, reverse=True)

        return fixes

    def _suggest_continuity_fixes(self, results: List[Any]) -> List[AutoFixSuggestion]:
        """生成连续性修复建议"""
        fixes = []

        for result in results:
            if not isinstance(result, dict) and hasattr(result, 'status'):
                if result.status != CheckStatus.PASSED:
                    fix = self._create_continuity_fix(result)
                    if fix:
                        fixes.append(fix)

        return fixes

    def _create_continuity_fix(self, result: Any) -> Optional[AutoFixSuggestion]:
        """创建连续性修复建议"""

        if result.continuity_type == "position" and result.position_discrepancy:
            # 位置连续性修复
            template = self.fix_templates["continuity_position"]

            # 提取参数
            character = result.affected_elements[0] if result.affected_elements else "character"
            position_a = "前一位置"
            position_b = "当前位置"

            fix_content = template["template"].format(
                character=character,
                position_a=position_a,
                position_b=position_b
            )

            return AutoFixSuggestion(
                fix_id=f"fix_continuity_position_{result.check_id}",
                fix_type=FixType.AUTO,
                target_issue_id=result.check_id,
                description="修复位置连续性",
                fix_content=fix_content,
                fix_instructions=[
                    "1. 识别前一镜头结束时的角色位置",
                    "2. 调整当前镜头起始位置以匹配",
                    "3. 使用平滑的位置过渡"
                ],
                expected_effectiveness=0.8,
                confidence_score=template["confidence"],
                complexity="low",
                estimated_time_minutes=3,
                can_be_auto_applied=True,
                auto_apply_confidence=0.7,
                fix_parameters={
                    "character": character,
                    "discrepancy": result.position_discrepancy,
                    "previous_segment": result.previous_segment_id,
                    "current_segment": result.current_segment_id
                },
                requires_verification=True,
                verification_check="检查位置是否匹配"
            )

        elif result.continuity_type == "appearance" and result.appearance_changes:
            # 外观连续性修复
            template = self.fix_templates["continuity_appearance"]

            # 提取第一个变化
            change = result.appearance_changes[0] if result.appearance_changes else ""
            parts = change.split(": ")
            if len(parts) >= 2:
                character_attr = parts[0]
                change_desc = parts[1]

                # 解析字符和属性
                if "服装" in change_desc:
                    attribute = "clothing"
                    value = change_desc.split("'")[1] if "'" in change_desc else "consistent clothing"
                elif "发型" in change_desc:
                    attribute = "hairstyle"
                    value = change_desc.split("'")[1] if "'" in change_desc else "consistent hairstyle"
                else:
                    attribute = "appearance"
                    value = "consistent appearance"

                character = character_attr.split(":")[0] if ":" in character_attr else character_attr

                fix_content = template["template"].format(
                    character=character,
                    attribute=attribute,
                    value=value
                )

                return AutoFixSuggestion(
                    fix_id=f"fix_continuity_appearance_{result.check_id}",
                    fix_type=FixType.AUTO,
                    target_issue_id=result.check_id,
                    description="修复外观连续性",
                    fix_content=fix_content,
                    fix_instructions=[
                        "1. 检查前一镜头的外观描述",
                        "2. 确保当前镜头使用相同描述",
                        "3. 更新提示词中的外观细节"
                    ],
                    expected_effectiveness=0.9,
                    confidence_score=template["confidence"],
                    complexity="low",
                    estimated_time_minutes=2,
                    can_be_auto_applied=True,
                    auto_apply_confidence=0.8,
                    fix_parameters={
                        "character": character,
                        "attribute": attribute,
                        "changes": result.appearance_changes
                    },
                    requires_verification=True,
                    verification_check="检查外观是否一致"
                )

        return None

    def _suggest_constraint_fixes(self, results: List[Any]) -> List[AutoFixSuggestion]:
        """生成约束修复建议"""
        fixes = []

        for result in results:
            if not isinstance(result, dict) and hasattr(result, 'status'):
                if result.status != CheckStatus.PASSED and not result.is_satisfied:
                    fix = self._create_constraint_fix(result)
                    if fix:
                        fixes.append(fix)

        return fixes

    def _create_constraint_fix(self, result: Any) -> Optional[AutoFixSuggestion]:
        """创建约束修复建议"""

        template = self.fix_templates["constraint_missing"]

        # 使用约束描述
        constraint_desc = result.constraint_description

        fix_content = template["template"].format(constraint=constraint_desc)

        # 生成修复代码（示例）
        fix_code = f"""
# 为镜头 {result.affected_shots[0] if result.affected_shots else 'unknown'} 添加约束
shot.full_sora_prompt += ". {constraint_desc}"
shot.satisfied_constraints.append("{result.constraint_id}")
"""

        return AutoFixSuggestion(
            fix_id=f"fix_constraint_{result.constraint_id}",
            fix_type=FixType.AUTO,
            target_issue_id=result.check_id,
            description="添加缺失约束",
            fix_content=fix_content,
            fix_code=fix_code,
            fix_instructions=[
                f"1. 在提示词中添加: '{constraint_desc}'",
                "2. 标记约束为已满足",
                "3. 验证约束是否在生成的视频中体现"
            ],
            expected_effectiveness=0.95,
            confidence_score=template["confidence"],
            complexity="low",
            estimated_time_minutes=1,
            can_be_auto_applied=True,
            auto_apply_confidence=0.9,
            fix_parameters={
                "constraint_id": result.constraint_id,
                "constraint_desc": constraint_desc,
                "shot_id": result.affected_shots[0] if result.affected_shots else "unknown",
                "priority": result.constraint_priority
            },
            requires_verification=True,
            verification_check="检查约束是否在提示词中"
        )

    def _suggest_visual_fixes(self, results: List[Any]) -> List[AutoFixSuggestion]:
        """生成视觉质量修复建议"""
        fixes = []

        for result in results:
            if not isinstance(result, dict) and hasattr(result, 'status'):
                if result.status != CheckStatus.PASSED:
                    fix = self._create_visual_fix(result)
                    if fix:
                        fixes.append(fix)

        return fixes

    def _create_visual_fix(self, result: Any) -> Optional[AutoFixSuggestion]:
        """创建视觉质量修复建议"""

        if result.quality_dimension == "composition" and result.composition_suggestions:
            # 构图修复
            suggestion = result.composition_suggestions[0] if result.composition_suggestions else "改进构图"

            # 提取缺失元素
            missing_element = "构图元素"
            if "三分法" in suggestion:
                missing_element = "rule of thirds composition"
            elif "层次" in suggestion:
                missing_element = "depth layers"
            elif "取景" in suggestion:
                missing_element = "proper framing"

            template = self.fix_templates["prompt_quality"]

            fix_content = template["template"].format(missing_element=missing_element)

            return AutoFixSuggestion(
                fix_id=f"fix_visual_{result.check_id}",
                fix_type=FixType.AUTO,
                target_issue_id=result.check_id,
                description="改进构图质量",
                fix_content=fix_content,
                fix_instructions=[
                    f"1. {suggestion}",
                    "2. 在提示词中添加构图描述",
                    "3. 验证构图改进效果"
                ],
                expected_effectiveness=0.7,
                confidence_score=0.6,
                complexity="low",
                estimated_time_minutes=2,
                can_be_auto_applied=True,
                auto_apply_confidence=0.6,
                fix_parameters={
                    "suggestion": suggestion,
                    "missing_element": missing_element,
                    "composition_score": result.composition_score
                },
                requires_verification=True,
                verification_check="检查构图是否改进"
            )

        return None

    def _suggest_technical_fixes(self, results: List[Any]) -> List[AutoFixSuggestion]:
        """生成技术质量修复建议"""
        fixes = []

        for result in results:
            if not isinstance(result, dict) and hasattr(result, 'status'):
                if result.status != CheckStatus.PASSED:
                    fix = self._create_technical_fix(result)
                    if fix:
                        fixes.append(fix)

        return fixes

    def _create_technical_fix(self, result: Any) -> Optional[AutoFixSuggestion]:
        """创建技术质量修复建议"""

        if result.technical_aspect == "camera_params" and result.camera_adjustments:
            # 相机参数修复
            adjustment = result.camera_adjustments[0] if result.camera_adjustments else ""

            # 解析参数
            parameter = "unknown"
            current_value = "unknown"
            suggested_value = "unknown"

            if "焦距" in adjustment or "lens" in adjustment.lower():
                parameter = "lens_focal_length"
                suggested_value = "35mm"  # 示例值

            template = self.fix_templates["camera_parameters"]

            fix_content = template["template"].format(
                parameter=parameter,
                current_value=current_value,
                suggested_value=suggested_value
            )

            return AutoFixSuggestion(
                fix_id=f"fix_technical_{result.check_id}",
                fix_type=FixType.AUTO,
                target_issue_id=result.check_id,
                description="调整相机参数",
                fix_content=fix_content,
                fix_instructions=[
                    f"1. {adjustment}",
                    "2. 更新相机参数设置",
                    "3. 验证参数合理性"
                ],
                expected_effectiveness=0.8,
                confidence_score=0.7,
                complexity="medium",
                estimated_time_minutes=3,
                can_be_auto_applied=True,
                auto_apply_confidence=0.7,
                fix_parameters={
                    "adjustment": adjustment,
                    "parameter": parameter,
                    "suggested_value": suggested_value
                },
                requires_verification=True,
                verification_check="检查相机参数是否合理"
            )

        return None

    def _deduplicate_fixes(self, fixes: List[AutoFixSuggestion]) -> List[AutoFixSuggestion]:
        """去重修复建议"""
        unique_fixes = []
        seen_contents = set()

        for fix in fixes:
            # 基于描述和参数创建唯一标识
            fix_key = f"{fix.description}_{fix.target_issue_id}"

            if fix_key not in seen_contents:
                seen_contents.add(fix_key)
                unique_fixes.append(fix)

        return unique_fixes

    def generate_manual_fixes(self, critical_issues: List[CriticalIssue],
                              warnings: List[IssueWarning]) -> List[ManualFixRecommendation]:
        """生成手动修复建议（针对严重问题）"""
        manual_fixes = []

        # 为关键问题生成手动修复建议
        for issue in critical_issues:
            fix = self._create_manual_fix_for_issue(issue)
            if fix:
                manual_fixes.append(fix)

        # 为高优先级警告生成手动修复建议
        high_priority_warnings = [w for w in warnings
                                  if w.suggested_priority == "high" and w.severity == IssueSeverity.HIGH]

        for warning in high_priority_warnings[:3]:  # 最多3个高优先级警告
            fix = self._create_manual_fix_for_warning(warning)
            if fix:
                manual_fixes.append(fix)

        return manual_fixes

    def _create_manual_fix_for_issue(self, issue: CriticalIssue) -> Optional[ManualFixRecommendation]:
        """为关键问题创建手动修复建议"""

        # 根据问题类型选择模板
        issue_type = issue.issue_type.lower()

        if "continuity" in issue_type:
            template_key = "critical_continuity"
        elif "constraint" in issue_type:
            template_key = "critical_constraint"
        elif "visual" in issue_type or "composition" in issue_type or "lighting" in issue_type:
            template_key = "visual_quality_issue"
        elif "technical" in issue_type or "feasibility" in issue_type:
            template_key = "technical_feasibility"
        else:
            template_key = "complex_multi_issue"

        template = self.manual_fix_templates.get(template_key,
                                                 self.manual_fix_templates["complex_multi_issue"])

        # 创建修复步骤（基于模板，但个性化）
        fix_steps = copy.deepcopy(template["steps"])

        # 个性化第一步
        if fix_steps:
            fix_steps[0] = f"分析问题：{issue.title}"

        # 创建前后对比示例
        before_after = {
            "before": issue.description[:100] + "..." if len(issue.description) > 100 else issue.description,
            "after": f"修复后的场景：{issue.suggested_fix if issue.suggested_fix else '问题已解决'}"
        }

        return ManualFixRecommendation(
            fix_id=f"manual_fix_critical_{issue.issue_id}",
            fix_type=FixType.MANUAL,
            target_issue_id=issue.issue_id,
            description=f"手动修复关键问题：{issue.title}",
            fix_content=issue.suggested_fix if issue.suggested_fix else f"需要手动解决：{issue.description}",
            fix_instructions=fix_steps,
            fix_steps=fix_steps,
            required_tools=template["required_tools"],
            skill_level_required=template["skill_level"],
            examples=template["examples"],
            before_after_comparison=before_after,
            risks=template["risks"],
            risk_mitigation=template["mitigation"],
            expected_effectiveness=0.85,
            confidence_score=0.7,
            complexity=issue.fix_effort if hasattr(issue, 'fix_effort') else "high",
            estimated_time_minutes=self._estimate_fix_time(issue, template_key)
        )

    def _create_manual_fix_for_warning(self, warning: IssueWarning) -> Optional[ManualFixRecommendation]:
        """为警告创建手动修复建议"""

        # 使用通用模板
        template = self.manual_fix_templates["visual_quality_issue"]

        # 创建修复步骤
        fix_steps = [
            f"分析警告：{warning.title}",
            "评估修复的必要性和优先级",
            warning.recommended_action if warning.recommended_action else "实施改进措施",
            "验证修复效果",
            "更新相关文档"
        ]

        return ManualFixRecommendation(
            fix_id=f"manual_fix_warning_{warning.issue_id}",
            fix_type=FixType.MANUAL,
            target_issue_id=warning.issue_id,
            description=f"手动修复警告：{warning.title}",
            fix_content=warning.recommended_action if warning.recommended_action else warning.description,
            fix_instructions=fix_steps,
            fix_steps=fix_steps,
            required_tools=["基本编辑工具", "质量检查工具"],
            skill_level_required="intermediate",
            examples=[{
                "before": warning.description[:80] + "..." if len(warning.description) > 80 else warning.description,
                "after": f"改进后的效果：{warning.expected_benefit if warning.expected_benefit else '质量提升'}"
            }],
            risks=["过度优化", "引入新问题"],
            risk_mitigation=["小步修改", "充分测试"],
            expected_effectiveness=0.7,
            confidence_score=0.6,
            complexity="medium",
            estimated_time_minutes=5
        )

    def _estimate_fix_time(self, issue: CriticalIssue, template_key: str) -> int:
        """估算修复时间（分钟）"""
        # 基于问题严重性和修复复杂度估算
        base_time = {
            "critical_continuity": 15,
            "critical_constraint": 20,
            "visual_quality_issue": 25,
            "technical_feasibility": 30,
            "complex_multi_issue": 45
        }.get(template_key, 20)

        # 基于修复复杂度调整
        if hasattr(issue, 'fix_effort'):
            if issue.fix_effort == "high":
                return base_time * 2
            elif issue.fix_effort == "low":
                return base_time // 2

        return base_time

    def generate_config_fixes(self, review_config: Any,
                              quality_scores: Any) -> List[ManualFixRecommendation]:
        """生成配置修复建议"""

        fixes = []

        # 检查配置是否适合当前质量水平
        if quality_scores.overall_quality_score < 0.6:
            # 质量较低，建议调整严格度
            fixes.append(self._create_config_fix_low_quality(review_config, quality_scores))

        # 检查是否需要调整阈值
        if (quality_scores.constraint_satisfaction_score > 0.95 and
                quality_scores.visual_quality_score < 0.6):
            # 约束满足度高但视觉质量低，可能需要调整权重
            fixes.append(self._create_config_fix_weight_adjustment(review_config, quality_scores))

        return fixes

    def _create_config_fix_low_quality(self, review_config: Any,
                                       quality_scores: Any) -> ManualFixRecommendation:
        """创建低质量配置修复建议"""

        fix_steps = [
            "1. 降低审查严格度级别",
            "2. 调整质量阈值以适应当前水平",
            "3. 重点关注关键问题，暂时忽略次要问题",
            "4. 生成新的分镜后逐步提高标准"
        ]

        return ManualFixRecommendation(
            fix_id="config_fix_low_quality",
            fix_type=FixType.CONFIG,
            target_issue_id="overall_low_quality",
            description="调整审查配置以适应低质量分镜",
            fix_content="降低严格度，调整阈值，分阶段改进",
            fix_instructions=fix_steps,
            fix_steps=fix_steps,
            required_tools=["配置编辑器", "质量分析工具"],
            skill_level_required="intermediate",
            examples=[{
                "before": f"当前严格度：{review_config.strictness_level}，总体质量：{quality_scores.overall_quality_score:.2%}",
                "after": "调整到lenient模式，允许更多宽容，逐步改进"
            }],
            risks=["标准过低导致质量差", "难以逐步提高"],
            risk_mitigation=["设置明确的改进里程碑", "定期重新评估"],
            expected_effectiveness=0.8,
            confidence_score=0.75,
            complexity="low",
            estimated_time_minutes=10
        )

    def _create_config_fix_weight_adjustment(self, review_config: Any,
                                             quality_scores: Any) -> ManualFixRecommendation:
        """创建权重调整配置修复建议"""

        fix_steps = [
            "1. 降低约束满足度的权重（当前可能过高）",
            "2. 提高视觉质量的权重",
            "3. 调整技术质量的权重平衡",
            "4. 重新计算总体评分"
        ]

        return ManualFixRecommendation(
            fix_id="config_fix_weight_adjustment",
            fix_type=FixType.CONFIG,
            target_issue_id="weight_imbalance",
            description="调整评分权重以更好反映质量",
            fix_content="重新分配各质量维度的权重比例",
            fix_instructions=fix_steps,
            fix_steps=fix_steps,
            required_tools=["权重配置工具", "评分计算器"],
            skill_level_required="expert",
            examples=[{
                "before": f"约束：{quality_scores.constraint_satisfaction_score:.2%}，视觉：{quality_scores.visual_quality_score:.2%}",
                "after": "平衡权重，使总体评分更准确反映综合质量"
            }],
            risks=["权重设置主观", "可能掩盖真实问题"],
            risk_mitigation=["基于数据分析", "多维度评估"],
            expected_effectiveness=0.7,
            confidence_score=0.6,
            complexity="medium",
            estimated_time_minutes=15
        )

    def prioritize_fixes(self, auto_fixes: List[AutoFixSuggestion],
                         manual_fixes: List[ManualFixRecommendation]) -> Dict[str, List]:
        """优先级排序修复建议"""

        # 按类型和优先级分组
        prioritized = {
            "immediate_auto": [],  # 可立即自动应用的高信心修复
            "review_auto": [],  # 需要审查的自动修复
            "high_priority_manual": [],  # 高优先级手动修复
            "medium_priority_manual": [],  # 中等优先级手动修复
            "low_priority_manual": [],  # 低优先级手动修复
            "config_adjustments": []  # 配置调整
        }

        # 分类自动修复
        for fix in auto_fixes:
            if (fix.can_be_auto_applied and
                    fix.auto_apply_confidence >= 0.8 and
                    fix.expected_effectiveness >= 0.7):
                prioritized["immediate_auto"].append(fix)
            else:
                prioritized["review_auto"].append(fix)

        # 分类手动修复
        for fix in manual_fixes:
            if fix.fix_type == FixType.CONFIG:
                prioritized["config_adjustments"].append(fix)
            elif fix.complexity == "high" and "critical" in fix.fix_id:
                prioritized["high_priority_manual"].append(fix)
            elif fix.complexity == "medium":
                prioritized["medium_priority_manual"].append(fix)
            else:
                prioritized["low_priority_manual"].append(fix)

        # 排序每个类别
        for category in prioritized:
            if category == "immediate_auto":
                prioritized[category].sort(key=lambda x: x.auto_apply_confidence, reverse=True)
            elif category in ["high_priority_manual", "medium_priority_manual", "low_priority_manual"]:
                prioritized[category].sort(key=lambda x: x.expected_effectiveness, reverse=True)

        return prioritized
