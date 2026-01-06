"""
@FileName: constraint_validator.py
@Description: 约束验证器 - 验证智能体4的输出是否满足智能体3的约束
@Author: HengLine
@Time: 2026/1/6 16:03
"""

import re
from typing import List, Dict, Any, Optional

from .model.check_models import ConstraintCheckResult, CheckStatus
from .model.issue_models import IssueSeverity


class ConstraintValidator:
    """约束验证器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.tolerance_levels = {
            "critical": 0.0,  # 零容忍
            "high": 0.1,  # 低容忍
            "medium": 0.3,  # 中等容忍
            "low": 0.5  # 高容忍
        }

    def validate_all_constraints(self, shots: List[Any],
                                 anchored_timeline: Any) -> List[ConstraintCheckResult]:
        """验证所有约束"""
        results = []

        # 获取所有约束
        constraints = []
        for segment in anchored_timeline.anchored_segments:
            constraints.extend(segment.hard_constraints)

        # 为每个约束检查每个相关镜头
        for constraint in constraints:
            # 确定哪些镜头需要检查
            applicable_shots = self._get_applicable_shots(shots, constraint)

            if not applicable_shots:
                continue

            # 检查约束
            for shot in applicable_shots:
                result = self.validate_constraint_for_shot(constraint, shot)
                if result:
                    results.append(result)

        return results

    def _get_applicable_shots(self, shots: List[Any], constraint: Any) -> List[Any]:
        """获取需要检查约束的镜头"""
        applicable_shots = []

        # 根据约束的适用范围筛选镜头
        for shot in shots:
            # 检查时间范围
            if self._is_shot_in_time_range(shot, constraint):
                # 检查片段ID
                if self._is_shot_in_segment(shot, constraint):
                    applicable_shots.append(shot)

        return applicable_shots

    def _is_shot_in_time_range(self, shot: Any, constraint: Any) -> bool:
        """检查镜头是否在约束的时间范围内"""
        if not constraint.temporal_range:
            return True  # 如果没有时间范围，检查所有镜头

        start_time, end_time = constraint.temporal_range
        shot_start, shot_end = shot.time_range

        # 检查是否有重叠
        return not (shot_end <= start_time or shot_start >= end_time)

    def _is_shot_in_segment(self, shot: Any, constraint: Any) -> bool:
        """检查镜头是否在约束适用的片段内"""
        if not constraint.applicable_segments:
            return True

        # 检查shot的segment_id是否在适用片段列表中
        # 注意：这里假设shot.segment_id格式为"s001_shot1"，需要提取基础片段ID
        base_segment_id = shot.segment_id.split('_')[0] if '_' in shot.segment_id else shot.segment_id
        return base_segment_id in constraint.applicable_segments

    def validate_constraint_for_shot(self, constraint: Any, shot: Any) -> Optional[ConstraintCheckResult]:
        """为单个镜头验证约束"""

        # 确定约束类型
        constraint_type = self._classify_constraint(constraint)

        # 根据类型选择验证方法
        if constraint_type == "character_appearance":
            return self._validate_character_appearance(constraint, shot)
        elif constraint_type == "prop_state":
            return self._validate_prop_state(constraint, shot)
        elif constraint_type == "camera_angle":
            return self._validate_camera_angle(constraint, shot)
        elif constraint_type == "environment":
            return self._validate_environment(constraint, shot)
        else:
            return self._validate_general_constraint(constraint, shot)

    def _classify_constraint(self, constraint: Any) -> str:
        """分类约束类型"""
        if hasattr(constraint, 'type'):
            return constraint.type
        else:
            # 从描述中推断
            description = constraint.description.lower()

            if any(keyword in description for keyword in ["穿着", "服装", "发型", "妆容"]):
                return "character_appearance"
            elif any(keyword in description for keyword in ["拿着", "手持", "杯子", "道具"]):
                return "prop_state"
            elif any(keyword in description for keyword in ["镜头", "特写", "角度", "拍摄"]):
                return "camera_angle"
            elif any(keyword in description for keyword in ["环境", "场景", "灯光", "房间"]):
                return "environment"
            else:
                return "general"

    def _validate_character_appearance(self, constraint: Any, shot: Any) -> ConstraintCheckResult:
        """验证角色外观约束"""

        # 从约束描述中提取要求
        requirements = self._parse_appearance_requirements(constraint.description)

        # 从镜头提示词中提取实际外观
        actual_appearance = self._extract_appearance_from_prompt(shot.full_sora_prompt)

        # 比较
        violations = []
        satisfied = True

        for requirement in requirements:
            if not self._check_appearance_requirement(requirement, actual_appearance):
                violations.append(f"不满足要求: {requirement}")
                satisfied = False

        # 计算满足度得分
        satisfaction_score = 1.0 if satisfied else 0.0

        # 如果有Sora指令，也检查
        sora_satisfied = False
        if hasattr(constraint, 'sora_instruction'):
            sora_satisfied = constraint.sora_instruction.lower() in shot.full_sora_prompt.lower()
            if not sora_satisfied and satisfied:  # 如果描述满足但指令不满足
                satisfaction_score = 0.5

        is_satisfied = satisfied or sora_satisfied
        final_score = max(satisfaction_score, 1.0 if sora_satisfied else 0.0)

        return ConstraintCheckResult(
            check_id=f"constraint_{constraint.constraint_id}_shot_{shot.shot_id}",
            check_name=f"约束验证: {constraint.constraint_id}",
            check_description=f"验证约束'{constraint.description}'",
            status=CheckStatus.PASSED if is_satisfied else CheckStatus.FAILED,
            severity=self._priority_to_severity(constraint.priority),
            score=final_score,
            constraint_id=constraint.constraint_id,
            constraint_type=constraint.type,
            constraint_description=constraint.description,
            constraint_priority=constraint.priority,
            expected_value=requirements,
            actual_value=actual_appearance,
            deviation=1.0 - final_score,
            tolerance=self.tolerance_levels.get(self._priority_to_tolerance_level(constraint.priority), 0.3),
            is_satisfied=is_satisfied,
            satisfaction_score=final_score,
            affected_shots=[shot.shot_id],
            fix_complexity="low" if not is_satisfied else "none",
            fix_suggestion="在提示词中明确添加约束要求" if not is_satisfied else None,
            details={
                "requirements": requirements,
                "actual_appearance": actual_appearance,
                "violations": violations if not is_satisfied else [],
                "sora_instruction_present": sora_satisfied
            },
            evidence=violations if not is_satisfied else ["约束已满足"]
        )

    def _parse_appearance_requirements(self, description: str) -> List[str]:
        """解析外观要求"""
        requirements = []

        # 提取关键词
        patterns = [
            r"穿着\s*([^，。]+)",
            r"服装\s*[：:]\s*([^，。]+)",
            r"发型\s*[：:]\s*([^，。]+)",
            r"妆容\s*[：:]\s*([^，。]+)"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, description)
            requirements.extend(matches)

        return requirements

    def _extract_appearance_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """从提示词中提取外观信息"""
        appearance = {
            "clothing": [],
            "hairstyle": [],
            "accessories": []
        }

        prompt_lower = prompt.lower()

        # 简单的关键词匹配
        clothing_keywords = ["wearing", "dressed", "clothing", "shirt", "dress", "jacket"]
        for keyword in clothing_keywords:
            if keyword in prompt_lower:
                # 提取相关短语
                appearance["clothing"].append(keyword)

        return appearance

    def _check_appearance_requirement(self, requirement: str, appearance: Dict[str, Any]) -> bool:
        """检查单个外观要求"""
        # 简化检查：如果要求中的关键词出现在外观描述中
        requirement_lower = requirement.lower()

        for category in appearance.values():
            if isinstance(category, list):
                for item in category:
                    if requirement_lower in item or item in requirement_lower:
                        return True
        return False

    def _validate_prop_state(self, constraint: Any, shot: Any) -> ConstraintCheckResult:
        """验证道具状态约束"""

        # 检查约束是否出现在提示词中
        constraint_in_prompt = False

        if hasattr(constraint, 'description'):
            constraint_in_prompt = constraint.description.lower() in shot.full_sora_prompt.lower()

        if hasattr(constraint, 'sora_instruction') and not constraint_in_prompt:
            constraint_in_prompt = constraint.sora_instruction.lower() in shot.full_sora_prompt.lower()

        is_satisfied = constraint_in_prompt
        score = 1.0 if is_satisfied else 0.0

        return ConstraintCheckResult(
            check_id=f"constraint_{constraint.constraint_id}_shot_{shot.shot_id}",
            check_name=f"道具约束验证: {constraint.constraint_id}",
            check_description=f"验证道具约束'{constraint.description}'",
            status=CheckStatus.PASSED if is_satisfied else CheckStatus.FAILED,
            severity=self._priority_to_severity(constraint.priority),
            score=score,
            constraint_id=constraint.constraint_id,
            constraint_type=constraint.type,
            constraint_description=constraint.description,
            constraint_priority=constraint.priority,
            is_satisfied=is_satisfied,
            satisfaction_score=score,
            affected_shots=[shot.shot_id],
            fix_complexity="low" if not is_satisfied else "none",
            fix_suggestion="在提示词中添加道具描述" if not is_satisfied else None,
            details={
                "constraint_in_prompt": constraint_in_prompt,
                "prompt_snippet": shot.full_sora_prompt[:100] + "..."
            },
            evidence=["约束在提示词中找到"] if is_satisfied else ["约束未在提示词中找到"]
        )

    def _validate_camera_angle(self, constraint: Any, shot: Any) -> ConstraintCheckResult:
        """验证相机角度约束"""

        # 从约束中提取相机要求
        camera_requirement = self._parse_camera_requirement(constraint)

        # 从镜头中提取相机参数
        actual_camera = self._extract_camera_from_shot(shot)

        # 比较
        match_score = self._compare_camera_requirements(camera_requirement, actual_camera)

        is_satisfied = match_score >= 0.7  # 70%匹配即认为满足
        score = match_score

        return ConstraintCheckResult(
            check_id=f"constraint_{constraint.constraint_id}_shot_{shot.shot_id}",
            check_name=f"相机约束验证: {constraint.constraint_id}",
            check_description=f"验证相机约束'{constraint.description}'",
            status=CheckStatus.PASSED if is_satisfied else CheckStatus.FAILED,
            severity=self._priority_to_severity(constraint.priority),
            score=score,
            constraint_id=constraint.constraint_id,
            constraint_type=constraint.type,
            constraint_description=constraint.description,
            constraint_priority=constraint.priority,
            expected_value=camera_requirement,
            actual_value=actual_camera,
            deviation=1.0 - score,
            tolerance=0.3,
            is_satisfied=is_satisfied,
            satisfaction_score=score,
            affected_shots=[shot.shot_id],
            fix_complexity="medium" if not is_satisfied else "none",
            fix_suggestion="调整相机参数以匹配约束要求" if not is_satisfied else None,
            details={
                "requirement": camera_requirement,
                "actual": actual_camera,
                "match_score": match_score
            },
            evidence=[f"匹配度: {match_score:.2%}"]
        )

    def _parse_camera_requirement(self, constraint: Any) -> Dict[str, Any]:
        """解析相机要求"""
        requirement = {}

        if hasattr(constraint, 'sora_instruction'):
            instruction = constraint.sora_instruction.lower()

            if "close-up" in instruction or "特写" in constraint.description:
                requirement["shot_size"] = "close_up"
            elif "wide" in instruction or "广角" in constraint.description:
                requirement["shot_size"] = "wide_shot"
            elif "medium" in instruction or "中景" in constraint.description:
                requirement["shot_size"] = "medium_shot"

        return requirement

    def _extract_camera_from_shot(self, shot: Any) -> Dict[str, Any]:
        """从镜头中提取相机参数"""
        camera = {}

        if hasattr(shot, 'camera_parameters'):
            camera["shot_size"] = shot.camera_parameters.shot_size.value
            camera["movement"] = shot.camera_parameters.camera_movement.value

        return camera

    def _compare_camera_requirements(self, requirement: Dict[str, Any],
                                     actual: Dict[str, Any]) -> float:
        """比较相机要求"""
        if not requirement or not actual:
            return 0.0

        match_count = 0
        total_count = 0

        for key, req_value in requirement.items():
            total_count += 1
            if key in actual and actual[key] == req_value:
                match_count += 1

        return match_count / total_count if total_count > 0 else 0.0

    def _validate_environment(self, constraint: Any, shot: Any) -> ConstraintCheckResult:
        """验证环境约束"""
        # 简化实现：检查约束是否出现在提示词中
        constraint_in_prompt = constraint.description.lower() in shot.full_sora_prompt.lower()

        if hasattr(constraint, 'sora_instruction') and not constraint_in_prompt:
            constraint_in_prompt = constraint.sora_instruction.lower() in shot.full_sora_prompt.lower()

        is_satisfied = constraint_in_prompt
        score = 1.0 if is_satisfied else 0.0

        return ConstraintCheckResult(
            check_id=f"constraint_{constraint.constraint_id}_shot_{shot.shot_id}",
            check_name=f"环境约束验证: {constraint.constraint_id}",
            check_description=f"验证环境约束'{constraint.description}'",
            status=CheckStatus.PASSED if is_satisfied else CheckStatus.FAILED,
            severity=self._priority_to_severity(constraint.priority),
            score=score,
            constraint_id=constraint.constraint_id,
            constraint_type=constraint.type,
            constraint_description=constraint.description,
            constraint_priority=constraint.priority,
            is_satisfied=is_satisfied,
            satisfaction_score=score,
            affected_shots=[shot.shot_id],
            fix_complexity="low" if not is_satisfied else "none",
            fix_suggestion="在提示词中添加环境描述" if not is_satisfied else None,
            details={
                "constraint_in_prompt": constraint_in_prompt
            },
            evidence=["环境约束在提示词中找到"] if is_satisfied else ["环境约束未在提示词中找到"]
        )

    def _validate_general_constraint(self, constraint: Any, shot: Any) -> ConstraintCheckResult:
        """验证通用约束"""
        # 检查约束是否出现在提示词中
        constraint_in_prompt = constraint.description.lower() in shot.full_sora_prompt.lower()

        if hasattr(constraint, 'sora_instruction') and not constraint_in_prompt:
            constraint_in_prompt = constraint.sora_instruction.lower() in shot.full_sora_prompt.lower()

        is_satisfied = constraint_in_prompt
        score = 1.0 if is_satisfied else 0.0

        return ConstraintCheckResult(
            check_id=f"constraint_{constraint.constraint_id}_shot_{shot.shot_id}",
            check_name=f"通用约束验证: {constraint.constraint_id}",
            check_description=f"验证约束'{constraint.description}'",
            status=CheckStatus.PASSED if is_satisfied else CheckStatus.FAILED,
            severity=self._priority_to_severity(constraint.priority),
            score=score,
            constraint_id=constraint.constraint_id,
            constraint_type=constraint.type,
            constraint_description=constraint.description,
            constraint_priority=constraint.priority,
            is_satisfied=is_satisfied,
            satisfaction_score=score,
            affected_shots=[shot.shot_id],
            fix_complexity="low" if not is_satisfied else "none",
            fix_suggestion="在提示词中明确添加约束要求" if not is_satisfied else None,
            details={
                "constraint_in_prompt": constraint_in_prompt
            },
            evidence=["约束在提示词中找到"] if is_satisfied else ["约束未在提示词中找到"]
        )

    def _priority_to_severity(self, priority: int) -> IssueSeverity:
        """将优先级转换为严重性级别"""
        if priority >= 9:
            return IssueSeverity.CRITICAL
        elif priority >= 7:
            return IssueSeverity.HIGH
        elif priority >= 5:
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.LOW

    def _priority_to_tolerance_level(self, priority: int) -> str:
        """将优先级转换为容忍度级别"""
        if priority >= 9:
            return "critical"
        elif priority >= 7:
            return "high"
        elif priority >= 5:
            return "medium"
        else:
            return "low"

    def calculate_constraint_scores(self, results: List[ConstraintCheckResult]) -> Dict[str, float]:
        """计算约束满足度评分"""
        if not results:
            return {
                "overall": 1.0,
                "critical": 1.0,
                "high": 1.0,
                "medium": 1.0,
                "low": 1.0
            }

        # 按优先级分组
        scores_by_priority = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }

        for result in results:
            priority_level = self._score_to_priority_level(result.constraint_priority)
            scores_by_priority[priority_level].append(result.satisfaction_score)

        # 计算各优先级平均分
        avg_scores = {}
        for level, scores in scores_by_priority.items():
            if scores:
                avg_scores[level] = sum(scores) / len(scores)
            else:
                avg_scores[level] = 1.0  # 默认满分

        # 计算总分（加权）
        weights = {
            "critical": 0.4,
            "high": 0.3,
            "medium": 0.2,
            "low": 0.1
        }

        overall_score = sum(avg_scores[level] * weights[level] for level in avg_scores)
        avg_scores["overall"] = overall_score

        return avg_scores

    def _score_to_priority_level(self, priority: int) -> str:
        """将分数转换为优先级级别"""
        if priority >= 9:
            return "critical"
        elif priority >= 7:
            return "high"
        elif priority >= 5:
            return "medium"
        else:
            return "low"
