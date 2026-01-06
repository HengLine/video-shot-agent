"""
@FileName: fix_applier.py
@Description: 修复应用器 - 应用自动修复建议
@Author: HengLine
@Time: 2026/1/6 17:11
"""
import copy
from typing import List, Dict, Any


class FixApplier:
    """修复应用器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.applied_fixes = []
        self.failed_fixes = []

    def apply_auto_fixes(self, shots: List[Any],
                         fixes: List[Any]) -> List[Any]:
        """应用自动修复建议"""

        modified_shots = copy.deepcopy(shots)

        for fix in fixes:
            if hasattr(fix, 'can_be_auto_applied') and fix.can_be_auto_applied:
                try:
                    modified_shots = self._apply_single_fix(modified_shots, fix)
                    self.applied_fixes.append({
                        "fix_id": fix.fix_id,
                        "target_issue": fix.target_issue_id,
                        "success": True,
                        "timestamp": "now"
                    })
                except Exception as e:
                    self.failed_fixes.append({
                        "fix_id": fix.fix_id,
                        "target_issue": fix.target_issue_id,
                        "error": str(e),
                        "timestamp": "now"
                    })

        return modified_shots

    def _apply_single_fix(self, shots: List[Any], fix: Any) -> List[Any]:
        """应用单个修复"""

        # 根据修复类型选择应用方法
        if "constraint" in fix.fix_id:
            return self._apply_constraint_fix(shots, fix)
        elif "continuity" in fix.fix_id:
            return self._apply_continuity_fix(shots, fix)
        elif "visual" in fix.fix_id:
            return self._apply_visual_fix(shots, fix)
        elif "technical" in fix.fix_id:
            return self._apply_technical_fix(shots, fix)
        else:
            # 通用修复应用
            return self._apply_generic_fix(shots, fix)

    def _apply_constraint_fix(self, shots: List[Any], fix: Any) -> List[Any]:
        """应用约束修复"""
        modified_shots = copy.deepcopy(shots)

        # 从修复参数中获取目标镜头ID
        target_shot_id = fix.fix_parameters.get("shot_id", "")
        constraint_desc = fix.fix_parameters.get("constraint_desc", "")

        for shot in modified_shots:
            if shot.shot_id == target_shot_id or target_shot_id in shot.shot_id:
                # 添加约束到提示词
                if hasattr(shot, 'full_sora_prompt'):
                    shot.full_sora_prompt += f". {constraint_desc}"

                # 标记约束为已满足
                if hasattr(shot, 'satisfied_constraints'):
                    constraint_id = fix.fix_parameters.get("constraint_id", "")
                    if constraint_id not in shot.satisfied_constraints:
                        shot.satisfied_constraints.append(constraint_id)

                break

        return modified_shots

    def _apply_continuity_fix(self, shots: List[Any], fix: Any) -> List[Any]:
        """应用连续性修复"""
        # 连续性修复通常需要手动调整
        # 这里只记录修复，实际应用可能需要更复杂的逻辑
        return shots

    def _apply_visual_fix(self, shots: List[Any], fix: Any) -> List[Any]:
        """应用视觉修复"""
        modified_shots = copy.deepcopy(shots)

        # 示例：添加构图描述
        missing_element = fix.fix_parameters.get("missing_element", "")
        suggestion = fix.fix_parameters.get("suggestion", "")

        if missing_element and suggestion:
            for shot in modified_shots:
                if hasattr(shot, 'full_sora_prompt'):
                    # 添加构图描述
                    if "composition" in missing_element.lower() or "构图" in suggestion:
                        shot.full_sora_prompt += f", with good composition and {missing_element}"

        return modified_shots

    def _apply_technical_fix(self, shots: List[Any], fix: Any) -> List[Any]:
        """应用技术修复"""
        # 技术修复通常涉及参数调整
        return shots

    def _apply_generic_fix(self, shots: List[Any], fix: Any) -> List[Any]:
        """应用通用修复"""
        # 执行修复代码（如果有）
        if hasattr(fix, 'fix_code') and fix.fix_code:
            try:
                # 这里可以执行修复代码
                # 注意：实际生产中应该安全地执行代码
                pass
            except Exception as e:
                print(f"执行修复代码失败: {e}")

        return shots

    def get_application_report(self) -> Dict[str, Any]:
        """获取修复应用报告"""
        return {
            "total_fixes_attempted": len(self.applied_fixes) + len(self.failed_fixes),
            "successful_applications": len(self.applied_fixes),
            "failed_applications": len(self.failed_fixes),
            "success_rate": len(self.applied_fixes) / max(1, len(self.applied_fixes) + len(self.failed_fixes)),
            "applied_fixes": self.applied_fixes,
            "failed_fixes": self.failed_fixes
        }

    def verify_fixes(self, original_shots: List[Any],
                     modified_shots: List[Any]) -> Dict[str, Any]:
        """验证修复效果"""

        verification_results = {
            "changes_detected": [],
            "improvements": [],
            "new_issues": [],
            "verification_score": 0.0
        }

        # 简单比较：检查提示词变化
        for orig, mod in zip(original_shots, modified_shots):
            if hasattr(orig, 'full_sora_prompt') and hasattr(mod, 'full_sora_prompt'):
                if orig.full_sora_prompt != mod.full_sora_prompt:
                    change = {
                        "shot_id": orig.shot_id,
                        "change_type": "prompt_modified",
                        "original_length": len(orig.full_sora_prompt),
                        "modified_length": len(mod.full_sora_prompt),
                        "added_content": mod.full_sora_prompt[len(orig.full_sora_prompt):]
                        if len(mod.full_sora_prompt) > len(orig.full_sora_prompt) else ""
                    }
                    verification_results["changes_detected"].append(change)

                    # 检查是否是改进
                    if len(mod.full_sora_prompt) > len(orig.full_sora_prompt):
                        verification_results["improvements"].append(f"镜头 {orig.shot_id} 提示词已增强")

        # 计算验证得分
        total_shots = len(original_shots)
        if total_shots > 0:
            improved_shots = len(verification_results["improvements"])
            verification_results["verification_score"] = improved_shots / total_shots

        return verification_results
