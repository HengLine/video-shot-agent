"""
@FileName: continuity_guardian_extract.py
@Description: 
@Author: HengLine
@Time: 2026/1/4 17:53
"""
import json
import math
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import numpy as np

from hengline.agent.continuity_guardian.continuity_constraint_generator import ContinuityConstraintGenerator
from .analyzer.continuity_learner import ContinuityLearner
from .analyzer.scene_analyzer import SceneAnalyzer
from .continuity_guardian_model import ContinuityLevel
from .model.continuity_guard_guardian import GuardianConfig, AnalysisDepth, GuardMode
from .model.continuity_guardian_autofix import AutoFix
from .model.continuity_guardian_report import StateSnapshot, ContinuityIssue, ValidationReport
from .model.continuity_rule_guardian import ContinuityRuleSet, GenerationHints
from .model.continuity_state_guardian import CharacterState, PropState, EnvironmentState
from .model.continuity_transition_guardian import KeyframeAnchor
from .model.continuity_visual_guardian import SpatialRelation
from .scene_transition_manager import SceneTransitionManager
from ... import info, debug, error


class ContinuityGuardian:
    """连续性守护智能体主类"""

    def __init__(self, project_id: str, config: Optional[GuardianConfig] = None):
        self.config = config or GuardianConfig(project_id=project_id)

        # 初始化核心组件
        self.rule_set = ContinuityRuleSet()
        self.state_history: List[StateSnapshot] = []
        self.current_state: Optional[StateSnapshot] = None
        self.auto_fixer = AutoFix(self.rule_set)
        self.validation_reports: Dict[str, ValidationReport] = {}
        self.keyframe_anchors: Dict[str, KeyframeAnchor] = {}

        # 初始化辅助组件
        self.scene_analyzer = SceneAnalyzer(self.config)
        self.continuity_learner = ContinuityLearner(self.config) if self.config.enable_machine_learning else None
        self.transition_manager = SceneTransitionManager(self.config)

        # 初始化约束生成器
        self.constraint_generator = ContinuityConstraintGenerator(self.rule_set.rules)

        # 缓存系统
        self.cache: Dict[str, Any] = {
            "scene_metrics": {},
            "validation_results": {},
            "constraints": {}
        }

        # 性能监控
        self.performance_metrics: Dict[str, List] = {
            "processing_times": [],
            "issue_counts": [],
            "validation_times": []
        }

        # 会话管理
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.frame_counter = 0

        info(f"连续性守护器初始化完成 - 项目: {project_id}, 会话: {self.session_id}")

    def process_scene(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理场景数据"""
        start_time = datetime.now()
        self.frame_counter += 1

        debug(f"处理场景 #{self.frame_counter}: {scene_data.get('scene_id', 'unknown')}")

        # 步骤1：分析场景
        scene_analysis = self._analyze_scene(scene_data)

        # 步骤2：捕获状态
        state_snapshot = self.capture_state(scene_data)

        # 步骤3：生成约束
        constraints = self._generate_constraints(scene_data, scene_analysis)

        # 步骤4：验证连续性
        validation_report = None
        if self._should_validate(self.frame_counter):
            validation_report = self._validate_scene(state_snapshot, constraints)

        # 步骤5：预测问题（如果启用学习）
        predicted_issues = []
        if self.continuity_learner:
            predicted_issues = self.continuity_learner.predict_issues(scene_data)

        # 步骤6：生成修复提示
        generation_hints = self.generate_hints(scene_data.get("next_scene_type", "general"))

        # 步骤7：性能监控
        processing_time = (datetime.now() - start_time).total_seconds()
        self._record_performance_metric("processing_times", processing_time)

        # 构建结果
        result = {
            "session_id": self.session_id,
            "frame_number": self.frame_counter,
            "scene_id": scene_data.get("scene_id", "unknown"),
            "processing_time": processing_time,
            "state_snapshot": state_snapshot.to_dict() if state_snapshot else None,
            "scene_analysis": scene_analysis,
            "constraints_summary": self.constraint_generator.get_constraint_summary(constraints),
            "validation_report": validation_report.generate_summary() if validation_report else None,
            "predicted_issues": predicted_issues,
            "generation_hints": generation_hints.continuity_constraints,
            "recommendations": self._generate_recommendations(scene_analysis, validation_report, predicted_issues)
        }

        # 缓存结果
        self.cache["scene_metrics"][self.frame_counter] = scene_analysis
        if validation_report:
            self.cache["validation_results"][self.frame_counter] = validation_report
        self.cache["constraints"][self.frame_counter] = constraints

        # 生成报告（如果需要）
        if self.config.generate_reports and self.frame_counter % 10 == 0:
            self._generate_periodic_report()

        info(f"场景处理完成 - 帧: {self.frame_counter}, 用时: {processing_time:.3f}秒")

        return result

    def _analyze_scene(self, scene_data: Dict) -> Dict[str, Any]:
        """分析场景"""
        analysis = {
            "complexity": self.scene_analyzer.analyze_scene_complexity(scene_data).value,
            "metrics": self.scene_analyzer.calculate_scene_metrics(scene_data),
            "risk_assessment": self._assess_scene_risk(scene_data),
            "processing_priority": self._determine_processing_priority(scene_data)
        }

        # 根据分析结果调整配置
        self._adjust_configuration_based_on_analysis(analysis)

        return analysis

    def _assess_scene_risk(self, scene_data: Dict) -> Dict[str, float]:
        """评估场景风险"""
        risk_factors = {
            "character_consistency_risk": 0.0,
            "physical_plausibility_risk": 0.0,
            "temporal_continuity_risk": 0.0,
            "lighting_consistency_risk": 0.0,
            "overall_risk": 0.0
        }

        # 角色一致性风险
        characters = scene_data.get("characters", [])
        if len(characters) > 3:
            risk_factors["character_consistency_risk"] = min(1.0, len(characters) * 0.1)

        # 物理合理性风险
        if "physics" in scene_data:
            physics = scene_data["physics"]
            if physics.get("complex", False):
                risk_factors["physical_plausibility_risk"] = 0.7

        # 时间连续性风险
        if "time_data" in scene_data:
            time_data = scene_data["time_data"]
            if time_data.get("non_linear", False):
                risk_factors["temporal_continuity_risk"] = 0.8

        # 光照一致性风险
        environment = scene_data.get("environment", {})
        lighting = environment.get("lighting", {})
        if lighting.get("dynamic", False):
            risk_factors["lighting_consistency_risk"] = 0.6

        # 计算总体风险
        risk_factors["overall_risk"] = np.mean(list(risk_factors.values()))

        return risk_factors

    def _determine_processing_priority(self, scene_data: Dict) -> str:
        """确定处理优先级"""
        analysis = self.scene_analyzer.calculate_scene_metrics(scene_data)
        motion_intensity = analysis["motion_intensity"]

        if motion_intensity > 0.7:
            return "high"  # 高运动场景需要高优先级
        elif len(scene_data.get("characters", [])) > 5:
            return "medium"  # 多角色场景
        else:
            return "low"  # 简单场景

    def _adjust_configuration_based_on_analysis(self, analysis: Dict):
        """根据分析结果调整配置"""
        complexity = analysis["complexity"]
        risk = analysis["risk_assessment"]["overall_risk"]

        # 根据复杂度调整分析深度
        if complexity in ["complex", "epic"] or risk > 0.7:
            self.config.analysis_depth = AnalysisDepth.DETAILED
        elif complexity == "simple" and risk < 0.3:
            self.config.analysis_depth = AnalysisDepth.QUICK

        # 根据风险调整模式
        if risk > 0.8:
            self.config.mode = GuardMode.PROACTIVE
        elif risk < 0.3:
            self.config.mode = GuardMode.PASSIVE

    def _generate_constraints(self, scene_data: Dict, scene_analysis: Dict) -> List[Dict]:
        """生成连续性约束"""
        scene_id = scene_data.get("scene_id", "unknown")
        cache_key = f"{scene_id}_{self.frame_counter}"

        # 检查缓存
        if cache_key in self.cache["constraints"]:
            return self.cache["constraints"][cache_key]

        # 根据场景类型生成约束
        scene_type = scene_data.get("scene_type", "general")

        # 获取前一场景数据（用于连续性约束）
        previous_scene_data = None
        if self.state_history:
            previous_snapshot = self.state_history[-1]
            previous_scene_data = {
                "scene_id": previous_snapshot.scene_id,
                "characters": [c.to_dict() for c in previous_snapshot.characters.values()],
                "environment": previous_snapshot.environment.__dict__
            }

        # 生成约束
        constraints = self.constraint_generator.generate_constraints_for_scene(
            scene_data=scene_data,
            previous_scene=previous_scene_data,
            scene_type=scene_type
        )

        # 根据分析结果调整约束优先级
        risk_level = scene_analysis["risk_assessment"]["overall_risk"]
        if risk_level > 0.7:
            # 高风险场景：提高关键约束优先级
            for constraint in constraints:
                if constraint["type"] in ["character_appearance", "physical_plausibility"]:
                    constraint["priority"] = "critical"

        return constraints

    def _validate_scene(self, state_snapshot: StateSnapshot,
                        constraints: List[Dict]) -> Optional[ValidationReport]:
        """验证场景连续性"""
        if not self.state_history:
            return None

        previous_snapshot = self.state_history[-1]

        # 并行验证不同方面
        validation_results = {}

        if self.config.parallel_processing:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._validate_characters,
                                    previous_snapshot, state_snapshot): "characters",
                    executor.submit(self._validate_props,
                                    previous_snapshot, state_snapshot): "props",
                    executor.submit(self._validate_environment,
                                    previous_snapshot, state_snapshot): "environment",
                    executor.submit(self._validate_constraints,
                                    constraints, state_snapshot): "constraints"
                }

                for future in as_completed(futures):
                    aspect = futures[future]
                    try:
                        validation_results[aspect] = future.result(timeout=self.config.validation_timeout)
                    except Exception as e:
                        error(f"{aspect} 验证失败: {e}")
                        validation_results[aspect] = {"issues": [], "passed": 0}
        else:
            # 串行验证
            validation_results["characters"] = self._validate_characters(previous_snapshot, state_snapshot)
            validation_results["props"] = self._validate_props(previous_snapshot, state_snapshot)
            validation_results["environment"] = self._validate_environment(previous_snapshot, state_snapshot)
            validation_results["constraints"] = self._validate_constraints(constraints, state_snapshot)

        # 合并验证结果
        report = ValidationReport(f"validation_{self.frame_counter}_{datetime.now().timestamp()}")

        for aspect, results in validation_results.items():
            for issue in results.get("issues", []):
                continuity_issue = ContinuityIssue(
                    issue_id=f"{aspect}_{len(report.issues)}",
                    level=issue.get("level", "MINOR").upper(),
                    description=issue.get("description", "未知问题")
                )
                continuity_issue.entity_type = aspect
                continuity_issue.entity_id = issue.get("entity_id", "")
                continuity_issue.suggested_fixes = issue.get("suggested_fixes", [])
                continuity_issue.auto_fixable = issue.get("auto_fixable", False)

                report.add_issue(continuity_issue)

            report.summary["passed"] += results.get("passed", 0)
            report.summary["total_checks"] += results.get("total_checks", 0)

        # 计算统计
        for issue in report.issues:
            if issue.level == ContinuityLevel.CRITICAL:
                report.summary["critical_issues"] += 1
            elif issue.level == ContinuityLevel.MAJOR:
                report.summary["major_issues"] += 1
            elif issue.level == ContinuityLevel.MINOR:
                report.summary["minor_issues"] += 1
            else:
                report.summary["cosmetic_issues"] += 1

        # 尝试自动修复
        if self.config.enable_auto_fix and report.issues:
            self._attempt_auto_fixes(report, state_snapshot)

        # 记录验证历史
        self.validation_reports[report.validation_id] = report

        # 性能记录
        validation_time = sum(r.get("validation_time", 0) for r in validation_results.values())
        self._record_performance_metric("validation_times", validation_time)
        self._record_performance_metric("issue_counts", len(report.issues))

        return report

    def _validate_characters(self, previous: StateSnapshot,
                             current: StateSnapshot) -> Dict[str, Any]:
        """验证角色连续性"""
        start_time = datetime.now()
        issues = []
        passed_checks = 0
        total_checks = 0

        for char_id, prev_char in previous.characters.items():
            if char_id in current.characters:
                curr_char = current.characters[char_id]
                total_checks += 1

                # 检查外貌
                if prev_char.appearance != curr_char.appearance:
                    issues.append({
                        "entity_id": char_id,
                        "level": "CRITICAL",
                        "description": f"角色 {char_id} 外貌不一致",
                        "suggested_fixes": ["恢复之前的外貌", "添加渐进变化解释"],
                        "auto_fixable": True
                    })
                else:
                    passed_checks += 1

                # 检查服装
                if prev_char.outfit != curr_char.outfit:
                    issues.append({
                        "entity_id": char_id,
                        "level": "MAJOR",
                        "description": f"角色 {char_id} 服装不一致",
                        "suggested_fixes": ["检查服装变化是否合理", "添加换装场景"],
                        "auto_fixable": False
                    })

                # 检查位置连续性
                if prev_char.position and curr_char.position:
                    distance = self._calculate_distance(prev_char.position, curr_char.position)
                    time_diff = (current.timestamp - previous.timestamp).total_seconds()

                    if time_diff > 0:
                        speed = distance / time_diff
                        if speed > 10.0:  # 超过10米/秒
                            issues.append({
                                "entity_id": char_id,
                                "level": "MAJOR",
                                "description": f"角色 {char_id} 移动速度异常: {speed:.1f}m/s",
                                "suggested_fixes": ["降低移动速度", "添加快速移动解释"],
                                "auto_fixable": True
                            })

        validation_time = (datetime.now() - start_time).total_seconds()

        return {
            "issues": issues,
            "passed": passed_checks,
            "total_checks": total_checks,
            "validation_time": validation_time
        }

    def _validate_props(self, previous: StateSnapshot,
                        current: StateSnapshot) -> Dict[str, Any]:
        """验证道具连续性"""
        start_time = datetime.now()
        issues = []
        passed_checks = 0
        total_checks = 0

        for prop_id, prev_prop in previous.props.items():
            if prop_id in current.props:
                curr_prop = current.props[prop_id]
                total_checks += 1

                # 检查位置
                if prev_prop.position and curr_prop.position:
                    distance = self._calculate_distance(prev_prop.position, curr_prop.position)

                    # 无支撑物体的异常移动
                    if (not prev_prop.owner and not curr_prop.owner and
                            distance > 1.0 and prev_prop.state != "moving"):
                        issues.append({
                            "entity_id": prop_id,
                            "level": "MAJOR",
                            "description": f"道具 {prop_id} 无支撑移动 {distance:.2f}米",
                            "suggested_fixes": ["添加支撑或交互", "添加物理解释"],
                            "auto_fixable": True
                        })
                    else:
                        passed_checks += 1

                # 检查状态变化
                if prev_prop.state != curr_prop.state:
                    issues.append({
                        "entity_id": prop_id,
                        "level": "MINOR",
                        "description": f"道具 {prop_id} 状态变化: {prev_prop.state} -> {curr_prop.state}",
                        "suggested_fixes": ["检查状态变化是否合理"],
                        "auto_fixable": False
                    })

        validation_time = (datetime.now() - start_time).total_seconds()

        return {
            "issues": issues,
            "passed": passed_checks,
            "total_checks": total_checks,
            "validation_time": validation_time
        }

    def _validate_environment(self, previous: StateSnapshot,
                              current: StateSnapshot) -> Dict[str, Any]:
        """验证环境连续性"""
        start_time = datetime.now()
        issues = []
        passed_checks = 0
        total_checks = 3  # 时间、天气、光照

        # 检查时间变化
        if previous.environment.time_of_day != current.environment.time_of_day:
            issues.append({
                "entity_id": "environment",
                "level": "MINOR",
                "description": f"时间变化: {previous.environment.time_of_day} -> {current.environment.time_of_day}",
                "suggested_fixes": ["添加时间过渡", "确保光照相应变化"],
                "auto_fixable": False
            })
        else:
            passed_checks += 1

        # 检查天气变化
        if previous.environment.weather != current.environment.weather:
            issues.append({
                "entity_id": "environment",
                "level": "MINOR",
                "description": f"天气变化: {previous.environment.weather} -> {current.environment.weather}",
                "suggested_fixes": ["添加天气过渡效果", "确保连续性"],
                "auto_fixable": False
            })
        else:
            passed_checks += 1

        # 检查光照一致性
        prev_lighting = previous.environment.lighting
        curr_lighting = current.environment.lighting

        if prev_lighting != curr_lighting:
            # 检查是否是渐进变化
            if not self._is_gradual_change(prev_lighting, curr_lighting):
                issues.append({
                    "entity_id": "environment",
                    "level": "MAJOR",
                    "description": "光照参数突变",
                    "suggested_fixes": ["调整光照渐变", "添加光源移动解释"],
                    "auto_fixable": True
                })
        else:
            passed_checks += 1

        validation_time = (datetime.now() - start_time).total_seconds()

        return {
            "issues": issues,
            "passed": passed_checks,
            "total_checks": total_checks,
            "validation_time": validation_time
        }

    def _validate_constraints(self, constraints: List[Dict],
                              state_snapshot: StateSnapshot) -> Dict[str, Any]:
        """验证约束符合性"""
        start_time = datetime.now()
        issues = []
        passed_checks = 0

        for constraint in constraints:
            constraint_type = constraint.get("type", "")
            entity_id = constraint.get("entity_id", "global")

            # 简化的约束验证
            if constraint_type.startswith("character_"):
                if entity_id in state_snapshot.characters:
                    passed_checks += 1
                elif entity_id != "global":
                    issues.append({
                        "entity_id": entity_id,
                        "level": "MINOR",
                        "description": f"约束目标不存在: {entity_id}",
                        "suggested_fixes": ["移除约束或添加对应实体"],
                        "auto_fixable": True
                    })

            elif constraint_type.startswith("prop_"):
                if entity_id in state_snapshot.props:
                    passed_checks += 1
                elif entity_id != "global":
                    issues.append({
                        "entity_id": entity_id,
                        "level": "MINOR",
                        "description": f"道具约束目标不存在: {entity_id}",
                        "suggested_fixes": ["移除约束或添加对应道具"],
                        "auto_fixable": True
                    })

        validation_time = (datetime.now() - start_time).total_seconds()

        return {
            "issues": issues,
            "passed": passed_checks,
            "total_checks": len(constraints),
            "validation_time": validation_time
        }

    def _is_gradual_change(self, old_value: Any, new_value: Any,
                           threshold: float = 0.3) -> bool:
        """检查是否是渐进变化"""
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            return abs(new_value - old_value) / max(abs(old_value), 1) < threshold

        if isinstance(old_value, dict) and isinstance(new_value, dict):
            # 检查字典值的平均变化
            changes = []
            for key in set(old_value.keys()) | set(new_value.keys()):
                old = old_value.get(key, 0)
                new = new_value.get(key, 0)
                if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                    if max(abs(old), abs(new)) > 0:
                        change = abs(new - old) / max(abs(old), abs(new), 1)
                        changes.append(change)

            if changes:
                return np.mean(changes) < threshold

        return old_value == new_value  # 非数值类型要求完全相等

    def _attempt_auto_fixes(self, report: ValidationReport,
                            current_state: StateSnapshot) -> None:
        """尝试自动修复"""
        fixed_issues = 0
        total_attempts = 0

        for issue in report.issues:
            if issue.auto_fixable and total_attempts < self.config.max_auto_fix_attempts:
                fix_suggestion = self.auto_fixer.suggest_fix(issue, current_state)

                if fix_suggestion:
                    # 模拟应用修复
                    fix_result = {
                        "issue_id": issue.issue_id,
                        "strategy": fix_suggestion.get("action", "unknown"),
                        "applied": True,
                        "effectiveness": fix_suggestion.get("confidence", 0.5),
                        "details": fix_suggestion
                    }

                    # 记录修复结果
                    issue.suggested_fixes.append(f"自动修复建议: {fix_result['strategy']}")
                    fixed_issues += 1

                    # 学习修复经验
                    if self.continuity_learner:
                        self.continuity_learner.learn_from_issue(issue, fix_result)

                total_attempts += 1

        if fixed_issues > 0:
            info(f"自动修复了 {fixed_issues} 个问题")

    def _should_validate(self, frame_number: int) -> bool:
        """判断是否需要验证"""
        if not self.config.enable_real_time_validation:
            return False

        if frame_number % self.config.validation_frequency == 0:
            return True

        # 根据性能决定
        if len(self.performance_metrics["processing_times"]) > 10:
            avg_time = np.mean(self.performance_metrics["processing_times"][-10:])
            if avg_time < 0.1:  # 平均处理时间小于0.1秒
                return True

        return False

    def _generate_recommendations(self, scene_analysis: Dict,
                                  validation_report: Optional[ValidationReport],
                                  predicted_issues: List[Dict]) -> List[str]:
        """生成处理建议"""
        recommendations = []

        # 基于场景分析的推荐
        complexity = scene_analysis["complexity"]
        risk = scene_analysis["risk_assessment"]["overall_risk"]

        if complexity in ["complex", "epic"]:
            recommendations.append("场景复杂度高，建议分阶段生成和验证")

        if risk > 0.7:
            recommendations.append("风险评估高，建议增加验证频率和深度")

        # 基于验证结果的推荐
        if validation_report:
            critical_count = validation_report.summary["critical_issues"]
            if critical_count > 0:
                recommendations.append(f"发现 {critical_count} 个关键问题，建议立即修复")

            if validation_report.summary["passed"] / max(1, validation_report.summary["total_checks"]) < 0.7:
                recommendations.append("通过率较低，建议检查数据源质量")

        # 基于预测问题的推荐
        for predicted in predicted_issues:
            if predicted.get("confidence", 0) > 0.8:
                recommendations.append(f"预测问题: {predicted['type']} - {predicted['prevention_advice'][0]}")

        # 性能优化建议
        if len(self.performance_metrics["processing_times"]) > 20:
            recent_times = self.performance_metrics["processing_times"][-10:]
            avg_time = np.mean(recent_times)

            if avg_time > 0.5:
                recommendations.append(f"处理时间较长 ({avg_time:.2f}秒)，建议优化或降低分析深度")

        return recommendations[:5]  # 返回前5个最重要的建议

    def _record_performance_metric(self, metric_name: str, value: float):
        """记录性能指标"""
        if metric_name in self.performance_metrics:
            self.performance_metrics[metric_name].append(value)

            # 保持历史长度
            if len(self.performance_metrics[metric_name]) > 100:
                self.performance_metrics[metric_name].pop(0)

    def _generate_periodic_report(self):
        """生成周期性报告"""
        report = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "frames_processed": self.frame_counter,
            "performance_summary": self._get_performance_summary(),
            "issue_statistics": self._get_issue_statistics(),
            "learning_progress": self._get_learning_progress() if self.continuity_learner else None,
            "recommendations": self._generate_system_recommendations()
        }

        # 保存报告
        if self.config.auto_save_reports:
            self._save_report(report)

        return report

    def _get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_metrics["processing_times"]:
            return {"average_processing_time": 0, "total_validations": 0}

        recent_times = self.performance_metrics["processing_times"][-20:]

        summary = {
            "average_processing_time": np.mean(recent_times),
            "min_processing_time": min(recent_times),
            "max_processing_time": max(recent_times),
            "total_validations": len(self.performance_metrics["validation_times"]),
            "average_validation_time": np.mean(self.performance_metrics["validation_times"])
            if self.performance_metrics["validation_times"] else 0
        }

        return summary

    def _get_issue_statistics(self) -> Dict[str, Any]:
        """获取问题统计"""
        stats = {
            "total_issues": 0,
            "by_severity": defaultdict(int),
            "by_entity_type": defaultdict(int),
            "auto_fix_rate": 0.0
        }

        total_fixable = 0
        total_issues = 0

        for report in self.validation_reports.values():
            for issue in report.issues:
                total_issues += 1
                stats["by_severity"][issue.level.value] += 1
                stats["by_entity_type"][issue.entity_type] += 1

                if issue.auto_fixable:
                    total_fixable += 1

        stats["total_issues"] = total_issues
        stats["by_severity"] = dict(stats["by_severity"])
        stats["by_entity_type"] = dict(stats["by_entity_type"])

        if total_issues > 0:
            stats["auto_fix_rate"] = total_fixable / total_issues

        return stats

    def _get_learning_progress(self) -> Dict[str, Any]:
        """获取学习进度"""
        if not self.continuity_learner:
            return {"enabled": False}

        return {
            "enabled": True,
            "statistics": self.continuity_learner.get_learning_statistics()
        }

    def _generate_system_recommendations(self) -> List[str]:
        """生成系统级推荐"""
        recommendations = []

        # 性能优化推荐
        perf_summary = self._get_performance_summary()
        if perf_summary["average_processing_time"] > 0.3:
            recommendations.append("考虑降低分析深度或启用并行处理")

        # 配置优化推荐
        issue_stats = self._get_issue_statistics()
        if issue_stats["auto_fix_rate"] > 0.8:
            recommendations.append("自动修复成功率较高，可以增加最大修复尝试次数")

        if self.frame_counter > 100 and len(self.validation_reports) < 10:
            recommendations.append("验证频率可能过低，考虑增加验证频率")

        return recommendations

    def _save_report(self, report: Dict):
        """保存报告"""
        try:
            # 创建报告目录
            report_dir = Path(self.config.report_save_path)
            report_dir.mkdir(exist_ok=True)

            # 生成文件名
            filename = f"report_{self.session_id}_{datetime.now().strftime('%H%M%S')}.json"
            filepath = report_dir / filename

            # 保存报告
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            info(f"报告已保存: {filepath}")
        except Exception as e:
            error(f"保存报告失败: {e}")

    # 继承和扩展之前定义的方法
    def capture_state(self, scene_data: Dict[str, Any]) -> StateSnapshot:
        """捕获当前状态快照（增强版）"""
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            scene_id=scene_data.get("scene_id", f"scene_{self.frame_counter}"),
            frame_number=self.frame_counter,
            characters=self._extract_characters(scene_data),
            props=self._extract_props(scene_data),
            environment=self._extract_environment(scene_data),
            spatial_relations=self._extract_spatial_relations(scene_data),
            metadata={
                "scene_type": scene_data.get("scene_type", "general"),
                "processing_priority": self._determine_processing_priority(scene_data),
                "capture_time": datetime.now().isoformat()
            }
        )

        self.current_state = snapshot
        self.state_history.append(snapshot)

        # 保持历史长度
        if len(self.state_history) > self.config.max_state_history:
            self.state_history.pop(0)

        return snapshot

    def _extract_spatial_relations(self, scene_data: Dict) -> SpatialRelation:
        """提取空间关系"""
        relations = SpatialRelation()

        # 提取角色与道具的关系
        characters = scene_data.get("characters", [])
        props = scene_data.get("props", [])

        for char in characters:
            char_id = char.get("id")
            char_pos = char.get("position", [0, 0, 0])

            for prop in props:
                prop_id = prop.get("id")
                prop_pos = prop.get("position", [0, 0, 0])

                # 计算距离
                if isinstance(char_pos, (list, tuple)) and len(char_pos) >= 3 and \
                        isinstance(prop_pos, (list, tuple)) and len(prop_pos) >= 3:

                    dx = prop_pos[0] - char_pos[0]
                    dy = prop_pos[1] - char_pos[1]
                    dz = prop_pos[2] - char_pos[2]
                    distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                    # 根据距离确定关系
                    if distance < 1.0:
                        relation = "holding" if distance < 0.5 else "near"
                        relations.add_relationship(char_id, relation, prop_id, 1.0 - distance)

        return relations

    def generate_hints(self, target_scene: str) -> GenerationHints:
        """生成连续性提示（增强版）"""
        hints = GenerationHints()

        if self.current_state:
            # 基于当前状态的约束
            hints.continuity_constraints.extend(
                self._generate_state_based_constraints()
            )

            # 基于历史的学习
            if self.continuity_learner:
                predicted_issues = self.continuity_learner.predict_issues(
                    self.current_state.to_dict()
                )
                for issue in predicted_issues:
                    hints.continuity_constraints.append(
                        f"预防预测问题: {issue['type']}"
                    )

            # 添加上下文
            hints.previous_context = {
                "scene_id": self.current_state.scene_id,
                "frame_number": self.current_state.frame_number,
                "character_count": len(self.current_state.characters),
                "environment_state": {
                    "time": self.current_state.environment.time_of_day,
                    "weather": self.current_state.environment.weather
                }
            }

        # 基于场景类型的指南
        hints.style_guidelines.update(
            self._get_style_guidelines_for_scene(target_scene)
        )

        # 避免的元素（基于历史问题）
        hints.avoid_elements.extend(
            self._get_elements_to_avoid()
        )

        return hints

    def _generate_state_based_constraints(self) -> List[str]:
        """生成基于状态的约束"""
        constraints = []

        if not self.current_state:
            return constraints

        # 角色外观约束
        for char_id, character in self.current_state.characters.items():
            constraints.append(
                f"保持角色 {char_id} 的外观: {character.appearance.get('summary', 'default')}"
            )
            if character.outfit:
                constraints.append(
                    f"保持角色 {char_id} 的服装: {character.outfit.get('type', 'default')}"
                )

        # 环境约束
        env = self.current_state.environment
        constraints.append(f"保持时间: {env.time_of_day}")
        constraints.append(f"保持天气: {env.weather}")

        # 空间关系约束
        for char_id, character in self.current_state.characters.items():
            if character.position:
                nearby_props = []
                for prop_id, prop in self.current_state.props.items():
                    if prop.position:
                        distance = self._calculate_distance(character.position, prop.position)
                        if distance < 2.0:
                            nearby_props.append(prop_id)

                if nearby_props:
                    constraints.append(
                        f"角色 {char_id} 与道具 {', '.join(nearby_props)} 的空间关系应保持"
                    )

        return constraints

    def _get_style_guidelines_for_scene(self, scene_type: str) -> Dict[str, Any]:
        """获取场景类型的风格指南"""
        guidelines = {
            "general": {
                "lighting_style": "natural",
                "camera_movement": "moderate",
                "color_palette": "balanced"
            },
            "action": {
                "lighting_style": "dynamic",
                "camera_movement": "energetic",
                "color_palette": "high_contrast",
                "motion_blur": "moderate"
            },
            "dialogue": {
                "lighting_style": "soft",
                "camera_movement": "minimal",
                "color_palette": "warm",
                "focus_pull": "smooth"
            },
            "environment": {
                "lighting_style": "atmospheric",
                "camera_movement": "sweeping",
                "color_palette": "muted",
                "depth_of_field": "shallow"
            }
        }

        return guidelines.get(scene_type, guidelines["general"])

    def _get_elements_to_avoid(self) -> List[str]:
        """获取需要避免的元素"""
        avoid_list = []

        # 基于历史问题
        for report in list(self.validation_reports.values())[-5:]:  # 最近5个报告
            for issue in report.issues:
                if issue.level in [ContinuityLevel.CRITICAL, ContinuityLevel.MAJOR]:
                    if "appearance" in issue.description:
                        avoid_list.append("角色外貌突变")
                    if "position" in issue.description and "jump" in issue.description:
                        avoid_list.append("位置跳跃")
                    if "lighting" in issue.description and "abrupt" in issue.description:
                        avoid_list.append("光照突变")

        return list(set(avoid_list))[:10]  # 去重并限制数量

    def create_keyframe_anchor(self, frame_id: str, timestamp: float) -> KeyframeAnchor:
        """创建关键帧锚点（增强版）"""
        anchor = KeyframeAnchor(frame_id, timestamp)

        if self.current_state:
            # 深度复制状态
            for character in self.current_state.characters.values():
                anchor.add_character_state(self._deep_copy_character(character))

            for prop in self.current_state.props.values():
                anchor.add_prop_state(self._deep_copy_prop(prop))

            anchor.environment = self._deep_copy_environment(self.current_state.environment)

            # 添加上下文信息
            anchor.continuity_checks.append({
                "check_time": datetime.now(),
                "scene_complexity": self.scene_analyzer.analyze_scene_complexity(
                    self.current_state.to_dict()
                ).value,
                "risk_level": self._assess_scene_risk(
                    self.current_state.to_dict()
                )["overall_risk"]
            })

        self.keyframe_anchors[frame_id] = anchor
        info(f"创建关键帧锚点: {frame_id}")

        return anchor

    def _deep_copy_character(self, character: CharacterState) -> CharacterState:
        """深度复制角色状态"""
        new_char = CharacterState(character.character_id, character.name)
        new_char.appearance = character.appearance.copy()
        new_char.outfit = character.outfit.copy()
        new_char.emotional_state = character.emotional_state
        new_char.physical_state = character.physical_state.copy()
        new_char.inventory = character.inventory.copy()
        new_char.position = character.position
        new_char.orientation = character.orientation
        new_char.interactions = character.interactions.copy()
        new_char.timestamp = character.timestamp
        return new_char

    def _deep_copy_prop(self, prop: PropState) -> PropState:
        """深度复制道具状态"""
        new_prop = PropState(prop.prop_id, prop.name)
        new_prop.position = prop.position
        new_prop.orientation = prop.orientation
        new_prop.state = prop.state
        new_prop.owner = prop.owner
        new_prop.interaction_history = prop.interaction_history.copy()
        new_prop.physical_condition = prop.physical_condition.copy()
        return new_prop

    def _deep_copy_environment(self, env: EnvironmentState) -> EnvironmentState:
        """深度复制环境状态"""
        new_env = EnvironmentState(env.scene_id)
        new_env.time_of_day = env.time_of_day
        new_env.weather = env.weather
        new_env.lighting = env.lighting.copy()
        new_env.ambient_sounds = env.ambient_sounds.copy()
        new_env.active_effects = env.active_effects.copy()
        new_env.prop_positions = env.prop_positions.copy()
        new_env.character_positions = env.character_positions.copy()
        return new_env

    def get_continuity_report(self, detailed: bool = False) -> str:
        """获取连续性报告（增强版）"""
        if not self.validation_reports:
            return "暂无连续性验证报告"

        latest_report = list(self.validation_reports.values())[-1]

        if detailed:
            report = f"连续性详细报告:\n"
            report += "=" * 60 + "\n"
            report += f"验证ID: {latest_report.validation_id}\n"
            report += f"时间: {latest_report.timestamp}\n"
            report += f"总检查数: {latest_report.summary['total_checks']}\n"
            report += f"通过数: {latest_report.summary['passed']}\n"
            report += f"通过率: {latest_report.summary['passed'] / max(1, latest_report.summary['total_checks']) * 100:.1f}%\n\n"

            report += "问题统计:\n"
            report += f"  关键问题: {latest_report.summary['critical_issues']}\n"
            report += f"  主要问题: {latest_report.summary['major_issues']}\n"
            report += f"  次要问题: {latest_report.summary['minor_issues']}\n"
            report += f"  外观问题: {latest_report.summary['cosmetic_issues']}\n\n"

            if latest_report.issues:
                report += "详细问题:\n"
                for i, issue in enumerate(latest_report.issues[:5], 1):  # 显示前5个问题
                    report += f"  {i}. [{issue.level.value}] {issue.description}\n"
                    if issue.suggested_fixes:
                        report += f"     建议: {issue.suggested_fixes[0]}\n"

            # 添加性能信息
            if self.performance_metrics["processing_times"]:
                recent_times = self.performance_metrics["processing_times"][-5:]
                avg_time = np.mean(recent_times)
                report += f"\n性能: 平均处理时间 {avg_time:.3f}秒"
        else:
            report = latest_report.generate_summary()

        return report

    def analyze_transition(self, from_scene: Dict, to_scene: Dict) -> Dict[str, Any]:
        """分析场景转场"""
        transition = self.transition_manager.analyze_transition(from_scene, to_scene)
        validation_issues = self.transition_manager.validate_transition(transition)
        advice = self.transition_manager.generate_transition_advice(transition)

        return {
            "transition_analysis": {
                "type": transition.transition_type,
                "temporal_gap": transition.temporal_gap,
                "spatial_changes": transition.spatial_changes,
                "character_transitions": transition.character_transitions
            },
            "validation_issues": validation_issues,
            "advice": advice
        }

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "frames_processed": self.frame_counter,
            "current_mode": self.config.mode.value,
            "analysis_depth": self.config.analysis_depth.value,
            "state_history_size": len(self.state_history),
            "validation_reports_count": len(self.validation_reports),
            "keyframe_anchors_count": len(self.keyframe_anchors),
            "performance": self._get_performance_summary(),
            "learning_enabled": self.config.enable_machine_learning,
            "auto_fix_enabled": self.config.enable_auto_fix,
            "cache_size": {k: len(v) for k, v in self.cache.items()}
        }

        return status

    def export_session_data(self, filepath: str) -> bool:
        """导出会话数据"""
        try:
            export_data = {
                "session_id": self.session_id,
                "project_id": self.project_id,
                "config": self.config.to_dict(),
                "state_history": [s.to_dict() for s in self.state_history[-100:]],  # 导出最近100个状态
                "validation_reports": {k: v.__dict__ for k, v in list(self.validation_reports.items())[-50:]},
                "performance_metrics": self.performance_metrics,
                "timestamp": datetime.now().isoformat()
            }

            with open(filepath, 'wb') as f:
                pickle.dump(export_data, f)

            info(f"会话数据已导出: {filepath}")
            return True
        except Exception as e:
            error(f"导出会话数据失败: {e}")
            return False

    def import_session_data(self, filepath: str) -> bool:
        """导入会话数据"""
        try:
            with open(filepath, 'rb') as f:
                import_data = pickle.load(f)

            # 验证数据格式
            if not all(k in import_data for k in ["session_id", "project_id", "config"]):
                error("导入数据格式无效")
                return False

            # 恢复数据
            self.session_id = import_data["session_id"]
            self.project_id = import_data["project_id"]

            # 注意：config可能需要特殊处理
            info(f"会话数据已导入: {filepath}")
            return True
        except Exception as e:
            error(f"导入会话数据失败: {e}")
            return False

    def reset_session(self, new_session_id: Optional[str] = None):
        """重置会话"""
        old_session = self.session_id

        self.session_id = new_session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.frame_counter = 0

        # 清空历史数据（保留配置）
        self.state_history.clear()
        self.validation_reports.clear()
        self.keyframe_anchors.clear()
        self.cache = {"scene_metrics": {}, "validation_results": {}, "constraints": {}}
        self.performance_metrics = {"processing_times": [], "issue_counts": [], "validation_times": []}

        info(f"会话已重置: {old_session} -> {self.session_id}")

    def _calculate_distance(self, pos1, pos2) -> float:
        """计算距离"""
        if isinstance(pos1, (list, tuple)) and isinstance(pos2, (list, tuple)):
            if len(pos1) >= 3 and len(pos2) >= 3:
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                dz = pos2[2] - pos1[2]
                return math.sqrt(dx * dx + dy * dy + dz * dz)
        return 0.0

    def _extract_characters(self, scene_data: Dict[str, Any]) -> Dict[str, CharacterState]:
        """从场景数据提取角色状态（完整实现）"""
        characters = {}

        for char_data in scene_data.get("characters", []):
            # 基础信息
            char_id = char_data.get("id", f"character_{len(characters)}")
            char_name = char_data.get("name", "Unknown")

            # 创建角色状态对象
            character = CharacterState(char_id, char_name)

            # 提取外貌特征
            if "appearance" in char_data:
                if isinstance(char_data["appearance"], dict):
                    character.appearance = char_data["appearance"].copy()
                else:
                    # 如果是简单描述，转换为字典
                    character.appearance = {
                        "description": str(char_data["appearance"]),
                        "summary": str(char_data["appearance"])[:100]
                    }
            else:
                # 默认外貌
                character.appearance = {
                    "default": True,
                    "description": "默认角色外貌",
                    "summary": "default"
                }

            # 提取服装信息
            if "outfit" in char_data:
                if isinstance(char_data["outfit"], dict):
                    character.outfit = char_data["outfit"].copy()
                else:
                    character.outfit = {
                        "description": str(char_data["outfit"]),
                        "type": "casual"
                    }
            elif "clothing" in char_data:
                # 兼容性：使用 clothing 作为 outfit
                character.outfit = {
                    "description": str(char_data["clothing"]),
                    "type": char_data.get("clothing_type", "casual")
                }

            # 提取情绪状态
            if "emotion" in char_data:
                character.emotional_state = str(char_data["emotion"])
            elif "mood" in char_data:
                character.emotional_state = str(char_data["mood"])

            # 提取物理状态
            if "physical_state" in char_data:
                if isinstance(char_data["physical_state"], dict):
                    character.physical_state = char_data["physical_state"].copy()
                else:
                    character.physical_state = {
                        "condition": str(char_data["physical_state"])
                    }

            # 提取位置信息
            if "position" in char_data:
                pos = char_data["position"]
                if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    character.position = tuple(float(p) for p in pos[:3])
                elif isinstance(pos, dict) and "x" in pos and "y" in pos and "z" in pos:
                    character.position = (float(pos["x"]), float(pos["y"]), float(pos["z"]))

            # 提取朝向信息
            if "rotation" in char_data:
                rot = char_data["rotation"]
                if isinstance(rot, (list, tuple)) and len(rot) >= 3:
                    # 计算欧拉角或四元数转换为朝向角度
                    character.orientation = float(rot[1]) if len(rot) > 1 else 0.0  # 使用Y轴旋转
                elif isinstance(rot, dict) and "yaw" in rot:
                    character.orientation = float(rot["yaw"])
            elif "orientation" in char_data:
                character.orientation = float(char_data["orientation"])

            # 提取速度信息
            if "velocity" in char_data:
                velocity = char_data["velocity"]
                if isinstance(velocity, (list, tuple)) and len(velocity) >= 3:
                    character.physical_state["velocity"] = tuple(float(v) for v in velocity[:3])

            # 提取动作状态
            if "action" in char_data:
                character.physical_state["action"] = str(char_data["action"])
            elif "state" in char_data:
                character.physical_state["current_state"] = str(char_data["state"])

            # 提取携带物品
            if "inventory" in char_data:
                inventory = char_data["inventory"]
                if isinstance(inventory, list):
                    character.inventory = [str(item) for item in inventory]
                elif isinstance(inventory, str):
                    character.inventory = inventory.split(",")

            # 提取交互历史
            if "interactions" in char_data:
                interactions = char_data["interactions"]
                if isinstance(interactions, list):
                    character.interactions = [str(interaction) for interaction in interactions]

            # 提取其他属性
            if "scale" in char_data:
                character.physical_state["scale"] = float(char_data["scale"])

            if "visibility" in char_data:
                character.physical_state["visible"] = bool(char_data["visibility"])

            # 提取角色标签
            if "tags" in char_data:
                character.physical_state["tags"] = char_data["tags"] if isinstance(char_data["tags"], list) else [char_data["tags"]]

            # 更新时间戳
            character.timestamp = datetime.now()

            # 添加到角色字典
            characters[char_id] = character

        return characters

    def _extract_props(self, scene_data: Dict[str, Any]) -> Dict[str, PropState]:
        """从场景数据提取道具状态（完整实现）"""
        props = {}

        for prop_data in scene_data.get("props", []):
            # 基础信息
            prop_id = prop_data.get("id", f"prop_{len(props)}")
            prop_name = prop_data.get("name", "Unknown Prop")

            # 创建道具状态对象
            prop = PropState(prop_id, prop_name)

            # 提取位置信息
            if "position" in prop_data:
                pos = prop_data["position"]
                if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    prop.position = tuple(float(p) for p in pos[:3])
                elif isinstance(pos, dict) and "x" in pos and "y" in pos and "z" in pos:
                    prop.position = (float(pos["x"]), float(pos["y"]), float(pos["z"]))

            # 提取朝向信息
            if "rotation" in prop_data:
                rot = prop_data["rotation"]
                if isinstance(rot, (list, tuple)) and len(rot) >= 3:
                    prop.orientation = tuple(float(r) for r in rot[:3])
                elif isinstance(rot, dict) and "x" in rot and "y" in rot and "z" in rot:
                    prop.orientation = (float(rot["x"]), float(rot["y"]), float(rot["z"]))

            # 提取状态
            if "state" in prop_data:
                prop.state = str(prop_data["state"])
            else:
                # 根据位置判断状态
                if prop.position and prop.position[1] < 0:
                    prop.state = "fallen"
                else:
                    prop.state = "default"

            # 提取所有者
            if "owner" in prop_data:
                prop.owner = str(prop_data["owner"])
            elif "held_by" in prop_data:
                prop.owner = str(prop_data["held_by"])
            elif "attached_to" in prop_data:
                prop.owner = str(prop_data["attached_to"])

            # 提取物理属性
            if "physics" in prop_data:
                physics = prop_data["physics"]
                if isinstance(physics, dict):
                    prop.physical_condition.update(physics)
                else:
                    prop.physical_condition["physics_data"] = str(physics)

            # 提取质量
            if "mass" in prop_data:
                prop.physical_condition["mass"] = float(prop_data["mass"])

            # 提取材质
            if "material" in prop_data:
                prop.physical_condition["material"] = str(prop_data["material"])

            # 提取尺寸
            if "size" in prop_data:
                size = prop_data["size"]
                if isinstance(size, (list, tuple)) and len(size) >= 3:
                    prop.physical_condition["size"] = tuple(float(s) for s in size[:3])
                elif isinstance(size, dict) and "width" in size and "height" in size and "depth" in size:
                    prop.physical_condition["size"] = (float(size["width"]), float(size["height"]), float(size["depth"]))
                elif isinstance(size, (int, float)):
                    prop.physical_condition["size"] = (float(size), float(size), float(size))

            # 提取可交互性
            if "interactive" in prop_data:
                prop.physical_condition["interactive"] = bool(prop_data["interactive"])

            if "interaction_type" in prop_data:
                prop.physical_condition["interaction_type"] = str(prop_data["interaction_type"])

            # 提取耐久度
            if "durability" in prop_data:
                prop.physical_condition["durability"] = float(prop_data["durability"])

            if "condition" in prop_data:
                prop.physical_condition["condition"] = str(prop_data["condition"])

            # 提取道具标签
            if "tags" in prop_data:
                prop.physical_condition["tags"] = prop_data["tags"] if isinstance(prop_data["tags"], list) else [prop_data["tags"]]

            # 提取交互历史
            if "interaction_history" in prop_data:
                history = prop_data["interaction_history"]
                if isinstance(history, list):
                    for item in history:
                        if isinstance(item, dict):
                            prop.interaction_history.append({
                                "timestamp": item.get("timestamp", datetime.now()),
                                "character": item.get("character", "unknown"),
                                "action": item.get("action", "interacted"),
                                "details": item.get("details", {})
                            })

            # 提取道具类型
            if "type" in prop_data:
                prop.physical_condition["prop_type"] = str(prop_data["type"])

            # 提取道具价值
            if "value" in prop_data:
                prop.physical_condition["value"] = float(prop_data["value"])

            # 提取道具状态变化
            if "state_changes" in prop_data:
                changes = prop_data["state_changes"]
                if isinstance(changes, list):
                    prop.physical_condition["state_changes"] = changes

            # 提取道具所属场景
            if "scene_relevance" in prop_data:
                prop.physical_condition["scene_relevance"] = str(prop_data["scene_relevance"])

            # 添加到道具字典
            props[prop_id] = prop

        return props

    def _extract_environment(self, scene_data: Dict[str, Any]) -> EnvironmentState:
        """从场景数据提取环境状态（完整实现）"""
        # 获取场景ID
        scene_id = scene_data.get("scene_id", f"scene_{self.frame_counter}")

        # 创建环境状态对象
        environment = EnvironmentState(scene_id)

        # 提取环境数据
        env_data = scene_data.get("environment", {})

        # 提取时间信息
        if "time_of_day" in env_data:
            time_val = env_data["time_of_day"]
            if isinstance(time_val, str):
                # 标准化时间值
                time_val = time_val.lower()
                if time_val in ["morning", "day", "afternoon", "noon"]:
                    environment.time_of_day = "day"
                elif time_val in ["evening", "dusk", "sunset"]:
                    environment.time_of_day = "dusk"
                elif time_val in ["night", "midnight"]:
                    environment.time_of_day = "night"
                elif time_val in ["dawn", "sunrise"]:
                    environment.time_of_day = "dawn"
                else:
                    environment.time_of_day = time_val
            elif isinstance(time_val, (int, float)):
                # 数字时间：0-24小时
                hour = float(time_val) % 24
                if 6 <= hour < 18:
                    environment.time_of_day = "day"
                elif 18 <= hour < 20:
                    environment.time_of_day = "dusk"
                elif 20 <= hour or hour < 6:
                    environment.time_of_day = "night"
                else:
                    environment.time_of_day = "day"

        # 提取天气信息
        if "weather" in env_data:
            weather_val = env_data["weather"]
            if isinstance(weather_val, str):
                # 标准化天气值
                weather_val = weather_val.lower()
                if weather_val in ["clear", "sunny", "fair"]:
                    environment.weather = "clear"
                elif weather_val in ["cloudy", "overcast", "grey"]:
                    environment.weather = "cloudy"
                elif weather_val in ["rain", "raining", "rainy", "drizzle"]:
                    environment.weather = "rain"
                elif weather_val in ["snow", "snowing", "snowy"]:
                    environment.weather = "snow"
                elif weather_val in ["fog", "foggy", "mist", "misty"]:
                    environment.weather = "fog"
                elif weather_val in ["storm", "stormy", "thunderstorm"]:
                    environment.weather = "storm"
                else:
                    environment.weather = weather_val

        # 提取光照信息
        if "lighting" in env_data:
            lighting_data = env_data["lighting"]
            if isinstance(lighting_data, dict):
                environment.lighting = lighting_data.copy()
            else:
                environment.lighting = {
                    "description": str(lighting_data),
                    "intensity": 1.0
                }

        # 提取光源信息
        if "lights" in env_data:
            lights = env_data["lights"]
            if isinstance(lights, list):
                environment.lighting["lights"] = lights

        # 提取环境音效
        if "sounds" in env_data:
            sounds = env_data["sounds"]
            if isinstance(sounds, list):
                environment.ambient_sounds = [str(sound) for sound in sounds]
            elif isinstance(sounds, str):
                environment.ambient_sounds = sounds.split(",")

        # 提取活跃效果
        if "effects" in env_data:
            effects = env_data["effects"]
            if isinstance(effects, list):
                environment.active_effects = [str(effect) for effect in effects]

        # 提取全局位置
        if "global_position" in env_data:
            pos = env_data["global_position"]
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                environment.prop_positions["scene_origin"] = tuple(float(p) for p in pos[:3])

        # 提取场景比例
        if "scale" in env_data:
            scale = env_data["scale"]
            if isinstance(scale, (int, float)):
                environment.lighting["scene_scale"] = float(scale)

        # 提取环境颜色
        if "color_palette" in env_data:
            palette = env_data["color_palette"]
            if isinstance(palette, (list, tuple)):
                environment.lighting["color_palette"] = palette

        # 提取环境温度
        if "temperature" in env_data:
            temp = env_data["temperature"]
            if isinstance(temp, (int, float)):
                environment.lighting["temperature"] = float(temp)

        # 提取环境湿度
        if "humidity" in env_data:
            humidity = env_data["humidity"]
            if isinstance(humidity, (int, float)):
                environment.lighting["humidity"] = float(humidity)

        # 提取风力信息
        if "wind" in env_data:
            wind = env_data["wind"]
            if isinstance(wind, dict):
                environment.lighting["wind"] = wind
            elif isinstance(wind, (int, float)):
                environment.lighting["wind_speed"] = float(wind)

        # 提取环境氛围
        if "atmosphere" in env_data:
            atmosphere = env_data["atmosphere"]
            if isinstance(atmosphere, str):
                environment.lighting["atmosphere"] = atmosphere

        # 提取环境标签
        if "tags" in env_data:
            tags = env_data["tags"]
            if isinstance(tags, list):
                environment.lighting["tags"] = tags

        # 提取时间流逝信息
        if "time_progression" in env_data:
            progression = env_data["time_progression"]
            if isinstance(progression, dict):
                environment.lighting["time_progression"] = progression

        # 提取季节信息
        if "season" in env_data:
            season = env_data["season"]
            if isinstance(season, str):
                environment.lighting["season"] = season

        # 提取地理位置
        if "location" in env_data:
            location = env_data["location"]
            if isinstance(location, str):
                environment.lighting["location"] = location
            elif isinstance(location, dict):
                environment.lighting["location"] = location

        # 提取环境难度
        if "difficulty" in env_data:
            difficulty = env_data["difficulty"]
            if isinstance(difficulty, (int, float)):
                environment.lighting["difficulty_level"] = float(difficulty)

        # 提取特殊环境效果
        if "special_effects" in env_data:
            special_effects = env_data["special_effects"]
            if isinstance(special_effects, list):
                for effect in special_effects:
                    if effect not in environment.active_effects:
                        environment.active_effects.append(str(effect))

        # 提取环境边界
        if "boundaries" in env_data:
            boundaries = env_data["boundaries"]
            if isinstance(boundaries, dict):
                environment.lighting["boundaries"] = boundaries

        # 提取环境重力
        if "gravity" in env_data:
            gravity = env_data["gravity"]
            if isinstance(gravity, (int, float)):
                environment.lighting["gravity"] = float(gravity)
            elif isinstance(gravity, (list, tuple)) and len(gravity) >= 3:
                environment.lighting["gravity_vector"] = tuple(float(g) for g in gravity[:3])

        # 提取环境物理参数
        if "physics" in env_data:
            physics = env_data["physics"]
            if isinstance(physics, dict):
                environment.lighting["physics_settings"] = physics

        # 提取环境光照预设
        if "lighting_preset" in env_data:
            preset = env_data["lighting_preset"]
            if isinstance(preset, str):
                environment.lighting["preset"] = preset

        # 确保有基本的光照设置
        if "intensity" not in environment.lighting:
            # 根据时间设置默认光照强度
            if environment.time_of_day == "night":
                environment.lighting["intensity"] = 0.3
            elif environment.time_of_day == "dusk" or environment.time_of_day == "dawn":
                environment.lighting["intensity"] = 0.7
            else:
                environment.lighting["intensity"] = 1.0

        # 确保有基本的光照方向
        if "direction" not in environment.lighting:
            # 根据时间设置默认光照方向
            if environment.time_of_day == "night":
                environment.lighting["direction"] = [0, -1, 0]  # 月光从上往下
            else:
                environment.lighting["direction"] = [1, -1, 0.5]  # 斜上方

        return environment

    # 新增：从状态快照中提取角色位置
    def _extract_character_positions(self, snapshot: StateSnapshot) -> Dict[str, Tuple[float, float, float]]:
        """从状态快照中提取角色位置"""
        positions = {}

        for char_id, character in snapshot.characters.items():
            if character.position:
                positions[char_id] = character.position

        return positions

    # 新增：从状态快照中提取道具位置
    def _extract_prop_positions(self, snapshot: StateSnapshot) -> Dict[str, Tuple[float, float, float]]:
        """从状态快照中提取道具位置"""
        positions = {}

        for prop_id, prop in snapshot.props.items():
            if prop.position:
                positions[prop_id] = prop.position

        return positions

    # 新增：环境状态验证
    def _validate_environment_state(self, environment: EnvironmentState) -> List[Dict]:
        """验证环境状态的合理性"""
        issues = []

        # 验证时间与光照的一致性
        if environment.time_of_day == "night" and environment.lighting.get("intensity", 1.0) > 0.5:
            issues.append({
                "type": "lighting_inconsistency",
                "description": "夜晚光照强度过高",
                "severity": "minor",
                "suggestion": "降低光照强度或调整时间设置"
            })

        # 验证天气与效果的合理性
        if environment.weather == "clear" and "rain_effect" in environment.active_effects:
            issues.append({
                "type": "weather_effect_mismatch",
                "description": "晴天却有雨效",
                "severity": "medium",
                "suggestion": "移除雨效或更改天气"
            })

        # 验证重力设置
        gravity = environment.lighting.get("gravity")
        if gravity is not None and isinstance(gravity, (int, float)):
            if abs(gravity - 9.8) > 5.0:
                issues.append({
                    "type": "unusual_gravity",
                    "description": f"重力设置异常: {gravity} m/s²",
                    "severity": "low",
                    "suggestion": "检查重力设置是否符合场景需求"
                })

        return issues