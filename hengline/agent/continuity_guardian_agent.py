# -*- coding: utf-8 -*-
"""
@FileName: continuity_guardian_agent.py
@Description: 连续性守护智能体，负责跟踪角色状态，生成/验证连续性锚点
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from .continuity_guardian.continuity_guardian_manager import IntegratedContinuityGuardian
from .continuity_guardian.model.continuity_guard_guardian import GuardianConfig, AnalysisDepth, GuardMode
from hengline.logger import info, error, debug
from .continuity_guardian.model.continuity_guardian_report import AnchoredTimeline
from .temporal_planner.temporal_planner_model import TimelinePlan


class ContinuityGuardianAgent:
    """连续性守护智能体 - 总接口"""

    def __init__(self, task_id: str, config: Optional[Dict] = None):
        """
        初始化连续性守护智能体

        Args:
            task_id: 任务ID
            config: 配置字典，可选
        """
        self.task_id = task_id

        # 解析配置
        self.config = self._parse_config(config) or {
            "mode": "adaptive",
            "analysis_depth": "standard",
            "enable_auto_fix": True,
            "validation_frequency": 5
        }

        # 初始化核心引擎
        self.guardian = None
        self._initialized = False

        info(f"连续性守护智能体初始化 - task_id: {task_id}")

    def _parse_config(self, config_dict: Optional[Dict]) -> GuardianConfig:
        """解析配置字典为GuardianConfig对象"""
        if config_dict is None:
            config_dict = {}

        # 基础配置
        base_config = {
            "task_id": self.task_id,
            "mode": GuardMode(config_dict.get("mode", "adaptive")),
            "analysis_depth": AnalysisDepth(config_dict.get("analysis_depth", "standard")),
            "enable_auto_fix": config_dict.get("enable_auto_fix", True),
            "max_state_history": config_dict.get("max_state_history", 1000),
            "validation_frequency": config_dict.get("validation_frequency", 10),
            "enable_real_time_validation": config_dict.get("enable_real_time_validation", True),
            "enable_machine_learning": config_dict.get("enable_machine_learning", False),
            "generate_reports": config_dict.get("generate_reports", True),
            "report_save_path": config_dict.get("report_save_path", f"./reports/{self.task_id}"),
            "parallel_processing": config_dict.get("parallel_processing", True),
            "max_workers": config_dict.get("max_workers", 4)
        }

        return GuardianConfig(**base_config)

    def initialize(self):
        """初始化智能体（延迟初始化）"""
        if not self._initialized:
            self.guardian = IntegratedContinuityGuardian(self.task_id, self.config)
            self._initialized = True
            info("连续性守护智能体初始化完成")
        return self

    def process(self, plan: TimelinePlan) -> AnchoredTimeline:
        """
        处理单个视频帧/场景

        Args:
            plan: 帧数据，包含场景信息

        Returns:
            处理结果字典
        """
        if not self._initialized:
            self.initialize()

        try:
            # 调用集成守护器处理场景
            result = self.guardian.process_scene(plan)

            # 添加智能体标识
            result["agent_info"] = {
                "agent_version": "1.0.0",
                "task_id": self.task_id
            }

            debug(f"帧处理完成 - 帧号: {result.get('frame_number', 'unknown')}")
            return result

        except Exception as e:
            error(f"处理帧数据失败: {e}")
            return self._create_error_response(plan, str(e))

    def process_sequence(self, plans: List[TimelinePlan]) -> Dict[str, Any]:
        """
        处理帧序列

        Args:
            plans: 帧序列列表

        Returns:
            序列处理结果
        """
        if not self._initialized:
            self.initialize()

        sequence_results = {
            "total_frames": len(plans),
            "processed_frames": 0,
            "frame_results": [],
            "sequence_summary": {},
            "continuity_score": 0.0,
            "issues_by_frame": [],
            "recommendations": []
        }

        for i, frame_data in enumerate(plans):
            try:
                frame_result = self.process(frame_data)
                sequence_results["frame_results"].append(frame_result)
                sequence_results["processed_frames"] += 1

                # 收集问题
                if frame_result.c.get("continuity_report"):
                    issues = self._extract_issues_from_report(frame_result["continuity_report"])
                    if issues:
                        sequence_results["issues_by_frame"].append({
                            "frame_index": i,
                            "frame_id": frame_data.get("scene_id", f"frame_{i}"),
                            "issues": issues
                        })

                info(f"序列处理进度: {i + 1}/{len(plans)}")

            except Exception as e:
                error(f"处理序列第{i}帧失败: {e}")
                error_result = self._create_error_response(frame_data, str(e))
                sequence_results["frame_results"].append(error_result)

        # 生成序列摘要
        sequence_results["sequence_summary"] = self._generate_sequence_summary(sequence_results)
        sequence_results["continuity_score"] = self._calculate_sequence_continuity_score(sequence_results)
        sequence_results["recommendations"] = self._generate_sequence_recommendations(sequence_results)

        return sequence_results

    def analyze_transition(self, from_scene: Dict, to_scene: Dict) -> Dict[str, Any]:
        """
        分析场景转场

        Args:
            from_scene: 来源场景数据
            to_scene: 目标场景数据

        Returns:
            转场分析结果
        """
        if not self._initialized:
            self.initialize()

        try:
            # 使用转场管理器分析
            transition_result = self.guardian.analyze_transition(from_scene, to_scene)

            # 增强结果
            enhanced_result = {
                "transition_analysis": transition_result,
                "continuity_assessment": self._assess_transition_continuity(transition_result),
                "suggestions": self._generate_transition_suggestions(transition_result),
                "risk_level": self._evaluate_transition_risk(transition_result)
            }

            return enhanced_result

        except Exception as e:
            error(f"转场分析失败: {e}")
            return {"error": str(e), "from_scene": from_scene.get("scene_id"), "to_scene": to_scene.get("scene_id")}

    def generate_constraints(self, scene_data: Dict,
                             scene_type: str = "general") -> Dict[str, Any]:
        """
        为场景生成连续性约束

        Args:
            scene_data: 场景数据
            scene_type: 场景类型

        Returns:
            约束生成结果
        """
        if not self._initialized:
            self.initialize()

        try:
            # 使用约束生成器
            constraints = self.guardian.constraint_generator.generate_constraints_for_scene(
                scene_data=scene_data,
                previous_scene=None,
                scene_type=scene_type
            )

            # 验证约束
            validation = self.guardian.constraint_generator.validate_constraints(scene_data, constraints)

            return {
                "constraints": constraints,
                "validation": validation,
                "summary": self.guardian.constraint_generator.get_constraint_summary(constraints),
                "optimization_suggestions": self._optimize_constraints(constraints, scene_data)
            }

        except Exception as e:
            error(f"约束生成失败: {e}")
            return {"error": str(e), "scene_id": scene_data.get("scene_id")}

    def validate_physics(self, scene_data: Dict) -> Dict[str, Any]:
        """
        验证场景物理合理性

        Args:
            scene_data: 场景数据

        Returns:
            物理验证结果
        """
        if not self._initialized:
            self.initialize()

        try:
            # 使用物理验证器
            physics_data = self.guardian._extract_physics_data(scene_data)
            validation_result = self.guardian.physics_validator.validate_scene_physics(physics_data)

            return {
                "plausibility_score": validation_result["overall_plausibility_score"],
                "issues": validation_result["issues"],
                "warnings": validation_result["warnings"],
                "detailed_analysis": validation_result.get("detailed_analysis", {}),
                "improvement_suggestions": self._generate_physics_improvements(validation_result)
            }

        except Exception as e:
            error(f"物理验证失败: {e}")
            return {"error": str(e), "scene_id": scene_data.get("scene_id")}

    def get_continuity_report(self, detailed: bool = True) -> Dict[str, Any]:
        """
        获取连续性报告

        Args:
            detailed: 是否获取详细报告

        Returns:
            连续性报告
        """
        if not self._initialized:
            self.initialize()

        try:
            # 获取守护器状态
            status = self.guardian.get_system_status()

            # 生成报告
            report = {
                "session_info": {
                    "task_id": self.task_id,
                    "frames_processed": status.get("frames_processed", 0),
                    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "performance_summary": status.get("performance", {}),
                "issue_statistics": self._collect_issue_statistics(),
                "continuity_score": self._calculate_overall_continuity_score(),
                "recommendations": self._generate_system_recommendations(status)
            }

            if detailed:
                report["detailed_analysis"] = {
                    "scene_complexity_distribution": self._analyze_scene_complexity(),
                    "common_issue_patterns": self._identify_common_issues(),
                    "learning_progress": self._get_learning_progress() if self.config.enable_machine_learning else None
                }

            return report

        except Exception as e:
            error(f"生成连续性报告失败: {e}")
            return {"error": str(e), "task_id": self.task_id}

    def export_session(self, export_path: Optional[str] = None) -> Dict[str, Any]:
        """
        导出会话数据

        Args:
            export_path: 导出路径，可选

        Returns:
            导出结果
        """
        if not self._initialized:
            self.initialize()

        try:
            if export_path is None:
                export_path = f"./exports/{self.task_id}.json"

            # 创建导出目录
            export_dir = Path(export_path).parent
            export_dir.mkdir(parents=True, exist_ok=True)

            # 收集导出数据
            export_data = {
                "metadata": {
                    "export_time": datetime.now().isoformat(),
                    "agent_version": "1.0.0",
                    "task_id": self.task_id
                },
                "config": self.config.to_dict(),
                "session_summary": self.get_continuity_report(detailed=False),
                "state_data": self._export_state_data(),
                "learning_data": self._export_learning_data() if self.config.enable_machine_learning else None
            }

            # 保存到文件
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            info(f"会话数据已导出: {export_path}")

            return {
                "success": True,
                "export_path": export_path,
                "export_size": Path(export_path).stat().st_size,
                "export_time": datetime.now().isoformat()
            }

        except Exception as e:
            error(f"导出会话失败: {e}")
            return {"success": False, "error": str(e)}

    def reset_session(self, task_id: Optional[str] = None):
        """
        重置会话

        Args:
            task_id: 新项目名称，可选
        """

        if task_id:
            self.task_id = task_id

        self.guardian = None
        self._initialized = False

        info(f"会话已重置: {task_id}")

        # 重新初始化
        self.initialize()

    # 辅助方法
    def _create_error_response(self, plan: TimelinePlan, error_message: str) -> AnchoredTimeline:
        """创建错误响应"""
        return AnchoredTimeline(
            anchored_segments=plan.timeline_segments
        )


    def _extract_issues_from_report(self, report_data: Any) -> List[Dict]:
        """从报告中提取问题"""
        issues = []

        if isinstance(report_data, str):
            # 如果是字符串报告，尝试解析
            try:
                if "关键问题:" in report_data:
                    # 简单解析文本报告
                    lines = report_data.split('\n')
                    for line in lines:
                        if line.strip() and (']' in line or ':' in line):
                            issues.append({"description": line.strip()})
            except:
                pass
        elif isinstance(report_data, dict):
            # 如果是字典，直接提取
            if "issues" in report_data:
                issues = report_data["issues"]

        return issues[:10]  # 限制数量

    def _generate_sequence_summary(self, sequence_results: Dict) -> Dict[str, Any]:
        """生成序列摘要"""
        total_issues = sum(len(frame.get("issues", [])) for frame in sequence_results["issues_by_frame"])
        processed_frames = sequence_results["processed_frames"]

        return {
            "total_frames": sequence_results["total_frames"],
            "successfully_processed": processed_frames,
            "success_rate": processed_frames / max(1, sequence_results["total_frames"]),
            "total_issues_found": total_issues,
            "average_issues_per_frame": total_issues / max(1, processed_frames),
            "processing_time_summary": self._calculate_processing_time_summary(sequence_results)
        }

    def _calculate_sequence_continuity_score(self, sequence_results: Dict) -> float:
        """计算序列连续性分数"""
        if not sequence_results["frame_results"]:
            return 0.0

        scores = []
        for result in sequence_results["frame_results"]:
            if result.get("success", True):
                # 如果有物理验证分数，使用它
                physics_score = result.get("physics_validation", {}).get("plausibility_score", 1.0)

                # 如果有问题，扣分
                issue_penalty = len(result.get("issues", [])) * 0.1
                score = max(0.0, physics_score - issue_penalty)
                scores.append(score)

        return np.mean(scores) if scores else 0.0

    def _generate_sequence_recommendations(self, sequence_results: Dict) -> List[str]:
        """生成序列级别的推荐"""
        recommendations = []

        # 基于问题频率
        issue_frames = len(sequence_results["issues_by_frame"])
        if issue_frames > sequence_results["total_frames"] * 0.3:
            recommendations.append(f"检测到 {issue_frames} 帧有问题，建议检查数据源质量")

        # 基于连续性分数
        continuity_score = sequence_results["continuity_score"]
        if continuity_score < 0.7:
            recommendations.append(f"序列连续性分数较低 ({continuity_score:.2f})，建议优化场景设计")

        # 基于处理性能
        time_summary = sequence_results["sequence_summary"].get("processing_time_summary", {})
        avg_time = time_summary.get("average", 0)
        if avg_time > 0.5:
            recommendations.append(f"平均处理时间较长 ({avg_time:.3f}秒)，建议简化场景或调整配置")

        return recommendations[:5]

    def _calculate_processing_time_summary(self, sequence_results: Dict) -> Dict[str, float]:
        """计算处理时间摘要"""
        times = []
        for result in sequence_results["frame_results"]:
            if "processing_time" in result:
                times.append(result["processing_time"])

        if not times:
            return {"average": 0, "min": 0, "max": 0}

        return {
            "average": np.mean(times),
            "min": min(times),
            "max": max(times),
            "total": sum(times)
        }

    def _assess_transition_continuity(self, transition_result: Dict) -> Dict[str, Any]:
        """评估转场连续性"""
        analysis = transition_result.get("transition_analysis", {})
        issues = transition_result.get("validation_issues", [])

        continuity_score = 1.0
        severity_factors = {
            "high": 0.3,
            "medium": 0.15,
            "low": 0.05
        }

        for issue in issues:
            severity = issue.get("severity", "low")
            continuity_score -= severity_factors.get(severity, 0.05)

        continuity_score = max(0.0, min(1.0, continuity_score))

        return {
            "continuity_score": continuity_score,
            "issue_count": len(issues),
            "transition_type_appropriateness": self._evaluate_transition_type(analysis.get("type", "unknown")),
            "temporal_consistency": analysis.get("temporal_gap", 0) < 10.0  # 时间间隔小于10秒
        }

    def _evaluate_transition_type(self, transition_type: str) -> str:
        """评估转场类型适当性"""
        appropriateness = {
            "cut": "适合快节奏场景切换",
            "fade": "适合时间/场景过渡",
            "dissolve": "适合梦境/回忆",
            "cross_dissolve": "通用过渡",
            "match_cut": "适合动作连续性"
        }

        return appropriateness.get(transition_type, "未知类型")

    def _generate_transition_suggestions(self, transition_result: Dict) -> List[str]:
        """生成转场建议"""
        suggestions = []
        analysis = transition_result.get("transition_analysis", {})
        issues = transition_result.get("validation_issues", [])

        # 基于转场类型
        transition_type = analysis.get("type", "")
        if transition_type == "cut":
            suggestions.append("硬切适合快速节奏，确保剪辑点在动作自然断点")
        elif transition_type == "fade":
            suggestions.append("淡入淡出适合表现时间流逝，持续时间建议1-2秒")

        # 基于问题
        for issue in issues[:3]:  # 前3个问题
            if "suggestion" in issue:
                suggestions.append(issue["suggestion"])

        return suggestions[:5]

    def _evaluate_transition_risk(self, transition_result: Dict) -> str:
        """评估转场风险"""
        issues = transition_result.get("validation_issues", [])
        high_severity = sum(1 for issue in issues if issue.get("severity") == "high")

        if high_severity > 2:
            return "high"
        elif high_severity > 0 or len(issues) > 5:
            return "medium"
        else:
            return "low"

    def _optimize_constraints(self, constraints: List[Dict], scene_data: Dict) -> List[str]:
        """优化约束建议"""
        suggestions = []

        # 检查约束数量
        if len(constraints) > 20:
            suggestions.append(f"约束数量较多 ({len(constraints)}个)，建议合并相似约束")

        # 检查约束优先级分布
        priority_count = {}
        for constraint in constraints:
            priority = constraint.get("priority", "medium")
            priority_count[priority] = priority_count.get(priority, 0) + 1

        if priority_count.get("critical", 0) > len(constraints) * 0.3:
            suggestions.append("关键优先级约束过多，考虑将部分降级")

        return suggestions

    def _generate_physics_improvements(self, validation_result: Dict) -> List[str]:
        """生成物理改进建议"""
        improvements = []

        issues = validation_result.get("issues", [])
        for issue in issues[:5]:  # 前5个问题
            if "suggested_fix" in issue:
                improvements.append(issue["suggested_fix"])

        score = validation_result.get("plausibility_score", 1.0)
        if score < 0.7:
            improvements.append(f"物理合理性分数较低 ({score:.2f})，建议检查物理参数设置")

        return improvements

    def _collect_issue_statistics(self) -> Dict[str, Any]:
        """收集问题统计"""
        # 这里需要实际的实现来收集历史问题统计
        # 简化实现
        return {
            "total_issues_detected": 0,
            "issues_by_severity": {"critical": 0, "major": 0, "minor": 0},
            "common_issue_types": [],
            "auto_fix_rate": 0.0
        }

    def _calculate_overall_continuity_score(self) -> float:
        """计算总体连续性分数"""
        # 这里需要实际的实现来计算总体分数
        # 简化实现
        return 0.85

    def _generate_system_recommendations(self, status: Dict) -> List[str]:
        """生成系统级推荐"""
        recommendations = []

        # 基于性能
        performance = status.get("performance", {})
        avg_time = performance.get("avg_processing_time", 0)
        if avg_time > 0.3:
            recommendations.append(f"平均处理时间 {avg_time:.3f}秒，建议优化配置或简化场景")

        # 基于配置
        if not self.config.enable_machine_learning:
            recommendations.append("机器学习功能未启用，启用后可以提供更好的预测和优化")

        return recommendations[:3]

    def _analyze_scene_complexity(self) -> Dict[str, int]:
        """分析场景复杂度分布"""
        # 这里需要实际的实现来分析历史场景复杂度
        # 简化实现
        return {
            "simple": 0,
            "moderate": 0,
            "complex": 0,
            "epic": 0
        }

    def _identify_common_issues(self) -> List[Dict]:
        """识别常见问题模式"""
        # 这里需要实际的实现来识别问题模式
        # 简化实现
        return []

    def _get_learning_progress(self) -> Dict[str, Any]:
        """获取学习进度"""
        # 这里需要实际的实现来获取学习进度
        # 简化实现
        return {"enabled": True, "patterns_learned": 0}

    def _export_state_data(self) -> Dict[str, Any]:
        """导出状态数据"""
        if self.guardian and self.guardian.state_tracker:
            return {
                "entities_tracked": len(self.guardian.state_tracker.entity_registry),
                "total_state_records": sum(len(h) for h in self.guardian.state_tracker.state_history.values())
            }
        return {}

    def _export_learning_data(self) -> Dict[str, Any]:
        """导出学习数据"""
        if self.guardian and self.guardian.continuity_learner:
            return self.guardian.continuity_learner.get_learning_statistics()
        return {}
