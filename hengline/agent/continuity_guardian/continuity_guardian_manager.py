"""
@FileName: continuity_guardian_manager.py
@Description: 连续性守护管理器
@Author: HengLine
@Time: 2026/1/5 16:17
"""
import math
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from hengline.agent.continuity_guardian.model.continuity_transition_guardian import TransitionInstruction, KeyframeAnchor
from hengline.logger import info, debug, error, warning
# 导入所有已创建的组件
from .analyzer.continuity_learner import ContinuityLearner
from .analyzer.pacing_adapter import PacingAdapter
from .analyzer.scene_analyzer import SceneAnalyzer
from .analyzer.spatial_Index_analyzer import SpatialIndex
from .analyzer.state_tracking_engine import StateTrackingEngine
from .analyzer.temporal_graph_analyzer import TemporalGraph
from .analyzer.timeline_analyzer import TimelineAnalyzer
from .anchor_manager import AnchorManager
from .continuity_constraint_generator import ContinuityConstraintGenerator
from .continuity_guardian_model import ContinuityLevel
from .detector.change_detector import ChangeDetector
from .detector.consistency_detector import ConsistencyChecker
from .model.continuity_guard_guardian import GuardianConfig, SceneComplexity
from .model.continuity_guardian_report import ValidationReport, ContinuityIssue, AutoFix
from .model.continuity_rule_guardian import ContinuityRuleSet, GenerationHints
from .model.continuity_state_guardian import StateSnapshot, CharacterState, PropState, EnvironmentState
from .model.continuity_visual_guardian import SpatialRelation, VisualSignature, VisualMatchRequirements
from .scene_transition_manager import SceneTransitionManager
from .validator.bounding_validator import BoundingVolumeType
from .validator.collision_validator import CollisionDetector, CollisionType
from .validator.material_validator import MaterialDatabase
from .validator.motion_validator import MotionAnalyzer
from .validator.physical_validator import PhysicalPlausibilityValidator
from ..temporal_planner.temporal_planner_model import TimelinePlan, TimeSegment, ContinuityAnchor


class ContinuityGuardian:
    """连续性守护智能体（优化版）- 接收TimelinePlan作为输入"""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.timeline_plan: Optional[TimelinePlan] = None

        # 重用之前创建的所有分析器和检查器
        self.constraint_generator = ContinuityConstraintGenerator()
        self.physical_validator = PhysicalPlausibilityValidator()
        self.state_tracker = StateTrackingEngine()

        # 添加专门针对TimelinePlan的分析器
        self.timeline_analyzer = TimelineAnalyzer()
        self.anchor_manager = AnchorManager()
        self.pacing_adapter = PacingAdapter()

        # 性能监控
        self.performance_metrics: Dict[str, List] = {
            "analysis_times": [],
            "validation_times": [],
            "issue_counts": []
        }

        info(f"连续性守护器初始化完成 - 项目: {project_id}")

    def analyze_timeline(self, timeline_plan: TimelinePlan) -> Dict[str, Any]:
        """分析时序规划，生成连续性守护策略"""
        start_time = datetime.now()
        self.timeline_plan = timeline_plan

        info(f"开始分析时序规划，包含 {len(timeline_plan.timeline_segments)} 个片段")

        # 步骤1：分析时序结构
        timeline_analysis = self.timeline_analyzer.analyze_timeline_structure(timeline_plan)

        # 步骤2：分析连续性锚点
        anchor_analysis = self.anchor_manager.analyze_continuity_anchors(timeline_plan.continuity_anchors)

        # 步骤3：根据节奏调整分析策略
        pacing_strategy = self.pacing_adapter.adapt_to_pacing(timeline_plan.pacing_analysis)

        # 步骤4：生成分段的连续性约束
        segment_constraints = self._generate_segment_constraints(timeline_plan.timeline_segments)

        # 步骤5：评估整体连续性风险
        continuity_risk = self._assess_continuity_risk(timeline_plan, segment_constraints)

        # 步骤6：生成连续性守护策略
        guardian_strategy = self._generate_guardian_strategy(
            timeline_analysis, anchor_analysis, pacing_strategy,
            segment_constraints, continuity_risk
        )

        # 性能记录
        analysis_time = (datetime.now() - start_time).total_seconds()
        self._record_performance_metric("analysis_times", analysis_time)

        result = {
            "project_id": self.project_id,
            "timeline_analysis": timeline_analysis,
            "anchor_analysis": anchor_analysis,
            "pacing_strategy": pacing_strategy,
            "segment_constraints": segment_constraints,
            "continuity_risk": continuity_risk,
            "guardian_strategy": guardian_strategy,
            "analysis_time": analysis_time,
            "summary": self._generate_summary(timeline_plan, guardian_strategy)
        }

        info(f"时序规划分析完成，用时: {analysis_time:.2f}秒")

        return result

    def _generate_segment_constraints(self, segments: List[TimeSegment]) -> Dict[str, List[Dict]]:
        """为每个时间片段生成连续性约束"""
        constraints = {}

        for segment in segments:
            segment_id = segment.segment_id

            # 根据片段内容生成约束
            segment_data = {
                "segment_id": segment_id,
                "content_description": segment.content_description,
                "key_elements": segment.key_elements,
                "complexity_level": segment.complexity_level,
                "continuity_requirements": segment.continuity_requirements
            }

            # 使用约束生成器
            segment_constraints = self.constraint_generator.generate_constraints_for_scene(
                scene_data=segment_data,
                scene_type=self._determine_scene_type(segment)
            )

            constraints[segment_id] = segment_constraints

        return constraints

    def _determine_scene_type(self, segment: TimeSegment) -> str:
        """根据片段内容确定场景类型"""
        description = segment.content_description.lower()

        if any(word in description for word in ["对话", "交谈", "说话"]):
            return "dialogue"
        elif any(word in description for word in ["动作", "战斗", "追逐", "打斗"]):
            return "action_sequence"
        elif any(word in description for word in ["环境", "场景", "背景", "建立"]):
            return "environment_establishing"
        elif any(word in description for word in ["情感", "内心", "思考"]):
            return "emotional"
        else:
            return "general"

    def _assess_continuity_risk(self, timeline_plan: TimelinePlan,
                                segment_constraints: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """评估整体连续性风险"""
        risk_assessment = {
            "overall_risk_score": 0.0,
            "risk_factors": [],
            "high_risk_segments": [],
            "mitigation_strategies": []
        }

        risk_factors = []

        # 因素1：片段数量风险
        segment_count = len(timeline_plan.timeline_segments)
        if segment_count > 20:
            risk_factors.append({
                "factor": "excessive_segments",
                "severity": "medium",
                "description": f"片段数量过多 ({segment_count}个)，可能增加连续性风险",
                "mitigation": "考虑合并相关片段"
            })

        # 因素2：复杂性风险
        complex_segments = [s for s in timeline_plan.timeline_segments
                            if s.complexity_level == "high"]
        if len(complex_segments) > 5:
            risk_factors.append({
                "factor": "high_complexity_concentration",
                "severity": "high",
                "description": f"高复杂度片段过多 ({len(complex_segments)}个)",
                "mitigation": "分散高复杂度片段"
            })

        # 因素3：时间跳跃风险
        time_jumps = self._detect_time_jumps(timeline_plan.timeline_segments)
        if time_jumps:
            risk_factors.append({
                "factor": "time_jumps",
                "severity": "medium",
                "description": f"检测到 {len(time_jumps)} 个时间跳跃",
                "mitigation": "添加时间过渡或说明",
                "details": time_jumps
            })

        # 因素4：约束冲突风险
        constraint_conflicts = self._detect_constraint_conflicts(segment_constraints)
        if constraint_conflicts:
            risk_factors.append({
                "factor": "constraint_conflicts",
                "severity": "high",
                "description": f"检测到 {len(constraint_conflicts)} 个约束冲突",
                "mitigation": "解决约束冲突",
                "details": constraint_conflicts[:3]  # 只显示前3个
            })

        risk_assessment["risk_factors"] = risk_factors

        # 计算总体风险分数
        severity_scores = {"high": 1.0, "medium": 0.7, "low": 0.3}
        total_score = sum(severity_scores.get(factor["severity"], 0.5)
                          for factor in risk_factors)

        if risk_factors:
            risk_assessment["overall_risk_score"] = min(1.0, total_score / len(risk_factors))
        else:
            risk_assessment["overall_risk_score"] = 0.0

        # 识别高风险片段
        for segment in timeline_plan.timeline_segments:
            if segment.complexity_level == "high":
                risk_assessment["high_risk_segments"].append({
                    "segment_id": segment.segment_id,
                    "reason": "高复杂度",
                    "priority": "high"
                })

        # 生成缓解策略
        risk_assessment["mitigation_strategies"] = self._generate_risk_mitigation_strategies(risk_factors)

        return risk_assessment

    def _detect_time_jumps(self, segments: List[TimeSegment]) -> List[Dict]:
        """检测时间跳跃"""
        time_jumps = []

        for i in range(1, len(segments)):
            prev_segment = segments[i - 1]
            curr_segment = segments[i]

            # 检查内容描述中的时间变化
            prev_desc = prev_segment.content_description
            curr_desc = curr_segment.content_description

            time_keywords = ["第二天", "几天后", "多年后", "突然", "同时", "回到", "之前", "之后"]

            jump_detected = False
            jump_description = ""

            for keyword in time_keywords:
                if keyword in prev_desc or keyword in curr_desc:
                    jump_detected = True
                    jump_description = f"检测到时间关键词: '{keyword}'"
                    break

            # 检查连续性要求中的时间变化
            if "time_gap" in prev_segment.continuity_requirements:
                time_gap = prev_segment.continuity_requirements["time_gap"]
                if time_gap > 3600:  # 1小时以上
                    jump_detected = True
                    jump_description = f"时间间隔过大: {time_gap}秒"

            if jump_detected:
                time_jumps.append({
                    "position": i,
                    "from_segment": prev_segment.segment_id,
                    "to_segment": curr_segment.segment_id,
                    "description": jump_description
                })

        return time_jumps

    def _detect_constraint_conflicts(self, segment_constraints: Dict[str, List[Dict]]) -> List[Dict]:
        """检测约束冲突"""
        conflicts = []

        segment_ids = list(segment_constraints.keys())

        for i in range(len(segment_ids)):
            for j in range(i + 1, len(segment_ids)):
                seg1_id = segment_ids[i]
                seg2_id = segment_ids[j]

                constraints1 = segment_constraints[seg1_id]
                constraints2 = segment_constraints[seg2_id]

                # 检查相同元素的约束冲突
                for const1 in constraints1:
                    for const2 in constraints2:
                        if self._constraints_conflict(const1, const2):
                            conflicts.append({
                                "segment1": seg1_id,
                                "segment2": seg2_id,
                                "constraint1": const1.get("type", "unknown"),
                                "constraint2": const2.get("type", "unknown"),
                                "conflict_type": "cross_segment_constraint_conflict"
                            })

        return conflicts

    def _constraints_conflict(self, constraint1: Dict, constraint2: Dict) -> bool:
        """检查两个约束是否冲突"""
        # 简化的冲突检测
        type1 = constraint1.get("type", "")
        type2 = constraint2.get("type", "")

        # 物理约束与动作约束的冲突
        if "physical" in type1 and "motion" in type2:
            return True

        # 时间约束与叙事约束的冲突
        if "temporal" in type1 and "narrative" in type2:
            return True

        return False

    def _generate_risk_mitigation_strategies(self, risk_factors: List[Dict]) -> List[str]:
        """生成风险缓解策略"""
        strategies = []

        for factor in risk_factors:
            factor_type = factor["factor"]

            if factor_type == "excessive_segments":
                strategies.append("实施分段合并策略，减少片段数量")

            elif factor_type == "high_complexity_concentration":
                strategies.append("重新分配高复杂度片段，避免集中")
                strategies.append("为高复杂度片段分配更多资源和时间")

            elif factor_type == "time_jumps":
                strategies.append("为时间跳跃添加视觉过渡效果")
                strategies.append("确保时间跳跃有明确的叙事逻辑")

            elif factor_type == "constraint_conflicts":
                strategies.append("建立约束冲突解决机制")
                strategies.append("优化约束生成策略，减少冲突")

        # 通用策略
        if len(risk_factors) > 3:
            strategies.append("增加验证频率和深度")
            strategies.append("建立实时监控和预警系统")

        return list(set(strategies))[:10]  # 去重并限制数量

    def _generate_guardian_strategy(self, timeline_analysis: Dict, anchor_analysis: Dict,
                                    pacing_strategy: Dict, segment_constraints: Dict,
                                    continuity_risk: Dict) -> Dict[str, Any]:
        """生成连续性守护策略"""
        strategy = {
            "monitoring_plan": self._generate_monitoring_plan(timeline_analysis),
            "validation_schedule": self._generate_validation_schedule(anchor_analysis, pacing_strategy),
            "resource_allocation": self._allocate_resources(continuity_risk, segment_constraints),
            "intervention_protocol": self._define_intervention_protocol(continuity_risk),
            "quality_assurance": self._define_quality_assurance_measures(continuity_risk)
        }

        return strategy

    def _generate_monitoring_plan(self, timeline_analysis: Dict) -> Dict[str, Any]:
        """生成监控计划"""
        monitoring_plan = {
            "monitoring_points": [],
            "monitoring_frequency": "adaptive",
            "monitoring_depth": "standard",
            "alert_thresholds": {
                "critical_issues": 1,
                "major_issues": 3,
                "minor_issues": 10
            }
        }

        # 在关键点设置监控
        if self.timeline_plan:
            for anchor in self.timeline_plan.continuity_anchors:
                if anchor.importance > 0.7:  # 高重要性锚点
                    monitoring_plan["monitoring_points"].append({
                        "point_id": f"monitor_{anchor.anchor_id}",
                        "timestamp": anchor.timestamp,
                        "monitoring_focus": anchor.anchor_type,
                        "priority": "high"
                    })

            # 在高复杂度片段设置监控
            for segment in self.timeline_plan.timeline_segments:
                if segment.complexity_level == "high":
                    mid_time = (segment.time_range[0] + segment.time_range[1]) / 2
                    monitoring_plan["monitoring_points"].append({
                        "point_id": f"monitor_{segment.segment_id}",
                        "timestamp": mid_time,
                        "monitoring_focus": "complexity_management",
                        "priority": "medium"
                    })

        return monitoring_plan

    def _generate_validation_schedule(self, anchor_analysis: Dict,
                                      pacing_strategy: Dict) -> Dict[str, Any]:
        """生成验证计划"""
        validation_schedule = {
            "anchor_validations": [],
            "segment_validations": [],
            "full_validations": [],
            "validation_intensity": "adaptive"
        }

        if self.timeline_plan:
            # 锚点验证
            for anchor in self.timeline_plan.continuity_anchors:
                validation_schedule["anchor_validations"].append({
                    "anchor_id": anchor.anchor_id,
                    "timestamp": anchor.timestamp,
                    "validation_type": "anchor_continuity",
                    "priority": "high" if anchor.importance > 0.7 else "medium"
                })

            # 片段验证（根据节奏调整频率）
            pacing = self.timeline_plan.pacing_analysis.overall_pacing
            validation_frequency = {"slow": 3, "medium": 2, "fast": 1}[pacing]

            for i, segment in enumerate(self.timeline_plan.timeline_segments):
                if i % validation_frequency == 0:
                    validation_schedule["segment_validations"].append({
                        "segment_id": segment.segment_id,
                        "start_time": segment.time_range[0],
                        "validation_type": "segment_continuity",
                        "priority": "high" if segment.complexity_level == "high" else "medium"
                    })

        return validation_schedule

    def _allocate_resources(self, continuity_risk: Dict,
                            segment_constraints: Dict) -> Dict[str, Any]:
        """分配资源"""
        resource_allocation = {
            "computational_resources": {},
            "human_resources": {},
            "time_resources": {}
        }

        risk_score = continuity_risk["overall_risk_score"]

        # 计算资源分配
        if risk_score > 0.7:
            resource_allocation["computational_resources"] = {
                "cpu_cores": 8,
                "gpu_memory": "high",
                "processing_priority": "realtime"
            }
        elif risk_score > 0.4:
            resource_allocation["computational_resources"] = {
                "cpu_cores": 4,
                "gpu_memory": "medium",
                "processing_priority": "high"
            }
        else:
            resource_allocation["computational_resources"] = {
                "cpu_cores": 2,
                "gpu_memory": "low",
                "processing_priority": "normal"
            }

        # 时间资源分配（基于约束复杂度）
        total_constraints = sum(len(constraints) for constraints in segment_constraints.values())
        avg_constraints_per_segment = total_constraints / max(1, len(segment_constraints))

        if avg_constraints_per_segment > 10:
            resource_allocation["time_resources"]["validation_time_per_segment"] = "extended"
        elif avg_constraints_per_segment > 5:
            resource_allocation["time_resources"]["validation_time_per_segment"] = "standard"
        else:
            resource_allocation["time_resources"]["validation_time_per_segment"] = "quick"

        return resource_allocation

    def _define_intervention_protocol(self, continuity_risk: Dict) -> Dict[str, Any]:
        """定义干预协议"""
        protocol = {
            "auto_intervention_levels": {},
            "human_intervention_triggers": [],
            "recovery_strategies": []
        }

        risk_score = continuity_risk["overall_risk_score"]

        # 自动干预级别
        if risk_score > 0.8:
            protocol["auto_intervention_levels"] = {
                "level": "aggressive",
                "allowed_interventions": ["auto_fix", "constraint_relaxation", "quality_reduction"],
                "intervention_threshold": 0.5  # 超过50%问题自动干预
            }
        elif risk_score > 0.5:
            protocol["auto_intervention_levels"] = {
                "level": "moderate",
                "allowed_interventions": ["auto_fix", "constraint_relaxation"],
                "intervention_threshold": 0.7
            }
        else:
            protocol["auto_intervention_levels"] = {
                "level": "conservative",
                "allowed_interventions": ["auto_fix"],
                "intervention_threshold": 0.9
            }

        # 人工干预触发条件
        protocol["human_intervention_triggers"] = [
            "连续3个片段出现严重连续性错误",
            "自动修复失败超过3次",
            "用户明确请求人工审核",
            "系统检测到无法自动处理的复杂情况"
        ]

        # 恢复策略
        protocol["recovery_strategies"] = [
            "回滚到最近的有效状态",
            "应用约束松弛后的替代方案",
            "请求人工指定修复方案",
            "生成问题报告并继续处理"
        ]

        return protocol

    def _define_quality_assurance_measures(self, continuity_risk: Dict) -> Dict[str, Any]:
        """定义质量保证措施"""
        quality_measures = {
            "quality_checkpoints": [],
            "quality_metrics": {},
            "acceptance_criteria": {}
        }

        # 质量检查点
        if self.timeline_plan:
            # 在高潮点设置质量检查
            for climax_point in self.timeline_plan.pacing_analysis.climax_points:
                quality_measures["quality_checkpoints"].append({
                    "checkpoint_id": f"quality_climax_{len(quality_measures['quality_checkpoints'])}",
                    "timestamp": climax_point,
                    "check_type": "comprehensive",
                    "importance": "critical"
                })

        # 质量指标
        quality_measures["quality_metrics"] = {
            "continuity_score_target": 0.9,
            "physical_plausibility_target": 0.85,
            "visual_consistency_target": 0.95,
            "temporal_coherence_target": 0.88
        }

        # 验收标准
        risk_score = continuity_risk["overall_risk_score"]

        if risk_score > 0.7:
            quality_measures["acceptance_criteria"] = {
                "min_continuity_score": 0.7,
                "max_critical_issues": 5,
                "max_major_issues": 10,
                "allow_partial_completion": True
            }
        else:
            quality_measures["acceptance_criteria"] = {
                "min_continuity_score": 0.85,
                "max_critical_issues": 1,
                "max_major_issues": 5,
                "allow_partial_completion": False
            }

        return quality_measures

    def _generate_summary(self, timeline_plan: TimelinePlan,
                          guardian_strategy: Dict) -> Dict[str, Any]:
        """生成分析摘要"""
        total_duration = sum(
            de.estimated_duration
            for de in timeline_plan.duration_estimations.values()
        )

        high_complexity_segments = len([
            s for s in timeline_plan.timeline_segments
            if s.complexity_level == "high"
        ])

        summary = {
            "total_segments": len(timeline_plan.timeline_segments),
            "total_duration": total_duration,
            "high_complexity_segments": high_complexity_segments,
            "continuity_anchors": len(timeline_plan.continuity_anchors),
            "overall_pacing": timeline_plan.pacing_analysis.overall_pacing,
            "recommended_strategy": self._recommend_primary_strategy(guardian_strategy),
            "estimated_effort": self._estimate_effort(timeline_plan, guardian_strategy)
        }

        return summary

    def _recommend_primary_strategy(self, guardian_strategy: Dict) -> str:
        """推荐主要策略"""
        monitoring_points = len(guardian_strategy["monitoring_plan"]["monitoring_points"])

        if monitoring_points > 10:
            return "密集监控和主动干预策略"
        elif monitoring_points > 5:
            return "平衡监控和响应式干预策略"
        else:
            return "轻量监控和保守干预策略"

    def _estimate_effort(self, timeline_plan: TimelinePlan,
                         guardian_strategy: Dict) -> Dict[str, float]:
        """估算工作量"""
        total_segments = len(timeline_plan.timeline_segments)
        high_complexity = len([
            s for s in timeline_plan.timeline_segments
            if s.complexity_level == "high"
        ])

        # 基础工作量
        base_effort = total_segments * 0.5  # 每个片段0.5单位

        # 复杂度加成
        complexity_effort = high_complexity * 1.0

        # 监控工作量
        monitoring_effort = len(guardian_strategy["monitoring_plan"]["monitoring_points"]) * 0.3

        total_effort = base_effort + complexity_effort + monitoring_effort

        return {
            "total_effort_units": total_effort,
            "estimated_hours": total_effort * 2,  # 假设每单位2小时
            "breakdown": {
                "base_processing": base_effort,
                "complexity_handling": complexity_effort,
                "monitoring": monitoring_effort
            }
        }

    def _record_performance_metric(self, metric_name: str, value: float):
        """记录性能指标"""
        if metric_name in self.performance_metrics:
            self.performance_metrics[metric_name].append(value)

            # 保持历史长度
            if len(self.performance_metrics[metric_name]) > 100:
                self.performance_metrics[metric_name].pop(0)
