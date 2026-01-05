"""
@FileName: continuity_guardian_manager.py
@Description: 
@Author: HengLine
@Time: 2026/1/5 16:17
"""
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np

from hengline.agent.continuity_guardian.model.continuity_transition_guardian import TransitionInstruction
from hengline.logger import info, debug, error, warning
# 导入所有已创建的组件
from .analyzer.continuity_learner import ContinuityLearner
from .analyzer.scene_analyzer import SceneAnalyzer
from .analyzer.spatial_Index_analyzer import SpatialIndex
from .analyzer.state_tracking_engine import StateTrackingEngine
from .analyzer.temporal_graph_analyzer import TemporalGraph
from .continuity_constraint_generator import ContinuityConstraintGenerator
from .continuity_guardian_model import ContinuityLevel
from .detector.change_detector import ChangeDetector
from .detector.consistency_detector import ConsistencyChecker
from .model.continuity_guard_guardian import GuardianConfig, SceneComplexity
from .model.continuity_guardian_report import ValidationReport, ContinuityIssue, AutoFix
from .model.continuity_rule_guardian import ContinuityRuleSet, GenerationHints
from .scene_transition_manager import SceneTransitionManager
from .validator.bounding_validator import BoundingVolumeType
from .validator.collision_validator import CollisionDetector, CollisionType
from .validator.material_validator import MaterialDatabase
from .validator.motion_validator import MotionAnalyzer
from .validator.physical_validator import PhysicalPlausibilityValidator


class IntegratedContinuityGuardian:
    """集成式连续性守护智能体 - 正确复用所有组件"""

    def __init__(self, task_id: str, config: Optional[GuardianConfig] = None):
        self.task_id = task_id
        self.config = config or GuardianConfig(task_id=task_id)

        # 初始化所有核心组件（单例模式，避免重复创建）
        self._init_core_components()

        # 性能监控
        self.performance_metrics: Dict[str, List] = {
            "processing_times": [],
            "validation_times": [],
            "constraint_generation_times": [],
            "physics_validation_times": []
        }

        # 会话管理
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.frame_counter = 0

        info(f"集成式连续性守护器初始化完成 - task_id: {task_id}")

    def _init_core_components(self):
        """初始化所有核心组件（确保单例）"""
        # 1. 状态跟踪引擎（核心状态管理）
        self.state_tracker = StateTrackingEngine(
            tracking_config={
                "history_length": self.config.max_state_history,
                "tracking_precision": "high"
            }
        )

        # 2. 规则集和自动修复
        self.rule_set = ContinuityRuleSet()
        self.auto_fixer = AutoFix(self.rule_set)

        # 3. 约束生成器（复用已创建的）
        self.constraint_generator = ContinuityConstraintGenerator(
            rule_set=self.rule_set.rules
        )

        # 4. 物理合理性验证器（复用已创建的）
        self.physics_validator = PhysicalPlausibilityValidator()

        # 5. 场景分析器
        self.scene_analyzer = SceneAnalyzer(self.config)

        # 6. 学习器（如果启用）
        if self.config.enable_machine_learning:
            self.continuity_learner = ContinuityLearner(self.config)
        else:
            self.continuity_learner = None

        # 7. 转场管理器
        self.transition_manager = SceneTransitionManager(self.config)

        # 8. 辅助组件（按需初始化）
        self._init_auxiliary_components()

    def _init_auxiliary_components(self):
        """初始化辅助组件（延迟初始化）"""
        self._collision_detector = None
        self._motion_analyzer = None
        self._material_database = None
        self._spatial_index = None
        self._change_detector = None
        self._consistency_checker = None
        self._temporal_graph = None

    def _get_collision_detector(self):
        """获取碰撞检测器（懒加载）"""
        if self._collision_detector is None:
            self._collision_detector = CollisionDetector(collision_margin=0.01)
        return self._collision_detector

    def _get_motion_analyzer(self):
        """获取运动分析器（懒加载）"""
        if self._motion_analyzer is None:
            self._motion_analyzer = MotionAnalyzer(sampling_rate=0.1)
        return self._motion_analyzer

    def _get_material_database(self):
        """获取材料数据库（懒加载）"""
        if self._material_database is None:
            self._material_database = MaterialDatabase()
        return self._material_database

    def _get_spatial_index(self):
        """获取空间索引（懒加载）"""
        if self._spatial_index is None:
            self._spatial_index = SpatialIndex(cell_size=2.0)
        return self._spatial_index

    def _get_change_detector(self):
        """获取变化检测器（懒加载）"""
        if self._change_detector is None:
            self._change_detector = ChangeDetector()
        return self._change_detector

    def _get_consistency_checker(self):
        """获取一致性检查器（懒加载）"""
        if self._consistency_checker is None:
            self._consistency_checker = ConsistencyChecker()
        return self._consistency_checker

    def _get_temporal_graph(self):
        """获取时间图（懒加载）"""
        if self._temporal_graph is None:
            self._temporal_graph = TemporalGraph(max_events=10000)
        return self._temporal_graph

    def process_scene(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理场景数据（集成所有组件）"""
        start_time = datetime.now()
        self.frame_counter += 1

        debug(f"处理场景 #{self.frame_counter}: {scene_data.get('scene_id', 'unknown')}")

        # 步骤1：场景分析（复用SceneAnalyzer）
        scene_analysis = self.scene_analyzer.analyze_scene_complexity(scene_data)

        # 步骤2：状态跟踪（复用StateTrackingEngine）
        state_update_result = self._track_scene_state(scene_data)

        # 步骤3：约束生成（复用ContinuityConstraintGenerator）
        constraints = self._generate_constraints_integrated(scene_data, scene_analysis)

        # 步骤4：物理验证（复用PhysicalPlausibilityValidator）
        physics_validation = self._validate_physics_integrated(scene_data)

        # 步骤5：连续性验证（集成多个验证器）
        continuity_report = self._validate_continuity_integrated(scene_data, constraints)

        # 步骤6：碰撞检测（如果需要）
        collision_analysis = None
        if self._needs_collision_detection(scene_data):
            collision_analysis = self._analyze_collisions(scene_data)

        # 步骤7：运动分析（如果需要）
        motion_analysis = None
        if self._needs_motion_analysis(scene_data):
            motion_analysis = self._analyze_motions(scene_data)

        # 步骤8：生成提示和建议
        generation_hints = self._generate_integrated_hints(scene_data)

        # 步骤9：学习（如果启用）
        if self.continuity_learner and continuity_report:
            self._learn_from_results(continuity_report)

        # 构建结果
        result = {
            "session_id": self.session_id,
            "frame_number": self.frame_counter,
            "scene_id": scene_data.get("scene_id", "unknown"),
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "scene_analysis": scene_analysis,
            "state_tracking": state_update_result,
            "constraints_generated": len(constraints),
            "physics_validation": physics_validation,
            "continuity_report": continuity_report.generate_summary() if continuity_report else None,
            "collision_analysis": collision_analysis,
            "motion_analysis": motion_analysis,
            "generation_hints": generation_hints.continuity_constraints,
            "recommendations": self._generate_integrated_recommendations(
                scene_analysis, physics_validation, continuity_report
            )
        }

        # 性能记录
        self._record_performance("processing_times", result["processing_time"])

        info(f"场景处理完成 - 帧: {self.frame_counter}")

        return result

    def _track_scene_state(self, scene_data: Dict) -> Dict[str, Any]:
        """使用StateTrackingEngine跟踪场景状态"""
        tracking_result = {
            "characters_registered": 0,
            "props_registered": 0,
            "state_updates": []
        }

        # 注册和更新角色
        for character in scene_data.get("characters", []):
            char_id = character.get("id", f"char_{len(tracking_result['state_updates'])}")

            # 提取角色状态
            char_state = self._extract_character_state_for_tracking(character)

            # 注册或更新
            if not self.state_tracker.register_entity("character", char_id, char_state):
                # 已注册，更新状态
                update_result = self.state_tracker.update_entity_state(
                    char_id, char_state, datetime.now(), "scene_update"
                )
                tracking_result["state_updates"].append(update_result)
            else:
                tracking_result["characters_registered"] += 1

        # 注册和更新道具
        for prop in scene_data.get("props", []):
            prop_id = prop.get("id", f"prop_{len(tracking_result['state_updates'])}")

            # 提取道具状态
            prop_state = self._extract_prop_state_for_tracking(prop)

            # 注册或更新
            if not self.state_tracker.register_entity("prop", prop_id, prop_state):
                update_result = self.state_tracker.update_entity_state(
                    prop_id, prop_state, datetime.now(), "scene_update"
                )
                tracking_result["state_updates"].append(update_result)
            else:
                tracking_result["props_registered"] += 1

        # 更新环境状态
        environment = scene_data.get("environment", {})
        env_state = self._extract_environment_state_for_tracking(environment)
        self.state_tracker.register_entity("environment", "global_environment", env_state)

        return tracking_result

    def _extract_character_state_for_tracking(self, character_data: Dict) -> Dict:
        """为状态跟踪提取角色状态"""
        state = {}

        # 基础属性
        if "position" in character_data:
            state["position"] = character_data["position"]

        if "rotation" in character_data:
            state["rotation"] = character_data["rotation"]

        if "velocity" in character_data:
            state["velocity"] = character_data["velocity"]

        # 外观和状态
        if "appearance" in character_data:
            state["appearance"] = character_data["appearance"]

        if "outfit" in character_data:
            state["outfit"] = character_data["outfit"]
        elif "clothing" in character_data:
            state["outfit"] = {"type": character_data["clothing"]}

        if "emotion" in character_data:
            state["emotion"] = character_data["emotion"]

        if "action" in character_data:
            state["action"] = character_data["action"]

        return state

    def _extract_prop_state_for_tracking(self, prop_data: Dict) -> Dict:
        """为状态跟踪提取道具状态"""
        state = {}

        # 基础属性
        if "position" in prop_data:
            state["position"] = prop_data["position"]

        if "rotation" in prop_data:
            state["rotation"] = prop_data["rotation"]

        if "state" in prop_data:
            state["state"] = prop_data["state"]

        if "owner" in prop_data:
            state["owner"] = prop_data["owner"]

        # 物理属性
        if "physics" in prop_data:
            state["physics"] = prop_data["physics"]

        if "material" in prop_data:
            state["material"] = prop_data["material"]

        return state

    def _extract_environment_state_for_tracking(self, environment_data: Dict) -> Dict:
        """为状态跟踪提取环境状态"""
        state = {}

        if "time_of_day" in environment_data:
            state["time_of_day"] = environment_data["time_of_day"]

        if "weather" in environment_data:
            state["weather"] = environment_data["weather"]

        if "lighting" in environment_data:
            state["lighting"] = environment_data["lighting"]

        if "effects" in environment_data:
            state["effects"] = environment_data["effects"]

        return state

    def _generate_constraints_integrated(self, scene_data: Dict,
                                         scene_analysis: Any) -> List[Dict]:
        """使用ContinuityConstraintGenerator生成约束"""
        start_time = datetime.now()

        # 获取前一场景状态（用于连续性约束）
        previous_state = None
        state_summary = self.state_tracker.get_state_summary("global_environment")
        if state_summary and "current_state_summary" in state_summary:
            previous_state = {
                "environment": state_summary["current_state_summary"]
            }

        # 生成约束
        scene_type = scene_data.get("scene_type", "general")
        constraints = self.constraint_generator.generate_constraints_for_scene(
            scene_data=scene_data,
            previous_scene=previous_state,
            scene_type=scene_type
        )

        # 验证约束适用性
        validation = self.constraint_generator.validate_constraints(scene_data, constraints)

        # 根据验证结果优化约束
        if validation["inapplicable_constraints"]:
            constraints = [c for c in constraints
                           if c not in validation["inapplicable_constraints"]]

        # 性能记录
        gen_time = (datetime.now() - start_time).total_seconds()
        self._record_performance("constraint_generation_times", gen_time)

        debug(f"生成 {len(constraints)} 个约束，用时 {gen_time:.3f}秒")

        return constraints

    def _validate_physics_integrated(self, scene_data: Dict) -> Dict[str, Any]:
        """使用PhysicalPlausibilityValidator验证物理合理性"""
        start_time = datetime.now()

        # 提取物理相关数据
        physics_data = self._extract_physics_data(scene_data)

        # 使用物理验证器
        validation_result = self.physics_validator.validate_scene_physics(physics_data)

        # 性能记录
        val_time = (datetime.now() - start_time).total_seconds()
        self._record_performance("physics_validation_times", val_time)

        return {
            "plausibility_score": validation_result["overall_plausibility_score"],
            "issues": len(validation_result["issues"]),
            "warnings": len(validation_result["warnings"]),
            "validation_time": val_time,
            "detailed_analysis": validation_result.get("detailed_analysis", {})
        }

    def _extract_physics_data(self, scene_data: Dict) -> Dict:
        """提取物理验证所需的数据"""
        physics_data = {
            "scene_id": scene_data.get("scene_id", f"scene_{self.frame_counter}"),
            "characters": [],
            "props": [],
            "environment": scene_data.get("environment", {}),
            "motions": [],
            "collisions": []
        }

        # 提取角色物理数据
        for character in scene_data.get("characters", []):
            char_physics = {
                "id": character.get("id"),
                "position": character.get("position"),
                "velocity": character.get("velocity"),
                "action": character.get("action")
            }

            # 添加质量估计
            if "scale" in character:
                char_physics["mass"] = 70.0 * character["scale"]  # 基于比例估算

            physics_data["characters"].append(char_physics)

        # 提取道具物理数据
        for prop in scene_data.get("props", []):
            prop_physics = {
                "id": prop.get("id"),
                "position": prop.get("position"),
                "state": prop.get("state", "static")
            }

            # 添加物理属性
            if "physics" in prop:
                prop_physics.update(prop["physics"])

            physics_data["props"].append(prop_physics)

        return physics_data

    def _validate_continuity_integrated(self, scene_data: Dict,
                                        constraints: List[Dict]) -> Optional[ValidationReport]:
        """集成验证连续性"""
        if not self._should_validate_continuity():
            return None

        report = ValidationReport(f"continuity_{self.frame_counter}")

        # 1. 使用StateTrackingEngine检测异常
        anomalies = self._detect_state_anomalies()
        for anomaly in anomalies:
            issue = ContinuityIssue(
                issue_id=f"anomaly_{anomaly.get('type', 'unknown')}",
                level=ContinuityLevel.MAJOR,
                description=anomaly.get("description", "状态异常")
            )
            report.add_issue(issue)

        # 2. 验证约束符合性
        constraint_validation = self._validate_constraints_compliance(constraints, scene_data)
        report.summary["total_checks"] += constraint_validation["total_checks"]
        report.summary["passed"] += constraint_validation["passed"]

        # 3. 使用ConsistencyChecker验证一致性
        if self._needs_consistency_check(scene_data):
            consistency_result = self._check_temporal_consistency(scene_data)
            for issue in consistency_result.get("inconsistencies", []):
                continuity_issue = ContinuityIssue(
                    issue_id=f"consistency_{issue.get('rule', 'unknown')}",
                    level=issue.get("severity", "MINOR").upper(),
                    description=issue.get("description", "一致性违反")
                )
                report.add_issue(continuity_issue)

        # 4. 使用ChangeDetector检测变化
        change_analysis = self._analyze_state_changes(scene_data)
        if change_analysis.get("significant_changes"):
            for change in change_analysis["significant_changes"]:
                issue = ContinuityIssue(
                    issue_id=f"change_{change.get('attribute', 'unknown')}",
                    level=ContinuityLevel.MINOR,
                    description=f"显著变化: {change.get('attribute')}"
                )
                report.add_issue(issue)

        return report

    def _detect_state_anomalies(self) -> List[Dict]:
        """使用StateTrackingEngine检测状态异常"""
        anomalies = []

        # 获取所有实体的状态摘要
        for entity_id in self.state_tracker.entity_registry:
            entity_anomalies = self.state_tracker.detect_anomalies(entity_id, window_size=5)
            anomalies.extend(entity_anomalies)

        return anomalies

    def _validate_constraints_compliance(self, constraints: List[Dict],
                                         scene_data: Dict) -> Dict[str, int]:
        """验证约束符合性"""
        total_checks = len(constraints)
        passed = 0

        for constraint in constraints:
            # 简化验证：检查约束是否适用于当前场景
            if self._is_constraint_compliant(constraint, scene_data):
                passed += 1

        return {"total_checks": total_checks, "passed": passed}

    def _is_constraint_compliant(self, constraint: Dict, scene_data: Dict) -> bool:
        """检查约束是否符合"""
        constraint_type = constraint.get("type", "")

        # 简化的符合性检查
        if constraint_type.startswith("character_"):
            # 检查角色相关约束
            entity_id = constraint.get("entity_id", "")
            if entity_id == "global":
                return True
            else:
                # 检查特定角色是否存在
                characters = scene_data.get("characters", [])
                return any(c.get("id") == entity_id for c in characters)

        elif constraint_type.startswith("prop_"):
            # 检查道具相关约束
            entity_id = constraint.get("entity_id", "")
            if entity_id == "global":
                return True
            else:
                props = scene_data.get("props", [])
                return any(p.get("id") == entity_id for p in props)

        return True

    def _needs_consistency_check(self, scene_data: Dict) -> bool:
        """判断是否需要一致性检查"""
        # 基于场景复杂度决定
        complexity = self.scene_analyzer.analyze_scene_complexity(scene_data)
        return complexity in [SceneComplexity.COMPLEX, SceneComplexity.EPIC]

    def _check_temporal_consistency(self, scene_data: Dict) -> Dict[str, Any]:
        """检查时间一致性（修正版本）"""
        # 获取状态历史（修正参数）
        states = []

        # 计算时间范围：过去10秒内的状态
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=10)

        # 获取主要实体的状态历史
        entity_ids = list(self.state_tracker.entity_registry.keys())[:10]  # 限制实体数量

        for entity_id in entity_ids:
            try:
                # 修正：使用正确的参数 start_time, end_time
                history = self.state_tracker.get_state_history(
                    entity_id,
                    start_time=start_time,
                    end_time=end_time
                )

                # 转换为一致性检查器需要的格式
                for record in history:
                    # 提取时间戳和状态
                    state_with_time = {
                        "timestamp": record["timestamp"],
                        "state": record["state"],
                        "entity_id": entity_id
                    }
                    states.append(state_with_time)

            except Exception as e:
                warning(f"获取实体 {entity_id} 的状态历史失败: {e}")

        if len(states) < 2:
            return {"inconsistencies": [], "passed_checks": 0, "total_checks": 0}

        # 按时间戳排序
        states.sort(key=lambda x: x["timestamp"])

        # 转换为一致性检查器需要的格式
        state_sequence = []
        for state_record in states:
            # 创建标准化的状态表示
            standardized_state = self._standardize_state_for_consistency(
                state_record["state"],
                state_record["entity_id"]
            )
            standardized_state["timestamp"] = state_record["timestamp"]
            state_sequence.append(standardized_state)

        # 使用ConsistencyChecker检查时间一致性
        checker = self._get_consistency_checker()

        # 检查规则：主要关注时间进展和位置连续性
        temporal_rules = {
            "temporal_consistency": {
                "description": "时间一致性规则",
                "check_functions": [
                    self._check_time_progression_wrapper,
                    self._check_temporal_gaps_wrapper
                ],
                "thresholds": {
                    "max_time_gap": 5.0,  # 最大时间间隔
                    "max_reverse_jump": 0.5  # 最大时间回跳
                }
            },
            "spatial_consistency": {
                "description": "空间一致性规则",
                "check_functions": [
                    self._check_position_continuity_wrapper
                ],
                "thresholds": {
                    "max_teleport_distance": 3.0,  # 最大瞬移距离
                    "max_speed": 8.0  # 最大速度
                }
            }
        }

        # 设置检查器的规则
        checker.consistency_rules = temporal_rules

        # 执行一致性检查
        consistency_result = checker.check_consistency(state_sequence)

        return consistency_result

    def _standardize_state_for_consistency(self, raw_state: Dict, entity_id: str) -> Dict:
        """标准化状态以便进行一致性检查"""
        standardized = {
            "entity_id": entity_id,
            "position": None,
            "rotation": None,
            "velocity": None,
            "state": "active"
        }

        # 提取位置信息
        if "position" in raw_state:
            pos = raw_state["position"]
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                standardized["position"] = tuple(float(p) for p in pos[:3])

        # 提取旋转信息
        if "rotation" in raw_state:
            rot = raw_state["rotation"]
            if isinstance(rot, (list, tuple)) and len(rot) >= 3:
                standardized["rotation"] = tuple(float(r) for r in rot[:3])

        # 提取速度信息
        if "velocity" in raw_state:
            vel = raw_state["velocity"]
            if isinstance(vel, (list, tuple)) and len(vel) >= 3:
                standardized["velocity"] = tuple(float(v) for v in vel[:3])

        # 提取动作状态
        if "action" in raw_state:
            standardized["state"] = raw_state["action"]
        elif "state" in raw_state:
            standardized["state"] = raw_state["state"]

        # 提取实体类型
        if entity_id.startswith("char_"):
            standardized["entity_type"] = "character"
        elif entity_id.startswith("prop_"):
            standardized["entity_type"] = "prop"
        else:
            standardized["entity_type"] = "unknown"

        return standardized

    def _check_time_progression_wrapper(self, states: List[Dict], thresholds: Dict) -> Dict[str, Any]:
        """时间进展检查包装器"""
        return self._check_time_progression_custom(states, thresholds)

    def _check_temporal_gaps_wrapper(self, states: List[Dict], thresholds: Dict) -> Dict[str, Any]:
        """时间间隔检查包装器"""
        return self._check_temporal_gaps_custom(states, thresholds)

    def _check_position_continuity_wrapper(self, states: List[Dict], thresholds: Dict) -> Dict[str, Any]:
        """位置连续性检查包装器"""
        return self._check_position_continuity_custom(states, thresholds)

    def _check_time_progression_custom(self, states: List[Dict], thresholds: Dict) -> Dict[str, Any]:
        """自定义时间进展检查"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": len(states) - 1 if len(states) > 1 else 0
        }

        for i in range(1, len(states)):
            state1 = states[i - 1]
            state2 = states[i]

            time1 = state1.get("timestamp")
            time2 = state2.get("timestamp")

            if not time1 or not time2:
                results["warnings"].append({
                    "position": i,
                    "issue": "缺少时间戳",
                    "severity": "medium"
                })
                continue

            time_diff = (time2 - time1).total_seconds()

            # 检查时间是否前进
            if time_diff < 0:
                results["violations"].append({
                    "rule": "time_progression",
                    "position": i,
                    "issue": "时间倒流",
                    "details": f"时间从 {time1} 回到 {time2}",
                    "severity": "high"
                })

            # 检查时间间隔
            max_gap = thresholds.get("max_time_gap", 5.0)
            if time_diff > max_gap:
                results["inconsistencies"].append({
                    "rule": "time_progression",
                    "position": i,
                    "issue": "时间间隔过大",
                    "details": f"时间间隔 {time_diff:.1f}秒 > {max_gap}秒",
                    "severity": "medium"
                })

            if time_diff >= 0 and time_diff <= max_gap:
                results["passed_checks"].append({
                    "position": i,
                    "check": "time_progression",
                    "result": f"时间前进 {time_diff:.1f}秒"
                })

        return results

    def _check_temporal_gaps_custom(self, states: List[Dict], thresholds: Dict) -> Dict[str, Any]:
        """自定义时间间隔检查"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": len(states) - 1 if len(states) > 1 else 0
        }

        # 计算时间间隔
        time_diffs = []
        for i in range(1, len(states)):
            time1 = states[i - 1].get("timestamp")
            time2 = states[i].get("timestamp")

            if time1 and time2:
                time_diffs.append((time2 - time1).total_seconds())

        if len(time_diffs) < 2:
            return results

        # 分析间隔一致性
        avg_interval = np.mean(time_diffs)
        std_interval = np.std(time_diffs)

        for i, diff in enumerate(time_diffs):
            if std_interval > 0:
                z_score = abs(diff - avg_interval) / std_interval
                if z_score > 2.0:  # 异常间隔
                    results["inconsistencies"].append({
                        "rule": "temporal_gaps",
                        "position": i + 1,
                        "issue": "异常时间间隔",
                        "details": f"间隔 {diff:.1f}秒，平均 {avg_interval:.1f}秒",
                        "severity": "low"
                    })
                else:
                    results["passed_checks"].append({
                        "position": i + 1,
                        "check": "temporal_gaps",
                        "result": f"间隔 {diff:.1f}秒在正常范围内"
                    })

        return results

    def _check_position_continuity_custom(self, states: List[Dict], thresholds: Dict) -> Dict[str, Any]:
        """自定义位置连续性检查"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": len(states) - 1 if len(states) > 1 else 0
        }

        # 按实体分组
        entity_states = {}
        for state in states:
            entity_id = state.get("entity_id")
            if entity_id not in entity_states:
                entity_states[entity_id] = []
            entity_states[entity_id].append(state)

        # 检查每个实体的位置连续性
        for entity_id, entity_state_list in entity_states.items():
            if len(entity_state_list) < 2:
                continue

            # 按时间排序
            entity_state_list.sort(key=lambda x: x.get("timestamp"))

            for i in range(1, len(entity_state_list)):
                state1 = entity_state_list[i - 1]
                state2 = entity_state_list[i]

                pos1 = state1.get("position")
                pos2 = state2.get("position")

                if not pos1 or not pos2:
                    continue

                # 计算移动距离
                distance = self._calculate_distance(pos1, pos2)

                # 计算时间间隔
                time1 = state1.get("timestamp")
                time2 = state2.get("timestamp")
                time_diff = (time2 - time1).total_seconds() if time1 and time2 else 1.0

                # 计算速度
                speed = distance / time_diff if time_diff > 0 else 0

                # 检查瞬移
                max_teleport = thresholds.get("max_teleport_distance", 3.0)
                if distance > max_teleport:
                    results["violations"].append({
                        "rule": "position_continuity",
                        "entity": entity_id,
                        "position": i,
                        "issue": "瞬移检测",
                        "details": f"{entity_id} 移动距离 {distance:.1f}米 > {max_teleport}米",
                        "severity": "high"
                    })

                # 检查速度
                max_speed = thresholds.get("max_speed", 8.0)
                if speed > max_speed:
                    results["violations"].append({
                        "rule": "position_continuity",
                        "entity": entity_id,
                        "position": i,
                        "issue": "超速移动",
                        "details": f"{entity_id} 速度 {speed:.1f}米/秒 > {max_speed}米/秒",
                        "severity": "high"
                    })

                if distance <= max_teleport and speed <= max_speed:
                    results["passed_checks"].append({
                        "entity": entity_id,
                        "position": i,
                        "check": "position_continuity",
                        "result": f"移动 {distance:.2f}米，速度 {speed:.1f}米/秒"
                    })

        return results

    def _analyze_state_changes(self, scene_data: Dict) -> Dict[str, Any]:
        """分析状态变化（完整实现）"""
        change_detector = self._get_change_detector()

        # 获取当前状态
        current_state = self._extract_current_scene_state(scene_data)

        # 获取历史状态（最近一次的状态）
        previous_state = self._get_previous_scene_state()

        if previous_state is None:
            # 第一次处理，没有历史状态
            return {
                "significant_changes": [],
                "insignificant_changes": [],
                "change_score": 0.0,
                "baseline_established": True,
                "analysis": "首次处理，建立基线状态"
            }

        # 设置基线（如果是第一次或需要重置）
        if not change_detector.baseline_states.get("scene_baseline"):
            change_detector.set_baseline("scene_baseline", previous_state)

        # 检测当前状态与前一状态的变化
        current_changes = change_detector.detect_changes(previous_state, current_state, "scene")

        # 检测相对于基线的漂移
        baseline_drift = change_detector.detect_drift_from_baseline("scene_baseline", current_state)

        # 分析变化模式
        change_patterns = self._analyze_change_patterns(current_changes)

        # 更新基线（如果变化在合理范围内）
        if current_changes["change_score"] < 0.3:  # 变化较小
            change_detector.set_baseline("scene_baseline", current_state)

        # 构建完整结果
        result = {
            "current_changes": current_changes,
            "baseline_drift": baseline_drift,
            "change_patterns": change_patterns,
            "statistics": self._get_change_statistics(),
            "recommendations": self._generate_change_recommendations(current_changes, baseline_drift)
        }

        return result

    def _extract_current_scene_state(self, scene_data: Dict) -> Dict:
        """提取当前场景状态"""
        current_state = {
            "timestamp": datetime.now(),
            "scene_id": scene_data.get("scene_id", f"scene_{self.frame_counter}"),
            "scene_type": scene_data.get("scene_type", "unknown"),
            "characters": {},
            "props": {},
            "environment": {},
            "metadata": {}
        }

        # 提取角色状态
        for character in scene_data.get("characters", []):
            char_id = character.get("id")
            if char_id:
                char_state = {
                    "position": character.get("position"),
                    "rotation": character.get("rotation"),
                    "appearance": character.get("appearance"),
                    "outfit": character.get("outfit") or character.get("clothing"),
                    "emotion": character.get("emotion"),
                    "action": character.get("action"),
                    "velocity": character.get("velocity")
                }
                # 移除None值
                char_state = {k: v for k, v in char_state.items() if v is not None}
                current_state["characters"][char_id] = char_state

        # 提取道具状态
        for prop in scene_data.get("props", []):
            prop_id = prop.get("id")
            if prop_id:
                prop_state = {
                    "position": prop.get("position"),
                    "rotation": prop.get("rotation"),
                    "state": prop.get("state"),
                    "owner": prop.get("owner"),
                    "material": prop.get("material"),
                    "physics": prop.get("physics")
                }
                # 移除None值
                prop_state = {k: v for k, v in prop_state.items() if v is not None}
                current_state["props"][prop_id] = prop_state

        # 提取环境状态
        environment = scene_data.get("environment", {})
        current_state["environment"] = {
            "time_of_day": environment.get("time_of_day"),
            "weather": environment.get("weather"),
            "lighting": environment.get("lighting"),
            "effects": environment.get("effects", []),
            "sounds": environment.get("sounds", [])
        }

        # 元数据
        current_state["metadata"] = {
            "frame_number": self.frame_counter,
            "processing_time": datetime.now().isoformat(),
            "scene_complexity": self.scene_analyzer.analyze_scene_complexity(scene_data).value
        }

        return current_state

    def _get_previous_scene_state(self) -> Optional[Dict]:
        """获取前一场景状态"""
        # 从状态跟踪器中获取历史状态
        if not self.state_tracker.state_history:
            return None

        # 获取最近的状态记录
        latest_states = {}

        # 收集所有实体的最新状态
        for entity_id, history in self.state_tracker.state_history.items():
            if history:
                latest_record = history[-1]
                latest_states[entity_id] = {
                    "state": latest_record["state"],
                    "timestamp": latest_record["timestamp"]
                }

        if not latest_states:
            return None

        # 构建前一场景状态
        previous_state = {
            "timestamp": None,
            "characters": {},
            "props": {},
            "environment": {}
        }

        # 从实体状态重建场景状态
        for entity_id, entity_data in latest_states.items():
            state = entity_data["state"]

            # 根据实体类型分类
            if entity_id.startswith("char_"):
                previous_state["characters"][entity_id] = state
                # 更新时间戳
                if previous_state["timestamp"] is None or entity_data["timestamp"] > previous_state["timestamp"]:
                    previous_state["timestamp"] = entity_data["timestamp"]

            elif entity_id.startswith("prop_"):
                previous_state["props"][entity_id] = state

            elif entity_id == "global_environment":
                previous_state["environment"] = state

        return previous_state if previous_state["timestamp"] else None

    def _analyze_change_patterns(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """分析变化模式"""
        patterns = {
            "frequent_changes": [],
            "gradual_changes": [],
            "abrupt_changes": [],
            "cyclical_patterns": [],
            "trend_patterns": []
        }

        # 分析每个属性的变化模式
        for attribute in changes.get("changed_attributes", []):
            # 获取该属性的历史变化
            attribute_history = self._get_change_detector().change_history.get(attribute, [])

            if len(attribute_history) < 3:
                continue

            # 分析变化频率
            frequency = self._analyze_change_frequency(attribute_history)
            if frequency > 0.7:
                patterns["frequent_changes"].append({
                    "attribute": attribute,
                    "frequency": frequency,
                    "description": f"属性 '{attribute}' 变化频繁"
                })

            # 分析变化趋势
            trend = self._analyze_change_trend(attribute_history)
            if trend.get("detected"):
                patterns["trend_patterns"].append({
                    "attribute": attribute,
                    "trend": trend["direction"],
                    "strength": trend["strength"],
                    "description": f"属性 '{attribute}' 呈{trend['direction']}趋势"
                })

            # 分析周期性
            periodicity = self._analyze_change_periodicity(attribute_history)
            if periodicity.get("detected"):
                patterns["cyclical_patterns"].append({
                    "attribute": attribute,
                    "period": periodicity["period"],
                    "confidence": periodicity["confidence"],
                    "description": f"属性 '{attribute}' 有周期性变化"
                })

        return patterns

    def _analyze_change_frequency(self, history: List[Dict]) -> float:
        """分析变化频率"""
        if len(history) < 2:
            return 0.0

        # 计算平均变化间隔
        timestamps = [record["timestamp"] for record in history]
        intervals = []

        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i - 1]).total_seconds()
            intervals.append(interval)

        if not intervals:
            return 0.0

        avg_interval = np.mean(intervals)

        # 频率评分：间隔越短，频率越高
        if avg_interval < 1.0:
            return 1.0
        elif avg_interval < 5.0:
            return 0.7
        elif avg_interval < 10.0:
            return 0.4
        else:
            return 0.1

    def _analyze_change_trend(self, history: List[Dict]) -> Dict[str, Any]:
        """分析变化趋势"""
        if len(history) < 3:
            return {"detected": False}

        # 提取变化幅度
        magnitudes = [record.get("magnitude", 0) for record in history]

        # 线性回归分析趋势
        x = np.arange(len(magnitudes))
        try:
            slope, intercept = np.polyfit(x, magnitudes, 1)

            if abs(slope) > 0.01:
                return {
                    "detected": True,
                    "direction": "上升" if slope > 0 else "下降",
                    "strength": min(1.0, abs(slope) * 10),
                    "slope": slope,
                    "intercept": intercept
                }
        except:
            pass

        return {"detected": False}

    def _analyze_change_periodicity(self, history: List[Dict]) -> Dict[str, Any]:
        """分析变化周期性"""
        if len(history) < 5:
            return {"detected": False}

        # 提取时间序列
        timestamps = [record["timestamp"] for record in history]
        magnitudes = [record.get("magnitude", 0) for record in history]

        # 计算时间间隔
        time_diffs = []
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i - 1]).total_seconds()
            time_diffs.append(diff)

        # 检查间隔的规律性
        if len(time_diffs) >= 3:
            mean_diff = np.mean(time_diffs)
            std_diff = np.std(time_diffs)

            if mean_diff > 0 and std_diff / mean_diff < 0.3:  # 变异系数小，说明规律
                return {
                    "detected": True,
                    "period": mean_diff,
                    "confidence": 1.0 - (std_diff / mean_diff),
                    "mean_interval": mean_diff,
                    "std_interval": std_diff
                }

        return {"detected": False}

    def _get_change_statistics(self) -> Dict[str, Any]:
        """获取变化统计"""
        change_detector = self._get_change_detector()

        stats = {
            "total_changes_recorded": 0,
            "change_rate": 0.0,
            "most_changing_attributes": [],
            "recent_change_intensity": 0.0
        }

        # 计算总变化次数
        total_changes = 0
        for attribute, history in change_detector.change_history.items():
            total_changes += len(history)

        stats["total_changes_recorded"] = total_changes

        # 计算变化率（每帧的平均变化数）
        if self.frame_counter > 0:
            stats["change_rate"] = total_changes / self.frame_counter

        # 找到变化最频繁的属性
        attribute_counts = {}
        for attribute, history in change_detector.change_history.items():
            attribute_counts[attribute] = len(history)

        # 取前5个
        sorted_attributes = sorted(attribute_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        stats["most_changing_attributes"] = [
            {"attribute": attr, "change_count": count}
            for attr, count in sorted_attributes
        ]

        # 计算最近变化强度
        recent_changes = []
        for attribute, history in change_detector.change_history.items():
            if history:
                recent = history[-1]
                recent_changes.append(recent.get("magnitude", 0))

        if recent_changes:
            stats["recent_change_intensity"] = np.mean(recent_changes)

        return stats

    def _generate_change_recommendations(self, current_changes: Dict,
                                         baseline_drift: Dict) -> List[str]:
        """生成变化相关推荐"""
        recommendations = []

        # 基于当前变化的推荐
        change_score = current_changes.get("change_score", 0)
        if change_score > 0.7:
            recommendations.append("变化强度较高，检查是否合理")

        significant_changes = current_changes.get("significant_changes", [])
        if len(significant_changes) > 3:
            recommendations.append(f"显著变化过多 ({len(significant_changes)}个)，可能导致不连续")

        # 基于基线漂移的推荐
        drift_score = baseline_drift.get("drift_score", 0)
        if drift_score > 0.5:
            recommendations.append(f"相对于基线漂移较大 ({drift_score:.2f})，考虑重置基线")

        # 基于变化模式的推荐
        for change in significant_changes:
            attribute = change.get("attribute", "")
            if "position" in attribute and change.get("magnitude", 0) > 5.0:
                recommendations.append(f"位置变化过大: {attribute}")
            elif "appearance" in attribute:
                recommendations.append(f"外观变化: {attribute}，确保观众能识别")

        return recommendations[:5]  # 返回前5个推荐

    def _calculate_distance(self, pos1, pos2) -> float:
        """计算两点间距离"""
        if isinstance(pos1, (list, tuple)) and isinstance(pos2, (list, tuple)):
            if len(pos1) >= 3 and len(pos2) >= 3:
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                dz = pos2[2] - pos1[2]
                return math.sqrt(dx * dx + dy * dy + dz * dz)
        return 0.0

    def _extract_scene_state_summary(self, scene_data: Dict) -> Dict:
        """提取场景状态摘要"""
        summary = {
            "characters": {},
            "props": {},
            "environment": scene_data.get("environment", {})
        }

        for character in scene_data.get("characters", []):
            char_id = character.get("id")
            summary["characters"][char_id] = {
                "position": character.get("position"),
                "appearance": character.get("appearance"),
                "action": character.get("action")
            }

        for prop in scene_data.get("props", []):
            prop_id = prop.get("id")
            summary["props"][prop_id] = {
                "position": prop.get("position"),
                "state": prop.get("state")
            }

        return summary

    def _needs_collision_detection(self, scene_data: Dict) -> bool:
        """判断是否需要碰撞检测"""
        # 如果场景中有多个移动物体，可能需要碰撞检测
        moving_objects = 0

        for character in scene_data.get("characters", []):
            if character.get("velocity") or character.get("action") in ["running", "jumping"]:
                moving_objects += 1

        for prop in scene_data.get("props", []):
            if prop.get("state") == "moving":
                moving_objects += 1

        return moving_objects >= 2

    def _analyze_collisions(self, scene_data: Dict) -> Optional[Dict[str, Any]]:
        """分析碰撞"""
        try:
            detector = self._get_collision_detector()

            # 准备实体数据
            entities = []

            # 添加角色
            for character in scene_data.get("characters", []):
                entities.append({
                    "id": character.get("id"),
                    "position": character.get("position", [0, 0, 0]),
                    "bounding_volume": {
                        "type": BoundingVolumeType.CAPSULE,
                        "radius": 0.3,
                        "height": 1.7
                    }
                })

            # 添加道具
            for prop in scene_data.get("props", []):
                entities.append({
                    "id": prop.get("id"),
                    "position": prop.get("position", [0, 0, 0]),
                    "bounding_volume": {
                        "type": BoundingVolumeType.BOX,
                        "size": prop.get("size", [0.5, 0.5, 0.5])
                    }
                })

            # 检测碰撞
            collisions = detector.detect_collisions(entities)

            return {
                "total_collisions": len([c for c in collisions
                                         if c["type"] != CollisionType.NO_COLLISION]),
                "collision_details": collisions[:5] if collisions else []
            }
        except Exception as e:
            error(f"碰撞检测失败: {e}")
            return None

    def _needs_motion_analysis(self, scene_data: Dict) -> bool:
        """判断是否需要运动分析"""
        # 检查是否有复杂的运动轨迹
        for character in scene_data.get("characters", []):
            if "trajectory" in character:
                return True

        return False

    def _analyze_motions(self, scene_data: Dict) -> Optional[Dict[str, Any]]:
        """分析运动"""
        try:
            analyzer = self._get_motion_analyzer()

            # 提取运动轨迹
            motions = []

            for character in scene_data.get("characters", []):
                if "trajectory" in character:
                    trajectory = character["trajectory"]
                    if isinstance(trajectory, list) and len(trajectory) >= 2:
                        analysis = analyzer.analyze_motion(trajectory)
                        motions.append({
                            "character_id": character.get("id"),
                            "analysis": analysis
                        })

            return {
                "motions_analyzed": len(motions),
                "motion_details": motions
            }
        except Exception as e:
            error(f"运动分析失败: {e}")
            return None

    def _generate_integrated_hints(self, scene_data: Dict) -> GenerationHints:
        """生成集成提示"""
        hints = GenerationHints()

        # 1. 基于约束的提示
        constraints = self._generate_constraints_integrated(scene_data, {})
        for constraint in constraints[:5]:  # 前5个最重要的约束
            hints.continuity_constraints.append(
                f"{constraint.get('type', '约束')}: {constraint.get('constraint', '')}"
            )

        # 2. 基于物理验证的提示
        physics_result = self._validate_physics_integrated(scene_data)
        if physics_result["issues"] > 0:
            hints.continuity_constraints.append(
                f"注意物理合理性: {physics_result['issues']}个问题"
            )

        # 3. 基于学习器的预测
        if self.continuity_learner:
            predicted = self.continuity_learner.predict_issues(scene_data)
            for prediction in predicted[:3]:
                hints.continuity_constraints.append(
                    f"预测问题: {prediction.get('type', '未知')}"
                )

        # 4. 基于历史问题的提示
        hints.avoid_elements = self._get_avoid_elements_from_history()

        return hints

    def _get_avoid_elements_from_history(self) -> List[str]:
        """从历史问题中获取应避免的元素"""
        avoid_list = []

        # 这里需要实际的实现来从历史问题中提取
        # 简化实现
        if self.frame_counter > 10:
            avoid_list.append("突然的角色外貌变化")
            avoid_list.append("不合理的物理行为")

        return avoid_list

    def _learn_from_results(self, report: ValidationReport):
        """从验证结果中学习"""
        if not self.continuity_learner:
            return

        for issue in report.issues:
            # 创建简化的修复结果（实际应该基于实际修复）
            fix_result = {
                "applied": issue.auto_fixable,
                "effectiveness": 0.7 if issue.auto_fixable else 0.0,
                "strategy": "auto_fix" if issue.auto_fixable else "manual_review"
            }

            self.continuity_learner.learn_from_issue(issue, fix_result)

    def _generate_integrated_recommendations(self, scene_analysis: Any,
                                             physics_validation: Dict,
                                             continuity_report: Optional[ValidationReport]) -> List[str]:
        """生成集成推荐"""
        recommendations = []

        # 基于场景分析
        if hasattr(scene_analysis, 'value'):  # 如果是SceneComplexity枚举
            if scene_analysis in [SceneComplexity.COMPLEX, SceneComplexity.EPIC]:
                recommendations.append("复杂场景，建议分阶段处理")

        # 基于物理验证
        if physics_validation["issues"] > 5:
            recommendations.append(f"物理问题较多 ({physics_validation['issues']}个)，建议检查物理参数")

        if physics_validation["plausibility_score"] < 0.7:
            recommendations.append(f"物理合理性较低 ({physics_validation['plausibility_score']:.2f})，建议优化")

        # 基于连续性报告
        if continuity_report:
            critical_issues = continuity_report.summary.get("critical_issues", 0)
            if critical_issues > 0:
                recommendations.append(f"发现 {critical_issues} 个关键连续性问题，需要立即修复")

        # 基于性能
        if len(self.performance_metrics["processing_times"]) > 10:
            avg_time = np.mean(self.performance_metrics["processing_times"][-5:])
            if avg_time > 0.5:
                recommendations.append(f"处理时间较长 ({avg_time:.2f}秒)，建议优化")

        return recommendations[:5]

    def _should_validate_continuity(self) -> bool:
        """判断是否需要验证连续性"""
        # 基于配置和性能
        if not self.config.enable_real_time_validation:
            return False

        if self.frame_counter % self.config.validation_frequency == 0:
            return True

        # 如果最近处理时间短，可以增加验证频率
        if self.performance_metrics["processing_times"]:
            recent_times = self.performance_metrics["processing_times"][-3:]
            if all(t < 0.2 for t in recent_times):
                return True

        return False

    def _record_performance(self, metric_name: str, value: float):
        """记录性能指标"""
        if metric_name in self.performance_metrics:
            self.performance_metrics[metric_name].append(value)

            # 保持历史长度
            if len(self.performance_metrics[metric_name]) > 100:
                self.performance_metrics[metric_name].pop(0)

    # 公共接口方法
    def get_continuity_report(self, detailed: bool = False) -> str:
        """获取连续性报告"""
        # 获取最近的验证报告
        # 这里需要实际的实现
        return "连续性报告功能"

    def analyze_transition(self, from_scene: Dict, to_scene: Dict) -> TransitionInstruction:
        """分析场景转场"""
        return self.transition_manager.analyze_transition(from_scene, to_scene)

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "task_id": self.task_id,
            "frames_processed": self.frame_counter,
            "state_tracking": {
                "entities_tracked": len(self.state_tracker.entity_registry),
                "total_states": sum(len(h) for h in self.state_tracker.state_history.values())
            },
            "performance": {
                "avg_processing_time": np.mean(self.performance_metrics["processing_times"])
                if self.performance_metrics["processing_times"] else 0,
                "recent_validation_time": self.performance_metrics["validation_times"][-1]
                if self.performance_metrics["validation_times"] else 0
            }
        }

    def export_data(self, filepath: str) -> bool:
        """导出数据"""
        # 实现数据导出逻辑
        return True

    def import_data(self, filepath: str) -> bool:
        """导入数据"""
        # 实现数据导入逻辑
        return True
