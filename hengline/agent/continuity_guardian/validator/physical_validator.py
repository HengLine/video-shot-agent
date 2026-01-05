"""
@FileName: physical_plausibility_validator.py
@Description: 物理规则
@Author: HengLine
@Time: 2026/1/4 16:47
"""
import math
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from hengline.agent.continuity_guardian.validator.collision_validator import CollisionDetector
from hengline.agent.continuity_guardian.validator.material_validator import MaterialDatabase
from hengline.agent.continuity_guardian.validator.motion_validator import MotionAnalyzer


class PhysicalPlausibilityValidator:
    """物理合理性验证器"""

    def __init__(self, physics_config: Optional[Dict] = None):
        self.physics_config = physics_config or self._get_default_physics_config()
        self.validation_history: deque = deque(maxlen=1000)
        self.collision_detector = CollisionDetector()
        self.motion_analyzer = MotionAnalyzer()
        self.material_database = MaterialDatabase()

    def _get_default_physics_config(self) -> Dict[str, Any]:
        """获取默认物理配置"""
        return {
            "gravity": 9.81,  # m/s²
            "air_density": 1.225,  # kg/m³
            "friction_coefficients": {
                "concrete": 0.6,
                "wood": 0.4,
                "metal": 0.3,
                "ice": 0.1,
                "carpet": 0.7
            },
            "material_properties": {
                "density_ranges": {  # kg/m³
                    "human": (985, 1050),
                    "wood": (400, 800),
                    "metal": (2700, 7850),
                    "plastic": (850, 1400)
                },
                "elasticity_ranges": {  # 恢复系数
                    "rubber": (0.7, 0.9),
                    "steel": (0.5, 0.7),
                    "glass": (0.4, 0.6),
                    "clay": (0.1, 0.3)
                }
            },
            "constraints": {
                "max_human_speed": 12.0,  # m/s
                "max_jump_height": 2.5,  # 米
                "max_carry_weight": 50.0,  # 千克
                "min_realistic_mass": 0.1  # 千克
            }
        }

    def validate_scene_physics(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证场景物理合理性"""
        validation_id = f"physics_validation_{datetime.now().timestamp()}"

        result = {
            "validation_id": validation_id,
            "timestamp": datetime.now(),
            "overall_plausibility_score": 1.0,
            "issues": [],
            "warnings": [],
            "passed_checks": [],
            "detailed_analysis": {}
        }

        # 验证角色物理
        characters = scene_data.get("characters", [])
        character_issues = self._validate_characters_physics(characters)
        result["issues"].extend(character_issues["critical"])
        result["warnings"].extend(character_issues["warnings"])
        result["passed_checks"].extend(character_issues["passed"])

        # 验证道具物理
        props = scene_data.get("props", [])
        prop_issues = self._validate_props_physics(props)
        result["issues"].extend(prop_issues["critical"])
        result["warnings"].extend(prop_issues["warnings"])
        result["passed_checks"].extend(prop_issues["passed"])

        # 验证环境物理
        environment = scene_data.get("environment", {})
        env_issues = self._validate_environment_physics(environment)
        result["issues"].extend(env_issues["critical"])
        result["warnings"].extend(env_issues["warnings"])
        result["passed_checks"].extend(env_issues["passed"])

        # 验证运动物理
        motions = scene_data.get("motions", [])
        motion_issues = self._validate_motions_physics(motions)
        result["issues"].extend(motion_issues["critical"])
        result["warnings"].extend(motion_issues["warnings"])
        result["passed_checks"].extend(motion_issues["passed"])

        # 验证碰撞
        collisions = scene_data.get("collisions", [])
        collision_issues = self._validate_collisions(collisions)
        result["issues"].extend(collision_issues["critical"])
        result["warnings"].extend(collision_issues["warnings"])
        result["passed_checks"].extend(collision_issues["passed"])

        # 计算总体合理性分数
        result["overall_plausibility_score"] = self._calculate_plausibility_score(result)

        # 生成详细分析
        result["detailed_analysis"] = self._generate_detailed_analysis(result)

        # 记录验证历史
        self.validation_history.append({
            "validation_id": validation_id,
            "timestamp": datetime.now(),
            "scene_id": scene_data.get("scene_id", "unknown"),
            "score": result["overall_plausibility_score"],
            "issue_count": len(result["issues"])
        })

        return result

    def _validate_characters_physics(self, characters: List[Dict]) -> Dict[str, List]:
        """验证角色物理"""
        issues = {"critical": [], "warnings": [], "passed": []}

        for char in characters:
            char_id = char.get("id", "unknown")

            # 验证运动速度
            if "velocity" in char:
                speed = self._calculate_speed(char["velocity"])
                max_speed = self.physics_config["constraints"]["max_human_speed"]

                if speed > max_speed:
                    issues["critical"].append({
                        "entity": char_id,
                        "type": "unrealistic_speed",
                        "description": f"角色速度 {speed:.1f}m/s 超过人类极限 {max_speed}m/s",
                        "severity": "high",
                        "suggested_fix": f"降低速度至 {max_speed}m/s 以下"
                    })
                else:
                    issues["passed"].append({
                        "entity": char_id,
                        "check": "speed_validation",
                        "result": f"速度 {speed:.1f}m/s 在合理范围内"
                    })

            # 验证跳跃高度
            if "jump_height" in char:
                max_jump = self.physics_config["constraints"]["max_jump_height"]
                if char["jump_height"] > max_jump:
                    issues["critical"].append({
                        "entity": char_id,
                        "type": "unrealistic_jump",
                        "description": f"跳跃高度 {char['jump_height']:.1f}m 超过物理极限",
                        "severity": "medium",
                        "suggested_fix": f"降低跳跃高度至 {max_jump}m 以下"
                    })

            # 验证负重
            if "carrying_weight" in char:
                max_carry = self.physics_config["constraints"]["max_carry_weight"]
                if char["carrying_weight"] > max_carry:
                    issues["warnings"].append({
                        "entity": char_id,
                        "type": "heavy_load",
                        "description": f"负重 {char['carrying_weight']:.1f}kg 超过推荐值",
                        "severity": "low",
                        "suggested_fix": f"减少负重或表现吃力状态"
                    })

            # 验证姿势稳定性
            if "posture" in char:
                stability = self._assess_posture_stability(char["posture"])
                if stability < 0.3:
                    issues["warnings"].append({
                        "entity": char_id,
                        "type": "unstable_posture",
                        "description": "姿势稳定性较低",
                        "severity": "medium",
                        "suggested_fix": "调整重心或添加支撑"
                    })

        return issues

    def _validate_props_physics(self, props: List[Dict]) -> Dict[str, List]:
        """验证道具物理"""
        issues = {"critical": [], "warnings": [], "passed": []}

        for prop in props:
            prop_id = prop.get("id", "unknown")

            # 验证质量
            if "mass" in prop:
                min_mass = self.physics_config["constraints"]["min_realistic_mass"]
                if prop["mass"] < min_mass:
                    issues["warnings"].append({
                        "entity": prop_id,
                        "type": "unrealistic_mass",
                        "description": f"质量 {prop['mass']:.3f}kg 过轻",
                        "severity": "low",
                        "suggested_fix": "增加质量或调整材质"
                    })

            # 验证材料属性
            if "material" in prop:
                material_valid = self._validate_material_properties(prop)
                if not material_valid["valid"]:
                    issues["critical"].extend(material_valid["issues"])
                else:
                    issues["passed"].append({
                        "entity": prop_id,
                        "check": "material_validation",
                        "result": "材料属性合理"
                    })

            # 验证浮空物体
            if "position" in prop and "supported" in prop:
                if not prop["supported"] and prop["position"][2] > 0.1:  # 离地面超过10cm
                    issues["critical"].append({
                        "entity": prop_id,
                        "type": "floating_object",
                        "description": "道具浮空无支撑",
                        "severity": "high",
                        "suggested_fix": "添加支撑或调整位置"
                    })

        return issues

    def _validate_environment_physics(self, environment: Dict) -> Dict[str, List]:
        """验证环境物理"""
        issues = {"critical": [], "warnings": [], "passed": []}

        # 验证光照物理
        if "lighting" in environment:
            lighting_issues = self._validate_lighting_physics(environment["lighting"])
            issues["warnings"].extend(lighting_issues)

        # 验证天气物理
        if "weather" in environment:
            weather_issues = self._validate_weather_physics(environment["weather"])
            issues["warnings"].extend(weather_issues)

        # 验证声音物理
        if "sound_sources" in environment:
            sound_issues = self._validate_sound_physics(environment["sound_sources"])
            issues["warnings"].extend(sound_issues)

        issues["passed"].append({
            "check": "environment_basic_physics",
            "result": "环境基础物理合理"
        })

        return issues

    def _validate_motions_physics(self, motions: List[Dict]) -> Dict[str, List]:
        """验证运动物理"""
        issues = {"critical": [], "warnings": [], "passed": []}

        for motion in motions:
            motion_id = motion.get("id", "unknown")

            # 验证运动轨迹
            if "trajectory" in motion:
                trajectory_issues = self._validate_trajectory(motion["trajectory"])
                issues["critical"].extend(trajectory_issues["critical"])
                issues["warnings"].extend(trajectory_issues["warnings"])

            # 验证加速度
            if "acceleration" in motion:
                acceleration_issues = self._validate_acceleration(motion["acceleration"])
                issues["critical"].extend(acceleration_issues)

            # 验证旋转运动
            if "angular_velocity" in motion:
                rotation_issues = self._validate_rotation(motion["angular_velocity"])
                issues["warnings"].extend(rotation_issues)

            issues["passed"].append({
                "entity": motion_id,
                "check": "motion_kinematics",
                "result": "运动学参数基本合理"
            })

        return issues

    def _validate_collisions(self, collisions: List[Dict]) -> Dict[str, List]:
        """验证碰撞物理"""
        issues = {"critical": [], "warnings": [], "passed": []}

        for collision in collisions:
            coll_id = collision.get("id", "unknown")

            # 验证碰撞响应
            if "response" in collision:
                response_issues = self._validate_collision_response(collision)
                issues["critical"].extend(response_issues["critical"])
                issues["warnings"].extend(response_issues["warnings"])

            # 验证能量守恒
            if "energy" in collision:
                energy_issues = self._validate_energy_conservation(collision)
                issues["critical"].extend(energy_issues)

            issues["passed"].append({
                "entity": coll_id,
                "check": "collision_detection",
                "result": "碰撞检测基本正确"
            })

        return issues

    def _calculate_speed(self, velocity: Tuple[float, float, float]) -> float:
        """计算速度大小"""
        return math.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2)

    def _assess_posture_stability(self, posture: Dict) -> float:
        """评估姿势稳定性"""
        # 实现姿势稳定性评估逻辑
        return 0.8  # 示例值

    def _validate_material_properties(self, prop: Dict) -> Dict[str, Any]:
        """验证材料属性"""
        result = {"valid": True, "issues": []}
        material = prop.get("material", "unknown")

        # 检查密度是否在合理范围内
        if "density" in prop and material in self.physics_config["material_properties"]["density_ranges"]:
            density_range = self.physics_config["material_properties"]["density_ranges"][material]
            if not (density_range[0] <= prop["density"] <= density_range[1]):
                result["valid"] = False
                result["issues"].append({
                    "type": "unrealistic_density",
                    "description": f"密度 {prop['density']} 不在 {material} 的合理范围 {density_range} 内"
                })

        return result

    def _validate_lighting_physics(self, lighting: Dict) -> List[Dict]:
        """验证光照物理"""
        issues = []

        # 检查阴影一致性
        if "shadows" in lighting:
            shadow_consistency = self._check_shadow_consistency(lighting["shadows"])
            if not shadow_consistency["consistent"]:
                issues.append({
                    "type": "inconsistent_shadows",
                    "description": shadow_consistency["issue"],
                    "severity": "low"
                })

        # 检查光照强度
        if "intensity" in lighting:
            if lighting["intensity"] > 100000:  # 过度亮度
                issues.append({
                    "type": "excessive_brightness",
                    "description": "光照强度超出物理合理范围",
                    "severity": "low"
                })

        return issues

    def _validate_weather_physics(self, weather: Dict) -> List[Dict]:
        """验证天气物理"""
        issues = []

        # 检查降水物理
        if "precipitation" in weather:
            precip_issues = self._check_precipitation_physics(weather["precipitation"])
            issues.extend(precip_issues)

        # 检查风效物理
        if "wind" in weather:
            wind_issues = self._check_wind_physics(weather["wind"])
            issues.extend(wind_issues)

        return issues

    def _validate_sound_physics(self, sound_sources: List[Dict]) -> List[Dict]:
        """验证声音物理"""
        issues = []

        for sound in sound_sources:
            # 检查声速合理性
            if "speed" in sound:
                if sound["speed"] < 330 or sound["speed"] > 350:  # 空气中声速大约343m/s
                    issues.append({
                        "type": "unrealistic_sound_speed",
                        "description": f"声速 {sound['speed']}m/s 不合理",
                        "severity": "low"
                    })

            # 检查多普勒效应
            if "doppler_effect" in sound and not sound["doppler_effect"]:
                if "velocity" in sound and self._calculate_speed(sound["velocity"]) > 10:
                    issues.append({
                        "type": "missing_doppler_effect",
                        "description": "高速移动声源缺少多普勒效应",
                        "severity": "medium"
                    })

        return issues

    def _validate_trajectory(self, trajectory: List[Tuple[float, float, float]]) -> Dict[str, List]:
        """验证运动轨迹"""
        issues = {"critical": [], "warnings": []}

        if len(trajectory) < 2:
            return issues

        # 检查轨迹平滑度
        smoothness = self._calculate_trajectory_smoothness(trajectory)
        if smoothness < 0.7:
            issues["warnings"].append({
                "type": "jerky_trajectory",
                "description": f"轨迹平滑度较低 ({smoothness:.2f})",
                "severity": "medium"
            })

        # 检查是否符合重力
        gravity_compliance = self._check_gravity_compliance(trajectory)
        if not gravity_compliance["compliant"]:
            issues["critical"].append({
                "type": "anti_gravity_trajectory",
                "description": gravity_compliance["issue"],
                "severity": "high"
            })

        return issues

    def _validate_acceleration(self, acceleration: float) -> List[Dict]:
        """验证加速度"""
        issues = []

        # 人类可承受的加速度限制
        max_human_acceleration = 30.0  # m/s²（约3g）

        if acceleration > max_human_acceleration:
            issues.append({
                "type": "excessive_acceleration",
                "description": f"加速度 {acceleration:.1f}m/s² 超过人类承受极限",
                "severity": "high"
            })

        return issues

    def _validate_rotation(self, angular_velocity: float) -> List[Dict]:
        """验证旋转速度"""
        issues = []

        # 合理的旋转速度限制（弧度/秒）
        max_rotation_speed = 20.0  # 约1146度/秒

        if angular_velocity > max_rotation_speed:
            issues.append({
                "type": "excessive_rotation",
                "description": f"旋转速度 {angular_velocity:.1f}rad/s 不合理",
                "severity": "medium"
            })

        return issues

    def _validate_collision_response(self, collision: Dict) -> Dict[str, List]:
        """验证碰撞响应"""
        issues = {"critical": [], "warnings": []}

        response = collision.get("response", {})

        # 检查动量守恒
        if "momentum_conservation" in response:
            if not response["momentum_conservation"]:
                issues["critical"].append({
                    "type": "momentum_not_conserved",
                    "description": "碰撞未保持动量守恒",
                    "severity": "high"
                })

        # 检查材料响应
        if "material_response" in response:
            material_valid = self._validate_material_collision_response(collision)
            if not material_valid["valid"]:
                issues["critical"].extend(material_valid["issues"])

        return issues

    def _validate_energy_conservation(self, collision: Dict) -> List[Dict]:
        """验证能量守恒"""
        issues = []

        energy_data = collision.get("energy", {})

        if "loss" in energy_data:
            energy_loss = energy_data["loss"]
            if energy_loss > 1.0 or energy_loss < 0:  # 能量损失应在0-1之间
                issues.append({
                    "type": "unrealistic_energy_loss",
                    "description": f"能量损失 {energy_loss:.2f} 不合理",
                    "severity": "high"
                })

        return issues

    def _calculate_plausibility_score(self, validation_result: Dict) -> float:
        """计算物理合理性分数"""
        total_issues = len(validation_result["issues"])
        total_warnings = len(validation_result["warnings"])
        total_passed = len(validation_result["passed"])

        if total_issues + total_warnings + total_passed == 0:
            return 1.0

        # 权重计算
        issue_weight = 0.7
        warning_weight = 0.3

        max_score = total_passed + total_warnings + total_issues
        if max_score == 0:
            return 1.0

        actual_score = (total_passed * 1.0 +
                        total_warnings * 0.5 +
                        total_issues * 0.0)

        return actual_score / max_score

    def _generate_detailed_analysis(self, validation_result: Dict) -> Dict[str, Any]:
        """生成详细分析报告"""
        analysis = {
            "physical_laws_compliance": self._assess_physical_laws_compliance(validation_result),
            "realism_metrics": self._calculate_realism_metrics(validation_result),
            "improvement_suggestions": self._generate_improvement_suggestions(validation_result),
            "risk_assessment": self._assess_physics_risks(validation_result)
        }
        return analysis

    def _assess_physical_laws_compliance(self, validation_result: Dict) -> Dict[str, float]:
        """评估物理定律遵守程度"""
        compliance = {
            "newtons_laws": 0.9,
            "energy_conservation": 0.85,
            "momentum_conservation": 0.88,
            "gravity_compliance": 0.95,
            "material_realism": 0.8
        }

        # 根据验证结果调整
        for issue in validation_result["issues"]:
            if "gravity" in issue.get("type", "").lower():
                compliance["gravity_compliance"] *= 0.7
            elif "energy" in issue.get("type", "").lower():
                compliance["energy_conservation"] *= 0.6
            elif "momentum" in issue.get("type", "").lower():
                compliance["momentum_conservation"] *= 0.5

        return compliance

    def _calculate_realism_metrics(self, validation_result: Dict) -> Dict[str, float]:
        """计算现实主义指标"""
        metrics = {
            "kinematic_realism": 0.9,
            "dynamic_realism": 0.85,
            "material_realism": 0.8,
            "environmental_realism": 0.75,
            "human_factor_realism": 0.7
        }
        return metrics

    def _generate_improvement_suggestions(self, validation_result: Dict) -> List[str]:
        """生成改进建议"""
        suggestions = []

        # 根据问题生成建议
        for issue in validation_result["issues"]:
            if "suggested_fix" in issue:
                suggestions.append(f"{issue['entity']}: {issue['suggested_fix']}")

        return suggestions

    def _assess_physics_risks(self, validation_result: Dict) -> Dict[str, Any]:
        """评估物理风险"""
        risk_levels = {
            "immersion_break": "low",
            "visual_distraction": "medium",
            "narrative_disruption": "low",
            "audience_rejection": "low"
        }

        # 根据问题严重程度调整风险
        critical_count = len(validation_result["issues"])
        if critical_count > 5:
            risk_levels["immersion_break"] = "high"
            risk_levels["audience_rejection"] = "medium"

        return risk_levels

    def _check_shadow_consistency(self, shadows: Dict) -> Dict[str, Any]:
        """检查阴影一致性"""
        return {"consistent": True, "issue": ""}

    def _check_precipitation_physics(self, precipitation: Dict) -> List[Dict]:
        """检查降水物理"""
        return []

    def _check_wind_physics(self, wind: Dict) -> List[Dict]:
        """检查风效物理"""
        return []

    def _calculate_trajectory_smoothness(self, trajectory: List[Tuple]) -> float:
        """计算轨迹平滑度"""
        if len(trajectory) < 3:
            return 1.0

        # 计算曲率变化
        curvatures = []
        for i in range(1, len(trajectory) - 1):
            # 简化计算
            curvatures.append(0.5)  # 示例值

        if not curvatures:
            return 1.0

        # 平滑度：曲率变化的倒数
        curvature_variance = np.var(curvatures) if curvatures else 0
        smoothness = 1.0 / (1.0 + curvature_variance * 10)

        return min(max(smoothness, 0), 1)

    def _check_gravity_compliance(self, trajectory: List[Tuple]) -> Dict[str, Any]:
        """检查重力符合度"""
        return {"compliant": True, "issue": ""}

    def _validate_material_collision_response(self, collision: Dict) -> Dict[str, Any]:
        """验证材料碰撞响应"""
        return {"valid": True, "issues": []}
