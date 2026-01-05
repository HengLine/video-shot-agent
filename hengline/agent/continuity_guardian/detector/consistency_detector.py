"""
@FileName: consistency_checker.py
@Description: 一致性检查器
@Author: HengLine
@Time: 2026/1/4 22:29
"""
import math
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime
import numpy as np


class ConsistencyChecker:
    """一致性检查器"""

    def __init__(self, consistency_rules: Optional[Dict] = None):
        self.consistency_rules = consistency_rules or self._get_default_rules()
        self.consistency_history: Dict[str, List[Dict]] = defaultdict(list)
        self.inconsistency_patterns: Dict[str, Dict] = {}
        self.rule_violations: Dict[str, List] = defaultdict(list)

    def _get_default_rules(self) -> Dict[str, Dict]:
        """获取默认一致性规则"""
        return {
            "temporal_consistency": {
                "description": "时间一致性规则",
                "check_functions": [
                    self._check_time_progression,
                    self._check_temporal_gaps,
                    self._check_event_ordering
                ],
                "thresholds": {
                    "max_time_gap": 10.0,  # 最大时间间隔（秒）
                    "max_reverse_jump": 1.0  # 最大时间回跳（秒）
                }
            },
            "spatial_consistency": {
                "description": "空间一致性规则",
                "check_functions": [
                    self._check_position_continuity,
                    self._check_collision_consistency,
                    self._check_boundary_violations
                ],
                "thresholds": {
                    "max_teleport_distance": 5.0,  # 最大瞬移距离（米）
                    "max_speed": 10.0  # 最大速度（米/秒）
                }
            },
            "logical_consistency": {
                "description": "逻辑一致性规则",
                "check_functions": [
                    self._check_state_transitions,
                    self._check_property_consistency,
                    self._check_causality
                ],
                "thresholds": {
                    "invalid_transitions": 0,
                    "property_conflicts": 0
                }
            },
            "physical_consistency": {
                "description": "物理一致性规则",
                "check_functions": [
                    self._check_physics_laws,
                    self._check_material_consistency,
                    self._check_energy_conservation
                ],
                "thresholds": {
                    "physics_violations": 0
                }
            }
        }

    def check_consistency(self, states: List[Dict]) -> Dict[str, Any]:
        """检查状态序列的一致性"""
        if len(states) < 2:
            return {"error": "状态数量不足"}

        consistency_report = {
            "overall_consistency_score": 1.0,
            "rule_violations": {},
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "detailed_scores": {},
            "recommendations": []
        }

        # 应用所有一致性规则
        total_score = 0.0
        rule_count = 0

        for rule_name, rule_config in self.consistency_rules.items():
            rule_score, rule_results = self._apply_consistency_rule(
                rule_name, rule_config, states
            )

            consistency_report["detailed_scores"][rule_name] = rule_score
            consistency_report["rule_violations"][rule_name] = rule_results["violations"]

            total_score += rule_score
            rule_count += 1

            # 记录不一致性
            consistency_report["inconsistencies"].extend(rule_results["inconsistencies"])
            consistency_report["warnings"].extend(rule_results["warnings"])
            consistency_report["passed_checks"].extend(rule_results["passed_checks"])

        # 计算总体一致性分数
        if rule_count > 0:
            consistency_report["overall_consistency_score"] = total_score / rule_count

        # 生成建议
        consistency_report["recommendations"] = self._generate_consistency_recommendations(
            consistency_report
        )

        # 记录检查历史
        self._record_consistency_check(consistency_report)

        return consistency_report

    def _apply_consistency_rule(self, rule_name: str, rule_config: Dict,
                                states: List[Dict]) -> Tuple[float, Dict[str, Any]]:
        """应用一致性规则"""
        rule_results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": []
        }

        total_checks = 0
        passed_checks = 0

        # 执行所有检查函数
        for check_func in rule_config["check_functions"]:
            check_result = check_func(states, rule_config.get("thresholds", {}))

            total_checks += check_result.get("total_checks", 1)
            passed_checks += check_result.get("passed_checks", 0)

            # 记录结果
            rule_results["violations"].extend(check_result.get("violations", []))
            rule_results["inconsistencies"].extend(check_result.get("inconsistencies", []))
            rule_results["warnings"].extend(check_result.get("warnings", []))
            rule_results["passed_checks"].extend(check_result.get("passed_checks", []))

            # 记录规则违反
            for violation in check_result.get("violations", []):
                violation_key = f"{rule_name}_{check_func.__name__}"
                self.rule_violations[violation_key].append({
                    "timestamp": datetime.now(),
                    "violation": violation
                })

        # 计算规则分数
        if total_checks > 0:
            rule_score = passed_checks / total_checks
        else:
            rule_score = 1.0

        return rule_score, rule_results

    def _check_time_progression(self, states: List[Dict],
                                thresholds: Dict) -> Dict[str, Any]:
        """检查时间进展"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": len(states) - 1
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
                violation = {
                    "rule": "time_progression",
                    "position": i,
                    "issue": "时间倒流",
                    "details": f"时间从 {time1} 回到 {time2}",
                    "severity": "high"
                }
                results["violations"].append(violation)

            # 检查时间间隔是否过大
            max_gap = thresholds.get("max_time_gap", 10.0)
            if time_diff > max_gap:
                inconsistency = {
                    "rule": "time_progression",
                    "position": i,
                    "issue": "时间间隔过大",
                    "details": f"时间间隔 {time_diff:.1f}秒 > {max_gap}秒",
                    "severity": "medium"
                }
                results["inconsistencies"].append(inconsistency)

            # 检查时间回跳
            max_reverse = thresholds.get("max_reverse_jump", 1.0)
            if time_diff < -max_reverse:
                violation = {
                    "rule": "time_progression",
                    "position": i,
                    "issue": "时间回跳过大",
                    "details": f"时间回跳 {-time_diff:.1f}秒 > {max_reverse}秒",
                    "severity": "high"
                }
                results["violations"].append(violation)

            if time_diff >= 0 and time_diff <= max_gap:
                results["passed_checks"].append({
                    "position": i,
                    "check": "time_progression",
                    "result": f"时间前进 {time_diff:.1f}秒"
                })

        return results

    def _check_temporal_gaps(self, states: List[Dict],
                             thresholds: Dict) -> Dict[str, Any]:
        """检查时间间隙"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": len(states) - 1
        }

        # 计算平均时间间隔
        time_diffs = []
        for i in range(1, len(states)):
            time1 = states[i - 1].get("timestamp")
            time2 = states[i].get("timestamp")

            if time1 and time2:
                time_diffs.append((time2 - time1).total_seconds())

        if len(time_diffs) < 2:
            return results

        avg_interval = np.mean(time_diffs)
        std_interval = np.std(time_diffs)

        # 检查间隔一致性
        for i, diff in enumerate(time_diffs):
            if std_interval > 0:
                z_score = abs(diff - avg_interval) / std_interval
                if z_score > 2.0:  # 2个标准差以外
                    inconsistency = {
                        "rule": "temporal_gaps",
                        "position": i + 1,
                        "issue": "异常时间间隔",
                        "details": f"间隔 {diff:.1f}秒，平均 {avg_interval:.1f}秒",
                        "severity": "low"
                    }
                    results["inconsistencies"].append(inconsistency)
            else:
                results["passed_checks"].append({
                    "position": i + 1,
                    "check": "temporal_gaps",
                    "result": f"稳定间隔 {diff:.1f}秒"
                })

        return results

    def _check_event_ordering(self, states: List[Dict],
                              thresholds: Dict) -> Dict[str, Any]:
        """检查事件顺序"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": 0  # 根据事件数量动态计算
        }

        # 这里可以添加具体的事件顺序检查逻辑
        # 例如：某些事件必须在其他事件之前发生

        return results

    def _check_position_continuity(self, states: List[Dict],
                                   thresholds: Dict) -> Dict[str, Any]:
        """检查位置连续性"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": len(states) - 1
        }

        for i in range(1, len(states)):
            state1 = states[i - 1]
            state2 = states[i]

            pos1 = state1.get("position")
            pos2 = state2.get("position")

            if not pos1 or not pos2:
                results["warnings"].append({
                    "position": i,
                    "issue": "缺少位置信息",
                    "severity": "medium"
                })
                continue

            if not (isinstance(pos1, (list, tuple)) and isinstance(pos2, (list, tuple))):
                continue

            if len(pos1) < 3 or len(pos2) < 3:
                continue

            # 计算移动距离
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            dz = pos2[2] - pos1[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            # 计算时间间隔
            time1 = state1.get("timestamp")
            time2 = state2.get("timestamp")

            if time1 and time2:
                time_diff = (time2 - time1).total_seconds()
                if time_diff > 0:
                    speed = distance / time_diff
                else:
                    speed = float('inf')
            else:
                speed = float('inf')

            # 检查瞬移
            max_teleport = thresholds.get("max_teleport_distance", 5.0)
            if distance > max_teleport:
                violation = {
                    "rule": "position_continuity",
                    "position": i,
                    "issue": "瞬移检测",
                    "details": f"移动距离 {distance:.1f}米 > {max_teleport}米",
                    "severity": "high"
                }
                results["violations"].append(violation)

            # 检查速度
            max_speed = thresholds.get("max_speed", 10.0)
            if speed > max_speed:
                violation = {
                    "rule": "position_continuity",
                    "position": i,
                    "issue": "超速移动",
                    "details": f"速度 {speed:.1f}米/秒 > {max_speed}米/秒",
                    "severity": "high"
                }
                results["violations"].append(violation)

            if distance <= max_teleport and (time_diff <= 0 or speed <= max_speed):
                results["passed_checks"].append({
                    "position": i,
                    "check": "position_continuity",
                    "result": f"移动 {distance:.2f}米，速度 {speed:.1f}米/秒"
                })

        return results

    def _check_collision_consistency(self, states: List[Dict],
                                     thresholds: Dict) -> Dict[str, Any]:
        """检查碰撞一致性"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": len(states) - 1
        }

        # 简化的碰撞一致性检查
        # 可以扩展为检查物体是否穿过其他物体等

        return results

    def _check_boundary_violations(self, states: List[Dict],
                                   thresholds: Dict) -> Dict[str, Any]:
        """检查边界违反"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": len(states)
        }

        # 检查位置是否在合理范围内
        for i, state in enumerate(states):
            pos = state.get("position")

            if pos and isinstance(pos, (list, tuple)) and len(pos) >= 3:
                # 检查是否在合理的地面之上
                if pos[1] < -10:  # 地下10米
                    violation = {
                        "rule": "boundary_violations",
                        "position": i,
                        "issue": "位置在地下",
                        "details": f"Y坐标 {pos[1]:.1f}米 < -10米",
                        "severity": "medium"
                    }
                    results["violations"].append(violation)
                elif pos[1] > 1000:  # 天上1000米
                    violation = {
                        "rule": "boundary_violations",
                        "position": i,
                        "issue": "位置过高",
                        "details": f"Y坐标 {pos[1]:.1f}米 > 1000米",
                        "severity": "medium"
                    }
                    results["violations"].append(violation)
                else:
                    results["passed_checks"].append({
                        "position": i,
                        "check": "boundary_violations",
                        "result": f"位置在合理范围内 ({pos[1]:.1f}米)"
                    })

        return results

    def _check_state_transitions(self, states: List[Dict],
                                 thresholds: Dict) -> Dict[str, Any]:
        """检查状态转移"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": len(states) - 1
        }

        # 检查状态转移是否合理
        # 例如：物体不能从未破损直接变成修复完成

        return results

    def _check_property_consistency(self, states: List[Dict],
                                    thresholds: Dict) -> Dict[str, Any]:
        """检查属性一致性"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": len(states) - 1
        }

        # 检查属性是否一致
        for i in range(1, len(states)):
            state1 = states[i - 1]
            state2 = states[i]

            # 检查关键属性是否突变
            for key in ["material", "type", "name"]:
                val1 = state1.get(key)
                val2 = state2.get(key)

                if val1 is not None and val2 is not None and val1 != val2:
                    inconsistency = {
                        "rule": "property_consistency",
                        "position": i,
                        "issue": f"{key}属性突变",
                        "details": f"从 '{val1}' 变为 '{val2}'",
                        "severity": "high"
                    }
                    results["inconsistencies"].append(inconsistency)
                elif val1 == val2 and val1 is not None:
                    results["passed_checks"].append({
                        "position": i,
                        "check": "property_consistency",
                        "result": f"{key}属性一致: '{val1}'"
                    })

        return results

    def _check_causality(self, states: List[Dict],
                         thresholds: Dict) -> Dict[str, Any]:
        """检查因果关系"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": len(states) - 1
        }

        # 检查因果关系
        # 例如：碰撞必须在接触之后，破碎必须在受力之后

        return results

    def _check_physics_laws(self, states: List[Dict],
                            thresholds: Dict) -> Dict[str, Any]:
        """检查物理定律"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": len(states) - 1
        }

        # 检查重力
        for i in range(1, len(states)):
            state1 = states[i - 1]
            state2 = states[i]

            pos1 = state1.get("position")
            pos2 = state2.get("position")

            if (pos1 and isinstance(pos1, (list, tuple)) and len(pos1) >= 3 and
                    pos2 and isinstance(pos2, (list, tuple)) and len(pos2) >= 3):

                # 检查自由落体加速度
                time1 = state1.get("timestamp")
                time2 = state2.get("timestamp")

                if time1 and time2:
                    time_diff = (time2 - time1).total_seconds()
                    if time_diff > 0:
                        # 计算Y方向加速度
                        dy = pos2[1] - pos1[1]
                        vy1 = state1.get("velocity", [0, 0, 0])[1] if isinstance(state1.get("velocity"), (list, tuple)) else 0
                        vy2 = state2.get("velocity", [0, 0, 0])[1] if isinstance(state2.get("velocity"), (list, tuple)) else 0

                        # 简化检查：如果不是支持物体，应该下落
                        supported = state2.get("supported", True)
                        if not supported and dy > 0.1 and vy2 > vy1:
                            # 物体无支撑但上升且加速
                            warning = {
                                "rule": "physics_laws",
                                "position": i,
                                "issue": "违反重力",
                                "details": f"无支撑物体上升 {dy:.2f}米",
                                "severity": "medium"
                            }
                            results["warnings"].append(warning)
                        else:
                            results["passed_checks"].append({
                                "position": i,
                                "check": "physics_laws",
                                "result": "运动符合重力"
                            })

        return results

    def _check_material_consistency(self, states: List[Dict],
                                    thresholds: Dict) -> Dict[str, Any]:
        """检查材料一致性"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": len(states) - 1
        }

        # 检查材料属性是否合理
        for i, state in enumerate(states):
            material = state.get("material")
            density = state.get("density")

            if material and density:
                # 可以根据材料类型检查密度范围
                # 例如：木材密度应该在300-800 kg/m³之间
                material_ranges = {
                    "wood": (300, 800),
                    "steel": (7500, 8000),
                    "plastic": (800, 1500),
                    "rubber": (900, 1200)
                }

                if material in material_ranges:
                    min_density, max_density = material_ranges[material]
                    if density < min_density or density > max_density:
                        inconsistency = {
                            "rule": "material_consistency",
                            "position": i,
                            "issue": "密度与材料不匹配",
                            "details": f"{material}密度 {density}kg/m³ 不在范围 [{min_density}, {max_density}]",
                            "severity": "medium"
                        }
                        results["inconsistencies"].append(inconsistency)
                    else:
                        results["passed_checks"].append({
                            "position": i,
                            "check": "material_consistency",
                            "result": f"{material}密度 {density}kg/m³ 合理"
                        })

        return results

    def _check_energy_conservation(self, states: List[Dict],
                                   thresholds: Dict) -> Dict[str, Any]:
        """检查能量守恒"""
        results = {
            "violations": [],
            "inconsistencies": [],
            "warnings": [],
            "passed_checks": [],
            "total_checks": len(states) - 1
        }

        # 简化的能量守恒检查
        # 可以扩展为计算动能、势能变化

        return results

    def _record_consistency_check(self, report: Dict[str, Any]):
        """记录一致性检查"""
        check_record = {
            "timestamp": datetime.now(),
            "overall_score": report["overall_consistency_score"],
            "violation_count": sum(len(v) for v in report["rule_violations"].values()),
            "inconsistency_count": len(report["inconsistencies"]),
            "warning_count": len(report["warnings"]),
            "detailed_scores": report["detailed_scores"]
        }

        # 记录历史
        history_key = f"check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.consistency_history[history_key].append(check_record)

        # 保持历史长度
        if len(self.consistency_history) > 100:
            # 删除最旧的记录
            oldest_key = min(self.consistency_history.keys())
            del self.consistency_history[oldest_key]

    def _generate_consistency_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """生成一致性建议"""
        recommendations = []

        overall_score = report["overall_consistency_score"]

        if overall_score < 0.5:
            recommendations.append("整体一致性较差，建议重新检查数据源")
        elif overall_score < 0.7:
            recommendations.append("整体一致性一般，建议优化数据处理流程")

        # 根据具体问题生成建议
        for rule_name, violations in report["rule_violations"].items():
            if violations:
                recommendations.append(f"{rule_name} 规则有 {len(violations)} 个违反，需要修复")

        for inconsistency in report["inconsistencies"]:
            if inconsistency.get("severity") == "high":
                recommendations.append(f"发现高严重性不一致：{inconsistency.get('issue')}")

        return recommendations

    def get_consistency_trend(self) -> Dict[str, Any]:
        """获取一致性趋势"""
        if not self.consistency_history:
            return {"error": "无历史数据"}

        scores = []
        timestamps = []

        for check_records in self.consistency_history.values():
            for record in check_records:
                scores.append(record["overall_score"])
                timestamps.append(record["timestamp"])

        if len(scores) < 2:
            return {"average_score": scores[0] if scores else 0}

        # 计算趋势
        x = np.arange(len(scores))
        slope, intercept = np.polyfit(x, scores, 1)

        trend = {
            "average_score": np.mean(scores),
            "score_std": np.std(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "trend_slope": slope,
            "trend_direction": "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable",
            "data_points": len(scores),
            "time_span": (max(timestamps) - min(timestamps)).total_seconds()
        }

        return trend

    def generate_consistency_report(self) -> str:
        """生成一致性报告"""
        report = "一致性检查报告\n"
        report += "=" * 60 + "\n"

        # 趋势分析
        trend = self.get_consistency_trend()
        if "error" not in trend:
            report += f"平均一致性分数: {trend['average_score']:.3f}\n"
            report += f"分数标准差: {trend['score_std']:.3f}\n"
            report += f"趋势: {trend['trend_direction']} (斜率: {trend['trend_slope']:.4f})\n"
            report += f"数据点: {trend['data_points']}, 时间跨度: {trend['time_span']:.0f}秒\n"

        # 规则违反统计
        report += "\n规则违反统计:\n"
        for rule_key, violations in self.rule_violations.items():
            if violations:
                report += f"- {rule_key}: {len(violations)} 次违反\n"
                # 最近违反
                if violations:
                    recent = violations[-1]
                    report += f"  最近: {recent['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"

        # 不一致模式
        report += "\n检测到的不一致模式:\n"
        if self.inconsistency_patterns:
            for pattern_name, pattern_info in self.inconsistency_patterns.items():
                report += f"- {pattern_name}: {pattern_info.get('count', 0)} 次\n"
        else:
            report += "- 无显著不一致模式\n"

        # 建议
        report += "\n建议:\n"
        if trend.get('trend_direction') == 'declining':
            report += "- 一致性在下降，建议检查最近的变化\n"

        if any(len(v) > 10 for v in self.rule_violations.values()):
            report += "- 某些规则违反频繁，需要重点关注\n"

        report += "- 定期运行一致性检查以监控质量\n"

        return report