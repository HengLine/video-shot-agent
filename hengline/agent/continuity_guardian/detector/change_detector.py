"""
@FileName: change_detector.py
@Description: 变化检测器
@Author: HengLine
@Time: 2026/1/4 22:27
"""
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime
import numpy as np

from hengline.agent.continuity_guardian.detector.anomaly_detector import AnomalyDetector


class ChangeDetector:
    """变化检测器"""

    def __init__(self, change_thresholds: Optional[Dict] = None):
        self.change_thresholds = change_thresholds or self._get_default_thresholds()
        self.change_history: Dict[str, List[Dict]] = defaultdict(list)
        self.baseline_states: Dict[str, Dict] = {}
        self.anomaly_detector = AnomalyDetector()

    def _get_default_thresholds(self) -> Dict[str, float]:
        """获取默认变化阈值"""
        return {
            "position": 0.1,  # 位置变化阈值（米）
            "rotation": 5.0,  # 旋转变化阈值（度）
            "scale": 0.05,  # 缩放变化阈值
            "color": 0.1,  # 颜色变化阈值（0-1）
            "intensity": 0.2,  # 强度变化阈值
            "texture": 0.3,  # 纹理变化阈值
            "shape": 0.15,  # 形状变化阈值
            "motion": 0.5,  # 运动变化阈值
            "appearance": 0.1,  # 外观变化阈值
            "state": 0.01  # 状态变化阈值
        }

    def detect_changes(self, old_state: Dict, new_state: Dict,
                       entity_type: str = "general") -> Dict[str, Any]:
        """检测状态变化"""
        changes = {
            "changed_attributes": [],
            "unchanged_attributes": [],
            "change_magnitudes": {},
            "significant_changes": [],
            "insignificant_changes": [],
            "anomalies": [],
            "overall_change_level": "none",
            "change_score": 0.0
        }

        # 获取所有属性
        all_keys = set(old_state.keys()) | set(new_state.keys())

        if not all_keys:
            return changes

        change_scores = []

        for key in all_keys:
            old_val = old_state.get(key)
            new_val = new_state.get(key)

            # 检查属性是否存在
            if old_val is None and new_val is None:
                changes["unchanged_attributes"].append(key)
                continue

            # 属性添加或删除
            if old_val is None:
                changes["changed_attributes"].append(key)
                changes["change_magnitudes"][key] = "added"
                change_scores.append(1.0)  # 添加属性得高分
                changes["significant_changes"].append({
                    "attribute": key,
                    "change": "added",
                    "old_value": None,
                    "new_value": new_val
                })
                continue

            if new_val is None:
                changes["changed_attributes"].append(key)
                changes["change_magnitudes"][key] = "removed"
                change_scores.append(1.0)  # 删除属性得高分
                changes["significant_changes"].append({
                    "attribute": key,
                    "change": "removed",
                    "old_value": old_val,
                    "new_value": None
                })
                continue

            # 计算变化
            change_result = self._calculate_attribute_change(key, old_val, new_val, entity_type)

            if change_result["changed"]:
                changes["changed_attributes"].append(key)
                changes["change_magnitudes"][key] = change_result["magnitude"]
                change_scores.append(change_result["score"])

                change_record = {
                    "attribute": key,
                    "change": "modified",
                    "old_value": old_val,
                    "new_value": new_val,
                    "magnitude": change_result["magnitude"],
                    "score": change_result["score"]
                }

                # 判断是否显著变化
                if change_result["significant"]:
                    changes["significant_changes"].append(change_record)
                else:
                    changes["insignificant_changes"].append(change_record)

                # 检测异常
                anomaly = self.anomaly_detector.detect_attribute_anomaly(
                    key, old_val, new_val, change_result
                )
                if anomaly:
                    changes["anomalies"].append(anomaly)
            else:
                changes["unchanged_attributes"].append(key)

        # 计算总体变化水平
        if change_scores:
            avg_score = np.mean(change_scores)
            changes["change_score"] = avg_score

            if avg_score > 0.7:
                changes["overall_change_level"] = "major"
            elif avg_score > 0.3:
                changes["overall_change_level"] = "moderate"
            elif avg_score > 0.1:
                changes["overall_change_level"] = "minor"
            else:
                changes["overall_change_level"] = "none"

        # 记录变化历史
        self._record_change_history(old_state, new_state, changes)

        return changes

    def _calculate_attribute_change(self, attribute: str, old_val: Any,
                                    new_val: Any, entity_type: str) -> Dict[str, Any]:
        """计算属性变化"""
        result = {
            "changed": False,
            "magnitude": 0,
            "score": 0.0,
            "significant": False
        }

        # 根据属性类型计算变化
        if attribute in self.change_thresholds:
            threshold = self.change_thresholds[attribute]

            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                # 数值变化
                magnitude = abs(new_val - old_val)
                result["magnitude"] = magnitude
                result["changed"] = magnitude > threshold
                result["score"] = min(1.0, magnitude / (threshold * 10))
                result["significant"] = magnitude > threshold * 2

            elif isinstance(old_val, (list, tuple)) and isinstance(new_val, (list, tuple)):
                # 序列变化
                if len(old_val) != len(new_val):
                    result["changed"] = True
                    result["magnitude"] = abs(len(new_val) - len(old_val))
                    result["score"] = 1.0
                    result["significant"] = True
                else:
                    # 计算序列差异
                    diffs = []
                    for o, n in zip(old_val, new_val):
                        if isinstance(o, (int, float)) and isinstance(n, (int, float)):
                            diffs.append(abs(n - o))

                    if diffs:
                        magnitude = np.mean(diffs)
                        result["magnitude"] = magnitude
                        result["changed"] = magnitude > threshold
                        result["score"] = min(1.0, magnitude / (threshold * 10))
                        result["significant"] = magnitude > threshold * 2

            elif isinstance(old_val, dict) and isinstance(new_val, dict):
                # 字典变化（递归）
                sub_changes = self.detect_changes(old_val, new_val, entity_type)
                result["magnitude"] = sub_changes["change_score"]
                result["changed"] = sub_changes["change_score"] > threshold
                result["score"] = sub_changes["change_score"]
                result["significant"] = sub_changes["change_score"] > threshold * 2

            else:
                # 其他类型（字符串等）
                result["changed"] = old_val != new_val
                result["magnitude"] = 1.0 if old_val != new_val else 0.0
                result["score"] = 1.0 if old_val != new_val else 0.0
                result["significant"] = old_val != new_val

        else:
            # 无阈值定义的属性
            result["changed"] = old_val != new_val
            result["magnitude"] = 1.0 if old_val != new_val else 0.0
            result["score"] = 1.0 if old_val != new_val else 0.0
            result["significant"] = old_val != new_val

        return result

    def _record_change_history(self, old_state: Dict, new_state: Dict,
                               changes: Dict[str, Any]):
        """记录变化历史"""
        change_record = {
            "timestamp": datetime.now(),
            "old_state": old_state.copy(),
            "new_state": new_state.copy(),
            "changes": changes.copy(),
            "change_score": changes["change_score"],
            "change_level": changes["overall_change_level"]
        }

        # 为每个改变的属性记录历史
        for change in changes["significant_changes"]:
            attribute = change["attribute"]
            self.change_history[attribute].append({
                "timestamp": change_record["timestamp"],
                "old_value": change["old_value"],
                "new_value": change["new_value"],
                "magnitude": change.get("magnitude", 0)
            })

            # 保持历史长度
            if len(self.change_history[attribute]) > 100:
                self.change_history[attribute].pop(0)

    def set_baseline(self, entity_id: str, baseline_state: Dict):
        """设置基线状态"""
        self.baseline_states[entity_id] = baseline_state.copy()

    def detect_drift_from_baseline(self, entity_id: str,
                                   current_state: Dict) -> Dict[str, Any]:
        """检测相对于基线的漂移"""
        if entity_id not in self.baseline_states:
            return {"error": "No baseline set for entity"}

        baseline = self.baseline_states[entity_id]
        changes = self.detect_changes(baseline, current_state)

        drift_analysis = {
            "entity_id": entity_id,
            "baseline_time": "unknown",  # 可以扩展存储基线时间
            "current_time": datetime.now(),
            "change_summary": changes,
            "drift_score": changes["change_score"],
            "drift_level": changes["overall_change_level"],
            "attribute_drifts": []
        }

        # 分析每个属性的漂移
        for change in changes["significant_changes"]:
            drift_record = {
                "attribute": change["attribute"],
                "baseline_value": change["old_value"],
                "current_value": change["new_value"],
                "magnitude": change.get("magnitude", 0),
                "score": change.get("score", 0)
            }

            # 检测漂移模式
            drift_record["pattern"] = self._detect_drift_pattern(
                change["attribute"],
                change["old_value"],
                change["new_value"]
            )

            drift_analysis["attribute_drifts"].append(drift_record)

        return drift_analysis

    def _detect_drift_pattern(self, attribute: str, baseline_val: Any,
                              current_val: Any) -> str:
        """检测漂移模式"""
        if isinstance(baseline_val, (int, float)) and isinstance(current_val, (int, float)):
            diff = current_val - baseline_val

            if abs(diff) < 0.001:
                return "stable"
            elif diff > 0:
                return "increasing"
            else:
                return "decreasing"

        return "changed"

    def get_change_statistics(self, attribute: Optional[str] = None) -> Dict[str, Any]:
        """获取变化统计信息"""
        if attribute:
            history = self.change_history.get(attribute, [])
        else:
            # 合并所有属性的历史
            history = []
            for attr_history in self.change_history.values():
                history.extend(attr_history)

        if not history:
            return {"total_changes": 0, "recent_changes": []}

        time_span = (history[-1]["timestamp"] - history[0]["timestamp"]).total_seconds()

        stats = {
            "total_changes": len(history),
            "time_span": time_span,
            "change_frequency": len(history) / max(1, time_span),
            "magnitude_stats": {
                "mean": 0,
                "std": 0,
                "max": 0,
                "min": float('inf')
            },
            "recent_changes": history[-10:]  # 最近10次变化
        }

        # 计算幅度统计
        magnitudes = [change.get("magnitude", 0) for change in history]
        if magnitudes:
            stats["magnitude_stats"]["mean"] = np.mean(magnitudes)
            stats["magnitude_stats"]["std"] = np.std(magnitudes)
            stats["magnitude_stats"]["max"] = max(magnitudes)
            stats["magnitude_stats"]["min"] = min(magnitudes)

        return stats

    def detect_change_patterns(self, attribute: str) -> List[Dict[str, Any]]:
        """检测变化模式"""
        history = self.change_history.get(attribute, [])
        if len(history) < 3:
            return []

        patterns = []

        # 检测周期性变化
        periodic_pattern = self._detect_periodic_pattern(history)
        if periodic_pattern:
            patterns.append(periodic_pattern)

        # 检测趋势
        trend_pattern = self._detect_trend_pattern(history)
        if trend_pattern:
            patterns.append(trend_pattern)

        # 检测异常点
        anomaly_patterns = self._detect_anomaly_patterns(history)
        patterns.extend(anomaly_patterns)

        return patterns

    def _detect_periodic_pattern(self, history: List[Dict]) -> Optional[Dict]:
        """检测周期性模式"""
        if len(history) < 5:
            return None

        # 简化实现：检查时间间隔的规律性
        timestamps = [h["timestamp"] for h in history]
        intervals = []

        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i - 1]).total_seconds()
            intervals.append(interval)

        if len(intervals) < 2:
            return None

        # 计算间隔的变异系数
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        if mean_interval > 0:
            cv = std_interval / mean_interval
            if cv < 0.3:  # 变异系数小，说明规律
                return {
                    "type": "periodic",
                    "period": mean_interval,
                    "regularity": 1.0 - cv,
                    "confidence": min(1.0, 1.0 - cv * 2)
                }

        return None

    def _detect_trend_pattern(self, history: List[Dict]) -> Optional[Dict]:
        """检测趋势模式"""
        if len(history) < 3:
            return None

        magnitudes = [h.get("magnitude", 0) for h in history]

        # 简单线性趋势检测
        x = np.arange(len(magnitudes))
        slope, intercept = np.polyfit(x, magnitudes, 1)

        # 计算趋势强度
        if abs(slope) > 0.01:
            trend_type = "increasing" if slope > 0 else "decreasing"
            strength = min(1.0, abs(slope) * 10)

            return {
                "type": "trend",
                "trend": trend_type,
                "slope": slope,
                "strength": strength,
                "confidence": min(1.0, strength * 2)
            }

        return None

    def _detect_anomaly_patterns(self, history: List[Dict]) -> List[Dict]:
        """检测异常模式"""
        if len(history) < 5:
            return []

        magnitudes = [h.get("magnitude", 0) for h in history]

        # 使用简单阈值检测异常
        mean_mag = np.mean(magnitudes)
        std_mag = np.std(magnitudes)

        anomalies = []
        for i, mag in enumerate(magnitudes):
            if std_mag > 0 and abs(mag - mean_mag) > 2 * std_mag:
                anomalies.append({
                    "type": "anomaly",
                    "position": i,
                    "magnitude": mag,
                    "deviation": (mag - mean_mag) / std_mag,
                    "timestamp": history[i]["timestamp"]
                })

        return anomalies

    def generate_change_report(self, entity_id: Optional[str] = None) -> str:
        """生成变化报告"""
        report = "变化检测报告\n"
        report += "=" * 50 + "\n"

        if entity_id:
            report += f"实体: {entity_id}\n"

        # 总体统计
        stats = self.get_change_statistics()
        report += f"总变化次数: {stats['total_changes']}\n"

        if stats['total_changes'] > 0:
            report += f"时间跨度: {stats['time_span']:.1f}秒\n"
            report += f"变化频率: {stats['change_frequency']:.3f}次/秒\n"

            mag_stats = stats['magnitude_stats']
            report += f"变化幅度 - 平均: {mag_stats['mean']:.3f}, "
            report += f"标准差: {mag_stats['std']:.3f}\n"

        # 最近变化
        if stats['recent_changes']:
            report += "\n最近变化:\n"
            for i, change in enumerate(stats['recent_changes'][-5:], 1):
                report += f"{i}. 时间: {change['timestamp'].strftime('%H:%M:%S')}, "
                report += f"幅度: {change.get('magnitude', 0):.3f}\n"

        # 变化模式
        report += "\n检测到的模式:\n"
        all_patterns = []
        for attribute in self.change_history.keys():
            patterns = self.detect_change_patterns(attribute)
            if patterns:
                all_patterns.extend(patterns)

        if all_patterns:
            for pattern in all_patterns[:3]:  # 显示前3个模式
                report += f"- {pattern['type']}: {pattern.get('trend', 'N/A')}, "
                report += f"置信度: {pattern.get('confidence', 0):.2f}\n"
        else:
            report += "- 未检测到明显模式\n"

        return report