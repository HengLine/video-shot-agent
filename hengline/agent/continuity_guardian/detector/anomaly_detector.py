"""
@FileName: anomaly_detector.py
@Description: 异常检测器
@Author: HengLine
@Time: 2026/1/4 22:38
"""
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, Optional, List


class AnomalyDetector:
    """异常检测器（ChangeDetector的辅助类）"""

    def __init__(self):
        self.anomaly_models: Dict[str, Any] = {}
        self.anomaly_history: Dict[str, List] = defaultdict(list)

    def detect_attribute_anomaly(self, attribute: str, old_val: Any,
                                 new_val: Any, change_result: Dict) -> Optional[Dict]:
        """检测属性异常"""
        anomaly = None

        # 基于变化幅度检测异常
        magnitude = change_result.get("magnitude", 0)

        if isinstance(magnitude, (int, float)):
            # 简单阈值检测
            if magnitude > 10.0:  # 异常高变化
                anomaly = {
                    "attribute": attribute,
                    "type": "extreme_change",
                    "magnitude": magnitude,
                    "old_value": old_val,
                    "new_value": new_val,
                    "confidence": min(1.0, magnitude / 20.0),
                    "severity": "high"
                }
            elif magnitude > 5.0:
                anomaly = {
                    "attribute": attribute,
                    "type": "large_change",
                    "magnitude": magnitude,
                    "old_value": old_val,
                    "new_value": new_val,
                    "confidence": 0.7,
                    "severity": "medium"
                }

        # 基于值范围检测异常
        if isinstance(new_val, (int, float)):
            if new_val > 10000 or new_val < -10000:  # 异常值
                anomaly = {
                    "attribute": attribute,
                    "type": "out_of_range",
                    "value": new_val,
                    "expected_range": "[-10000, 10000]",
                    "confidence": 0.9,
                    "severity": "high"
                }

        # 记录异常
        if anomaly:
            self.anomaly_history[attribute].append({
                "timestamp": datetime.now(),
                "anomaly": anomaly
            })

            # 保持历史长度
            if len(self.anomaly_history[attribute]) > 100:
                self.anomaly_history[attribute].pop(0)

        return anomaly

    def get_anomaly_statistics(self, attribute: Optional[str] = None) -> Dict[str, Any]:
        """获取异常统计"""
        if attribute:
            history = self.anomaly_history.get(attribute, [])
        else:
            history = []
            for attr_history in self.anomaly_history.values():
                history.extend(attr_history)

        stats = {
            "total_anomalies": len(history),
            "by_type": defaultdict(int),
            "by_severity": defaultdict(int),
            "recent_anomalies": history[-10:] if history else []
        }

        for record in history:
            anomaly = record["anomaly"]
            stats["by_type"][anomaly["type"]] += 1
            stats["by_severity"][anomaly["severity"]] += 1

        stats["by_type"] = dict(stats["by_type"])
        stats["by_severity"] = dict(stats["by_severity"])

        return stats
