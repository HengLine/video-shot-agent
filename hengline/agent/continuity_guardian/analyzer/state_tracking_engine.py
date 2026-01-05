"""
@FileName: state_tracking_engine.py
@Description: 状态管理
@Author: HengLine
@Time: 2026/1/4 16:45
"""
import math
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any

import numpy as np

from hengline.agent.continuity_guardian.detector.change_detector import ChangeDetector
from .spatial_Index_analyzer import SpatialIndex
from .temporal_graph_analyzer import TemporalGraph
from ..detector.consistency_detector import ConsistencyChecker


class TemporalConsistency(Enum):
    """时间一致性级别"""
    PERFECT = "perfect"        # 完美一致
    HIGH = "high"              # 高度一致
    MODERATE = "moderate"      # 中等一致
    LOW = "low"                # 低一致
    BROKEN = "broken"          # 断裂


class StateTrackingEngine:
    """状态跟踪引擎"""

    def __init__(self, tracking_config: Optional[Dict] = None):
        self.tracking_config = tracking_config or self._get_default_config()
        self.state_history: Dict[str, List[Dict]] = defaultdict(list)
        self.entity_registry: Dict[str, Dict] = {}
        self.temporal_graph = TemporalGraph()
        self.spatial_index = SpatialIndex()
        self.change_detector = ChangeDetector()
        self.consistency_checker = ConsistencyChecker()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "tracking_precision": "high",
            "history_length": 1000,
            "sampling_rate": 1.0,  # 采样率（秒）
            "tracked_entities": ["characters", "props", "environment", "camera"],
            "state_attributes": {
                "characters": ["position", "rotation", "appearance", "emotion", "action"],
                "props": ["position", "rotation", "state", "interactions"],
                "environment": ["lighting", "weather", "time", "effects"],
                "camera": ["position", "rotation", "fov", "movement"]
            },
            "change_thresholds": {
                "position": 0.1,  # 米
                "rotation": 5.0,  # 度
                "appearance": 0.05,  # 外观变化阈值
                "lighting": 0.1  # 光照变化阈值
            }
        }

    def register_entity(self, entity_type: str, entity_id: str, initial_state: Dict) -> bool:
        """注册实体"""
        if entity_id in self.entity_registry:
            return False

        entity_record = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "initial_state": initial_state,
            "current_state": initial_state.copy(),
            "registration_time": datetime.now(),
            "state_history": [],
            "change_log": [],
            "metadata": {}
        }

        self.entity_registry[entity_id] = entity_record

        # 记录初始状态
        self._record_state(entity_id, initial_state, "initial")

        return True

    def update_entity_state(self, entity_id: str, new_state: Dict,
                            timestamp: Optional[datetime] = None,
                            change_reason: str = "update") -> Dict[str, Any]:
        """更新实体状态"""
        if entity_id not in self.entity_registry:
            return {"success": False, "error": "Entity not registered"}

        entity = self.entity_registry[entity_id]
        old_state = entity["current_state"]

        # 计算状态变化
        changes = self._detect_state_changes(old_state, new_state, entity["entity_type"])

        # 验证状态变化
        validation = self._validate_state_changes(entity_id, old_state, new_state, changes)

        if not validation["valid"] and validation["block_update"]:
            return {
                "success": False,
                "error": "State change validation failed",
                "validation_result": validation
            }

        # 更新当前状态
        entity["current_state"] = new_state.copy()

        # 记录状态
        record_result = self._record_state(entity_id, new_state, change_reason, timestamp)

        # 记录变化
        if changes["significant_changes"]:
            change_record = {
                "timestamp": timestamp or datetime.now(),
                "old_state": old_state,
                "new_state": new_state,
                "changes": changes,
                "reason": change_reason,
                "validation_result": validation
            }
            entity["change_log"].append(change_record)

        # 更新时空索引
        self._update_spatial_temporal_index(entity_id, new_state, timestamp)

        return {
            "success": True,
            "changes": changes,
            "validation_result": validation,
            "state_record_id": record_result.get("record_id")
        }

    def _record_state(self, entity_id: str, state: Dict,
                      change_reason: str, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """记录状态"""
        record_time = timestamp or datetime.now()
        record_id = f"{entity_id}_{record_time.timestamp()}"

        record = {
            "record_id": record_id,
            "entity_id": entity_id,
            "timestamp": record_time,
            "state": state.copy(),
            "change_reason": change_reason,
            "metadata": {}
        }

        # 添加到历史
        self.state_history[entity_id].append(record)

        # 保持历史长度
        if len(self.state_history[entity_id]) > self.tracking_config["history_length"]:
            self.state_history[entity_id].pop(0)

        # 更新实体注册表中的历史
        if entity_id in self.entity_registry:
            self.entity_registry[entity_id]["state_history"].append(record)

        return record

    def _detect_state_changes(self, old_state: Dict, new_state: Dict,
                              entity_type: str) -> Dict[str, Any]:
        """检测状态变化"""
        changes = {
            "changed_attributes": [],
            "unchanged_attributes": [],
            "significant_changes": False,
            "change_magnitudes": {},
            "temporal_consistency": TemporalConsistency.PERFECT
        }

        # 获取要跟踪的属性
        tracked_attrs = self.tracking_config["state_attributes"].get(entity_type, [])

        for attr in tracked_attrs:
            old_val = old_state.get(attr)
            new_val = new_state.get(attr)

            if old_val is None and new_val is None:
                changes["unchanged_attributes"].append(attr)
                continue

            if old_val is None or new_val is None:
                changes["changed_attributes"].append(attr)
                changes["change_magnitudes"][attr] = "added" if new_val else "removed"
                changes["significant_changes"] = True
                continue

            # 比较值
            if self._compare_values(old_val, new_val, attr, entity_type):
                changes["unchanged_attributes"].append(attr)
            else:
                changes["changed_attributes"].append(attr)
                magnitude = self._calculate_change_magnitude(old_val, new_val, attr)
                changes["change_magnitudes"][attr] = magnitude

                # 检查是否显著变化
                threshold = self.tracking_config["change_thresholds"].get(attr, 0.0)
                if isinstance(magnitude, (int, float)) and magnitude > threshold:
                    changes["significant_changes"] = True

        # 评估时间一致性
        changes["temporal_consistency"] = self._assess_temporal_consistency(changes)

        return changes

    def _compare_values(self, old_val: Any, new_val: Any, attr: str, entity_type: str) -> bool:
        """比较值是否相等"""
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            threshold = self.tracking_config["change_thresholds"].get(attr, 0.001)
            return abs(old_val - new_val) <= threshold

        if isinstance(old_val, list) and isinstance(new_val, list):
            if len(old_val) != len(new_val):
                return False
            return all(self._compare_values(o, n, attr, entity_type)
                       for o, n in zip(old_val, new_val))

        if isinstance(old_val, dict) and isinstance(new_val, dict):
            old_keys = set(old_val.keys())
            new_keys = set(new_val.keys())
            if old_keys != new_keys:
                return False
            return all(self._compare_values(old_val[k], new_val[k], attr, entity_type)
                       for k in old_keys)

        return old_val == new_val

    def _calculate_change_magnitude(self, old_val: Any, new_val: Any, attr: str) -> Any:
        """计算变化幅度"""
        if attr == "position" and isinstance(old_val, (list, tuple)) and isinstance(new_val, (list, tuple)):
            if len(old_val) >= 3 and len(new_val) >= 3:
                dx = new_val[0] - old_val[0]
                dy = new_val[1] - old_val[1]
                dz = new_val[2] - old_val[2]
                return math.sqrt(dx * dx + dy * dy + dz * dz)

        if attr == "rotation" and isinstance(old_val, (list, tuple)) and isinstance(new_val, (list, tuple)):
            # 计算角度差
            if len(old_val) >= 3 and len(new_val) >= 3:
                # 简化计算
                return 0.0

        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            return abs(new_val - old_val)

        return "changed"  # 无法量化的变化

    def _assess_temporal_consistency(self, changes: Dict) -> TemporalConsistency:
        """评估时间一致性"""
        if not changes["significant_changes"]:
            return TemporalConsistency.PERFECT

        change_count = len(changes["changed_attributes"])
        magnitude_sum = 0

        for mag in changes["change_magnitudes"].values():
            if isinstance(mag, (int, float)):
                magnitude_sum += mag

        if change_count == 0:
            return TemporalConsistency.PERFECT
        elif change_count <= 2 and magnitude_sum < 0.5:
            return TemporalConsistency.HIGH
        elif change_count <= 5 and magnitude_sum < 2.0:
            return TemporalConsistency.MODERATE
        elif change_count <= 10:
            return TemporalConsistency.LOW
        else:
            return TemporalConsistency.BROKEN

    def _validate_state_changes(self, entity_id: str, old_state: Dict,
                                new_state: Dict, changes: Dict) -> Dict[str, Any]:
        """验证状态变化"""
        validation = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "block_update": False,
            "consistency_score": 1.0
        }

        entity_type = self.entity_registry[entity_id]["entity_type"]

        # 检查物理合理性
        if entity_type == "characters":
            physical_validation = self._validate_character_state_change(old_state, new_state)
            validation["warnings"].extend(physical_validation["warnings"])
            if not physical_validation["valid"]:
                validation["errors"].extend(physical_validation["errors"])
                validation["valid"] = False

        # 检查时间一致性
        if changes["temporal_consistency"] == TemporalConsistency.BROKEN:
            validation["warnings"].append("时间一致性断裂")
            validation["consistency_score"] *= 0.5

        # 检查空间连续性
        if "position" in changes["changed_attributes"]:
            spatial_validation = self._validate_spatial_continuity(entity_id, old_state, new_state)
            validation["warnings"].extend(spatial_validation["warnings"])
            if not spatial_validation["valid"]:
                validation["block_update"] = True
                validation["valid"] = False

        # 更新一致性分数
        validation["consistency_score"] = max(0.0, min(1.0, validation["consistency_score"]))

        return validation

    def _validate_character_state_change(self, old_state: Dict, new_state: Dict) -> Dict[str, Any]:
        """验证角色状态变化"""
        result = {"valid": True, "warnings": [], "errors": []}

        # 检查位置变化速度
        if "position" in old_state and "position" in new_state:
            old_pos = old_state["position"]
            new_pos = new_state["position"]

            if isinstance(old_pos, (list, tuple)) and isinstance(new_pos, (list, tuple)):
                if len(old_pos) >= 2 and len(new_pos) >= 2:
                    dx = new_pos[0] - old_pos[0]
                    dy = new_pos[1] - old_pos[1]
                    distance = math.sqrt(dx * dx + dy * dy)

                    # 假设时间间隔为1秒（简化）
                    speed = distance  # 米/秒

                    if speed > 10.0:  # 超过10米/秒
                        result["warnings"].append(f"移动速度过快: {speed:.1f}m/s")
                    elif speed > 6.0:  # 超过6米/秒
                        result["warnings"].append(f"移动速度较快: {speed:.1f}m/s")

        return result

    def _validate_spatial_continuity(self, entity_id: str, old_state: Dict,
                                     new_state: Dict) -> Dict[str, Any]:
        """验证空间连续性"""
        result = {"valid": True, "warnings": []}

        # 检查是否穿过固体物体
        if self._check_collision_with_obstacles(entity_id, old_state, new_state):
            result["valid"] = False
            result["warnings"].append("检测到穿过障碍物的移动")

        return result

    def _check_collision_with_obstacles(self, entity_id: str, old_state: Dict,
                                        new_state: Dict) -> bool:
        """检查与障碍物碰撞"""
        # 简化的碰撞检测
        return False

    def _update_spatial_temporal_index(self, entity_id: str, state: Dict,
                                       timestamp: Optional[datetime]):
        """更新时空索引"""
        if "position" in state:
            position = state["position"]
            if isinstance(position, (list, tuple)) and len(position) >= 2:
                self.spatial_index.update_position(entity_id, position[0], position[1])

        self.temporal_graph.add_event(entity_id, timestamp or datetime.now(), state)

    def get_entity_state(self, entity_id: str,
                         timestamp: Optional[datetime] = None) -> Optional[Dict]:
        """获取实体状态"""
        if entity_id not in self.entity_registry:
            return None

        if timestamp is None:
            return self.entity_registry[entity_id]["current_state"]

        # 查找指定时间点的状态
        history = self.state_history.get(entity_id, [])
        if not history:
            return None

        # 二分查找最接近的时间点
        left, right = 0, len(history) - 1
        while left <= right:
            mid = (left + right) // 2
            record_time = history[mid]["timestamp"]

            if record_time < timestamp:
                left = mid + 1
            elif record_time > timestamp:
                right = mid - 1
            else:
                return history[mid]["state"]

        # 返回最接近的时间点
        if right < 0:
            return history[0]["state"]
        if left >= len(history):
            return history[-1]["state"]

        # 选择更接近的时间点
        left_diff = abs((history[left]["timestamp"] - timestamp).total_seconds())
        right_diff = abs((history[right]["timestamp"] - timestamp).total_seconds())

        return history[left]["state"] if left_diff < right_diff else history[right]["state"]

    def get_state_history(self, entity_id: str,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> List[Dict]:
        """获取状态历史"""
        if entity_id not in self.state_history:
            return []

        history = self.state_history[entity_id]

        if start_time is None and end_time is None:
            return history

        filtered = []
        for record in history:
            record_time = record["timestamp"]

            if start_time and record_time < start_time:
                continue
            if end_time and record_time > end_time:
                continue

            filtered.append(record)

        return filtered

    def get_state_summary(self, entity_id: str) -> Dict[str, Any]:
        """获取状态摘要"""
        if entity_id not in self.entity_registry:
            return {"error": "Entity not found"}

        entity = self.entity_registry[entity_id]
        history = self.state_history.get(entity_id, [])

        summary = {
            "entity_id": entity_id,
            "entity_type": entity["entity_type"],
            "registration_time": entity["registration_time"],
            "state_count": len(history),
            "change_count": len(entity["change_log"]),
            "current_state_summary": self._summarize_state(entity["current_state"]),
            "recent_changes": entity["change_log"][-5:] if entity["change_log"] else [],
            "temporal_consistency": self._calculate_overall_consistency(entity_id)
        }

        return summary

    def _summarize_state(self, state: Dict) -> Dict[str, Any]:
        """总结状态"""
        summary = {}

        for key, value in state.items():
            if isinstance(value, (int, float)):
                summary[key] = value
            elif isinstance(value, str):
                summary[key] = value[:50]  # 截断长字符串
            elif isinstance(value, (list, tuple)):
                summary[key] = f"{type(value).__name__}[{len(value)}]"
            elif isinstance(value, dict):
                summary[key] = f"dict[{len(value)}]"
            else:
                summary[key] = str(type(value))

        return summary

    def _calculate_overall_consistency(self, entity_id: str) -> Dict[str, float]:
        """计算整体一致性"""
        history = self.state_history.get(entity_id, [])
        if len(history) < 2:
            return {"temporal": 1.0, "spatial": 1.0, "logical": 1.0}

        # 计算时间一致性
        temporal_consistency = self._calculate_temporal_consistency_score(history)

        # 计算空间一致性
        spatial_consistency = self._calculate_spatial_consistency_score(history)

        # 计算逻辑一致性
        logical_consistency = self._calculate_logical_consistency_score(history)

        return {
            "temporal": temporal_consistency,
            "spatial": spatial_consistency,
            "logical": logical_consistency
        }

    def _calculate_temporal_consistency_score(self, history: List[Dict]) -> float:
        """计算时间一致性分数"""
        if len(history) < 2:
            return 1.0

        time_diffs = []
        for i in range(1, len(history)):
            diff = (history[i]["timestamp"] - history[i - 1]["timestamp"]).total_seconds()
            time_diffs.append(diff)

        if not time_diffs:
            return 1.0

        # 时间间隔的稳定性
        mean_diff = np.mean(time_diffs)
        std_diff = np.std(time_diffs)

        if mean_diff == 0:
            return 1.0

        stability = 1.0 / (1.0 + std_diff / mean_diff)
        return min(max(stability, 0), 1)

    def _calculate_spatial_consistency_score(self, history: List[Dict]) -> float:
        """计算空间一致性分数"""
        positions = []
        for record in history:
            state = record["state"]
            if "position" in state and isinstance(state["position"], (list, tuple)):
                if len(state["position"]) >= 2:
                    positions.append((state["position"][0], state["position"][1]))

        if len(positions) < 2:
            return 1.0

        # 计算移动平滑度
        distances = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            distances.append(math.sqrt(dx * dx + dy * dy))

        if not distances:
            return 1.0

        # 距离变化的平滑度
        distance_changes = []
        for i in range(1, len(distances)):
            distance_changes.append(abs(distances[i] - distances[i - 1]))

        if not distance_changes:
            return 1.0

        mean_change = np.mean(distance_changes)
        max_distance = max(distances) if distances else 1.0

        smoothness = 1.0 / (1.0 + mean_change / max_distance)
        return min(max(smoothness, 0), 1)

    def _calculate_logical_consistency_score(self, history: List[Dict]) -> float:
        """计算逻辑一致性分数"""
        if len(history) < 2:
            return 1.0

        logical_errors = 0

        for i in range(1, len(history)):
            prev_state = history[i - 1]["state"]
            curr_state = history[i]["state"]

            # 检查逻辑矛盾
            if self._has_logical_contradiction(prev_state, curr_state):
                logical_errors += 1

        consistency = 1.0 - (logical_errors / (len(history) - 1))
        return max(consistency, 0)

    def _has_logical_contradiction(self, prev_state: Dict, curr_state: Dict) -> bool:
        """检查逻辑矛盾"""
        # 简化的逻辑检查
        return False

    def detect_anomalies(self, entity_id: str, window_size: int = 10) -> List[Dict]:
        """检测异常状态"""
        if entity_id not in self.state_history:
            return []

        history = self.state_history[entity_id]
        if len(history) < window_size:
            return []

        anomalies = []

        # 分析最近的状态序列
        recent_states = history[-window_size:]

        # 检测突变
        for i in range(1, len(recent_states)):
            prev_state = recent_states[i - 1]["state"]
            curr_state = recent_states[i]["state"]

            changes = self._detect_state_changes(prev_state, curr_state,
                                                 self.entity_registry[entity_id]["entity_type"])

            if changes["temporal_consistency"] == TemporalConsistency.BROKEN:
                anomalies.append({
                    "timestamp": recent_states[i]["timestamp"],
                    "type": "temporal_discontinuity",
                    "severity": "high",
                    "description": "时间连续性断裂",
                    "changes": changes
                })

            # 检查位置突变
            if "position" in changes["changed_attributes"]:
                magnitude = changes["change_magnitudes"].get("position", 0)
                if isinstance(magnitude, (int, float)) and magnitude > 5.0:
                    anomalies.append({
                        "timestamp": recent_states[i]["timestamp"],
                        "type": "position_jump",
                        "severity": "medium",
                        "description": f"位置突变: {magnitude:.1f}米",
                        "changes": changes
                    })

        return anomalies

    def generate_continuity_report(self, entity_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """生成连续性报告"""
        if entity_ids is None:
            entity_ids = list(self.entity_registry.keys())

        report = {
            "generation_time": datetime.now(),
            "entities_analyzed": len(entity_ids),
            "overall_continuity_score": 0.0,
            "entity_reports": [],
            "cross_entity_consistency": {},
            "recommendations": []
        }

        total_score = 0.0
        for entity_id in entity_ids:
            entity_report = self._generate_entity_continuity_report(entity_id)
            report["entity_reports"].append(entity_report)

            if "overall_score" in entity_report:
                total_score += entity_report["overall_score"]

        if entity_ids:
            report["overall_continuity_score"] = total_score / len(entity_ids)

        # 生成跨实体一致性分析
        report["cross_entity_consistency"] = self._analyze_cross_entity_consistency(entity_ids)

        # 生成建议
        report["recommendations"] = self._generate_continuity_recommendations(report)

        return report

    def _generate_entity_continuity_report(self, entity_id: str) -> Dict[str, Any]:
        """生成实体连续性报告"""
        if entity_id not in self.entity_registry:
            return {"error": "Entity not found"}

        entity = self.entity_registry[entity_id]
        history = self.state_history.get(entity_id, [])

        if len(history) < 2:
            return {
                "entity_id": entity_id,
                "state_count": len(history),
                "insufficient_data": True
            }

        # 计算各项指标
        consistency_scores = self._calculate_overall_consistency(entity_id)

        # 检测异常
        anomalies = self.detect_anomalies(entity_id)

        # 计算总体分数
        overall_score = (consistency_scores["temporal"] * 0.4 +
                         consistency_scores["spatial"] * 0.4 +
                         consistency_scores["logical"] * 0.2)

        report = {
            "entity_id": entity_id,
            "entity_type": entity["entity_type"],
            "state_count": len(history),
            "change_count": len(entity["change_log"]),
            "consistency_scores": consistency_scores,
            "overall_score": overall_score,
            "anomalies_detected": len(anomalies),
            "recent_anomalies": anomalies[-3:] if anomalies else [],
            "temporal_pattern": self._analyze_temporal_pattern(history),
            "stability_assessment": self._assess_stability(history)
        }

        return report

    def _analyze_temporal_pattern(self, history: List[Dict]) -> Dict[str, Any]:
        """分析时间模式"""
        if len(history) < 3:
            return {"pattern": "insufficient_data"}

        # 分析状态变化频率
        time_diffs = []
        for i in range(1, len(history)):
            diff = (history[i]["timestamp"] - history[i - 1]["timestamp"]).total_seconds()
            time_diffs.append(diff)

        pattern = {
            "update_frequency_mean": np.mean(time_diffs) if time_diffs else 0,
            "update_frequency_std": np.std(time_diffs) if time_diffs else 0,
            "pattern_type": "regular" if len(time_diffs) > 2 and np.std(time_diffs) < 1.0 else "irregular"
        }

        return pattern

    def _assess_stability(self, history: List[Dict]) -> Dict[str, Any]:
        """评估稳定性"""
        if len(history) < 2:
            return {"stability": "unknown", "confidence": 0.0}

        # 计算状态变化率
        change_counts = []
        for i in range(1, len(history)):
            changes = self._detect_state_changes(history[i - 1]["state"], history[i]["state"], "unknown")
            change_counts.append(len(changes["changed_attributes"]))

        mean_changes = np.mean(change_counts) if change_counts else 0
        std_changes = np.std(change_counts) if change_counts else 0

        if mean_changes == 0:
            stability = "perfect"
            confidence = 1.0
        elif mean_changes < 2 and std_changes < 1:
            stability = "high"
            confidence = 0.8
        elif mean_changes < 5:
            stability = "moderate"
            confidence = 0.6
        else:
            stability = "low"
            confidence = 0.4

        return {
            "stability": stability,
            "confidence": confidence,
            "mean_changes_per_update": mean_changes,
            "change_consistency": 1.0 / (1.0 + std_changes) if std_changes > 0 else 1.0
        }

    def _analyze_cross_entity_consistency(self, entity_ids: List[str]) -> Dict[str, Any]:
        """分析跨实体一致性"""
        if len(entity_ids) < 2:
            return {"analysis": "insufficient_entities"}

        analysis = {
            "entity_pairs_analyzed": 0,
            "consistent_pairs": 0,
            "inconsistent_pairs": 0,
            "pairwise_consistency_scores": {},
            "global_inconsistencies": []
        }

        # 分析实体对
        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                entity1 = entity_ids[i]
                entity2 = entity_ids[j]

                pair_score = self._calculate_pairwise_consistency(entity1, entity2)
                analysis["pairwise_consistency_scores"][f"{entity1}_{entity2}"] = pair_score
                analysis["entity_pairs_analyzed"] += 1

                if pair_score > 0.7:
                    analysis["consistent_pairs"] += 1
                else:
                    analysis["inconsistent_pairs"] += 1

        return analysis

    def _calculate_pairwise_consistency(self, entity1: str, entity2: str) -> float:
        """计算成对一致性"""
        # 简化的成对一致性计算
        return 0.8

    def _generate_continuity_recommendations(self, report: Dict) -> List[str]:
        """生成连续性建议"""
        recommendations = []

        for entity_report in report.get("entity_reports", []):
            if "overall_score" in entity_report:
                score = entity_report["overall_score"]

                if score < 0.5:
                    recommendations.append(
                        f"实体 {entity_report.get('entity_id')} 连续性较差，建议检查状态更新逻辑"
                    )
                elif score < 0.7:
                    recommendations.append(
                        f"实体 {entity_report.get('entity_id')} 连续性一般，建议优化状态管理"
                    )

        if report.get("overall_continuity_score", 1.0) < 0.6:
            recommendations.append("整体连续性较差，建议进行系统性优化")

        return recommendations
