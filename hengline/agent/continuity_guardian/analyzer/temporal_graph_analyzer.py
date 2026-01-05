"""
@FileName: temporal_graph_analyzer.py
@Description: 时间图 - 用于时间序列分析和关系建模
@Author: HengLine
@Time: 2026/1/4 22:16
"""
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np


class TemporalGraph:
    """时间图 - 用于时间序列分析和关系建模"""

    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: List[Dict] = []
        self.entity_timelines: Dict[str, List[int]] = defaultdict(list)  # 实体ID到事件索引的映射
        self.time_index: List[datetime] = []
        self.event_types: Set[str] = set()
        self.relationships: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)  # 事件间关系

    def add_event(self, entity_id: str, timestamp: datetime, state: Dict,
                  event_type: str = "state_update") -> int:
        """添加事件"""
        if len(self.events) >= self.max_events:
            # 移除最旧的事件
            self._remove_oldest_event()

        event_index = len(self.events)

        event = {
            "index": event_index,
            "entity_id": entity_id,
            "timestamp": timestamp,
            "state": state.copy(),
            "event_type": event_type,
            "relationships": []
        }

        self.events.append(event)
        self.entity_timelines[entity_id].append(event_index)
        self.time_index.append(timestamp)
        self.event_types.add(event_type)

        return event_index

    def _remove_oldest_event(self):
        """移除最旧的事件"""
        if not self.events:
            return

        # 找到时间最早的事件
        oldest_index = 0
        oldest_time = self.time_index[0]

        for i, timestamp in enumerate(self.time_index):
            if timestamp < oldest_time:
                oldest_time = timestamp
                oldest_index = i

        # 移除事件
        removed_event = self.events.pop(oldest_index)
        removed_time = self.time_index.pop(oldest_index)

        # 更新实体时间线
        entity_id = removed_event["entity_id"]
        if entity_id in self.entity_timelines:
            self.entity_timelines[entity_id].remove(oldest_index)
            # 更新后续事件的索引
            for i in range(len(self.entity_timelines[entity_id])):
                if self.entity_timelines[entity_id][i] > oldest_index:
                    self.entity_timelines[entity_id][i] -= 1

        # 更新所有其他实体的时间线
        for other_entity, indices in self.entity_timelines.items():
            if other_entity != entity_id:
                for i in range(len(indices)):
                    if indices[i] > oldest_index:
                        indices[i] -= 1

        # 更新关系
        self._update_relationships_after_removal(oldest_index)

        # 更新后续事件的索引
        for i in range(oldest_index, len(self.events)):
            self.events[i]["index"] = i

    def _update_relationships_after_removal(self, removed_index: int):
        """移除事件后更新关系"""
        # 移除包含被删除事件的关系
        for rel_type in list(self.relationships.keys()):
            new_rels = []
            for rel in self.relationships[rel_type]:
                if rel[0] != removed_index and rel[1] != removed_index:
                    # 调整索引
                    new_rel = (
                        rel[0] if rel[0] < removed_index else rel[0] - 1,
                        rel[1] if rel[1] < removed_index else rel[1] - 1,
                        rel[2]
                    )
                    new_rels.append(new_rel)
            self.relationships[rel_type] = new_rels

        # 更新事件中的关系
        for event in self.events:
            if event["index"] > removed_index:
                event["index"] -= 1

    def add_relationship(self, from_event_index: int, to_event_index: int,
                         rel_type: str, strength: float = 1.0) -> bool:
        """添加事件间关系"""
        if (from_event_index < 0 or from_event_index >= len(self.events) or
                to_event_index < 0 or to_event_index >= len(self.events)):
            return False

        # 添加全局关系
        self.relationships[rel_type].append((from_event_index, to_event_index, strength))

        # 添加到事件的关系列表
        self.events[from_event_index]["relationships"].append({
            "to_event": to_event_index,
            "type": rel_type,
            "strength": strength
        })

        return True

    def get_entity_events(self, entity_id: str,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> List[Dict]:
        """获取实体的所有事件"""
        if entity_id not in self.entity_timelines:
            return []

        event_indices = self.entity_timelines[entity_id]
        events = [self.events[i] for i in event_indices]

        # 时间过滤
        if start_time or end_time:
            filtered = []
            for event in events:
                event_time = event["timestamp"]

                if start_time and event_time < start_time:
                    continue
                if end_time and event_time > end_time:
                    continue

                filtered.append(event)
            events = filtered

        return events

    def get_events_in_time_range(self, start_time: datetime,
                                 end_time: datetime) -> List[Dict]:
        """获取时间范围内的所有事件"""
        events = []

        for event in self.events:
            event_time = event["timestamp"]
            if start_time <= event_time <= end_time:
                events.append(event)

        return events

    def find_correlated_events(self, event_index: int,
                               max_time_gap: timedelta = timedelta(seconds=10),
                               min_correlation: float = 0.5) -> List[Dict]:
        """查找相关事件"""
        if event_index < 0 or event_index >= len(self.events):
            return []

        source_event = self.events[event_index]
        source_time = source_event["timestamp"]
        source_state = source_event["state"]

        correlated = []

        # 查找时间接近的事件
        for other_event in self.events:
            if other_event["index"] == event_index:
                continue

            time_diff = abs((other_event["timestamp"] - source_time).total_seconds())
            if time_diff > max_time_gap.total_seconds():
                continue

            # 计算状态相似度
            similarity = self._calculate_state_similarity(source_state, other_event["state"])

            if similarity >= min_correlation:
                correlated.append({
                    "event": other_event,
                    "time_difference": time_diff,
                    "similarity": similarity,
                    "relationship_strength": self._get_relationship_strength(
                        event_index, other_event["index"]
                    )
                })

        # 按相关性排序
        correlated.sort(key=lambda x: x["similarity"], reverse=True)

        return correlated

    def _calculate_state_similarity(self, state1: Dict, state2: Dict) -> float:
        """计算状态相似度"""
        if not state1 or not state2:
            return 0.0

        # 获取所有键
        all_keys = set(state1.keys()) | set(state2.keys())
        if not all_keys:
            return 1.0

        similarities = []

        for key in all_keys:
            val1 = state1.get(key)
            val2 = state2.get(key)

            if val1 is None or val2 is None:
                # 一个状态有值，一个没有
                similarities.append(0.0)
                continue

            # 根据类型计算相似度
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 数值相似度
                if val1 == 0 and val2 == 0:
                    similarity = 1.0
                else:
                    max_val = max(abs(val1), abs(val2))
                    similarity = 1.0 - abs(val1 - val2) / max_val
            elif isinstance(val1, str) and isinstance(val2, str):
                # 字符串相似度（简化版）
                similarity = 1.0 if val1 == val2 else 0.0
            elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                # 序列相似度
                if len(val1) != len(val2):
                    similarity = 0.0
                else:
                    # 计算平均相似度
                    item_similarities = []
                    for v1, v2 in zip(val1, val2):
                        # 递归计算
                        item_similarities.append(
                            self._calculate_state_similarity(
                                {"value": v1}, {"value": v2}
                            )
                        )
                    similarity = np.mean(item_similarities) if item_similarities else 0.0
            elif isinstance(val1, dict) and isinstance(val2, dict):
                # 字典相似度（递归）
                similarity = self._calculate_state_similarity(val1, val2)
            else:
                # 类型不匹配
                similarity = 0.0

            similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def _get_relationship_strength(self, event_index1: int, event_index2: int) -> float:
        """获取事件间关系强度"""
        max_strength = 0.0

        for rel_type, relationships in self.relationships.items():
            for from_idx, to_idx, strength in relationships:
                if (from_idx == event_index1 and to_idx == event_index2) or \
                        (from_idx == event_index2 and to_idx == event_index1):
                    max_strength = max(max_strength, strength)

        return max_strength

    def analyze_temporal_patterns(self, entity_id: Optional[str] = None) -> Dict[str, Any]:
        """分析时间模式"""
        if entity_id:
            events = self.get_entity_events(entity_id)
        else:
            events = self.events

        if len(events) < 2:
            return {"error": "事件数量不足"}

        # 计算事件间隔
        intervals = []
        events_sorted = sorted(events, key=lambda x: x["timestamp"])

        for i in range(1, len(events_sorted)):
            interval = (events_sorted[i]["timestamp"] -
                        events_sorted[i - 1]["timestamp"]).total_seconds()
            intervals.append(interval)

        time_span = (events_sorted[-1]["timestamp"] -
                          events_sorted[0]["timestamp"]).total_seconds()
        # 分析模式
        analysis = {
            "total_events": len(events),
            "time_span": time_span,
            "interval_stats": {
                "mean": np.mean(intervals) if intervals else 0,
                "std": np.std(intervals) if intervals else 0,
                "min": min(intervals) if intervals else 0,
                "max": max(intervals) if intervals else 0
            },
            "event_frequency": len(events) / max(1, time_span),
            "regularity_score": self._calculate_regularity_score(intervals),
            "event_type_distribution": defaultdict(int)
        }

        # 统计事件类型分布
        for event in events:
            analysis["event_type_distribution"][event["event_type"]] += 1

        analysis["event_type_distribution"] = dict(analysis["event_type_distribution"])

        # 检测周期性
        analysis["periodicity"] = self._detect_periodicity(intervals)

        return analysis

    def _calculate_regularity_score(self, intervals: List[float]) -> float:
        """计算规律性分数"""
        if len(intervals) < 2:
            return 1.0

        # 计算间隔的变异系数
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        if mean_interval > 0:
            cv = std_interval / mean_interval  # 变异系数
            # 变异系数越小越规律
            regularity = 1.0 / (1.0 + cv)
        else:
            regularity = 1.0

        return max(0.0, min(1.0, regularity))

    def _detect_periodicity(self, intervals: List[float]) -> Dict[str, Any]:
        """检测周期性"""
        if len(intervals) < 5:
            return {"detected": False, "period": 0, "confidence": 0}

        try:
            # 使用自相关检测周期性
            from scipy import signal

            # 标准化间隔
            intervals_norm = (intervals - np.mean(intervals)) / (np.std(intervals) + 1e-10)

            # 计算自相关
            autocorr = signal.correlate(intervals_norm, intervals_norm, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]  # 取后半部分

            # 找到峰值（除零滞后外）
            peaks, properties = signal.find_peaks(autocorr[1:], height=0.3)

            if len(peaks) > 0:
                # 第一个峰值对应的滞后
                first_peak_lag = peaks[0] + 1
                period = first_peak_lag

                # 计算置信度
                peak_height = properties['peak_heights'][0]
                confidence = min(1.0, peak_height)

                return {
                    "detected": True,
                    "period": period,
                    "confidence": confidence,
                    "peak_heights": properties['peak_heights'].tolist()
                }
        except ImportError:
            pass  # scipy不可用

        return {"detected": False, "period": 0, "confidence": 0}

    def predict_next_event(self, entity_id: str,
                           current_time: datetime) -> Optional[Dict[str, Any]]:
        """预测下一个事件"""
        if entity_id not in self.entity_timelines:
            return None

        events = self.get_entity_events(entity_id)
        if len(events) < 2:
            return None

        # 计算平均间隔
        events_sorted = sorted(events, key=lambda x: x["timestamp"])
        intervals = []

        for i in range(1, len(events_sorted)):
            interval = (events_sorted[i]["timestamp"] -
                        events_sorted[i - 1]["timestamp"]).total_seconds()
            intervals.append(interval)

        avg_interval = np.mean(intervals)

        # 预测下一个时间
        last_event = events_sorted[-1]
        predicted_time = last_event["timestamp"] + timedelta(seconds=avg_interval)

        # 预测状态（使用最后一个状态）
        predicted_state = last_event["state"].copy()

        return {
            "entity_id": entity_id,
            "predicted_time": predicted_time,
            "predicted_state": predicted_state,
            "time_until_predicted": (predicted_time - current_time).total_seconds(),
            "confidence": self._calculate_regularity_score(intervals)
        }

    def get_temporal_graph_summary(self) -> str:
        """获取时间图摘要"""
        summary = "时间图摘要:\n"
        summary += "=" * 50 + "\n"
        summary += f"总事件数: {len(self.events)}\n"
        summary += f"实体数: {len(self.entity_timelines)}\n"
        summary += f"事件类型: {len(self.event_types)}\n"
        summary += f"关系类型: {len(self.relationships)}\n"

        if self.events:
            time_span = self.time_index[-1] - self.time_index[0]
            summary += f"时间跨度: {time_span}\n"
            summary += f"最早事件: {self.time_index[0]}\n"
            summary += f"最晚事件: {self.time_index[-1]}\n"

        # 事件类型统计
        summary += "\n事件类型分布:\n"
        type_counts = defaultdict(int)
        for event in self.events:
            type_counts[event["event_type"]] += 1

        for event_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.events)) * 100
            summary += f"  {event_type}: {count} ({percentage:.1f}%)\n"

        # 关系统计
        summary += "\n关系统计:\n"
        for rel_type, relationships in self.relationships.items():
            summary += f"  {rel_type}: {len(relationships)} 个关系\n"

        return summary

    def export_graph_data(self) -> Dict[str, Any]:
        """导出图数据"""
        return {
            "events": self.events,
            "entity_timelines": dict(self.entity_timelines),
            "relationships": dict(self.relationships),
            "metadata": {
                "total_events": len(self.events),
                "total_entities": len(self.entity_timelines),
                "time_range": {
                    "start": self.time_index[0].isoformat() if self.time_index else None,
                    "end": self.time_index[-1].isoformat() if self.time_index else None
                }
            }
        }
