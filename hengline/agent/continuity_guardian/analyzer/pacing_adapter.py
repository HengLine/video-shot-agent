"""
@FileName: pacing_adapter.py
@Description: 
@Author: HengLine
@Time: 2026/1/9 18:42
"""
from typing import Dict, Any

from hengline.agent.temporal_planner.temporal_planner_model import PacingAnalysis


class PacingAdapter:
    """节奏适配器"""

    def adapt_to_pacing(self, pacing_analysis: PacingAnalysis) -> Dict[str, Any]:
        """根据节奏调整分析策略"""
        pacing = pacing_analysis.overall_pacing

        adaptation = {
            "pacing_profile": pacing,
            "monitoring_strategy": self._get_monitoring_strategy(pacing),
            "validation_intensity": self._get_validation_intensity(pacing),
            "intervention_style": self._get_intervention_style(pacing),
            "resource_allocation": self._allocate_resources_by_pacing(pacing)
        }

        return adaptation

    def _get_monitoring_strategy(self, pacing: str) -> str:
        """获取监控策略"""
        strategies = {
            "slow": "comprehensive",  # 慢节奏：全面监控
            "medium": "balanced",  # 中节奏：平衡监控
            "fast": "focused"  # 快节奏：重点监控
        }

        return strategies.get(pacing, "balanced")

    def _get_validation_intensity(self, pacing: str) -> str:
        """获取验证强度"""
        intensities = {
            "slow": "detailed",  # 慢节奏：详细验证
            "medium": "standard",  # 中节奏：标准验证
            "fast": "quick"  # 快节奏：快速验证
        }

        return intensities.get(pacing, "standard")

    def _get_intervention_style(self, pacing: str) -> str:
        """获取干预风格"""
        styles = {
            "slow": "proactive",  # 慢节奏：主动干预
            "medium": "responsive",  # 中节奏：响应式干预
            "fast": "aggressive"  # 快节奏：积极干预
        }

        return styles.get(pacing, "responsive")

    def _allocate_resources_by_pacing(self, pacing: str) -> Dict[str, Any]:
        """根据节奏分配资源"""
        allocations = {
            "slow": {
                "cpu_priority": "normal",
                "memory_allocation": "standard",
                "processing_batch_size": "large",
                "real_time_requirements": "relaxed"
            },
            "medium": {
                "cpu_priority": "high",
                "memory_allocation": "generous",
                "processing_batch_size": "medium",
                "real_time_requirements": "moderate"
            },
            "fast": {
                "cpu_priority": "realtime",
                "memory_allocation": "maximum",
                "processing_batch_size": "small",
                "real_time_requirements": "strict"
            }
        }

        return allocations.get(pacing, allocations["medium"])
