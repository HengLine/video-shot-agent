"""
@FileName: continuity_guard_guardian.py
@Description: 
@Author: HengLine
@Time: 2026/1/5 15:44
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class GuardMode(Enum):
    """守护模式"""
    REACTIVE = "reactive"  # 反应式：检测到问题后修复
    PROACTIVE = "proactive"  # 主动式：预测并预防问题
    ADAPTIVE = "adaptive"  # 自适应：根据情况调整模式
    PASSIVE = "passive"  # 被动式：仅检测不修复


class SceneComplexity(Enum):
    """场景复杂度"""
    SIMPLE = "simple"  # 简单场景：1-3角色，少量道具
    MODERATE = "moderate"  # 中等场景：4-6角色，中等道具
    COMPLEX = "complex"  # 复杂场景：7+角色，大量道具
    EPIC = "epic"  # 史诗场景：大规模场景，特效密集


class AnalysisDepth(Enum):
    """分析深度"""
    QUICK = "quick"  # 快速分析：仅关键元素
    STANDARD = "standard"  # 标准分析：主要元素
    DETAILED = "detailed"  # 详细分析：所有元素
    EXHAUSTIVE = "exhaustive"  # 详尽分析：包括微观细节


@dataclass
class GuardianConfig:
    """连续性守护器配置"""
    # 基础配置
    task_id: str = "123"
    mode: GuardMode = GuardMode.ADAPTIVE
    analysis_depth: AnalysisDepth = AnalysisDepth.STANDARD
    enable_auto_fix: bool = True
    max_auto_fix_attempts: int = 3

    enable_state_tracking: bool = True
    enable_learning: bool = False
    strict_mode: bool = False
    max_issues_per_scene: int = 50
    check_interval_frames: int = 1

    # 性能配置
    max_state_history: int = 1000
    cache_size: int = 100
    parallel_processing: bool = True
    max_workers: int = 4

    # 检测配置
    detection_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "critical": 0.9,  # 关键问题检测阈值
        "major": 0.7,  # 主要问题检测阈值
        "minor": 0.5,  # 次要问题检测阈值
        "cosmetic": 0.3  # 外观问题检测阈值
    })

    # 验证配置
    validation_frequency: int = 10  # 每N帧验证一次
    enable_real_time_validation: bool = True
    validation_timeout: float = 5.0  # 验证超时时间（秒）

    # 报告配置
    generate_reports: bool = True
    report_format: str = "json"  # json, html, text
    auto_save_reports: bool = True
    report_save_path: str = "./continuity_reports"

    # 学习配置
    enable_machine_learning: bool = False
    learn_from_mistakes: bool = True
    pattern_recognition: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "mode": self.mode.value,
            "analysis_depth": self.analysis_depth.value,
            "enable_auto_fix": self.enable_auto_fix,
            "max_state_history": self.max_state_history,
            "detection_thresholds": self.detection_thresholds
        }
