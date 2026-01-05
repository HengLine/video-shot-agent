"""
@FileName: scene_analyzer.py
@Description: 
@Author: HengLine
@Time: 2026/1/5 15:45
"""
from typing import Dict, Any, List
from datetime import datetime
import math

from hengline.agent.continuity_guardian.model.continuity_guard_guardian import AnalysisDepth, SceneComplexity, GuardianConfig


class SceneAnalyzer:
    """场景分析器"""

    def __init__(self, config: GuardianConfig):
        self.config = config
        self.metrics_history: List[Dict] = []

    def analyze_scene_complexity(self, scene_data: Dict) -> SceneComplexity:
        """分析场景复杂度"""
        # 统计角色和道具数量
        character_count = len(scene_data.get("characters", []))
        prop_count = len(scene_data.get("props", []))
        effect_count = len(scene_data.get("effects", []))

        # 计算复杂度分数
        complexity_score = (
                character_count * 0.4 +
                prop_count * 0.3 +
                effect_count * 0.3
        )

        # 判断复杂度级别
        if complexity_score < 2:
            return SceneComplexity.SIMPLE
        elif complexity_score < 5:
            return SceneComplexity.MODERATE
        elif complexity_score < 10:
            return SceneComplexity.COMPLEX
        else:
            return SceneComplexity.EPIC

    def calculate_scene_metrics(self, scene_data: Dict) -> Dict[str, Any]:
        """计算场景指标"""
        metrics = {
            "timestamp": datetime.now(),
            "character_count": len(scene_data.get("characters", [])),
            "prop_count": len(scene_data.get("props", [])),
            "environment_complexity": self._calculate_env_complexity(scene_data.get("environment", {})),
            "motion_intensity": self._calculate_motion_intensity(scene_data),
            "lighting_complexity": self._calculate_lighting_complexity(scene_data),
            "camera_movement": self._analyze_camera_movement(scene_data.get("camera", {})),
            "estimated_processing_time": self._estimate_processing_time(scene_data)
        }

        # 记录历史
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.config.max_state_history:
            self.metrics_history.pop(0)

        return metrics

    def _calculate_env_complexity(self, environment: Dict) -> float:
        """计算环境复杂度"""
        complexity = 0.0

        # 光照复杂度
        if "lighting" in environment:
            lights = environment["lighting"].get("lights", [])
            complexity += len(lights) * 0.2

        # 天气效果
        if "weather" in environment:
            weather = environment["weather"]
            if weather != "clear":
                complexity += 0.3

        # 特效
        if "effects" in environment:
            complexity += len(environment["effects"]) * 0.1

        return min(complexity, 1.0)

    def _calculate_motion_intensity(self, scene_data: Dict) -> float:
        """计算运动强度"""
        intensity = 0.0

        # 角色运动
        for character in scene_data.get("characters", []):
            if "velocity" in character:
                velocity = character["velocity"]
                if isinstance(velocity, (list, tuple)) and len(velocity) >= 3:
                    speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2)
                    intensity += min(speed / 10.0, 0.5)  # 标准化

        # 相机运动
        camera = scene_data.get("camera", {})
        if "movement" in camera:
            movement = camera["movement"]
            if movement != "static":
                intensity += 0.3

        return min(intensity, 1.0)

    def _calculate_lighting_complexity(self, scene_data: Dict) -> float:
        """计算光照复杂度"""
        environment = scene_data.get("environment", {})
        lighting = environment.get("lighting", {})

        complexity = 0.0

        # 光源数量
        lights = lighting.get("lights", [])
        complexity += len(lights) * 0.1

        # 动态光照
        if lighting.get("dynamic", False):
            complexity += 0.3

        # 阴影质量
        shadows = lighting.get("shadows", {})
        if shadows.get("enabled", False):
            complexity += 0.2

        return min(complexity, 1.0)

    def _analyze_camera_movement(self, camera: Dict) -> Dict[str, Any]:
        """分析相机运动"""
        analysis = {
            "type": camera.get("movement", "static"),
            "stability": 1.0,
            "smoothness": 1.0
        }

        if "velocity" in camera:
            velocity = camera["velocity"]
            if isinstance(velocity, (list, tuple)) and len(velocity) >= 3:
                speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2)
                analysis["speed"] = speed
                analysis["stability"] = 1.0 / (1.0 + speed * 10)

        return analysis

    def _estimate_processing_time(self, scene_data: Dict) -> float:
        """估计处理时间"""
        base_time = 0.1  # 基础处理时间

        # 根据复杂度调整
        complexity = self.analyze_scene_complexity(scene_data)
        complexity_multiplier = {
            SceneComplexity.SIMPLE: 1.0,
            SceneComplexity.MODERATE: 1.5,
            SceneComplexity.COMPLEX: 2.5,
            SceneComplexity.EPIC: 4.0
        }

        # 根据分析深度调整
        depth_multiplier = {
            AnalysisDepth.QUICK: 0.5,
            AnalysisDepth.STANDARD: 1.0,
            AnalysisDepth.DETAILED: 2.0,
            AnalysisDepth.EXHAUSTIVE: 4.0
        }

        estimated_time = base_time * \
                         complexity_multiplier[complexity] * \
                         depth_multiplier[self.config.analysis_depth]

        return estimated_time