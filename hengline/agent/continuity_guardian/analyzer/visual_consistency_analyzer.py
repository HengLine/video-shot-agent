"""
@FileName: visual_consistency_analyzer.py
@Description: 
@Author: HengLine
@Time: 2026/1/4 18:06
"""
import hashlib
import json
from typing import Dict, List, Any, Tuple, Union

import numpy as np


class VisualConsistencyAnalyzer:
    """视觉一致性分析器"""

    def __init__(self):
        self.feature_cache: Dict[str, Dict[str, Any]] = {}
        self.similarity_thresholds = {
            "character": 0.85,
            "prop": 0.75,
            "environment": 0.90,
            "style": 0.80
        }

    def extract_visual_features(self, image_data: Union[str, np.ndarray]) -> Dict[str, Any]:
        """提取视觉特征"""
        # 实际实现中这里会使用CV/NLP技术
        features = {
            "color_histogram": self._compute_color_histogram(image_data),
            "texture_features": self._extract_texture_features(image_data),
            "edge_density": self._compute_edge_density(image_data),
            "composition_analysis": self._analyze_composition(image_data),
            "dominant_colors": self._get_dominant_colors(image_data),
            "brightness_level": self._compute_brightness(image_data),
            "contrast_level": self._compute_contrast(image_data),
            "key_visual_elements": self._detect_visual_elements(image_data)
        }

        # 生成特征哈希
        feature_hash = self._generate_feature_hash(features)
        self.feature_cache[feature_hash] = features

        return features

    def compare_frames(self, frame1_features: Dict[str, Any],
                       frame2_features: Dict[str, Any]) -> Dict[str, float]:
        """比较两帧的视觉特征"""
        comparison_results = {
            "color_similarity": self._compare_color_histograms(
                frame1_features.get("color_histogram", {}),
                frame2_features.get("color_histogram", {})
            ),
            "texture_similarity": self._compare_texture_features(
                frame1_features.get("texture_features", {}),
                frame2_features.get("texture_features", {})
            ),
            "composition_similarity": self._compare_composition(
                frame1_features.get("composition_analysis", {}),
                frame2_features.get("composition_analysis", {})
            ),
            "brightness_difference": abs(
                frame1_features.get("brightness_level", 0) -
                frame2_features.get("brightness_level", 0)
            ),
            "contrast_difference": abs(
                frame1_features.get("contrast_level", 0) -
                frame2_features.get("contrast_level", 0)
            ),
            "element_matching": self._match_visual_elements(
                frame1_features.get("key_visual_elements", []),
                frame2_features.get("key_visual_elements", [])
            )
        }

        # 计算总体相似度分数
        comparison_results["overall_similarity"] = self._calculate_overall_similarity(comparison_results)

        return comparison_results

    def _compute_color_histogram(self, image_data: Any) -> Dict[str, List[float]]:
        """计算颜色直方图（简化实现）"""
        # 实际实现中使用OpenCV
        return {"histogram": [0.1, 0.2, 0.3, 0.4], "bins": 256}

    def _extract_texture_features(self, image_data: Any) -> Dict[str, float]:
        """提取纹理特征"""
        return {
            "smoothness": 0.7,
            "regularity": 0.5,
            "directionality": 0.3,
            "contrast": 0.6
        }

    def _compute_edge_density(self, image_data: Any) -> float:
        """计算边缘密度"""
        return 0.25

    def _analyze_composition(self, image_data: Any) -> Dict[str, Any]:
        """分析构图"""
        return {
            "rule_of_thirds": True,
            "symmetry_score": 0.8,
            "balance_score": 0.7,
            "focal_points": [{"x": 0.5, "y": 0.5, "strength": 0.9}]
        }

    def _get_dominant_colors(self, image_data: Any) -> List[Tuple[int, int, int]]:
        """获取主色调"""
        return [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    def _compute_brightness(self, image_data: Any) -> float:
        """计算亮度"""
        return 0.65

    def _compute_contrast(self, image_data: Any) -> float:
        """计算对比度"""
        return 0.45

    def _detect_visual_elements(self, image_data: Any) -> List[Dict[str, Any]]:
        """检测视觉元素"""
        return [
            {"type": "face", "position": (0.3, 0.3), "confidence": 0.95},
            {"type": "object", "position": (0.7, 0.7), "confidence": 0.85}
        ]

    def _generate_feature_hash(self, features: Dict[str, Any]) -> str:
        """生成特征哈希"""
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()

    def _compare_color_histograms(self, hist1: Dict, hist2: Dict) -> float:
        """比较颜色直方图"""
        # 简化实现
        return 0.95 if hist1 and hist2 else 0.0

    def _compare_texture_features(self, tex1: Dict, tex2: Dict) -> float:
        """比较纹理特征"""
        if not tex1 or not tex2:
            return 0.0
        similarities = []
        for key in tex1:
            if key in tex2:
                similarities.append(1.0 - abs(tex1[key] - tex2[key]))
        return np.mean(similarities) if similarities else 0.0

    def _compare_composition(self, comp1: Dict, comp2: Dict) -> float:
        """比较构图"""
        if not comp1 or not comp2:
            return 0.0
        return 0.9 if comp1.get("rule_of_thirds") == comp2.get("rule_of_thirds") else 0.5

    def _match_visual_elements(self, elements1: List, elements2: List) -> float:
        """匹配视觉元素"""
        if not elements1 or not elements2:
            return 0.0
        matches = 0
        for e1 in elements1:
            for e2 in elements2:
                if e1.get("type") == e2.get("type"):
                    matches += 1
                    break
        return matches / max(len(elements1), len(elements2))

    def _calculate_overall_similarity(self, comparison_results: Dict[str, float]) -> float:
        """计算总体相似度"""
        weights = {
            "color_similarity": 0.3,
            "texture_similarity": 0.2,
            "composition_similarity": 0.25,
            "element_matching": 0.25
        }

        weighted_sum = 0
        total_weight = 0

        for metric, weight in weights.items():
            if metric in comparison_results:
                weighted_sum += comparison_results[metric] * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def analyze_visual_continuity(self, frame_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析视觉连续性"""
        results = []

        for i in range(len(frame_sequence) - 1):
            frame1 = frame_sequence[i]
            frame2 = frame_sequence[i + 1]

            # 提取特征
            features1 = self.extract_visual_features(frame1.get("image_data", ""))
            features2 = self.extract_visual_features(frame2.get("image_data", ""))

            # 比较特征
            comparison = self.compare_frames(features1, features2)

            # 评估连续性
            continuity_assessment = self._assess_continuity(
                comparison,
                frame1.get("context", {}),
                frame2.get("context", {})
            )

            results.append({
                "frame_pair": (i, i + 1),
                "similarity_scores": comparison,
                "continuity_assessment": continuity_assessment,
                "issues": continuity_assessment.get("issues", [])
            })

        return results

    def _assess_continuity(self, comparison: Dict[str, float],
                           context1: Dict, context2: Dict) -> Dict[str, Any]:
        """评估连续性"""
        issues = []

        # 检查颜色相似度
        if comparison.get("color_similarity", 0) < self.similarity_thresholds["style"]:
            issues.append({
                "type": "color_inconsistency",
                "severity": "medium",
                "description": "颜色调性不一致",
                "score": comparison.get("color_similarity", 0)
            })

        # 检查构图相似度
        if comparison.get("composition_similarity", 0) < self.similarity_thresholds["style"]:
            issues.append({
                "type": "composition_inconsistency",
                "severity": "high",
                "description": "构图风格不一致",
                "score": comparison.get("composition_similarity", 0)
            })

        # 检查亮度差异
        if comparison.get("brightness_difference", 0) > 0.3:
            issues.append({
                "type": "brightness_jump",
                "severity": "low",
                "description": "亮度跳跃过大",
                "difference": comparison.get("brightness_difference", 0)
            })

        return {
            "overall_continuity_score": comparison.get("overall_similarity", 0),
            "issues": issues,
            "is_acceptable": len([i for i in issues if i["severity"] in ["high", "critical"]]) == 0
        }
