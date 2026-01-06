"""
@FileName: visual_quality_assessor.py
@Description: 视觉质量评估器 - 评估视觉质量
@Author: HengLine
@Time: 2026/1/6 16:03
"""

from typing import List, Dict, Any, Optional

from .model.check_models import CheckStatus, VisualQualityCheckResult
from .model.issue_models import IssueSeverity


class VisualQualityAssessor:
    """视觉质量评估器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.composition_rules = self.config.get("composition_rules", {})
        self.lighting_standards = self.config.get("lighting_standards", {})
        self.color_standards = self.config.get("color_standards", {})

    def assess_all_shots(self, shots: List[Any]) -> List[VisualQualityCheckResult]:
        """评估所有镜头的视觉质量"""
        results = []

        for shot in shots:
            # 1. 构图质量评估
            composition_result = self.assess_composition(shot)
            if composition_result:
                results.append(composition_result)

            # 2. 灯光质量评估
            lighting_result = self.assess_lighting(shot)
            if lighting_result:
                results.append(lighting_result)

            # 3. 色彩质量评估
            color_result = self.assess_color(shot)
            if color_result:
                results.append(color_result)

            # 4. 风格一致性评估
            style_result = self.assess_style_consistency(shot, shots)
            if style_result:
                results.append(style_result)

        return results

    def assess_composition(self, shot: Any) -> VisualQualityCheckResult:
        """评估构图质量"""

        prompt = shot.full_sora_prompt.lower()

        # 检查构图关键词
        composition_keywords = {
            "rule_of_thirds": ["rule of thirds", "off-center", "balanced composition"],
            "leading_lines": ["leading lines", "lines leading", "converging lines"],
            "symmetry": ["symmetrical", "balanced", "centered composition"],
            "framing": ["framed by", "frame within frame", "natural frame"],
            "depth": ["depth of field", "layers", "foreground midground background"]
        }

        found_composition = []
        composition_issues = []
        composition_score = 0.5  # 基础分

        for comp_type, keywords in composition_keywords.items():
            for keyword in keywords:
                if keyword in prompt:
                    found_composition.append(comp_type)
                    composition_score += 0.1  # 每个好的构图元素加分
                    break

        # 检查构图问题
        problem_keywords = {
            "poor_framing": ["poorly framed", "awkward framing", "cut off"],
            "cluttered": ["cluttered", "busy", "too many elements"],
            "empty": ["empty space", "too much negative space"],
            "distracting": ["distracting", "competes with subject"]
        }

        for problem_type, keywords in problem_keywords.items():
            for keyword in keywords:
                if keyword in prompt:
                    composition_issues.append(problem_type)
                    composition_score -= 0.2  # 每个问题减分
                    break

        # 限制分数在0-1之间
        composition_score = max(0.0, min(1.0, composition_score))

        # 确定状态和严重性
        if composition_score >= 0.7:
            status = CheckStatus.PASSED
            severity = IssueSeverity.INFO
        elif composition_score >= 0.5:
            status = CheckStatus.WARNING
            severity = IssueSeverity.LOW
        else:
            status = CheckStatus.FAILED
            severity = IssueSeverity.MEDIUM

        # 生成建议
        suggestions = []
        if "rule_of_thirds" not in found_composition:
            suggestions.append("考虑使用三分法构图增强视觉平衡")
        if "depth" not in found_composition:
            suggestions.append("添加前景/中景/背景层次增强深度感")
        if "poor_framing" in composition_issues:
            suggestions.append("调整取景，避免切割重要元素")

        return VisualQualityCheckResult(
            check_id=f"composition_{shot.shot_id}",
            check_name="构图质量评估",
            check_description="评估镜头构图质量",
            status=status,
            severity=severity,
            score=composition_score,
            quality_dimension="composition",
            composition_score=composition_score,
            composition_issues=composition_issues,
            composition_suggestions=suggestions,
            details={
                "found_composition": found_composition,
                "composition_issues": composition_issues,
                "prompt_analysis": prompt[:100]
            },
            evidence=[
                f"构图元素: {len(found_composition)}个",
                f"构图问题: {len(composition_issues)}个"
            ] if composition_issues else ["构图良好"]
        )

    def assess_lighting(self, shot: Any) -> VisualQualityCheckResult:
        """评估灯光质量"""

        prompt = shot.full_sora_prompt.lower()

        # 检查灯光关键词
        lighting_keywords = {
            "good_lighting": [
                "natural lighting", "soft light", "diffused light",
                "golden hour", "cinematic lighting", "dramatic lighting",
                "key light", "fill light", "backlight"
            ],
            "lighting_problems": [
                "harsh shadows", "overexposed", "underexposed",
                "flat lighting", "unflattering light", "uneven lighting"
            ]
        }

        good_lighting_count = 0
        lighting_issues = []
        lighting_score = 0.5  # 基础分

        for keyword in lighting_keywords["good_lighting"]:
            if keyword in prompt:
                good_lighting_count += 1
                lighting_score += 0.05  # 每个好的灯光描述加分

        for keyword in lighting_keywords["lighting_problems"]:
            if keyword in prompt:
                lighting_issues.append(keyword)
                lighting_score -= 0.1  # 每个问题减分

        # 检查灯光描述完整性
        has_time_of_day = any(word in prompt for word in ["morning", "afternoon", "evening", "night", "day", "sunset"])
        has_light_quality = any(word in prompt for word in ["soft", "hard", "diffused", "direct", "indirect"])
        has_light_direction = any(word in prompt for word in ["from the left", "from the right", "front", "back", "side"])

        if has_time_of_day:
            lighting_score += 0.05
        if has_light_quality:
            lighting_score += 0.05
        if has_light_direction:
            lighting_score += 0.05

        # 限制分数在0-1之间
        lighting_score = max(0.0, min(1.0, lighting_score))

        # 确定状态和严重性
        if lighting_score >= 0.7:
            status = CheckStatus.PASSED
            severity = IssueSeverity.INFO
        elif lighting_score >= 0.5:
            status = CheckStatus.WARNING
            severity = IssueSeverity.LOW
        else:
            status = CheckStatus.FAILED
            severity = IssueSeverity.MEDIUM

        # 生成建议
        suggestions = []
        if good_lighting_count == 0:
            suggestions.append("添加灯光描述，如'自然光'、'电影感灯光'")
        if not has_time_of_day:
            suggestions.append("指定时间（早晨、午后、黄昏等）增强氛围")
        if not has_light_direction:
            suggestions.append("指定灯光方向（侧光、逆光等）增强立体感")

        return VisualQualityCheckResult(
            check_id=f"lighting_{shot.shot_id}",
            check_name="灯光质量评估",
            check_description="评估镜头灯光质量",
            status=status,
            severity=severity,
            score=lighting_score,
            quality_dimension="lighting",
            lighting_consistency_score=lighting_score,
            lighting_issues=lighting_issues,
            lighting_suggestions=suggestions,
            details={
                "good_lighting_count": good_lighting_count,
                "lighting_issues": lighting_issues,
                "has_time_of_day": has_time_of_day,
                "has_light_quality": has_light_quality,
                "has_light_direction": has_light_direction
            },
            evidence=[
                f"良好灯光描述: {good_lighting_count}个",
                f"灯光问题: {len(lighting_issues)}个"
            ] if lighting_issues else ["灯光描述充分"]
        )

    def assess_color(self, shot: Any) -> VisualQualityCheckResult:
        """评估色彩质量"""

        prompt = shot.full_sora_prompt.lower()

        # 检查色彩关键词
        color_keywords = {
            "color_descriptions": [
                "color palette", "color grading", "warm tones", "cool tones",
                "saturated", "desaturated", "vibrant", "muted", "contrast"
            ],
            "specific_colors": [
                "blue", "red", "green", "yellow", "orange", "purple",
                "teal", "gold", "silver", "black", "white", "gray"
            ]
        }

        color_descriptions = 0
        specific_colors = 0
        color_issues = []
        color_score = 0.5  # 基础分

        for keyword in color_keywords["color_descriptions"]:
            if keyword in prompt:
                color_descriptions += 1
                color_score += 0.03  # 每个色彩描述加分

        for keyword in color_keywords["specific_colors"]:
            if keyword in prompt:
                specific_colors += 1
                color_score += 0.02  # 每个具体颜色加分

        # 检查色彩问题
        if "clashing colors" in prompt or "color clash" in prompt:
            color_issues.append("clashing_colors")
            color_score -= 0.2

        if "washed out" in prompt or "dull colors" in prompt:
            color_issues.append("washed_out")
            color_score -= 0.15

        # 检查色彩一致性
        has_color_theme = any(word in prompt for word in ["color theme", "dominant color", "color scheme"])
        has_color_mood = any(word in prompt for word in ["warm mood", "cool mood", "color sets mood"])

        if has_color_theme:
            color_score += 0.1
        if has_color_mood:
            color_score += 0.05

        # 限制分数在0-1之间
        color_score = max(0.0, min(1.0, color_score))

        # 确定状态和严重性
        if color_score >= 0.7:
            status = CheckStatus.PASSED
            severity = IssueSeverity.INFO
        elif color_score >= 0.5:
            status = CheckStatus.WARNING
            severity = IssueSeverity.LOW
        else:
            status = CheckStatus.FAILED
            severity = IssueSeverity.MEDIUM

        # 生成建议
        suggestions = []
        if color_descriptions < 2:
            suggestions.append("添加更多色彩描述，如'色彩调色板'、'色彩分级'")
        if not has_color_theme:
            suggestions.append("指定主色调或色彩主题")
        if "clashing_colors" in color_issues:
            suggestions.append("调整色彩搭配，避免冲突色")

        return VisualQualityCheckResult(
            check_id=f"color_{shot.shot_id}",
            check_name="色彩质量评估",
            check_description="评估镜头色彩质量",
            status=status,
            severity=severity,
            score=color_score,
            quality_dimension="color",
            color_harmony_score=color_score,
            color_issues=color_issues,
            color_suggestions=suggestions,
            details={
                "color_descriptions": color_descriptions,
                "specific_colors": specific_colors,
                "color_issues": color_issues,
                "has_color_theme": has_color_theme,
                "has_color_mood": has_color_mood
            },
            evidence=[
                f"色彩描述: {color_descriptions}个",
                f"具体颜色: {specific_colors}个",
                f"色彩问题: {len(color_issues)}个"
            ]
        )

    def assess_style_consistency(self, shot: Any, all_shots: List[Any]) -> Optional[VisualQualityCheckResult]:
        """评估风格一致性"""

        # 收集所有镜头的风格特征
        all_style_features = []
        for s in all_shots:
            features = self._extract_style_features(s)
            all_style_features.append(features)

        # 计算当前镜头与平均风格的差异
        current_features = self._extract_style_features(shot)

        if not all_style_features:
            return None

        # 计算一致性得分
        consistency_score = self._calculate_style_consistency(current_features, all_style_features)

        # 确定状态和严重性
        if consistency_score >= 0.8:
            status = CheckStatus.PASSED
            severity = IssueSeverity.INFO
        elif consistency_score >= 0.6:
            status = CheckStatus.WARNING
            severity = IssueSeverity.LOW
        else:
            status = CheckStatus.FAILED
            severity = IssueSeverity.MEDIUM

        # 分析风格差异
        style_issues = []
        suggestions = []

        if consistency_score < 0.7:
            style_issues.append("风格不一致")
            suggestions.append("调整视觉风格以匹配其他镜头")

        # 检查具体风格元素
        prompt = shot.full_sora_prompt.lower()

        has_cinematic_style = "cinematic" in prompt or "film" in prompt
        has_realistic_style = "realistic" in prompt or "photorealistic" in prompt
        has_stylized_style = "stylized" in prompt or "artistic" in prompt

        style_elements = []
        if has_cinematic_style:
            style_elements.append("cinematic")
        if has_realistic_style:
            style_elements.append("realistic")
        if has_stylized_style:
            style_elements.append("stylized")

        # 如果没有任何风格描述
        if not style_elements and consistency_score < 0.9:
            suggestions.append("添加风格描述，如'电影感'、'写实'、'艺术风格'")

        return VisualQualityCheckResult(
            check_id=f"style_consistency_{shot.shot_id}",
            check_name="风格一致性评估",
            check_description="评估镜头风格一致性",
            status=status,
            severity=severity,
            score=consistency_score,
            quality_dimension="style",
            style_consistency_score=consistency_score,
            style_issues=style_issues,
            style_suggestions=suggestions,
            details={
                "consistency_score": consistency_score,
                "style_elements": style_elements,
                "total_shots": len(all_shots)
            },
            evidence=[f"风格一致性: {consistency_score:.2%}"] if consistency_score < 0.8 else ["风格一致"]
        )

    def _extract_style_features(self, shot: Any) -> Dict[str, Any]:
        """提取风格特征"""
        features = {}
        prompt = shot.full_sora_prompt.lower()

        # 提取关键词频率
        style_keywords = [
            "cinematic", "realistic", "stylized", "artistic",
            "photorealistic", "painterly", "anime", "cartoon",
            "vintage", "modern", "futuristic", "retro"
        ]

        for keyword in style_keywords:
            features[keyword] = prompt.count(keyword)

        # 提取技术描述
        tech_keywords = [
            "shallow depth", "deep focus", "motion blur",
            "film grain", "vignette", "lens flare"
        ]

        for keyword in tech_keywords:
            features[keyword] = 1 if keyword in prompt else 0

        return features

    def _calculate_style_consistency(self, current_features: Dict[str, Any],
                                     all_features: List[Dict[str, Any]]) -> float:
        """计算风格一致性"""
        if not all_features or len(all_features) < 2:
            return 1.0

        # 计算平均特征
        avg_features = {}
        for features in all_features:
            for key, value in features.items():
                if key not in avg_features:
                    avg_features[key] = []
                avg_features[key].append(value)

        for key in avg_features:
            avg_features[key] = sum(avg_features[key]) / len(avg_features[key])

        # 计算当前特征与平均特征的相似度
        total_similarity = 0
        total_features = 0

        for key, avg_value in avg_features.items():
            if key in current_features:
                current_value = current_features[key]
                # 简单相似度计算
                if avg_value == 0 and current_value == 0:
                    similarity = 1.0
                elif avg_value == 0 or current_value == 0:
                    similarity = 0.0
                else:
                    similarity = min(current_value, avg_value) / max(current_value, avg_value)

                total_similarity += similarity
                total_features += 1

        if total_features == 0:
            return 1.0

        return total_similarity / total_features

    def calculate_visual_quality_scores(self, results: List[VisualQualityCheckResult]) -> Dict[str, float]:
        """计算视觉质量评分"""
        if not results:
            return {
                "overall": 1.0,
                "composition": 1.0,
                "lighting": 1.0,
                "color": 1.0,
                "style": 1.0
            }

        scores = {
            "composition": [],
            "lighting": [],
            "color": [],
            "style": []
        }

        for result in results:
            if isinstance(result, VisualQualityCheckResult):
                if result.quality_dimension == "composition":
                    scores["composition"].append(result.score)
                elif result.quality_dimension == "lighting":
                    scores["lighting"].append(result.score)
                elif result.quality_dimension == "color":
                    scores["color"].append(result.score)
                elif result.quality_dimension == "style":
                    scores["style"].append(result.score)

        # 计算平均分
        avg_scores = {}
        for key, value_list in scores.items():
            if value_list:
                avg_scores[key] = sum(value_list) / len(value_list)
            else:
                avg_scores[key] = 1.0

        # 计算总分（加权平均）
        weights = {
            "composition": 0.3,
            "lighting": 0.25,
            "color": 0.25,
            "style": 0.2
        }

        overall_score = sum(avg_scores[key] * weights[key] for key in avg_scores)
        avg_scores["overall"] = overall_score

        return avg_scores
