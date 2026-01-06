"""
@FileName: technical_validator.py
@Description: 技术验证器 - 验证技术质量
@Author: HengLine
@Time: 2026/1/6 16:03
"""

from typing import List, Dict, Any, Optional

from .model.check_models import TechnicalCheckResult, CheckStatus
from .model.issue_models import IssueSeverity


class TechnicalValidator:
    """技术验证器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.prompt_standards = self.config.get("prompt_standards", {})
        self.camera_standards = self.config.get("camera_standards", {})
        self.feasibility_standards = self.config.get("feasibility_standards", {})

    def validate_all_shots(self, shots: List[Any]) -> List[TechnicalCheckResult]:
        """验证所有镜头的技术质量"""
        results = []

        for shot in shots:
            # 1. 提示词质量验证
            prompt_result = self.validate_prompt_quality(shot)
            if prompt_result:
                results.append(prompt_result)

            # 2. 相机参数验证
            camera_result = self.validate_camera_parameters(shot)
            if camera_result:
                results.append(camera_result)

            # 3. 可行性验证
            feasibility_result = self.validate_feasibility(shot)
            if feasibility_result:
                results.append(feasibility_result)

        return results

    def validate_prompt_quality(self, shot: Any) -> TechnicalCheckResult:
        """验证提示词质量"""

        prompt = shot.full_sora_prompt

        # 基本指标
        prompt_length = len(prompt)
        word_count = len(prompt.split())

        # 检查提示词结构
        has_subject = self._check_prompt_element(prompt, "subject")
        has_action = self._check_prompt_element(prompt, "action")
        has_environment = self._check_prompt_element(prompt, "environment")
        has_style = self._check_prompt_element(prompt, "style")
        has_technical = self._check_prompt_element(prompt, "technical")

        # 检查问题
        prompt_issues = []
        prompt_improvements = []

        # 长度检查
        if prompt_length < 30:
            prompt_issues.append("提示词过短")
            prompt_improvements.append("添加更多细节描述")
        elif prompt_length > 500:
            prompt_issues.append("提示词过长")
            prompt_improvements.append("精简描述，保留关键信息")

        # 结构检查
        elements_present = [has_subject, has_action, has_environment, has_style, has_technical]
        elements_count = sum(elements_present)

        if elements_count < 3:
            prompt_issues.append("提示词结构不完整")
            missing = []
            if not has_subject:
                missing.append("主体")
            if not has_action:
                missing.append("动作")
            if not has_environment:
                missing.append("环境")
            if not has_style:
                missing.append("风格")
            if not has_technical:
                missing.append("技术参数")

            prompt_improvements.append(f"添加缺失元素: {', '.join(missing)}")

        # 清晰度检查
        clarity_issues = self._check_prompt_clarity(prompt)
        if clarity_issues:
            prompt_issues.extend(clarity_issues)
            prompt_improvements.append("使用更具体的词汇，避免歧义")

        # 计算质量得分
        base_score = 0.5
        score = base_score

        # 长度得分
        if 50 <= prompt_length <= 300:
            score += 0.2
        elif prompt_length < 50:
            score += 0.1
        elif prompt_length > 300:
            score += 0.1

        # 结构得分
        score += (elements_count * 0.06)  # 每个元素加0.06分

        # 问题扣分
        score -= (len(prompt_issues) * 0.05)

        # 限制分数在0-1之间
        score = max(0.0, min(1.0, score))

        # 计算清晰度得分
        clarity_score = 1.0 - (len(clarity_issues) * 0.1)
        clarity_score = max(0.0, min(1.0, clarity_score))

        # 确定状态和严重性
        if score >= 0.7:
            status = CheckStatus.PASSED
            severity = IssueSeverity.INFO
        elif score >= 0.5:
            status = CheckStatus.WARNING
            severity = IssueSeverity.LOW
        else:
            status = CheckStatus.FAILED
            severity = IssueSeverity.MEDIUM

        return TechnicalCheckResult(
            check_id=f"prompt_quality_{shot.shot_id}",
            check_name="提示词质量验证",
            check_description="验证Sora提示词质量",
            status=status,
            severity=severity,
            score=score,
            technical_aspect="prompt_quality",
            prompt_issues=prompt_issues,
            prompt_length=prompt_length,
            prompt_clarity_score=clarity_score,
            prompt_improvements=prompt_improvements,
            details={
                "word_count": word_count,
                "elements_present": {
                    "subject": has_subject,
                    "action": has_action,
                    "environment": has_environment,
                    "style": has_style,
                    "technical": has_technical
                },
                "clarity_issues": clarity_issues
            },
            evidence=[
                f"长度: {prompt_length}字符",
                f"完整元素: {elements_count}/5",
                f"问题: {len(prompt_issues)}个"
            ]
        )

    def _check_prompt_element(self, prompt: str, element_type: str) -> bool:
        """检查提示词是否包含特定元素"""
        prompt_lower = prompt.lower()

        element_keywords = {
            "subject": ["man", "woman", "character", "person", "figure"],
            "action": ["sitting", "standing", "walking", "talking", "looking"],
            "environment": ["room", "indoors", "outdoors", "street", "garden"],
            "style": ["cinematic", "realistic", "photorealistic", "artistic"],
            "technical": ["shot", "camera", "lens", "lighting", "depth"]
        }

        if element_type not in element_keywords:
            return False

        keywords = element_keywords[element_type]
        return any(keyword in prompt_lower for keyword in keywords)

    def _check_prompt_clarity(self, prompt: str) -> List[str]:
        """检查提示词清晰度问题"""
        issues = []

        # 检查模糊词汇
        vague_words = ["something", "somehow", "kind of", "sort of", "maybe", "perhaps"]
        for word in vague_words:
            if word in prompt.lower():
                issues.append(f"模糊词汇: '{word}'")

        # 检查矛盾描述
        contradictions = [
            ("small", "large"),
            ("bright", "dark"),
            ("fast", "slow"),
            ("close", "far")
        ]

        for word1, word2 in contradictions:
            if word1 in prompt.lower() and word2 in prompt.lower():
                issues.append(f"矛盾描述: '{word1}'和'{word2}'")

        # 检查过于技术性的术语（可能不被Sora理解）
        technical_jargon = ["aperture f/", "shutter speed", "ISO", "color grading LUT"]
        for jargon in technical_jargon:
            if jargon in prompt:
                issues.append(f"技术术语: '{jargon}'可能不被完全理解")

        return issues

    def validate_camera_parameters(self, shot: Any) -> Optional[TechnicalCheckResult]:
        """验证相机参数"""

        if not hasattr(shot, 'camera_parameters'):
            return None

        camera = shot.camera_parameters
        camera_issues = []
        camera_adjustments = []

        # 检查参数有效性
        if hasattr(camera, 'lens_focal_length'):
            if camera.lens_focal_length < 10 or camera.lens_focal_length > 1000:
                camera_issues.append(f"异常焦距: {camera.lens_focal_length}mm")
                camera_adjustments.append("使用常见焦距: 24mm, 35mm, 50mm, 85mm等")

        if hasattr(camera, 'framerate'):
            common_framerates = [24, 25, 30, 48, 50, 60]
            if camera.framerate not in common_framerates:
                camera_issues.append(f"非常见帧率: {camera.framerate}fps")
                camera_adjustments.append(f"使用常见帧率: {common_framerates}")

        # 检查镜头运动与时长匹配
        if hasattr(camera, 'camera_movement'):
            movement = camera.camera_movement.value
            shot_duration = shot.duration

            # 快速运动需要足够时长
            fast_movements = ["dolly_in", "dolly_out", "handheld_shaky"]
            if movement in fast_movements and shot_duration < 2.0:
                camera_issues.append(f"运动'{movement}'在{shot_duration}s内可能太快")
                camera_adjustments.append("增加镜头时长或使用较慢运动")

            # 静态镜头不宜过长
            if movement == "static" and shot_duration > 8.0:
                camera_issues.append(f"静态镜头{shot_duration}s可能过长")
                camera_adjustments.append("添加轻微运动或缩短时长")

        # 计算得分
        base_score = 0.8
        score = base_score - (len(camera_issues) * 0.1)
        score = max(0.0, min(1.0, score))

        # 确定状态和严重性
        if score >= 0.7:
            status = CheckStatus.PASSED
            severity = IssueSeverity.INFO
        elif score >= 0.5:
            status = CheckStatus.WARNING
            severity = IssueSeverity.LOW
        else:
            status = CheckStatus.FAILED
            severity = IssueSeverity.MEDIUM

        return TechnicalCheckResult(
            check_id=f"camera_params_{shot.shot_id}",
            check_name="相机参数验证",
            check_description="验证相机参数合理性",
            status=status,
            severity=severity,
            score=score,
            technical_aspect="camera_params",
            camera_issues=camera_issues,
            camera_parameter_validity=len(camera_issues) == 0,
            camera_adjustments=camera_adjustments,
            details={
                "camera_parameters": {
                    "shot_size": camera.shot_size.value if hasattr(camera, 'shot_size') else "unknown",
                    "movement": camera.camera_movement.value if hasattr(camera, 'camera_movement') else "unknown",
                    "lens": camera.lens_focal_length if hasattr(camera, 'lens_focal_length') else "unknown",
                    "framerate": camera.framerate if hasattr(camera, 'framerate') else "unknown"
                }
            },
            evidence=camera_issues if camera_issues else ["相机参数合理"]
        )

    def validate_feasibility(self, shot: Any) -> TechnicalCheckResult:
        """验证可行性"""

        prompt = shot.full_sora_prompt.lower()

        # 检查Sora可能难以实现的元素
        feasibility_issues = []
        feasibility_suggestions = []
        feasibility_score = 0.8  # 基础分

        # 复杂交互
        complex_interactions = [
            "complex choreography",
            "intricate dance",
            "multiple characters interacting precisely",
            "synchronized movements"
        ]

        for interaction in complex_interactions:
            if interaction in prompt:
                feasibility_issues.append(f"复杂交互: {interaction}")
                feasibility_suggestions.append("简化交互或分多个镜头拍摄")
                feasibility_score -= 0.1

        # 精确时间控制
        if "exactly at the same time" in prompt or "perfectly synchronized" in prompt:
            feasibility_issues.append("精确时间控制可能困难")
            feasibility_suggestions.append("允许轻微时间差异")
            feasibility_score -= 0.05

        # 物理上不可能
        impossible_physics = [
            "defying gravity",
            "flying without support",
            "impossible physics",
            "contradicting laws of physics"
        ]

        for physics in impossible_physics:
            if physics in prompt:
                feasibility_issues.append(f"物理上不可能: {physics}")
                feasibility_suggestions.append("调整描述使其物理上可行")
                feasibility_score -= 0.15

        # 极端特写细节
        if "microscopic detail" in prompt or "atomic level" in prompt:
            feasibility_issues.append("极端特写细节可能不清晰")
            feasibility_suggestions.append("使用合理特写级别")
            feasibility_score -= 0.05

        # 限制分数在0-1之间
        feasibility_score = max(0.0, min(1.0, feasibility_score))

        # 确定状态和严重性
        if feasibility_score >= 0.7:
            status = CheckStatus.PASSED
            severity = IssueSeverity.INFO
        elif feasibility_score >= 0.5:
            status = CheckStatus.WARNING
            severity = IssueSeverity.LOW
        else:
            status = CheckStatus.FAILED
            severity = IssueSeverity.MEDIUM

        return TechnicalCheckResult(
            check_id=f"feasibility_{shot.shot_id}",
            check_name="可行性验证",
            check_description="验证镜头生成可行性",
            status=status,
            severity=severity,
            score=feasibility_score,
            technical_aspect="feasibility",
            feasibility_issues=feasibility_issues,
            feasibility_score=feasibility_score,
            feasibility_suggestions=feasibility_suggestions,
            details={
                "feasibility_issues": feasibility_issues,
                "prompt_complexity": "high" if len(feasibility_issues) > 2 else "medium" if len(feasibility_issues) > 0 else "low"
            },
            evidence=feasibility_issues if feasibility_issues else ["生成可行性良好"]
        )

    def calculate_technical_scores(self, results: List[TechnicalCheckResult]) -> Dict[str, float]:
        """计算技术质量评分"""
        if not results:
            return {
                "overall": 1.0,
                "prompt": 1.0,
                "camera": 1.0,
                "feasibility": 1.0
            }

        scores = {
            "prompt": [],
            "camera": [],
            "feasibility": []
        }

        for result in results:
            if isinstance(result, TechnicalCheckResult):
                if result.technical_aspect == "prompt_quality":
                    scores["prompt"].append(result.score)
                elif result.technical_aspect == "camera_params":
                    scores["camera"].append(result.score)
                elif result.technical_aspect == "feasibility":
                    scores["feasibility"].append(result.score)

        # 计算平均分
        avg_scores = {}
        for key, value_list in scores.items():
            if value_list:
                avg_scores[key] = sum(value_list) / len(value_list)
            else:
                avg_scores[key] = 1.0

        # 计算总分（加权平均）
        weights = {
            "prompt": 0.4,
            "camera": 0.3,
            "feasibility": 0.3
        }

        overall_score = sum(avg_scores[key] * weights[key] for key in avg_scores)
        avg_scores["overall"] = overall_score

        return avg_scores
