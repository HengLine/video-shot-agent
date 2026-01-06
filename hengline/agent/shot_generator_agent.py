# -*- coding: utf-8 -*-
"""
@FileName: shot_generator_agent.py
@Description: 分镜生成智能体，负责生成符合AI视频模型要求的提示词
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import copy
import time
from typing import Dict, List, Any, Optional, Tuple

from hengline.prompts.prompts_manager import prompt_manager
from .continuity_guardian.continuity_guardian_model import AnchoredSegment, HardConstraint
from .shot_generator.camera_optimizer import CameraOptimizer
from .shot_generator.constraint_handler import ConstraintHandler
from .shot_generator.model.data_models import TechnicalSettings, ContinuityAnchoredInput, VisualEffect, GenerationMetadata
from .shot_generator.model.shot_models import SoraReadyShots, ShotSize, ShotTransition, SoraPromptStructure, CameraParameters, GenerationHints, CameraMovement, SoraShot
from .shot_generator.model.style_models import StyleGuide, LightingStyle, LightingScheme, ColorPalette
from .shot_generator.prompt_engine import PromptEngine
from .shot_generator.style_manager import StyleConsistencyManager
from .temporal_planner.temporal_planner_model import TimeSegment, ContentType
from hengline.logger import debug, info, warning, error


class ShotGeneratorAgent:
    """分镜生成智能体"""

    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None):
        """
        初始化分镜生成智能体
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        # 分镜生成提示词模板 - 从YAML加载或使用默认
        self.shot_generation_template = prompt_manager.get_shot_generator_prompt()
        #############
        self.config = config or {}
        self.prompt_engine = PromptEngine()
        self.camera_optimizer = CameraOptimizer()
        self.constraint_handler = ConstraintHandler()

        # 默认风格指南
        self.default_style_guide = StyleGuide(
            visual_theme="modern_cinematic",
            era_period="contemporary",
            dominant_colors=["#2c3e50", "#ecf0f1", "#3498db"],
            color_temperature="warm",
            color_grading="teal_and_orange",
            lighting_style=LightingStyle.NATURALISTIC,
            key_light_direction="45_degree_right",
            preferred_camera_styles=["steadycam", "static"],
            framing_preferences={
                "dialogue": "medium_close_up",
                "emotional": "close_up",
                "establishing": "wide_shot"
            },
            artistic_influences=["Roger Deakins", "Hayao Miyazaki"],
            reference_films=["Blade Runner 2049", "Your Name"]
        )

        # 默认技术设置
        self.default_technical_settings = TechnicalSettings(
            resolution="1920x1080",
            aspect_ratio="16:9",
            framerate=24,
            bit_depth=10,
            color_space="rec709",
            render_engine="sora_v2",
            cfg_scale=7.5,
            steps=50,
            sampler="ddim"
        )

    def process(self, input_data: ContinuityAnchoredInput) -> SoraReadyShots:
        """处理输入数据，生成Sora就绪的分镜指令"""
        start_time = time.time()

        debug(f"开始处理 {len(input_data.anchored_segments)} 个片段...")

        # 1. 初始化风格管理器
        style_guide = self._get_style_guide(input_data)
        style_manager = StyleConsistencyManager(style_guide)

        # 2. 处理每个片段
        shot_sequence = []
        all_satisfied_constraints = []
        all_constraint_violations = []
        previous_shot = None

        for i, anchored_segment in enumerate(input_data.anchored_segments):
            debug(f"  处理片段 {i + 1}/{len(input_data.anchored_segments)}: {anchored_segment.segment_id}")

            # 为片段生成镜头
            segment_shots = self._generate_shots_for_segment(
                anchored_segment, style_guide, previous_shot
            )

            # 确保风格一致性
            for shot in segment_shots:
                consistent_shot = style_manager.ensure_consistency(shot)

                # 添加到序列
                shot_sequence.append(consistent_shot)

                # 更新约束满足记录
                all_satisfied_constraints.extend(consistent_shot.satisfied_constraints)
                all_constraint_violations.extend(consistent_shot.constraint_violations)

                # 更新前一镜头
                previous_shot = consistent_shot

        # 3. 计算质量指标
        processing_time = time.time() - start_time
        quality_metrics = self._calculate_quality_metrics(
            shot_sequence, all_satisfied_constraints, all_constraint_violations
        )

        # 4. 生成约束摘要
        constraints_summary = self._generate_constraints_summary(
            input_data.anchored_segments, all_satisfied_constraints, all_constraint_violations
        )

        # 5. 生成元数据
        generation_metadata = GenerationMetadata(
            generator_version="1.0.0",
            processing_time=processing_time,
            total_segments=len(input_data.anchored_segments),
            total_shots=len(shot_sequence),
            constraint_satisfaction_rate=quality_metrics["constraint_satisfaction"],
            style_consistency_score=quality_metrics["style_consistency_score"],
            visual_appeal_score=quality_metrics["visual_appeal_score"],
            warnings=quality_metrics["warnings"],
            suggestions=quality_metrics["suggestions"]
        )

        # 6. 获取风格报告
        style_report = style_manager.get_style_report()

        debug(f"处理完成！生成 {len(shot_sequence)} 个镜头，用时 {processing_time:.2f}秒")
        debug(f"约束满足率: {quality_metrics['constraint_satisfaction']:.2%}")
        debug(f"风格一致性: {style_report['consistency_score']:.2%}")

        # 7. 返回最终结果
        return SoraReadyShots(
            shot_sequence=shot_sequence,
            technical_settings=self.default_technical_settings,
            style_consistency=style_guide,
            constraints_summary=constraints_summary,
            generation_metadata=generation_metadata,
            visual_appeal_score=quality_metrics["visual_appeal_score"],
            constraint_satisfaction=quality_metrics["constraint_satisfaction"]
        )

    def _get_style_guide(self, input_data: ContinuityAnchoredInput) -> StyleGuide:
        """获取风格指南（可扩展从输入数据中提取）"""
        # 这里可以从input_data.metadata中提取自定义风格指南
        # 目前使用默认指南
        return self.default_style_guide

    def _generate_shots_for_segment(self, anchored_segment: AnchoredSegment,
                                    style_guide: StyleGuide,
                                    previous_shot: Optional[SoraShot] = None) -> List[SoraShot]:
        """为单个片段生成镜头"""
        shots = []

        # 1. 生成提示词结构
        prompt_structure = self.prompt_engine.generate_shot_prompt(
            anchored_segment, style_guide
        )

        # 2. 确定相机参数
        camera_params = self.camera_optimizer.optimize_for_content(
            anchored_segment.base_segment, previous_shot
        )

        # 3. 确定灯光方案
        lighting_scheme = self._determine_lighting_scheme(
            anchored_segment.base_segment, style_guide
        )

        # 4. 确定色彩调色板
        color_palette = self._determine_color_palette(
            anchored_segment.base_segment, style_guide
        )

        # 5. 确定满足的约束
        satisfied_constraints, constraint_violations = self._check_constraint_compliance(
            anchored_segment.hard_constraints, prompt_structure, camera_params
        )

        # 6. 创建镜头
        shot = SoraShot(
            shot_id=f"{anchored_segment.segment_id}_shot1",
            segment_id=anchored_segment.segment_id,
            time_range=anchored_segment.base_segment.time_range,
            duration=anchored_segment.base_segment.duration,

            # 提示词
            primary_prompt=prompt_structure.subject_description,
            style_prompt=prompt_structure.style_enhancement,
            technical_prompt=prompt_structure.technical_specs,
            full_sora_prompt=prompt_structure.compose_full_prompt(),

            # 约束满足
            satisfied_constraints=satisfied_constraints,
            constraint_violations=constraint_violations,
            constraint_compliance_score=len(satisfied_constraints) /
                                        max(1, len(anchored_segment.hard_constraints)),

            # 视觉参数
            camera_parameters=camera_params,
            lighting_scheme=lighting_scheme,
            color_palette=color_palette,

            # 特效和过渡
            visual_effects=self._determine_visual_effects(anchored_segment.base_segment),
            transition_to_next=self._determine_transition(anchored_segment),

            # 生成提示
            generation_hints=self._extract_generation_hints(anchored_segment),

            # 元数据
            content_type=anchored_segment.base_segment.content_type or ContentType.DIALOGUE_INTIMATE,
            emotional_tone=anchored_segment.base_segment.emotional_tone or "neutral",
            action_intensity=anchored_segment.base_segment.action_intensity
        )

        shots.append(shot)

        # 如果有复杂内容，可能需要多个镜头
        if anchored_segment.base_segment.duration > 8.0:
            # 创建第二个镜头（不同角度）
            second_shot = self._create_alternate_angle_shot(shot, anchored_segment)
            shots.append(second_shot)

        return shots

    def _determine_lighting_scheme(self, segment: TimeSegment,
                                   style_guide: StyleGuide) -> LightingScheme:
        """确定灯光方案"""
        # 基于内容类型决定基础灯光
        content_type = segment.content_type or "dialogue_intimate"

        if content_type == "emotional_reveal":
            base_style = LightingStyle.DRAMATIC
        elif content_type == "action_fast":
            base_style = LightingStyle.HIGH_CONTRAST
        else:
            base_style = style_guide.lighting_style

        return LightingScheme(
            style=base_style,
            key_light_direction=style_guide.key_light_direction,
            fill_light_ratio=0.3,
            backlight_intensity=0.5,
            ambient_light=style_guide.color_temperature,
            color_temperature=3200 if style_guide.color_temperature == "warm" else 6500,
            mood_description=self._get_mood_description(segment.emotional_tone),
            time_of_day=self._extract_time_of_day(segment.visual_content)
        )

    def _get_mood_description(self, emotional_tone: Optional[str]) -> str:
        """获取情绪描述"""
        mood_mapping = {
            "happy": "joyful uplifting mood",
            "sad": "melancholy emotional mood",
            "tense": "suspenseful tense mood",
            "romantic": "romantic dreamy mood",
            "neutral": "neutral realistic mood"
        }
        return mood_mapping.get(emotional_tone or "neutral", "cinematic mood")

    def _extract_time_of_day(self, visual_content: str) -> Optional[str]:
        """从视觉内容中提取时间"""
        content = visual_content.lower()

        if "morning" in content or "早晨" in content:
            return "morning"
        elif "afternoon" in content or "下午" in content:
            return "afternoon"
        elif "evening" in content or "傍晚" in content:
            return "evening"
        elif "night" in content or "夜晚" in content:
            return "night"

        return None

    def _determine_color_palette(self, segment: TimeSegment,
                                 style_guide: StyleGuide) -> ColorPalette:
        """确定色彩调色板"""

        color_palette = ColorPalette(
            dominant_color=style_guide.dominant_colors[0],
            secondary_colors=style_guide.dominant_colors[1:3],
            accent_color="#e74c3c",  # 强调色
            color_temperature=style_guide.color_temperature,
            saturation_level="natural",
            brightness="normal",
            contrast_ratio=0.6,
            color_grading=style_guide.color_grading,
            hex_colors={
                "primary": style_guide.dominant_colors[0],
                "secondary": style_guide.dominant_colors[1] if len(style_guide.dominant_colors) > 1 else "#ecf0f1",
                "accent": "#e74c3c"
            }
        )

        if segment.emotional_tone == "sad":
            color_palette.dominant_color = "#2c3e50"  # 冷色调
            color_palette.saturation_level = "muted"
            color_palette.brightness = "dark"

        elif segment.emotional_tone == "happy":
            color_palette.dominant_color = "#f1c40f"  # 温暖黄色
            color_palette.saturation_level = "vibrant"
            color_palette.brightness = "bright"

        return color_palette

    def _check_constraint_compliance(self, constraints: List[HardConstraint],
                                     prompt_structure: SoraPromptStructure,
                                     camera_params: CameraParameters) -> Tuple[List[str], List[str]]:
        """检查约束符合性"""
        satisfied = []
        violated = []

        for constraint in constraints:
            if not constraint.is_enforced:
                continue

            # 检查是否在提示词中
            in_prompt = (
                    constraint.description.lower() in prompt_structure.subject_description.lower() or
                    constraint.sora_instruction.lower() in prompt_structure.technical_specs.lower()
            )

            # 检查是否在相机参数中
            in_camera = self._check_constraint_in_camera(constraint, camera_params)

            if in_prompt or in_camera:
                satisfied.append(constraint.constraint_id)
            else:
                violated.append(constraint.constraint_id)

        return satisfied, violated

    def _check_constraint_in_camera(self, constraint: HardConstraint,
                                    camera_params: CameraParameters) -> bool:
        """检查约束是否在相机参数中满足"""
        if constraint.type != "camera_angle":
            return False

        # 解析相机约束
        instruction = constraint.sora_instruction.lower()

        if "close-up" in instruction or "特写" in instruction:
            return camera_params.shot_size in [ShotSize.CLOSE_UP, ShotSize.EXTREME_CLOSE_UP]
        elif "wide" in instruction or "广角" in instruction:
            return camera_params.shot_size in [ShotSize.WIDE_SHOT, ShotSize.EXTREME_WIDE_SHOT]
        elif "medium" in instruction or "中景" in instruction:
            return camera_params.shot_size in [ShotSize.MEDIUM_SHOT, ShotSize.MEDIUM_CLOSE_UP]

        return False

    def _determine_visual_effects(self, segment: TimeSegment) -> List[VisualEffect]:
        """确定视觉特效"""
        effects = []

        # 基于内容类型添加效果
        if segment.content_type == "action_fast":
            effects.append(VisualEffect(
                effect_type="motion_blur",
                intensity=0.7,
                parameters={"direction": "following", "amount": 0.5}
            ))

        # 基于情绪添加效果
        if segment.emotional_tone == "dreamy" or "dream" in segment.visual_content.lower():
            effects.append(VisualEffect(
                effect_type="bloom",
                intensity=0.4,
                parameters={"threshold": 0.8, "size": 5}
            ))

        return effects

    def _determine_transition(self, anchored_segment: AnchoredSegment) -> ShotTransition:
        """确定镜头过渡"""
        transition_type = anchored_segment.transition_to_next.transition_type

        # 根据内容类型调整过渡
        if anchored_segment.base_segment.content_type == "emotional_reveal":
            transition_type = "dissolve"
            duration = 0.8
        elif anchored_segment.base_segment.action_intensity > 1.5:
            transition_type = "cut"
            duration = 0.1
        else:
            duration = 0.3

        return ShotTransition(
            transition_type=transition_type,
            duration=duration,
            style="smooth",
            timing_curve="ease_in_out"
        )

    def _extract_generation_hints(self, anchored_segment: AnchoredSegment) -> GenerationHints:
        """提取生成提示"""
        # 从视觉内容中提取关键词
        content = anchored_segment.base_segment.visual_content.lower()

        emphasis_keywords = []
        avoid_keywords = ["blurry", "pixelated", "low quality"]

        # 根据内容类型添加强调关键词
        if "face" in content or "expression" in content:
            emphasis_keywords.append("detailed facial expressions")
            emphasis_keywords.append("clear eye contact")

        if "hand" in content or "holding" in content:
            emphasis_keywords.append("detailed hand gestures")
            emphasis_keywords.append("natural hand movements")

        # 从约束中提取
        for constraint in anchored_segment.hard_constraints[:3]:
            if constraint.priority >= 8:
                # 从描述中提取关键词
                words = constraint.description.split()
                emphasis_keywords.extend(words[:3])

        return GenerationHints(
            emphasis_keywords=list(set(emphasis_keywords[:5])),  # 最多5个
            avoid_keywords=avoid_keywords,
            reference_styles=["cinematic realism", "photorealistic"],
            visual_references=[],
            technical_requirements={"consistency": "high", "detail": "ultra"}
        )

    def _create_alternate_angle_shot(self, base_shot: SoraShot,
                                     anchored_segment: AnchoredSegment) -> SoraShot:
        """创建备用角度镜头"""
        # 创建副本
        alternate = copy.deepcopy(base_shot)

        # 修改ID
        alternate.shot_id = alternate.shot_id.replace("shot1", "shot2")

        # 修改相机参数
        alternate.camera_parameters = self._get_alternate_angle(
            alternate.camera_parameters
        )

        # 修改时间范围（接续）
        start_time = alternate.time_range[1]
        alternate.time_range = (start_time, start_time + alternate.duration)

        # 修改提示词
        alternate.full_sora_prompt = alternate.full_sora_prompt.replace(
            "medium close-up", "over the shoulder shot"
        )

        return alternate

    def _get_alternate_angle(self, camera_params: CameraParameters) -> CameraParameters:
        """获取备用角度"""
        alternate = copy.deepcopy(camera_params)

        # 修改镜头大小
        if alternate.shot_size in [ShotSize.MEDIUM_CLOSE_UP, ShotSize.CLOSE_UP]:
            alternate.shot_size = ShotSize.MEDIUM_SHOT
        elif alternate.shot_size == ShotSize.MEDIUM_SHOT:
            alternate.shot_size = ShotSize.FULL_SHOT

        # 修改相机高度
        if alternate.camera_height == "eye_level":
            alternate.camera_height = "low_angle"
        else:
            alternate.camera_height = "eye_level"

        # 修改运动
        if alternate.camera_movement == CameraMovement.SLOW_PUSH_IN:
            alternate.camera_movement = CameraMovement.SLOW_PULL_OUT

        return alternate

    def _calculate_quality_metrics(self, shot_sequence: List[SoraShot],
                                   satisfied_constraints: List[str],
                                   violated_constraints: List[str]) -> Dict[str, Any]:
        """计算质量指标"""
        total_constraints = len(satisfied_constraints) + len(violated_constraints)

        if total_constraints == 0:
            constraint_satisfaction = 1.0
        else:
            constraint_satisfaction = len(satisfied_constraints) / total_constraints

        # 计算视觉吸引力（基于启发式）
        visual_appeal = 0.0
        for shot in shot_sequence:
            # 基于镜头多样性
            appeal_factors = []

            # 镜头大小多样性
            shot_size_score = {
                ShotSize.EXTREME_CLOSE_UP: 0.8,
                ShotSize.CLOSE_UP: 0.9,
                ShotSize.MEDIUM_CLOSE_UP: 0.8,
                ShotSize.MEDIUM_SHOT: 0.7,
                ShotSize.FULL_SHOT: 0.6,
                ShotSize.WIDE_SHOT: 0.7,
                ShotSize.EXTREME_WIDE_SHOT: 0.8
            }.get(shot.camera_parameters.shot_size, 0.5)

            # 运动多样性
            movement_score = 0.5
            if shot.camera_parameters.camera_movement != CameraMovement.STATIC:
                movement_score = 0.8

            # 内容类型得分
            content_score = {
                "dialogue_intimate": 0.7,
                "action_fast": 0.9,
                "emotional_reveal": 0.8,
                "establishing_shot": 0.6
            }.get(shot.content_type.value, 0.5)

            # 综合得分
            shot_appeal = (shot_size_score + movement_score + content_score) / 3
            visual_appeal += shot_appeal

        if shot_sequence:
            visual_appeal /= len(shot_sequence)

        # 检查潜在问题
        warnings = []
        suggestions = []

        # 检查镜头重复
        shot_sizes = [shot.camera_parameters.shot_size for shot in shot_sequence]
        if len(set(shot_sizes)) < len(shot_sizes) * 0.7:  # 超过30%重复
            warnings.append("镜头大小变化不足，可能导致视觉单调")
            suggestions.append("尝试使用更多样的镜头大小，如特写、中景、全景交替")

        # 检查过渡一致性
        transition_types = [shot.transition_to_next.transition_type for shot in shot_sequence]
        if "cut" in transition_types and "dissolve" in transition_types:
            warnings.append("混合使用硬切和淡入淡出过渡，可能导致节奏不一致")
            suggestions.append("统一过渡类型以获得更一致的节奏感")

        return {
            "constraint_satisfaction": constraint_satisfaction,
            "visual_appeal_score": visual_appeal,
            "style_consistency_score": 0.9,  # 由StyleManager提供
            "warnings": warnings,
            "suggestions": suggestions
        }

    def _generate_constraints_summary(self, anchored_segments: List[AnchoredSegment],
                                      satisfied_constraints: List[str],
                                      violated_constraints: List[str]) -> Dict[str, List[str]]:
        """生成约束摘要"""
        # 收集所有约束
        all_constraints = []
        for segment in anchored_segments:
            for constraint in segment.hard_constraints:
                all_constraints.append(constraint)

        # 按类型分类
        by_type = {}
        for constraint in all_constraints:
            if constraint.type not in by_type:
                by_type[constraint.type] = []
            by_type[constraint.type].append(constraint.constraint_id)

        # 按优先级分类
        high_priority = [c.constraint_id for c in all_constraints if c.priority >= 8]
        medium_priority = [c.constraint_id for c in all_constraints if 5 <= c.priority < 8]
        low_priority = [c.constraint_id for c in all_constraints if c.priority < 5]

        return {
            "satisfied_constraints": satisfied_constraints,
            "violated_constraints": violated_constraints,
            "by_type": by_type,
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "low_priority": low_priority,
            "total_constraints": len(all_constraints),
            "satisfaction_rate": f"{len(satisfied_constraints) / max(1, len(all_constraints)):.2%}"
        }
