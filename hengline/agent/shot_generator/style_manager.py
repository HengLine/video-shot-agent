"""
@FileName: style_manager.py
@Description: 风格一致性管理器
@Author: HengLine
@Time: 2026/1/5 23:09
"""

from .model.shot_models import ShotSize, CameraMovement, VisualEffect, CameraParameters, SoraShot
from .model.style_models import StyleGuide, LightingScheme, LightingStyle, ColorPalette

"""
风格管理器 - 确保视觉风格一致性
"""
import copy
from typing import Dict, List, Any, Tuple
import colorsys


class ColorConsistencyChecker:
    """色彩一致性检查器"""

    def __init__(self):
        self.color_tolerance = 30  # 颜色距离容差

    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """十六进制颜色转RGB"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c * 2 for c in hex_color])
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def rgb_to_hsl(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """RGB转HSL"""
        r, g, b = [x / 255.0 for x in rgb]
        return colorsys.rgb_to_hls(r, g, b)

    def calculate_color_distance(self, color1: str, color2: str) -> float:
        """计算两个颜色的距离"""
        rgb1 = self.hex_to_rgb(color1)
        rgb2 = self.hex_to_rgb(color2)

        # 欧几里得距离
        distance = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb1, rgb2)) ** 0.5
        return distance

    def check_color_consistency(self, palette: ColorPalette,
                                style_guide: StyleGuide) -> bool:
        """检查色彩是否符合全局风格"""
        # 检查主导色
        if style_guide.dominant_colors:
            min_distance = min(
                self.calculate_color_distance(palette.dominant_color, global_color)
                for global_color in style_guide.dominant_colors[:3]  # 检查前三个主导色
            )
            if min_distance > self.color_tolerance:
                return False

        # 检查色彩温度
        if style_guide.color_temperature and palette.color_temperature:
            if style_guide.color_temperature != palette.color_temperature:
                # 计算HSL中的色调来判断冷暖
                rgb = self.hex_to_rgb(palette.dominant_color)
                h, l, s = self.rgb_to_hsl(rgb)

                # 暖色：红色到黄色 (0-60度)
                # 冷色：青色到蓝色 (180-240度)
                is_warm = h < 0.166 or h > 0.833  # 0-60度或300-360度

                expected_warm = style_guide.color_temperature == "warm"
                if is_warm != expected_warm:
                    return False

        return True

    def adjust_to_global_palette(self, palette: ColorPalette,
                                 style_guide: StyleGuide) -> ColorPalette:
        """调整到全局调色板"""
        adjusted = copy.deepcopy(palette)

        if style_guide.dominant_colors:
            # 找到最接近的全局主导色
            closest_color = min(
                style_guide.dominant_colors,
                key=lambda c: self.calculate_color_distance(palette.dominant_color, c)
            )
            adjusted.dominant_color = closest_color

        # 更新次要颜色，保持与主导色的关系
        if palette.secondary_colors:
            # 计算原主导色与次要颜色的关系
            original_hsl = self.rgb_to_hsl(self.hex_to_rgb(palette.dominant_color))
            adjusted_hsl = self.rgb_to_hsl(self.hex_to_rgb(adjusted.dominant_color))

            # 调整次要颜色，保持色相差
            adjusted_secondary = []
            for sec_color in palette.secondary_colors:
                sec_rgb = self.hex_to_rgb(sec_color)
                sec_hsl = self.rgb_to_hsl(sec_rgb)

                # 保持明度和饱和度，调整色调
                hue_diff = sec_hsl[0] - original_hsl[0]
                new_hue = (adjusted_hsl[0] + hue_diff) % 1.0

                # 转换回RGB
                new_rgb = colorsys.hls_to_rgb(new_hue, sec_hsl[1], sec_hsl[2])
                new_hex = self.rgb_to_hex(
                    int(new_rgb[0] * 255),
                    int(new_rgb[1] * 255),
                    int(new_rgb[2] * 255)
                )
                adjusted_secondary.append(new_hex)

            adjusted.secondary_colors = adjusted_secondary

        adjusted.color_temperature = style_guide.color_temperature
        adjusted.color_grading = style_guide.color_grading

        return adjusted

    def rgb_to_hex(self, r: int, g: int, b: int) -> str:
        """RGB转十六进制"""
        return f"#{r:02x}{g:02x}{b:02x}"


class StyleConsistencyManager:
    """风格一致性管理器"""

    def __init__(self, global_style_guide: StyleGuide):
        self.style_guide = global_style_guide
        self.applied_styles = []
        self.color_checker = ColorConsistencyChecker()

    def ensure_consistency(self, shot: SoraShot) -> SoraShot:
        """确保镜头符合全局风格指南"""
        adjusted_shot = copy.deepcopy(shot)

        # 1. 检查并调整色彩一致性
        if not self.color_checker.check_color_consistency(
                adjusted_shot.color_palette, self.style_guide
        ):
            adjusted_shot.color_palette = self.color_checker.adjust_to_global_palette(
                adjusted_shot.color_palette, self.style_guide
            )

        # 2. 检查并调整灯光一致性
        if not self._check_lighting_consistency(adjusted_shot.lighting_scheme):
            adjusted_shot.lighting_scheme = self._adjust_lighting(
                adjusted_shot.lighting_scheme
            )

        # 3. 检查并调整相机风格一致性
        if not self._check_camera_style_consistency(adjusted_shot.camera_parameters):
            adjusted_shot.camera_parameters = self._adjust_camera_style(
                adjusted_shot.camera_parameters
            )

        # 4. 更新风格提示词以反映调整
        adjusted_shot.style_prompt = self._update_style_prompt(
            adjusted_shot.style_prompt,
            adjusted_shot.color_palette,
            adjusted_shot.lighting_scheme
        )

        # 5. 添加全局视觉特效
        adjusted_shot.visual_effects = self._add_global_visual_effects(
            adjusted_shot.visual_effects
        )

        # 记录应用的风格
        self.applied_styles.append({
            "shot_id": adjusted_shot.shot_id,
            "original_style": shot.style_prompt[:50] + "...",
            "adjusted_styles": self._get_adjusted_style_names(shot, adjusted_shot),
            "timestamp": adjusted_shot.timestamp_generated
        })

        return adjusted_shot

    def _check_lighting_consistency(self, lighting: LightingScheme) -> bool:
        """检查灯光一致性"""
        # 检查灯光风格
        if self.style_guide.lighting_style:
            if lighting.style != self.style_guide.lighting_style:
                return False

        # 检查主光方向（如果指定）
        if self.style_guide.key_light_direction:
            if lighting.key_light_direction != self.style_guide.key_light_direction:
                return False

        # 检查时间一致性（如果指定）
        if hasattr(self.style_guide, 'time_of_day') and self.style_guide.time_of_day:
            if lighting.time_of_day and lighting.time_of_day != self.style_guide.time_of_day:
                return False

        return True

    def _adjust_lighting(self, lighting: LightingScheme) -> LightingScheme:
        """调整灯光方案"""
        adjusted = copy.deepcopy(lighting)

        # 调整灯光风格
        if self.style_guide.lighting_style:
            adjusted.style = self.style_guide.lighting_style

        # 调整主光方向
        if self.style_guide.key_light_direction:
            adjusted.key_light_direction = self.style_guide.key_light_direction

        # 调整色温以匹配全局风格
        if self.style_guide.color_temperature == "warm":
            adjusted.color_temperature = 3200  # 暖白
            adjusted.ambient_light = "warm"
        elif self.style_guide.color_temperature == "cool":
            adjusted.color_temperature = 6500  # 冷白
            adjusted.ambient_light = "cool"

        # 调整情绪描述
        mood_mapping = {
            LightingStyle.DRAMATIC: "dramatic mood",
            LightingStyle.SOFT: "soft romantic mood",
            LightingStyle.NATURALISTIC: "natural realistic mood",
            LightingStyle.HIGH_CONTRAST: "high contrast cinematic mood"
        }
        adjusted.mood_description = mood_mapping.get(
            adjusted.style, "cinematic mood"
        )

        return adjusted

    def _check_camera_style_consistency(self, camera: CameraParameters) -> bool:
        """检查相机风格一致性"""
        # 检查是否在偏好相机风格列表中
        if self.style_guide.preferred_camera_styles:
            camera_style = self._camera_params_to_style(camera)
            if camera_style not in self.style_guide.preferred_camera_styles:
                return False

        # 检查构图偏好
        if self.style_guide.framing_preferences:
            content_type = self._get_camera_content_type(camera)
            if content_type in self.style_guide.framing_preferences:
                expected_framing = self.style_guide.framing_preferences[content_type]
                if camera.shot_size.value != expected_framing:
                    return False

        return True

    def _camera_params_to_style(self, camera: CameraParameters) -> str:
        """将相机参数转换为风格标签"""
        if camera.camera_movement == CameraMovement.HANDHELD_SHAKY:
            return "handheld"
        elif camera.camera_movement in [CameraMovement.SMOOTH_TRACKING, CameraMovement.DOLLY_IN]:
            return "steadycam"
        elif camera.camera_movement == CameraMovement.STATIC:
            return "static"
        else:
            return "cinematic"

    def _get_camera_content_type(self, camera: CameraParameters) -> str:
        """根据相机参数推断内容类型"""
        if camera.shot_size in [ShotSize.EXTREME_CLOSE_UP, ShotSize.CLOSE_UP]:
            return "emotional"
        elif camera.shot_size in [ShotSize.MEDIUM_CLOSE_UP, ShotSize.MEDIUM_SHOT]:
            return "dialogue"
        elif camera.shot_size in [ShotSize.WIDE_SHOT, ShotSize.EXTREME_WIDE_SHOT]:
            return "establishing"
        else:
            return "general"

    def _adjust_camera_style(self, camera: CameraParameters) -> CameraParameters:
        """调整相机风格"""
        adjusted = copy.deepcopy(camera)

        # 如果有偏好风格，调整
        if self.style_guide.preferred_camera_styles:
            preferred_style = self.style_guide.preferred_camera_styles[0]

            if preferred_style == "handheld":
                adjusted.camera_movement = CameraMovement.HANDHELD_SHAKY
                adjusted.movement_speed = "medium"
            elif preferred_style == "steadycam":
                adjusted.camera_movement = CameraMovement.SMOOTH_TRACKING
                adjusted.movement_speed = "slow"
            elif preferred_style == "static":
                adjusted.camera_movement = CameraMovement.STATIC
                adjusted.movement_speed = "static"

        # 根据构图偏好调整
        if self.style_guide.framing_preferences:
            content_type = self._get_camera_content_type(camera)
            if content_type in self.style_guide.framing_preferences:
                framing = self.style_guide.framing_preferences[content_type]
                # 将字符串转换为ShotSize枚举
                try:
                    adjusted.shot_size = ShotSize(framing)
                except ValueError:
                    pass  # 保持原样

        return adjusted

    def _update_style_prompt(self, original_prompt: str,
                             color_palette: ColorPalette,
                             lighting_scheme: LightingScheme) -> str:
        """更新风格提示词"""
        # 移除原有的颜色和灯光描述
        prompt_parts = original_prompt.split(', ')

        # 过滤掉颜色相关描述
        color_keywords = ["color", "grading", "warm", "cool", "teal", "orange", "palette"]
        filtered_parts = [
            part for part in prompt_parts
            if not any(keyword in part.lower() for keyword in color_keywords)
        ]

        # 过滤掉灯光相关描述
        lighting_keywords = ["lighting", "light", "shadow", "bright", "dark", "glow"]
        filtered_parts = [
            part for part in filtered_parts
            if not any(keyword in part.lower() for keyword in lighting_keywords)
        ]

        # 添加新的风格描述
        new_parts = []

        # 添加颜色描述
        color_desc = f"{color_palette.color_temperature} color grading"
        if color_palette.color_grading:
            color_desc += f", {color_palette.color_grading} palette"
        new_parts.append(color_desc)

        # 添加灯光描述
        lighting_desc = f"{lighting_scheme.style.value} lighting"
        if lighting_scheme.mood_description:
            lighting_desc += f", {lighting_scheme.mood_description}"
        new_parts.append(lighting_desc)

        # 合并所有部分
        updated_prompt = ', '.join(filtered_parts + new_parts)
        return updated_prompt

    def _add_global_visual_effects(self, effects: List[VisualEffect]) -> List[VisualEffect]:
        """添加全局视觉特效"""
        global_effects = []

        # 添加胶片颗粒
        if hasattr(self.style_guide, 'film_grain_level') and self.style_guide.film_grain_level > 0:
            global_effects.append(VisualEffect(
                effect_type="film_grain",
                intensity=self.style_guide.film_grain_level,
                parameters={"size": "fine", "monochromatic": True}
            ))

        # 添加色差（如果启用）
        if hasattr(self.style_guide, 'chromatic_aberration') and self.style_guide.chromatic_aberration:
            global_effects.append(VisualEffect(
                effect_type="chromatic_aberration",
                intensity=0.3,
                parameters={"fringe_size": 1.5, "red_shift": 0.5, "blue_shift": -0.5}
            ))

        # 添加镜头畸变（如果启用）
        if hasattr(self.style_guide, 'lens_distortion') and self.style_guide.lens_distortion:
            global_effects.append(VisualEffect(
                effect_type="lens_distortion",
                intensity=0.1,
                parameters={"distortion_type": "barrel", "curvature": 0.05}
            ))

        # 合并效果，避免重复
        existing_types = {effect.effect_type for effect in effects}
        for effect in global_effects:
            if effect.effect_type not in existing_types:
                effects.append(effect)

        return effects

    def _get_adjusted_style_names(self, original: SoraShot,
                                  adjusted: SoraShot) -> List[str]:
        """获取已调整的风格名称"""
        adjusted_styles = []

        if original.color_palette.dominant_color != adjusted.color_palette.dominant_color:
            adjusted_styles.append("color_palette")

        if original.lighting_scheme.style != adjusted.lighting_scheme.style:
            adjusted_styles.append("lighting_style")

        if original.camera_parameters.camera_movement != adjusted.camera_parameters.camera_movement:
            adjusted_styles.append("camera_movement")

        if len(original.visual_effects) != len(adjusted.visual_effects):
            adjusted_styles.append("visual_effects")

        return adjusted_styles

    def get_style_report(self) -> Dict[str, Any]:
        """获取风格调整报告"""
        return {
            "total_shots_adjusted": len(self.applied_styles),
            "adjustments_by_type": self._count_adjustment_types(),
            "most_adjusted_style": self._get_most_adjusted_style(),
            "consistency_score": self._calculate_consistency_score(),
            "adjustment_details": self.applied_styles
        }

    def _count_adjustment_types(self) -> Dict[str, int]:
        """统计调整类型"""
        counts = {}
        for adjustment in self.applied_styles:
            for style_type in adjustment["adjusted_styles"]:
                counts[style_type] = counts.get(style_type, 0) + 1
        return counts

    def _get_most_adjusted_style(self) -> str:
        """获取最常调整的风格"""
        counts = self._count_adjustment_types()
        if counts:
            return max(counts.items(), key=lambda x: x[1])[0]
        return "none"

    def _calculate_consistency_score(self) -> float:
        """计算一致性得分"""
        if not self.applied_styles:
            return 1.0

        # 计算平均调整数量
        total_adjustments = sum(len(adj["adjusted_styles"]) for adj in self.applied_styles)
        avg_adjustments = total_adjustments / len(self.applied_styles)

        # 转换为得分（调整越少，得分越高）
        # 0调整：1.0，1调整：0.8，2调整：0.6，3+调整：0.4
        if avg_adjustments == 0:
            return 1.0
        elif avg_adjustments == 1:
            return 0.8
        elif avg_adjustments == 2:
            return 0.6
        else:
            return 0.4
