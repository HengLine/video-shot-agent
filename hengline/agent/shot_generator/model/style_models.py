"""
@FileName: style_models.py
@Description: 风格相关模型
@Author: HengLine
@Time: 2026/1/5 23:05
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional


class LightingStyle(str, Enum):
    NATURALISTIC = "naturalistic"
    DRAMATIC = "dramatic"
    SOFT = "soft"
    HIGH_CONTRAST = "high_contrast"
    LOW_KEY = "low_key"
    HIGH_KEY = "high_key"
    CHIAROSCURO = "chiaroscuro"
    RIM_LIGHT = "rim_light"


@dataclass
class LightingScheme:
    """灯光方案"""
    style: LightingStyle
    key_light_direction: str
    fill_light_ratio: float = 0.3
    backlight_intensity: float = 0.5
    ambient_light: str = "warm"
    color_temperature: int = 5600  # Kelvin
    shadows_intensity: float = 0.7
    highlights_intensity: float = 0.8
    mood_description: str = ""
    time_of_day: Optional[str] = None


@dataclass
class ColorPalette:
    """色彩调色板"""
    dominant_color: str
    secondary_colors: List[str]
    accent_color: str
    color_temperature: str  # "warm", "cool", "neutral"
    saturation_level: str = "natural"  # "muted", "natural", "vibrant"
    brightness: str = "normal"  # "dark", "normal", "bright"
    contrast_ratio: float = 0.6
    color_grading: str = "teal_and_orange"
    hex_colors: Dict[str, str] = field(default_factory=dict)


@dataclass
class StyleGuide:
    """风格指南"""
    visual_theme: str
    era_period: str
    dominant_colors: List[str]
    color_temperature: str
    color_grading: str
    lighting_style: LightingStyle
    key_light_direction: str
    preferred_camera_styles: List[str]
    framing_preferences: Dict[str, str]
    artistic_influences: List[str]
    reference_films: List[str]
    texture_quality: str = "high_detail"
    film_grain_level: float = 0.1
    chromatic_aberration: bool = False
    lens_distortion: bool = False
