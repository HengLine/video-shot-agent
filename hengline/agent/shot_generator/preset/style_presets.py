"""
@FileName: style_presets.py
@Description: 风格预设配置
@Author: HengLine
@Time: 2026/1/5 23:44
"""
from hengline.agent.shot_generator.model.style_models import StyleGuide, LightingStyle

# 电影感现代风格
CINEMATIC_MODERN = StyleGuide(
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
    artistic_influences=["Roger Deakins", "Christopher Nolan"],
    reference_films=["Blade Runner 2049", "Inception"],
    texture_quality="high_detail",
    film_grain_level=0.1,
    chromatic_aberration=True
)

# 动漫风格
ANIME_STYLE = StyleGuide(
    visual_theme="anime_cinematic",
    era_period="contemporary",
    dominant_colors=["#87CEEB", "#FFB6C1", "#98FB98"],
    color_temperature="vibrant",
    color_grading="saturated_vivid",
    lighting_style=LightingStyle.HIGH_KEY,
    key_light_direction="front_light",
    preferred_camera_styles=["dynamic", "expressive"],
    framing_preferences={
        "dialogue": "medium_close_up",
        "emotional": "extreme_close_up",
        "action": "wide_dynamic"
    },
    artistic_influences=["Hayao Miyazaki", "Makoto Shinkai"],
    reference_films=["Your Name", "Spirited Away"],
    texture_quality="stylized",
    film_grain_level=0.05,
    chromatic_aberration=False
)

# 复古胶片风格
VINTAGE_FILM = StyleGuide(
    visual_theme="vintage_film",
    era_period="1990s",
    dominant_colors=["#8B7355", "#696969", "#8B0000"],
    color_temperature="warm",
    color_grading="sepia_tone",
    lighting_style=LightingStyle.CHIAROSCURO,
    key_light_direction="side_light",
    preferred_camera_styles=["handheld", "organic"],
    framing_preferences={
        "dialogue": "medium_shot",
        "emotional": "close_up",
        "establishing": "wide_shot_with_vignette"
    },
    artistic_influences=["Wong Kar-wai", "Quentin Tarantino"],
    reference_films=["Chungking Express", "Pulp Fiction"],
    texture_quality="film_grain_heavy",
    film_grain_level=0.3,
    chromatic_aberration=True,
    lens_distortion=True
)

# 科幻风格
SCIFI_FUTURISTIC = StyleGuide(
    visual_theme="futuristic_scifi",
    era_period="future",
    dominant_colors=["#0A2342", "#00A8E8", "#FFFFFF"],
    color_temperature="cool",
    color_grading="neon_blue",
    lighting_style=LightingStyle.LOW_KEY,
    key_light_direction="backlight",
    preferred_camera_styles=["smooth_tracking", "drone"],
    framing_preferences={
        "dialogue": "medium_close_up_glowing",
        "emotional": "close_up_with_reflections",
        "establishing": "ultra_wide_futuristic"
    },
    artistic_influences=["Denis Villeneuve", "Ridley Scott"],
    reference_films=["Blade Runner 2049", "Arrival"],
    texture_quality="ultra_clean",
    film_grain_level=0.0,
    chromatic_aberration=True,
    lens_distortion=False
)

# 预设映射
STYLE_PRESETS = {
    "cinematic_modern": CINEMATIC_MODERN,
    "anime_style": ANIME_STYLE,
    "vintage_film": VINTAGE_FILM,
    "scifi_futuristic": SCIFI_FUTURISTIC
}