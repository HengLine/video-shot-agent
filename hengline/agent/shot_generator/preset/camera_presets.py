"""
@FileName: camera_presets.py
@Description: 相机预设配置
@Author: HengLine
@Time: 2026/1/6 12:29
"""

from typing import Dict, List, Any

from hengline.agent.shot_generator.model.shot_models import ShotSize, CameraMovement


class CameraPresets:
    """相机预设配置类"""

    # 镜头语言风格预设
    CAMERA_STYLE_PRESETS = {
        # 1. 电影感对话风格
        "cinematic_dialogue": {
            "shot_size_sequence": [
                ShotSize.MEDIUM_CLOSE_UP,
                ShotSize.CLOSE_UP,
                ShotSize.MEDIUM_SHOT
            ],
            "movement_patterns": [
                CameraMovement.SLOW_PUSH_IN,
                CameraMovement.STATIC,
                CameraMovement.SLOW_PULL_OUT
            ],
            "lens_preferences": [50, 85, 35],  # 焦距mm
            "framing_style": "rule_of_thirds",
            "depth_of_field": "shallow",
            "framerate": 24,
            "movement_speed": "slow",
            "transition_style": "smooth_cuts",
            "description": "经典电影对话镜头风格，注重角色表情和情绪传递"
        },

        # 2. 动作场景风格
        "action_dynamic": {
            "shot_size_sequence": [
                ShotSize.WIDE_SHOT,
                ShotSize.MEDIUM_SHOT,
                ShotSize.CLOSE_UP,
                ShotSize.EXTREME_CLOSE_UP
            ],
            "movement_patterns": [
                CameraMovement.HANDHELD_SHAKY,
                CameraMovement.DOLLY_IN,
                CameraMovement.PAN_LEFT,
                CameraMovement.PAN_RIGHT
            ],
            "lens_preferences": [24, 35, 50],
            "framing_style": "dynamic_asymmetric",
            "depth_of_field": "medium",
            "framerate": 48,  # 为慢动作留余地
            "movement_speed": "fast",
            "transition_style": "quick_cuts",
            "description": "动态动作镜头风格，快速切换和手持摄影感"
        },

        # 3. 情感揭示风格
        "emotional_reveal": {
            "shot_size_sequence": [
                ShotSize.MEDIUM_SHOT,
                ShotSize.CLOSE_UP,
                ShotSize.EXTREME_CLOSE_UP
            ],
            "movement_patterns": [
                CameraMovement.STATIC,
                CameraMovement.SLOW_PUSH_IN,
                CameraMovement.STATIC
            ],
            "lens_preferences": [85, 105, 135],
            "framing_style": "centered_intimate",
            "depth_of_field": "very_shallow",
            "framerate": 24,
            "movement_speed": "very_slow",
            "transition_style": "slow_dissolves",
            "description": "情感揭示镜头风格，极度特写和缓慢运动"
        },

        # 4. 建立场景风格
        "establishing_epic": {
            "shot_size_sequence": [
                ShotSize.EXTREME_WIDE_SHOT,
                ShotSize.WIDE_SHOT,
                ShotSize.FULL_SHOT
            ],
            "movement_patterns": [
                CameraMovement.SLOW_PAN,
                CameraMovement.CRANE_SHOT,
                CameraMovement.DRONE_SHOT
            ],
            "lens_preferences": [16, 24, 35],
            "framing_style": "symmetrical_epic",
            "depth_of_field": "deep",
            "framerate": 24,
            "movement_speed": "very_slow",
            "transition_style": "long_fades",
            "description": "史诗感建立场景风格，广角和缓慢移动"
        },

        # 5. 纪录片风格
        "documentary_realistic": {
            "shot_size_sequence": [
                ShotSize.MEDIUM_SHOT,
                ShotSize.MEDIUM_CLOSE_UP,
                ShotSize.FULL_SHOT
            ],
            "movement_patterns": [
                CameraMovement.HANDHELD_SHAKY,
                CameraMovement.SMOOTH_TRACKING,
                CameraMovement.STATIC
            ],
            "lens_preferences": [35, 50, 85],
            "framing_style": "observational_loose",
            "depth_of_field": "natural",
            "framerate": 30,
            "movement_speed": "natural",
            "transition_style": "jump_cuts",
            "description": "纪录片写实风格，手持感和观察性构图"
        },

        # 6. 悬疑惊悚风格
        "suspense_thriller": {
            "shot_size_sequence": [
                ShotSize.CLOSE_UP,
                ShotSize.MEDIUM_SHOT,
                ShotSize.WIDE_SHOT
            ],
            "movement_patterns": [
                CameraMovement.SLOW_DOLLY,
                CameraMovement.STATIC,
                CameraMovement.SLOW_PAN
            ],
            "lens_preferences": [35, 50, 24],
            "framing_style": "unbalanced_tense",
            "depth_of_field": "selective",
            "framerate": 24,
            "movement_speed": "slow",
            "transition_style": "tense_cuts",
            "description": "悬疑惊悚风格，不平衡构图和缓慢张力"
        },

        # 7. 浪漫爱情风格
        "romantic_love": {
            "shot_size_sequence": [
                ShotSize.CLOSE_UP,
                ShotSize.MEDIUM_CLOSE_UP,
                ShotSize.MEDIUM_SHOT
            ],
            "movement_patterns": [
                CameraMovement.SMOOTH_TRACKING,
                CameraMovement.SLOW_PUSH_IN,
                CameraMovement.CIRCULAR_MOVEMENT
            ],
            "lens_preferences": [50, 85, 135],
            "framing_style": "soft_framing",
            "depth_of_field": "shallow_with_bokeh",
            "framerate": 24,
            "movement_speed": "gentle",
            "transition_style": "soft_dissolves",
            "description": "浪漫爱情风格，柔和运动和浅景深"
        },

        # 8. 喜剧风格
        "comedy_light": {
            "shot_size_sequence": [
                ShotSize.MEDIUM_SHOT,
                ShotSize.FULL_SHOT,
                ShotSize.WIDE_SHOT
            ],
            "movement_patterns": [
                CameraMovement.STATIC,
                CameraMovement.QUICK_PAN,
                CameraMovement.ZOOM_IN
            ],
            "lens_preferences": [35, 28, 24],
            "framing_style": "balanced_clean",
            "depth_of_field": "medium",
            "framerate": 30,
            "movement_speed": "brisk",
            "transition_style": "snappy_cuts",
            "description": "喜剧轻松风格，干净构图和快速切换"
        }
    }

    # 导演/电影风格映射
    DIRECTOR_STYLES = {
        "christopher_nolan": {
            "base_style": "cinematic_dialogue",
            "modifications": {
                "framing_style": "precise_symmetrical",
                "movement_speed": "deliberate",
                "preferred_transitions": ["hard_cuts", "match_cuts"],
                "visual_characteristics": ["practical_lighting", "minimal_cgi", "IMAX_ratio"]
            }
        },
        "hayao_miyazaki": {
            "base_style": "emotional_reveal",
            "modifications": {
                "framing_style": "painterly_composition",
                "movement_speed": "flowing",
                "preferred_transitions": ["fluid_dissolves", "morph_transitions"],
                "visual_characteristics": ["hand_drawn_feel", "natural_lighting", "detailed_backgrounds"]
            }
        },
        "wong_kar_wai": {
            "base_style": "romantic_love",
            "modifications": {
                "framing_style": "off_center_poetic",
                "movement_speed": "dreamy_slow",
                "preferred_transitions": ["step_prints", "slow_motion"],
                "visual_characteristics": ["color_filters", "smoke_effects", "close_ups"]
            }
        },
        "steven_spielberg": {
            "base_style": "establishing_epic",
            "modifications": {
                "framing_style": "classic_hollywood",
                "movement_speed": "story_driven",
                "preferred_transitions": ["invisible_cuts", "tracking_shots"],
                "visual_characteristics": ["warm_lighting", "deep_focus", "fluid_camera"]
            }
        },
        "quentin_tarantino": {
            "base_style": "suspense_thriller",
            "modifications": {
                "framing_style": "iconic_framing",
                "movement_speed": "conversational",
                "preferred_transitions": ["chapter_cards", "long_takes"],
                "visual_characteristics": ["low_angles", "trunk_shots", "pop_references"]
            }
        }
    }

    # 内容类型到相机风格的映射
    CONTENT_TYPE_TO_CAMERA_STYLE = {
        "dialogue_intimate": "cinematic_dialogue",
        "dialogue_group": "documentary_realistic",
        "action_fight": "action_dynamic",
        "action_chase": "action_dynamic",
        "emotional_reveal": "emotional_reveal",
        "emotional_breakdown": "emotional_reveal",
        "establishing_location": "establishing_epic",
        "establishing_mood": "suspense_thriller",
        "romantic_moment": "romantic_love",
        "comedy_gag": "comedy_light",
        "montage_sequence": "documentary_realistic",
        "flashback_memory": "romantic_love",
        "dream_sequence": "emotional_reveal"
    }

    # 情绪到相机运动的映射
    EMOTION_TO_CAMERA_MOVEMENT = {
        "happy": {
            "primary": CameraMovement.SMOOTH_TRACKING,
            "secondary": CameraMovement.CIRCULAR_MOVEMENT,
            "speed": "brisk",
            "stability": "smooth"
        },
        "sad": {
            "primary": CameraMovement.STATIC,
            "secondary": CameraMovement.SLOW_PULL_OUT,
            "speed": "slow",
            "stability": "stable"
        },
        "angry": {
            "primary": CameraMovement.HANDHELD_SHAKY,
            "secondary": CameraMovement.QUICK_ZOOM,
            "speed": "fast",
            "stability": "unstable"
        },
        "tense": {
            "primary": CameraMovement.SLOW_DOLLY,
            "secondary": CameraMovement.STATIC,
            "speed": "very_slow",
            "stability": "tense"
        },
        "romantic": {
            "primary": CameraMovement.SLOW_PUSH_IN,
            "secondary": CameraMovement.SMOOTH_TRACKING,
            "speed": "gentle",
            "stability": "fluid"
        },
        "mysterious": {
            "primary": CameraMovement.SLOW_PAN,
            "secondary": CameraMovement.STATIC,
            "speed": "slow",
            "stability": "deliberate"
        },
        "epic": {
            "primary": CameraMovement.CRANE_SHOT,
            "secondary": CameraMovement.DRONE_SHOT,
            "speed": "majestic",
            "stability": "steady"
        }
    }

    # 镜头大小到使用场景的映射
    SHOT_SIZE_USAGE_GUIDE = {
        ShotSize.EXTREME_CLOSE_UP: {
            "best_for": ["emotional intensity", "detail focus", "dramatic moments"],
            "avoid_for": ["establishing shots", "action sequences"],
            "typical_duration": 2.0,
            "lens_recommendation": [85, 105, 135],
            "composition": "centered or rule_of_thirds"
        },
        ShotSize.CLOSE_UP: {
            "best_for": ["dialogue", "facial expressions", "intimate moments"],
            "avoid_for": ["wide action", "establishing context"],
            "typical_duration": 3.0,
            "lens_recommendation": [50, 85, 105],
            "composition": "rule_of_thirds"
        },
        ShotSize.MEDIUM_CLOSE_UP: {
            "best_for": ["conversations", "character introduction", "emotional scenes"],
            "avoid_for": ["wide landscapes", "group action"],
            "typical_duration": 4.0,
            "lens_recommendation": [35, 50, 85],
            "composition": "rule_of_thirds"
        },
        ShotSize.MEDIUM_SHOT: {
            "best_for": ["dialogue with body language", "character interactions", "standard scenes"],
            "avoid_for": ["extreme close-ups", "epic wide shots"],
            "typical_duration": 5.0,
            "lens_recommendation": [35, 50, 70],
            "composition": "balanced"
        },
        ShotSize.FULL_SHOT: {
            "best_for": ["character full body", "action movements", "costume showing"],
            "avoid_for": ["intimate moments", "facial detail"],
            "typical_duration": 4.0,
            "lens_recommendation": [35, 50, 85],
            "composition": "centered or rule_of_thirds"
        },
        ShotSize.WIDE_SHOT: {
            "best_for": ["establishing location", "group scenes", "action sequences"],
            "avoid_for": ["intimate dialogue", "facial expressions"],
            "typical_duration": 6.0,
            "lens_recommendation": [24, 35, 50],
            "composition": "rule_of_thirds or leading_lines"
        },
        ShotSize.EXTREME_WIDE_SHOT: {
            "best_for": ["epic landscapes", "architectural shots", "dramatic reveals"],
            "avoid_for": ["character focus", "dialogue scenes"],
            "typical_duration": 8.0,
            "lens_recommendation": [16, 24, 35],
            "composition": "symmetrical or epic_scale"
        }
    }

    # 镜头过渡指南
    TRANSITION_GUIDE = {
        "cut": {
            "best_for": ["standard scene changes", "action sequences", "conversations"],
            "emotional_effect": "neutral, direct",
            "typical_duration": 0.1,
            "timing": "on_action_or_dialogue"
        },
        "dissolve": {
            "best_for": ["time passage", "dream sequences", "emotional transitions"],
            "emotional_effect": "soft, dreamy, reflective",
            "typical_duration": 1.0,
            "timing": "between_scenes_or_moods"
        },
        "fade": {
            "best_for": ["scene beginnings/endings", "dramatic pauses", "chapter breaks"],
            "emotional_effect": "final, contemplative",
            "typical_duration": 1.5,
            "timing": "scene_boundaries"
        },
        "wipe": {
            "best_for": ["stylized transitions", "geometric patterns", "retro style"],
            "emotional_effect": "playful, stylistic",
            "typical_duration": 0.8,
            "timing": "stylized_changes"
        },
        "match_cut": {
            "best_for": ["thematic connections", "time transitions", "visual metaphors"],
            "emotional_effect": "clever, meaningful",
            "typical_duration": 0.5,
            "timing": "on_visual_similarity"
        },
        "jump_cut": {
            "best_for": ["documentary style", "time compression", "uneasy feeling"],
            "emotional_effect": "raw, urgent, modern",
            "typical_duration": 0.2,
            "timing": "within_same_scene"
        }
    }

    @staticmethod
    def get_camera_style(style_name: str) -> Dict[str, Any]:
        """获取相机风格预设"""
        return CameraPresets.CAMERA_STYLE_PRESETS.get(
            style_name,
            CameraPresets.CAMERA_STYLE_PRESETS["cinematic_dialogue"]
        )

    @staticmethod
    def get_style_for_content_type(content_type: str) -> str:
        """根据内容类型获取相机风格"""
        return CameraPresets.CONTENT_TYPE_TO_CAMERA_STYLE.get(
            content_type,
            "cinematic_dialogue"
        )

    @staticmethod
    def get_movement_for_emotion(emotion: str) -> Dict[str, Any]:
        """根据情绪获取相机运动"""
        return CameraPresets.EMOTION_TO_CAMERA_MOVEMENT.get(
            emotion.lower(),
            CameraPresets.EMOTION_TO_CAMERA_MOVEMENT["neutral"]
        )

    @staticmethod
    def get_shot_size_guide(shot_size: ShotSize) -> Dict[str, Any]:
        """获取镜头大小使用指南"""
        return CameraPresets.SHOT_SIZE_USAGE_GUIDE.get(
            shot_size,
            CameraPresets.SHOT_SIZE_USAGE_GUIDE[ShotSize.MEDIUM_SHOT]
        )

    @staticmethod
    def get_transition_guide(transition_type: str) -> Dict[str, Any]:
        """获取镜头过渡指南"""
        return CameraPresets.TRANSITION_GUIDE.get(
            transition_type,
            CameraPresets.TRANSITION_GUIDE["cut"]
        )

    @staticmethod
    def get_director_style(director_name: str) -> Dict[str, Any]:
        """获取导演风格"""
        if director_name in CameraPresets.DIRECTOR_STYLES:
            style = CameraPresets.DIRECTOR_STYLES[director_name]
            base_style = CameraPresets.get_camera_style(style["base_style"])
            # 合并基本风格和导演特定修改
            merged_style = {**base_style, **style["modifications"]}
            return merged_style
        else:
            return CameraPresets.get_camera_style("cinematic_dialogue")

    @staticmethod
    def generate_shot_sequence(content_type: str,
                               duration: float,
                               emotion: str = None) -> List[Dict[str, Any]]:
        """生成镜头序列"""
        style_name = CameraPresets.get_style_for_content_type(content_type)
        style = CameraPresets.get_camera_style(style_name)

        shot_sizes = style["shot_size_sequence"]
        movements = style["movement_patterns"]

        # 计算每个镜头的时长
        avg_shot_duration = duration / len(shot_sizes)

        sequence = []
        for i, (shot_size, movement) in enumerate(zip(shot_sizes, movements)):
            shot_info = {
                "shot_number": i + 1,
                "shot_size": shot_size,
                "camera_movement": movement,
                "duration": avg_shot_duration,
                "lens_recommendation": style["lens_preferences"][i % len(style["lens_preferences"])],
                "framing": style["framing_style"],
                "notes": f"{shot_size.value.replace('_', ' ')} shot with {movement.value.replace('_', ' ')} movement"
            }
            sequence.append(shot_info)

        return sequence

    @staticmethod
    def recommend_shot_size(content_description: str,
                            character_count: int = 1) -> ShotSize:
        """根据内容描述推荐镜头大小"""
        description = content_description.lower()

        # 检查关键词
        if any(word in description for word in ["extreme close", "eye detail", "micro expression"]):
            return ShotSize.EXTREME_CLOSE_UP

        elif any(word in description for word in ["close up", "face", "expression", "emotion"]):
            return ShotSize.CLOSE_UP

        elif any(word in description for word in ["medium close", "bust shot", "shoulders up"]):
            return ShotSize.MEDIUM_CLOSE_UP

        elif any(word in description for word in ["medium shot", "waist up", "dialogue", "conversation"]):
            return ShotSize.MEDIUM_SHOT

        elif any(word in description for word in ["full shot", "full body", "costume", "standing"]):
            return ShotSize.FULL_SHOT

        elif any(word in description for word in ["wide shot", "establishing", "location", "environment"]):
            return ShotSize.WIDE_SHOT

        elif any(word in description for word in ["extreme wide", "epic", "landscape", "aerial"]):
            return ShotSize.EXTREME_WIDE_SHOT

        # 根据角色数量决定
        if character_count == 1:
            return ShotSize.MEDIUM_CLOSE_UP
        elif character_count == 2:
            return ShotSize.MEDIUM_SHOT
        elif character_count <= 4:
            return ShotSize.FULL_SHOT
        else:
            return ShotSize.WIDE_SHOT

    @staticmethod
    def recommend_transition(current_mood: str,
                             next_mood: str) -> str:
        """根据情绪变化推荐过渡类型"""
        # 情绪强度映射
        mood_intensity = {
            "calm": 1,
            "neutral": 2,
            "happy": 3,
            "sad": 3,
            "tense": 4,
            "angry": 5,
            "epic": 5
        }

        current_intensity = mood_intensity.get(current_mood.lower(), 2)
        next_intensity = mood_intensity.get(next_mood.lower(), 2)

        intensity_diff = abs(current_intensity - next_intensity)

        if intensity_diff >= 3:
            return "fade"  # 大情绪变化用淡入淡出
        elif intensity_diff >= 2:
            return "dissolve"  # 中等变化用淡入淡出
        elif current_mood == next_mood:
            return "cut"  # 相同情绪用硬切
        else:
            return "dissolve"  # 小变化用淡入淡出


# 测试函数
def test_camera_presets():
    """测试相机预设"""
    presets = CameraPresets

    # 测试获取风格
    style = presets.get_camera_style("cinematic_dialogue")
    print(f"Cinematic dialogue style: {style['description']}")

    # 测试内容类型映射
    content_style = presets.get_style_for_content_type("emotional_reveal")
    print(f"Style for emotional reveal: {content_style}")

    # 测试情绪到运动映射
    emotion_movement = presets.get_movement_for_emotion("happy")
    print(f"Camera movement for happy: {emotion_movement['primary'].value}")

    # 测试镜头大小推荐
    description = "A close up of the character's face showing emotion"
    shot_size = presets.recommend_shot_size(description)
    print(f"Recommended shot size: {shot_size.value}")

    # 测试镜头序列生成
    sequence = presets.generate_shot_sequence("dialogue_intimate", 15.0)
    print(f"\nGenerated shot sequence:")
    for shot in sequence:
        print(f"  Shot {shot['shot_number']}: {shot['shot_size'].value} - {shot['camera_movement'].value}")

    # 测试过渡推荐
    transition = presets.recommend_transition("calm", "tense")
    print(f"\nRecommended transition from calm to tense: {transition}")


if __name__ == "__main__":
    test_camera_presets()