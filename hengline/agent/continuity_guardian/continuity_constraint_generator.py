"""
@FileName: continuity_constraint_generator.py
@Description: 约束生成
@Author: HengLine
@Time: 2026/1/4 16:46
"""
import math
import random
from collections import deque, defaultdict
from datetime import datetime
from typing import Optional, List, Dict, Callable, Any, Tuple


class ContinuityConstraintGenerator:
    """连续性约束生成器"""

    def __init__(self, rule_set: Optional[Dict] = None):
        self.rule_set = rule_set or self._get_default_rules()
        self.constraint_cache: Dict[str, List[Dict]] = {}
        self.constraint_templates = self._load_constraint_templates()
        self.context_history: deque = deque(maxlen=100)

    def _get_default_rules(self) -> Dict[str, Any]:
        """获取默认规则集"""
        return {
            "character_consistency": {
                "strictness": "high",
                "tracked_attributes": ["appearance", "outfit", "posture", "facial_expression"],
                "allowed_changes": ["emotion", "hair_movement", "sweat", "dirt"],
                "change_thresholds": {
                    "appearance": 0.1,  # 外观变化阈值
                    "outfit": 0.05,  # 服装变化阈值
                    "posture": 0.15  # 姿态变化阈值
                }
            },
            "prop_consistency": {
                "strictness": "medium",
                "tracked_attributes": ["position", "orientation", "state", "condition"],
                "movement_constraints": {
                    "max_speed": 10.0,  # 最大移动速度
                    "max_rotation": 90.0,  # 最大旋转角度（度/秒）
                    "gravity": True  # 启用重力约束
                }
            },
            "environment_consistency": {
                "strictness": "high",
                "tracked_attributes": ["lighting", "weather", "time_of_day", "ambient_conditions"],
                "transition_rules": {
                    "time_progression": "gradual",  # 时间渐变
                    "weather_changes": "progressive",  # 天气渐进变化
                    "lighting_changes": "smooth"  # 光照平滑变化
                }
            },
            "camera_consistency": {
                "strictness": "medium",
                "tracked_attributes": ["position", "orientation", "focal_length", "movement"],
                "movement_constraints": {
                    "max_shake": 0.1,  # 最大抖动
                    "smooth_transitions": True,  # 平滑转场
                    "consistent_framing": True  # 一致取景
                }
            }
        }

    def _load_constraint_templates(self) -> Dict[str, Callable]:
        """加载约束模板"""
        templates = {
            "character_appearance": self._generate_character_appearance_constraints,
            "prop_position": self._generate_prop_position_constraints,
            "environment_lighting": self._generate_environment_lighting_constraints,
            "camera_movement": self._generate_camera_movement_constraints,
            "temporal_consistency": self._generate_temporal_constraints,
            "spatial_relationships": self._generate_spatial_constraints,
            "physical_plausibility": self._generate_physical_constraints,
            "narrative_continuity": self._generate_narrative_constraints
        }
        return templates

    def generate_constraints_for_scene(self,
                                       scene_data: Dict[str, Any],
                                       previous_scene: Optional[Dict] = None,
                                       scene_type: str = "general") -> List[Dict]:
        """为场景生成约束"""
        scene_id = scene_data.get("scene_id", "unknown")

        # 检查缓存
        cache_key = f"{scene_id}_{scene_type}"
        if cache_key in self.constraint_cache:
            return self.constraint_cache[cache_key]

        constraints = []

        # 根据场景类型应用不同的约束模板
        if scene_type == "character_focus":
            constraints.extend(self._generate_character_focus_constraints(scene_data))
        elif scene_type == "action_sequence":
            constraints.extend(self._generate_action_sequence_constraints(scene_data))
        elif scene_type == "environment_establishing":
            constraints.extend(self._generate_environment_constraints(scene_data))
        elif scene_type == "dialogue":
            constraints.extend(self._generate_dialogue_constraints(scene_data))

        # 通用约束
        constraints.extend(self._generate_general_constraints(scene_data))

        # 如果有前一场景，生成连续性约束
        if previous_scene:
            constraints.extend(self._generate_continuity_constraints(scene_data, previous_scene))

        # 缓存约束
        self.constraint_cache[cache_key] = constraints

        # 记录上下文
        self.context_history.append({
            "scene_id": scene_id,
            "timestamp": datetime.now(),
            "constraint_count": len(constraints),
            "scene_type": scene_type
        })

        return constraints

    def _generate_character_focus_constraints(self, scene_data: Dict) -> List[Dict]:
        """生成角色聚焦场景的约束"""
        constraints = []
        characters = scene_data.get("characters", [])

        for char in characters:
            char_id = char.get("id")

            # 外貌一致性约束
            constraints.append({
                "type": "character_appearance",
                "entity_id": char_id,
                "constraint": "maintain_consistent_appearance",
                "parameters": {
                    "strictness": self.rule_set["character_consistency"]["strictness"],
                    "allowed_variations": self.rule_set["character_consistency"]["allowed_changes"],
                    "track_attributes": self.rule_set["character_consistency"]["tracked_attributes"]
                },
                "priority": "high"
            })

            # 表情连续性约束
            constraints.append({
                "type": "facial_expression",
                "entity_id": char_id,
                "constraint": "gradual_expression_changes",
                "parameters": {
                    "max_change_rate": 0.3,  # 最大变化率
                    "blend_duration": 0.2  # 混合持续时间（秒）
                },
                "priority": "medium"
            })

            # 视线方向约束
            if "gaze_target" in char:
                constraints.append({
                    "type": "gaze_direction",
                    "entity_id": char_id,
                    "constraint": "maintain_gaze_consistency",
                    "parameters": {
                        "target": char["gaze_target"],
                        "allowance_angle": 15.0,  # 允许的角度偏差
                        "transition_time": 0.5  # 转移时间
                    },
                    "priority": "medium"
                })

        return constraints

    def _generate_action_sequence_constraints(self, scene_data: Dict) -> List[Dict]:
        """生成动作序列的约束"""
        constraints = []

        # 运动连续性约束
        constraints.append({
            "type": "motion_continuity",
            "constraint": "smooth_motion_transitions",
            "parameters": {
                "max_acceleration": 5.0,  # 最大加速度
                "max_jerk": 20.0,  # 最大急动度
                "min_smoothness": 0.8  # 最小平滑度
            },
            "priority": "high"
        })

        # 物理合理性约束
        constraints.append({
            "type": "physical_plausibility",
            "constraint": "enforce_physics",
            "parameters": {
                "gravity": 9.8,
                "friction_coefficient": 0.3,
                "air_resistance": True
            },
            "priority": "high"
        })

        # 碰撞检测约束
        constraints.append({
            "type": "collision_detection",
            "constraint": "prevent_unrealistic_collisions",
            "parameters": {
                "collision_margin": 0.1,  # 碰撞边距
                "response_type": "realistic",  # 响应类型
                "material_properties": True  # 考虑材质属性
            },
            "priority": "critical"
        })

        return constraints

    def _generate_environment_constraints(self, scene_data: Dict) -> List[Dict]:
        """生成环境场景的约束"""
        constraints = []
        environment = scene_data.get("environment", {})

        # 光照一致性约束
        if "lighting" in environment:
            constraints.append({
                "type": "lighting_consistency",
                "constraint": "maintain_lighting_continuity",
                "parameters": {
                    "intensity_variance": 0.1,
                    "color_temperature_range": (3000, 6500),  # 色温范围
                    "shadow_consistency": True,
                    "reflection_accuracy": 0.8
                },
                "priority": "high"
            })

        # 天气连续性约束
        if "weather" in environment:
            constraints.append({
                "type": "weather_continuity",
                "constraint": "gradual_weather_transitions",
                "parameters": {
                    "transition_duration": 30.0,  # 过渡持续时间（秒）
                    "particle_density_consistency": True,
                    "effect_intensity_gradient": True
                },
                "priority": "medium"
            })

        # 时间进展约束
        if "time_of_day" in environment:
            constraints.append({
                "type": "temporal_progression",
                "constraint": "consistent_time_flow",
                "parameters": {
                    "time_scale": 1.0,  # 时间缩放
                    "shadow_movement": True,
                    "light_angle_progression": True
                },
                "priority": "high"
            })

        return constraints

    def _generate_dialogue_constraints(self, scene_data: Dict) -> List[Dict]:
        """生成对话场景的约束"""
        constraints = []

        # 唇形同步约束
        constraints.append({
            "type": "lip_sync",
            "constraint": "synchronize_lip_movement",
            "parameters": {
                "phoneme_accuracy": 0.9,
                "timing_tolerance": 0.1,  # 时间容差（秒）
                "viseme_blending": True  # 视位混合
            },
            "priority": "high"
        })

        # 肢体语言连续性
        constraints.append({
            "type": "body_language",
            "constraint": "consistent_gestures",
            "parameters": {
                "gesture_repetition_consistency": True,
                "posture_maintenance": True,
                "micro_expression_continuity": True
            },
            "priority": "medium"
        })

        # 视线交流约束
        constraints.append({
            "type": "eye_contact",
            "constraint": "realistic_eye_interaction",
            "parameters": {
                "gaze_duration_range": (0.5, 3.0),  # 注视时长范围
                "blink_rate": 0.3,  # 眨眼率（次/秒）
                "glance_patterns": True  # 真实扫视模式
            },
            "priority": "medium"
        })

        return constraints

    def _generate_general_constraints(self, scene_data: Dict) -> List[Dict]:
        """生成通用约束"""
        constraints = []

        # 颜色一致性约束
        constraints.append({
            "type": "color_consistency",
            "constraint": "maintain_color_palette",
            "parameters": {
                "palette_variance": 0.05,
                "brightness_consistency": True,
                "saturation_limits": (0.7, 1.3)
            },
            "priority": "medium"
        })

        # 纹理连续性约束
        constraints.append({
            "type": "texture_continuity",
            "constraint": "preserve_texture_details",
            "parameters": {
                "texture_resolution": "consistent",
                "specular_consistency": True,
                "normal_map_continuity": True
            },
            "priority": "medium"
        })

        # 风格一致性约束
        constraints.append({
            "type": "style_consistency",
            "constraint": "maintain_artistic_style",
            "parameters": {
                "style_attributes": scene_data.get("style", {}),
                "rendering_technique": "consistent",
                "artistic_elements": "preserved"
            },
            "priority": "high"
        })

        return constraints

    def _generate_continuity_constraints(self, current_scene: Dict, previous_scene: Dict) -> List[Dict]:
        """生成场景间连续性约束"""
        constraints = []

        # 场景过渡约束
        constraints.append({
            "type": "scene_transition",
            "constraint": "smooth_scene_change",
            "parameters": {
                "transition_type": self._determine_transition_type(previous_scene, current_scene),
                "temporal_alignment": True,
                "spatial_continuity": True
            },
            "priority": "high"
        })

        # 角色位置连续性
        prev_chars = {c["id"]: c for c in previous_scene.get("characters", [])}
        curr_chars = {c["id"]: c for c in current_scene.get("characters", [])}

        for char_id in set(prev_chars.keys()) & set(curr_chars.keys()):
            constraints.append({
                "type": "character_position_continuity",
                "entity_id": char_id,
                "constraint": "consistent_character_placement",
                "parameters": {
                    "max_movement_distance": self._calculate_max_movement(
                        prev_chars[char_id],
                        curr_chars[char_id],
                        previous_scene.get("time_elapsed", 0)
                    ),
                    "orientation_consistency": True
                },
                "priority": "high"
            })

        return constraints

    def _determine_transition_type(self, prev_scene: Dict, curr_scene: Dict) -> str:
        """确定转场类型"""
        prev_type = prev_scene.get("scene_type", "general")
        curr_type = curr_scene.get("scene_type", "general")

        if prev_type == curr_type:
            return "cut"
        elif "action" in prev_type and "action" in curr_type:
            return "match_cut"
        elif "environment" in prev_type or "environment" in curr_type:
            return "fade"
        else:
            return "dissolve"

    def _calculate_max_movement(self, prev_char: Dict, curr_char: Dict, time_elapsed: float) -> float:
        """计算最大允许移动距离"""
        base_speed = 2.0  # 基础移动速度（米/秒）
        if "action" in prev_char:
            if prev_char["action"] == "running":
                base_speed = 6.0
            elif prev_char["action"] == "walking":
                base_speed = 1.4

        return base_speed * time_elapsed

    def _generate_character_appearance_constraints(self, character: Dict) -> List[Dict]:
        """生成角色外貌约束"""
        constraints = []
        char_id = character.get("id", "unknown")

        # 基础外貌约束
        constraints.append({
            "type": "character_appearance",
            "entity_id": char_id,
            "constraint": "consistent_hair_style",
            "parameters": {
                "style": character.get("hair_style", "default"),
                "color_variance": 0.1,  # 颜色允许变化范围
                "length_variance": 0.05,  # 长度允许变化范围
                "rigidity": 0.8  # 刚性程度（防止头发过度飘动）
            },
            "priority": "high"
        })

        # 服装约束
        if "clothing" in character:
            constraints.append({
                "type": "character_appearance",
                "entity_id": char_id,
                "constraint": "consistent_clothing",
                "parameters": {
                    "outfit_type": character["clothing"].get("type", "casual"),
                    "wrinkle_consistency": True,  # 褶皱一致性
                    "dirt_pattern_consistency": True,  # 污渍模式一致性
                    "wear_and_tear_progression": "gradual"  # 磨损渐进性
                },
                "priority": "high"
            })

        # 肤色约束
        if "skin_tone" in character:
            constraints.append({
                "type": "character_appearance",
                "entity_id": char_id,
                "constraint": "consistent_skin_tone",
                "parameters": {
                    "base_color": character["skin_tone"],
                    "lighting_variance": 0.15,  # 光照引起的颜色变化范围
                    "tone_consistency": "strict",  # 色调一致性
                    "specular_consistency": True  # 高光一致性
                },
                "priority": "high"
            })

        # 年龄特征约束
        if "age_features" in character:
            constraints.append({
                "type": "character_appearance",
                "entity_id": char_id,
                "constraint": "consistent_age_features",
                "parameters": {
                    "wrinkle_pattern": character["age_features"].get("wrinkles", "none"),
                    "skin_texture": character["age_features"].get("texture", "smooth"),
                    "feature_progression": "time_locked"  # 特征随时间锁定
                },
                "priority": "medium"
            })

        # 配饰约束
        if "accessories" in character:
            for i, accessory in enumerate(character["accessories"]):
                constraints.append({
                    "type": "character_appearance",
                    "entity_id": char_id,
                    "constraint": "consistent_accessory",
                    "parameters": {
                        "accessory_type": accessory.get("type"),
                        "position_constraint": accessory.get("position", "fixed"),
                        "movement_range": accessory.get("movement_range", 0.0),
                        "interaction_aware": True  # 交互感知
                    },
                    "priority": "medium"
                })

        return constraints

    def _generate_prop_position_constraints(self, prop: Dict) -> List[Dict]:
        """生成道具位置约束"""
        constraints = []
        prop_id = prop.get("id", "unknown")

        # 基础位置约束
        constraints.append({
            "type": "prop_position",
            "entity_id": prop_id,
            "constraint": "stable_position",
            "parameters": {
                "base_position": prop.get("position", [0, 0, 0]),
                "movement_threshold": prop.get("movement_threshold", 0.1),  # 移动阈值
                "rotation_threshold": prop.get("rotation_threshold", 5.0),  # 旋转阈值
                "support_check": True  # 支撑检查
            },
            "priority": "high"
        })

        # 物理约束
        if "physics_properties" in prop:
            physics = prop["physics_properties"]
            constraints.append({
                "type": "prop_position",
                "entity_id": prop_id,
                "constraint": "physical_behavior",
                "parameters": {
                    "mass": physics.get("mass", 1.0),
                    "friction": physics.get("friction", 0.5),
                    "bounciness": physics.get("bounciness", 0.3),
                    "gravity_affected": physics.get("gravity", True)
                },
                "priority": "medium"
            })

        # 交互约束
        if "interaction_type" in prop:
            constraints.append({
                "type": "prop_position",
                "entity_id": prop_id,
                "constraint": "interaction_consistency",
                "parameters": {
                    "interaction_type": prop["interaction_type"],
                    "user_handedness": prop.get("handedness", "right"),  # 使用手型
                    "grip_position": prop.get("grip_position", "default"),
                    "usage_pattern": prop.get("usage_pattern", "normal")
                },
                "priority": "medium"
            })

        # 环境关系约束
        if "environment_relation" in prop:
            env_rel = prop["environment_relation"]
            constraints.append({
                "type": "prop_position",
                "entity_id": prop_id,
                "constraint": "environmental_context",
                "parameters": {
                    "surface_type": env_rel.get("surface", "solid"),
                    "attachment_type": env_rel.get("attachment", "none"),
                    "contextual_relevance": env_rel.get("relevance", "high"),
                    "scene_coherence": True  # 场景连贯性
                },
                "priority": "low"
            })

        # 时间相关约束
        if "temporal_behavior" in prop:
            temp = prop["temporal_behavior"]
            constraints.append({
                "type": "prop_position",
                "entity_id": prop_id,
                "constraint": "temporal_consistency",
                "parameters": {
                    "movement_pattern": temp.get("pattern", "static"),
                    "periodicity": temp.get("periodicity", None),
                    "phase_locking": temp.get("phase_lock", False),
                    "drift_correction": temp.get("drift_correction", True)
                },
                "priority": "medium"
            })

        return constraints

    def _generate_environment_lighting_constraints(self, environment: Dict) -> List[Dict]:
        """生成环境光照约束"""
        constraints = []

        # 全局光照约束
        if "global_lighting" in environment:
            global_light = environment["global_lighting"]
            constraints.append({
                "type": "environment_lighting",
                "constraint": "global_light_consistency",
                "parameters": {
                    "intensity": global_light.get("intensity", 1.0),
                    "color_temperature": global_light.get("color_temp", 6500),  # 色温
                    "direction_consistency": global_light.get("direction_lock", True),
                    "shadow_quality": global_light.get("shadow_quality", "high")
                },
                "priority": "critical"
            })

        # 关键光源约束
        if "key_lights" in environment:
            for i, light in enumerate(environment["key_lights"]):
                constraints.append({
                    "type": "environment_lighting",
                    "constraint": "key_light_consistency",
                    "parameters": {
                        "light_id": f"key_light_{i}",
                        "position": light.get("position"),
                        "intensity": light.get("intensity", 1.0),
                        "color": light.get("color", [1, 1, 1]),
                        "falloff_curve": light.get("falloff", "quadratic"),
                        "specular_contribution": light.get("specular", 0.5)
                    },
                    "priority": "high"
                })

        # 填充光约束
        if "fill_lights" in environment:
            for i, light in enumerate(environment["fill_lights"]):
                constraints.append({
                    "type": "environment_lighting",
                    "constraint": "fill_light_consistency",
                    "parameters": {
                        "light_id": f"fill_light_{i}",
                        "softness": light.get("softness", 0.7),
                        "color_bias": light.get("color_bias", "neutral"),
                        "shadow_fill": light.get("shadow_fill", 0.3),
                        "ambient_contribution": light.get("ambient", 0.2)
                    },
                    "priority": "medium"
                })

        # 环境光遮蔽约束
        constraints.append({
            "type": "environment_lighting",
            "constraint": "ambient_occlusion_consistency",
            "parameters": {
                "ao_intensity": environment.get("ao_intensity", 0.5),
                "ao_radius": environment.get("ao_radius", 0.5),
                "ao_quality": environment.get("ao_quality", "medium"),
                "contact_shadows": environment.get("contact_shadows", True)
            },
            "priority": "medium"
        })

        # 反射约束
        if "reflections" in environment:
            reflections = environment["reflections"]
            constraints.append({
                "type": "environment_lighting",
                "constraint": "reflection_consistency",
                "parameters": {
                    "reflection_intensity": reflections.get("intensity", 0.3),
                    "reflection_blur": reflections.get("blur", 0.1),
                    "environment_map_consistency": reflections.get("env_map_lock", True),
                    "fresnel_effect": reflections.get("fresnel", True)
                },
                "priority": "low"
            })

        # 体积光约束
        if "volumetric_lighting" in environment:
            vol_light = environment["volumetric_lighting"]
            constraints.append({
                "type": "environment_lighting",
                "constraint": "volumetric_light_consistency",
                "parameters": {
                    "density": vol_light.get("density", 0.1),
                    "scattering": vol_light.get("scattering", 0.8),
                    "anisotropy": vol_light.get("anisotropy", 0.0),
                    "light_bands": vol_light.get("bands", False)
                },
                "priority": "low"
            })

        return constraints

    def _generate_camera_movement_constraints(self, camera: Dict) -> List[Dict]:
        """生成相机运动约束"""
        constraints = []

        # 基础相机参数约束
        constraints.append({
            "type": "camera_movement",
            "constraint": "camera_parameter_consistency",
            "parameters": {
                "focal_length": camera.get("focal_length", 35.0),  # 焦距（mm）
                "sensor_size": camera.get("sensor_size", [36, 24]),  # 传感器尺寸
                "aperture": camera.get("aperture", 2.8),  # 光圈
                "iso": camera.get("iso", 100),  # ISO
                "shutter_speed": camera.get("shutter_speed", 1 / 60),  # 快门速度
                "depth_of_field": camera.get("dof", True)  # 景深
            },
            "priority": "high"
        })

        # 相机位置约束
        if "position" in camera:
            constraints.append({
                "type": "camera_movement",
                "constraint": "camera_position_continuity",
                "parameters": {
                    "base_position": camera["position"],
                    "movement_smoothness": camera.get("smoothness", 0.8),
                    "max_shake": camera.get("max_shake", 0.05),  # 最大抖动
                    "movement_curve": camera.get("movement_curve", "bezier")
                },
                "priority": "high"
            })

        # 相机旋转约束
        if "rotation" in camera:
            constraints.append({
                "type": "camera_movement",
                "constraint": "camera_rotation_continuity",
                "parameters": {
                    "base_rotation": camera["rotation"],
                    "pan_smoothness": camera.get("pan_smoothness", 0.7),
                    "tilt_limit": camera.get("tilt_limit", 30.0),  # 倾斜限制（度）
                    "roll_consistency": camera.get("roll_lock", True)  # 滚动锁定
                },
                "priority": "high"
            })

        # 镜头运动约束
        if "movement_type" in camera:
            movement = camera["movement_type"]
            constraints.append({
                "type": "camera_movement",
                "constraint": "lens_movement_consistency",
                "parameters": {
                    "movement_pattern": movement,
                    "zoom_curve": camera.get("zoom_curve", "smooth"),
                    "focus_pull_speed": camera.get("focus_pull", 1.0),
                    "breathing_effect": camera.get("breathing", 0.1)  # 呼吸效应
                },
                "priority": "medium"
            })

        # 帧率与动态模糊约束
        constraints.append({
            "type": "camera_movement",
            "constraint": "temporal_consistency",
            "parameters": {
                "frame_rate": camera.get("frame_rate", 30),
                "motion_blur_samples": camera.get("motion_blur_samples", 8),
                "shutter_angle": camera.get("shutter_angle", 180.0),
                "temporal_aliasing": camera.get("temporal_aa", True)
            },
            "priority": "medium"
        })

        # 取景约束
        if "framing" in camera:
            framing = camera["framing"]
            constraints.append({
                "type": "camera_movement",
                "constraint": "framing_consistency",
                "parameters": {
                    "rule_of_thirds": framing.get("rule_of_thirds", True),
                    "headroom": framing.get("headroom", 0.1),  # 头顶空间
                    "look_space": framing.get("look_space", 0.15),  # 视线空间
                    "composition_lock": framing.get("composition_lock", False)
                },
                "priority": "medium"
            })

        # 稳定性约束
        if "stabilization" in camera:
            stab = camera["stabilization"]
            constraints.append({
                "type": "camera_movement",
                "constraint": "camera_stabilization",
                "parameters": {
                    "stabilization_type": stab.get("type", "electronic"),
                    "stabilization_strength": stab.get("strength", 0.7),
                    "horizon_lock": stab.get("horizon_lock", True),
                    "micro_jitter_reduction": stab.get("jitter_reduction", True)
                },
                "priority": "low"
            })

        return constraints

    def _generate_temporal_constraints(self, temporal_data: Dict) -> List[Dict]:
        """生成时间约束"""
        constraints = []

        # 时间流约束
        constraints.append({
            "type": "temporal_consistency",
            "constraint": "time_flow_consistency",
            "parameters": {
                "time_scale": temporal_data.get("time_scale", 1.0),
                "frame_rate_consistency": True,
                "time_dilation_consistency": True,
                "temporal_coherence": temporal_data.get("coherence", 0.9)
            },
            "priority": "critical"
        })

        # 时序事件约束
        if "timed_events" in temporal_data:
            for event in temporal_data["timed_events"]:
                constraints.append({
                    "type": "temporal_consistency",
                    "constraint": "event_timing_consistency",
                    "parameters": {
                        "event_id": event.get("id"),
                        "start_time": event.get("start"),
                        "duration": event.get("duration"),
                        "temporal_tolerance": event.get("tolerance", 0.1),  # 时间容差
                        "synchronization_group": event.get("sync_group", None)
                    },
                    "priority": "high"
                })

        # 动画时间约束
        if "animation_timing" in temporal_data:
            anim = temporal_data["animation_timing"]
            constraints.append({
                "type": "temporal_consistency",
                "constraint": "animation_timing_consistency",
                "parameters": {
                    "easing_curve": anim.get("easing", "easeInOut"),
                    "keyframe_spacing": anim.get("keyframe_spacing", "consistent"),
                    "timing_function": anim.get("timing_function", "bezier"),
                    "overshoot_control": anim.get("overshoot", 0.0)
                },
                "priority": "medium"
            })

        # 物理模拟时间约束
        constraints.append({
            "type": "temporal_consistency",
            "constraint": "physics_timing_consistency",
            "parameters": {
                "simulation_rate": temporal_data.get("sim_rate", 60),  # 模拟频率
                "substep_consistency": True,
                "delta_time_stability": temporal_data.get("delta_stability", 0.95),
                "time_step_invariance": True  # 时间步长不变性
            },
            "priority": "high"
        })

        # 时间效果约束
        if "time_effects" in temporal_data:
            effects = temporal_data["time_effects"]
            constraints.append({
                "type": "temporal_consistency",
                "constraint": "time_effect_consistency",
                "parameters": {
                    "motion_blur_consistency": effects.get("motion_blur", True),
                    "temporal_anti_aliasing": effects.get("taa", True),
                    "time_remapping": effects.get("time_remap", None),
                    "slow_motion_quality": effects.get("slow_mo_quality", "high")
                },
                "priority": "medium"
            })

        # 同步约束
        if "synchronization" in temporal_data:
            sync = temporal_data["synchronization"]
            constraints.append({
                "type": "temporal_consistency",
                "constraint": "temporal_synchronization",
                "parameters": {
                    "sync_targets": sync.get("targets", []),
                    "sync_tolerance": sync.get("tolerance", 0.01),  # 同步容差
                    "drift_correction": sync.get("drift_correction", True),
                    "clock_source": sync.get("clock", "master")
                },
                "priority": "high"
            })

        # 时间连续性约束
        constraints.append({
            "type": "temporal_consistency",
            "constraint": "temporal_continuity",
            "parameters": {
                "time_gap_tolerance": temporal_data.get("gap_tolerance", 0.5),
                "flashback_consistency": temporal_data.get("flashback_markers", True),
                "time_loop_handling": temporal_data.get("time_loop", "seamless"),
                "temporal_discontinuity_markers": True
            },
            "priority": "medium"
        })

        return constraints

    def _generate_spatial_constraints(self, spatial_data: Dict) -> List[Dict]:
        """生成空间约束"""
        constraints = []

        # 空间关系约束
        if "spatial_relationships" in spatial_data:
            for rel in spatial_data["spatial_relationships"]:
                constraints.append({
                    "type": "spatial_relationships",
                    "constraint": "maintain_spatial_relationship",
                    "parameters": {
                        "entity_a": rel.get("entity_a"),
                        "entity_b": rel.get("entity_b"),
                        "relationship_type": rel.get("type", "relative"),
                        "distance_constraint": rel.get("distance", None),
                        "angle_constraint": rel.get("angle", None),
                        "relative_orientation": rel.get("orientation", None)
                    },
                    "priority": "high"
                })

        # 比例约束
        constraints.append({
            "type": "spatial_relationships",
            "constraint": "scale_consistency",
            "parameters": {
                "world_scale": spatial_data.get("world_scale", 1.0),
                "relative_scale_consistency": True,
                "perspective_consistency": spatial_data.get("perspective_lock", True),
                "forced_perspective": spatial_data.get("forced_perspective", False)
            },
            "priority": "critical"
        })

        # 碰撞约束
        if "collision_settings" in spatial_data:
            coll = spatial_data["collision_settings"]
            constraints.append({
                "type": "spatial_relationships",
                "constraint": "collision_consistency",
                "parameters": {
                    "collision_enabled": coll.get("enabled", True),
                    "collision_precision": coll.get("precision", "high"),
                    "penetration_resolution": coll.get("penetration_resolution", "push"),
                    "continuous_collision_detection": coll.get("ccd", True)
                },
                "priority": "high"
            })

        # 物理空间约束
        constraints.append({
            "type": "spatial_relationships",
            "constraint": "physical_space_consistency",
            "parameters": {
                "gravity_direction": spatial_data.get("gravity", [0, -9.8, 0]),
                "air_resistance": spatial_data.get("air_resistance", True),
                "friction_model": spatial_data.get("friction_model", "standard"),
                "material_interaction": spatial_data.get("material_interaction", True)
            },
            "priority": "medium"
        })

        # 遮挡约束
        if "occlusion_settings" in spatial_data:
            occ = spatial_data["occlusion_settings"]
            constraints.append({
                "type": "spatial_relationships",
                "constraint": "occlusion_consistency",
                "parameters": {
                    "occlusion_culling": occ.get("culling", True),
                    "soft_occlusion": occ.get("soft_edges", True),
                    "occlusion_fade": occ.get("fade_distance", 1.0),
                    "depth_sorting": occ.get("depth_sort", "accurate")
                },
                "priority": "medium"
            })

        # 空间连续性约束
        constraints.append({
            "type": "spatial_relationships",
            "constraint": "spatial_continuity",
            "parameters": {
                "teleportation_restriction": spatial_data.get("no_teleport", True),
                "portal_consistency": spatial_data.get("portal_rules", "seamless"),
                "spatial_wrap": spatial_data.get("wrap_behavior", "none"),
                "boundary_handling": spatial_data.get("boundaries", "solid")
            },
            "priority": "high"
        })

        # 层级约束
        if "hierarchy" in spatial_data:
            hierarchy = spatial_data["hierarchy"]
            constraints.append({
                "type": "spatial_relationships",
                "constraint": "hierarchical_consistency",
                "parameters": {
                    "parent_child_relationships": hierarchy.get("relationships", {}),
                    "transformation_inheritance": hierarchy.get("inherit_transform", True),
                    "local_space_consistency": hierarchy.get("local_space", True),
                    "pivot_point_consistency": hierarchy.get("pivot_consistency", True)
                },
                "priority": "medium"
            })

        # 网格与拓扑约束
        if "mesh_topology" in spatial_data:
            mesh = spatial_data["mesh_topology"]
            constraints.append({
                "type": "spatial_relationships",
                "constraint": "topological_consistency",
                "parameters": {
                    "vertex_count_consistency": mesh.get("vertex_consistency", True),
                    "edge_flow_consistency": mesh.get("edge_flow", True),
                    "normal_consistency": mesh.get("normal_consistency", "smooth"),
                    "uv_consistency": mesh.get("uv_consistency", True)
                },
                "priority": "low"
            })

        return constraints

    def _generate_physical_constraints(self, physical_data: Dict) -> List[Dict]:
        """生成物理约束"""
        constraints = []

        # 牛顿力学约束
        constraints.append({
            "type": "physical_plausibility",
            "constraint": "newtonian_physics",
            "parameters": {
                "gravity_constant": physical_data.get("gravity", 9.8),
                "inertia_consistency": True,
                "momentum_conservation": physical_data.get("momentum_conservation", True),
                "energy_conservation": physical_data.get("energy_conservation", True)
            },
            "priority": "critical"
        })

        # 材料物理约束
        if "material_physics" in physical_data:
            mat_phys = physical_data["material_physics"]
            constraints.append({
                "type": "physical_plausibility",
                "constraint": "material_behavior",
                "parameters": {
                    "density_consistency": mat_phys.get("density_consistency", True),
                    "elasticity_range": mat_phys.get("elasticity_range", [0.1, 0.9]),
                    "plastic_deformation": mat_phys.get("plastic_deformation", False),
                    "fracture_pattern": mat_phys.get("fracture_pattern", "brittle")
                },
                "priority": "high"
            })

        # 流体动力学约束
        if "fluid_dynamics" in physical_data:
            fluid = physical_data["fluid_dynamics"]
            constraints.append({
                "type": "physical_plausibility",
                "constraint": "fluid_behavior",
                "parameters": {
                    "viscosity": fluid.get("viscosity", 1.0),
                    "surface_tension": fluid.get("surface_tension", 0.07),
                    "turbulence": fluid.get("turbulence", 0.1),
                    "splash_consistency": fluid.get("splash_consistency", True)
                },
                "priority": "medium"
            })

        # 刚体动力学约束
        constraints.append({
            "type": "physical_plausibility",
            "constraint": "rigid_body_dynamics",
            "parameters": {
                "collision_response": physical_data.get("collision_response", "realistic"),
                "friction_model": physical_data.get("friction_model", "coulomb"),
                "restitution_consistency": physical_data.get("restitution_consistency", True),
                "center_of_mass": physical_data.get("center_of_mass", "calculated")
            },
            "priority": "high"
        })

        # 软体与布料约束
        if "soft_body" in physical_data:
            soft = physical_data["soft_body"]
            constraints.append({
                "type": "physical_plausibility",
                "constraint": "soft_body_dynamics",
                "parameters": {
                    "stiffness": soft.get("stiffness", 0.5),
                    "damping": soft.get("damping", 0.1),
                    "stretch_resistance": soft.get("stretch_resistance", 0.8),
                    "bend_resistance": soft.get("bend_resistance", 0.3)
                },
                "priority": "medium"
            })

        # 热力学约束
        if "thermodynamics" in physical_data:
            thermo = physical_data["thermodynamics"]
            constraints.append({
                "type": "physical_plausibility",
                "constraint": "thermodynamic_consistency",
                "parameters": {
                    "temperature_effects": thermo.get("temperature_effects", False),
                    "thermal_expansion": thermo.get("thermal_expansion", 0.0),
                    "heat_conduction": thermo.get("heat_conduction", False),
                    "phase_transitions": thermo.get("phase_transitions", False)
                },
                "priority": "low"
            })

        # 光学物理约束
        constraints.append({
            "type": "physical_plausibility",
            "constraint": "optical_physics",
            "parameters": {
                "refraction_index": physical_data.get("refraction_index", 1.0),
                "diffraction_effects": physical_data.get("diffraction", False),
                "polarization": physical_data.get("polarization", False),
                "dispersion": physical_data.get("dispersion", False)
            },
            "priority": "low"
        })

        # 声音物理约束
        if "acoustics" in physical_data:
            acoustics = physical_data["acoustics"]
            constraints.append({
                "type": "physical_plausibility",
                "constraint": "acoustic_consistency",
                "parameters": {
                    "sound_speed": acoustics.get("sound_speed", 343.0),
                    "doppler_effect": acoustics.get("doppler_effect", True),
                    "absorption_coefficients": acoustics.get("absorption", {}),
                    "reverberation": acoustics.get("reverberation", True)
                },
                "priority": "low"
            })

        # 生物力学约束
        if "biomechanics" in physical_data:
            bio = physical_data["biomechanics"]
            constraints.append({
                "type": "physical_plausibility",
                "constraint": "biomechanical_consistency",
                "parameters": {
                    "joint_limits": bio.get("joint_limits", True),
                    "muscle_simulation": bio.get("muscle_sim", False),
                    "balance_physics": bio.get("balance", True),
                    "fatigue_model": bio.get("fatigue", False)
                },
                "priority": "medium"
            })

        return constraints

    def _generate_narrative_constraints(self, narrative_data: Dict) -> List[Dict]:
        """生成叙事约束"""
        constraints = []

        # 角色发展约束
        if "character_development" in narrative_data:
            char_dev = narrative_data["character_development"]
            constraints.append({
                "type": "narrative_continuity",
                "constraint": "character_arc_consistency",
                "parameters": {
                    "arc_progression": char_dev.get("arc_progression", "linear"),
                    "personality_consistency": char_dev.get("personality_lock", True),
                    "motivation_tracking": char_dev.get("motivation_tracking", True),
                    "growth_markers": char_dev.get("growth_markers", [])
                },
                "priority": "high"
            })

        # 情节连续性约束
        constraints.append({
            "type": "narrative_continuity",
            "constraint": "plot_continuity",
            "parameters": {
                "causal_chain": narrative_data.get("causal_chain", True),
                "plot_hole_prevention": narrative_data.get("plot_hole_check", True),
                "foreshadowing_consistency": narrative_data.get("foreshadowing", True),
                "payoff_tracking": narrative_data.get("payoff_tracking", True)
            },
            "priority": "critical"
        })

        # 时间线约束
        if "timeline" in narrative_data:
            timeline = narrative_data["timeline"]
            constraints.append({
                "type": "narrative_continuity",
                "constraint": "timeline_consistency",
                "parameters": {
                    "chronology": timeline.get("chronology", "linear"),
                    "flashback_integration": timeline.get("flashback_rules", "marked"),
                    "time_travel_rules": timeline.get("time_travel", "none"),
                    "aging_consistency": timeline.get("aging", "realistic")
                },
                "priority": "high"
            })

        # 对话连续性约束
        if "dialogue" in narrative_data:
            dialogue = narrative_data["dialogue"]
            constraints.append({
                "type": "narrative_continuity",
                "constraint": "dialogue_consistency",
                "parameters": {
                    "voice_consistency": dialogue.get("voice_consistency", True),
                    "accent_consistency": dialogue.get("accent_lock", True),
                    "speech_pattern": dialogue.get("speech_pattern", "consistent"),
                    "conversation_flow": dialogue.get("conversation_flow", "natural")
                },
                "priority": "high"
            })

        # 主题一致性约束
        constraints.append({
            "type": "narrative_continuity",
            "constraint": "thematic_consistency",
            "parameters": {
                "theme_development": narrative_data.get("theme_dev", "progressive"),
                "symbolism_consistency": narrative_data.get("symbolism", True),
                "tone_consistency": narrative_data.get("tone_lock", True),
                "mood_transitions": narrative_data.get("mood_transitions", "gradual")
            },
            "priority": "medium"
        })

        # 环境叙事约束
        if "environmental_storytelling" in narrative_data:
            env_story = narrative_data["environmental_storytelling"]
            constraints.append({
                "type": "narrative_continuity",
                "constraint": "environmental_narrative",
                "parameters": {
                    "set_dressing_consistency": env_story.get("set_dressing", True),
                    "prop_narrative": env_story.get("prop_story", True),
                    "lighting_mood": env_story.get("lighting_mood", True),
                    "atmosphere_consistency": env_story.get("atmosphere", True)
                },
                "priority": "medium"
            })

        # 冲突与解决约束
        if "conflict_resolution" in narrative_data:
            conflict = narrative_data["conflict_resolution"]
            constraints.append({
                "type": "narrative_continuity",
                "constraint": "conflict_consistency",
                "parameters": {
                    "conflict_escalation": conflict.get("escalation", "logical"),
                    "resolution_satisfaction": conflict.get("resolution_satisfaction", True),
                    "stakes_consistency": conflict.get("stakes", True),
                    "tension_curve": conflict.get("tension_curve", "standard")
                },
                "priority": "high"
            })

        # 观众认知约束
        constraints.append({
            "type": "narrative_continuity",
            "constraint": "audience_cognition",
            "parameters": {
                "exposition_timing": narrative_data.get("exposition_timing", "balanced"),
                "information_reveal": narrative_data.get("info_reveal", "progressive"),
                "cognitive_load": narrative_data.get("cognitive_load", "moderate"),
                "suspension_of_disbelief": narrative_data.get("suspension", 0.8)
            },
            "priority": "medium"
        })

        # 类型约束
        if "genre_constraints" in narrative_data:
            genre = narrative_data["genre_constraints"]
            constraints.append({
                "type": "narrative_continuity",
                "constraint": "genre_consistency",
                "parameters": {
                    "genre_conventions": genre.get("conventions", []),
                    "trope_usage": genre.get("trope_usage", "appropriate"),
                    "pacing_by_genre": genre.get("genre_pacing", True),
                    "expectation_management": genre.get("expectations", True)
                },
                "priority": "high"
            })

        # 文化一致性约束
        if "cultural_consistency" in narrative_data:
            cultural = narrative_data["cultural_consistency"]
            constraints.append({
                "type": "narrative_continuity",
                "constraint": "cultural_authenticity",
                "parameters": {
                    "historical_accuracy": cultural.get("historical", "contextual"),
                    "cultural_sensitivity": cultural.get("sensitivity", True),
                    "language_accuracy": cultural.get("language", True),
                    "custom_consistency": cultural.get("customs", True)
                },
                "priority": "medium"
            })

        return constraints

    # 新增：验证约束适用性的完整实现
    def _is_constraint_applicable(self, scene_data: Dict, constraint: Dict) -> bool:
        """检查约束是否适用于当前场景"""
        constraint_type = constraint.get("type", "")
        entity_id = constraint.get("entity_id", "")
        priority = constraint.get("priority", "medium")

        # 根据约束类型进行检查
        if constraint_type.startswith("character_"):
            # 角色相关约束：检查角色是否存在
            if entity_id == "global":
                return any(c for c in scene_data.get("characters", []))
            else:
                return any(c["id"] == entity_id for c in scene_data.get("characters", []))

        elif constraint_type.startswith("prop_"):
            # 道具相关约束：检查道具是否存在
            if entity_id == "global":
                return any(p for p in scene_data.get("props", []))
            else:
                return any(p["id"] == entity_id for p in scene_data.get("props", []))

        elif constraint_type.startswith("environment_"):
            # 环境相关约束：检查环境数据是否存在
            return "environment" in scene_data

        elif constraint_type.startswith("camera_"):
            # 相机相关约束：检查相机数据是否存在
            return "camera" in scene_data

        elif constraint_type.startswith("temporal_"):
            # 时间相关约束：需要时间数据
            return any(k in scene_data for k in ["timing", "frame_rate", "time_progression"])

        elif constraint_type.startswith("spatial_"):
            # 空间相关约束：需要空间数据
            return any(k in scene_data for k in ["spatial_data", "positions", "bounding_volumes"])

        elif constraint_type.startswith("physical_"):
            # 物理相关约束：需要物理数据
            return any(k in scene_data for k in ["physics", "dynamics", "collisions"])

        elif constraint_type.startswith("narrative_"):
            # 叙事相关约束：需要叙事数据
            return any(k in scene_data for k in ["narrative", "story", "plot"])

        # 默认情况：如果约束没有特定要求，则适用
        return True

    # 新增：查找约束冲突的完整实现
    def _find_constraint_conflicts(self, constraints: List[Dict]) -> List[Tuple[int, int, str]]:
        """查找约束之间的冲突"""
        conflicts = []

        # 按约束类型分组
        constraint_groups = {}
        for i, constraint in enumerate(constraints):
            constraint_type = constraint.get("type", "unknown")
            if constraint_type not in constraint_groups:
                constraint_groups[constraint_type] = []
            constraint_groups[constraint_type].append((i, constraint))

        # 检查同类型约束间的冲突
        for constraint_type, group_constraints in constraint_groups.items():
            if len(group_constraints) < 2:
                continue

            # 检查位置约束冲突
            if constraint_type in ["prop_position", "character_position"]:
                conflicts.extend(self._find_position_conflicts(group_constraints))

            # 检查时间约束冲突
            elif constraint_type in ["temporal_consistency", "event_timing"]:
                conflicts.extend(self._find_timing_conflicts(group_constraints))

            # 检查物理约束冲突
            elif constraint_type in ["physical_plausibility", "physics_timing"]:
                conflicts.extend(self._find_physics_conflicts(group_constraints))

        # 检查不同类型约束间的冲突
        for i, constraint1 in enumerate(constraints):
            for j, constraint2 in enumerate(constraints[i + 1:], i + 1):
                if self._constraints_conflict(constraint1, constraint2):
                    conflicts.append((i, j, f"约束 {constraint1.get('type')} 与 {constraint2.get('type')} 冲突"))

        return conflicts

    def _find_position_conflicts(self, constraints: List[Tuple[int, Dict]]) -> List[Tuple[int, int, str]]:
        """查找位置约束冲突"""
        conflicts = []

        # 收集所有实体的位置约束
        position_constraints = {}
        for idx, constraint in constraints:
            entity_id = constraint.get("entity_id")
            if entity_id and entity_id != "global":
                if entity_id not in position_constraints:
                    position_constraints[entity_id] = []
                position_constraints[entity_id].append((idx, constraint))

        # 检查同一实体的位置约束冲突
        for entity_id, entity_constraints in position_constraints.items():
            if len(entity_constraints) < 2:
                continue

            # 检查是否有相互矛盾的位置要求
            base_positions = []
            movement_limits = []

            for idx, constraint in entity_constraints:
                params = constraint.get("parameters", {})
                if "base_position" in params:
                    base_positions.append((idx, params["base_position"]))
                if "movement_threshold" in params:
                    movement_limits.append((idx, params["movement_threshold"]))

            # 检查基础位置冲突
            if len(base_positions) > 1:
                for k in range(len(base_positions)):
                    for l in range(k + 1, len(base_positions)):
                        idx1, pos1 = base_positions[k]
                        idx2, pos2 = base_positions[l]
                        if self._calculate_position_distance(pos1, pos2) > 0.1:
                            conflicts.append((idx1, idx2,
                                              f"实体 {entity_id} 的基础位置冲突"))

            # 检查移动限制冲突
            if len(movement_limits) > 1:
                thresholds = [limit for _, limit in movement_limits]
                if max(thresholds) - min(thresholds) > 1.0:
                    for k in range(len(movement_limits)):
                        for l in range(k + 1, len(movement_limits)):
                            idx1, thresh1 = movement_limits[k]
                            idx2, thresh2 = movement_limits[l]
                            if abs(thresh1 - thresh2) > 0.5:
                                conflicts.append((idx1, idx2,
                                                  f"实体 {entity_id} 的移动限制冲突"))

        return conflicts

    def _find_timing_conflicts(self, constraints: List[Tuple[int, Dict]]) -> List[Tuple[int, int, str]]:
        """查找时间约束冲突"""
        conflicts = []

        # 收集所有时间事件
        timed_events = []
        for idx, constraint in constraints:
            params = constraint.get("parameters", {})
            if "start_time" in params and "duration" in params:
                timed_events.append((idx, params["start_time"], params["duration"]))

        # 检查时间重叠
        for i in range(len(timed_events)):
            for j in range(i + 1, len(timed_events)):
                idx1, start1, duration1 = timed_events[i]
                idx2, start2, duration2 = timed_events[j]

                end1 = start1 + duration1
                end2 = start2 + duration2

                # 检查重叠
                if not (end1 <= start2 or end2 <= start1):
                    conflicts.append((idx1, idx2, "时间事件重叠冲突"))

        return conflicts

    def _find_physics_conflicts(self, constraints: List[Tuple[int, Dict]]) -> List[Tuple[int, int, str]]:
        """查找物理约束冲突"""
        conflicts = []

        physics_settings = {}
        for idx, constraint in constraints:
            params = constraint.get("parameters", {})

            # 检查重力设置冲突
            if "gravity_constant" in params:
                gravity = params["gravity_constant"]
                if "gravity" not in physics_settings:
                    physics_settings["gravity"] = []
                physics_settings["gravity"].append((idx, gravity))

            # 检查物理模型冲突
            if "collision_response" in params:
                response = params["collision_response"]
                if "collision_response" not in physics_settings:
                    physics_settings["collision_response"] = []
                physics_settings["collision_response"].append((idx, response))

        # 检查重力冲突
        if "gravity" in physics_settings and len(physics_settings["gravity"]) > 1:
            gravities = [g for _, g in physics_settings["gravity"]]
            if max(gravities) - min(gravities) > 2.0:
                for i in range(len(physics_settings["gravity"])):
                    for j in range(i + 1, len(physics_settings["gravity"])):
                        idx1, g1 = physics_settings["gravity"][i]
                        idx2, g2 = physics_settings["gravity"][j]
                        if abs(g1 - g2) > 1.0:
                            conflicts.append((idx1, idx2, "重力设置冲突"))

        # 检查碰撞响应冲突
        if "collision_response" in physics_settings and len(physics_settings["collision_response"]) > 1:
            responses = [r for _, r in physics_settings["collision_response"]]
            if len(set(responses)) > 1:
                for i in range(len(physics_settings["collision_response"])):
                    for j in range(i + 1, len(physics_settings["collision_response"])):
                        idx1, r1 = physics_settings["collision_response"][i]
                        idx2, r2 = physics_settings["collision_response"][j]
                        if r1 != r2:
                            conflicts.append((idx1, idx2, "碰撞响应模型冲突"))

        return conflicts

    def _constraints_conflict(self, constraint1: Dict, constraint2: Dict) -> bool:
        """检查两个约束是否冲突"""
        type1 = constraint1.get("type", "")
        type2 = constraint2.get("type", "")

        # 特定类型冲突检测
        conflict_patterns = [
            # 位置约束与运动约束冲突
            ({"prop_position", "character_position"}, {"camera_movement"},
             lambda c1, c2: self._check_position_movement_conflict(c1, c2)),

            # 物理约束与动画约束冲突
            ({"physical_plausibility"}, {"temporal_consistency", "animation_timing"},
             lambda c1, c2: self._check_physics_animation_conflict(c1, c2)),

            # 时间约束与叙事约束冲突
            ({"temporal_consistency"}, {"narrative_continuity"},
             lambda c1, c2: self._check_time_narrative_conflict(c1, c2))
        ]

        for types1, types2, check_func in conflict_patterns:
            if type1 in types1 and type2 in types2:
                return check_func(constraint1, constraint2)
            if type2 in types1 and type1 in types2:
                return check_func(constraint2, constraint1)

        return False

    def _check_position_movement_conflict(self, position_constraint: Dict,
                                          movement_constraint: Dict) -> bool:
        """检查位置约束与运动约束的冲突"""
        pos_params = position_constraint.get("parameters", {})
        move_params = movement_constraint.get("parameters", {})

        # 如果位置约束要求固定位置，但运动约束要求自由移动
        if (pos_params.get("movement_threshold", 0.0) < 0.1 and
                move_params.get("movement_smoothness", 0.0) > 0.5):
            return True

        # 如果位置约束有严格的边界，但运动约束允许大范围移动
        if (pos_params.get("boundary_handling") == "strict" and
                move_params.get("movement_range", 100.0) > 10.0):
            return True

        return False

    def _check_physics_animation_conflict(self, physics_constraint: Dict,
                                          animation_constraint: Dict) -> bool:
        """检查物理约束与动画约束的冲突"""
        physics_params = physics_constraint.get("parameters", {})
        animation_params = animation_constraint.get("parameters", {})

        # 如果物理要求动量守恒，但动画允许违反物理规律
        if (physics_params.get("momentum_conservation", True) and
                animation_params.get("overshoot_control", 0.0) > 0.5):
            return True

        # 如果物理要求刚体行为，但动画要求弹性变形
        if (physics_params.get("collision_response") == "rigid" and
                animation_params.get("easing_curve") == "bounce"):
            return True

        return False

    def _check_time_narrative_conflict(self, time_constraint: Dict,
                                       narrative_constraint: Dict) -> bool:
        """检查时间约束与叙事约束的冲突"""
        time_params = time_constraint.get("parameters", {})
        narrative_params = narrative_constraint.get("parameters", {})

        # 如果时间要求线性进展，但叙事要求时间跳跃
        if (time_params.get("time_flow_consistency") == "strict" and
                narrative_params.get("chronology") in ["nonlinear", "fragmented"]):
            return True

        # 如果时间要求固定帧率，但叙事要求慢动作
        if (time_params.get("frame_rate_consistency", True) and
                narrative_params.get("tension_curve") == "slow_motion_heavy"):
            return True

        return False

    def _calculate_position_distance(self, pos1, pos2) -> float:
        """计算位置距离"""
        if isinstance(pos1, (list, tuple)) and isinstance(pos2, (list, tuple)):
            if len(pos1) >= 3 and len(pos2) >= 3:
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                dz = pos2[2] - pos1[2]
                return math.sqrt(dx * dx + dy * dy + dz * dz)
        return float('inf')

    # 新增：约束优先级调整方法
    def adjust_constraint_priorities(self, constraints: List[Dict],
                                     scene_context: Dict) -> List[Dict]:
        """根据场景上下文调整约束优先级"""
        adjusted_constraints = []

        scene_type = scene_context.get("scene_type", "general")
        time_of_day = scene_context.get("time_of_day", "day")
        importance_level = scene_context.get("importance", "medium")

        for constraint in constraints:
            adjusted_constraint = constraint.copy()
            original_priority = constraint.get("priority", "medium")

            # 根据场景类型调整优先级
            if scene_type in ["action", "fight", "chase"]:
                # 动作场景：物理和运动约束更重要
                if constraint["type"] in ["physical_plausibility", "motion_continuity"]:
                    adjusted_constraint["priority"] = self._increase_priority(original_priority)
                elif constraint["type"] in ["narrative_continuity", "thematic_consistency"]:
                    adjusted_constraint["priority"] = self._decrease_priority(original_priority)

            elif scene_type in ["dialogue", "emotional"]:
                # 对话/情感场景：角色和叙事约束更重要
                if constraint["type"].startswith("character_") or constraint["type"].startswith("narrative_"):
                    adjusted_constraint["priority"] = self._increase_priority(original_priority)
                elif constraint["type"] in ["physical_plausibility", "camera_movement"]:
                    adjusted_constraint["priority"] = self._decrease_priority(original_priority)

            # 根据时间调整光照相关约束
            if time_of_day in ["night", "dusk", "dawn"]:
                if constraint["type"].startswith("environment_lighting"):
                    adjusted_constraint["priority"] = self._increase_priority(original_priority)

            # 根据重要性级别调整
            if importance_level == "high":
                # 重要场景：所有约束优先级提高
                adjusted_constraint["priority"] = self._increase_priority(original_priority)
            elif importance_level == "low":
                # 次要场景：所有约束优先级降低
                adjusted_constraint["priority"] = self._decrease_priority(original_priority)

            adjusted_constraints.append(adjusted_constraint)

        return adjusted_constraints

    def _increase_priority(self, priority: str) -> str:
        """提高优先级"""
        priority_order = ["low", "medium", "high", "critical"]
        current_index = priority_order.index(priority) if priority in priority_order else 1
        new_index = min(current_index + 1, len(priority_order) - 1)
        return priority_order[new_index]

    def _decrease_priority(self, priority: str) -> str:
        """降低优先级"""
        priority_order = ["low", "medium", "high", "critical"]
        current_index = priority_order.index(priority) if priority in priority_order else 1
        new_index = max(current_index - 1, 0)
        return priority_order[new_index]

    # 新增：约束优化方法
    def optimize_constraints(self, constraints: List[Dict],
                             optimization_goal: str = "performance") -> List[Dict]:
        """优化约束集合"""
        if optimization_goal == "performance":
            return self._optimize_for_performance(constraints)
        elif optimization_goal == "quality":
            return self._optimize_for_quality(constraints)
        elif optimization_goal == "balanced":
            return self._optimize_balanced(constraints)
        else:
            return constraints

    def _optimize_for_performance(self, constraints: List[Dict]) -> List[Dict]:
        """为性能优化约束"""
        optimized = []

        # 移除低优先级约束
        for constraint in constraints:
            priority = constraint.get("priority", "medium")
            if priority in ["high", "critical"]:
                optimized.append(constraint)
            elif priority == "medium":
                # 保留50%的中等优先级约束
                if random.random() < 0.5:
                    optimized.append(constraint)
            # 低优先级约束全部移除

        # 合并相似约束
        optimized = self._merge_similar_constraints(optimized)

        return optimized

    def _optimize_for_quality(self, constraints: List[Dict]) -> List[Dict]:
        """为质量优化约束"""
        optimized = []

        # 保留所有约束
        optimized.extend(constraints)

        # 添加额外的质量相关约束
        quality_constraints = self._generate_quality_constraints(constraints)
        optimized.extend(quality_constraints)

        # 确保没有冲突
        conflicts = self._find_constraint_conflicts(optimized)
        if conflicts:
            # 解决冲突：保留优先级更高的约束
            optimized = self._resolve_conflicts(optimized, conflicts)

        return optimized

    def _optimize_balanced(self, constraints: List[Dict]) -> List[Dict]:
        """平衡优化"""
        optimized = []

        # 按优先级分组
        priority_groups = {"critical": [], "high": [], "medium": [], "low": []}
        for constraint in constraints:
            priority = constraint.get("priority", "medium")
            priority_groups[priority].append(constraint)

        # 保留所有critical和high优先级约束
        optimized.extend(priority_groups["critical"])
        optimized.extend(priority_groups["high"])

        # 保留75%的medium优先级约束
        medium_count = len(priority_groups["medium"])
        keep_count = int(medium_count * 0.75)
        optimized.extend(random.sample(priority_groups["medium"], min(keep_count, medium_count)))

        # 保留25%的low优先级约束
        low_count = len(priority_groups["low"])
        keep_count = int(low_count * 0.25)
        optimized.extend(random.sample(priority_groups["low"], min(keep_count, low_count)))

        return optimized

    def _merge_similar_constraints(self, constraints: List[Dict]) -> List[Dict]:
        """合并相似约束"""
        merged = []
        merged_indices = set()

        for i in range(len(constraints)):
            if i in merged_indices:
                continue

            constraint_i = constraints[i]
            similar_found = False

            for j in range(i + 1, len(constraints)):
                if j in merged_indices:
                    continue

                constraint_j = constraints[j]

                if self._are_constraints_similar(constraint_i, constraint_j):
                    # 合并约束
                    merged_constraint = self._merge_two_constraints(constraint_i, constraint_j)
                    merged.append(merged_constraint)
                    merged_indices.add(i)
                    merged_indices.add(j)
                    similar_found = True
                    break

            if not similar_found:
                merged.append(constraint_i)
                merged_indices.add(i)

        return merged

    def _are_constraints_similar(self, constraint1: Dict, constraint2: Dict) -> bool:
        """判断两个约束是否相似"""
        # 类型相同
        if constraint1["type"] != constraint2["type"]:
            return False

        # 实体相同或都是全局约束
        entity1 = constraint1.get("entity_id", "")
        entity2 = constraint2.get("entity_id", "")
        if entity1 != entity2 and not (entity1 == "global" and entity2 == "global"):
            return False

        # 约束条件相似
        params1 = constraint1.get("parameters", {})
        params2 = constraint2.get("parameters", {})

        # 检查关键参数相似度
        key_params = ["base_position", "intensity", "time_scale", "gravity_constant"]
        for key in key_params:
            if key in params1 and key in params2:
                val1 = params1[key]
                val2 = params2[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if abs(val1 - val2) > 0.1 * max(abs(val1), abs(val2)):
                        return False

        return True

    def _merge_two_constraints(self, constraint1: Dict, constraint2: Dict) -> Dict:
        """合并两个约束"""
        merged = constraint1.copy()

        # 合并参数：取更严格的参数值
        params1 = constraint1.get("parameters", {})
        params2 = constraint2.get("parameters", {})

        merged_params = params1.copy()
        for key, val2 in params2.items():
            if key in merged_params:
                val1 = merged_params[key]

                # 数值参数：取平均值
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    merged_params[key] = (val1 + val2) / 2

                # 布尔参数：取逻辑与
                elif isinstance(val1, bool) and isinstance(val2, bool):
                    merged_params[key] = val1 and val2

                # 字符串参数：如果不同，使用第一个
                elif isinstance(val1, str) and isinstance(val2, str):
                    if val1 != val2:
                        merged_params[key] = val1  # 保留第一个
            else:
                merged_params[key] = val2

        merged["parameters"] = merged_params

        # 调整优先级：取更高的优先级
        priority_order = ["critical", "high", "medium", "low"]
        pri1 = constraint1.get("priority", "medium")
        pri2 = constraint2.get("priority", "medium")

        idx1 = priority_order.index(pri1) if pri1 in priority_order else 2
        idx2 = priority_order.index(pri2) if pri2 in priority_order else 2

        merged["priority"] = priority_order[min(idx1, idx2)]

        return merged

    def _generate_quality_constraints(self, base_constraints: List[Dict]) -> List[Dict]:
        """生成额外的质量相关约束"""
        quality_constraints = []

        # 检测可能需要额外质量约束的情况
        has_characters = any(c["type"].startswith("character_") for c in base_constraints)
        has_environment = any(c["type"].startswith("environment_") for c in base_constraints)
        has_physics = any(c["type"].startswith("physical_") for c in base_constraints)

        if has_characters:
            # 添加角色细节约束
            quality_constraints.append({
                "type": "character_appearance",
                "constraint": "high_detail_facial_expressions",
                "parameters": {
                    "micro_expression_detail": "high",
                    "eye_moisture": True,
                    "subsurface_scattering": True,
                    "pore_detail": "medium"
                },
                "priority": "medium"
            })

        if has_environment:
            # 添加环境细节约束
            quality_constraints.append({
                "type": "environment_lighting",
                "constraint": "advanced_lighting_effects",
                "parameters": {
                    "global_illumination": True,
                    "ray_tracing": "partial",
                    "volumetric_fog": True,
                    "screen_space_reflections": True
                },
                "priority": "medium"
            })

        if has_physics:
            # 添加物理细节约束
            quality_constraints.append({
                "type": "physical_plausibility",
                "constraint": "advanced_physics_simulation",
                "parameters": {
                    "cloth_simulation_quality": "high",
                    "fluid_simulation": True,
                    "destruction_detail": "medium",
                    "particle_count": 10000
                },
                "priority": "medium"
            })

        return quality_constraints

    def _resolve_conflicts(self, constraints: List[Dict],
                           conflicts: List[Tuple[int, int, str]]) -> List[Dict]:
        """解决约束冲突"""
        # 创建冲突图
        conflict_graph = defaultdict(set)
        for idx1, idx2, _ in conflicts:
            conflict_graph[idx1].add(idx2)
            conflict_graph[idx2].add(idx1)

        # 按优先级排序约束索引
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        constraint_priorities = [
            (i, priority_order.get(constraints[i].get("priority", "medium"), 2))
            for i in range(len(constraints))
        ]
        constraint_priorities.sort(key=lambda x: x[1])

        # 选择要保留的约束
        selected = set()
        removed = set()

        for idx, _ in constraint_priorities:
            if idx in removed:
                continue

            # 检查与已选约束的冲突
            conflicting_with_selected = any(conflict in selected for conflict in conflict_graph[idx])

            if not conflicting_with_selected:
                selected.add(idx)
            else:
                # 移除当前约束（保留更高优先级的冲突约束）
                removed.add(idx)

        # 返回选中的约束
        return [constraints[i] for i in range(len(constraints)) if i in selected]

    # 新增：约束验证和报告方法
    def validate_constraint_set(self, constraints: List[Dict],
                                scene_data: Dict) -> Dict[str, Any]:
        """验证约束集的有效性"""
        validation_result = {
            "total_constraints": len(constraints),
            "applicable_constraints": 0,
            "inapplicable_constraints": 0,
            "conflicting_constraints": 0,
            "coverage_score": 0.0,
            "efficiency_score": 0.0,
            "recommendations": [],
            "detailed_analysis": {}
        }

        # 检查适用性
        applicable = []
        inapplicable = []
        for constraint in constraints:
            if self._is_constraint_applicable(scene_data, constraint):
                applicable.append(constraint)
            else:
                inapplicable.append(constraint)

        validation_result["applicable_constraints"] = len(applicable)
        validation_result["inapplicable_constraints"] = len(inapplicable)

        # 检查冲突
        conflicts = self._find_constraint_conflicts(applicable)
        validation_result["conflicting_constraints"] = len(conflicts)

        # 计算覆盖率
        scene_elements = self._count_scene_elements(scene_data)
        covered_elements = self._count_covered_elements(applicable, scene_data)

        if scene_elements["total"] > 0:
            validation_result["coverage_score"] = covered_elements["total"] / scene_elements["total"]

        # 计算效率分数
        validation_result["efficiency_score"] = self._calculate_efficiency_score(
            applicable, conflicts, scene_elements
        )

        # 生成详细分析
        validation_result["detailed_analysis"] = {
            "constraint_distribution": self._analyze_constraint_distribution(applicable),
            "conflict_analysis": self._analyze_conflicts(conflicts, applicable),
            "coverage_gaps": self._identify_coverage_gaps(applicable, scene_data)
        }

        # 生成建议
        validation_result["recommendations"] = self._generate_validation_recommendations(
            validation_result, applicable, inapplicable, conflicts
        )

        return validation_result

    def _count_scene_elements(self, scene_data: Dict) -> Dict[str, int]:
        """统计场景元素"""
        counts = {
            "characters": len(scene_data.get("characters", [])),
            "props": len(scene_data.get("props", [])),
            "lights": len(scene_data.get("environment", {}).get("lights", [])),
            "cameras": 1 if "camera" in scene_data else 0,
            "total": 0
        }

        counts["total"] = sum(counts.values())
        return counts

    def _count_covered_elements(self, constraints: List[Dict], scene_data: Dict) -> Dict[str, int]:
        """统计被约束覆盖的元素"""
        covered = {
            "characters": set(),
            "props": set(),
            "lights": set(),
            "cameras": set(),
            "total": 0
        }

        for constraint in constraints:
            entity_id = constraint.get("entity_id", "")
            constraint_type = constraint.get("type", "")

            if entity_id == "global":
                # 全局约束覆盖所有元素
                for char in scene_data.get("characters", []):
                    covered["characters"].add(char.get("id"))
                for prop in scene_data.get("props", []):
                    covered["props"].add(prop.get("id"))
                # 添加其他元素...
            else:
                # 特定实体约束
                if constraint_type.startswith("character_"):
                    covered["characters"].add(entity_id)
                elif constraint_type.startswith("prop_"):
                    covered["props"].add(entity_id)
                elif constraint_type.startswith("environment_lighting"):
                    # 光照约束可能涉及特定灯光
                    covered["lights"].add(entity_id)
                elif constraint_type.startswith("camera_"):
                    covered["cameras"].add(entity_id)

        # 计算总数
        covered["total"] = (len(covered["characters"]) + len(covered["props"]) +
                            len(covered["lights"]) + len(covered["cameras"]))

        return covered

    def _calculate_efficiency_score(self, constraints: List[Dict],
                                    conflicts: List[Tuple],
                                    scene_elements: Dict) -> float:
        """计算约束集效率分数"""
        if len(constraints) == 0:
            return 1.0

        # 冲突惩罚
        conflict_penalty = len(conflicts) / max(1, len(constraints))

        # 冗余惩罚（基于相似约束数量）
        similarity_groups = defaultdict(list)
        for constraint in constraints:
            key = f"{constraint['type']}_{constraint.get('entity_id', 'global')}"
            similarity_groups[key].append(constraint)

        redundancy = 0
        for group_constraints in similarity_groups.values():
            if len(group_constraints) > 1:
                redundancy += len(group_constraints) - 1

        redundancy_penalty = redundancy / max(1, len(constraints))

        # 覆盖奖励
        coverage_reward = min(1.0, scene_elements["total"] / max(1, len(constraints) * 2))

        # 计算效率分数
        efficiency = 1.0 - conflict_penalty * 0.5 - redundancy_penalty * 0.3 + coverage_reward * 0.2

        return max(0.0, min(1.0, efficiency))

    def _analyze_constraint_distribution(self, constraints: List[Dict]) -> Dict[str, Any]:
        """分析约束分布"""
        distribution = {
            "by_type": defaultdict(int),
            "by_priority": defaultdict(int),
            "by_entity": defaultdict(int),
            "type_coverage": defaultdict(set)
        }

        for constraint in constraints:
            constraint_type = constraint.get("type", "unknown")
            priority = constraint.get("priority", "medium")
            entity_id = constraint.get("entity_id", "global")

            distribution["by_type"][constraint_type] += 1
            distribution["by_priority"][priority] += 1
            distribution["by_entity"][entity_id] += 1

            # 记录每种类型覆盖的实体
            distribution["type_coverage"][constraint_type].add(entity_id)

        # 转换为普通字典
        distribution["by_type"] = dict(distribution["by_type"])
        distribution["by_priority"] = dict(distribution["by_priority"])
        distribution["by_entity"] = dict(distribution["by_entity"])
        distribution["type_coverage"] = {k: list(v) for k, v in distribution["type_coverage"].items()}

        return distribution

    def _analyze_conflicts(self, conflicts: List[Tuple],
                           constraints: List[Dict]) -> Dict[str, Any]:
        """分析冲突"""
        conflict_analysis = {
            "by_type": defaultdict(int),
            "by_priority": defaultdict(int),
            "severe_conflicts": [],
            "minor_conflicts": []
        }

        for idx1, idx2, reason in conflicts:
            if idx1 < len(constraints) and idx2 < len(constraints):
                constraint1 = constraints[idx1]
                constraint2 = constraints[idx2]

                # 按类型统计
                type_pair = f"{constraint1['type']}-{constraint2['type']}"
                conflict_analysis["by_type"][type_pair] += 1

                # 按优先级分类
                pri1 = constraint1.get("priority", "medium")
                pri2 = constraint2.get("priority", "medium")

                if "critical" in [pri1, pri2]:
                    conflict_analysis["severe_conflicts"].append({
                        "constraint1": constraint1,
                        "constraint2": constraint2,
                        "reason": reason
                    })
                else:
                    conflict_analysis["minor_conflicts"].append({
                        "constraint1": constraint1,
                        "constraint2": constraint2,
                        "reason": reason
                    })

        conflict_analysis["by_type"] = dict(conflict_analysis["by_type"])

        return conflict_analysis

    def _identify_coverage_gaps(self, constraints: List[Dict],
                                scene_data: Dict) -> List[Dict]:
        """识别覆盖缺口"""
        gaps = []

        # 检查角色覆盖
        characters = scene_data.get("characters", [])
        character_constraints = [c for c in constraints if c["type"].startswith("character_")]

        for character in characters:
            char_id = character.get("id")
            char_constraints = [c for c in character_constraints
                                if c.get("entity_id") == char_id or c.get("entity_id") == "global"]

            if len(char_constraints) == 0:
                gaps.append({
                    "element_type": "character",
                    "element_id": char_id,
                    "missing_constraint_types": ["character_appearance", "character_motion"],
                    "severity": "high"
                })

        # 检查环境光照覆盖
        environment = scene_data.get("environment", {})
        if environment and not any(c["type"].startswith("environment_") for c in constraints):
            gaps.append({
                "element_type": "environment",
                "element_id": "global",
                "missing_constraint_types": ["environment_lighting"],
                "severity": "medium"
            })

        return gaps

    def _generate_validation_recommendations(self, validation_result: Dict,
                                             applicable: List[Dict],
                                             inapplicable: List[Dict],
                                             conflicts: List[Tuple]) -> List[str]:
        """生成验证建议"""
        recommendations = []

        # 关于不适用约束的建议
        if validation_result["inapplicable_constraints"] > 0:
            recommendations.append(
                f"移除 {validation_result['inapplicable_constraints']} 个不适用于当前场景的约束"
            )

        # 关于冲突的建议
        if validation_result["conflicting_constraints"] > 0:
            recommendations.append(
                f"解决 {validation_result['conflicting_constraints']} 个约束冲突"
            )

        # 关于覆盖率的建议
        if validation_result["coverage_score"] < 0.5:
            recommendations.append("增加约束覆盖率，确保关键场景元素都有相应约束")

        # 关于效率的建议
        if validation_result["efficiency_score"] < 0.7:
            recommendations.append("优化约束集，减少冗余和冲突，提高效率")

        # 基于约束分布的建议
        distribution = self._analyze_constraint_distribution(applicable)

        # 检查优先级分布
        priority_counts = distribution["by_priority"]
        total_constraints = len(applicable)

        if total_constraints > 0:
            critical_ratio = priority_counts.get("critical", 0) / total_constraints
            if critical_ratio > 0.3:
                recommendations.append("关键优先级约束过多，考虑将部分降级为高优先级")

            if priority_counts.get("low", 0) / total_constraints > 0.4:
                recommendations.append("低优先级约束过多，考虑移除部分以提高性能")

        return recommendations

    # 新增：约束导出和导入方法
    def export_constraints(self, constraints: List[Dict],
                           filename: str = "constraints.json") -> bool:
        """导出约束到文件"""
        try:
            import json

            export_data = {
                "constraints": constraints,
                "metadata": {
                    "export_time": datetime.now().isoformat(),
                    "constraint_count": len(constraints),
                    "generator_version": "1.0"
                }
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"导出约束失败: {e}")
            return False

    def import_constraints(self, filename: str) -> Optional[List[Dict]]:
        """从文件导入约束"""
        try:
            import json

            with open(filename, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            if "constraints" in import_data:
                return import_data["constraints"]
            else:
                return None
        except Exception as e:
            print(f"导入约束失败: {e}")
            return None

    # 新增：约束模板管理方法
    def get_available_templates(self) -> Dict[str, str]:
        """获取可用约束模板"""
        templates = {}
        for name, func in self.constraint_templates.items():
            templates[name] = func.__doc__ or "无描述"
        return templates

    def add_template(self, name: str, template_func: Callable) -> bool:
        """添加新约束模板"""
        if name in self.constraint_templates:
            return False

        self.constraint_templates[name] = template_func
        return True

    def remove_template(self, name: str) -> bool:
        """移除约束模板"""
        if name in self.constraint_templates:
            del self.constraint_templates[name]
            return True
        return False

    def get_constraint_summary(self, constraints: List[Dict]) -> Dict[str, Any]:
        """获取约束摘要"""
        summary = {
            "total_constraints": len(constraints),
            "by_type": defaultdict(int),
            "by_priority": defaultdict(int),
            "by_entity": defaultdict(int)
        }

        for constraint in constraints:
            constraint_type = constraint.get("type", "unknown")
            priority = constraint.get("priority", "medium")
            entity_id = constraint.get("entity_id", "global")

            summary["by_type"][constraint_type] += 1
            summary["by_priority"][priority] += 1
            summary["by_entity"][entity_id] += 1

        return dict(summary)

    def validate_constraints(self, scene_data: Dict, constraints: List[Dict]) -> Dict[str, Any]:
        """验证约束的适用性"""
        validation_result = {
            "applicable_constraints": [],
            "inapplicable_constraints": [],
            "conflicting_constraints": [],
            "recommendations": []
        }

        # 实现约束验证逻辑
        for constraint in constraints:
            if self._is_constraint_applicable(scene_data, constraint):
                validation_result["applicable_constraints"].append(constraint)
            else:
                validation_result["inapplicable_constraints"].append(constraint)

        # 检查冲突
        validation_result["conflicting_constraints"] = self._find_constraint_conflicts(
            validation_result["applicable_constraints"]
        )

        return validation_result
