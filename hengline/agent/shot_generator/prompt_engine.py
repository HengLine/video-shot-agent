"""
@FileName: prompt_engine.py
@Description: 提示词生成引擎
@Author: HengLine
@Time: 2026/1/5 23:03
"""
from hengline.agent.continuity_guardian.continuity_guardian_model import AnchoredSegment, HardConstraint
from .constraint_handler import ConstraintHandler
from .model.shot_models import SoraPromptStructure
from .model.style_models import StyleGuide

import random
from typing import Dict, List, Any


class PromptVocabulary:
    """提示词词汇库"""

    # 风格描述词汇
    STYLE_ADJECTIVES = {
        "cinematic": [
            "cinematic", "film-like", "movie-quality", "Hollywood-style",
            "blockbuster", "arthouse", "indie-film", "documentary-style"
        ],
        "lighting": [
            "soft natural lighting", "dramatic chiaroscuro", "high-key lighting",
            "low-key moody lighting", "golden hour glow", "blue hour ambiance",
            "studio lighting", "practical lighting", "neon glow", "candlelight"
        ],
        "texture": [
            "detailed textures", "hyperrealistic details", "tactile surfaces",
            "material realism", "fabric details", "skin pores visible",
            "hair strands detailed", "reflective surfaces"
        ],
        "atmosphere": [
            "cozy atmosphere", "tense mood", "romantic ambiance",
            "mysterious vibe", "epic scale", "intimate setting",
            "dreamlike quality", "surreal feeling"
        ]
    }

    # 镜头描述词汇
    CAMERA_VOCAB = {
        "movement": [
            "slow push-in", "gentle pull-out", "smooth dolly shot",
            "steady tracking shot", "subtle pan", "graceful tilt",
            "dynamic handheld", "floating crane shot", "soaring drone shot"
        ],
        "quality": [
            "steady shot", "stable camera", "smooth motion",
            "fluid movement", "precise framing", "balanced composition"
        ]
    }

    # 技术参数词汇
    TECHNICAL_TERMS = {
        "lens": ["35mm lens", "50mm lens", "85mm lens", "24mm wide lens", "70-200mm zoom"],
        "framerate": ["24fps cinematic", "30fps smooth", "60fps slow motion"],
        "aperture": ["f/1.8 shallow depth", "f/2.8 cinematic", "f/4 balanced", "f/8 sharp"],
        "color": ["warm color grading", "cool teal and orange", "desaturated look", "vibrant colors"]
    }


class PromptEnhancer:
    """提示词增强器"""

    def __init__(self, vocabulary: PromptVocabulary = None):
        self.vocab = vocabulary or PromptVocabulary()

    def enhance_subject_description(self, base_description: str,
                                    character_info: Dict[str, Any] = None,
                                    prop_info: Dict[str, Any] = None) -> str:
        """增强主体描述"""
        enhanced = base_description

        # 添加角色细节
        if character_info:
            enhanced = self._add_character_details(enhanced, character_info)

        # 添加道具细节
        if prop_info:
            enhanced = self._add_prop_details(enhanced, prop_info)

        # 添加动作细节
        enhanced = self._add_action_descriptors(enhanced)

        # 添加空间关系
        enhanced = self._add_spatial_context(enhanced)

        return enhanced

    def _add_character_details(self, text: str, character_info: Dict[str, Any]) -> str:
        """添加角色细节"""
        details = []

        if "clothing" in character_info and character_info["clothing"]:
            clothing = character_info["clothing"]
            if "color" in clothing and "type" in clothing:
                details.append(f"wearing a {clothing['color']} {clothing['type']}")
            elif "type" in clothing:
                details.append(f"wearing a {clothing['type']}")

        if "expression" in character_info and character_info["expression"]:
            details.append(f"with a {character_info['expression']} expression")

        if "posture" in character_info and character_info["posture"]:
            details.append(f"{character_info['posture']}")

        if details:
            text += f", {' and '.join(details)}"

        return text

    def _add_prop_details(self, text: str, prop_info: Dict[str, Any]) -> str:
        """添加道具细节"""
        if "name" in prop_info and prop_info["name"]:
            details = []

            if "state" in prop_info and prop_info["state"]:
                state_map = {
                    "half_full": "half-full",
                    "empty": "empty",
                    "full": "full"
                }
                details.append(state_map.get(prop_info["state"], ""))

            if "location" in prop_info and prop_info["location"]:
                location_map = {
                    "in_hand": "holding in hand",
                    "on_table": "placed on table",
                    "on_floor": "lying on floor"
                }
                details.append(location_map.get(prop_info["location"], ""))

            if details:
                text += f", {prop_info['name']} {' '.join([d for d in details if d])}"

        return text

    def _add_action_descriptors(self, text: str) -> str:
        """添加动作描述词"""
        action_descriptors = [
            "gently", "slowly", "deliberately", "gracefully",
            "suddenly", "quickly", "carefully", "nervously"
        ]

        # 根据文本内容选择合适描述词
        if any(word in text for word in ["sit", "stand", "walk"]):
            descriptor = random.choice(action_descriptors[:4])  # 温和的动作
            text = text.replace("sitting", f"{descriptor} sitting")
            text = text.replace("standing", f"{descriptor} standing")
            text = text.replace("walking", f"{descriptor} walking")

        return text

    def _add_spatial_context(self, text: str) -> str:
        """添加空间上下文"""
        spatial_phrases = [
            "in the foreground", "in the background", "centered in frame",
            "slightly to the left", "on the right side", "occupying most of the frame"
        ]

        # 如果文本中没有空间描述，添加一个
        if not any(phrase in text for phrase in spatial_phrases):
            text += f", {random.choice(spatial_phrases)}"

        return text

    def generate_style_enhancement(self, content_type: str,
                                   emotional_tone: str = None) -> str:
        """生成风格增强提示词"""
        style_parts = []

        # 基于内容类型的风格
        style_by_type = {
            "dialogue_intimate": [
                "soft natural lighting",
                "shallow depth of field",
                "warm color tones",
                "intimate framing"
            ],
            "action_fast": [
                "dynamic lighting",
                "motion blur",
                "high contrast",
                "handheld camera feel"
            ],
            "emotional_reveal": [
                "dramatic chiaroscuro lighting",
                "extreme close-up detail",
                "heightened textures",
                "emotionally charged color grading"
            ]
        }

        base_styles = style_by_type.get(content_type, style_by_type["dialogue_intimate"])
        style_parts.extend(base_styles)

        # 基于情绪的增强
        if emotional_tone:
            emotion_enhancements = {
                "happy": ["bright lighting", "warm colors", "uplifting atmosphere"],
                "sad": ["muted colors", "soft diffused lighting", "melancholy mood"],
                "tense": ["high contrast", "dramatic shadows", "uneasy atmosphere"],
                "romantic": ["soft glow", "warm tones", "dreamlike quality"]
            }
            if emotional_tone in emotion_enhancements:
                style_parts.extend(emotion_enhancements[emotional_tone])

        # 添加随机风格形容词
        style_adjectives = random.sample(self.vocab.STYLE_ADJECTIVES["cinematic"], 2)
        style_parts.extend(style_adjectives)

        # 添加纹理和质量描述
        texture_desc = random.choice(self.vocab.STYLE_ADJECTIVES["texture"])
        style_parts.append(texture_desc)

        # 添加氛围描述
        atmosphere_desc = random.choice(self.vocab.STYLE_ADJECTIVES["atmosphere"])
        style_parts.append(atmosphere_desc)

        return ", ".join(style_parts)

    def generate_technical_specs(self, camera_params: Dict[str, Any],
                                 shot_complexity: str = "medium") -> str:
        """生成技术规格提示词"""
        tech_parts = []

        # 镜头运动
        if "camera_movement" in camera_params:
            movement_desc = camera_params["camera_movement"]
            quality = random.choice(self.vocab.CAMERA_VOCAB["quality"])
            tech_parts.append(f"{movement_desc}, {quality}")

        # 镜头类型
        if "lens" in camera_params:
            tech_parts.append(camera_params["lens"])

        # 帧率
        if shot_complexity == "high":
            tech_parts.append("60fps for smooth slow motion potential")
        else:
            tech_parts.append("cinematic 24fps")

        # 景深
        if "depth_of_field" in camera_params:
            if camera_params["depth_of_field"] == "shallow":
                tech_parts.append("shallow depth of field, background slightly blurred")
            elif camera_params["depth_of_field"] == "deep":
                tech_parts.append("deep focus, everything in sharp detail")

        # 添加随机的技术增强
        tech_enhancements = random.sample(list(self.vocab.TECHNICAL_TERMS["color"]), 1)
        tech_parts.extend(tech_enhancements)

        return ", ".join(tech_parts)


class PromptEngine:
    """主提示词生成引擎"""

    def __init__(self):
        self.constraint_handler = ConstraintHandler()
        self.prompt_enhancer = PromptEnhancer()

    def generate_shot_prompt(self, anchored_segment: AnchoredSegment,
                             style_guide: StyleGuide) -> SoraPromptStructure:
        """为锚点片段生成完整提示词结构"""

        # 1. 处理约束
        constraints = anchored_segment.hard_constraints
        categorized_constraints = self.constraint_handler.categorize_constraints(constraints)
        key_elements = self.constraint_handler.extract_key_elements(constraints)

        # 2. 生成基础主体描述
        base_description = self._create_base_description(
            anchored_segment.base_segment.visual_content,
            key_elements
        )

        # 3. 增强主体描述
        character_info = key_elements["characters"][0] if key_elements["characters"] else None
        prop_info = key_elements["props"][0] if key_elements["props"] else None

        enhanced_subject = self.prompt_enhancer.enhance_subject_description(
            base_description, character_info, prop_info
        )

        # 4. 添加硬约束的具体要求
        subject_with_constraints = self._incorporate_hard_constraints(
            enhanced_subject, constraints
        )

        # 5. 生成风格增强
        content_type = anchored_segment.base_segment.content_type or "dialogue_intimate"
        emotional_tone = anchored_segment.base_segment.emotional_tone

        style_enhancement = self.prompt_enhancer.generate_style_enhancement(
            content_type, emotional_tone
        )

        # 6. 生成技术规格
        camera_params = self._determine_camera_parameters(anchored_segment)
        technical_specs = self.prompt_enhancer.generate_technical_specs(
            camera_params, anchored_segment.base_segment.shot_complexity
        )

        return SoraPromptStructure(
            subject_description=subject_with_constraints,
            style_enhancement=style_enhancement,
            technical_specs=technical_specs
        )

    def _create_base_description(self, visual_content: str,
                                 key_elements: Dict[str, Any]) -> str:
        """创建基础描述"""
        description = visual_content

        # 如果视觉内容太简单，用关键元素增强
        if len(description.split()) < 10:
            enhanced_parts = []

            # 添加角色
            if key_elements["characters"]:
                for char in key_elements["characters"][:2]:  # 最多两个主要角色
                    if char["name"]:
                        enhanced_parts.append(f"{char['name']}")

            # 添加主要动作
            action_keywords = ["sitting", "standing", "walking", "talking", "looking"]
            for keyword in action_keywords:
                if keyword in description.lower():
                    enhanced_parts.append(keyword)
                    break

            if enhanced_parts:
                description = f"{' '.join(enhanced_parts)}, {description}"

        return description.capitalize()

    def _incorporate_hard_constraints(self, base_description: str,
                                      constraints: List[HardConstraint]) -> str:
        """融入硬约束到描述中"""
        if not constraints:
            return base_description

        # 提取高优先级约束
        high_priority = [c for c in constraints if c.priority >= 8]

        constraint_descriptions = []
        for constraint in high_priority[:3]:  # 最多三个高优先级约束
            # 将Sora指令转化为自然语言描述
            if constraint.sora_instruction:
                # 移除技术指令，保留视觉描述
                instruction = constraint.sora_instruction
                # 移除括号内的技术参数
                import re
                instruction = re.sub(r'\([^)]*\)', '', instruction)
                constraint_descriptions.append(instruction.strip())

        if constraint_descriptions:
            constraints_text = ", ".join(constraint_descriptions)
            return f"{base_description}. Important: {constraints_text}"

        return base_description

    def _determine_camera_parameters(self, anchored_segment: AnchoredSegment) -> Dict[str, Any]:
        """确定相机参数"""
        # 基于内容类型决定基础参数
        content_type = anchored_segment.base_segment.content_type

        base_params = {
            "dialogue_intimate": {
                "camera_movement": "slow push-in",
                "lens": "50mm lens",
                "depth_of_field": "shallow"
            },
            "action_fast": {
                "camera_movement": "handheld tracking",
                "lens": "35mm wide lens",
                "depth_of_field": "medium"
            },
            "emotional_reveal": {
                "camera_movement": "static shot",
                "lens": "85mm portrait lens",
                "depth_of_field": "very shallow"
            }
        }

        # 检查是否有镜头相关的硬约束
        camera_constraints = [c for c in anchored_segment.hard_constraints
                              if c.type == "camera_angle"]

        if camera_constraints:
            # 使用约束中的相机指令
            camera_instruction = camera_constraints[0].sora_instruction.lower()

            # 解析相机指令
            params = base_params.get(content_type, base_params["dialogue_intimate"]).copy()

            if "close-up" in camera_instruction or "特写" in camera_instruction:
                params["lens"] = "85mm portrait lens"
                params["depth_of_field"] = "very shallow"
            elif "wide" in camera_instruction or "广角" in camera_instruction:
                params["lens"] = "24mm wide lens"
                params["depth_of_field"] = "deep"

            return params
        else:
            return base_params.get(content_type, base_params["dialogue_intimate"])
