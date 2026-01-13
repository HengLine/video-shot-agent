"""
@FileName: dialogue_estimator.py
@Description: 基于规则的对话时长估算器
@Author: HengLine
@Time: 2026/1/12 15:40
"""
import re
from abc import ABC
from datetime import datetime
from typing import Dict, List, Any

from hengline.agent.script_parser.script_parser_models import Dialogue
from hengline.agent.temporal_planner.estimator.rule_base_estimator import BaseDurationEstimator
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, ElementType
from hengline.logger import debug, error, info
from utils.log_utils import print_log_exception

class DialogueDurationEstimator(BaseDurationEstimator, ABC):
    """基于规则的对话时长估算器"""

    def _initialize_rules(self) -> Dict[str, Any]:
        """从配置加载对话估算规则"""
        dialogue_config = self.planner_config.dialogue_estimator
        self.dialogue_keywords = self.keyword_config.get_dialogue_keywords()

        rules = {
            "speech_rate_baselines": dialogue_config.get("speech_rate_baselines", {}),
            "dialogue_type_factors": dialogue_config.get("dialogue_type_factors", {}),
            "emotion_speed_adjustments": dialogue_config.get("emotion_speed_adjustments", {}),
            "punctuation_adjustments": dialogue_config.get("punctuation_adjustments", {}),
            "silence_baselines": dialogue_config.get("silence_baselines", {}),
            "parenthetical_adjustments": dialogue_config.get("parenthetical_adjustments", {}),
            "keywords": self.dialogue_keywords,
            "complexity_thresholds": dialogue_config.get("complexity_thresholds", {}),
            "confidence_config": dialogue_config.get("confidence_config", {}),
            "config": dialogue_config.get("config", {}),
        }

        debug("对话规则加载完成")
        return rules

    def estimate(self, dialogue_data: Dialogue) -> DurationEstimation:
        """估算对话时长"""
        dialogue_id = dialogue_data.dialogue_id
        content = dialogue_data.content

        debug(f"开始估算对话: {dialogue_id}")
        debug(f"对话内容: '{content[:50]}...'" if len(content) > 50 else f"对话内容: '{content}'")

        try:
            # 检查是否为沉默
            if self._is_silence_dialogue(dialogue_data):
                return self._estimate_silence_duration(dialogue_data)

            # 1. 基础分析
            speaker = dialogue_data.speaker
            emotion = dialogue_data.emotion
            voice_quality = dialogue_data.voice_quality
            parenthetical = dialogue_data.parenthetical

            # 2. 对话类型分析
            dialogue_type = self._determine_dialogue_type(content, emotion, parenthetical)
            debug(f"对话类型: {dialogue_type}")

            # 3. 文本分析
            text_analysis = self._analyze_dialogue_text(content, emotion)
            debug(f"文本分析: {text_analysis.get('word_count')}词, 标点: {text_analysis.get('punctuation_type')}")

            # 4. 情感强度分析
            emotional_intensity = self._analyze_emotional_intensity(emotion, content, parenthetical)
            debug(f"情感强度: {emotional_intensity}")

            # 5. 计算基础时长
            base_duration = self._calculate_base_duration(content, text_analysis, emotional_intensity)
            debug(f"基础时长: {base_duration}秒")

            # 6. 应用调整因子
            adjusted_duration = self._apply_dialogue_adjustments(
                base_duration, dialogue_data, text_analysis, emotional_intensity, dialogue_type
            )
            debug(f"调整后时长: {adjusted_duration}秒")

            # 7. 确保在合理范围内
            final_duration = self._clamp_duration(adjusted_duration, "dialogue")

            # 8. 计算置信度
            confidence = self._calculate_dialogue_confidence(dialogue_data, text_analysis, emotional_intensity)
            debug(f"置信度: {confidence}")

            # 9. 创建 DurationEstimation 对象
            estimation = self._create_duration_estimation(
                dialogue_id=dialogue_id,
                original_duration=dialogue_data.duration,
                rule_estimated_duration=final_duration,
                confidence=confidence,
                dialogue_data=dialogue_data,
                text_analysis=text_analysis,
                emotional_intensity=emotional_intensity,
                dialogue_type=dialogue_type
            )

            info(f"估算完成: {final_duration}秒")
            return estimation

        except Exception as e:
            print_log_exception()
            error(f"估算失败: {str(e)}")
            return self._create_fallback_estimation(dialogue_id, dialogue_data)

    def _is_silence_dialogue(self, dialogue_data: Dialogue) -> bool:
        """检查是否为沉默对话"""
        content = dialogue_data.content.strip()
        dialogue_type = dialogue_data.type

        # 类型为silence或内容为空
        if dialogue_type == "silence" or not content:
            return True

        # 检查是否只有标点或空格
        if re.sub(r'[^\w\s\u4e00-\u9fff]', '', content).strip() == "":
            return True

        return False

    def _estimate_silence_duration(self, dialogue_data: Dialogue) -> DurationEstimation:
        """估算沉默时长"""
        dialogue_id = dialogue_data.dialogue_id

        debug(f"处理沉默对话: {dialogue_id}")

        # 1. 分析沉默类型
        silence_type = self._determine_silence_type(dialogue_data)
        debug(f"沉默类型: {silence_type}")

        # 2. 获取基准时长
        base_duration = self.rules["silence_baselines"].get(
            silence_type,
            self.rules["silence_baselines"].get("default", 1.0)
        )

        # 3. 根据表演提示调整
        parenthetical = dialogue_data.parenthetical
        adjusted_duration = self._adjust_silence_duration(base_duration, parenthetical)

        # 4. 确保在合理范围内
        final_duration = self._clamp_duration(adjusted_duration, "silence")

        # 5. 计算置信度（沉默通常置信度较高）
        confidence = self._calculate_silence_confidence(dialogue_data)

        # 6. 创建估算结果
        estimation = self._create_silence_estimation(
            dialogue_id=dialogue_id,
            original_duration=dialogue_data.duration,
            rule_estimated_duration=final_duration,
            confidence=confidence,
            dialogue_data=dialogue_data,
            silence_type=silence_type
        )

        info(f"沉默估算完成: {final_duration}秒")
        return estimation

    def _determine_silence_type(self, dialogue_data: Dialogue) -> str:
        """确定沉默类型"""
        emotion = dialogue_data.emotion
        parenthetical = dialogue_data.parenthetical

        # 基于情感判断
        if any(word in emotion for word in ["震惊", "震惊的", "shocked"]):
            return "shock_silence"
        elif any(word in emotion for word in ["悲伤", "哽咽", "悲伤的", "sad"]):
            return "emotional_pause"
        elif any(word in emotion for word in ["思考", "犹豫", "thinking"]):
            return "thinking_silence"

        # 基于表演提示判断
        if any(word in parenthetical for word in ["震惊", "愣住", "呆住", "shocked"]):
            return "shock_silence"
        elif any(word in parenthetical for word in ["思考", "犹豫", "考虑", "thinking"]):
            return "thinking_silence"
        elif any(word in parenthetical for word in ["停顿", "暂停", "pause"]):
            return "brief_pause"

        # 默认短暂停顿
        return "brief_pause"

    def _adjust_silence_duration(self, base_duration: float, parenthetical: str) -> float:
        """调整沉默时长"""
        adjusted = base_duration

        # 根据表演提示调整
        if "长久的" in parenthetical or "长时间" in parenthetical or "long" in parenthetical:
            adjusted *= 2.0
        elif "短暂的" in parenthetical or "短时间" in parenthetical or "brief" in parenthetical:
            adjusted *= 0.5
        elif "深深地" in parenthetical or "deep" in parenthetical:
            adjusted *= 1.5

        # 根据上下文调整
        context_adjustment = self._apply_context_adjustments(adjusted)
        if context_adjustment != adjusted:
            adjusted = context_adjustment

        return adjusted

    def _calculate_silence_confidence(self, dialogue_data: Dialogue) -> float:
        """计算沉默置信度"""
        # 沉默通常置信度较高
        base_confidence = 0.8

        # 如果有明确的表演提示，置信度更高
        parenthetical = dialogue_data.parenthetical
        if parenthetical:
            base_confidence = min(base_confidence + 0.1, 0.95)

        # 如果有情感描述，置信度更高
        emotion = dialogue_data.emotion
        if emotion:
            base_confidence = min(base_confidence + 0.05, 0.95)

        return round(base_confidence, 2)

    def _create_silence_estimation(self, dialogue_id: str, original_duration: float,
                                   rule_estimated_duration: float, confidence: float,
                                   dialogue_data: Dialogue, silence_type: str) -> DurationEstimation:
        """创建沉默估算结果"""
        # 计算情感权重（沉默通常情感权重高）
        emotional_weight = self._calculate_silence_emotional_weight(dialogue_data)

        # 构建推理详情
        reasoning_breakdown = {
            "silence_type": silence_type,
            "base_duration": self.rules["silence_baselines"].get(silence_type, 1.0),
            "adjustments_applied": True if dialogue_data.parenthetical else False
        }

        # 构建关键因素
        key_factors = [f"沉默类型: {silence_type}"]
        emotion = dialogue_data.emotion
        if emotion:
            key_factors.append(f"情感: {emotion}")

        # 构建视觉提示
        visual_hints = self._create_silence_visual_hints(dialogue_data, silence_type)

        return DurationEstimation(
            element_id=dialogue_id,
            element_type=ElementType.SILENCE,
            original_duration=original_duration,
            estimated_duration=rule_estimated_duration,
            confidence=confidence,
            rule_based_estimate=rule_estimated_duration,
            adjustment_reason=self._build_silence_adjustment_reason(dialogue_data),
            emotional_weight=emotional_weight,
            visual_complexity=1.0,  # 沉默视觉复杂度低
            pacing_factor=self._calculate_silence_pacing_factor(dialogue_data),
            reasoning_breakdown=reasoning_breakdown,
            visual_hints=visual_hints,
            key_factors=key_factors,
            pacing_notes=self._generate_silence_pacing_notes(silence_type),
            estimated_at=datetime.now().isoformat()
        )

    def _calculate_silence_emotional_weight(self, dialogue_data: Dialogue) -> float:
        """计算沉默情感权重"""
        base_weight = 1.5  # 沉默基础情感权重较高

        emotion = dialogue_data.emotion
        if any(word in emotion for word in ["震惊", "悲伤", "哽咽"]):
            base_weight = 2.0
        elif any(word in emotion for word in ["愤怒", "激动"]):
            base_weight = 1.8
        elif any(word in emotion for word in ["思考", "犹豫"]):
            base_weight = 1.3

        return round(base_weight, 2)

    def _create_silence_visual_hints(self, dialogue_data: Dialogue, silence_type: str) -> Dict[str, Any]:
        """创建沉默视觉提示"""
        hints = {}

        # 镜头建议
        shot_suggestions = {
            "shock_silence": ["extreme_close_up", "slow_zoom_in"],
            "emotional_pause": ["close_up", "soft_focus"],
            "thinking_silence": ["medium_close_up", "shallow_depth_of_field"],
            "brief_pause": ["reaction_shot", "cutaway"],
            "transition_silence": ["wide_shot", "panning"]
        }

        hints["suggested_shot_types"] = shot_suggestions.get(silence_type, ["close_up"])

        # 焦点建议
        hints["focus_elements"] = ["facial_expression", "eye_movement"]

        # 时长拉伸建议
        if silence_type in ["shock_silence", "emotional_pause"]:
            hints["time_stretch_suggestion"] = "slight_slow_motion"

        return hints

    def _build_silence_adjustment_reason(self, dialogue_data: Dialogue) -> str:
        """构建沉默调整原因"""
        reasons = []

        emotion = dialogue_data.emotion
        if emotion:
            reasons.append(f"情感: {emotion}")

        parenthetical = dialogue_data.parenthetical
        if parenthetical:
            reasons.append(f"表演: {parenthetical[:20]}...")

        if reasons:
            return f"沉默调整: {', '.join(reasons)}"
        return "标准沉默时长"

    def _calculate_silence_pacing_factor(self, dialogue_data: Dialogue) -> float:
        """计算沉默节奏因子"""
        factor = 0.7  # 沉默通常减慢节奏

        # 根据沉默类型调整
        silence_type = self._determine_silence_type(dialogue_data)
        if silence_type == "brief_pause":
            factor = 0.9
        elif silence_type == "shock_silence":
            factor = 0.5

        return round(factor, 2)

    def _generate_silence_pacing_notes(self, silence_type: str) -> str:
        """生成沉默节奏说明"""
        notes = {
            "shock_silence": "需要足够时间传达震惊感",
            "emotional_pause": "情感停顿需要观众共鸣时间",
            "thinking_silence": "思考过程需要表现时间",
            "brief_pause": "短暂自然停顿",
            "transition_silence": "场景或情绪转换过渡"
        }
        return notes.get(silence_type, "情感或思考时刻")

    # ============== 普通对话估算方法 ==============

    def _determine_dialogue_type(self, content: str, emotion: str, parenthetical: str) -> str:
        """确定对话类型"""
        content_lower = content.lower()

        # 从配置获取对话类型关键词
        dialogue_type_keywords = self.dialogue_keywords.get("dialogue_types")

        # 检查各种对话类型
        for d_type, keywords in dialogue_type_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    return d_type

        # 基于情感判断
        if any(word in emotion for word in ["激动", "愤怒", "悲伤", "喜悦"]):
            return "emotional"

        # 基于表演提示判断
        if any(word in parenthetical for word in ["颤抖", "哭泣", "哽咽"]):
            return "emotional"

        # 默认类型
        return self.rules["config"].get("default_dialogue_type", "casual")

    def _analyze_dialogue_text(self, content: str, emotion: str) -> Dict[str, Any]:
        """分析对话文本"""
        if not content:
            return {
                "word_count": 0,
                "sentence_count": 0,
                "punctuation_type": "none",
                "complexity_level": "very_simple"
            }

        # 基础文本分析
        text_analysis = self._analyze_text_complexity(content)
        word_count = text_analysis["word_count"]

        # 标点分析
        punctuation_type = self._analyze_punctuation(content)

        # 确定复杂度级别
        complexity_level = self._determine_dialogue_complexity(word_count)

        # 检测疑问句
        is_question = self._is_question(content)

        text_analysis.update({
            "punctuation_type": punctuation_type,
            "complexity_level": complexity_level,
            "is_question": is_question,
            "has_emotional_content": self._has_emotional_content(content, emotion)
        })

        return text_analysis

    def _analyze_punctuation(self, text: str) -> str:
        """分析标点类型"""
        punctuation_keywords = self.dialogue_keywords.get("punctuation_patterns")

        for p_type, patterns in punctuation_keywords.items():
            for pattern in patterns:
                if pattern in text:
                    return p_type

        return "none"

    def _determine_dialogue_complexity(self, word_count: int) -> str:
        """确定对话复杂度级别"""
        thresholds = self.rules.get("complexity_thresholds", {})

        if word_count <= thresholds.get("very_simple", 2):
            return "very_simple"
        elif word_count <= thresholds.get("simple", 4):
            return "simple"
        elif word_count <= thresholds.get("medium", 8):
            return "medium"
        elif word_count <= thresholds.get("complex", 15):
            return "complex"
        else:
            return "very_complex"

    def _is_question(self, text: str) -> bool:
        """检查是否为疑问句"""
        question_keywords = self.dialogue_keywords.get("punctuation_patterns", {}).get("question")
        chinese_questions = ["吗", "呢", "什么", "为什么", "怎么", "如何", "何时", "哪里"]

        # 检查标点
        for pattern in question_keywords:
            if pattern in text:
                return True

        # 检查中文疑问词
        for keyword in chinese_questions:
            if keyword in text:
                return True

        return False

    def _has_emotional_content(self, content: str, emotion: str) -> bool:
        """检查是否有情感内容"""
        if emotion and emotion != "平静":
            return True

        # 检查情感关键词
        emotional_indicators = self.dialogue_keywords.get("emotional_indicators")
        all_emotional_words = []
        for words in emotional_indicators.values():
            all_emotional_words.extend(words)

        for word in all_emotional_words:
            if word in content:
                return True

        return False

    def _analyze_emotional_intensity(self, emotion: str, content: str, parenthetical: str) -> Dict[str, Any]:
        """分析情感强度"""
        intensity_score = 1.0  # 默认

        # 从情感描述获取
        emotion_adjustments = self.rules.get("emotion_speed_adjustments", {})
        if emotion in emotion_adjustments:
            intensity_score = emotion_adjustments[emotion]
        elif emotion:  # 如果有情感描述但不在配置中，默认中等强度
            intensity_score = 0.8

        # 从内容获取情感词汇
        emotional_indicators = self.dialogue_keywords.get("emotional_indicators")
        for intensity_level, words in emotional_indicators.items():
            for word in words:
                if word in content:
                    if intensity_level == "high_intensity":
                        intensity_score = min(intensity_score, 0.6)  # 高强度情感减慢语速
                    elif intensity_level == "medium_intensity":
                        intensity_score = min(intensity_score, 0.8)

        # 从表演提示获取
        parenthetical_keywords = self.dialogue_keywords.get("parenthetical_keywords")
        for p_type, keywords in parenthetical_keywords.items():
            for keyword in keywords:
                if keyword in parenthetical:
                    parenthetical_adjustments = self.rules.get("parenthetical_adjustments", {})
                    if p_type in parenthetical_adjustments:
                        intensity_score = min(intensity_score, parenthetical_adjustments[p_type])

        return {
            "score": round(intensity_score, 2),
            "emotion": emotion,
            "has_parenthetical": bool(parenthetical),
            "intensity_level": self._get_intensity_level(intensity_score)
        }

    def _get_intensity_level(self, intensity_score: float) -> str:
        """获取强度级别"""
        if intensity_score <= 0.5:
            return "very_high"
        elif intensity_score <= 0.7:
            return "high"
        elif intensity_score <= 0.9:
            return "medium"
        else:
            return "low"

    def _calculate_base_duration(self, content: str, text_analysis: Dict[str, Any],
                                 emotional_intensity: Dict[str, Any]) -> float:
        """计算基础时长"""
        word_count = text_analysis.get("word_count", 0)

        if word_count == 0:
            return 0.0

        # 1. 确定基础语速
        default_speech_rate = self.rules["config"].get("default_speech_rate", "normal")
        speech_rate = self.rules["speech_rate_baselines"].get(
            default_speech_rate,
            self.rules["speech_rate_baselines"].get("default", 2.5)
        )

        # 2. 应用情感强度调整
        intensity_score = emotional_intensity.get("score", 1.0)
        adjusted_speech_rate = speech_rate * intensity_score

        # 3. 计算基础时长
        base_duration = word_count / adjusted_speech_rate if adjusted_speech_rate > 0 else 0

        # 4. 应用标点调整
        punctuation_type = text_analysis.get("punctuation_type", "none")
        punctuation_adjustment = self.rules["punctuation_adjustments"].get(punctuation_type, 1.0)
        base_duration *= punctuation_adjustment

        # 5. 添加基础停顿（对话间自然停顿）
        if word_count > 0:
            base_duration += 0.3

        return base_duration

    def _apply_dialogue_adjustments(self, base_duration: float, dialogue_data: Dialogue,
                                    text_analysis: Dict[str, Any], emotional_intensity: Dict[str, Any],
                                    dialogue_type: str) -> float:
        """应用对话特有调整"""
        adjusted = base_duration
        applied_factors = {}

        config = self.rules.get("config", {})

        # 1. 对话类型调整
        dialogue_type_factor = self.rules["dialogue_type_factors"].get(dialogue_type, 1.0)
        adjusted *= dialogue_type_factor
        applied_factors["dialogue_type"] = dialogue_type_factor
        debug(f"对话类型调整: {dialogue_type} -> {dialogue_type_factor}")

        # 2. 疑问句调整
        if text_analysis.get("is_question", False) and config.get("apply_contextual_pauses", True):
            question_adjustment = 1.15  # 疑问需要反应时间
            adjusted *= question_adjustment
            applied_factors["question_adjustment"] = question_adjustment
            debug(f"疑问句调整: 1.15")

        # 3. 情感内容调整
        if text_analysis.get("has_emotional_content", False):
            emotional_adjustment = 1.1  # 情感内容需要更多时间
            adjusted *= emotional_adjustment
            applied_factors["emotional_content"] = emotional_adjustment
            debug(f"情感内容调整: 1.1")

        # 4. 复杂度调整
        complexity_level = text_analysis.get("complexity_level", "medium")
        complexity_factors = {
            "very_simple": 0.9,
            "simple": 0.95,
            "medium": 1.0,
            "complex": 1.1,
            "very_complex": 1.2
        }
        complexity_factor = complexity_factors.get(complexity_level, 1.0)
        adjusted *= complexity_factor
        applied_factors["complexity"] = complexity_factor
        debug(f"复杂度调整: {complexity_level} -> {complexity_factor}")

        # 5. 表演提示调整
        parenthetical = dialogue_data.parenthetical
        if parenthetical:
            parenthetical_factor = self._get_parenthetical_adjustment(parenthetical)
            adjusted *= parenthetical_factor
            applied_factors["parenthetical"] = parenthetical_factor
            debug(f"表演提示调整: {parenthetical_factor}")

        # 6. 整体节奏调整
        pacing_adjusted = self._apply_pacing_adjustment(adjusted)
        if pacing_adjusted != adjusted:
            applied_factors["pacing_adjustment"] = pacing_adjusted / adjusted
            adjusted = pacing_adjusted
            debug(f"节奏调整: {applied_factors['pacing_adjustment']}")

        # 7. 上下文调整
        context_adjusted = self._apply_context_adjustments(adjusted)
        if context_adjusted != adjusted:
            applied_factors["context_adjustment"] = context_adjusted / adjusted
            adjusted = context_adjusted
            debug(f"上下文调整: {applied_factors['context_adjustment']}")

        # 保存应用的调整因子
        self._current_adjustment_factors = applied_factors

        return adjusted

    def _get_parenthetical_adjustment(self, parenthetical: str) -> float:
        """获取表演提示调整因子"""
        parenthetical_keywords = self.dialogue_keywords.get("parenthetical_keywords")
        parenthetical_adjustments = self.rules.get("parenthetical_adjustments", {})

        for p_type, keywords in parenthetical_keywords.items():
            for keyword in keywords:
                if keyword in parenthetical:
                    return parenthetical_adjustments.get(p_type, 1.0)

        return 1.0

    def _clamp_duration(self, duration: float, duration_type: str = "dialogue") -> float:
        """确保时长在合理范围内"""
        config = self.rules.get("config", {})

        if duration_type == "silence":
            min_duration = config.get("min_silence_duration", 0.3)
            max_duration = config.get("max_silence_duration", 8.0)
        else:
            min_duration = config.get("min_dialogue_duration", 0.5)
            max_duration = config.get("max_dialogue_duration", 10.0)

        clamped = max(min_duration, min(duration, max_duration))
        return round(clamped, 2)

    def _calculate_dialogue_confidence(self, dialogue_data: Dialogue,
                                       text_analysis: Dict[str, Any],
                                       emotional_intensity: Dict[str, Any]) -> float:
        """计算对话估算置信度"""
        confidence_config = self.rules.get("confidence_config", {})

        # 词数得分
        word_count = text_analysis.get("word_count", 0)
        word_count_score = min(word_count / 10, 1.0)  # 10词为满分

        # 情感清晰度得分
        emotion_clarity_score = 0.7
        emotion = dialogue_data.emotion
        if emotion:
            emotion_clarity_score = 0.9
        if emotional_intensity.get("has_parenthetical", False):
            emotion_clarity_score = min(emotion_clarity_score + 0.1, 1.0)

        # 标点清晰度得分
        punctuation_clarity_score = 0.8
        punctuation_type = text_analysis.get("punctuation_type", "none")
        if punctuation_type != "none":
            punctuation_clarity_score = 0.9

        # 表演提示得分
        parenthetical_score = 0.6
        parenthetical = dialogue_data.parenthetical
        if parenthetical:
            parenthetical_score = 0.9

        # 上下文得分（基于说话者信息）
        context_score = 0.7
        speaker = dialogue_data.speaker
        if speaker:
            context_score = 0.9

        # 加权平均
        weights = {
            "word_count": confidence_config.get("word_count_weight", 0.3),
            "emotion_clarity": confidence_config.get("emotion_clarity_weight", 0.25),
            "punctuation_clarity": confidence_config.get("punctuation_clarity_weight", 0.2),
            "parenthetical": confidence_config.get("parenthetical_weight", 0.15),
            "context": confidence_config.get("context_weight", 0.1)
        }

        confidence = (
                word_count_score * weights["word_count"] +
                emotion_clarity_score * weights["emotion_clarity"] +
                punctuation_clarity_score * weights["punctuation_clarity"] +
                parenthetical_score * weights["parenthetical"] +
                context_score * weights["context"]
        )

        # 调用基类方法进行最终调整
        base_confidence = self._calculate_confidence(dialogue_data, text_analysis)

        # 综合置信度
        final_confidence = (confidence + base_confidence) / 2

        debug(f"置信度计算: 词数={word_count_score:.2f}, 情感={emotion_clarity_score:.2f}, "
              f"标点={punctuation_clarity_score:.2f}, 表演={parenthetical_score:.2f}, "
              f"上下文={context_score:.2f}, 最终={final_confidence:.2f}")

        return round(final_confidence, 2)

    def _create_duration_estimation(self, dialogue_id: str, original_duration: float,
                                    rule_estimated_duration: float, confidence: float,
                                    dialogue_data: Dialogue, text_analysis: Dict[str, Any],
                                    emotional_intensity: Dict[str, Any], dialogue_type: str) -> DurationEstimation:
        """创建对话估算结果"""
        # 计算情感权重和视觉复杂度
        emotional_weight = self._calculate_dialogue_emotional_weight(emotional_intensity, text_analysis)
        visual_complexity = self._calculate_dialogue_visual_complexity(dialogue_data, text_analysis)

        # 提取状态信息
        character_states = self._extract_dialogue_states(dialogue_data)

        # 构建推理详情
        reasoning_breakdown = self._create_dialogue_reasoning_breakdown(
            dialogue_data, text_analysis, emotional_intensity, dialogue_type, rule_estimated_duration
        )

        # 构建关键因素
        key_factors = self._extract_dialogue_key_factors(dialogue_data, text_analysis, emotional_intensity)

        # 构建视觉提示
        visual_hints = self._create_dialogue_visual_hints(dialogue_data, emotional_intensity)

        # 构建调整原因
        adjustment_reason = self._build_dialogue_adjustment_reason()

        # 确定元素类型
        element_type = ElementType.DIALOGUE

        return DurationEstimation(
            element_id=dialogue_id,
            element_type=element_type,
            original_duration=original_duration,
            estimated_duration=rule_estimated_duration,
            confidence=confidence,
            rule_based_estimate=rule_estimated_duration,
            adjustment_reason=adjustment_reason,
            emotional_weight=emotional_weight,
            visual_complexity=visual_complexity,
            pacing_factor=self._calculate_dialogue_pacing_factor(dialogue_data, emotional_intensity),
            character_states=character_states,
            reasoning_breakdown=reasoning_breakdown,
            visual_hints=visual_hints,
            key_factors=key_factors,
            pacing_notes=self._generate_dialogue_pacing_notes(dialogue_data, text_analysis),
            estimated_at=datetime.now().isoformat()
        )

    def _calculate_dialogue_emotional_weight(self, emotional_intensity: Dict[str, Any],
                                             text_analysis: Dict[str, Any]) -> float:
        """计算对话情感权重"""
        base_weight = 1.0

        # 情感强度影响
        intensity_level = emotional_intensity.get("intensity_level", "low")
        intensity_weights = {
            "very_high": 2.0,
            "high": 1.7,
            "medium": 1.4,
            "low": 1.0
        }
        base_weight = intensity_weights.get(intensity_level, 1.0)

        # 情感内容加成
        if text_analysis.get("has_emotional_content", False):
            base_weight *= 1.2

        return round(base_weight, 2)

    def _calculate_dialogue_visual_complexity(self, dialogue_data: Dialogue,
                                              text_analysis: Dict[str, Any]) -> float:
        """计算对话视觉复杂度"""
        base_complexity = 1.0

        # 词数影响
        word_count = text_analysis.get("word_count", 0)
        if word_count > 0:
            base_complexity += min(word_count / 20, 1.0)  # 最多增加1.0

        # 疑问句需要更多视觉变化
        if text_analysis.get("is_question", False):
            base_complexity += 0.3

        # 情感对话需要更多微表情
        if text_analysis.get("has_emotional_content", False):
            base_complexity += 0.4

        return round(min(base_complexity, 3.0), 2)

    def _extract_dialogue_states(self, dialogue_data: Dialogue) -> Dict[str, str]:
        """提取对话状态信息"""
        character_states = {}

        speaker = dialogue_data.speaker
        emotion = dialogue_data.emotion
        parenthetical = dialogue_data.parenthetical

        if speaker:
            state_description = []
            if emotion:
                state_description.append(f"emotion:{emotion}")
            if parenthetical:
                # 简化表演提示
                simplified_parenthetical = parenthetical[:30] + "..." if len(parenthetical) > 30 else parenthetical
                state_description.append(f"action:{simplified_parenthetical}")

            if state_description:
                character_states[speaker] = " | ".join(state_description)

        return character_states

    def _create_dialogue_reasoning_breakdown(self, dialogue_data: Dialogue,
                                             text_analysis: Dict[str, Any],
                                             emotional_intensity: Dict[str, Any],
                                             dialogue_type: str,
                                             total_duration: float) -> Dict[str, Any]:
        """创建对话推理详情"""
        breakdown = {}

        # 词数基础时长
        word_count = text_analysis.get("word_count", 0)
        speech_rate = self.rules["speech_rate_baselines"].get(
            self.rules["config"].get("default_speech_rate", "normal"), 2.5
        )
        base_word_duration = word_count / speech_rate if speech_rate > 0 else 0
        breakdown["word_count_base"] = round(base_word_duration, 2)

        # 情感调整
        intensity_score = emotional_intensity.get("score", 1.0)
        if intensity_score != 1.0:
            breakdown["emotion_adjustment"] = round(base_word_duration * (intensity_score - 1.0), 2)

        # 标点调整
        punctuation_type = text_analysis.get("punctuation_type", "none")
        punctuation_adjustment = self.rules["punctuation_adjustments"].get(punctuation_type, 1.0)
        if punctuation_adjustment != 1.0:
            current_total = sum(v for v in breakdown.values() if isinstance(v, (int, float)))
            breakdown["punctuation_adjustment"] = round(current_total * (punctuation_adjustment - 1.0), 2)

        # 对话类型调整
        dialogue_type_factor = self.rules["dialogue_type_factors"].get(dialogue_type, 1.0)
        if dialogue_type_factor != 1.0:
            current_total = sum(v for v in breakdown.values() if isinstance(v, (int, float)))
            breakdown["dialogue_type_adjustment"] = round(current_total * (dialogue_type_factor - 1.0), 2)

        # 基础停顿
        if word_count > 0:
            breakdown["natural_pause"] = 0.3

        # 应用的所有调整因子
        adjustment_factors = getattr(self, '_current_adjustment_factors', {})
        if adjustment_factors:
            breakdown["adjustment_factors"] = adjustment_factors

        return breakdown

    def _extract_dialogue_key_factors(self, dialogue_data: Dialogue,
                                      text_analysis: Dict[str, Any],
                                      emotional_intensity: Dict[str, Any]) -> List[str]:
        """提取对话关键因素"""
        factors = []

        # 词数因素
        word_count = text_analysis.get("word_count", 0)
        if word_count > 10:
            factors.append(f"长对话({word_count}词)")
        elif word_count < 3:
            factors.append(f"简短回应")

        # 情感因素
        emotion = dialogue_data.emotion
        if emotion and emotion != "平静":
            factors.append(f"情感: {emotion}")

        # 对话类型因素
        dialogue_type = self._determine_dialogue_type(
            dialogue_data.content,
            emotion,
            dialogue_data.parenthetical
        )
        if dialogue_type != "casual":
            type_names = {
                "exposition": "信息交代",
                "emotional": "情感表达",
                "argument": "争论辩论",
                "dramatic": "戏剧性",
                "questioning": "疑问"
            }
            factors.append(type_names.get(dialogue_type, dialogue_type))

        # 表演提示因素
        parenthetical = dialogue_data.parenthetical
        if parenthetical:
            factors.append("有表演提示")

        # 标点因素
        punctuation_type = text_analysis.get("punctuation_type", "none")
        if punctuation_type != "none":
            punctuation_names = {
                "ellipsis": "省略号(犹豫)",
                "question_mark": "问号(疑问)",
                "exclamation": "感叹号(强调)",
                "dash": "破折号(转折)"
            }
            factors.append(punctuation_names.get(punctuation_type, "特殊标点"))

        return factors

    def _create_dialogue_visual_hints(self, dialogue_data: Dialogue,
                                      emotional_intensity: Dict[str, Any]) -> Dict[str, Any]:
        """创建对话视觉提示"""
        hints = {}

        speaker = dialogue_data.speaker
        emotion = dialogue_data.emotion

        # 镜头建议
        intensity_level = emotional_intensity.get("intensity_level", "low")

        shot_suggestions = {
            "very_high": ["extreme_close_up", "shallow_focus"],
            "high": ["close_up", "medium_close_up"],
            "medium": ["medium_close_up", "over_shoulder"],
            "low": ["medium_shot", "two_shot"]
        }

        hints["suggested_shot_types"] = shot_suggestions.get(intensity_level, ["medium_shot"])

        # 焦点建议
        hints["focus_elements"] = ["facial_expression", "lip_movement"]
        if emotion in ["愤怒", "激动"]:
            hints["focus_elements"].append("body_tension")

        # 灯光建议
        if intensity_level in ["very_high", "high"]:
            hints["lighting_notes"] = "dramatic_lighting, high_contrast"
        elif emotion in ["悲伤", "忧郁"]:
            hints["lighting_notes"] = "soft_lighting, cool_tone"
        else:
            hints["lighting_notes"] = "natural_lighting"

        # 说话者信息
        if speaker:
            hints["speaker"] = speaker

        return hints

    def _build_dialogue_adjustment_reason(self) -> str:
        """构建对话调整原因"""
        reasons = []
        adjustment_factors = getattr(self, '_current_adjustment_factors', {})

        factor_names = {
            "dialogue_type": "对话类型",
            "question_adjustment": "疑问句",
            "emotional_content": "情感内容",
            "complexity": "复杂度",
            "parenthetical": "表演提示",
            "pacing_adjustment": "节奏",
            "context_adjustment": "上下文"
        }

        for factor, value in adjustment_factors.items():
            if value != 1.0:
                change = "增加" if value > 1.0 else "减少"
                percent = abs(value - 1.0) * 100
                name = factor_names.get(factor, factor)
                reasons.append(f"{name}{change}{percent:.0f}%")

        if reasons:
            return f"应用调整: {', '.join(reasons)}"
        return "标准估算"

    def _calculate_dialogue_pacing_factor(self, dialogue_data: Dialogue,
                                          emotional_intensity: Dict[str, Any]) -> float:
        """计算对话节奏因子"""
        factor = 1.0

        # 情感强度影响节奏
        intensity_level = emotional_intensity.get("intensity_level", "low")
        if intensity_level in ["very_high", "high"]:
            factor *= 1.3  # 高强度情感加快节奏
        elif intensity_level == "medium":
            factor *= 1.1

        # 对话类型影响
        dialogue_type = self._determine_dialogue_type(
            dialogue_data.content,
            dialogue_data.emotion,
            dialogue_data.parenthetical
        )
        if dialogue_type in ["argument", "urgent"]:
            factor *= 1.2
        elif dialogue_type in ["emotional", "dramatic"]:
            factor *= 0.9

        # 整体节奏目标
        if self.context.overall_pacing_target == "fast":
            factor *= 1.1
        elif self.context.overall_pacing_target == "slow":
            factor *= 0.9

        return round(factor, 2)

    def _generate_dialogue_pacing_notes(self, dialogue_data: Dialogue,
                                        text_analysis: Dict[str, Any]) -> str:
        """生成对话节奏说明"""
        notes = []

        # 词数影响
        word_count = text_analysis.get("word_count", 0)
        if word_count > 15:
            notes.append("长句需要自然停顿")
        elif word_count < 3:
            notes.append("简短回应适合快速剪辑")

        # 情感影响
        emotion = dialogue_data.emotion
        if emotion in ["悲伤", "哽咽"]:
            notes.append("悲伤情绪需要缓慢表达")
        elif emotion in ["愤怒", "激动"]:
            notes.append("激动情绪节奏较快")

        # 疑问句
        if text_analysis.get("is_question", False):
            notes.append("疑问需要反应时间")

        if notes:
            return "; ".join(notes)
        return "自然对话节奏"

    def _create_fallback_estimation(self, dialogue_id: str, dialogue_data: Dialogue) -> DurationEstimation:
        """创建降级估算"""
        # 检查是否为沉默
        if self._is_silence_dialogue(dialogue_data):
            element_type = ElementType.SILENCE
        else:
            element_type = ElementType.DIALOGUE

        return DurationEstimation(
            element_id=dialogue_id,
            element_type=element_type,
            original_duration=dialogue_data.duration,
            estimated_duration=dialogue_data.duration,
            confidence=0.3,
            adjustment_reason="规则估算失败，使用原始值",
            estimated_at=datetime.now().isoformat()
        )

    def _get_required_fields(self) -> List[str]:
        """获取必需字段"""
        return ["dialogue_id", "content"]
