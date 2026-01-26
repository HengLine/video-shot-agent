"""
@FileName: scene_estimator.py
@Description: 场景时长估算器
@Author: HengLine
@Time: 2026/1/12 15:40
"""
from abc import ABC
from datetime import datetime
from typing import Dict, List, Any, Tuple

from hengline.agent.script_parser2.script_parser_models import Scene
from hengline.agent.shot_generator.estimator.rule_base_estimator import BaseRuleDurationEstimator
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, ElementType, EstimationSource
from hengline.logger import debug, error
from utils.log_utils import print_log_exception


class RuleSceneDurationEstimator(BaseRuleDurationEstimator, ABC):
    """使用YAML配置的场景时长估算器"""

    def _initialize_rules(self) -> Dict[str, Any]:
        """从配置加载场景估算规则"""
        scene_config = self.planner_config.scene_estimator
        self.scene_keywords = self.keyword_config.get_scene_keywords()

        # 加载所有规则
        rules = {
            "scene_type_baselines": scene_config.get("scene_type_baselines", {}),
            "text_complexity_factors": scene_config.get("text_complexity_factors", {}),
            "complexity_thresholds": scene_config.get("complexity_thresholds", {}),
            "key_visual_durations": scene_config.get("key_visual_durations", {}),
            "mood_adjustments": scene_config.get("mood_adjustments", {}),
            "keywords": self.scene_keywords,
            "time_weather_adjustments": scene_config.get("time_weather_adjustments", {}),
            "character_count_adjustments": scene_config.get("character_count_adjustments", {}),
            "config": scene_config.get("config", {}),
        }

        debug("场景规则加载完成")
        return rules

    def estimate(self, scene_data: Scene, context: Dict = None) -> DurationEstimation:
        """估算场景时长"""
        scene_id = scene_data.scene_id

        debug(f"开始估算场景: {scene_id}")

        try:
            # 1. 基础分析
            description = scene_data.description
            mood = scene_data.mood
            key_visuals = scene_data.key_visuals
            location = scene_data.location

            # 2. 确定场景类型
            scene_type = self._determine_scene_type(description, location, mood)
            debug(f"场景类型: {scene_type}")

            # 3. 文本复杂度分析
            text_analysis = self._analyze_scene_text(description)
            debug(f"文本分析: {text_analysis.get('word_count')}词, 复杂度: {text_analysis.get('complexity_level')}")

            # 4. 视觉元素分析
            visual_analysis = self._analyze_key_visuals(key_visuals)
            debug(f"视觉分析: {visual_analysis.get('count')}个元素")

            # 5. 计算基础时长
            base_duration = self._calculate_base_duration(scene_type, text_analysis, visual_analysis)
            debug(f"基础时长: {base_duration}秒")

            # 6. 应用调整因子
            adjusted_duration = self._apply_scene_adjustments(base_duration, scene_data, text_analysis, visual_analysis)
            debug(f"调整后时长: {adjusted_duration}秒")

            # 7. 确保在合理范围内
            final_duration = self._clamp_duration(adjusted_duration)

            # 8. 计算置信度
            confidence = self._calculate_scene_confidence(scene_data, text_analysis, visual_analysis)
            debug(f"置信度: {confidence}")

            # 9. 创建 DurationEstimation 对象
            estimation = self._create_duration_estimation(
                scene_id=scene_id,
                original_duration=scene_data.duration,
                rule_estimated_duration=final_duration,
                confidence=confidence,
                scene_data=scene_data,
                text_analysis=text_analysis,
                visual_analysis=visual_analysis,
                scene_type=scene_type
            )

            debug(f"估算完成: {final_duration}秒")
            return estimation

        except Exception as e:
            print_log_exception()
            error(f"估算失败: {str(e)}")
            return self._create_fallback_estimation(scene_id, scene_data)

    def _determine_scene_type(self, description: str, location: str, mood: str) -> str:
        """确定场景类型"""
        description_lower = description.lower()

        # 从配置获取关键词
        scene_type_keywords = self.scene_keywords.get("scene_type", {})

        # 检查各种场景类型
        for scene_type, keywords in scene_type_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return scene_type

        # 基于情绪判断
        emotional_moods = ["悲伤", "喜悦", "愤怒", "恐惧", "孤独", "压抑", "紧张"]
        if any(emotion in mood for emotion in emotional_moods):
            return "emotional_background"

        # 默认类型
        return self.rules.get("config", {}).get("default_scene_type", "detailed")

    def _analyze_scene_text(self, description: str) -> Dict[str, Any]:
        """分析场景文本"""
        # 基础文本分析
        text_analysis = self._analyze_text_complexity(description)

        # 获取词数
        word_count = text_analysis["word_count"]

        # 根据阈值确定复杂度级别
        complexity_level = self._determine_complexity_level(word_count)

        # 检测描述类型
        description_type = self._determine_description_type(description)

        # 检测空间元素
        spatial_elements = self._count_spatial_elements(description)

        text_analysis.update({
            "complexity_level": complexity_level,
            "description_type": description_type,
            "spatial_elements": spatial_elements,
            "has_detailed_description": description_type == "detailed"
        })

        return text_analysis

    def _determine_complexity_level(self, word_count: int) -> str:
        """根据词数确定复杂度级别"""
        thresholds = self.rules.get("complexity_thresholds", {})

        if word_count <= thresholds.get("very_low", 20):
            return "very_low"
        elif word_count <= thresholds.get("low", 40):
            return "low"
        elif word_count <= thresholds.get("medium", 80):
            return "medium"
        elif word_count <= thresholds.get("high", 120):
            return "high"
        else:
            return "very_high"

    def _determine_description_type(self, description: str) -> str:
        """确定描述类型"""
        description_type_keywords = self.scene_keywords.get("description_type")

        for desc_type, keywords in description_type_keywords.items():
            for keyword in keywords:
                if keyword in description:
                    return desc_type

        return "neutral"

    def _count_spatial_elements(self, description: str) -> int:
        """计算空间元素数量"""
        spatial_keywords = self.keyword_config.get_position_keywords().get("spatial_keywords", {})
        count = 0

        for keyword in spatial_keywords:
            if keyword in description:
                count += 1

        return count

    def _analyze_key_visuals(self, key_visuals: List[str]) -> Dict[str, Any]:
        """分析关键视觉元素"""
        if not key_visuals:
            return {
                "count": 0,
                "types": [],
                "total_duration": 0,
                "breakdown": {},
                "has_important_visuals": False
            }

        # 获取视觉类型关键词
        visual_type_keywords = self._get_keywords("visual_types")

        breakdown = {}
        total_duration = 0

        for visual in key_visuals:
            visual_lower = visual.lower()
            assigned_type = "background_detail"  # 默认

            # 确定视觉类型
            for vtype, keywords in visual_type_keywords.items():
                if any(keyword in visual_lower for keyword in keywords):
                    assigned_type = vtype
                    break

            # 获取该类型的时长
            duration_per_element = self.rules["key_visual_durations"].get(
                assigned_type,
                self.rules["key_visual_durations"].get("default", 0.3)
            )

            # 如果是第一个主要元素，时间稍长
            if assigned_type == "major_setting" and breakdown.get(assigned_type, 0) == 0:
                duration_per_element *= 1.5

            breakdown[assigned_type] = breakdown.get(assigned_type, 0) + duration_per_element
            total_duration += duration_per_element

        return {
            "count": len(key_visuals),
            "types": list(breakdown.keys()),
            "total_duration": total_duration,
            "breakdown": breakdown,
            "has_important_visuals": any(t != "background_detail" for t in breakdown.keys())
        }

    def _calculate_base_duration(self, scene_type: str,
                                 text_analysis: Dict[str, Any],
                                 visual_analysis: Dict[str, Any]) -> float:
        """计算基础时长"""
        # 1. 场景类型基础值
        base = self.rules["scene_type_baselines"].get(
            scene_type,
            self.rules["scene_type_baselines"].get("default", 3.0)
        )

        # 2. 文本复杂度调整
        complexity_level = text_analysis.get("complexity_level", "medium")
        complexity_factor = self.rules["text_complexity_factors"].get(complexity_level, 1.0)
        base *= complexity_factor

        # 3. 视觉元素时长
        visual_duration = visual_analysis.get("total_duration", 0)
        base += visual_duration

        # 4. 空间元素加成
        spatial_elements = text_analysis.get("spatial_elements", 0)
        if spatial_elements > 0:
            base += spatial_elements * 0.2

        return base

    def _apply_scene_adjustments(self, base_duration: float,
                                 scene_data: Scene,
                                 text_analysis: Dict[str, Any],
                                 visual_analysis: Dict[str, Any]) -> float:
        """应用场景特有调整"""
        adjusted = base_duration
        applied_factors = {}

        config = self.rules.get("config", {})

        # 1. 氛围调整
        if config.get("apply_mood_adjustments", True):
            mood = scene_data.mood
            mood_adjustment = self.rules["mood_adjustments"].get(mood, 1.0)
            adjusted *= mood_adjustment
            applied_factors["mood"] = mood_adjustment
            debug(f"氛围调整: {mood} -> {mood_adjustment}")

        # 2. 时间/天气调整
        if config.get("apply_time_weather_adjustments", True):
            time_adjustment = self._get_time_of_day_adjustment()
            weather_adjustment = self._get_weather_adjustment()

            adjusted *= time_adjustment
            adjusted *= weather_adjustment

            applied_factors["time_of_day"] = time_adjustment
            applied_factors["weather"] = weather_adjustment

            debug(f"时间调整: {self.context.time_of_day} -> {time_adjustment}")
            debug(f"天气调整: {self.context.weather} -> {weather_adjustment}")

        # 3. 角色数量调整
        if config.get("apply_character_adjustments", True):
            char_adjustment = self._get_character_adjustment()
            adjusted *= char_adjustment
            applied_factors["character_count"] = char_adjustment
            debug(f"角色调整: {self.context.character_count}人 -> {char_adjustment}")

        # 4. 详细描述加成
        if text_analysis.get("has_detailed_description", False):
            detailed_adjustment = 1.15
            adjusted *= detailed_adjustment
            applied_factors["detailed_description"] = detailed_adjustment
            debug("详细描述加成: 1.15")

        # 5. 重要视觉元素加成
        if visual_analysis.get("has_important_visuals", False):
            important_visuals_adjustment = 1.1
            adjusted *= important_visuals_adjustment
            applied_factors["important_visuals"] = important_visuals_adjustment
            debug("重要视觉元素加成: 1.1")

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

    def _get_emotional_adjustment(self) -> float:
        """获取情绪调整因子"""
        # 使用mood_adjustments规则
        emotional_tone = self.context.emotional_tone
        mood_adjustments = self.rules.get("mood_adjustments", {})

        # 尝试英文key
        adjustment = mood_adjustments.get(emotional_tone, 1.0)

        # 如果没有找到，尝试从中文映射
        if adjustment == 1.0:
            # 简单映射
            mood_map = {
                "tense": "紧张",
                "emotional": "悲伤",  # 假设emotional对应悲伤
                "relaxed": "喜悦",  # 假设relaxed对应喜悦
                "neutral": "平静"
            }
            chinese_mood = mood_map.get(emotional_tone, "平静")
            adjustment = mood_adjustments.get(chinese_mood, 1.0)

        return adjustment

    def _get_time_of_day_adjustment(self) -> float:
        """获取时间调整因子"""
        time_adjustments = self.rules.get("time_weather_adjustments", {}).get("time_of_day", {})
        return time_adjustments.get(self.context.time_of_day,
                                    time_adjustments.get("default", 1.0))

    def _get_weather_adjustment(self) -> float:
        """获取天气调整因子"""
        weather_adjustments = self.rules.get("time_weather_adjustments", {}).get("weather", {})
        return weather_adjustments.get(self.context.weather,
                                       weather_adjustments.get("default", 1.0))

    def _get_character_adjustment(self) -> float:
        """获取角色数量调整因子"""
        char_count = self.context.character_count
        adjustments = self.rules.get("character_count_adjustments", {})

        # 找到适用的调整因子
        adjustment = adjustments.get("default", 1.2)

        # 过滤并排序
        int_adjustments = [(k, v) for k, v in adjustments.items() if isinstance(k, int)]
        int_adjustments.sort(key=lambda x: x[0], reverse=True)
        # 检查具体的数量
        for count, adj in int_adjustments:
            if isinstance(count, int) and char_count >= count:
                adjustment = adj
                break

        return adjustment

    def _clamp_duration(self, duration: float) -> float:
        """确保时长在合理范围内"""
        config = self.rules.get("config", {})
        min_duration = config.get("min_scene_duration", 1.5)
        max_duration = config.get("max_scene_duration", 15.0)

        clamped = max(min_duration, min(duration, max_duration))
        return round(clamped, 2)

    def _calculate_scene_confidence(self, scene_data: Scene,
                                    text_analysis: Dict[str, Any],
                                    visual_analysis: Dict[str, Any]) -> float:
        """计算场景估算置信度"""
        config = self.rules.get("config", {})
        confidence_weights = config.get("confidence_weights", {})

        # 文本完整性得分
        text_completeness = 0.7
        if scene_data.description:
            word_count = text_analysis.get("word_count", 0)
            text_completeness = min(1.0, word_count / 50)

        # 关键视觉元素得分
        key_visuals_score = 0.5
        if visual_analysis.get("has_important_visuals", False):
            key_visuals_score = 0.8
        if visual_analysis.get("count", 0) >= 3:
            key_visuals_score = min(1.0, key_visuals_score + 0.2)

        # 氛围清晰度得分
        mood_clarity = 0.6
        mood = scene_data.mood
        if mood and mood != "平静":
            mood_clarity = 0.8
        if mood in ["紧张", "压抑", "悲伤", "喜悦"]:
            mood_clarity = 0.9

        # 加权平均
        confidence = (
                text_completeness * confidence_weights.get("text_completeness", 0.4) +
                key_visuals_score * confidence_weights.get("key_visuals", 0.3) +
                mood_clarity * confidence_weights.get("mood_clarity", 0.3)
        )

        # 调用基类方法进行最终调整
        base_confidence = self._calculate_confidence(scene_data, text_analysis)

        # 综合置信度
        final_confidence = (confidence + base_confidence) / 2

        debug(f"置信度计算: 文本={text_completeness:.2f}, 视觉={key_visuals_score:.2f}, "
              f"氛围={mood_clarity:.2f}, 基础={base_confidence:.2f}, 最终={final_confidence:.2f}")

        return round(final_confidence, 2)

    def _create_duration_estimation(self, scene_id: str, original_duration: float,
                                    rule_estimated_duration: float, confidence: float,
                                    scene_data: Scene, text_analysis: Dict[str, Any],
                                    visual_analysis: Dict[str, Any], scene_type: str) -> DurationEstimation:
        """创建 DurationEstimation 对象"""
        # 计算情感权重和视觉复杂度
        emotional_weight = self._calculate_emotional_weight(scene_data, text_analysis)
        visual_complexity = self._calculate_visual_complexity(visual_analysis, text_analysis)

        # 提取状态信息
        character_states, prop_states = self._extract_state_changes(scene_data)

        # 构建推理详情
        reasoning_breakdown = self._create_reasoning_breakdown(
            scene_type, text_analysis, visual_analysis, scene_data, rule_estimated_duration
        )

        # 构建关键因素
        key_factors = self._extract_key_factors(scene_data, text_analysis, visual_analysis)

        # 构建视觉提示
        visual_hints = self._create_visual_hints(scene_data, visual_analysis)

        # 构建调整原因
        adjustment_reason = self._build_adjustment_reason()

        # 创建 DurationEstimation
        return DurationEstimation(
            element_id=scene_id,
            element_type=ElementType.SCENE,
            original_duration=original_duration,
            estimated_duration=rule_estimated_duration,  # 对于规则估算器，这是规则估算值
            confidence=confidence,
            rule_estimated=rule_estimated_duration,  # 规则估算值
            estimator_source=EstimationSource.LOCAL_RULE,
            adjustment_reason=adjustment_reason,
            emotional_weight=emotional_weight,
            visual_complexity=visual_complexity,
            pacing_factor=self._calculate_pacing_factor(scene_data),
            character_states=character_states,
            prop_states=prop_states,
            reasoning_breakdown=reasoning_breakdown,
            visual_hints=visual_hints,
            key_factors=key_factors,
            pacing_notes=self._generate_pacing_notes(scene_data, text_analysis),
            estimated_at=datetime.now().isoformat()
        )

    def _calculate_emotional_weight(self, scene_data: Scene,
                                    text_analysis: Dict[str, Any]) -> float:
        """计算情感权重"""
        mood = scene_data.mood

        # 基于氛围的情感权重
        mood_weights = {
            "紧张": 1.8,
            "压抑": 1.7,
            "悲伤": 1.6,
            "孤独": 1.5,
            "恐惧": 1.7,
            "愤怒": 1.6,
            "喜悦": 1.3,
            "平静": 1.0,
            "中性": 1.0
        }

        base_weight = mood_weights.get(mood, 1.0)

        # 文本复杂度加成
        complexity = text_analysis.get("complexity_score", 1.0)
        if complexity > 2.0:
            base_weight *= 1.1

        return round(base_weight, 2)

    def _calculate_visual_complexity(self, visual_analysis: Dict[str, Any],
                                     text_analysis: Dict[str, Any]) -> float:
        """计算视觉复杂度"""
        base_complexity = 1.0

        # 视觉元素数量
        visual_count = visual_analysis.get("count", 0)
        if visual_count > 0:
            base_complexity += min(visual_count / 5, 1.0)  # 最多增加1.0

        # 视觉元素类型多样性
        visual_types = len(visual_analysis.get("types", []))
        if visual_types > 1:
            base_complexity += (visual_types - 1) * 0.2

        # 空间元素加成
        spatial_elements = text_analysis.get("spatial_elements", 0)
        if spatial_elements > 0:
            base_complexity += min(spatial_elements * 0.1, 0.5)

        return round(min(base_complexity, 3.0), 2)  # 限制在3.0以内

    def _extract_state_changes(self, scene_data: Scene) -> Tuple[Dict[str, str], Dict[str, str]]:
        """提取状态变化信息"""
        character_states = {}
        prop_states = {}

        # 从场景描述中提取可能的状态变化
        description = scene_data.description

        # 简单检测（实际可用更复杂的NLP）
        if "蜷缩" in description or "蜷坐" in description:
            character_states["林然"] = "posture:蜷缩在沙发上"

        if "电视播放" in description:
            prop_states["电视"] = "state:静音播放黑白电影"

        if "半杯凉茶" in description:
            prop_states["凉茶"] = "state:半杯已凉，凝出水雾"

        return character_states, prop_states

    def _create_reasoning_breakdown(self, scene_type: str,
                                    text_analysis: Dict[str, Any],
                                    visual_analysis: Dict[str, Any],
                                    scene_data: Scene,
                                    total_duration: float) -> Dict[str, Any]:
        """创建推理详情"""
        breakdown = {}

        # 场景类型基础
        scene_base = self.rules["scene_type_baselines"].get(
            scene_type,
            self.rules["scene_type_baselines"].get("default", 3.0)
        )
        breakdown["scene_type_base"] = scene_base

        # 文本复杂度
        complexity_level = text_analysis.get("complexity_level", "medium")
        complexity_factor = self.rules["text_complexity_factors"].get(complexity_level, 1.0)
        if complexity_factor != 1.0:
            breakdown["text_complexity_adjustment"] = round(scene_base * (complexity_factor - 1.0), 2)

        # 视觉元素
        visual_breakdown = visual_analysis.get("breakdown", {})
        for vtype, duration in visual_breakdown.items():
            breakdown[f"visual_{vtype}"] = duration

        # 空间元素
        spatial_elements = text_analysis.get("spatial_elements", 0)
        if spatial_elements > 0:
            breakdown["spatial_elements"] = round(spatial_elements * 0.2, 2)

        # 应用的所有调整因子
        adjustment_factors = getattr(self, '_current_adjustment_factors', {})
        if adjustment_factors:
            breakdown["adjustment_factors"] = adjustment_factors

        return {k: round(v, 2) if isinstance(v, (int, float)) else v
                for k, v in breakdown.items()}

    def _extract_key_factors(self, scene_data: Scene,
                             text_analysis: Dict[str, Any],
                             visual_analysis: Dict[str, Any]) -> List[str]:
        """提取关键因素"""
        factors = []

        # 氛围因素
        mood = scene_data.mood
        if mood and mood != "平静":
            factors.append(f"氛围: {mood}")

        # 视觉复杂度因素
        visual_count = visual_analysis.get("count", 0)
        if visual_count >= 3:
            factors.append(f"多视觉元素({visual_count}个)")

        # 文本复杂度因素
        word_count = text_analysis.get("word_count", 0)
        if word_count > 50:
            factors.append("详细描述")

        # 时间天气因素
        if self.context.time_of_day != "day":
            factors.append(f"时间: {self.context.time_of_day}")
        if self.context.weather != "clear":
            factors.append(f"天气: {self.context.weather}")

        return factors

    def _create_visual_hints(self, scene_data: Scene,
                             visual_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """创建视觉提示"""
        hints = {}

        # 基于场景类型
        scene_type = self._determine_scene_type(
            scene_data.description,
            scene_data.location,
            scene_data.mood
        )

        # 镜头建议
        shot_suggestions = {
            "establishing": ["wide_shot", "panning"],
            "detailed": ["medium_shot", "close_up_details"],
            "atmospheric": ["slow_pan", "mood_shot"],
            "emotional_background": ["close_up", "shallow_focus"]
        }

        hints["suggested_shot_types"] = shot_suggestions.get(scene_type, ["medium_shot"])

        # 灯光建议
        mood = scene_data.mood
        if mood in ["紧张", "压抑", "孤独"]:
            hints["lighting_notes"] = "low_key_lighting, high_contrast"
        elif mood in ["悲伤", "忧郁"]:
            hints["lighting_notes"] = "soft_lighting, cool_tone"
        else:
            hints["lighting_notes"] = "natural_lighting"

        # 焦点元素
        if visual_analysis.get("has_important_visuals", False):
            hints["focus_elements"] = ["key_props", "character_expressions"]

        return hints

    def _build_adjustment_reason(self) -> str:
        """构建调整原因"""
        reasons = []
        adjustment_factors = getattr(self, '_current_adjustment_factors', {})

        for factor, value in adjustment_factors.items():
            if value != 1.0:
                change = "增加" if value > 1.0 else "减少"
                percent = abs(value - 1.0) * 100
                reasons.append(f"{factor}{change}{percent:.0f}%")

        if reasons:
            return f"应用调整: {', '.join(reasons)}"
        return "标准估算"

    def _calculate_pacing_factor(self, scene_data: Scene) -> float:
        """计算节奏因子"""
        factor = 1.0

        # 氛围影响节奏
        mood = scene_data.mood
        if mood in ["紧张", "愤怒"]:
            factor *= 1.2  # 更快节奏
        elif mood in ["悲伤", "孤独", "压抑"]:
            factor *= 0.8  # 更慢节奏

        # 整体节奏目标
        if self.context.overall_pacing == "fast":
            factor *= 1.1
        elif self.context.overall_pacing == "slow":
            factor *= 0.9

        return round(factor, 2)

    def _generate_pacing_notes(self, scene_data: Scene,
                               text_analysis: Dict[str, Any]) -> str:
        """生成节奏说明"""
        notes = []

        mood = scene_data.mood
        if mood:
            notes.append(f"氛围: {mood}")

        complexity = text_analysis.get("complexity_score", 1.0)
        if complexity > 1.5:
            notes.append("复杂描述需要更多展示时间")
        elif complexity < 0.8:
            notes.append("简洁描述适合快速过渡")

        if notes:
            return "; ".join(notes)
        return "标准节奏"

    def _create_fallback_estimation(self, scene_id: str, scene_data: Scene) -> DurationEstimation:
        """创建降级估算"""
        return DurationEstimation(
            element_id=scene_id,
            element_type=ElementType.SCENE,
            original_duration=scene_data.duration,
            estimated_duration=scene_data.duration,
            confidence=0.3,
            adjustment_reason="规则估算失败，使用原始值",
            estimated_at=datetime.now().isoformat()
        )

    def _get_applied_rules(self, scene_type: str, scene_data: Scene) -> List[str]:
        """获取应用的规则"""
        rules = []

        # 场景类型规则
        rules.append(f"scene_type:{scene_type}")

        # 氛围规则
        mood = scene_data.mood
        if mood and mood != "平静":
            rules.append(f"mood:{mood}")

        # 时间规则
        if self.context.time_of_day != "day":
            rules.append(f"time:{self.context.time_of_day}")

        # 天气规则
        if self.context.weather != "clear":
            rules.append(f"weather:{self.context.weather}")

        # 角色数量规则
        if self.context.character_count > 1:
            rules.append(f"characters:{self.context.character_count}")

        return rules

    def _get_required_fields(self) -> List[str]:
        """获取必需字段"""
        return ["scene_id", "description"]
