"""
@FileName: scene_estimator.py
@Description: 场景时长估算器
@Author: HengLine
@Time: 2026/1/12 15:40
"""
from datetime import datetime
from typing import Dict, List, Any

from hengline.agent.temporal_planner.estimator.rule_base_estimator import BaseDurationEstimator
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, ElementType


class SceneDurationEstimator(BaseDurationEstimator):
    """场景时长估算器"""
    def _initialize_rules(self) -> Dict[str, Any]:
        """初始化场景估算规则"""
        return {
            # 场景类型基准时长（秒）
            "scene_type_baselines": {
                "establishing": 3.5,  # 建立镜头
                "detailed": 5.0,  # 详细场景
                "atmospheric": 4.0,  # 氛围场景
                "transitional": 2.0,  # 过渡场景
                "action_background": 3.0,  # 动作背景
                "emotional_background": 4.5  # 情感背景
            },

            # 文本复杂度系数
            "text_complexity_factors": {
                "very_low": 0.7,  # 0-0.5
                "low": 0.85,  # 0.5-1.0
                "medium": 1.0,  # 1.0-2.0
                "high": 1.2,  # 2.0-3.0
                "very_high": 1.4  # 3.0+
            },

            # 关键视觉元素时长（秒/元素）
            "key_visual_durations": {
                "major_setting": 0.8,  # 主要布景元素
                "character_intro": 1.2,  # 角色引入
                "important_prop": 0.6,  # 重要道具
                "atmospheric_element": 0.5,  # 氛围元素
                "background_detail": 0.3  # 背景细节
            },

            # 氛围类型调整因子
            "mood_adjustments": {
                "紧张": 0.9,  # 紧张场景节奏更快
                "压抑": 1.15,  # 压抑场景需要更多时间
                "孤独": 1.1,  # 孤独感需要时间建立
                "悲伤": 1.2,  # 悲伤情绪需要时间传达
                "喜悦": 1.05,  # 喜悦场景稍快
                "平静": 1.0,  # 平静场景标准
                "神秘": 1.25,  # 神秘感需要时间营造
                "浪漫": 1.15  # 浪漫场景节奏较慢
            },

            # 时间/天气调整
            "time_weather_adjustments": {
                "night": 1.15,  # 夜晚需要更多时间看清细节
                "dawn": 1.1,  # 黎明光线变化需要时间
                "dusk": 1.1,  # 黄昏同理
                "rain": 1.1,  # 雨景需要氛围时间
                "snow": 1.2,  # 雪景节奏更慢
                "fog": 1.25  # 雾景最慢
            },

            # 角色数量调整
            "character_count_adjustments": {
                1: 1.0,  # 单人场景
                2: 1.05,  # 双人场景
                3: 1.1,  # 三人场景
                4: 1.15,  # 四人场景
                5: 1.2  # 五人及以上
            },

            # 场景复杂度阈值
            "complexity_thresholds": {
                "simple": {"max_words": 30, "max_visuals": 2},
                "medium": {"max_words": 60, "max_visuals": 4},
                "complex": {"max_words": 100, "max_visuals": 6}
            }
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "enable_detailed_breakdown": True,
            "use_key_visual_analysis": True,
            "apply_mood_adjustments": True,
            "default_scene_type": "detailed",
            "min_scene_duration": 1.5,
            "max_scene_duration": 15.0,
            "confidence_calibration": {
                "text_completeness_weight": 0.4,
                "key_visuals_weight": 0.3,
                "mood_clarity_weight": 0.3
            }
        }

    def estimate(self, scene_data: Dict[str, Any]) -> DurationEstimation:
        """估算场景时长"""
        scene_id = scene_data.get("scene_id", "unknown_scene")

        # 1. 基础分析
        description = scene_data.get("description", "")
        mood = scene_data.get("mood", "平静")
        key_visuals = scene_data.get("key_visuals", [])
        location = scene_data.get("location", "")

        # 2. 确定场景类型
        scene_type = self._determine_scene_type(description, location, mood)

        # 3. 文本复杂度分析
        text_analysis = self._analyze_scene_text(description)

        # 4. 视觉元素分析
        visual_analysis = self._analyze_key_visuals(key_visuals)

        # 5. 计算基础时长
        base_duration = self._calculate_base_duration(scene_type, text_analysis, visual_analysis)

        # 6. 应用调整因子
        adjusted_duration = self._apply_scene_adjustments(base_duration, scene_data, text_analysis, visual_analysis)

        # 7. 确保在合理范围内
        final_duration = self._clamp_duration(adjusted_duration)

        # 8. 计算置信度
        confidence = self._calculate_scene_confidence(scene_data, text_analysis, visual_analysis)

        # 9. 计算时长范围
        # min_duration, max_duration = self._calculate_duration_range(final_duration, confidence)

        # 10. 构建结果
        # duration_breakdown = self._create_duration_breakdown(
        #     scene_type, text_analysis, visual_analysis, scene_data
        # ),
        # adjustment_factors = self._get_applied_adjustments(scene_data),
        # applied_rules = self._get_applied_rules(scene_type, scene_data)
        estimation = DurationEstimation(
            element_id=scene_id,
            element_type=ElementType.SCENE,
            original_duration=final_duration,
            estimated_duration=round(final_duration, 2),
            confidence=confidence,
            reasoning_breakdown={
                "scene_type": scene_type,
                "text_analysis": text_analysis,
                "visual_analysis": visual_analysis,
                "mood": mood,
                "location": location
            },
            visual_hints={},
            key_factors=[],
            pacing_notes="",
            emotional_weight=1,
            visual_complexity=1,
            character_states={},
            prop_states={},
            estimated_at=datetime.now().isoformat()
        )

        return estimation

    def _determine_scene_type(self, description: str, location: str, mood: str) -> str:
        """确定场景类型"""
        # 基于描述关键词判断
        description_lower = description.lower()
        location_lower = location.lower()

        # 建立镜头关键词
        establishing_keywords = ["远处", "全景", "俯瞰", "鸟瞰", "远景", "wide shot", "establishing"]
        if any(keyword in description_lower for keyword in establishing_keywords):
            return "establishing"

        # 过渡场景关键词
        transition_keywords = ["转场", "切换到", "接着", "随后", "transition", "cut to"]
        if any(keyword in description_lower for keyword in transition_keywords):
            return "transitional"

        # 氛围场景关键词
        atmospheric_keywords = ["氛围", "气氛", "感觉", "情绪", "atmosphere", "mood"]
        if any(keyword in description_lower for keyword in atmospheric_keywords):
            return "atmospheric"

        # 动作背景（如果描述中有动作发生）
        action_keywords = ["动作", "移动", "跑", "走", "打斗", "action", "movement"]
        if any(keyword in description_lower for keyword in action_keywords):
            return "action_background"

        # 情感背景（基于情绪）
        emotional_moods = ["悲伤", "喜悦", "愤怒", "恐惧", "孤独", "压抑", "紧张"]
        if any(emotion in mood for emotion in emotional_moods):
            return "emotional_background"

        # 默认：详细场景
        return "detailed"

    def _analyze_scene_text(self, description: str) -> Dict[str, Any]:
        """分析场景文本"""
        # 基础文本分析
        text_analysis = self._analyze_text_complexity(description)

        # 场景特有分析
        word_count = text_analysis["word_count"]

        # 判断复杂度级别
        complexity_level = "medium"
        if word_count <= 20:
            complexity_level = "very_low"
        elif word_count <= 40:
            complexity_level = "low"
        elif word_count <= 80:
            complexity_level = "medium"
        elif word_count <= 120:
            complexity_level = "high"
        else:
            complexity_level = "very_high"

        # 检测描述类型
        description_type = "neutral"
        if any(word in description for word in ["细节", "详细", "具体", "细致"]):
            description_type = "detailed"
        elif any(word in description for word in ["快速", "简短", "简洁", "概括"]):
            description_type = "brief"

        # 检测空间描述
        spatial_elements = 0
        spatial_keywords = ["左边", "右边", "前面", "后面", "上方", "下方", "中间", "旁边", "之间"]
        for keyword in spatial_keywords:
            if keyword in description:
                spatial_elements += 1

        text_analysis.update({
            "complexity_level": complexity_level,
            "description_type": description_type,
            "spatial_elements": spatial_elements,
            "has_detailed_description": description_type == "detailed"
        })

        return text_analysis

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

        visual_types = {
            "major_setting": ["房间", "客厅", "街道", "建筑", "城市", "森林", "setting", "room"],
            "character_intro": ["角色", "人物", "主角", "配角", "character", "person"],
            "important_prop": ["道具", "物品", "杯子", "手机", "书", "武器", "prop", "item"],
            "atmospheric_element": ["光线", "阴影", "颜色", "色调", "灯光", "light", "shadow", "color"],
            "background_detail": ["背景", "细节", "纹理", "装饰", "background", "detail"]
        }

        breakdown = {}
        total_duration = 0

        for visual in key_visuals:
            visual_lower = visual.lower()
            assigned_type = "background_detail"  # 默认

            for vtype, keywords in visual_types.items():
                if any(keyword in visual_lower for keyword in keywords):
                    assigned_type = vtype
                    break

            # 计算该视觉元素的时长
            duration_per_element = self.rules["key_visual_durations"].get(assigned_type, 0.3)

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
        base = self.rules["scene_type_baselines"].get(scene_type, 3.0)

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
                                 scene_data: Dict[str, Any],
                                 text_analysis: Dict[str, Any],
                                 visual_analysis: Dict[str, Any]) -> float:
        """应用场景特有调整"""
        adjusted = base_duration
        applied_factors = {}

        # 1. 氛围调整
        mood = scene_data.get("mood", "平静")
        mood_adjustment = self.rules["mood_adjustments"].get(mood, 1.0)
        adjusted *= mood_adjustment
        applied_factors["mood"] = mood_adjustment

        # 2. 时间/天气调整
        time_of_day = self.context.time_of_day
        weather = self.context.weather

        time_adjustment = self.rules["time_weather_adjustments"].get(time_of_day, 1.0)
        weather_adjustment = self.rules["time_weather_adjustments"].get(weather, 1.0)

        adjusted *= time_adjustment
        adjusted *= weather_adjustment
        applied_factors["time_of_day"] = time_adjustment
        applied_factors["weather"] = weather_adjustment

        # 3. 角色数量调整
        char_count = self.context.character_count
        char_adjustment = 1.0
        for count, adjustment in sorted(self.rules["character_count_adjustments"].items(), reverse=True):
            if char_count >= count:
                char_adjustment = adjustment
                break

        adjusted *= char_adjustment
        applied_factors["character_count"] = char_adjustment

        # 4. 详细描述加成
        if text_analysis.get("has_detailed_description", False):
            detailed_adjustment = 1.15
            adjusted *= detailed_adjustment
            applied_factors["detailed_description"] = detailed_adjustment

        # 5. 重要视觉元素加成
        if visual_analysis.get("has_important_visuals", False):
            important_visuals_adjustment = 1.1
            adjusted *= important_visuals_adjustment
            applied_factors["important_visuals"] = important_visuals_adjustment

        # 6. 整体节奏调整
        pacing_adjusted = self._apply_pacing_adjustment(adjusted)
        if pacing_adjusted != adjusted:
            applied_factors["pacing_adjustment"] = pacing_adjusted / adjusted
            adjusted = pacing_adjusted

        # 7. 上下文调整
        context_adjusted = self._apply_context_adjustments(adjusted, scene_data)
        if context_adjusted != adjusted:
            applied_factors["context_adjustment"] = context_adjusted / adjusted
            adjusted = context_adjusted

        # 保存应用的调整因子
        self._current_adjustment_factors = applied_factors

        return adjusted

    def _clamp_duration(self, duration: float) -> float:
        """确保时长在合理范围内"""
        min_duration = self.config.get("min_scene_duration", 1.5)
        max_duration = self.config.get("max_scene_duration", 15.0)

        return round(max(min_duration, min(duration, max_duration)), 2)

    def _calculate_scene_confidence(self, scene_data: Dict[str, Any],
                                    text_analysis: Dict[str, Any],
                                    visual_analysis: Dict[str, Any]) -> float:
        """计算场景估算置信度"""
        calibration = self.config.get("confidence_calibration", {})

        # 文本完整性得分
        text_completeness = 0.7  # 基础分
        if scene_data.get("description"):
            text_completeness = min(1.0, text_analysis.get("word_count", 0) / 50)

        # 关键视觉元素得分
        key_visuals_score = 0.5  # 基础分
        if visual_analysis.get("has_important_visuals", False):
            key_visuals_score = 0.8
        if visual_analysis.get("count", 0) >= 3:
            key_visuals_score = min(1.0, key_visuals_score + 0.2)

        # 氛围清晰度得分
        mood_clarity = 0.6  # 基础分
        mood = scene_data.get("mood", "")
        if mood and mood != "平静":
            mood_clarity = 0.8
        if mood in ["紧张", "压抑", "悲伤", "喜悦"]:  # 明确的情感
            mood_clarity = 0.9

        # 加权平均
        weights = {
            "text": calibration.get("text_completeness_weight", 0.4),
            "visuals": calibration.get("key_visuals_weight", 0.3),
            "mood": calibration.get("mood_clarity_weight", 0.3)
        }

        confidence = (
                text_completeness * weights["text"] +
                key_visuals_score * weights["visuals"] +
                mood_clarity * weights["mood"]
        )

        # 调用基类方法进行最终调整
        base_confidence = self._calculate_confidence(scene_data, text_analysis)

        # 综合置信度
        final_confidence = (confidence + base_confidence) / 2

        return round(final_confidence, 2)

    def _create_duration_breakdown(self, scene_type: str,
                                   text_analysis: Dict[str, Any],
                                   visual_analysis: Dict[str, Any],
                                   scene_data: Dict[str, Any]) -> Dict[str, float]:
        """创建时长分解详情"""
        if not self.config.get("enable_detailed_breakdown", True):
            return {}

        breakdown = {}

        # 1. 场景类型基础
        scene_base = self.rules["scene_type_baselines"].get(scene_type, 3.0)
        breakdown[f"scene_type_{scene_type}"] = scene_base

        # 2. 文本复杂度
        complexity_level = text_analysis.get("complexity_level", "medium")
        complexity_factor = self.rules["text_complexity_factors"].get(complexity_level, 1.0)
        text_duration = scene_base * (complexity_factor - 1.0)
        if text_duration > 0:
            breakdown["text_complexity"] = text_duration

        # 3. 视觉元素
        visual_breakdown = visual_analysis.get("breakdown", {})
        for vtype, duration in visual_breakdown.items():
            breakdown[f"visual_{vtype}"] = duration

        # 4. 空间元素
        spatial_elements = text_analysis.get("spatial_elements", 0)
        if spatial_elements > 0:
            breakdown["spatial_elements"] = spatial_elements * 0.2

        # 5. 氛围调整
        mood = scene_data.get("mood", "平静")
        mood_adjustment = self.rules["mood_adjustments"].get(mood, 1.0)
        if mood_adjustment != 1.0:
            mood_duration = sum(breakdown.values()) * (mood_adjustment - 1.0)
            if mood_duration != 0:
                breakdown["mood_adjustment"] = mood_duration

        return {k: round(v, 2) for k, v in breakdown.items()}

    def _get_applied_adjustments(self, scene_data: Dict[str, Any]) -> Dict[str, float]:
        """获取应用的调整因子"""
        adjustments = getattr(self, '_current_adjustment_factors', {})

        # 添加氛围调整
        mood = scene_data.get("mood", "平静")
        mood_adjustment = self.rules["mood_adjustments"].get(mood, 1.0)
        if "mood" not in adjustments:
            adjustments["mood"] = mood_adjustment

        return adjustments

    def _get_applied_rules(self, scene_type: str, scene_data: Dict[str, Any]) -> List[str]:
        """获取应用的规则"""
        rules = []

        # 场景类型规则
        rules.append(f"scene_type:{scene_type}")

        # 氛围规则
        mood = scene_data.get("mood", "")
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
