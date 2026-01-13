"""
@FileName: action_estimator.py
@Description: 基于规则的动作时长估算器
@Author: HengLine
@Time: 2026/1/12 15:41
"""
from abc import ABC
from datetime import datetime
from typing import Dict, List, Any, Tuple

from hengline.agent.script_parser.script_parser_models import Action
from hengline.agent.temporal_planner.estimator.rule_base_estimator import BaseDurationEstimator
from hengline.agent.temporal_planner.temporal_planner_model import DurationEstimation, ElementType
from hengline.logger import debug, error, info, warning
from utils.log_utils import print_log_exception

class ActionDurationEstimator(BaseDurationEstimator, ABC):
    """基于规则的动作时长估算器"""

    def _initialize_rules(self) -> Dict[str, Any]:
        """从配置加载动作估算规则"""
        action_config = self.planner_config.action_estimator
        self.action_keywords = self.keyword_config.get_action_keywords()

        if not action_config:
            warning(f"[WARNING] 未找到动作估算器配置，使用默认配置")
            action_config = self._get_default_action_config()

        rules = {
            "action_type_baselines": action_config.get("action_type_baselines", {}),
            "complexity_factors": action_config.get("complexity_factors", {}),
            "actor_type_adjustments": action_config.get("actor_type_adjustments", {}),
            "target_type_adjustments": action_config.get("target_type_adjustments", {}),
            "speed_keyword_adjustments": action_config.get("speed_keyword_adjustments", {}),
            "emotion_intensity_adjustments": action_config.get("emotion_intensity_adjustments", {}),
            "keywords": self.action_keywords,
            "config": action_config.get("config", {}),
        }

        info(f"[INFO] 动作规则加载完成，包含 {len(rules['action_type_baselines'])} 种动作类型")
        return rules

    def _get_default_action_config(self) -> Dict[str, Any]:
        """获取默认动作配置"""
        return {
            "action_type_baselines": {
                "posture": 2.0, "gaze": 1.5, "gesture": 1.2, "facial": 1.0,
                "physiological": 0.8, "interaction": 1.5, "prop_fall": 1.0,
                "device_alert": 2.0, "movement": 2.5, "complex_sequence": 4.0,
                "default": 1.5
            },
            "config": {
                "min_action_duration": 0.5,
                "max_action_duration": 10.0
            }
        }

    def estimate(self, action_data: Action) -> DurationEstimation:
        """估算动作时长"""
        action_id = action_data.action_id

        info(f"[INFO] 开始估算动作: {action_id}")

        try:
            # 1. 基础分析
            description = action_data.description
            actor = action_data.actor
            action_type = action_data.type
            target = action_data.target

            debug(f"[DEBUG] 动作描述: {description[:50]}...")
            debug(f"[DEBUG] 执行者: {actor}, 类型: {action_type}, 目标: {target}")

            # 2. 确定动作类型
            determined_action_type = self._determine_action_type(description, action_type)
            debug(f"[DEBUG] 确定动作类型: {determined_action_type}")

            # 3. 分析动作复杂度
            complexity_analysis = self._analyze_action_complexity(description)
            debug(f"[DEBUG] 复杂度分析: 级别={complexity_analysis.get('complexity_level')}, "
                  f"得分={complexity_analysis.get('complexity_score')}")

            # 4. 分析执行者和目标
            actor_analysis = self._analyze_actor(actor)
            target_analysis = self._analyze_target(target, description)

            # 5. 计算基础时长
            base_duration = self._calculate_base_duration(
                determined_action_type, complexity_analysis
            )
            debug(f"[DEBUG] 基础时长: {base_duration}秒")

            # 6. 应用调整因子
            adjusted_duration = self._apply_action_adjustments(
                base_duration, description, actor_analysis, target_analysis,
                complexity_analysis, action_data
            )
            debug(f"[DEBUG] 调整后时长: {adjusted_duration}秒")

            # 7. 确保在合理范围内
            final_duration = self._clamp_duration(adjusted_duration)
            debug(f"[DEBUG] 最终时长: {final_duration}秒")

            # 8. 计算置信度
            confidence = self._calculate_action_confidence(
                action_data, complexity_analysis, actor_analysis, target_analysis
            )
            debug(f"[DEBUG] 置信度: {confidence}")

            # 9. 创建 DurationEstimation 对象
            estimation = self._create_duration_estimation(
                action_id=action_id,
                original_duration=action_data.duration,
                rule_estimated_duration=final_duration,
                confidence=confidence,
                action_data=action_data,
                action_type=determined_action_type,
                complexity_analysis=complexity_analysis,
                actor_analysis=actor_analysis,
                target_analysis=target_analysis
            )

            info(f"[INFO] 动作估算完成: {final_duration}秒 (置信度: {confidence})")
            return estimation

        except Exception as e:
            print_log_exception()
            error(f"[ERROR] 动作估算失败 {action_id}: {str(e)}")
            return self._create_fallback_estimation(action_id, action_data)

    def _determine_action_type(self, description: str,
                               declared_type: str = "") -> str:
        """确定动作类型"""
        # 优先使用声明的类型
        if declared_type and declared_type in self.rules["action_type_baselines"]:
            return declared_type

        # 从描述中推断类型
        description_lower = description.lower()
        action_type_keywords = self.action_keywords.get("action_types")

        for action_type, keywords in action_type_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return action_type

        # 使用默认类型
        return self.rules["config"].get("default_action_type", "gesture")

    def _analyze_action_complexity(self, description: str) -> Dict[str, Any]:
        """分析动作复杂度"""
        # 基础文本分析
        text_analysis = self._analyze_text_complexity(description)
        word_count = text_analysis["word_count"]

        # 分析动作组成部分
        components = self._extract_action_components(description)
        component_count = len(components)

        # 检测精细动作
        fine_motor_count = self._count_fine_motor_actions(description)

        # 检测速度关键词
        speed_keywords = self._detect_speed_keywords(description)

        # 检测情感强度
        emotion_intensity = self._detect_emotion_intensity(description)

        # 计算复杂度得分
        complexity_score = self._calculate_complexity_score(
            word_count, component_count, fine_motor_count, emotion_intensity
        )

        # 确定复杂度级别
        complexity_level = self._determine_complexity_level(complexity_score)

        return {
            "word_count": word_count,
            "component_count": component_count,
            "fine_motor_count": fine_motor_count,
            "speed_keywords": speed_keywords,
            "emotion_intensity": emotion_intensity,
            "complexity_score": complexity_score,
            "complexity_level": complexity_level,
            "components": components
        }

    def _extract_action_components(self, description: str) -> List[str]:
        """提取动作组成部分"""
        components = []

        # 动作动词列表
        action_verbs = [
            "走", "跑", "跳", "坐", "站", "躺", "看", "盯", "望",
            "拿", "放", "接", "递", "按", "摸", "拍", "打", "推",
            "拉", "转", "扭", "弯", "伸", "缩", "呼吸", "颤抖",
            "笑", "哭", "皱眉", "点头", "摇头", "挥手"
        ]

        for verb in action_verbs:
            if verb in description:
                components.append(verb)

        return components

    def _count_fine_motor_actions(self, description: str) -> int:
        """计算精细动作数量"""
        fine_motor_keywords = self.action_keywords.get("body_parts", {}).get("fine_motor", [])
        count = 0

        for keyword in fine_motor_keywords:
            if keyword in description:
                count += 1

        return count

    def _detect_speed_keywords(self, description: str) -> Dict[str, List[str]]:
        """检测速度关键词"""
        speed_keywords_config = self.action_keywords.get("speed_keywords")
        detected = {"fast": [], "slow": []}

        for speed_type, keywords in speed_keywords_config.items():
            for keyword in keywords:
                if keyword in description:
                    detected[speed_type].append(keyword)

        return detected

    def _detect_emotion_intensity(self, description: str) -> str:
        """检测情感强度"""
        emotion_keywords = self.action_keywords.get("emotion_intensity_keywords")

        for intensity_level, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in description:
                    return intensity_level

        return "mild"  # 默认轻微



    def _determine_complexity_level(self, complexity_score: float) -> str:
        """确定复杂度级别"""
        if complexity_score < 1.0:
            return "simple"
        elif complexity_score < 2.0:
            return "medium"
        elif complexity_score < 3.5:
            return "complex"
        else:
            return "very_complex"

    def _analyze_actor(self, actor: str) -> Dict[str, Any]:
        """分析执行者"""
        # 判断执行者类型
        actor_type = "human_main"  # 默认主要角色

        if not actor or actor == "":
            actor_type = "none"
        elif "道具" in actor or "物品" in actor:
            actor_type = "prop"
        elif any(word in actor for word in ["主角", "主要", "主演", "女主", "男主", "主人公", "女主角", "男主角"]):
            actor_type = "human_main"
        elif any(word in actor for word in ["配角", "次要", "其他"]):
            actor_type = "human_supporting"

        return {
            "actor_name": actor,
            "actor_type": actor_type,
            "adjustment_factor": self.rules["actor_type_adjustments"].get(
                actor_type, self.rules["actor_type_adjustments"].get("default", 1.0)
            )
        }

    def _analyze_target(self, target: str, description: str) -> Dict[str, Any]:
        """分析目标"""
        # 判断目标类型
        target_type = "none"
        target_keywords = self.action_keywords.get("target_keywords")

        if not target or target == "":
            # 从描述中推断
            for ttype, keywords in target_keywords.items():
                for keyword in keywords:
                    if keyword in description:
                        target_type = ttype
                        break
                if target_type != "none":
                    break
        else:
            # 根据目标内容判断
            target_lower = target.lower()
            for ttype, keywords in target_keywords.items():
                for keyword in keywords:
                    if keyword in target_lower:
                        target_type = ttype
                        break
                if target_type != "none":
                    break

        return {
            "target_name": target,
            "target_type": target_type,
            "adjustment_factor": self.rules["target_type_adjustments"].get(
                target_type, self.rules["target_type_adjustments"].get("default", 1.0)
            )
        }

    def _calculate_base_duration(self, action_type: str,
                                 complexity_analysis: Dict[str, Any]) -> float:
        """计算基础时长"""
        # 1. 动作类型基础值
        base = self.rules["action_type_baselines"].get(
            action_type,
            self.rules["action_type_baselines"].get("default", 1.5)
        )

        # 2. 复杂度调整
        complexity_level = complexity_analysis.get("complexity_level", "simple")
        complexity_factor = self.rules["complexity_factors"].get(complexity_level, 1.0)
        base *= complexity_factor

        return base

    def _apply_action_adjustments(self, base_duration: float, description: str,
                                  actor_analysis: Dict[str, Any],
                                  target_analysis: Dict[str, Any],
                                  complexity_analysis: Dict[str, Any],
                                  action_data: Action) -> float:
        """应用动作特有调整"""
        adjusted = base_duration
        applied_factors = {}

        config = self.rules.get("config", {})

        # 1. 执行者调整
        if config.get("apply_actor_adjustments", True):
            actor_factor = actor_analysis.get("adjustment_factor", 1.0)
            adjusted *= actor_factor
            applied_factors["actor"] = actor_factor
            debug(f"[DEBUG] 执行者调整: {actor_factor}")

        # 2. 目标调整
        if config.get("apply_target_adjustments", True):
            target_factor = target_analysis.get("adjustment_factor", 1.0)
            adjusted *= target_factor
            applied_factors["target"] = target_factor
            debug(f"[DEBUG] 目标调整: {target_factor}")

        # 3. 速度关键词调整
        if config.get("apply_speed_adjustments", True):
            speed_factor = self._calculate_speed_adjustment(description)
            adjusted *= speed_factor
            applied_factors["speed"] = speed_factor
            debug(f"[DEBUG] 速度调整: {speed_factor}")

        # 4. 情感强度调整
        if config.get("apply_emotion_adjustments", True):
            emotion_factor = self._calculate_emotion_adjustment(description)
            adjusted *= emotion_factor
            applied_factors["emotion"] = emotion_factor
            debug(f"[DEBUG] 情感调整: {emotion_factor}")

        # 5. 精细动作加成
        fine_motor_count = complexity_analysis.get("fine_motor_count", 0)
        if fine_motor_count > 0:
            fine_motor_factor = 1.0 + (fine_motor_count * 0.05)
            adjusted *= fine_motor_factor
            applied_factors["fine_motor"] = fine_motor_factor
            debug(f"[DEBUG] 精细动作调整: {fine_motor_factor}")

        # 6. 整体节奏调整
        pacing_adjusted = self._apply_pacing_adjustment(adjusted)
        if pacing_adjusted != adjusted:
            applied_factors["pacing"] = pacing_adjusted / adjusted
            adjusted = pacing_adjusted
            debug(f"[DEBUG] 节奏调整: {applied_factors['pacing']}")

        # 7. 上下文调整
        context_adjusted = self._apply_context_adjustments(adjusted)
        if context_adjusted != adjusted:
            applied_factors["context"] = context_adjusted / adjusted
            adjusted = context_adjusted
            debug(f"[DEBUG] 上下文调整: {applied_factors['context']}")

        # 保存应用的调整因子
        self._current_adjustment_factors = applied_factors

        return adjusted

    def _calculate_speed_adjustment(self, description: str) -> float:
        """计算速度调整因子"""
        speed_keywords = self._detect_speed_keywords(description)
        adjustment = 1.0

        # 检查快速关键词
        if speed_keywords.get("fast"):
            for keyword in speed_keywords["fast"]:
                keyword_adjustment = self.rules["speed_keyword_adjustments"].get(
                    keyword, self.rules["speed_keyword_adjustments"].get("default", 1.0)
                )
                adjustment = min(adjustment, keyword_adjustment)  # 取最小值（最快）

        # 检查慢速关键词
        if speed_keywords.get("slow"):
            for keyword in speed_keywords["slow"]:
                keyword_adjustment = self.rules["speed_keyword_adjustments"].get(
                    keyword, self.rules["speed_keyword_adjustments"].get("default", 1.0)
                )
                adjustment = max(adjustment, keyword_adjustment)  # 取最大值（最慢）

        return adjustment

    def _calculate_emotion_adjustment(self, description: str) -> float:
        """计算情感调整因子"""
        emotion_intensity = self._detect_emotion_intensity(description)

        return self.rules["emotion_intensity_adjustments"].get(
            emotion_intensity,
            self.rules["emotion_intensity_adjustments"].get("default", 1.0)
        )

    def _clamp_duration(self, duration: float) -> float:
        """确保时长在合理范围内"""
        config = self.rules.get("config", {})
        min_duration = config.get("min_action_duration", 0.5)
        max_duration = config.get("max_action_duration", 10.0)

        clamped = max(min_duration, min(duration, max_duration))
        return round(clamped, 2)

    def _calculate_action_confidence(self, action_data: Action,
                                     complexity_analysis: Dict[str, Any],
                                     actor_analysis: Dict[str, Any],
                                     target_analysis: Dict[str, Any]) -> float:
        """计算动作估算置信度"""
        config = self.rules.get("config", {})
        confidence_weights = config.get("confidence_weights", {})

        # 1. 描述完整性得分
        description_completeness = 0.6
        description = action_data.description
        if description:
            word_count = len(description.split())
            description_completeness = min(1.0, word_count / 15)  # 15词为完整

        # 2. 动作类型清晰度得分
        action_type_clarity = 0.7
        declared_type = action_data.type
        if declared_type and declared_type in self.rules["action_type_baselines"]:
            action_type_clarity = 0.9

        # 3. 目标明确性得分
        target_specificity = 0.5
        target = action_data.target
        if target:
            target_specificity = 0.8
        target_type = target_analysis.get("target_type", "none")
        if target_type != "none":
            target_specificity = 0.9

        # 4. 情感明确性得分
        emotion_clarity = 0.5
        emotion_intensity = complexity_analysis.get("emotion_intensity", "mild")
        if emotion_intensity != "mild":
            emotion_clarity = 0.8

        # 加权平均
        confidence = (
                description_completeness * confidence_weights.get("description_completeness", 0.4) +
                action_type_clarity * confidence_weights.get("action_type_clarity", 0.3) +
                target_specificity * confidence_weights.get("target_specificity", 0.2) +
                emotion_clarity * confidence_weights.get("emotion_clarity", 0.1)
        )

        # 调用基类方法进行最终调整
        base_confidence = self._calculate_confidence(action_data, complexity_analysis)

        # 综合置信度
        final_confidence = (confidence + base_confidence) / 2

        debug(f"[DEBUG] 置信度计算: 描述={description_completeness:.2f}, "
              f"类型={action_type_clarity:.2f}, 目标={target_specificity:.2f}, "
              f"情感={emotion_clarity:.2f}, 最终={final_confidence:.2f}")

        return round(final_confidence, 2)

    def _create_duration_estimation(self, action_id: str, original_duration: float,
                                    rule_estimated_duration: float, confidence: float,
                                    action_data: Action, action_type: str,
                                    complexity_analysis: Dict[str, Any],
                                    actor_analysis: Dict[str, Any],
                                    target_analysis: Dict[str, Any]) -> DurationEstimation:
        """创建 DurationEstimation 对象"""
        # 计算情感权重和视觉复杂度
        emotional_weight = self._calculate_emotional_weight(action_data, complexity_analysis)
        visual_complexity = self._calculate_visual_complexity(complexity_analysis)

        # 提取状态信息
        character_states, prop_states = self._extract_state_changes(action_data)

        # 构建推理详情
        reasoning_breakdown = self._create_reasoning_breakdown(
            action_type, complexity_analysis, actor_analysis,
            target_analysis, rule_estimated_duration
        )

        # 构建关键因素
        key_factors = self._extract_key_factors(
            action_data, complexity_analysis, actor_analysis, target_analysis
        )

        # 构建视觉提示
        visual_hints = self._create_visual_hints(action_data, complexity_analysis)

        # 构建调整原因
        adjustment_reason = self._build_adjustment_reason()

        # 创建 DurationEstimation
        return DurationEstimation(
            element_id=action_id,
            element_type=ElementType.ACTION,
            original_duration=original_duration,
            estimated_duration=rule_estimated_duration,
            confidence=confidence,
            rule_based_estimate=rule_estimated_duration,
            adjustment_reason=adjustment_reason,
            emotional_weight=emotional_weight,
            visual_complexity=visual_complexity,
            pacing_factor=self._calculate_pacing_factor(action_data, complexity_analysis),
            character_states=character_states,
            prop_states=prop_states,
            reasoning_breakdown=reasoning_breakdown,
            visual_hints=visual_hints,
            key_factors=key_factors,
            pacing_notes=self._generate_pacing_notes(action_data, complexity_analysis),
            estimated_at=datetime.now().isoformat()
        )

    def _calculate_emotional_weight(self, action_data: Action,
                                    complexity_analysis: Dict[str, Any]) -> float:
        """计算情感权重"""
        emotion_intensity = complexity_analysis.get("emotion_intensity", "mild")

        intensity_weights = {
            "mild": 1.0,
            "moderate": 1.3,
            "strong": 1.6,
            "dramatic": 2.0
        }

        return intensity_weights.get(emotion_intensity, 1.0)

    def _calculate_visual_complexity(self, complexity_analysis: Dict[str, Any]) -> float:
        """计算视觉复杂度"""
        base_complexity = 1.0

        # 动作组成部分
        component_count = complexity_analysis.get("component_count", 0)
        base_complexity += min(component_count / 5, 1.0)

        # 精细动作
        fine_motor_count = complexity_analysis.get("fine_motor_count", 0)
        base_complexity += fine_motor_count * 0.2

        # 情感强度
        emotion_intensity = complexity_analysis.get("emotion_intensity", "mild")
        if emotion_intensity in ["strong", "dramatic"]:
            base_complexity += 0.3

        return round(min(base_complexity, 3.0), 2)

    def _extract_state_changes(self, action_data: Action) -> Tuple[Dict[str, str], Dict[str, str]]:
        """提取状态变化信息"""
        character_states = {}
        prop_states = {}

        description = action_data.description
        actor = action_data.actor

        # 角色状态变化
        if "林然" in actor:
            if "坐直" in description:
                character_states["林然"] = "posture:从蜷坐到挺直"
            elif "手指收紧" in description or "指节泛白" in description:
                character_states["林然"] = "hand_tension:放松到紧绷"
            elif "泪水" in description:
                character_states["林然"] = "emotional:泪水盈眶"
            elif "盯着" in description:
                character_states["林然"] = "gaze:专注注视手机"

        # 道具状态变化
        if "旧羊毛毯" in actor and "滑落" in description:
            prop_states["旧羊毛毯"] = "position:从肩头滑落到地板"
        elif "手机" in actor and "亮起" in description:
            prop_states["手机"] = "state:屏幕从暗到亮"
        elif "凉茶" in description and "凝出" in description:
            prop_states["凉茶"] = "state:凝出水雾"

        return character_states, prop_states

    def _create_reasoning_breakdown(self, action_type: str,
                                    complexity_analysis: Dict[str, Any],
                                    actor_analysis: Dict[str, Any],
                                    target_analysis: Dict[str, Any],
                                    total_duration: float) -> Dict[str, Any]:
        """创建推理详情"""
        breakdown = {}

        # 动作类型基础
        action_base = self.rules["action_type_baselines"].get(
            action_type,
            self.rules["action_type_baselines"].get("default", 1.5)
        )
        breakdown["action_type_base"] = action_base

        # 复杂度调整
        complexity_level = complexity_analysis.get("complexity_level", "simple")
        complexity_factor = self.rules["complexity_factors"].get(complexity_level, 1.0)
        if complexity_factor != 1.0:
            breakdown["complexity_adjustment"] = action_base * (complexity_factor - 1.0)

        # 执行者和目标调整
        actor_factor = actor_analysis.get("adjustment_factor", 1.0)
        if actor_factor != 1.0:
            breakdown["actor_adjustment"] = action_base * (actor_factor - 1.0)

        target_factor = target_analysis.get("adjustment_factor", 1.0)
        if target_factor != 1.0:
            breakdown["target_adjustment"] = action_base * (target_factor - 1.0)

        # 应用的所有调整因子
        adjustment_factors = getattr(self, '_current_adjustment_factors', {})
        if adjustment_factors:
            breakdown["adjustment_factors"] = adjustment_factors

        return breakdown

    def _extract_key_factors(self, action_data: Action,
                             complexity_analysis: Dict[str, Any],
                             actor_analysis: Dict[str, Any],
                             target_analysis: Dict[str, Any]) -> List[str]:
        """提取关键因素"""
        factors = []

        # 动作类型
        action_type = self._determine_action_type(
            action_data.description,
            action_data.type
        )
        factors.append(f"动作类型: {action_type}")

        # 复杂度
        complexity_level = complexity_analysis.get("complexity_level", "simple")
        if complexity_level != "simple":
            factors.append(f"复杂度: {complexity_level}")

        # 情感强度
        emotion_intensity = complexity_analysis.get("emotion_intensity", "mild")
        if emotion_intensity != "mild":
            factors.append(f"情感强度: {emotion_intensity}")

        # 执行者类型
        actor_type = actor_analysis.get("actor_type", "human_main")
        if actor_type != "human_main":
            factors.append(f"执行者: {actor_type}")

        # 目标类型
        target_type = target_analysis.get("target_type", "none")
        if target_type != "none":
            factors.append(f"目标: {target_type}")

        return factors

    def _create_visual_hints(self, action_data: Action,
                             complexity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """创建视觉提示"""
        hints = {}

        description = action_data.description

        # 镜头建议
        if any(word in description for word in ["手指", "指尖", "眼睛", "泪水"]):
            hints["suggested_shot_types"] = ["extreme_close_up", "close_up"]
        elif any(word in description for word in ["走", "跑", "移动"]):
            hints["suggested_shot_types"] = ["medium_shot", "tracking_shot"]
        else:
            hints["suggested_shot_types"] = ["medium_close_up"]

        # 动作速度
        speed_keywords = complexity_analysis.get("speed_keywords", {})
        if speed_keywords.get("fast"):
            hints["motion_suggestion"] = "fast_pacing, quick_cuts"
        elif speed_keywords.get("slow"):
            hints["motion_suggestion"] = "slow_motion, lingering"

        # 焦点建议
        if any(word in description for word in ["手机", "杯子", "道具"]):
            hints["focus_elements"] = ["prop_in_hand", "hand_prop_interaction"]
        elif any(word in description for word in ["看", "盯", "注视"]):
            hints["focus_elements"] = ["eye_line", "gaze_direction"]

        return hints

    def _build_adjustment_reason(self) -> str:
        """构建调整原因"""
        reasons = []
        adjustment_factors = getattr(self, '_current_adjustment_factors', {})

        for factor, value in adjustment_factors.items():
            if value != 1.0:
                change = "加速" if value < 1.0 else "减速"
                percent = abs(value - 1.0) * 100
                reasons.append(f"{factor}{change}{percent:.0f}%")

        if reasons:
            return f"动作调整: {', '.join(reasons)}"
        return "标准动作估算"

    def _calculate_pacing_factor(self, action_data: Action,
                                 complexity_analysis: Dict[str, Any]) -> float:
        """计算节奏因子"""
        factor = 1.0

        # 速度关键词影响
        speed_keywords = complexity_analysis.get("speed_keywords", {})
        if speed_keywords.get("fast"):
            factor *= 1.2
        elif speed_keywords.get("slow"):
            factor *= 0.8

        # 情感强度影响
        emotion_intensity = complexity_analysis.get("emotion_intensity", "mild")
        if emotion_intensity in ["strong", "dramatic"]:
            factor *= 1.1  # 强烈情感通常需要更多时间

        return round(factor, 2)

    def _generate_pacing_notes(self, action_data: Action,
                               complexity_analysis: Dict[str, Any]) -> str:
        """生成节奏说明"""
        notes = []

        # 速度说明
        speed_keywords = complexity_analysis.get("speed_keywords", {})
        if speed_keywords.get("fast"):
            notes.append("快速动作")
        elif speed_keywords.get("slow"):
            notes.append("缓慢动作")

        # 复杂度说明
        complexity_level = complexity_analysis.get("complexity_level", "simple")
        if complexity_level != "simple":
            notes.append(f"{complexity_level}动作")

        if notes:
            return "; ".join(notes)
        return "标准动作节奏"

    def _create_fallback_estimation(self, action_id: str,
                                    action_data: Action) -> DurationEstimation:
        """创建降级估算"""
        return DurationEstimation(
            element_id=action_id,
            element_type=ElementType.ACTION,
            original_duration=action_data.duration,
            estimated_duration=action_data.duration,
            confidence=0.3,
            adjustment_reason="动作估算失败，使用原始值",
            estimated_at=datetime.now().isoformat()
        )

    def _get_required_fields(self) -> List[str]:
        """获取必需字段"""
        return ["action_id", "description"]
