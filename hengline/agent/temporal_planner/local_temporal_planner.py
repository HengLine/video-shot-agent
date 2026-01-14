# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: 基于规则的启发式算法，实现时序规划（负责将剧本按5秒粒度切分，估算动作时长）
@Author: HengLine
@Time: 2025/10 - 2025/12
"""
from datetime import datetime
from typing import List, Dict, Tuple

from hengline.agent.script_parser.script_parser_models import UnifiedScript
from hengline.agent.temporal_planner.base_temporal_planner import TemporalPlanner
from hengline.agent.temporal_planner.estimator.rule_action_estimator import RuleActionDurationEstimator
from hengline.agent.temporal_planner.estimator.rule_base_estimator import EstimationContext, ElementWithContext
from hengline.agent.temporal_planner.estimator.rule_dialogue_estimator import RuleDialogueDurationEstimator
from hengline.agent.temporal_planner.estimator.rule_scene_estimator import RuleSceneDurationEstimator
from hengline.agent.temporal_planner.temporal_planner_model import TimelinePlan, ElementType, TimeSegment, DurationEstimation, PacingAnalysis, ContinuityAnchor
from hengline.logger import debug, error, info, warning
from utils.log_utils import print_log_exception


class LocalRuleTemporalPlanner(TemporalPlanner):
    """基于规则的动作合并算法
            规则优先级：
            1. 情感强烈变化点（如震惊）→ 必须独立镜头
            2. 对话前后 → 通常拆分
            3. 物理位置/视角变化 → 建议拆分
            4. 时长填充与合并
    """

    def __init__(self):
        """初始化时序规划智能体"""

        # 初始化估算器
        self.performance_stats = {}
        self.unified_script = None
        self.estimators = {
            ElementType.SCENE: RuleSceneDurationEstimator(),
            ElementType.DIALOGUE: RuleDialogueDurationEstimator(),
            ElementType.ACTION: RuleActionDurationEstimator(),
            ElementType.SILENCE: RuleDialogueDurationEstimator()  # 沉默也使用对话估算器
        }

        # 状态跟踪
        self.element_order = []
        self.character_states = {}
        self.prop_states = {}
        self.estimations = {}

        self.max_segment_duration = 5.0  # 每个片段的最大时长（秒）

        # 当前上下文
        self.current_context = EstimationContext()

        # 初始化LangChain记忆工具（替代原有的向量记忆+状态机）
        # self.memory_tool = LangChainMemoryTool()

    def plan_timeline(self, structured_script: UnifiedScript) -> TimelinePlan | None:
        """
        规划剧本的时序分段
        
        Args:
            structured_script: 结构化的剧本

        Returns:
            分段计划列表
        """
        # 加载剧本
        self.load_unified_script(structured_script)

        # 1. 估算所有元素
        estimations = self.estimate_all_elements()

        # 2. 分割为片段
        segments = self.split_into_segments(self.max_segment_duration)

        # 3. 分析节奏
        pacing_analysis = self.analyze_pacing(segments)

        # 4. 生成连续性锚点
        continuity_anchors = self.generate_continuity_anchors(segments)

        # 5. 计算总时长
        total_duration = segments[-1].time_range[1] if segments else 0

        # 7. 确定全局视觉风格和主导情感
        global_visual_style = "写实"
        dominant_emotion = "平稳"

        # 8. 找到关键转折点
        key_transition_points = []
        for turning_point in pacing_analysis.key_points:
            if turning_point["type"] == "peak":
                key_transition_points.append(turning_point["time"])

        # 创建 TimelinePlan
        timeline_plan = TimelinePlan(
            timeline_segments=segments,
            duration_estimations=estimations,
            pacing_analysis=pacing_analysis,
            continuity_anchors=continuity_anchors,
            total_duration=total_duration,
            segments_count=len(segments),
            elements_count=len(estimations),
            global_visual_style=global_visual_style,
            dominant_emotion=dominant_emotion,
            key_transition_points=key_transition_points
        )

        debug("时序规划创建完成")
        return timeline_plan

    def load_unified_script(self, unified_script: UnifiedScript) -> 'LocalRuleTemporalPlanner':
        """加载统一剧本"""
        self.unified_script = unified_script
        self._extract_element_order()
        self._update_global_context()
        return self

    def _extract_element_order(self):
        """提取元素的原始时间顺序"""
        if not self.unified_script:
            return

        # 从剧本数据中提取所有元素
        all_elements = []

        # 场景
        for scene in self.unified_script.scenes:
            all_elements.append(ElementWithContext(
                element_id=scene.scene_id,
                element_type=ElementType.SCENE,
                data=scene,
                time_offset=scene.start_time
            ))

        # 对话
        for dialogue in self.unified_script.dialogues:
            element_type = ElementType.SILENCE if dialogue.type == "silence" else ElementType.DIALOGUE
            all_elements.append(ElementWithContext(
                element_id=dialogue.dialogue_id,
                element_type=element_type,
                data=dialogue,
                time_offset=dialogue.time_offset
            ))

        # 动作
        for action in self.unified_script.actions:
            all_elements.append(ElementWithContext(
                element_id=action.action_id,
                element_type=ElementType.ACTION,
                data=action,
                time_offset=action.time_offset
            ))

        # 按时间偏移排序
        all_elements.sort(key=lambda x: x.time_offset)
        self.element_order = all_elements

        debug(f"提取了 {len(self.element_order)} 个元素")

    def _update_global_context(self):
        """更新全局上下文"""
        if not self.unified_script:
            return

        # 从剧本数据更新上下文
        for estimator in self.estimators.values():
            # 将 UnifiedScript 转换为字典格式（假设的方法）
            estimator.update_context_from_script(self.unified_script)
            estimator.set_context(self.current_context)

        debug("全局上下文已更新")

    def estimate_all_elements(self) -> Dict[str, DurationEstimation]:
        """估算所有元素的时长"""
        if not self.unified_script:
            raise ValueError("请先加载统一剧本")

        info(f"开始估算 {len(self.element_order)} 个元素...")
        self.estimations = {}

        start_time = datetime.now()

        # 顺序估算每个元素
        for i, element in enumerate(self.element_order):
            debug(f"估算元素 {i + 1}/{len(self.element_order)}: {element.element_id} ({element.element_type.value})")

            try:
                # 获取对应的估算器
                estimator = self.estimators.get(element.element_type)
                if not estimator:
                    warning(f"没有找到 {element.element_type.value} 类型的估算器，跳过")
                    continue

                # 更新元素级上下文
                self._update_element_context(element, i)
                estimator.set_context(self.current_context)

                # 执行估算
                estimation = estimator.estimate(element.data)

                # 验证估算结果
                validated_estimation = self._validate_estimation(estimation, element)

                self.estimations[element.element_id] = validated_estimation
                debug(f"  -> {validated_estimation.estimated_duration}秒 (置信度: {validated_estimation.confidence})")

            except Exception as e:
                print_log_exception()
                error(f"元素 {element.element_id} 估算失败: {str(e)}")
                # 创建降级估算
                fallback = self._create_fallback_estimation(element)
                self.estimations[element.element_id] = fallback

        # 计算性能统计
        elapsed = (datetime.now() - start_time).total_seconds()
        self.performance_stats = {
            "total_elements": len(self.element_order),
            "estimated_elements": len(self.estimations),
            "estimation_time_seconds": round(elapsed, 2),
            "avg_time_per_element": round(elapsed / len(self.element_order), 3) if self.element_order else 0
        }

        info(f"时长估算完成，成功 {len(self.estimations)}/{len(self.element_order)} 个元素")
        return self.estimations

    def _update_element_context(self, element: ElementWithContext, position: int):
        """更新元素级上下文"""
        # 更新节奏上下文
        if position > 0:
            # 基于前一个元素的节奏
            prev_element = self.element_order[position - 1]
            prev_estimation = self.estimations.get(prev_element.element_id)
            if prev_estimation:
                # 根据前一个元素的节奏因子调整
                if prev_estimation.pacing_factor > 1.2:
                    self.current_context.previous_pacing = "fast"
                elif prev_estimation.pacing_factor < 0.8:
                    self.current_context.previous_pacing = "slow"
                else:
                    self.current_context.previous_pacing = "normal"

        # 基于元素类型更新情感基调
        if element.element_type == ElementType.SCENE:
            mood = element.data.mood
            if mood:
                self.current_context.emotional_tone = self._normalize_emotional_tone(mood)

        # 如果是对话，考虑对话情感
        elif element.element_type == ElementType.DIALOGUE:
            emotion = element.data.emotion
            if emotion:
                self.current_context.emotional_tone = self._normalize_emotional_tone(emotion)

    def _normalize_emotional_tone(self, mood_str: str) -> str:
        """规范化情绪基调"""
        if any(word in mood_str for word in ["紧张", "激动", "愤怒", "恐惧"]):
            return "tense"
        elif any(word in mood_str for word in ["悲伤", "忧郁", "孤独", "压抑"]):
            return "emotional"
        elif any(word in mood_str for word in ["喜悦", "兴奋", "轻松"]):
            return "relaxed"
        return "neutral"

    def _validate_estimation(self, estimation: DurationEstimation,
                             element: ElementWithContext) -> DurationEstimation:
        """验证估算结果"""
        # 检查时长是否在合理范围内
        element_type = element.element_type
        min_duration, max_duration = self._get_duration_limits(element_type)

        if estimation.estimated_duration < min_duration:
            warning(f"元素 {element.element_id} 时长过短，调整为 {min_duration}秒")
            estimation.estimated_duration = min_duration
            estimation.confidence = min(estimation.confidence, 0.6)

        elif estimation.estimated_duration > max_duration:
            warning(f"元素 {element.element_id} 时长过长，调整为 {max_duration}秒")
            estimation.estimated_duration = max_duration
            estimation.confidence = min(estimation.confidence, 0.6)

        return estimation

    def _get_duration_limits(self, element_type: ElementType) -> Tuple[float, float]:
        """获取时长限制"""
        limits = {
            ElementType.SCENE: (1.5, 15.0),
            ElementType.DIALOGUE: (0.5, 8.0),
            ElementType.SILENCE: (0.5, 6.0),
            ElementType.ACTION: (0.3, 10.0)
        }
        return limits.get(element_type, (0.5, 5.0))

    def _create_fallback_estimation(self, element: ElementWithContext) -> DurationEstimation:
        """创建降级估算"""
        original_duration = element.data.duration

        # 基于元素类型的默认时长
        default_durations = {
            ElementType.SCENE: 4.0,
            ElementType.DIALOGUE: 2.5,
            ElementType.SILENCE: 2.0,
            ElementType.ACTION: 1.5
        }

        fallback_duration = default_durations.get(element.element_type, original_duration)

        return DurationEstimation(
            element_id=element.element_id,
            element_type=element.element_type,
            original_duration=original_duration,
            estimated_duration=fallback_duration,
            confidence=0.3,
            adjustment_reason="规则估算失败，使用默认值",
            estimated_at=datetime.now().isoformat()
        )

    def split_into_segments(self, max_segment_duration: float = 5.0) -> List[TimeSegment]:
        """将估算后的元素切分为5秒片段"""
        if not self.estimations:
            raise ValueError("请先执行时长估算")

        debug(f"开始{max_segment_duration}秒分片...")

        segments = []
        current_segment_elements = []
        current_segment_duration = 0.0
        current_start_time = 0.0
        segment_index = 1

        for element in self.element_order:
            element_id = element.element_id

            if element_id not in self.estimations:
                continue

            estimation = self.estimations[element_id]
            element_duration = estimation.estimated_duration

            # 检查是否可以放入当前片段
            if current_segment_duration + element_duration <= max_segment_duration:
                # 可以放入当前片段
                current_segment_elements.append({
                    "element_id": element_id,
                    "element_type": estimation.element_type.value,
                    "duration": element_duration,
                    "start_in_segment": current_segment_duration,
                    "estimation": estimation.to_dict()
                })
                current_segment_duration += element_duration
            else:
                # 当前片段已满，创建新片段
                if current_segment_elements:
                    segment = self._create_segment(
                        segment_index, current_start_time, current_segment_duration,
                        current_segment_elements
                    )
                    segments.append(segment)
                    segment_index += 1
                    current_start_time += current_segment_duration

                # 开始新片段
                current_segment_elements = [{
                    "element_id": element_id,
                    "element_type": estimation.element_type.value,
                    "duration": element_duration,
                    "start_in_segment": 0.0,
                    "estimation": estimation.to_dict()
                }]
                current_segment_duration = element_duration

        # 处理最后一个片段
        if current_segment_elements:
            segment = self._create_segment(
                segment_index, current_start_time, current_segment_duration,
                current_segment_elements
            )
            segments.append(segment)

        info(f"分片完成，共{len(segments)}个片段")
        return segments

    def _create_segment(self, segment_index: int, start_time: float, duration: float,
                        elements: List[Dict]) -> TimeSegment:
        """创建单个时间片段"""
        segment_id = f"seg_{segment_index:03d}"
        end_time = start_time + duration

        # 生成视觉摘要
        visual_summary = self._generate_visual_summary(elements)

        # 生成连续性锚点
        start_anchor, end_anchor = self._generate_continuity_anchors(elements)

        # 生成镜头建议
        shot_type, lighting, focus_elements = self._generate_shot_suggestions(elements)

        # 连续性要求
        continuity_reqs = self._extract_continuity_requirements(elements)

        return TimeSegment(
            segment_id=segment_id,
            time_range=(round(start_time, 2), round(end_time, 2)),
            duration=round(duration, 2),
            visual_summary=visual_summary,
            contained_elements=elements,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
            continuity_requirements=continuity_reqs,
            shot_type_suggestion=shot_type,
            lighting_suggestion=lighting,
            focus_elements=focus_elements
        )

    def _generate_visual_summary(self, elements: List[Dict]) -> str:
        """生成视觉内容摘要"""
        if not elements:
            return "空片段"

        summaries = []
        element_count = {}

        for elem in elements:
            element_type = elem["element_type"]
            element_count[element_type] = element_count.get(element_type, 0) + 1

        # 构建摘要
        if element_count.get("scene", 0) > 0:
            summaries.append("场景")
        if element_count.get("dialogue", 0) > 0:
            summaries.append("对话")
        if element_count.get("silence", 0) > 0:
            summaries.append("沉默")
        if element_count.get("action", 0) > 0:
            summaries.append("动作")

        # 添加关键元素信息
        key_elements = []
        for elem in elements[:3]:
            elem_id = elem["element_id"]
            if elem_id in self.estimations:
                est = self.estimations[elem_id]
                if est.emotional_weight > 1.5:
                    key_elements.append("情感时刻")
                    break
                elif est.visual_complexity > 1.5:
                    key_elements.append("复杂视觉")
                    break

        if key_elements:
            summaries.append(f"({', '.join(set(key_elements))})")

        return " | ".join(summaries)

    def _generate_continuity_anchors(self, elements: List[Dict]) -> Tuple[Dict, Dict]:
        """生成连续性锚点"""
        start_anchor = {
            "type": "segment_start",
            "constraints": [],
            "character_states": {},
            "prop_states": {}
        }

        end_anchor = {
            "type": "segment_end",
            "constraints": [],
            "character_states": {},
            "prop_states": {},
            "transition_hints": []
        }

        # 收集所有元素的状态变化
        all_character_states = {}
        all_prop_states = {}

        for elem in elements:
            elem_id = elem["element_id"]
            if elem_id in self.estimations:
                est = self.estimations[elem_id]
                all_character_states.update(est.character_states)
                all_prop_states.update(est.prop_states)

        # 设置开始状态
        if elements:
            first_elem_id = elements[0]["element_id"]
            if first_elem_id in self.estimations:
                first_est = self.estimations[first_elem_id]
                start_anchor["character_states"].update(first_est.character_states)
                start_anchor["prop_states"].update(first_est.prop_states)

        # 设置结束状态
        if elements:
            end_anchor["character_states"].update(all_character_states)
            end_anchor["prop_states"].update(all_prop_states)

            # 添加过渡提示
            last_elem_id = elements[-1]["element_id"]
            if last_elem_id in self.estimations:
                last_est = self.estimations[last_elem_id]
                if last_est.element_type == ElementType.DIALOGUE:
                    end_anchor["transition_hints"].append("对话反应时间")
                elif last_est.element_type == ElementType.ACTION:
                    end_anchor["transition_hints"].append("动作缓冲")

        # 添加约束
        if all_character_states:
            start_anchor["constraints"].append("角色状态连贯")
            end_anchor["constraints"].append("角色状态自然过渡")

        return start_anchor, end_anchor

    def _generate_shot_suggestions(self, elements: List[Dict]) -> Tuple[str, str, List[str]]:
        """生成镜头建议"""
        shot_type = "medium_shot"
        lighting = "natural"
        focus_elements = []

        if not elements:
            return shot_type, lighting, focus_elements

        # 分析元素类型
        element_types = [elem["element_type"] for elem in elements]

        # 确定镜头类型
        if "silence" in element_types:
            shot_type = "close_up"
            lighting = "soft_dramatic"
        elif "dialogue" in element_types:
            shot_type = "medium_close_up"
            lighting = "natural_with_emphasis"
        elif "action" in element_types:
            shot_type = "medium_shot"
            lighting = "dynamic"

        # 确定焦点元素
        for elem in elements:
            if elem["element_type"] == "dialogue":
                focus_elements.append("speaker_face")
            elif elem["element_type"] == "silence":
                focus_elements.append("emotional_expression")

        # 去重并限制数量
        focus_elements = list(set(focus_elements))[:2]

        return shot_type, lighting, focus_elements

    def _extract_continuity_requirements(self, elements: List[Dict]) -> List[str]:
        """提取连续性要求"""
        requirements = []

        # 检查是否有状态变化
        has_state_changes = False
        for elem in elements:
            elem_id = elem["element_id"]
            if elem_id in self.estimations:
                est = self.estimations[elem_id]
                if est.character_states or est.prop_states:
                    has_state_changes = True
                    break

        if has_state_changes:
            requirements.append("状态变化连贯")

        # 对话相关要求
        if any(elem["element_type"] == "dialogue" for elem in elements):
            requirements.append("对话情绪自然")

        return requirements

    def analyze_pacing(self, segments: List[TimeSegment]) -> PacingAnalysis | None:
        """分析整体节奏"""
        if not segments:
            return None

        debug("开始节奏分析...")

        intensities = []
        emotional_points = []

        # 计算每个片段的节奏强度
        for i, segment in enumerate(segments):
            intensity = self._calculate_segment_intensity(segment)
            intensities.append(intensity)

            # 检测情感高点
            if intensity > 2.0:
                emotional_points.append({
                    "segment_id": segment.segment_id,
                    "time": segment.time_range[0],
                    "intensity": intensity
                })

        # 分析节奏模式
        pacing_profile = self._classify_pacing_profile(intensities)

        # 找到关键转折点
        turning_points = self._find_turning_points(intensities, segments)

        # 计算统计信息
        total_duration = segments[-1].time_range[1] if segments else 0
        avg_intensity = sum(intensities) / len(intensities) if intensities else 0

        # 确定主导情感
        # dominant_emotion = self._determine_dominant_emotion()

        # 确定视觉风格
        # global_style = self._determine_visual_style(segments)

        debug(f"节奏分析完成: {pacing_profile}")

        return PacingAnalysis(
            pacing_profile=pacing_profile,
            intensity_curve=intensities,
            emotional_arc=emotional_points,
            visual_complexity=[],
            key_points=turning_points,
            statistics={
                "total_duration": round(total_duration, 2),
                "segments_count": len(segments),
                "avg_intensity": round(avg_intensity, 2),
                "max_intensity": round(max(intensities), 2) if intensities else 0,
                "min_intensity": round(min(intensities), 2) if intensities else 0
            },
            recommendations=self._generate_pacing_recommendations(pacing_profile, intensities),
            analysis_notes="基于规则的节奏分析完成"
        )

    def _calculate_segment_intensity(self, segment: TimeSegment) -> float:
        """计算片段节奏强度"""
        intensity = 0.0

        for elem in segment.contained_elements:
            elem_id = elem["element_id"]
            if elem_id in self.estimations:
                est = self.estimations[elem_id]

                # 不同类型元素贡献不同的强度
                if est.element_type == ElementType.SCENE:
                    intensity += 0.5 * est.visual_complexity
                elif est.element_type == ElementType.DIALOGUE:
                    intensity += 0.8 * est.emotional_weight
                elif est.element_type == ElementType.SILENCE:
                    intensity += 1.2 * est.emotional_weight
                elif est.element_type == ElementType.ACTION:
                    intensity += 1.0 * est.visual_complexity

        # 归一化到0-3范围
        return min(intensity, 3.0)

    def _classify_pacing_profile(self, intensities: List[float]) -> str:
        """分类节奏模式"""
        if len(intensities) < 2:
            return "平稳"

        # 计算波动性
        avg_intensity = sum(intensities) / len(intensities)
        variance = sum((i - avg_intensity) ** 2 for i in intensities) / len(intensities)

        if max(intensities) > 2.5 and variance > 0.8:
            return "起伏激烈"
        elif variance > 0.4:
            return "起伏适中"
        elif max(intensities) > 2.0:
            return "渐强累积"
        else:
            return "平稳舒缓"

    def _find_turning_points(self, intensities: List[float],
                             segments: List[TimeSegment]) -> List[Dict]:
        """找到节奏转折点"""
        turning_points = []

        for i in range(1, len(intensities) - 1):
            prev, curr, next_ = intensities[i - 1], intensities[i], intensities[i + 1]

            # 峰值点
            if curr > prev and curr > next_ and curr > 1.5:
                turning_points.append({
                    "type": "peak",
                    "segment_id": segments[i].segment_id,
                    "time": segments[i].time_range[0],
                    "intensity": round(curr, 2),
                    "description": "节奏高潮点"
                })

            # 谷值点
            elif curr < prev and curr < next_ and curr < 1.0:
                turning_points.append({
                    "type": "valley",
                    "segment_id": segments[i].segment_id,
                    "time": segments[i].time_range[0],
                    "intensity": round(curr, 2),
                    "description": "节奏低点"
                })

        return turning_points

    def _determine_dominant_emotion(self) -> str:
        """确定主导情感"""
        emotion_scores = {}

        for est in self.estimations.values():
            emotion_level = est.emotional_weight

            if emotion_level > 1.8:
                emotion = "强烈情感"
            elif emotion_level > 1.5:
                emotion = "中等情感"
            else:
                emotion = "平稳"

            emotion_scores[emotion] = emotion_scores.get(emotion, 0) + 1

        if emotion_scores:
            dominant = max(emotion_scores.items(), key=lambda x: x[1])
            return dominant[0]

        return "平稳"

    def _determine_visual_style(self, segments: List[TimeSegment]) -> str:
        """确定视觉风格"""
        styles = []

        for segment in segments:
            for elem in segment.contained_elements:
                elem_id = elem["element_id"]
                if elem_id in self.estimations:
                    est = self.estimations[elem_id]

                    # 基于情感权重推断风格
                    if est.emotional_weight > 1.5:
                        styles.append("情感化")
                    elif est.visual_complexity > 1.5:
                        styles.append("视觉复杂")
                    else:
                        styles.append("写实")

        # 统计最常见的风格
        if styles:
            style_counts = {}
            for style in styles:
                style_counts[style] = style_counts.get(style, 0) + 1

            dominant = max(style_counts.items(), key=lambda x: x[1])
            return dominant[0]

        return "写实"

    def _generate_pacing_recommendations(self, profile: str, intensities: List[float]) -> List[str]:
        """生成节奏建议"""
        recommendations = []

        if profile == "起伏激烈":
            recommendations.append("保持强烈的节奏对比")
            recommendations.append("高潮点使用快速剪辑")
        elif profile == "起伏适中":
            recommendations.append("节奏自然流畅")
            recommendations.append("注意高潮点的视觉强调")
        elif profile == "渐强累积":
            recommendations.append("节奏逐渐加强")
            recommendations.append("注意最终高潮的冲击力")
        elif profile == "平稳舒缓":
            recommendations.append("节奏平稳")
            recommendations.append("适合情感细腻的场景")

        return recommendations

    def generate_continuity_anchors(self, segments: List[TimeSegment]) -> List[ContinuityAnchor]:
        """生成连续性锚点"""
        anchors = []

        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]

            # 角色状态连续性锚点
            char_anchor = ContinuityAnchor(
                anchor_id=f"char_anchor_{current.segment_id}_to_{next_seg.segment_id}",
                type="character_continuity",
                from_segment=current.segment_id,
                to_segment=next_seg.segment_id,
                time_point=current.time_range[1],
                requirements={},
                priority=8,
                description="角色状态必须连贯"
            )
            anchors.append(char_anchor)

            # 环境一致性锚点
            env_anchor = ContinuityAnchor(
                anchor_id=f"env_anchor_{current.segment_id}_to_{next_seg.segment_id}",
                type="environment_consistency",
                from_segment=current.segment_id,
                to_segment=next_seg.segment_id,
                time_point=current.time_range[1],
                requirements={},
                priority=5,
                description="环境必须一致"
            )
            anchors.append(env_anchor)

        return anchors