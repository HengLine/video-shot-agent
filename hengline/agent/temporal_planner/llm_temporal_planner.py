# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: LLM + 规则约束实现的时序规划（负责将剧本按5秒粒度切分，估算动作时长，使用LangChain实现状态记忆）
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import hashlib
import json
from typing import List, Dict, Any

from hengline.agent.script_parser.script_parser_model import UnifiedScript, Scene
from hengline.agent.temporal_planner.base_temporal_planner import TemporalPlanner
from hengline.agent.temporal_planner.temporal_planner_model import TimelinePlan, DurationEstimation
from hengline.logger import debug, error
from hengline.prompts.prompts_manager import prompt_manager


class LLMTemporalPlanner(TemporalPlanner):
    """
        # 提示词模板
            你是一个专业的分镜师。请将以下动作序列拆分成若干个镜头，每个镜头约{max_seconds}秒。
                ## 规则：
                1. 每个镜头必须包含1-3个连续动作
                2. 情感转折点（如震惊）必须作为新镜头的开始
                3. 对话通常与其前后的反应拆分开
                4. 总时长尽量接近但不超过{max_seconds}秒
                5. 保持叙事的流畅性

                ## 动作序列：
                {actions_json}

                ## 输出格式：
                返回JSON数组，每个元素是一个镜头对象：
                {{
                  "shot_id": 数字,
                  "included_actions": [动作ID列表],
                  "estimated_duration": 数字（秒）,
                  "rationale": "合并理由，如'展现从犹豫到决定的完整过程'"
                }}
    """

    def __init__(self, llm_client):
        """初始化时序规划智能体"""
        super().__init__()
        self.llm = llm_client
        # 缓存系统，避免重复调用相同的场景
        self.cache = {}

        # 统计信息
        self.stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "successful_adjustments": 0,
            "failed_adjustments": 0
        }

    def plan_timeline(self, structured_script: UnifiedScript) -> TimelinePlan | None:
        """
        规划剧本的时序分段
        
        Args:
            structured_script: 结构化的剧本

        Returns:
            分段计划列表
        """

    def adjust(self, estimations: Dict[str, DurationEstimation],
               script_input: UnifiedScript) -> Dict[str, DurationEstimation]:
        """
        主调整方法

        策略：
        1. 识别需要AI调整的复杂场景
        2. 为复杂场景构建上下文
        3. 调用LLM进行时长调整
        4. 有限度地应用调整结果
        """
        self.stats["total_calls"] += 1

        # 1. 识别需要调整的复杂场景
        complex_scenes = self._identify_complex_scenes(script_input, estimations)

        if not complex_scenes:
            print("没有检测到需要AI调整的复杂场景，使用规则估算")
            return estimations

        # 2. 为每个复杂场景构建调整上下文
        adjustment_contexts = []
        for scene in complex_scenes:
            context = self._build_adjustment_context(scene, script_input, estimations)
            adjustment_contexts.append(context)

        # 3. 批量调用AI（减少API调用次数）
        adjusted_results = self._batch_adjust_with_ai(adjustment_contexts)

        # 4. 应用调整（有限度地）
        final_estimations = estimations.copy()
        for scene_id, adjustments in adjusted_results.items():
            final_estimations = self._apply_adjustments(
                final_estimations,
                adjustments,
                self.config.max_adjustment_ratio
            )

        self.stats["successful_adjustments"] += 1

        return final_estimations

    def _identify_complex_scenes(self, script_input: UnifiedScript,
                                 estimations: Dict[str, DurationEstimation]) -> List[Scene]:
        """
        识别需要AI调整的复杂场景

        复杂度评估标准：
        1. 情感强度高
        2. 多角色交互复杂
        3. 动作序列复杂
        4. 对话内容复杂
        5. 规则估算置信度低
        """
        complex_scenes = []

        for scene in script_input.scenes:
            complexity_score = self._calculate_scene_complexity(scene, script_input, estimations)

            if complexity_score >= self.config.complexity_threshold:
                complex_scenes.append(scene)
                print(f"场景 {scene.scene_id} 被识别为复杂场景，复杂度分数: {complexity_score:.2f}")

        return complex_scenes

    def _calculate_scene_complexity(self, scene: Scene,
                                    script_input: UnifiedScript,
                                    estimations: Dict[str, DurationEstimation]) -> float:
        """
        计算场景复杂度分数（0-1）
        """
        scores = []
        weights = self.config.complexity_weights

        # 1. 情感复杂度
        emotional_score = self._calculate_emotional_complexity(scene, script_input)
        scores.append(emotional_score * weights["emotional"])

        # 2. 角色交互复杂度
        interaction_score = self._calculate_interaction_complexity(scene, script_input)
        scores.append(interaction_score * weights["interaction"])

        # 3. 动作复杂度
        action_score = self._calculate_action_complexity(scene, script_input)
        scores.append(action_score * weights["action"])

        # 4. 对话复杂度
        dialogue_score = self._calculate_dialogue_complexity(scene, script_input)
        scores.append(dialogue_score * weights["dialogue"])

        # 5. 规则估算置信度（置信度越低，越需要AI调整）
        confidence_score = self._calculate_confidence_score(scene, estimations)
        scores.append(confidence_score * weights["confidence"])

        # 加权平均
        total_score = sum(scores)

        return min(1.0, total_score)  # 限制在0-1范围内

    def _calculate_emotional_complexity(self, scene: Scene,
                                        script_input: UnifiedScript) -> float:
        """计算情感复杂度"""
        # 收集场景中的情绪
        emotions = []

        # 从对话中提取情绪
        scene_dialogues = [d for d in script_input.dialogues if d.scene_ref == scene.scene_id]
        for dialogue in scene_dialogues:
            if dialogue.emotion and dialogue.emotion != "平静":
                emotions.append(dialogue.emotion)

        if not emotions:
            return 0.0

        # 情绪多样性
        unique_emotions = set(emotions)
        diversity_score = min(1.0, len(unique_emotions) / 5)  # 最多5种情绪

        # 情绪强度
        emotion_intensity = {
            "平静": 1,
            "疑问": 2,
            "喜悦": 3,
            "惊讶": 4,
            "悲伤": 5,
            "愤怒": 6,
            "恐惧": 7,
            "大喊": 8
        }

        max_intensity = max(emotion_intensity.get(e, 1) for e in emotions)
        intensity_score = min(1.0, max_intensity / 8)

        # 情绪变化（如果场景中有情绪转变）
        change_score = 0.0
        if len(emotions) >= 2:
            # 检查情绪是否有明显变化
            first_emotion = emotions[0]
            last_emotion = emotions[-1]
            if first_emotion != last_emotion:
                change_score = 0.7
            # 检查中间是否有变化
            for i in range(1, len(emotions)):
                if emotions[i] != emotions[i - 1]:
                    change_score = 1.0
                    break

        return (diversity_score * 0.3 + intensity_score * 0.4 + change_score * 0.3)

    def _calculate_interaction_complexity(self, scene: Scene,
                                          script_input: UnifiedScript) -> float:
        """计算角色交互复杂度"""
        if not scene.character_refs:
            return 0.0

        # 角色数量
        character_count = len(scene.character_refs)
        count_score = min(1.0, character_count / 5)  # 最多5个角色

        # 对话轮次
        scene_dialogues = [d for d in script_input.dialogues if d.scene_ref == scene.scene_id]
        dialogue_turns = len(scene_dialogues)
        turn_score = min(1.0, dialogue_turns / 10)  # 最多10轮对话

        # 角色切换频率
        if len(scene_dialogues) >= 2:
            speaker_changes = 0
            for i in range(1, len(scene_dialogues)):
                if scene_dialogues[i].speaker != scene_dialogues[i - 1].speaker:
                    speaker_changes += 1
            change_score = min(1.0, speaker_changes / len(scene_dialogues))
        else:
            change_score = 0.0

        return (count_score * 0.4 + turn_score * 0.3 + change_score * 0.3)

    def _build_adjustment_context(self, scene: Scene,
                                  script_input: UnifiedScript,
                                  estimations: Dict[str, DurationEstimation]) -> Dict:
        """
        为单个场景构建调整上下文
        """
        # 收集场景相关信息
        scene_dialogues = [d for d in script_input.dialogues if d.scene_ref == scene.scene_id]
        scene_actions = [a for a in script_input.actions if a.scene_ref == scene.scene_id]
        scene_characters = [c for c in script_input.characters if c.name in scene.character_refs]

        # 构建对话详情
        dialogue_details = []
        for dialogue in scene_dialogues:
            est_key = dialogue.dialogue_id
            original_est = estimations.get(est_key, DurationEstimation(estimated_duration=3.0))

            dialogue_details.append({
                "dialogue_id": dialogue.dialogue_id,
                "speaker": dialogue.speaker,
                "text": dialogue.text,
                "emotion": dialogue.emotion,
                "original_estimate": original_est.estimated_duration,
                "word_count": len(dialogue.text),
                "speaker_age": next((c.age for c in scene_characters if c.name == dialogue.speaker), None)
            })

        # 构建动作详情
        action_details = []
        for action in scene_actions:
            est_key = action.action_id
            original_est = estimations.get(est_key, DurationEstimation(estimated_duration=2.0))

            action_details.append({
                "action_id": action.action_id,
                "type": action.type,
                "actor": action.actor,
                "description": action.description,
                "intensity": action.intensity,
                "original_estimate": original_est.estimated_duration
            })

        # 构建上下文
        context = {
            "scene_id": scene.scene_id,
            "scene_description": scene.description,
            "location": scene.location,
            "time_of_day": scene.time_of_day,
            "mood": scene.mood,
            "characters": scene_characters,
            "dialogues": dialogue_details,
            "actions": action_details,
            "complexity_analysis": {
                "emotional_score": self._calculate_emotional_complexity(scene, script_input),
                "interaction_score": self._calculate_interaction_complexity(scene, script_input),
                "action_score": self._calculate_action_complexity(scene, script_input),
                "dialogue_score": self._calculate_dialogue_complexity(scene, script_input)
            }
        }

        return context

        def _create_adjustment_prompt(self, context: Dict) -> str:
            """
            创建AI调整提示词

            重点：让AI理解影视节奏和时长分配的艺术
            """
            return f"""
    你是一个专业的影视剪辑师和节奏大师。请分析以下场景，调整时间分配。

    ## 场景信息：
    - 场景ID: {context['scene_id']}
    - 地点: {context['location']}
    - 时间: {context['time_of_day']}
    - 氛围: {context['mood']}
    - 角色: {', '.join([c['name'] for c in context['characters']])}

    ## 场景描述：
    {context['scene_description'][:500]}

    ## 对话内容（需要调整时长）：
    {self._format_dialogues_for_prompt(context['dialogues'])}

    ## 动作序列（需要调整时长）：
    {self._format_actions_for_prompt(context['actions'])}

    ## 复杂度分析：
    {json.dumps(context['complexity_analysis'], indent=2, ensure_ascii=False)}

    ## 调整原则：
    1. **情绪表达完整性**：给足情绪表达的时间（悲伤需要时间酝酿，愤怒需要时间爆发）
    2. **角色特征尊重**：老人动作慢，年轻人语速快，儿童反应时间短
    3. **节奏感控制**：紧张场景节奏快，抒情场景节奏慢
    4. **视觉叙事需求**：重要动作要给足展示时间，关键表情要给特写时间
    5. **对话自然度**：给足思考和反应时间，避免对话过于急促

    ## 具体考虑因素：
    - 对话中的情绪强度：{self._extract_emotion_intensity(context['dialogues'])}
    - 动作的复杂程度：{self._extract_action_complexity(context['actions'])}
    - 角色年龄分布：{self._extract_age_distribution(context['characters'])}
    - 场景氛围需求：{context['mood']}需要特定的节奏

    ## 调整约束：
    1. 每个元素的调整幅度不超过原始时长的±40%
    2. 场景总时长变化不超过±20%
    3. 保持对话和动作的逻辑顺序
    4. 确保关键情感时刻有足够时间

    ## 输出格式（严格的JSON）：
    {{
      "scene_id": "{context['scene_id']}",
      "adjustments": [
        {{
          "element_id": "元素ID",
          "element_type": "dialogue|action",
          "original_duration": 原时长,
          "adjusted_duration": 调整后时长,
          "adjustment_percentage": 调整百分比,
          "reason": "调整理由（具体说明，如：老人说话需要更多时间表达情感）",
          "confidence": 0-1的置信度,
          "key_moment": "是否为关键时刻（true/false）"
        }}
      ],
      "overall_analysis": {{
        "emotional_arc": "情绪弧线分析",
        "pacing_recommendation": "节奏建议",
        "key_moments": ["关键时刻列表"],
        "total_adjustment_percentage": "总调整百分比"
      }},
      "adjustment_confidence": 0.9
    }}

    ## 特别注意：
    1. 考虑老人的缓慢动作和深思熟虑的对话
    2. 考虑年轻人的快速反应和急促对话
    3. 给情感高潮足够的时间
    4. 保持叙事的流畅性和连贯性
    """

        def _format_dialogues_for_prompt(self, dialogues: List[Dict]) -> str:
            """格式化对话信息用于提示词"""
            if not dialogues:
                return "无对话内容"

            formatted = []
            for d in dialogues:
                speaker_info = f"{d['speaker']}"
                if d.get('speaker_age'):
                    speaker_info += f"({d['speaker_age']}岁)"

                formatted.append(
                    f"- {speaker_info} ({d['emotion']}): \"{d['text'][:100]}\" "
                    f"[原估算: {d['original_estimate']:.1f}秒, {d['word_count']}字]"
                )

            return "\n".join(formatted)

        def _format_actions_for_prompt(self, actions: List[Dict]) -> str:
            """格式化动作信息用于提示词"""
            if not actions:
                return "无动作内容"

            formatted = []
            for a in actions:
                intensity_desc = {1: "轻微", 2: "一般", 3: "较强", 4: "强烈", 5: "剧烈"}.get(a['intensity'], "一般")
                formatted.append(
                    f"- {a['actor']}的{a['type']}动作（强度: {intensity_desc}）: "
                    f"\"{a['description'][:80]}\" [原估算: {a['original_estimate']:.1f}秒]"
                )

            return "\n".join(formatted)

        def _batch_adjust_with_ai(self, contexts: List[Dict]) -> Dict[str, List[Dict]]:
            """
            批量调用AI进行时长调整

            策略：将多个场景合并到一个请求中，减少API调用次数
            """
            if not contexts:
                return {}

            # 检查缓存
            cache_key = self._generate_cache_key(contexts)
            if cache_key in self.cache and not self.config.force_refresh:
                self.stats["cache_hits"] += 1
                print(f"使用缓存结果，缓存命中: {self.stats['cache_hits']}")
                return self.cache[cache_key]

            # 根据场景数量决定是否分批
            if len(contexts) > self.config.max_scenes_per_request:
                # 分批处理
                batches = [contexts[i:i + self.config.max_scenes_per_request]
                           for i in range(0, len(contexts), self.config.max_scenes_per_request)]
                all_results = {}

                for batch in batches:
                    batch_results = self._process_batch(batch)
                    all_results.update(batch_results)

                # 缓存结果
                self.cache[cache_key] = all_results
                return all_results
            else:
                # 单批处理
                results = self._process_batch(contexts)
                self.cache[cache_key] = results
                return results

        def _process_batch(self, contexts: List[Dict]) -> Dict[str, List[Dict]]:
            """处理一批场景的调整"""
            try:
                # 构建批量提示词
                batch_prompt = self._create_batch_prompt(contexts)

                # 调用LLM
                response = self._call_llm_for_adjustment(batch_prompt)

                # 解析响应
                adjustments_by_scene = self._parse_batch_response(response, contexts)

                return adjustments_by_scene

            except Exception as e:
                print(f"AI批量调整失败: {e}")
                # 返回空调整，使用原始估算
                return {ctx["scene_id"]: [] for ctx in contexts}

        def _create_batch_prompt(self, contexts: List[Dict]) -> str:
            """创建批量调整提示词"""
            scene_summaries = []

            for i, context in enumerate(contexts):
                scene_summary = f"""
    ## 场景 {i + 1}: {context['scene_id']}
    地点: {context['location']}, 时间: {context['time_of_day']}, 氛围: {context['mood']}
    角色: {', '.join([c['name'] for c in context['characters']])}

    对话数量: {len(context['dialogues'])}
    动作数量: {len(context['actions'])}
    情感强度: {context['complexity_analysis']['emotional_score']:.2f}
                """
                scene_summaries.append(scene_summary)

            return f"""
    你是一个专业的影视剪辑师。请为以下多个场景进行时长调整。

    {''.join(scene_summaries)}

    ## 调整要求（适用于所有场景）：
    1. 尊重角色特征（年龄、性格）
    2. 保持情绪表达的完整性
    3. 确保节奏适合场景氛围
    4. 每个元素调整幅度不超过±40%
    5. 每个场景总时长变化不超过±20%

    ## 输出格式：
    {{
      "scenes_adjustments": [
        {{
          "scene_id": "场景ID",
          "adjustments": [
            {{
              "element_id": "元素ID",
              "element_type": "dialogue|action",
              "original_duration": 原时长,
              "adjusted_duration": 调整后时长,
              "reason": "调整理由",
              "confidence": 置信度
            }}
          ]
        }}
      ],
      "cross_scene_analysis": {{
        "pacing_consistency": "节奏一致性分析",
        "emotional_progression": "情绪进展分析",
        "recommendations": ["跨场景建议"]
      }}
    }}
    """

    def _call_llm_for_adjustment(self, prompt: str) -> str:
        """
        调用LLM进行时长调整
        """
        try:
            response = self.llm.chat_complete(
                messages=[
                    {
                        "role": "system",
                        "content": """你是一个专业的影视剪辑师和节奏分析师。你精通：
                        1. 对话节奏控制（语速、停顿、情绪表达）
                        2. 动作时长估算（复杂动作、简单动作、反应时间）
                        3. 角色特征考量（年龄、性格、情绪状态）
                        4. 场景氛围营造（紧张、舒缓、浪漫、悬疑）
                        请输出严格的JSON格式。"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                response_format={"type": "json_object"}
            )
            return response

        except Exception as e:
            error(f"LLM调用失败: {e}")
            raise Exception(f"AI调整服务暂时不可用: {str(e)}")

    def _parse_batch_response(self, response: str, contexts: List[Dict]) -> Dict[str, List[Dict]]:
        """
        解析批量调整响应
        """
        try:
            result = json.loads(response)
            adjustments_by_scene = {}

            # 提取每个场景的调整
            if "scenes_adjustments" in result:
                for scene_adj in result["scenes_adjustments"]:
                    scene_id = scene_adj.get("scene_id")
                    adjustments = scene_adj.get("adjustments", [])

                    # 验证和清理调整数据
                    validated_adjustments = self._validate_adjustments(adjustments, scene_id, contexts)
                    adjustments_by_scene[scene_id] = validated_adjustments

            # 如果没有找到场景调整，尝试其他格式
            elif "adjustments" in result:
                # 可能是单场景格式
                for context in contexts:
                    scene_id = context["scene_id"]
                    adjustments_by_scene[scene_id] = self._validate_adjustments(
                        result.get("adjustments", []), scene_id, contexts
                    )

            # 记录跨场景分析
            if "cross_scene_analysis" in result:
                print("跨场景分析:", result["cross_scene_analysis"])

            return adjustments_by_scene

        except json.JSONDecodeError as e:
            print(f"AI响应JSON解析失败: {e}")
            print(f"响应内容: {response[:500]}...")
            # 返回空调整
            return {ctx["scene_id"]: [] for ctx in contexts}

    def _validate_adjustments(self, adjustments: List[Dict],
                              scene_id: str, contexts: List[Dict]) -> List[Dict]:
        """
        验证和清理调整数据
        """
        validated = []

        # 查找场景上下文
        scene_context = next((ctx for ctx in contexts if ctx["scene_id"] == scene_id), None)
        if not scene_context:
            return []

        for adj in adjustments:
            # 基本验证
            if not all(k in adj for k in ["element_id", "adjusted_duration", "reason"]):
                continue

            element_id = adj["element_id"]
            adjusted_duration = adj["adjusted_duration"]

            # 验证元素类型
            element_type = adj.get("element_type", "")
            if element_type not in ["dialogue", "action"]:
                # 尝试推断类型
                if "dialogue" in element_id:
                    element_type = "dialogue"
                elif "action" in element_id:
                    element_type = "action"
                else:
                    continue

            # 查找原始估算
            original_duration = None
            if element_type == "dialogue":
                dialogue = next((d for d in scene_context["dialogues"]
                                 if d["dialogue_id"] == element_id), None)
                if dialogue:
                    original_duration = dialogue["original_estimate"]
            elif element_type == "action":
                action = next((a for a in scene_context["actions"]
                               if a["action_id"] == element_id), None)
                if action:
                    original_duration = action["original_estimate"]

            if original_duration is None:
                continue

            # 验证调整幅度
            adjustment_ratio = abs(adjusted_duration - original_duration) / original_duration
            if adjustment_ratio > self.config.max_adjustment_ratio:
                print(f"警告：调整幅度过大 ({adjustment_ratio:.1%})，限制为{self.config.max_adjustment_ratio:.0%}")
                # 限制调整幅度
                if adjusted_duration > original_duration:
                    adjusted_duration = original_duration * (1 + self.config.max_adjustment_ratio)
                else:
                    adjusted_duration = original_duration * (1 - self.config.max_adjustment_ratio)

            # 验证时长合理性
            if element_type == "dialogue":
                if adjusted_duration < self.config.min_dialogue_duration:
                    adjusted_duration = self.config.min_dialogue_duration
                elif adjusted_duration > self.config.max_dialogue_duration:
                    adjusted_duration = self.config.max_dialogue_duration
            elif element_type == "action":
                if adjusted_duration < self.config.min_action_duration:
                    adjusted_duration = self.config.min_action_duration
                elif adjusted_duration > self.config.max_action_duration:
                    adjusted_duration = self.config.max_action_duration

            # 构建验证后的调整
            validated_adj = {
                "element_id": element_id,
                "element_type": element_type,
                "original_duration": original_duration,
                "adjusted_duration": round(adjusted_duration, 1),
                "adjustment_percentage": round(((adjusted_duration / original_duration) - 1) * 100, 1),
                "reason": adj.get("reason", "AI优化调整"),
                "confidence": min(1.0, max(0.0, adj.get("confidence", 0.7))),
                "key_moment": adj.get("key_moment", False)
            }

            validated.append(validated_adj)

        return validated

    def _apply_adjustments(self, estimations: Dict[str, DurationEstimation],
                           adjustments_by_scene: Dict[str, List[Dict]],
                           max_ratio: float) -> Dict[str, DurationEstimation]:
        """
        应用AI调整到原始估算

        有限度地应用调整，确保不会过度改变
        """
        adjusted_estimations = estimations.copy()

        for scene_id, adjustments in adjustments_by_scene.items():
            for adjustment in adjustments:
                element_id = adjustment["element_id"]

                if element_id in adjusted_estimations:
                    original = adjusted_estimations[element_id]
                    ai_adjusted = adjustment["adjusted_duration"]

                    # 计算混合权重（基于AI置信度和调整幅度）
                    ai_confidence = adjustment.get("confidence", 0.7)

                    # 调整幅度越大，应用权重越小（避免过度调整）
                    adjustment_ratio = abs(ai_adjusted - original.estimated_duration) / original.estimated_duration
                    amplitude_factor = max(0.3, 1.0 - adjustment_ratio)  # 调整越大，权重越小

                    # 最终权重 = AI置信度 × 幅度因子 × 配置权重
                    ai_weight = ai_confidence * amplitude_factor * self.config.ai_weight

                    # 限制权重范围
                    ai_weight = min(self.config.max_ai_weight, max(self.config.min_ai_weight, ai_weight))

                    # 计算最终时长（AI调整和原始估算的加权平均）
                    final_duration = (
                            original.estimated_duration * (1 - ai_weight) +
                            ai_adjusted * ai_weight
                    )

                    # 确保在合理范围内
                    if original.element_type == "dialogue":
                        final_duration = max(self.config.min_dialogue_duration,
                                             min(self.config.max_dialogue_duration, final_duration))
                    elif original.element_type == "action":
                        final_duration = max(self.config.min_action_duration,
                                             min(self.config.max_action_duration, final_duration))

                    # 更新估算
                    adjusted_estimations[element_id] = DurationEstimation(
                        element_id=original.element_id,
                        element_type=original.element_type,
                        estimated_duration=round(final_duration, 1),
                        confidence=min(1.0, original.confidence * 0.9 + ai_confidence * 0.1),
                        factors_considered=original.factors_considered + [
                            f"AI调整: {adjustment['reason']} (权重: {ai_weight:.2f})"
                        ],
                        adjustment_notes=f"AI调整: {adjustment.get('reason', '优化')} "
                                         f"[原: {original.estimated_duration:.1f}s → "
                                         f"AI建议: {ai_adjusted:.1f}s → "
                                         f"最终: {final_duration:.1f}s]"
                    )

        return adjusted_estimations

    def _generate_cache_key(self, contexts: List[Dict]) -> str:
        """生成缓存键"""
        # 基于场景ID和内容生成简单的哈希
        scene_info = []
        for ctx in contexts:
            scene_info.append(f"{ctx['scene_id']}:{len(ctx['dialogues'])}:{len(ctx['actions'])}")

        content = "|".join(sorted(scene_info))
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _calculate_action_complexity(self, scene: Scene,
                                     script_input: UnifiedScript) -> float:
        """计算动作复杂度"""
        scene_actions = [a for a in script_input.actions if a.scene_ref == scene.scene_id]

        if not scene_actions:
            return 0.0

        # 动作数量
        count_score = min(1.0, len(scene_actions) / 8)

        # 动作强度
        intensity_sum = sum(a.intensity for a in scene_actions)
        avg_intensity = intensity_sum / len(scene_actions)
        intensity_score = min(1.0, avg_intensity / 5)  # 最大强度5

        # 动作类型多样性
        action_types = set(a.type for a in scene_actions)
        diversity_score = min(1.0, len(action_types) / 5)

        # 复杂动作比例
        complex_actions = ["fight", "chase", "fall", "climb", "dance"]
        complex_count = sum(1 for a in scene_actions if a.type in complex_actions)
        complexity_score = min(1.0, complex_count / len(scene_actions))

        return (count_score * 0.2 + intensity_score * 0.3 +
                diversity_score * 0.2 + complexity_score * 0.3)

    def _calculate_dialogue_complexity(self, scene: Scene,
                                       script_input: UnifiedScript) -> float:
        """计算对话复杂度"""
        scene_dialogues = [d for d in script_input.dialogues if d.scene_ref == scene.scene_id]

        if not scene_dialogues:
            return 0.0

        # 对话长度复杂度
        total_words = sum(len(d.text) for d in scene_dialogues)
        avg_words = total_words / len(scene_dialogues)
        length_score = min(1.0, avg_words / 50)  # 平均50字为满分

        # 对话内容复杂度（基于标点符号）
        complex_punctuation = ["？", "！", "…", "——"]
        complex_count = sum(1 for d in scene_dialogues
                            if any(p in d.text for p in complex_punctuation))
        punctuation_score = min(1.0, complex_count / len(scene_dialogues))

        return (length_score * 0.6 + punctuation_score * 0.4)

    def _calculate_confidence_score(self, scene: Scene,
                                    estimations: Dict[str, DurationEstimation]) -> float:
        """计算置信度分数（置信度越低，越需要AI调整）"""
        # 收集场景相关的估算
        scene_estimations = []

        # 查找对话估算
        for key, est in estimations.items():
            if key.startswith("dialogue_"):
                # 简单匹配：对话ID可能包含场景信息
                # 实际实现需要更精确的匹配逻辑
                scene_estimations.append(est)

        if not scene_estimations:
            return 0.5  # 中等置信度需求

        # 计算平均置信度（置信度越低，分数越高）
        avg_confidence = sum(e.confidence for e in scene_estimations) / len(scene_estimations)

        # 转换：置信度越低 -> 分数越高（越需要AI调整）
        confidence_score = 1.0 - avg_confidence

        return confidence_score

    def _extract_emotion_intensity(self, dialogues: List[Dict]) -> str:
        """提取情绪强度描述"""
        if not dialogues:
            return "无情绪内容"

        emotions = [d['emotion'] for d in dialogues if d.get('emotion')]
        if not emotions:
            return "情绪平静"

        # 情绪强度映射
        intensity_map = {
            "平静": 1, "疑问": 2, "喜悦": 3, "惊讶": 4,
            "悲伤": 5, "愤怒": 6, "恐惧": 7, "大喊": 8
        }

        max_emotion = max(emotions, key=lambda e: intensity_map.get(e, 1))
        intensity_value = intensity_map.get(max_emotion, 1)

        if intensity_value >= 6:
            return "高强度情绪"
        elif intensity_value >= 4:
            return "中等强度情绪"
        else:
            return "低强度情绪"

    def _extract_action_complexity(self, actions: List[Dict]) -> str:
        """提取动作复杂度描述"""
        if not actions:
            return "无复杂动作"

        complex_types = ["fight", "chase", "fall", "climb", "dance"]
        complex_count = sum(1 for a in actions if a['type'] in complex_types)

        if complex_count >= 3:
            return "高复杂度动作序列"
        elif complex_count >= 1:
            return "中等复杂度动作"
        else:
            return "简单动作"

    def _extract_age_distribution(self, characters: List[Dict]) -> str:
        """提取年龄分布描述"""
        if not characters:
            return "无年龄信息"

        ages = [c.get('age') for c in characters if c.get('age') is not None]
        if not ages:
            return "年龄未知"

        avg_age = sum(ages) / len(ages)

        if avg_age >= 60:
            return "主要为老年角色"
        elif avg_age >= 30:
            return "主要为中年角色"
        else:
            return "主要为年轻角色"