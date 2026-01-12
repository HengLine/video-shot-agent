"""
@FileName: duration_prompt_templates.py
@Description: 时长估算提示词模板
@Author: HengLine
@Time: 2026/1/12 16:40
"""
import json
from dataclasses import dataclass
from typing import Dict, List

from hengline.prompts.prompts_manager import prompt_manager


@dataclass
class PromptConfig:
    """提示词配置"""
    enable_enhanced_analysis: bool = True
    include_visual_suggestions: bool = True
    include_continuity_hints: bool = True


class PromptComponent:
    """提示词组件基类"""

    @staticmethod
    def format_list(items: List[str], max_items: int = 5) -> str:
        """格式化列表为字符串"""
        if not items:
            return "无"

        displayed = items[:max_items]
        result = ", ".join(displayed)
        if len(items) > max_items:
            result += f" 等{len(items)}个"
        return result

    @staticmethod
    def truncate_text(text: str, max_length: int = 150) -> str:
        """截断文本"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    @staticmethod
    def analyze_punctuation(text: str) -> Dict[str, bool]:
        """分析标点特征"""
        return {
            "has_ellipsis": "..." in text or "……" in text,
            "has_exclamation": "!" in text,
            "has_question": "?" in text,
            "has_dash": "—" in text or "-" in text,
            "has_comma": "," in text
        }

    @staticmethod
    def punctuation_analysis_str(text: str) -> str:
        """生成标点分析字符串"""
        analysis = PromptComponent.analyze_punctuation(text)
        features = []

        if analysis["has_ellipsis"]:
            features.append("省略号（犹豫/未说完）")
        if analysis["has_question"]:
            features.append("问号（疑问）")
        if analysis["has_exclamation"]:
            features.append("感叹号（强调）")
        if analysis["has_dash"]:
            features.append("破折号（转折）")

        if features:
            return "包含：" + "、".join(features)
        return "标准标点"

    @staticmethod
    def extract_emotional_words(text: str, word_list: List[str] = None) -> List[str]:
        """提取情感词汇"""
        if word_list is None:
            word_list = ["爱", "恨", "悲伤", "喜悦", "愤怒", "恐惧",
                         "惊讶", "沉默", "颤抖", "哽咽", "紧张", "放松"]

        found = []
        for word in word_list:
            if word in text:
                found.append(word)
        return found

    @staticmethod
    def assess_sentence_complexity(text: str) -> str:
        """评估句子复杂度"""
        words = text.split()
        if len(words) <= 3:
            return "简单"
        elif len(words) <= 8:
            return "中等"
        elif len(words) <= 15:
            return "复杂"
        else:
            return "非常复杂"


class DurationPromptTemplates:
    """重构的提示词模板管理器"""

    def __init__(self, config: PromptConfig = None):
        self.config = config or PromptConfig()
        self.components = PromptComponent()
        self.prompt_manager = prompt_manager

    def scene_duration_prompt(self, scene_data: Dict, context: Dict = None) -> str:
        """场景时长估算提示词"""
        # 基础提示词
        prompt = prompt_manager.get_name_prompt("temporal_planner")

        # 获取模板
        template = prompt_manager.get_name_prompt("scene_planner")
        if not template:
            return prompt + "请估算以下场景的时长。\n\n" + json.dumps(scene_data, ensure_ascii=False)

        # 准备数据
        scene_id = scene_data.get("scene_id", "N/A")
        location = scene_data.get("location", "未指定")
        time_of_day = scene_data.get("time_of_day", "未指定")
        mood = scene_data.get("mood", "未指定")
        description = scene_data.get("description", "")

        key_visuals = scene_data.get("key_visuals", [])
        key_visuals_str = self.components.format_list(key_visuals)

        # 填充模板
        formatted = template.format(
            scene_id=scene_id,
            location=location,
            time_of_day=time_of_day,
            mood=mood,
            key_visuals_str=key_visuals_str,
            description=description
        )

        prompt += formatted

        # 添加上下文信息
        if context:
            prompt += self._add_context_info(context, "场景")

        # 添加增强分析
        if self.config.enable_enhanced_analysis:
            prompt += "\n\n" + """
            ### 视觉连续性考虑
                1. 哪些视觉元素需要在后续片段中保持一致？
                2. 这个场景结束时，角色和环境应该处于什么状态？
                3. 为下一个片段需要预留什么视觉锚点？
            """

        # 添加额外指令
        if self.config.include_visual_suggestions:
            prompt += "\n\n" + """
            ### 输出要求
                请在visual_hints中包含以下信息（如果适用）：
                - suggested_shot_types: 建议的镜头类型数组
                - lighting_notes: 灯光建议
                - focus_transitions: 焦点转移顺序
                - continuity_requirements: 连续性要求数组
            """

        return prompt

    def dialogue_duration_prompt(self, dialogue_data: Dict, context: Dict = None) -> str:
        """对话时长估算提示词"""
        # 检查是否为沉默
        if dialogue_data.get("type") == "silence" or not dialogue_data.get("content", "").strip():
            return self.silence_duration_prompt(dialogue_data, context)

        # 基础提示词
        prompt = prompt_manager.get_name_prompt("temporal_planner")

        # 获取模板
        template = prompt_manager.get_name_prompt("dialogue_planner")
        if not template:
            return prompt + "请估算以下对话的时长。\n\n" + json.dumps(dialogue_data, ensure_ascii=False)

        # 准备数据
        dialogue_id = dialogue_data.get("dialogue_id", "N/A")
        speaker = dialogue_data.get("speaker", "未指定")
        emotion = dialogue_data.get("emotion", "未指定")
        voice_quality = dialogue_data.get("voice_quality", "未指定")
        parenthetical = dialogue_data.get("parenthetical", "无")
        dialogue_type = dialogue_data.get("type", "speech")
        content = dialogue_data.get("content", "")

        # 分析对话特征
        word_count = len(content.split())
        punctuation_analysis = self.components.punctuation_analysis_str(content)
        emotional_words = self.components.extract_emotional_words(content)
        emotional_words_str = self.components.format_list(emotional_words)
        sentence_complexity = self.components.assess_sentence_complexity(content)

        # 填充基础模板
        formatted = template.format(
            dialogue_id=dialogue_id,
            speaker=speaker,
            emotion=emotion,
            voice_quality=voice_quality,
            parenthetical=parenthetical,
            dialogue_type=dialogue_type,
            content=content,

            # 添加语言学分析
            word_count=word_count,
            punctuation_analysis=punctuation_analysis,
            emotional_words=emotional_words_str,
            sentence_complexity=sentence_complexity
        )

        prompt += formatted

        # 添加上下文信息
        if context:
            prompt += self._add_context_info(context, "对话")

        # 添加情感分析组件
        if self.config.enable_enhanced_analysis:
            emotional_indicators = prompt_manager.get_name_prompt("emotional_indicators_planner")
            if emotional_indicators:
                prompt += "\n\n### 情感分析参考\n" + emotional_indicators

        return prompt

    def silence_duration_prompt(self, dialogue_data: Dict, context: Dict = None) -> str:
        """沉默时长估算提示词"""
        prompt = prompt_manager.get_name_prompt("temporal_planner")

        # 获取沉默模板
        template = prompt_manager.get_name_prompt("silence_planner")
        if not template:
            template = "## 沉默时长估算\n\n请估算以下沉默时刻的合理时长。\n\n{silence_info}"

        # 准备数据
        dialogue_id = dialogue_data.get("dialogue_id", "N/A")
        emotion = dialogue_data.get("emotion", "未指定")
        parenthetical = dialogue_data.get("parenthetical", "无动作描述")

        # 填充模板
        formatted = template.format(
            dialogue_id=dialogue_id,
            emotion=emotion,
            parenthetical=parenthetical
        )

        prompt += formatted

        # 添加视觉建议（沉默通常需要特定的镜头）
        if self.config.include_visual_suggestions:
            shot_suggestions = prompt_manager.get_name_prompt("shot_type_suggestions_planner")
            if shot_suggestions:
                prompt += "\n\n### 视觉表现建议\n适合沉默的镜头类型：\n" + shot_suggestions

        # 添加上下文信息
        if context:
            prompt += self._add_context_info(context, "沉默")

        return prompt

    def action_duration_prompt(self, action_data: Dict, context: Dict = None) -> str:
        """动作时长估算提示词"""
        prompt = prompt_manager.get_name_prompt("temporal_planner")

        # 获取沉默模板
        template = prompt_manager.get_name_prompt("action_planner")
        if not template:
            return prompt + "请估算以下动作的时长。\n\n" + json.dumps(action_data, ensure_ascii=False)

        # 准备数据
        action_id = action_data.get("action_id", "N/A")
        actor = action_data.get("actor", "未指定")
        action_type = action_data.get("type", "未指定")
        target = action_data.get("target", "无")
        description = action_data.get("description", "")

        # 分析动作复杂度
        component_count = self._count_action_components(description)
        fineness_level = self._assess_action_fineness(description)
        movement_range = self._assess_movement_range(description)

        # 填充基础模板
        formatted = template.format(
            action_id=action_id,
            actor=actor,
            action_type=action_type,
            target=target,
            description=description,
            # 添加复杂度评估
            component_count=component_count,
            fineness_level=fineness_level,
            movement_range=movement_range
        )

        prompt += formatted

        # 添加上下文信息
        if context:
            prompt += self._add_context_info(context, "动作")

        # 添加连续性提示
        if self.config.include_continuity_hints:
            prompt += "\n\n### 连续性考虑\n"
            prompt += "请考虑：\n"
            prompt += "1. 这个动作开始前，角色/道具处于什么状态？\n"
            prompt += "2. 动作完成后，状态如何变化？\n"
            prompt += "3. 需要为下一个动作预留什么过渡时间？\n"

        return prompt

    def batch_scene_prompt(self, scenes: List[Dict], context: Dict = None) -> str:
        """批量场景估算提示词"""
        # 获取模板
        prompt = prompt_manager.get_name_prompt("temporal_planner")
        template = prompt_manager.get_name_prompt("scene_batch_planner")

        if not template:
            return prompt + f"请估算以下{len(scenes)}个场景的时长。\n\n" + json.dumps(scenes[:3], ensure_ascii=False)

        # 准备场景摘要
        scenes_summary = []
        for i, scene in enumerate(scenes[:10]):  # 限制数量
            scene_id = scene.get("scene_id", f"scene_{i}")
            location = scene.get("location", "未指定")
            mood = scene.get("mood", "未指定")
            desc_preview = self.components.truncate_text(scene.get("description", ""), 80)

            scenes_summary.append({
                "scene_id": scene_id,
                "location": location,
                "mood": mood,
                "description_preview": desc_preview
            })

        scenes_summary_str = json.dumps(scenes_summary, ensure_ascii=False, indent=2)

        # 填充模板
        formatted = template.format(
            count=len(scenes),
            scenes_summary=scenes_summary_str
        )

        prompt += formatted

        return prompt

    def batch_dialogue_prompt(self, dialogues: List[Dict], context: Dict = None) -> str:
        """批量对话估算提示词"""
        prompt = prompt_manager.get_name_prompt("temporal_planner")
        template = prompt_manager.get_name_prompt("dialogue_batch_planner")
        if not template:
            return prompt + f"请估算以下{len(dialogues)}个对话的时长。\n\n" + json.dumps(dialogues[:5], ensure_ascii=False)

        # 准备对话摘要
        dialogues_summary = []
        for i, dialogue in enumerate(dialogues[:15]):  # 限制数量
            dialogue_id = dialogue.get("dialogue_id", f"dialogue_{i}")
            speaker = dialogue.get("speaker", "未指定")
            emotion = dialogue.get("emotion", "未指定")
            content = dialogue.get("content", "")
            content_preview = self.components.truncate_text(content, 50)
            is_silence = dialogue.get("type") == "silence" or not content.strip()

            dialogues_summary.append({
                "dialogue_id": dialogue_id,
                "speaker": speaker,
                "emotion": emotion,
                "content_preview": content_preview,
                "is_silence": is_silence
            })

        dialogues_summary_str = json.dumps(dialogues_summary, ensure_ascii=False, indent=2)

        # 填充模板
        formatted = template.format(
            count=len(dialogues),
            dialogues_summary=dialogues_summary_str
        )

        prompt += formatted

        return prompt

    def batch_action_prompt(self, actions: List[Dict], context: Dict = None) -> str:
        """批量动作估算提示词"""
        prompt = prompt_manager.get_name_prompt("temporal_planner")
        template = prompt_manager.get_name_prompt("action_batch_planner")
        if not template:
            return prompt + f"请估算以下{len(actions)}个动作的时长。\n\n" + json.dumps(actions[:5], ensure_ascii=False)

        # 准备动作摘要
        actions_summary = []
        for i, action in enumerate(actions[:20]):  # 限制数量
            action_id = action.get("action_id", f"action_{i}")
            actor = action.get("actor", "未指定")
            action_type = action.get("type", "未指定")
            description = action.get("description", "")
            desc_preview = self.components.truncate_text(description, 60)

            actions_summary.append({
                "action_id": action_id,
                "actor": actor,
                "action_type": action_type,
                "description_preview": desc_preview
            })

        actions_summary_str = json.dumps(actions_summary, ensure_ascii=False, indent=2)

        # 填充模板
        formatted = template.format(
            count=len(actions),
            actions_summary=actions_summary_str
        )

        prompt += formatted

        return prompt

    def _add_context_info(self, context: Dict, element_type: str) -> str:
        """添加上下文信息"""
        if not context:
            return ""

        context_str = "\n\n### 上下文信息\n"

        if "previous_elements" in context:
            prev_count = len(context.get("previous_elements", {}))
            if prev_count > 0:
                context_str += f"- 前序{prev_count}个{element_type}元素已估算\n"

        if "position_in_sequence" in context:
            pos = context.get("position_in_sequence", 0)
            total = context.get("total_elements", 1)
            context_str += f"- 在序列中的位置：{pos + 1}/{total}\n"

        if "overall_pacing" in context:
            pacing = context.get("overall_pacing", "未指定")
            context_str += f"- 整体节奏：{pacing}\n"

        if "key_emotional_state" in context:
            emotion = context.get("key_emotional_state", {})
            if emotion:
                context_str += f"- 当前情感状态：{emotion.get('emotion', '未指定')}（强度{emotion.get('intensity', 5)}/10）\n"

        return context_str

    def _count_action_components(self, description: str) -> int:
        """估算动作部件数量"""
        # 基于动词和连接词估算
        action_verbs = ["走", "跑", "跳", "坐", "站", "躺", "拿", "放", "看",
                        "盯", "按", "贴", "收", "缩", "转", "滑", "滚", "吸", "呼"]
        count = 0
        for verb in action_verbs:
            if verb in description:
                count += 1
        return max(count, 1)

    def _assess_action_fineness(self, description: str) -> str:
        """评估动作精细程度"""
        fine_indicators = ["指尖", "喉头", "指节", "瞳孔", "眼眶", "嘴角", "眉梢", "睫毛"]
        for indicator in fine_indicators:
            if indicator in description:
                return "精细"
        return "普通"

    def _assess_movement_range(self, description: str) -> str:
        """评估动作移动范围"""
        if any(word in description for word in ["滑落", "坐直", "走过", "站起", "跑过", "跳下"]):
            return "大范围"
        elif any(word in description for word in ["收紧", "滚动", "打转", "眨眼", "点头"]):
            return "小范围"
        else:
            return "原位"
