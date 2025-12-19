# -*- coding: utf-8 -*-
"""
@FileName: script_parser_agent.py
@Description: LLM 剧本解析功能，AI 生成的剧本解析器
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import json
from typing import Dict, Any, Optional, List

from hengline.agent.workflow_models import ScriptType
from hengline.logger import debug
from hengline.prompts.prompts_manager import prompt_manager
from .base_script_parser import ScriptParser
from .script_extractor.script_action_extractor import action_extractor
from .script_extractor.script_character_extractor import character_extractor
from .script_extractor.script_dialogue_extractor import dialogue_extractor
from .script_extractor.script_field_extractor import StoryboardFieldExtractor
from .script_extractor.script_environment_extractor import environment_extractor
from .script_extractor.script_time_extractor import TimeSegmentExtractor
from .script_parser_model import Scene, Character, Dialogue, Action


class AIStoryboardParser(ScriptParser):
    """
        AI生成剧本解析器

        特点：
            1. 专门处理AI生成的分镜格式（[0-5秒] 画面：... 声音：...）
            2. 支持本地规则解析和AI增强解析
            3. 输出格式与自然语言解析器保持一致
        """

    def __init__(self, llm_client):
        # 初始化各个处理器
        super().__init__(llm_client)
        self.time_extractor = TimeSegmentExtractor()
        self.field_extractor = StoryboardFieldExtractor()

    def _enhance(self, scenes_data, script_text: str) -> List[Dict]:
        """
        使用AI增强分镜解析

        增强内容包括：
            1. 更准确的场景描述
            2. 角色关系的深度理解
            3. 情绪和氛围的识别
            4. 动作序列的连贯性分析
        """
        parser_prompt = prompt_manager.get_script_parser_prompt("ai_storyboard_parser")

        # 1. 构建提示词
        prompt = parser_prompt.format(script_text=script_text)

        # 2. 调用LLM
        response = self._call_llm_with_retry(prompt)
        debug(f"{ScriptType.AI_STORYBOARD.value} 类型的 LLM 解析结果:\n {response}")

        # 3. 解析增强结果
        enhanced_data = self._parse_enhancement_response(response, scenes_data)

        return enhanced_data

    def _parse_enhancement_response(self, response: str, original_scenes: List[Dict]) -> List[Dict]:
        """解析AI增强响应"""
        try:
            enhanced_data = json.loads(response)
            enhanced_scenes = enhanced_data.get("enhanced_scenes", [])

            # 合并增强信息到原始场景
            merged_scenes = []

            for original in original_scenes:
                scene_id = original["scene"]["scene_id"]

                # 查找对应的增强场景
                enhanced_info = None
                for enhanced in enhanced_scenes:
                    if enhanced.get("scene_id") == scene_id:
                        enhanced_info = enhanced
                        break

                # 合并信息
                merged_scene = original.copy()

                if enhanced_info:
                    # 更新场景描述
                    merged_scene["scene"]["description"] = enhanced_info.get(
                        "enhanced_description",
                        merged_scene["scene"]["description"]
                    )

                    # 添加情绪信息
                    if "emotional_tone" in enhanced_info:
                        merged_scene["scene"]["emotional_tone"] = enhanced_info["emotional_tone"]

                    # 添加关键元素
                    if "key_elements" in enhanced_info:
                        merged_scene["scene"]["key_elements"] = enhanced_info["key_elements"]

                merged_scenes.append(merged_scene)

            return merged_scenes

        except json.JSONDecodeError:
            print("AI增强响应JSON解析失败，返回原始数据")
            return original_scenes

    def extract_with_local(self, script_text: str) -> Optional[Dict[str, Any]]:
        """本地规则解析剧本"""
        """
                解析AI分镜剧本

                Args:
                    script_text: AI分镜格式文本
                    use_ai_enhancement: 是否使用AI增强解析

                Returns:
                    统一格式的解析结果
                """
        # 1. 基础解析：提取时间分段和字段
        raw_segments = self._parse_raw_segments(script_text)

        # 2. 转换为场景结构
        scenes_data = self._segments_to_scenes(raw_segments)

        # 3. AI增强（可选）
        enhanced_scenes = self._enhance(scenes_data, script_text)

        # 4. 构建最终结果
        result = self._build_final_result(enhanced_scenes)

        return result

    def _parse_raw_segments(self, script_text: str) -> List[Dict]:
        """
        解析原始分镜分段
        """
        segments = []
        lines = script_text.strip().split('\n')

        current_segment = None
        current_field = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测时间标记
            time_match = self.time_extractor.parse_time_marker(line)
            if time_match:
                # 保存前一个分段
                if current_segment:
                    segments.append(current_segment)

                # 开始新分段
                current_segment = {
                    "time_marker": time_match["original"],
                    "start_second": time_match["start"],
                    "end_second": time_match["end"],
                    "duration": time_match.get("duration", 0),
                    "fields": {}
                }
                current_field = None
                continue

            # 检测字段标记
            field_match = self.field_extractor.extract_field(line)
            if field_match and current_segment:
                field_name = field_match["field_name"]
                field_value = field_match["field_value"]

                current_segment["fields"][field_name] = field_value
                current_field = field_name
            elif current_field and current_segment:
                # 续行内容
                current_segment["fields"][current_field] += " " + line

        # 添加最后一个分段
        if current_segment:
            segments.append(current_segment)

        # 计算缺失的时长
        segments = self._calculate_missing_durations(segments)

        return segments

    def _calculate_missing_durations(self, segments: List[Dict]) -> List[Dict]:
        """计算缺失的时长"""
        for i, segment in enumerate(segments):
            if "duration" not in segment or segment["duration"] <= 0:
                # 默认每个分镜5秒
                segment["duration"] = 5

                # 计算起止时间
                if i == 0:
                    segment["start_second"] = 0
                    segment["end_second"] = segment["duration"]
                else:
                    prev_segment = segments[i - 1]
                    segment["start_second"] = prev_segment["end_second"]
                    segment["end_second"] = segment["start_second"] + segment["duration"]

        return segments

    def _segments_to_scenes(self, segments: List[Dict]) -> List[Dict]:
        """
        将分镜分段转换为场景结构
        """
        scenes = []

        for i, segment in enumerate(segments):
            scene_id = f"scene_{i + 1:03d}"

            # 合并字段内容创建场景描述
            scene_description = self._create_scene_description(segment)

            # 提取场景元素
            characters = self._extract_characters_from_segment(segment)
            dialogues = self._extract_dialogues_from_segment(segment, scene_id)
            actions = self._extract_actions_from_segment(segment, scene_id)
            environment = self._extract_environment_from_segment(segment)

            # 构建场景
            scene = Scene(
                scene_id=scene_id,
                order_index=i,
                location=environment.get("location", ""),
                time_of_day=environment.get("time_of_day", ""),
                mood=environment.get("mood", ""),
                summary=scene_description,
                character_refs=[char.name for char in characters],
                dialogue_refs=[d.dialogue_id for d in dialogues],
                action_refs=[a.action_id for a in actions],
                start_time=segment["start_second"],
                end_time=segment["end_second"],
                duration=segment["duration"]
            )

            scenes.append({
                "scene": scene,
                "characters": characters,
                "dialogues": dialogues,
                "actions": actions
            })

        return scenes

    def _build_final_result(self, scenes_data: List[Dict]) -> Dict:
        """构建最终的统一格式结果"""
        # 收集所有数据
        all_scenes = []
        all_characters = []
        all_dialogues = []
        all_actions = []
        all_descriptions = []

        for scene_info in scenes_data:
            scene = scene_info["scene"]
            all_scenes.append(scene)
            all_descriptions.append(scene["description"])

            # 收集角色（去重）
            for char in scene_info["characters"]:
                if char["name"] not in [c["name"] for c in all_characters]:
                    all_characters.append(char)

            all_dialogues.extend(scene_info["dialogues"])
            all_actions.extend(scene_info["actions"])

        # 构建结果
        result = {
            "format_type": "ai_storyboard",
            "scenes": all_scenes,
            "characters": all_characters,
            "dialogues": all_dialogues,
            "actions": all_actions,
            "descriptions": all_descriptions,
            "parsing_confidence": self._calculate_confidence(scenes_data)
        }

        return result

    def _calculate_confidence(self, scenes_data: List[Dict]) -> Dict:
        """计算解析置信度"""
        if not scenes_data:
            return {"overall": 0.0}

        # 评估标准
        total_score = 0
        criteria_count = 0

        for scene_info in scenes_data:
            scene = scene_info["scene"]

            # 1. 字段完整性评分
            field_score = self._evaluate_field_completeness(scene.get("original_fields", {}))
            total_score += field_score
            criteria_count += 1

            # 2. 时长合理性评分
            duration = scene.get("duration", 0)
            duration_score = 1.0 if 2 <= duration <= 10 else 0.5  # 2-10秒为合理范围
            total_score += duration_score
            criteria_count += 1

            # 3. 内容充实度评分
            content_score = min(1.0, len(scene["description"]) / 50)  # 50字符为满分
            total_score += content_score
            criteria_count += 1

        # 计算平均分
        avg_score = total_score / criteria_count if criteria_count > 0 else 0

        return {
            "field_completeness": round(avg_score * 0.9, 2),
            "time_accuracy": round(avg_score * 0.8, 2),
            "content_richness": round(avg_score * 0.85, 2),
            "character_recognition": round(avg_score * 0.75, 2),
            "overall": round(avg_score, 2)
        }

    def _evaluate_field_completeness(self, fields: Dict) -> float:
        """评估字段完整性"""
        # 重要字段
        important_fields = ["visual", "audio", "action"]

        # 计算重要字段的存在比例
        present_count = sum(1 for field in important_fields if field in fields)
        completeness = present_count / len(important_fields)

        # 如果有额外字段，加分
        extra_fields = len(fields) - present_count
        if extra_fields > 0:
            completeness = min(1.0, completeness + 0.1 * extra_fields)

        return completeness

    def _create_scene_description(self, segment: Dict) -> str:
        """创建场景描述"""
        fields = segment.get("fields", {})

        # 使用分镜字段提取器合并描述
        description = self.field_extractor.merge_fields_to_description(fields)

        # 如果没有字段，使用时间标记作为描述
        if not description:
            description = f"第{segment['start_second']}-{segment['end_second']}秒的场景"

        return description

    def _extract_characters_from_segment(self, segment: Dict) -> List[Character]:
        """从分镜分段中提取角色"""
        characters = []

        # 从各个字段中提取角色
        fields = segment.get("fields", {})

        # 合并所有字段文本
        all_text = " ".join(fields.values())

        # 复用通用角色提取器
        if all_text:
            characters = character_extractor.extract(all_text)

        return characters

    def _extract_dialogues_from_segment(self, segment: Dict, scene_id: str) -> List[Dialogue]:
        """从分镜分段中提取对话"""
        dialogues = []

        # 主要从音频字段提取
        audio_text = segment.get("fields", {}).get("audio", "")

        if audio_text:
            # 复用通用对话提取器
            dialogues = dialogue_extractor.extract(audio_text, scene_id)

        # 如果没有音频字段，尝试从其他字段提取
        if not dialogues:
            all_text = " ".join(segment.get("fields", {}).values())
            if all_text:
                dialogues = dialogue_extractor.extract(all_text, scene_id)

        return dialogues

    def _extract_actions_from_segment(self, segment: Dict, scene_id: str) -> List[Action]:
        """从分镜分段中提取动作"""
        actions = []

        # 主要从动作字段提取
        action_text = segment.get("fields", {}).get("action", "")

        if action_text:
            # 复用通用动作提取器
            actions = action_extractor.extract(action_text, scene_id)

        # 如果没有动作字段，从视觉字段提取
        if not actions:
            visual_text = segment.get("fields", {}).get("visual", "")
            if visual_text:
                actions = action_extractor.extract(visual_text, scene_id)

        return actions

    def _extract_environment_from_segment(self, segment: Dict) -> Dict:
        """从分镜分段中提取环境信息"""
        environment = {}

        # 合并所有字段文本
        all_text = " ".join(segment.get("fields", {}).values())

        if all_text:
            # 复用通用环境提取器
            env_info = environment_extractor.extract(all_text)

            # 从视觉字段提取地点
            visual_text = segment.get("fields", {}).get("visual", "")
            if visual_text:
                # 简单的地点提取逻辑
                locations = ["客厅", "卧室", "办公室", "街道", "公园", "餐厅", "学校", "商店", "医院",
                             "机场", "火车站", "咖啡馆", "图书馆", "电影院"]
                for location in locations:
                    if location in visual_text:
                        env_info["location"] = location
                        break

            environment = env_info

        return environment
