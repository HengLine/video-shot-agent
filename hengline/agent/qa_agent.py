# -*- coding: utf-8 -*-
"""
@FileName: qa_agent.py
@Description: 分镜审查智能体，负责审查分镜质量和连续性
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
import json
from pathlib import Path
from typing import Dict, List, Any

from hengline.logger import debug, warning
from hengline.prompts.prompts_manager import PromptManager


class QAAgent:
    """质量审查智能体"""

    def __init__(self, llm=None):
        """
        初始化质量审查智能体
        
        Args:
            llm: 语言模型实例（可选，用于高级审查）
        """
        self.llm = llm
        self.max_shot_duration = 5.5  # 最大允许时长（秒）

    def review_single_shot(self, shot: Dict[str, Any], segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        审查单个分镜
        
        Args:
            shot: 分镜对象
            segment: 对应的分段信息
            
        Returns:
            审查结果
        """
        debug(f"审查分镜，ID: {shot.get('shot_id')}")

        critical_issues = []  # 关键错误，需要修正
        warnings = []         # 警告，不阻止继续处理
        suggestions = []

        # 检查基本字段
        basic_check = self._check_basic_fields(shot)
        critical_issues.extend(basic_check["critical_issues"])
        warnings.extend(basic_check["warnings"])
        suggestions.extend(basic_check["suggestions"])

        # 检查时长
        duration_check = self._check_duration(shot)
        critical_issues.extend(duration_check["critical_issues"])
        warnings.extend(duration_check["warnings"])
        suggestions.extend(duration_check["suggestions"])

        # 检查角色状态
        character_check = self._check_character_states(shot)
        critical_issues.extend(character_check["critical_issues"])
        warnings.extend(character_check["warnings"])
        suggestions.extend(character_check["suggestions"])

        # 检查提示词质量
        prompt_check = self._check_prompt_quality(shot)
        critical_issues.extend(prompt_check["critical_issues"])
        warnings.extend(prompt_check["warnings"])
        suggestions.extend(prompt_check["suggestions"])

        # 如果有LLM，进行高级审查
        if self.llm:
            advanced_check = self._advanced_review_with_llm(shot, segment)
            critical_issues.extend(advanced_check.get("critical_issues", []))
            warnings.extend(advanced_check.get("warnings", []))
            suggestions.extend(advanced_check.get("suggestions", []))

        result = {
            "shot_id": shot.get("shot_id"),
            "is_valid": len(critical_issues) == 0,
            "critical_issues": critical_issues,
            "warnings": warnings,
            "suggestions": suggestions
        }

        if critical_issues:
            warning(f"分镜 {shot.get('shot_id')} 审查发现关键问题: {critical_issues}")
        if warnings:
            debug(f"分镜 {shot.get('shot_id')} 审查发现警告: {warnings}")
        if not critical_issues and not warnings:
            debug(f"分镜 {shot.get('shot_id')} 审查通过")

        return result

    def review_shot_sequence(self, shots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        审查分镜序列的连续性
        
        Args:
            shots: 分镜列表
            
        Returns:
            审查结果
        """
        debug(f"审查分镜序列，共 {len(shots)} 个分镜")

        continuity_issues = []
        continuity_suggestions = []

        # 检查分镜之间的连续性
        for i in range(1, len(shots)):
            prev_shot = shots[i - 1]
            current_shot = shots[i]

            # 检查时间连续性
            time_continuity = self._check_time_continuity(prev_shot, current_shot)
            if not time_continuity["is_continuous"]:
                continuity_issues.append(f"分镜 {prev_shot.get('shot_id')} 和 {current_shot.get('shot_id')} 时间不连续")
                continuity_suggestions.extend(time_continuity["suggestions"])

            # 检查角色连续性
            character_continuity = self._check_character_continuity(prev_shot, current_shot)
            # 合并critical_issues和warnings作为连续性问题
            continuity_issues.extend(character_continuity["critical_issues"])
            continuity_issues.extend(character_continuity["warnings"])
            continuity_suggestions.extend(character_continuity["suggestions"])

            # 检查场景连续性
            scene_continuity = self._check_scene_continuity(prev_shot, current_shot)
            if not scene_continuity["is_continuous"]:
                continuity_issues.append(f"分镜 {prev_shot.get('shot_id')} 和 {current_shot.get('shot_id')} 场景不连续")
                continuity_suggestions.extend(scene_continuity["suggestions"])

        # 检查整体叙事连贯性
        narrative_check = self._check_narrative_coherence(shots)
        # 合并critical_issues和warnings作为连续性问题
        continuity_issues.extend(narrative_check["critical_issues"])
        continuity_issues.extend(narrative_check["warnings"])
        continuity_suggestions.extend(narrative_check["suggestions"])

        result = {
            "total_shots": len(shots),
            "has_continuity_issues": len(continuity_issues) > 0,
            "continuity_issues": continuity_issues,
            "continuity_suggestions": continuity_suggestions,
            "overall_assessment": "通过" if len(continuity_issues) == 0 else "需要修正"
        }

        if continuity_issues:
            warning(f"分镜序列审查发现连续性问题: {continuity_issues}")
        else:
            debug("分镜序列连续性审查通过")

        return result

    def _check_basic_fields(self, shot: Dict[str, Any]) -> Dict[str, Any]:
        """检查基本字段是否完整"""
        critical_issues = []
        warnings = []
        suggestions = []

        # 核心必填字段（缺少会导致功能错误）
        core_required_fields = ["shot_id", "chinese_description"]
        # 推荐字段（缺少会影响质量但不阻止继续）
        recommended_fields = ["ai_prompt", "camera", "characters_in_frame"]

        for field in core_required_fields:
            if field not in shot or (isinstance(shot[field], (str, list)) and not shot[field]):
                critical_issues.append(f"缺少必要字段: {field}")
                suggestions.append(f"请添加 {field}")

        for field in recommended_fields:
            if field not in shot or (isinstance(shot[field], (str, list)) and not shot[field]):
                warnings.append(f"缺少推荐字段: {field}")
                suggestions.append(f"建议添加 {field}")

        return {"critical_issues": critical_issues, "warnings": warnings, "suggestions": suggestions}

    def _check_duration(self, shot: Dict[str, Any]) -> Dict[str, Any]:
        """检查分镜时长"""
        critical_issues = []
        warnings = []
        suggestions = []

        time_range = shot.get("time_range_sec", [0, 5])
        if not isinstance(time_range, list) or len(time_range) != 2:
            critical_issues.append("时间范围格式错误")
            suggestions.append("请设置正确的时间范围格式 [开始时间, 结束时间]")
            return {"critical_issues": critical_issues, "warnings": warnings, "suggestions": suggestions}

        duration = time_range[1] - time_range[0]

        # 允许一定的容忍度，轻微超出范围只作为警告
        if duration > self.max_shot_duration + 0.5:
            critical_issues.append(f"分镜时长长于允许的最大值 {self.max_shot_duration} 秒")
            suggestions.append("请缩短分镜时长或拆分为多个分镜")
        elif duration > self.max_shot_duration:
            warnings.append(f"分镜时长略长于推荐值 {self.max_shot_duration} 秒")
            suggestions.append("建议适当缩短分镜时长")

        return {"critical_issues": critical_issues, "warnings": warnings, "suggestions": suggestions}

    def _check_character_states(self, shot: Dict[str, Any]) -> Dict[str, Any]:
        """检查角色状态是否合理"""
        critical_issues = []
        warnings = []
        suggestions = []

        # 检查初始状态和结束状态
        initial_state = shot.get("initial_state", [])
        final_state = shot.get("final_state", [])
        characters_in_frame = shot.get("characters_in_frame", [])

        # 特殊处理电话角色（标记为off-screen）
        is_phone_scene = any("电话" in action.get("action", "") for action in shot.get("actions", []))
        phone_characters = []
        if is_phone_scene:
            for character in characters_in_frame:
                if "电话" in character or "对面" in character:
                    phone_characters.append(character)

        # 构建角色状态映射
        initial_char_map = {s.get("character_name"): s for s in initial_state}
        final_char_map = {s.get("character_name"): s for s in final_state}
        all_state_characters = set(initial_char_map.keys()) | set(final_char_map.keys())

        # 检查画面内角色是否都有状态信息
        for character in characters_in_frame:
            if character not in phone_characters and character not in all_state_characters:
                warnings.append(f"角色 {character} 缺少状态信息")
                suggestions.append(f"建议添加 {character} 的状态信息")

        # 检查角色状态的合理性
        for character_name, state in {**initial_char_map, **final_char_map}.items():
            # 电话角色的特殊规则
            if character_name in phone_characters:
                if state.get("position") != "off-screen":
                    warnings.append(f"电话角色 {character_name} 位置应为 'off-screen'")
                    suggestions.append(f"设置 {character_name} 的位置为 'off-screen'")
            else:
                # 普通角色的状态检查
                if "position" not in state:
                    critical_issues.append(f"角色 {character_name} 缺少位置信息")
                    suggestions.append(f"请添加 {character_name} 的位置信息")
                # 姿势和情绪信息降级为警告
                if "pose" not in state:
                    warnings.append(f"角色 {character_name} 缺少姿势信息")
                    suggestions.append(f"建议添加 {character_name} 的姿势信息")
                if "emotion" not in state:
                    warnings.append(f"角色 {character_name} 缺少情绪信息")
                    suggestions.append(f"建议添加 {character_name} 的情绪信息")

        return {"critical_issues": critical_issues, "warnings": warnings, "suggestions": suggestions}

    def _check_prompt_quality(self, shot: Dict[str, Any]) -> Dict[str, Any]:
        """检查提示词质量"""
        critical_issues = []
        warnings = []
        suggestions = []

        ai_prompt = shot.get("ai_prompt", "")

        # 提示词为空是警告级别
        if not ai_prompt:
            warnings.append("AI提示词为空")
            suggestions.append("请添加AI提示词以提高生成质量")
        else:
            # 检查提示词长度，轻微过短只作为警告
            if len(ai_prompt) < 10:
                critical_issues.append("AI提示词过短")
                suggestions.append("请添加更多细节到提示词")
            elif len(ai_prompt) < 20:
                warnings.append("AI提示词较短")
                suggestions.append("建议添加更多细节到提示词")

            # 检查是否包含必要元素
            essential_elements = ["shot", "lighting", "style"]
            for element in essential_elements:
                if element not in ai_prompt.lower():
                    suggestions.append(f"建议在提示词中添加 {element} 相关描述")

        return {"critical_issues": critical_issues, "warnings": warnings, "suggestions": suggestions}

    def _advanced_review_with_llm(self, shot: Dict[str, Any], segment: Dict[str, Any]) -> Dict[str, Any]:
        """使用LLM进行高级审查"""
        try:
            # 使用PromptManager获取提示词，使用正确的提示词目录路径
            prompt_manager = PromptManager(prompt_dir=Path(__file__).parent.parent)
            prompt = prompt_manager.get_prompt("qa_review_prompt")

            # 填充提示词模板
            filled_prompt = prompt.format(
                shot_info=json.dumps(shot, ensure_ascii=False),
                segment_info=json.dumps(segment, ensure_ascii=False)
            )

            # 调用LLM
            response = self.llm.invoke(filled_prompt)

            # 处理可能的响应对象
            response_text = response.content if hasattr(response, 'content') else response

            # 检查响应是否为空
            if not response_text or not str(response_text).strip():
                warning("LLM高级审查响应为空")
                return {"issues": [], "suggestions": []}

            # 确保response_text是字符串
            response_text = str(response_text).strip()

            # 使用JSON响应解析器解析响应
            from hengline.tools import parse_json_response
            result = parse_json_response(response_text)
            return result
        except json.JSONDecodeError as e:
            warning(f"LLM高级审查JSON解析失败: {str(e)}, 响应文本: {str(response_text)[:100]}...")
            return {"issues": [], "suggestions": []}
        except Exception as e:
            warning(f"LLM高级审查失败: {str(e)}")
            return {"issues": [], "suggestions": []}

    def _check_time_continuity(self, prev_shot: Dict[str, Any], current_shot: Dict[str, Any]) -> Dict[str, Any]:
        """检查时间连续性"""
        prev_end = prev_shot.get("time_range_sec", [0, 5])[1]
        current_start = current_shot.get("time_range_sec", [5, 10])[0]

        is_continuous = abs(prev_end - current_start) < 0.1

        suggestions = []
        if not is_continuous:
            suggestions.append("修正时间范围以确保连续")

        return {
            "is_continuous": is_continuous,
            "suggestions": suggestions
        }

    def _check_character_continuity(self, prev_shot: Dict[str, Any], current_shot: Dict[str, Any]) -> Dict[str, Any]:
        """检查角色连续性"""
        critical_issues = []
        warnings = []
        suggestions = []

        # 获取前一个分镜的结束状态和当前分镜的初始状态
        prev_final_state = {s.get("character_name"): s for s in prev_shot.get("final_state", [])}
        current_initial_state = {s.get("character_name"): s for s in current_shot.get("initial_state", [])}

        # 特殊处理电话场景
        is_prev_phone = any("电话" in action.get("action", "") for action in prev_shot.get("actions", []))
        is_current_phone = any("电话" in action.get("action", "") for action in current_shot.get("actions", []))
        
        # 检查共同角色
        common_characters = set(prev_final_state.keys()) & set(current_initial_state.keys())

        for character in common_characters:
            prev_state = prev_final_state[character]
            current_state = current_initial_state[character]

            # 电话角色特殊处理
            is_phone_character = "电话" in character or "对面" in character
            
            # 检查位置连续性（允许合理的位置变化）
            prev_pos = prev_state.get("position")
            current_pos = current_state.get("position")
            
            if is_phone_character:
                # 电话角色位置应该始终为off-screen
                if prev_pos != "off-screen" and current_pos != "off-screen":
                    warnings.append(f"电话角色 {character} 位置应为 'off-screen'")
                    suggestions.append(f"设置 {character} 的位置为 'off-screen'")
            elif prev_pos and current_pos and prev_pos != current_pos:
                # 允许合理的位置变化（例如：站立->坐下、门口->窗边等）
                # 只有剧烈的位置变化才作为警告
                剧烈变化 = [
                    ("客厅", "卧室"), ("卧室", "客厅"),
                    ("室内", "室外"), ("室外", "室内")
                ]
                if (prev_pos, current_pos) in 剧烈变化 or (current_pos, prev_pos) in 剧烈变化:
                    warnings.append(f"角色 {character} 位置变化较大")
                    suggestions.append(f"确保 {character} 的位置变化有合理过渡")

            # 检查情绪连续性
            prev_emotion = prev_state.get("emotion")
            current_emotion = current_state.get("emotion")
            if prev_emotion and current_emotion and prev_emotion != current_emotion:
                # 检查是否是合理的情绪过渡
                if not self._is_valid_emotion_transition(prev_emotion, current_emotion):
                    warnings.append(f"角色 {character} 情绪过渡可能不合理")
                    suggestions.append(f"建议添加 {character} 的情绪过渡描述")

        # 检查角色突然出现或消失
        prev_characters = set(prev_final_state.keys())
        current_characters = set(current_initial_state.keys())
        
        new_characters = current_characters - prev_characters
        if new_characters:
            warnings.append(f"新角色突然出现: {', '.join(new_characters)}")
            suggestions.append("建议添加角色入场的自然过渡")
            
        disappeared_characters = prev_characters - current_characters
        if disappeared_characters:
            warnings.append(f"角色突然消失: {', '.join(disappeared_characters)}")
            suggestions.append("建议添加角色退场的自然过渡")

        return {"critical_issues": critical_issues, "warnings": warnings, "suggestions": suggestions}

    def _check_scene_continuity(self, prev_shot: Dict[str, Any], current_shot: Dict[str, Any]) -> Dict[str, Any]:
        """检查场景连续性"""
        prev_scene = prev_shot.get("scene_context", {})
        current_scene = current_shot.get("scene_context", {})

        is_continuous = True
        suggestions = []

        # 检查位置、时间、氛围是否连续
        if prev_scene.get("location") != current_scene.get("location"):
            is_continuous = False
            suggestions.append("场景位置发生变化，请确保有合理的转场")

        if prev_scene.get("time") != current_scene.get("time"):
            is_continuous = False
            suggestions.append("场景时间发生变化，请添加时间过渡说明")

        return {
            "is_continuous": is_continuous,
            "suggestions": suggestions
        }

    def _check_narrative_coherence(self, shots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检查叙事连贯性"""
        critical_issues = []
        warnings = []
        suggestions = []

        # 简单的叙事连贯性检查
        total_duration = len(shots) * 5
        if total_duration > 300:  # 超过5分钟
            warnings.append("视频总时长过长，可能影响叙事连贯性")
            suggestions.append("考虑精简内容或分章节制作")

        # 检查角色出现频率
        character_counts = {}
        for shot in shots:
            for character in shot.get("characters_in_frame", []):
                character_counts[character] = character_counts.get(character, 0) + 1

        # 检查是否有角色突然消失
        for i in range(1, len(shots)):
            prev_characters = set(shots[i - 1].get("characters_in_frame", []))
            current_characters = set(shots[i].get("characters_in_frame", []))

            disappeared_characters = prev_characters - current_characters
            for character in disappeared_characters:
                # 如果角色出现过多次但突然消失，可能是问题
                if character_counts.get(character, 0) > 2:
                    warnings.append(f"角色 {character} 突然消失")
                    suggestions.append(f"请添加 {character} 的离开场景")

        return {"critical_issues": critical_issues, "warnings": warnings, "suggestions": suggestions}

    def _is_valid_emotion_transition(self, prev_emotion: str, current_emotion: str) -> bool:
        """检查情绪过渡是否合理"""
        # 定义有效的情绪过渡对
        valid_transitions = {
            "平静": ["惊讶", "注意", "思考", "微笑"],
            "惊讶": ["震惊", "恐惧", "困惑", "平静"],
            "震惊": ["恐惧", "悲伤", "愤怒", "平静"],
            "愤怒": ["攻击", "冷静", "悲伤"],
            "悲伤": ["哭泣", "平静", "接受"],
            "快乐": ["大笑", "平静", "兴奋"],
            "紧张": ["焦虑", "恐惧", "平静"],
            "恐惧": ["逃跑", "震惊", "平静"],
        }

        if prev_emotion in valid_transitions:
            return current_emotion in valid_transitions[prev_emotion] or current_emotion == prev_emotion

        return True
