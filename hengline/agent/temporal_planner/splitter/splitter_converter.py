"""
@FileName: splitter_converter.py
@Description: 数据格式的转换
@Author: HengLine
@Time: 2026/1/15 16:19
"""

from typing import Dict, List, Any, Tuple

from hengline.agent.script_parser.script_parser_models import UnifiedScript


class UnifiedScriptConverter:
    """将 UnifiedScript 转换为分片器所需格式"""

    @staticmethod
    def convert_to_splitter_format(unified_script: UnifiedScript) -> Tuple[Dict[str, Dict], List[str]]:
        """
        将 UnifiedScript 转换为分片器所需格式

        Args:
            unified_script: 智能体1输出的完整结构化剧本

        Returns:
            Tuple[original_data_dict, element_order_list]
        """
        original_data = {}
        element_order = []

        # 1. 提取场景
        for scene in unified_script.scenes:
            scene_id = scene.get("scene_id", f"scene_{len(original_data)}")
            original_data[scene_id] = {
                "type": "scene",
                "description": scene.get("description", ""),
                "location": scene.get("location", ""),
                "time_of_day": scene.get("time_of_day", ""),
                "weather": scene.get("weather", ""),
                "mood": scene.get("mood", ""),
                "key_visuals": scene.get("key_visuals", []),
                "props": scene.get("props", []),
                "character_refs": scene.get("character_refs", []),
                "original_start_time": scene.get("start_time"),
                "original_end_time": scene.get("end_time")
            }
            element_order.append(scene_id)

        # 2. 提取对话（按 time_offset 排序）
        dialogues = sorted(
            unified_script.get("dialogues", []),
            key=lambda d: d.get("time_offset", 0)
        )

        for dialogue in dialogues:
            dialogue_id = dialogue.get("dialogue_id", f"dialogue_{len(original_data)}")
            original_data[dialogue_id] = {
                "type": "dialogue",
                "content": dialogue.get("content", ""),
                "speaker": dialogue.get("speaker", ""),
                "target": dialogue.get("target", ""),
                "emotion": dialogue.get("emotion", ""),
                "voice_quality": dialogue.get("voice_quality", ""),
                "parenthetical": dialogue.get("parenthetical", ""),
                "dialogue_type": dialogue.get("type", "speech"),
                "scene_ref": dialogue.get("scene_ref", ""),
                "original_time_offset": dialogue.get("time_offset"),
                "original_duration": dialogue.get("duration")
            }
            element_order.append(dialogue_id)

        # 3. 提取动作（按 time_offset 排序）
        actions = sorted(
            unified_script.get("actions", []),
            key=lambda a: a.get("time_offset", 0)
        )

        for action in actions:
            action_id = action.get("action_id", f"action_{len(original_data)}")
            original_data[action_id] = {
                "type": "action",
                "description": action.get("description", ""),
                "actor": action.get("actor", ""),
                "target": action.get("target", ""),
                "action_type": action.get("type", ""),
                "scene_ref": action.get("scene_ref", ""),
                "original_time_offset": action.get("time_offset"),
                "original_duration": action.get("duration")
            }
            element_order.append(action_id)

        return original_data, element_order

    @staticmethod
    def create_estimations_from_original(original_data: Dict[str, Dict]) -> Dict[str, Any]:
        """
        从原始数据创建基础的时长估算（如果没有AI/规则估算时使用）
        """
        estimations = {}

        for elem_id, elem_data in original_data.items():
            elem_type = elem_data.get("type", "")

            # 根据类型确定默认时长
            if elem_type == "scene":
                base_duration = 4.0
                confidence = 0.3
            elif elem_type == "dialogue":
                content = elem_data.get("content", "")
                word_count = len(content.split())
                base_duration = word_count * 0.4 if word_count > 0 else 2.5
                confidence = 0.5
            elif elem_type == "action":
                base_duration = 1.5
                confidence = 0.4
            else:
                base_duration = 2.0
                confidence = 0.3

            estimations[elem_id] = {
                "estimated_duration": base_duration,
                "confidence": confidence,
                "element_type": elem_type,
                "original_duration": elem_data.get("original_duration", base_duration)
            }

        return estimations


class ElementOrderOptimizer:
    """元素顺序优化器"""

    @staticmethod
    def optimize_element_order(original_data: Dict[str, Dict],
                               element_order: List[str]) -> List[str]:
        """
        优化元素顺序，确保时间逻辑合理

        根据原始数据中的时间信息重新排序
        """
        # 提取时间信息
        elements_with_time = []

        for elem_id in element_order:
            if elem_id in original_data:
                elem = original_data[elem_id]

                # 获取时间信息
                time_info = elem.get("original_time_offset")
                if time_info is None:
                    # 对于场景，可能没有time_offset，使用0或特殊值
                    if elem.get("type") == "scene":
                        time_info = 0
                    else:
                        time_info = 9999  # 放到最后

                elements_with_time.append((time_info, elem_id))

        # 按时间排序
        elements_with_time.sort(key=lambda x: x[0])

        # 返回排序后的元素ID列表
        return [elem_id for _, elem_id in elements_with_time]

    @staticmethod
    def validate_element_order(original_data: Dict[str, Dict],
                               element_order: List[str]) -> List[str]:
        """
        验证元素顺序的合理性

        Returns:
            问题描述列表
        """
        issues = []

        # 检查场景是否在对应对话和动作之前
        scene_elements = {}
        for elem_id in element_order:
            if elem_id in original_data:
                elem = original_data[elem_id]
                if elem.get("type") == "scene":
                    scene_ref = elem.get("scene_ref", "")
                    if scene_ref:
                        scene_elements[scene_ref] = elem_id

        for elem_id in element_order:
            if elem_id in original_data:
                elem = original_data[elem_id]
                scene_ref = elem.get("scene_ref", "")

                if scene_ref and scene_ref in scene_elements:
                    scene_position = element_order.index(scene_elements[scene_ref])
                    elem_position = element_order.index(elem_id)

                    if elem_position < scene_position:
                        issues.append(f"元素 {elem_id} 出现在其场景 {scene_ref} 之前")

        return issues