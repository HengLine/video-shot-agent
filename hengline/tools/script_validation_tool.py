"""
@FileName: script_validation_tool.py
@Description: 剧本验证工具，验证结构化剧本的完整性和一致性
@Author: HengLine
@Time: 2025/12/18 23:35
"""
import re
from typing import List, Dict

from hengline.agent.script_parser.script_parser_model import UnifiedScript


class BasicScriptValidator:
    """剧本基础验证器"""

    def validate(self, unified_script: UnifiedScript) -> tuple[bool, list[dict]]:
        """
        验证统一格式剧本的基本完整性

        返回：(是否有效, 问题列表)
        """
        issues = []

        # 1. 结构完整性验证
        issues.extend(self._validate_structure(unified_script))

        # 2. 数据一致性验证
        issues.extend(self._validate_consistency(unified_script))

        # 3. 引用完整性验证
        issues.extend(self._validate_references(unified_script))

        # 4. 内容合理性验证
        issues.extend(self._validate_content_reasonableness(unified_script))

        is_valid = len([i for i in issues if i["severity"] == "error"]) == 0

        return is_valid, issues

    def _validate_structure(self, script: UnifiedScript) -> List[Dict]:
        """验证基础结构"""
        issues = []

        # 检查必需字段
        required_fields = ["scenes", "characters", "dialogues", "actions"]
        for field in required_fields:
            if not hasattr(script, field) or not getattr(script, field):
                issues.append({
                    "type": "missing_field",
                    "field": field,
                    "severity": "error",
                    "message": f"缺少必需字段: {field}",
                    "suggestion": f"请确保剧本包含{field}"
                })

        # 检查场景数量
        if hasattr(script, "scenes") and script.scenes:
            if len(script.scenes) > 100:
                issues.append({
                    "type": "too_many_scenes",
                    "severity": "warning",
                    "message": f"场景数量过多 ({len(script.scenes)}个)",
                    "suggestion": "考虑分割剧本或合并简单场景"
                })
            elif len(script.scenes) == 0:
                issues.append({
                    "type": "no_scenes",
                    "severity": "error",
                    "message": "未提取到任何场景",
                    "suggestion": "请检查剧本格式或内容"
                })

        return issues

    def _validate_consistency(self, script: UnifiedScript) -> List[Dict]:
        """验证数据一致性"""
        issues = []

        if not hasattr(script, "scenes") or not script.scenes:
            return issues

        # 1. 角色一致性验证
        all_referenced_chars = set()
        for scene in script.scenes:
            all_referenced_chars.update(scene.character_refs)

        # 检查是否有场景引用了不存在的角色
        defined_char_names = {char.name for char in script.characters}
        undefined_chars = all_referenced_chars - defined_char_names

        for char_name in undefined_chars:
            issues.append({
                "type": "undefined_character",
                "severity": "warning",
                "character": char_name,
                "message": f"场景引用了未定义的角色: {char_name}",
                "suggestion": "请在角色列表中定义此角色，或检查角色名拼写"
            })

        # 2. ID唯一性验证
        scene_ids = [s.scene_id for s in script.scenes]
        duplicate_scenes = self._find_duplicates(scene_ids)
        if duplicate_scenes:
            issues.append({
                "type": "duplicate_scene_ids",
                "severity": "error",
                "ids": duplicate_scenes,
                "message": f"发现重复的场景ID: {duplicate_scenes}",
                "suggestion": "确保每个场景有唯一ID"
            })

        # 3. 对话角色引用验证
        for dialogue in script.dialogues:
            if dialogue.speaker not in defined_char_names:
                issues.append({
                    "type": "dialogue_undefined_speaker",
                    "severity": "warning",
                    "dialogue_id": dialogue.dialogue_id,
                    "speaker": dialogue.speaker,
                    "message": f"对话{dialogue.dialogue_id}的说话者'{dialogue.speaker}'未在角色列表中定义",
                    "suggestion": f"将'{dialogue.speaker}'添加到角色列表"
                })

        return issues

    def _validate_references(self, script: UnifiedScript) -> List[Dict]:
        """验证引用完整性"""
        issues = []

        if not hasattr(script, "scenes") or not script.scenes:
            return issues

        # 建立索引
        scene_ids = {s.scene_id for s in script.scenes}
        dialogue_ids = {d.dialogue_id for d in script.dialogues}
        action_ids = {a.action_id for a in script.actions}

        # 检查场景中的引用
        for scene in script.scenes:
            # 检查对话引用
            for dialogue_id in scene.dialogue_refs:
                if dialogue_id not in dialogue_ids:
                    issues.append({
                        "type": "invalid_dialogue_ref",
                        "severity": "warning",
                        "scene_id": scene.scene_id,
                        "dialogue_id": dialogue_id,
                        "message": f"场景{scene.scene_id}引用了不存在的对话: {dialogue_id}",
                        "suggestion": "检查对话ID或重新解析剧本"
                    })

            # 检查动作引用
            for action_id in scene.action_refs:
                if action_id not in action_ids:
                    issues.append({
                        "type": "invalid_action_ref",
                        "severity": "warning",
                        "scene_id": scene.scene_id,
                        "action_id": action_id,
                        "message": f"场景{scene.scene_id}引用了不存在的动作: {action_id}",
                        "suggestion": "检查动作ID或重新解析剧本"
                    })

        # 检查对话的场景引用
        for dialogue in script.dialogues:
            if dialogue.scene_ref and dialogue.scene_ref not in scene_ids:
                issues.append({
                    "type": "invalid_scene_ref_in_dialogue",
                    "severity": "warning",
                    "dialogue_id": dialogue.dialogue_id,
                    "scene_ref": dialogue.scene_ref,
                    "message": f"对话{dialogue.dialogue_id}引用了不存在的场景: {dialogue.scene_ref}",
                    "suggestion": "检查场景ID或重新解析剧本"
                })

        # 检查动作的场景引用
        for action in script.actions:
            if action.scene_ref and action.scene_ref not in scene_ids:
                issues.append({
                    "type": "invalid_scene_ref_in_action",
                    "severity": "warning",
                    "action_id": action.action_id,
                    "scene_ref": action.scene_ref,
                    "message": f"动作{action.action_id}引用了不存在的场景: {action.scene_ref}",
                    "suggestion": "检查场景ID或重新解析剧本"
                })

        return issues

    def _validate_content_reasonableness(self, script: UnifiedScript) -> List[Dict]:
        """验证内容合理性"""
        issues = []

        # 1. 检查异常长的对话
        for dialogue in script.dialogues:
            if len(dialogue.text) > 500:  # 超过500字符的对话
                issues.append({
                    "type": "excessively_long_dialogue",
                    "severity": "warning",
                    "dialogue_id": dialogue.dialogue_id,
                    "length": len(dialogue.text),
                    "message": f"对话{dialogue.dialogue_id}过长 ({len(dialogue.text)}字符)",
                    "suggestion": "考虑分割长对话为多个较短对话"
                })

        # 2. 检查空内容
        for scene in script.scenes:
            if not scene.summary.strip():
                issues.append({
                    "type": "empty_scene_description",
                    "severity": "warning",
                    "scene_id": scene.scene_id,
                    "message": f"场景{scene.scene_id}描述为空",
                    "suggestion": "添加场景描述或检查解析结果"
                })

        # 3. 检查重复内容
        scene_summaries = [s.summary.strip() for s in script.scenes]
        duplicate_scenes = self._find_similar_texts(scene_summaries, similarity_threshold=0.9)
        if duplicate_scenes:
            issues.append({
                "type": "possible_duplicate_scenes",
                "severity": "warning",
                "scene_indices": duplicate_scenes,
                "message": f"发现可能重复的场景: 位置{duplicate_scenes}",
                "suggestion": "检查是否需要合并重复场景"
            })

        return issues

    def _find_duplicates(self, items: List) -> List:
        """查找重复项"""
        seen = set()
        duplicates = set()
        for item in items:
            if item in seen:
                duplicates.add(item)
            seen.add(item)
        return list(duplicates)

    def _find_similar_texts(self, texts: List[str], similarity_threshold: float = 0.8) -> list[tuple[int, int]]:
        """查找相似文本"""
        similar_pairs = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if texts[i] and texts[j]:
                    similarity = self._text_similarity(texts[i], texts[j])
                    if similarity > similarity_threshold:
                        similar_pairs.append((i, j))

        return similar_pairs

    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简化版）"""
        if not text1 or not text2:
            return 0.0

        # 使用Jaccard相似度
        words1 = set(re.findall(r'[\u4e00-\u9fa5]+', text1))
        words2 = set(re.findall(r'[\u4e00-\u9fa5]+', text2))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0
