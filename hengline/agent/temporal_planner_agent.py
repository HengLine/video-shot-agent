# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: 时序规划智能体，负责将剧本按5秒粒度切分，估算动作时长
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from pathlib import Path
from typing import List, Dict, Any

from hengline.logger import debug, warning, info
from hengline.prompts.prompts_manager import PromptManager
from hengline.config.temporal_planner_config import get_planner_config
from hengline.tools.action_duration_tool import ActionDurationEstimator


class TemporalPlannerAgent:
    """时序规划智能体"""

    def __init__(self):
        """初始化时序规划智能体"""
        # 初始化PromptManager，使用正确的提示词目录路径
        self.prompt_manager = PromptManager(prompt_dir=Path(__file__).parent.parent)
        
        # 获取配置实例
        self.config = get_planner_config()
        
        # 初始化动作时长估算器
        self.duration_estimator = ActionDurationEstimator(self.config.config_path)
        
        debug(f"时序规划智能体初始化完成，加载了 {len(self.config.base_actions)} 个基础动作配置")

    def plan_timeline(self, structured_script: Dict[str, Any], target_duration: int = None) -> List[Dict[str, Any]]:
        """
        规划剧本的时序分段
        
        Args:
            structured_script: 结构化的剧本
            target_duration: 目标分段时长（秒）
            
        Returns:
            分段计划列表
        """
        debug("开始时序规划")

        if target_duration:
            self.config.target_segment_duration = target_duration

        # 获取提示词模板（供后续扩展使用）
        try:
            self.timeline_planning_template = self.prompt_manager.get_prompt("temporal_planner_prompt")
        except Exception as e:
            debug(f"未找到时序规划提示词模板: {e}")
            # 使用默认处理逻辑

        segments = []
        current_segment = {
            "id": 1,
            "actions": [],
            "est_duration": 0.0,
            "scene_id": 0
        }

        # 遍历所有场景
        scenes = structured_script.get("scenes", [])
        for scene_idx, scene in enumerate(scenes):
            scene_actions = scene.get("actions", [])

            # 确保场景有动作
            if not scene_actions:
                # 如果场景没有动作，创建一个默认动作
                default_action = {
                    "character": "默认角色",
                    "action": "站立",
                    "emotion": "平静"
                }
                scene_actions = [default_action]

            for action in scene_actions:
                action_duration = self._estimate_action_duration(action)

                # 检查是否需要分段
                if current_segment["est_duration"] + action_duration > self.config.target_segment_duration + self.config.max_duration_deviation:
                    # 保存当前分段
                    segments.append(current_segment)

                    # 开始新分段
                    current_segment = {
                        "id": len(segments) + 1,
                        "actions": [],
                        "est_duration": 0.0,
                        "scene_id": scene_idx
                    }

                # 添加动作到当前分段
                current_segment["actions"].append(action)
                current_segment["est_duration"] += action_duration
                current_segment["scene_id"] = scene_idx

        # 添加最后一个分段
        if current_segment["actions"]:
            segments.append(current_segment)
        elif scenes:
            # 如果没有任何动作，但有场景，创建一个默认分段
            warning("未找到任何动作，创建默认分段")
            segments.append({
                "id": 1,
                "actions": [{
                    "character": "默认角色",
                    "action": "站立",
                    "emotion": "平静"
                }],
                "est_duration": self.config.target_segment_duration,
                "scene_id": 0
            })

        # 优化分段
        optimized_segments = self._optimize_segments(segments)

        # 确保至少有一个分段
        if not optimized_segments and scenes:
            warning("分段优化后为空，创建保底分段")
            optimized_segments = [{
                "id": 1,
                "actions": [{
                    "character": "默认角色",
                    "action": "站立",
                    "emotion": "平静"
                }],
                "est_duration": self.config.target_segment_duration,
                "scene_id": 0
            }]

        debug(f"时序规划完成，生成了 {len(optimized_segments)} 个分段")
        return optimized_segments

    def _estimate_action_duration(self, action: Dict[str, Any]) -> float:
        """
        使用ActionDurationEstimator估算单个动作的时长
        
        Args:
            action: 动作字典
            
        Returns:
            估算的时长（秒）
        """
        if not action:
            return 0.0
            
        # 准备参数
        action_text = action.get("action", "")
        dialogue = action.get("dialogue", "")
        
        # 组合动作和对话文本
        combined_text = action_text
        if dialogue:
            # 如果有对话，构造对话描述
            combined_text = f"说：'{dialogue}'"
        
        emotion = action.get("emotion", "")
        character_type = action.get("appearance", {}).get("type", "default")
        
        # 使用动作时长估算器进行估算
        duration = self.duration_estimator.estimate(
            action_text=combined_text,
            emotion=emotion,
            character_type=character_type
        )
        
        # 确保不小于最小动作时长
        duration = max(duration, self.config.min_action_duration)
        debug(f"使用ActionDurationEstimator估算({action_text})的动作时长: {duration:.2f}s")
        return duration
    
    # 以下方法已被ActionDurationEstimator替代
    # 配置现在由 ActionDurationEstimator 管理

    def _optimize_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        优化分段，确保时长合理
        
        Args:
            segments: 原始分段列表
            
        Returns:
            优化后的分段列表
        """
        optimized_segments = []

        for i, segment in enumerate(segments):
            # 检查时长是否过短，最后一个分段允许时长过短
            if segment["est_duration"] < self.config.target_segment_duration * 0.6 and i < len(segments) - 1:
                warning(f"分段 {segment['id']} 时长过短: {segment['est_duration']}秒")
                # 可以考虑合并到前一个分段，但这里暂时保留

            # 检查时长是否过长
            if segment["est_duration"] > self.config.target_segment_duration + self.config.max_duration_deviation:
                warning(f"分段 {segment['id']} 时长过长: {segment['est_duration']}秒")
                # 尝试拆分过长的分段
                split_segments = self._split_long_segment(segment)
                optimized_segments.extend(split_segments)
            else:
                optimized_segments.append(segment)

        # 重新分配ID
        for idx, segment in enumerate(optimized_segments):
            segment["id"] = idx + 1

        return optimized_segments

    def _split_long_segment(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        拆分过长的分段
        
        Args:
            segment: 过长的分段
            
        Returns:
            拆分后的分段列表
        """
        split_segments = []
        current_split = {
            "id": segment["id"],
            "actions": [],
            "est_duration": 0.0,
            "scene_id": segment["scene_id"]
        }

        for action in segment["actions"]:
            # 清理对话字段中的JSON格式残留
            if "dialogue" in action and action["dialogue"]:
                # 移除任何JSON格式的残留文本
                dialogue = action["dialogue"].strip()
                if dialogue.startswith('"""') and dialogue.endswith('"""'):
                    dialogue = dialogue[3:-3].strip()
                # 移除任何看起来像JSON的部分
                if '{' in dialogue and '}' in dialogue:
                    # 这是一个简化的处理，实际项目中可能需要更复杂的逻辑
                    import re
                    dialogue = re.sub(r'\{[^}]*\}', '', dialogue).strip()
                action["dialogue"] = dialogue
            
            action_duration = self._estimate_action_duration(action)

            if current_split["est_duration"] + action_duration > self.config.target_segment_duration:
                # 确保当前拆分不为空才添加
                if current_split["actions"]:
                    split_segments.append(current_split)

                    # 开始新的拆分
                    current_split = {
                        "id": segment["id"],  # ID暂时保留原分段ID
                        "actions": [],
                        "est_duration": 0.0,
                        "scene_id": segment["scene_id"]
                    }

            current_split["actions"].append(action)
            current_split["est_duration"] += action_duration

        # 添加最后一个拆分
        if current_split["actions"]:
            split_segments.append(current_split)

        # 如果拆分失败（没有生成任何分段），返回原始分段的副本
        if not split_segments:
            debug(f"拆分分段 {segment['id']} 失败，返回原始分段")
            return [segment.copy()]

        return split_segments
