# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: 时序规划，负责将剧本按5秒粒度切分，估算动作时长
@Author: HengLine
@Time: 2025/10 - 2025/12
"""
from typing import List, Dict, Any

from hengline.config.temporal_planner_config import get_planner_config
from hengline.logger import debug
from hengline.tools.action_duration_tool import ActionDurationEstimator
from hengline.tools.langchain_memory_tool import LangChainMemoryTool


class TemporalPlanner:
    """时序规划"""

    def __init__(self):
        """初始化时序规划智能体"""
        # 获取配置实例
        self.config = get_planner_config()

        # 初始化动作时长估算器
        self.duration_estimator = ActionDurationEstimator(self.config.config_path)

        # 初始化LangChain记忆工具（替代原有的向量记忆+状态机）
        self.memory_tool = LangChainMemoryTool()

    def plan_timeline(self, structured_script: Dict[str, Any], target_duration: int = 5) -> List[Dict[str, Any]] | None:
        """
        规划剧本的时序分段
        
        Args:
            structured_script: 结构化的剧本
            target_duration: 目标分段时长（秒）
            
        Returns:
            分段计划列表
        """
        pass

    def _store_action_state(self, action: Dict[str, Any], scene_idx: int) -> None:
        """
        存储动作状态到LangChain记忆中

        Args:
            action: 动作信息
            scene_idx: 场景索引
        """
        try:
            # 构建完整的状态信息
            state = {
                "action": action.get("action", ""),
                "emotion": action.get("emotion", ""),
                "character": action.get("character", ""),
                "scene_id": scene_idx,
                "timestamp": len(self.memory_tool.retrieve_similar_states("")) + 1  # 简化的时间戳
            }

            # 添加上下文信息
            context = f"场景 {scene_idx} 中的动作"

            # 存储到记忆中
            self.memory_tool.store_state(state, context)
        except Exception as e:
            debug(f"存储动作状态失败: {e}")
