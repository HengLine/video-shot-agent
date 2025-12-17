# -*- coding: utf-8 -*-
"""
@FileName: temporal_planner_agent.py
@Description: LLM + 规则约束实现的时序规划（负责将剧本按5秒粒度切分，估算动作时长，使用LangChain实现状态记忆）
@Author: HengLine
@Time: 2025/10 - 2025/11
"""
from typing import List, Dict, Any

from hengline.agent.temporal_planner.base_temporal_planner import TemporalPlanner
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

    def __init__(self):
        """初始化时序规划智能体"""
        debug(f"LLM + 规则约束的时序规划，加载了 {len(self.config.base_actions)} 个基础动作配置")

    def plan_timeline(self, structured_script: Dict[str, Any], target_duration: int = 5) -> List[Dict[str, Any]] | None:
        """
        规划剧本的时序分段
        
        Args:
            structured_script: 结构化的剧本
            target_duration: 目标分段时长（秒）
            
        Returns:
            分段计划列表
        """

        # 获取提示词模板（供后续扩展使用）
        try:
            timeline_planning_template = prompt_manager.get_temporal_planner_prompt()

            return None
        except Exception as e:
            error(f"未找到时序规划提示词模板: {e}")
            return None
