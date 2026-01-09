"""
@FileName: workflow_models.py
@Description:  工作流模型定义模块
@Author: HengLine
@Time: 2025/11/30 19:12
"""
import json
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any


@unique
class AgentType(Enum):
    PARSER = "parser"  # 智能体1：剧本解析
    PLANNER = "planner"  # 智能体2：时序规划
    CONTINUITY = "continuity"  # 智能体3：连贯性
    VISUAL = "visual"  # 智能体4：视觉生成
    REVIEWER = "reviewer"  # 智能体5：质量审查


@unique
class ScriptType(Enum):
    NATURAL_LANGUAGE = "natural_language"  # 自然语言描述
    STRUCTURED_SCENE = "structured_scene"  # 结构化分场剧本
    AI_STORYBOARD = "ai_storyboard"  # AI生成的分镜剧本
    SCREENPLAY_FORMAT = "screenplay_format"  # 标准剧本格式


@unique
class ParserType(Enum):
    LLM_PARSER = "llm_parser"  # LLM 解析器
    RULE_PARSER = "rule_parser"  # 本地规则解析器


@unique
class VideoStyle(Enum):
    # 逼真
    REALISTIC = 'realistic'
    # 动漫
    ANIME = 'anime'
    # 电影
    CINEMATIC = 'cinematic'
    # 卡通
    CARTOON = 'cartoon'


@dataclass
class AIConfig:
    """AI配置"""
    model: str = "gpt-4"  # 或 "claude-3", "deepseek-chat"
    temperature: float = 0.2
    max_tokens: int = 4000
    system_prompt: str = ""
    json_mode: bool = True  # 强制JSON输出


@dataclass
class AIBaseAgent:
    """AI智能体基类"""
    agent_type: AgentType
    config: AIConfig
    name: str = ""
    description: str = ""

    def __post_init__(self):
        self.name = self.name or f"{self.agent_type.value}_agent"
        self.description = self.description or f"{self.agent_type.value.capitalize()} Agent"

    def call_ai(self, prompt: str, system_prompt: str = "") -> str:
        """调用AI的通用方法（简化版）"""
        # 这里可以替换为实际的AI API调用
        # 返回AI的响应文本
        return self._mock_ai_call(prompt, system_prompt or self.config.system_prompt)


    def process(self, input_data: Any) -> Any:
        """处理输入数据（子类实现）"""
        raise NotImplementedError("子类必须实现process方法")



    def _mock_ai_call(self, prompt: str, system_prompt: str) -> str:
        """模拟AI调用（实际使用时替换为真实API）"""
        # 根据agent_type返回不同的模拟响应
        if self.agent_type == AgentType.PARSER:
            return self._mock_parser_response()
        elif self.agent_type == AgentType.PLANNER:
            return self._mock_planner_response()
        elif self.agent_type == AgentType.CONTINUITY:
            return self._mock_continuity_response()
        elif self.agent_type == AgentType.VISUAL:
            return self._mock_visual_response()
        elif self.agent_type == AgentType.REVIEWER:
            return self._mock_reviewer_response()

        return '{"status": "success", "message": "AI processing completed"}'

    def _mock_parser_response(self) -> str:
        return json.dumps({
            "format_type": "standard_script",
            "scenes": [
                {
                    "scene_id": "s1",
                    "description": "林然坐在客厅沙发上，手中拿着咖啡杯",
                    "characters": ["林然"],
                    "location": "客厅"
                }
            ],
            "characters": [
                {
                    "name": "林然",
                    "description": "年轻男性，穿着休闲服装"
                }
            ],
            "dialogues": [
                {
                    "speaker": "林然",
                    "content": "你好吗？",
                    "emotional_tone": "neutral"
                }
            ]
        }, ensure_ascii=False)

    def _mock_planner_response(self) -> str:
        return json.dumps({
            "timeline_segments": [
                {
                    "segment_id": "s001",
                    "time_range": [0.0, 5.0],
                    "duration": 5.0,
                    "content": "林然坐在沙发上，拿着咖啡杯",
                    "characters": ["林然"],
                    "estimated_duration": 5.0
                }
            ],
            "pacing_analysis": "normal",
            "total_duration": 5.0
        }, ensure_ascii=False)

    def _mock_continuity_response(self) -> str:
        return json.dumps({
            "anchored_segments": [
                {
                    "segment_id": "s001",
                    "start_constraint": "林然坐在沙发左侧，手持咖啡杯",
                    "end_constraint": "林然保持坐姿，咖啡杯在手中",
                    "continuity_hooks": ["coffee_cup_position", "sitting_posture"]
                }
            ],
            "continuity_rules": [
                "咖啡杯必须始终在画面中",
                "林然的服装保持一致"
            ]
        }, ensure_ascii=False)

    def _mock_visual_response(self) -> str:
        return json.dumps({
            "shots": [
                {
                    "shot_id": "s001_shot1",
                    "time_range": [0.0, 5.0],
                    "prompt": "A young man (Lin Ran) sitting on a modern gray sofa, holding a white ceramic coffee cup in his right hand. Soft natural lighting from window, cinematic shallow depth of field, warm color grading. Medium close-up shot, slow subtle push-in.",
                    "camera": "medium_close_up",
                    "movement": "slow_push_in"
                }
            ],
            "style_guide": "cinematic natural lighting"
        }, ensure_ascii=False)

    def _mock_reviewer_response(self) -> str:
        return json.dumps({
            "decision": "approved",
            "score": 0.85,
            "issues": [],
            "suggestions": ["可以考虑添加更多镜头变化"],
            "can_proceed": True
        }, ensure_ascii=False)
