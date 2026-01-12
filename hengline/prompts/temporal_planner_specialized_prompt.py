"""
@FileName: _specialized_prompt.py
@Description: 
@Author: HengLine
@Time: 2026/1/12 18:32
"""
from typing import List, Dict, Any

from hengline.prompts.prompts_manager import prompt_manager
from hengline.prompts.temporal_planner_prompt import DurationPromptTemplates, PromptConfig


class SpecializedPromptTemplates(DurationPromptTemplates):
    """专业化的提示词模板（支持特定类型扩展）"""

    def __init__(self, config: PromptConfig = None):
        super().__init__(config)
        self.specialized_templates = self._load_specialized_templates()

    def _load_specialized_templates(self) -> Dict[str, Any]:
        """加载专业化模板"""
        # 这里可以加载额外的专业化模板文件
        # 例如：emotional_dialogue_prompts.yaml, action_sequence_prompts.yaml
        return {}

    def emotional_dialogue_prompt(self, dialogue_data: Dict, intensity: str = "medium") -> str:
        """情感化对话提示词（针对高强度情感场景）"""
        base_prompt = self.dialogue_duration_prompt(dialogue_data)

        # 添加情感强度特定指导
        intensity_guides = {
            "high": """
            ### 高强度情感指导
            这是情感高潮时刻，请特别注意：
            1. **情感释放时间**：强烈情感需要充分的时间表达
            2. **观众共鸣时间**：给观众时间感受和共鸣
            3. **视觉强调时间**：特写镜头需要足够时间展示微表情变化
            4. **情感回落时间**：强烈情感后的平静需要过渡时间

            建议情感高潮时刻比普通对话多50-100%的时间。
            """,
            "medium": """
            ### 中等强度情感指导
            这是情感发展时刻，请考虑：
            1. **情感建立时间**：情感需要时间逐渐展现
            2. **内心冲突时间**：矛盾情感需要表现时间
            3. **关系发展时间**：角色关系变化需要观众理解时间
            """,
            "low": """
            ### 低强度情感指导
            这是情感铺垫时刻，保持自然节奏：
            1. **情感暗示时间**：细微情感需要细心观察
            2. **氛围营造时间**：为后续情感高潮铺垫
            3. **角色塑造时间**：通过对话建立角色性格
            """
        }

        if intensity in intensity_guides:
            base_prompt += "\n\n" + intensity_guides[intensity]

        return base_prompt

    def action_sequence_prompt(self, actions: List[Dict], sequence_type: str = "continuous") -> str:
        """动作序列提示词（针对连续动作）"""
        if len(actions) == 1:
            return self.action_duration_prompt(actions[0])

        prompt = prompt_manager.get_name_prompt("action_sequence_planner")

        sequence_guides = {
            "continuous": """
                ### 连续动作序列分析

                你收到一个连续动作序列，包含{count}个相互关联的动作。
                请分析：
                1. **动作流畅性**：动作之间如何自然过渡？
                2. **节奏变化**：序列中是否有节奏变化点？
                3. **高潮时刻**：哪个动作是序列的情感/动作高潮？
                4. **视觉连贯性**：如何保证视觉上的流畅性？

                请为每个动作提供时长估算，并说明动作间的时间关系。
                """,
            "parallel": """
                ### 并行动作序列分析

                你收到一个并行动作序列，包含{count}个同时或交替发生的动作。
                请分析：
                1. **时间同步**：这些动作在时间上如何同步？
                2. **焦点转移**：视觉焦点如何在动作间转移？
                3. **节奏协调**：如何协调不同动作的节奏？
                4. **整体时长**：并行动作的总时长如何确定？

                请考虑动作的并行性和交互性。
                """,
            "reaction": """
                ### 反应动作序列分析

                你收到一个反应动作序列，包含{count}个因果相关的动作。
                请分析：
                1. **刺激-反应时间**：每个反应需要多少处理时间？
                2. **情感发展**：情感如何在反应中演变？
                3. **连锁反应**：一个动作如何引发下一个动作？
                4. **累积效果**：反应序列如何构建紧张感或情感？

                请特别注意反应的自然时间和情感真实性。
                """
        }

        guide = sequence_guides.get(sequence_type, sequence_guides["continuous"])
        prompt += guide.format(count=len(actions))

        # 添加动作列表
        prompt += "\n\n### 动作序列详情\n"
        for i, action in enumerate(actions):
            action_id = action.get("action_id", f"action_{i}")
            actor = action.get("actor", "未指定")
            description = self.components.truncate_text(action.get("description", ""), 60)
            prompt += f"{i + 1}. [{action_id}] {actor}：{description}\n"

        # 输出格式要求
        prompt += "\n\n### 输出要求\n"
        prompt += """请输出以下格式：
            {
              "sequence_analysis": "序列整体分析",
              "total_duration": 总时长,
              "individual_estimations": [
                {
                  "action_id": "动作ID",
                  "estimated_duration": 时长,
                  "position_in_sequence": 位置,
                  "relationship_to_previous": "与前一个动作的关系"
                }
              ],
              "pacing_curve": "节奏曲线描述",
              "key_transitions": ["关键过渡点"]
            }"""

        return prompt
