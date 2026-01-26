"""
@FileName: generate_agent.py
@Description: 
@Author: HengLine
@Time: 2025/10/23 15:51
"""
from typing import Dict, Optional

from hengline.agent import MultiAgentPipeline
from hengline.client.client_factory import get_default_llm


# 对外暴露的主函数，供 LangGraph 或 A2A 调用
async def generate_storyboard(
        script_text: str,
        duration_per_shot: int = 5,
        task_id: Optional[str] = None,
        llm=None
) -> Dict:
    """
    剧本分镜生成主接口（可嵌入 LangGraph 或 A2A 调用）

    Args:
        script_text: 用户输入的中文剧本（自然语言）
        duration_per_shot: 每段目标时长（秒）
        task_id: 任务ID，用于关联多次调用。 同一个剧本任务ID应该一致
        llm: 可选的LLM实例，如果不提供，将自动初始化（需要配置env参数）

    Returns:
        包含分镜列表的完整结果
    """
    # 创建并运行多智能体管道
    workflow = MultiAgentPipeline(llm=llm if llm else get_default_llm(), task_id=task_id)
    return await  workflow.run_process(
        raw_script=script_text,
        duration_per_shot=duration_per_shot
    )
