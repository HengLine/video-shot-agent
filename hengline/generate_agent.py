"""
@FileName: generate_agent.py
@Description: 
@Author: HengLine
@Time: 2025/10/23 15:51
"""
from typing import Dict, Any, Optional

from hengline.agent import MultiAgentPipeline
from hengline.agent.workflow.workflow_models import VideoStyle
from hengline.client.client_factory import get_default_llm


# 对外暴露的主函数，供 LangGraph 或 A2A 调用
def generate_storyboard(
        script_text: str,
        style: VideoStyle = VideoStyle.REALISTIC,
        duration_per_shot: int = 5,
        prev_continuity_state: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        llm=None
) -> Dict[str, Any]:
    """
    剧本分镜生成主接口（可嵌入 LangGraph 或 A2A 调用）

    Args:
        script_text: 用户输入的中文剧本（自然语言）
        style: 视频风格（realistic / anime / cinematic）
        duration_per_shot: 每段目标时长（秒）
        prev_continuity_state: 上一段的 continuity_anchor（用于长剧本续生成）
        task_id: 任务ID，用于关联多次调用。 同一个剧本任务ID应该一致
        llm: 可选的LLM实例，如果不提供，将自动初始化（需要配置env参数）

    Returns:
        包含分镜列表的完整结果
    """
    # 创建并运行多智能体管道
    pipeline = MultiAgentPipeline(llm=llm if llm else get_default_llm(), task_id=task_id)
    return pipeline.run_pipeline(
        script_text=script_text,
        style=style,
        duration_per_shot=duration_per_shot,
        task_id=task_id,
        prev_continuity_state=prev_continuity_state
    )
