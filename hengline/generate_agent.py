"""
@FileName: generate_agent.py
@Description: 
@Author: HengLine
@Time: 2025/10/23 15:51
"""
from typing import Dict, Any, Optional

from hengline.agent import MultiAgentPipeline
from hengline.logger import warning, info, error
from utils.log_utils import print_log_exception


# 对外暴露的主函数
def generate_storyboard(
        script_text: str,
        style: str = "realistic",
        duration_per_shot: int = 5,
        prev_continuity_state: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    剧本分镜生成主接口（可嵌入 LangGraph 或 A2A 调用）

    Args:
        script_text: 用户输入的中文剧本（自然语言）
        style: 视频风格（realistic / anime / cinematic）
        duration_per_shot: 每段目标时长（秒）
        prev_continuity_state: 上一段的 continuity_anchor（用于长剧本续生成）

    Returns:
        包含分镜列表的完整结果
    """
    # 尝试初始化LLM（从配置中获取AI提供商）
    llm = None
    try:
        from config.config import get_ai_config
        from hengline.client.client_factory import ai_client_factory

        # 获取配置
        ai_config = get_ai_config()
        provider = ai_config.get("provider", "openai").lower()
        model = ai_config.get("default_model", "gpt-4o")
        temperature = ai_config.get("temperature", 0.7)

        info(f"使用AI提供商: {provider}, 模型: {model}")

        # 创建完整的LLM配置
        llm_config = {
            'model': model,
            'temperature': temperature,
            **ai_config  # 包含API密钥等配置
        }

        # 使用client_factory获取对应的LangChain LLM实例
        llm = ai_client_factory.get_langchain_llm(provider=provider, config=llm_config)

        if not llm:
            warning(f"AI模型初始化失败（未能获取 {provider} 的LLM实例），系统将自动使用规则引擎模式继续工作")
    except Exception as e:
        print_log_exception()
        error(f"AI模型初始化失败（错误: {str(e)}），系统将自动使用规则引擎模式继续工作")

    # 创建并运行多智能体管道
    pipeline = MultiAgentPipeline(llm=llm)
    return pipeline.run_pipeline(
        script_text=script_text,
        style=style,
        duration_per_shot=duration_per_shot,
        task_id=task_id,
        prev_continuity_state=prev_continuity_state
    )
