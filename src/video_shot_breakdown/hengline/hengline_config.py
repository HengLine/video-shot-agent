"""
@FileName: hengline_config.py
@Description: 
@Author: HengLine
@Github: https://github.com/HengLine/video-shot-agent
@Time: 2026/1/28 12:25
"""
from video_shot_breakdown.hengline.client.client_config import AIConfig


class HengLineConfig(AIConfig):
    """用户请求的参数"""
    prev_continuity_state = None        # 前一个分镜的连续性状态，用于保持连续性
    enable_llm: bool = True    # 开启 LLM 解析，否则使用规则解析
    enable_continuity_check: bool = False   # 开启连续性检查
    # 流程控制
    max_total_loops: int = 30  # 最大总循环次数
    loop_warning_issued: bool = False  # 是否已发出循环警告
    global_loop_exceeded: bool = False  # 全局循环超限标记

    # =====================剧本解析
    use_local_rules: bool = False  # 是否启用本地规则校验和补全

    # ======================镜头拆分
    max_shot_duration: float = 60.0  # 镜头允许的时长范围
    min_shot_duration: float = 1.0
    default_shot_duration: float = 3.0

    # ======================视频分割
    max_fragment_duration: float = 5.5  # 每个分镜的最大持续时间（秒）
    min_fragment_duration: float = 1.0  # 最小片段时长
    split_strategy: str = "simple"  # 简单拆分策略

    # ======================指令转换
    target_model: str = "runway_gen2"
    default_negative_prompt: str = "blurry, distorted, low quality, cartoonish, bad anatomy"
    default_style: str = "cinematic"
    max_prompt_length: int = 1000
    min_prompt_length: int = 10


