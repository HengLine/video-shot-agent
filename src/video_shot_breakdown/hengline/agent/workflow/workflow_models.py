"""
@FileName: workflow_models.py
@Description: 工作流模型定义文件，包含工作流状态和条件的枚举类
@Author: HengLine
@Github: https://github.com/HengLine/video-shot-agent
@Time: 2026/1/27 19:12
"""
from enum import Enum, unique


@unique
class AgentStage(str, Enum):
    """智能体工作流阶段枚举类，定义工作流的不同阶段状态"""
    START = "loaded"  # 开始状态
    INIT = "initialized"  # 初始化状态
    PARSER = "parsed"  # 剧本解析完成
    SEGMENTER = "segmented"  # 分镜拆分完成
    SPLITTER = "split"  # 片段分隔完成
    CONVERTER = "converted"  # 指令转换完成
    AUDITOR = "audited"  # 质量审查完成
    CONTINUITY = "continuity_check"  # 连续性检查完成
    ERROR = "error_handling"  # 错误处理中
    HUMAN = "human_intervention"  # 人工干预中
    END = "completed"  # 工作流完成


class ConditionalEdges(str, Enum):
    """工作流条件分支枚举类，定义工作流中的条件分支"""
    PARSE_SCRIPT = "parse_script"  # 剧本解析分支
    SEGMENT_SHOT = "segment_shot"  # 分镜拆分分支
    SPLIT_VIDEO = "split_video"  # 片段分隔分支
    CONVERT_PROMPT = "convert_prompt"  # 指令转换分支
    AUDIT_QUALITY = "audit_quality"  # 质量审查分支


from enum import Enum, unique


# ============================================================================
# 决策状态枚举 (DecisionState)
# ============================================================================
@unique
class DecisionState(str, Enum):
    """
    决策状态枚举 - 表示单个操作或检查的结果

    说明:
    - 用于决策函数的返回值
    - 表示操作的成功、失败、需要调整等状态
    - 是工作流分支决策的依据

    设计原则:
    1. 语义清晰: 每个状态都有明确的业务含义
    2. 正交性: 状态之间互斥，不重叠
    3. 可操作性: 每个状态都能映射到具体的下一步操作
    """

    # ========== 成功状态 (1xx) 表示操作完全成功，可以继续下一步流程==========
    SUCCESS = "success"
    """
    完全成功
    含义: 操作执行成功，所有要求都满足，可以直接进入下一阶段
    示例: 剧本解析成功、质量审查通过
    下一步: 进入流程中的下一个常规节点
    """

    VALID = "valid"
    """
    验证通过
    含义: 经过验证确认符合要求，但可能有一些小问题或警告
    示例: 质量审查有轻微警告但可以通过、连续性检查有可接受偏差
    下一步: 可以继续，但可能需要记录警告信息
    """

    # ========== 需要调整状态 (2xx) 表示操作需要调整或修复，但无需重试整个操作==========
    NEEDS_ADJUSTMENT = "needs_adjustment"
    """
    需要调整
    含义: 结果基本可用，但需要小范围调整优化
    示例: 提示词质量可以但需要优化、片段时长稍微超限
    下一步: 返回调整节点进行优化，不重启整个流程
    """

    NEEDS_FIX = "needs_fix"
    """
    需要修复
    含义: 存在必须修复的问题，但问题范围有限
    示例: 有个别片段超时、部分连续性错误
    下一步: 返回特定修复节点处理，针对性修复
    """

    NEEDS_OPTIMIZATION = "needs_optimization"
    """
    需要优化
    含义: 功能正常，但质量或效果有待提升
    示例: 提示词可以生成视频但质量不高
    下一步: 进入优化流程，提升输出质量
    """

    # ========== 重试状态 (3xx) 表示操作需要重新执行，可能因为临时问题或配置问题==========
    SHOULD_RETRY = "should_retry"
    """
    应该重试
    含义: 操作失败，但可能通过重试解决（如网络问题、API限制）
    示例: AI模型调用失败、临时资源不足
    下一步: 重新执行当前节点，可能增加重试计数
    """

    RETRY_WITH_ADJUSTMENT = "retry_with_adjustment"
    """
    调整后重试
    含义: 操作失败，需要调整参数后重新尝试
    示例: 镜头拆分结果不合理、分段策略需要调整
    下一步: 调整配置参数后重新执行当前节点
    """

    # ========== 人工干预状态 (4xx) 表示需要人工介入进行判断或处理==========
    REQUIRE_HUMAN = "require_human"
    """
    需要人工干预
    含义: 系统无法自动处理，需要人工判断或操作
    示例: 剧本内容有歧义、质量审查发现严重但不确定的问题
    下一步: 暂停流程，等待人工输入
    """

    ESCALATE = "escalate"
    """
    需要升级处理
    含义: 问题超出当前处理能力，需要更高层级的处理
    示例: 发现系统性错误、多次重试仍然失败
    下一步: 暂停流程，记录问题，可能需要管理员介入
    """

    # ========== 失败状态 (5xx) 表示操作完全失败，流程可能需要终止==========
    FAILED = "failed"
    """
    决策失败
    含义: 操作执行失败，但可能是可预期的业务失败
    示例: 输入格式不支持、资源权限不足
    下一步: 进入错误处理流程，记录错误原因
    """

    CRITICAL_FAILURE = "critical_failure"
    """
    严重失败
    含义: 遇到无法处理的严重错误，流程无法继续
    示例: 数据损坏、系统错误、配置错误
    下一步: 立即停止流程，记录错误日志
    """

    ABORT_PROCESS = "abort_process"
    """
    中止流程
    含义: 主动中止整个工作流，通常是人工干预的结果
    示例: 用户手动取消、超过时间限制
    下一步: 清理资源，结束工作流
    """


# ============================================================================
# 工作流阶段枚举 (PipelineState)
# ============================================================================
@unique
class PipelineState(str, Enum):
    """
    工作流阶段枚举 - 表示工作流中的处理节点或阶段

    说明:
    - 用于标识工作流图中的节点
    - 每个节点对应一个具体的处理功能
    - 决策状态会映射到这些节点

    分类说明:
    A. 核心处理阶段: 视频生成的主流程
    B. 质量控制阶段: 质量审查和检查
    C. 调整修复阶段: 针对问题的修复流程
    D. 特殊处理阶段: 人工干预和错误处理
    E. 流程状态: 流程的开始和结束
    """

    # ========== 核心处理阶段 (A系列) 视频生成的主流程，按顺序执行==========
    PARSE_SCRIPT = "parse_script"
    """
    解析剧本
    功能: 将原始剧本文本解析为结构化数据
    输入: 原始剧本文本
    输出: ParsedScript对象
    依赖: 无
    """

    SPLIT_SHOTS = "split_shots"
    """
    拆分镜头
    功能: 将剧本场景拆分为具体的镜头
    输入: ParsedScript对象
    输出: ShotSequence对象
    依赖: PARSE_SCRIPT
    """

    CUT_FRAGMENTS = "cut_fragments"
    """
    切割片段
    功能: 将镜头切割为适合AI生成的短视频片段
    输入: ShotSequence对象
    输出: FragmentSequence对象
    依赖: SPLIT_SHOTS
    """

    GENERATE_PROMPTS = "generate_prompts"
    """
    生成提示词
    功能: 为每个视频片段生成AI提示词
    输入: FragmentSequence对象
    输出: AIVideoInstructions对象
    依赖: CUT_FRAGMENTS
    """

    GENERATE_VIDEO = "generate_video"
    """
    生成视频
    功能: 使用AI模型生成视频片段
    输入: AIVideoInstructions对象
    输出: 视频文件或URL
    依赖: GENERATE_PROMPTS, AUDIT_QUALITY
    """

    # ========== 质量控制阶段 (B系列) 质量检查和审查，确保输出质量==========
    CHECK_CONTINUITY = "check_continuity"
    """
    检查连续性
    功能: 检查视频片段之间的连续性（角色、场景、道具等）
    输入: AIVideoInstructions对象或相关数据
    输出: ContinuityReport对象
    依赖: GENERATE_PROMPTS
    """

    AUDIT_QUALITY = "audit_quality"
    """
    审计质量
    功能: 全面审查生成内容的质量
    输入: AIVideoInstructions对象
    输出: QualityAuditReport对象
    依赖: GENERATE_PROMPTS
    """

    # ========== 调整修复阶段 (C系列) 针对发现的问题进行修复和优化 ==========
    ADJUST_PROMPTS = "adjust_prompts"
    """
    调整提示词
    功能: 根据质量审查结果调整AI提示词
    输入: AIVideoInstructions对象, QualityAuditReport
    输出: 优化后的AIVideoInstructions对象
    依赖: AUDIT_QUALITY, CHECK_CONTINUITY
    """

    REGENERATE_FRAGMENTS = "regenerate_fragments"
    """
    重新生成片段
    功能: 重新切割视频片段（如时长问题）
    输入: ShotSequence对象
    输出: 新的FragmentSequence对象
    依赖: SPLIT_SHOTS
    """

    REGENERATE_SHOTS = "regenerate_shots"
    """
    重新生成镜头
    功能: 重新拆分镜头（如结构问题）
    输入: ParsedScript对象
    输出: 新的ShotSequence对象
    依赖: PARSE_SCRIPT
    """

    # ========== 特殊处理阶段 (D系列) ==========
    # 人工干预和错误处理
    HUMAN_INTERVENTION = "human_intervention"
    """
    人工干预
    功能: 等待人工输入或决策
    输入: 当前状态、问题描述
    输出: 人工决策结果
    依赖: 任意阶段（当需要人工时）
    """

    ERROR_HANDLING = "error_handling"
    """
    错误处理
    功能: 处理系统错误和异常
    输入: 错误信息、当前状态
    输出: 错误处理结果（重试、跳过、中止等）
    依赖: 任意阶段（当发生错误时）
    """

    # ========== 流程状态 (E系列) ==========
    # 工作流的开始和结束状态
    WORKFLOW_START = "workflow_start"
    """
    工作流开始
    功能: 工作流的起始点
    输入: 用户输入（剧本、配置）
    输出: 初始工作流状态
    依赖: 无
    """

    WORKFLOW_COMPLETE = "workflow_complete"
    """
    工作流完成
    功能: 工作流的结束点
    输入: 最终输出结果
    输出: 无
    依赖: GENERATE_VIDEO 或 中止流程
    """

    WORKFLOW_PAUSED = "workflow_paused"
    """
    工作流暂停
    功能: 临时暂停工作流
    输入: 当前状态
    输出: 无
    依赖: 任意阶段（当需要暂停时）
    """

@unique
class PipelineNode(str, Enum):
    """工作流管道节点枚举类，定义工作流管道中的节点"""
    PARSE_SCRIPT = "parse_script"  # 剧本解析节点
    SEGMENT_SHOT = "segment_shot"  # 分镜拆分节点
    SPLIT_VIDEO = "split_video"  # 片段分隔节点
    CONVERT_PROMPT = "convert_prompt"  # 指令转换节点
    AUDIT_QUALITY = "audit_quality"  # 质量审查节点
    CONTINUITY_CHECK = "continuity_check"  # 连续性检查节点
    HUMAN_INTERVENTION = "human_intervention"  # 人工干预节点
    ERROR_HANDLER = "error_handler"  # 错误处理节点
    GENERATE_OUTPUT = "generate_output"  # 生成输出节点
    END = "end"  # 结束节点