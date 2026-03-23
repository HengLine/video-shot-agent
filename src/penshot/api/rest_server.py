"""
@FileName: rest_server.py
@Description: REST API 服务器 - 供非Python智能体通过HTTP调用
@Author: HiPeng
@Time: 2026/3/23 18:54
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel

from penshot.logger import info, error, log_with_context
from penshot.neopen.shot_config import ShotConfig
from penshot.neopen.shot_context import task_id_ctx
from penshot.neopen.shot_language import set_language
from penshot.neopen.task.task_manager import TaskManager
from penshot.neopen.task.task_models import (
    ProcessRequest, ProcessResult, ProcessingStatus,
    BatchProcessRequest, BatchProcessResult
)
from penshot.neopen.task.task_processor import AsyncTaskProcessor
from penshot.utils.log_utils import print_log_exception


# ============================================================================
# 扩展的请求/响应模型
# ============================================================================

class TaskListResponse(BaseModel):
    """任务列表响应"""
    tasks: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int


class CancelTaskResponse(BaseModel):
    """取消任务响应"""
    task_id: str
    status: str
    message: str
    cancelled_at: datetime


# 初始化组件
router = APIRouter(prefix="/api/v1", tags=["Penshot"])
task_manager = TaskManager()
task_processor = AsyncTaskProcessor(task_manager)


# ========================================================================
# 分镜生成接口
# ========================================================================

@router.post("/storyboard", response_model=ProcessResult)
async def generate_storyboard(
        request: ProcessRequest,
        background_tasks: BackgroundTasks
):
    """
    分镜生成接口

    提交剧本进行分镜生成，立即返回任务ID

    - **script**: 剧本文本内容
    - **task_id**: 可选，自定义任务ID
    - **language**: 输出语言 (zh/en)
    - **config**: 可选配置参数
    """
    try:
        log_with_context(
            "INFO",
            "接收到分镜生成请求",
            {
                "task_id": request.task_id,
                "script_length": len(request.script),
                "language": request.language
            }
        )

        # 设置上下文
        if request.task_id:
            task_id_ctx.set(request.task_id)

        # 设置语言
        set_language(request.language)

        # 创建任务
        task_id = task_manager.create_task(
            script=request.script,
            config=request.config,
            task_id=request.task_id
        )

        # 后台异步处理
        background_tasks.add_task(
            task_processor.process_script_task,
            task_id
        )

        info(f"任务创建成功: {task_id}")

        return ProcessResult(
            success=True,
            task_id=task_id,
            status="pending",
            message="任务已提交，请使用任务ID查询状态",
            created_at=datetime.now(timezone.utc)
        )

    except ValueError as e:
        print_log_exception()
        error(f"参数错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        print_log_exception()
        error(f"分镜生成失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"内部服务器错误: {str(e)}"
        )


@router.post("/storyboard/sync", response_model=ProcessResult)
async def generate_storyboard_sync(
        request: ProcessRequest,
        timeout: float = 300
):
    """
    同步分镜生成接口

    等待任务完成后返回结果

    - **script**: 剧本文本内容
    - **timeout**: 等待超时时间（秒）
    """
    try:
        log_with_context(
            "INFO",
            "接收到同步分镜生成请求",
            {"script_length": len(request.script)}
        )

        set_language(request.language)

        # 创建任务
        task_id = task_manager.create_task(
            script=request.script,
            config=request.config,
            task_id=request.task_id
        )

        info(f"任务创建成功: {task_id}")

        # 等待任务完成
        import asyncio
        start_time = asyncio.get_event_loop().time()
        poll_interval = 0.5

        while True:
            task = task_manager.get_task(task_id)

            if not task:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"任务不存在: {task_id}"
                )

            task_status = task.get("status")

            if task_status == "completed":
                # 计算处理时间
                processing_time = None
                if task.get("completed_at") and task.get("created_at"):
                    try:
                        completed_at = datetime.fromisoformat(task["completed_at"]) if isinstance(task["completed_at"], str) else task["completed_at"]
                        created_at = datetime.fromisoformat(task["created_at"]) if isinstance(task["created_at"], str) else task["created_at"]
                        processing_time = int((completed_at - created_at).total_seconds() * 1000)
                    except Exception:
                        pass

                return ProcessResult(
                    task_id=task_id,
                    success=True,
                    status="success",
                    data=task.get("result", {}).get("data"),
                    processing_time_ms=processing_time,
                    created_at=task.get("created_at"),
                    completed_at=task.get("completed_at")
                )

            if task_status == "failed":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=task.get("error", "任务处理失败")
                )

            # 检查超时
            if asyncio.get_event_loop().time() - start_time > timeout:
                # 任务仍在处理，标记为取消
                task_manager.fail_task(task_id, "同步等待超时")
                raise HTTPException(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    detail=f"等待超时 ({timeout}秒)"
                )

            await asyncio.sleep(poll_interval)

    except HTTPException:
        raise
    except Exception as e:
        print_log_exception()
        error(f"同步分镜生成失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ========================================================================
# 批量处理接口
# ========================================================================

@router.post("/storyboard/batch", response_model=BatchProcessResult)
async def batch_process_scripts(
        request: BatchProcessRequest,
        background_tasks: BackgroundTasks
):
    """
    批量处理多个剧本

    - **scripts**: 剧本列表（最多50个）
    - **batch_id**: 可选，自定义批量ID
    - **config**: 统一配置（可选）
    - **language**: 输出语言
    """
    try:
        batch_id = request.batch_id or str(uuid.uuid4())

        log_with_context(
            "INFO",
            "接收到批量处理请求",
            {
                "batch_id": batch_id,
                "script_count": len(request.scripts)
            }
        )

        set_language(request.language)

        # 限制批量大小
        max_batch_size = 50
        if len(request.scripts) > max_batch_size:
            raise ValueError(f"批量处理最多支持 {max_batch_size} 个剧本")

        # 后台异步处理批量任务
        background_tasks.add_task(
            task_processor.process_batch,
            batch_id,
            request.scripts,
            request.config
        )

        return BatchProcessResult(
            batch_id=batch_id,
            total_tasks=len(request.scripts),
            completed_tasks=0,
            failed_tasks=0,
            pending_tasks=len(request.scripts),
            created_at=datetime.now(timezone.utc)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        print_log_exception()
        error(f"批量处理创建失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量处理创建失败: {str(e)}"
        )


@router.get("/storyboard/batch/{batch_id}", response_model=BatchProcessResult)
async def get_batch_status(batch_id: str):
    """
    获取批量任务状态

    - **batch_id**: 批量任务ID
    """
    # 获取该批次下所有任务
    # 注意：这里简化实现，实际需要从 task_manager 获取批次信息
    tasks = task_manager.get_tasks_by_batch(batch_id) if hasattr(task_manager, 'get_tasks_by_batch') else []

    completed = sum(1 for t in tasks if t.get("status") == "completed")
    failed = sum(1 for t in tasks if t.get("status") == "failed")
    pending = len(tasks) - completed - failed

    return BatchProcessResult(
        batch_id=batch_id,
        total_tasks=len(tasks),
        completed_tasks=completed,
        failed_tasks=failed,
        pending_tasks=pending,
        created_at=datetime.now(timezone.utc)
    )


# ========================================================================
# 任务管理接口
# ========================================================================

@router.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_task_status(task_id: str):
    """
    获取任务状态

    - **task_id**: 任务ID
    """
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {task_id}"
        )

    # 解析时间
    try:
        created_at = datetime.fromisoformat(task["created_at"]) if isinstance(task["created_at"], str) else task["created_at"]
    except Exception:
        created_at = datetime.now(timezone.utc)

    try:
        updated_at = datetime.fromisoformat(task["updated_at"]) if isinstance(task["updated_at"], str) else task["updated_at"]
    except Exception:
        updated_at = datetime.now(timezone.utc)

    # 计算预估剩余时间
    estimated_time = None
    if task["status"] == "processing" and task.get("progress") and task["progress"] > 0:
        try:
            elapsed = (datetime.now(timezone.utc) - created_at).total_seconds()
            estimated_time = int((elapsed / task["progress"]) * (100 - task["progress"]))
        except Exception:
            pass

    return ProcessingStatus(
        task_id=task_id,
        status=task["status"],
        stage=task.get("stage", "unknown"),
        progress=task.get("progress"),
        estimated_time_remaining=estimated_time,
        created_at=created_at,
        updated_at=updated_at,
        error_message=task.get("error")
    )


@router.get("/result/{task_id}", response_model=ProcessResult)
async def get_task_result(task_id: str):
    """
    获取任务结果

    - **task_id**: 任务ID
    """
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {task_id}"
        )

    # 任务处理中
    if task["status"] in ["pending", "processing"]:
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"任务仍在处理中，当前状态: {task['status']}"
        )

    # 解析时间并计算处理时间
    processing_time = None
    try:
        if task.get("completed_at"):
            completed_at = datetime.fromisoformat(task["completed_at"]) if isinstance(task["completed_at"], str) else task["completed_at"]
            created_at = datetime.fromisoformat(task["created_at"]) if isinstance(task["created_at"], str) else task["created_at"]
            processing_time = int((completed_at - created_at).total_seconds() * 1000)
    except Exception:
        pass

    return ProcessResult(
        task_id=task_id,
        success=task["status"] == "completed",
        status="success" if task["status"] == "completed" else "failed",
        data=task.get("result", {}).get("data") if task.get("result") else None,
        message=task.get("error"),
        processing_time_ms=processing_time,
        created_at=task.get("created_at"),
        completed_at=task.get("completed_at")
    )


@router.delete("/task/{task_id}", response_model=CancelTaskResponse)
async def cancel_task(task_id: str):
    """
    取消任务

    - **task_id**: 任务ID
    """
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {task_id}"
        )

    if task["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"任务已结束，无法取消: {task['status']}"
        )

    # 标记为取消
    task_manager.fail_task(task_id, "任务被用户取消")

    return CancelTaskResponse(
        task_id=task_id,
        status="cancelled",
        message="任务已取消",
        cancelled_at=datetime.now(timezone.utc)
    )


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
        status_filter: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
):
    """
    列出任务列表

    - **status_filter**: 可选，按状态筛选 (pending/processing/completed/failed)
    - **limit**: 返回数量限制
    - **offset**: 偏移量
    """
    # 注意：这里需要根据实际 task_manager 实现
    # 简化实现，返回空列表
    tasks = []

    return TaskListResponse(
        tasks=tasks,
        total=len(tasks),
        page=offset // limit + 1 if limit > 0 else 1,
        page_size=limit
    )


# ========================================================================
# 配置接口
# ========================================================================

@router.get("/config", response_model=ShotConfig)
async def get_default_config():
    """获取默认配置"""
    return ShotConfig()


@router.get("/languages")
async def get_supported_languages():
    """获取支持的语言列表"""
    return {
        "languages": [
            {"code": "zh", "name": "中文"},
            {"code": "en", "name": "English"}
        ],
        "default": "zh"
    }
