"""
@FileName: function_calls.py
@Description: Function Call接口 - 供其他Python智能体调用
@Author: HiPeng
@Time: 2026/3/23 18:39
"""

import asyncio
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List, Any, Callable

from penshot.logger import log_with_context
from penshot.neopen.shot_config import ShotConfig
from penshot.neopen.shot_language import Language, set_language
from penshot.neopen.task.task_manager import TaskManager
from penshot.neopen.task.task_processor import AsyncTaskProcessor


@dataclass
class PenshotResult:
    """Penshot 执行结果"""
    task_id: str
    success: bool
    status: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[int] = None


class PenshotFunction:
    """
    Penshot 智能体功能调用接口

    复用现有的 TaskManager 进行任务管理
    """

    def __init__(
            self,
            config: Optional[ShotConfig] = None,
            language: Language = Language.ZH,
            task_manager: Optional[TaskManager] = None
    ):
        """
        初始化 Penshot 功能接口

        Args:
            config: 系统配置
            language: 输出语言
            task_manager: 任务管理器（可选）
        """
        self.config = config or ShotConfig()
        self.language = language

        # 复用任务管理器
        self.task_manager = task_manager or TaskManager()
        self.task_processor = AsyncTaskProcessor(self.task_manager)

        # 回调存储
        self._callbacks: Dict[str, Callable] = {}

        # 后台任务事件循环
        self._background_loop: Optional[asyncio.AbstractEventLoop] = None
        self._background_thread: Optional[threading.Thread] = None
        self._start_background_loop()

    def _start_background_loop(self):
        """启动后台事件循环"""

        def run_loop():
            self._background_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._background_loop)
            self._background_loop.run_forever()

        self._background_thread = threading.Thread(target=run_loop, daemon=True)
        self._background_thread.start()

        # 等待循环启动
        while self._background_loop is None:
            pass

    def _run_async_in_background(self, coro):
        """
        在后台事件循环中运行协程

        使用 run_coroutine_threadsafe 是正确的方式
        """
        if self._background_loop is None:
            raise RuntimeError("后台事件循环未启动")

        future = asyncio.run_coroutine_threadsafe(coro, self._background_loop)
        return future

    def _call_soon_in_background(self, func, *args):
        """
        在后台事件循环中同步执行函数

        正确使用 call_soon_threadsafe，需要传递函数和参数

        Args:
            func: 要执行的函数
            *args: 函数参数
        """
        if self._background_loop is None:
            raise RuntimeError("后台事件循环未启动")

        self._background_loop.call_soon_threadsafe(func, *args)

    def _call_soon_with_context(self, func, *args, context=None):
        """
        在后台事件循环中执行函数（带上下文）

        Args:
            func: 要执行的函数
            *args: 函数参数
            context: 上下文对象
        """
        if self._background_loop is None:
            raise RuntimeError("后台事件循环未启动")

        if context is not None:
            self._background_loop.call_soon_threadsafe(func, *args, context=context)
        else:
            self._background_loop.call_soon_threadsafe(func, *args)

    def breakdown_script(
            self,
            script_text: str,
            task_id: Optional[str] = None,
            language: Optional[Language] = None,
            wait_timeout: float = 300.0
    ) -> PenshotResult:
        """
        同步执行剧本分镜拆分（等待完成）
        """
        task_id = self.breakdown_script_async(script_text, task_id, language)
        return self.wait_for_result(task_id, timeout=wait_timeout)

    def breakdown_script_async(
            self,
            script_text: str,
            task_id: Optional[str] = None,
            language: Optional[Language] = None,
            callback: Optional[Callable] = None
    ) -> str:
        """
        异步执行剧本分镜拆分（立即返回 task_id）
        """
        task_id = task_id or uuid.uuid4().hex
        lang = language or self.language

        if callback:
            self._callbacks[task_id] = callback

        set_language(lang)

        # 创建任务
        self.task_manager.create_task(
            script=script_text,
            config=self.config,
            task_id=task_id
        )

        # 在后台事件循环中处理任务
        self._run_async_in_background(
            self._process_and_callback(task_id)
        )

        return task_id

    async def _process_and_callback(self, task_id: str):
        """处理任务并触发回调"""
        try:
            await self.task_processor.process_script_task(task_id)

            result = self.get_task_result(task_id)

            if task_id in self._callbacks:
                callback = self._callbacks[task_id]
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, callback, result)
                del self._callbacks[task_id]

        except Exception as e:
            log_with_context("ERROR", f"任务处理失败: {str(e)}", {"task_id": task_id})

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        return self.task_manager.get_task(task_id)

    def get_task_result(self, task_id: str) -> Optional[PenshotResult]:
        """获取任务结果"""
        task = self.task_manager.get_task(task_id)
        if not task:
            return None
        return self._task_to_result(task_id, task)

    def wait_for_result(
            self,
            task_id: str,
            timeout: float = 300.0,
            poll_interval: float = 0.5
    ) -> PenshotResult:
        """等待任务完成"""
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            task = self.task_manager.get_task(task_id)

            if not task:
                return PenshotResult(
                    task_id=task_id,
                    success=False,
                    status="not_found",
                    error=f"任务不存在: {task_id}"
                )

            status = task.get("status")

            if status == "completed":
                return self._task_to_result(task_id, task)

            if status == "failed":
                return PenshotResult(
                    task_id=task_id,
                    success=False,
                    status="failed",
                    error=task.get("error", "未知错误")
                )

            time.sleep(poll_interval)

        return PenshotResult(
            task_id=task_id,
            success=False,
            status="timeout",
            error=f"等待超时 ({timeout}秒)"
        )

    async def wait_for_result_async(
            self,
            task_id: str,
            timeout: float = 300.0,
            poll_interval: float = 0.5
    ) -> PenshotResult:
        """异步等待任务完成"""
        import asyncio

        start_time = asyncio.get_event_loop().time()

        while True:
            task = self.task_manager.get_task(task_id)

            if not task:
                return PenshotResult(
                    task_id=task_id,
                    success=False,
                    status="not_found",
                    error=f"任务不存在: {task_id}"
                )

            status = task.get("status")

            if status == "completed":
                return self._task_to_result(task_id, task)

            if status == "failed":
                return PenshotResult(
                    task_id=task_id,
                    success=False,
                    status="failed",
                    error=task.get("error", "未知错误")
                )

            if asyncio.get_event_loop().time() - start_time > timeout:
                return PenshotResult(
                    task_id=task_id,
                    success=False,
                    status="timeout",
                    error=f"等待超时 ({timeout}秒)"
                )

            await asyncio.sleep(poll_interval)

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.task_manager.get_task(task_id)
        if not task:
            return False

        if task.get("status") in ["completed", "failed", "cancelled"]:
            return False

        self.task_manager.fail_task(task_id, "任务被用户取消")
        return True

    def batch_breakdown(
            self,
            scripts: List[str],
            language: Optional[Language] = None,
            wait_timeout: float = 600.0
    ) -> List[PenshotResult]:
        """批量处理多个剧本"""
        task_ids = []

        for script in scripts:
            task_id = self.breakdown_script_async(script, language=language)
            task_ids.append(task_id)

        results = []
        for task_id in task_ids:
            result = self.wait_for_result(task_id, timeout=wait_timeout)
            results.append(result)

        return results

    async def batch_breakdown_async(
            self,
            scripts: List[str],
            language: Optional[Language] = None,
            max_concurrent: int = 3
    ) -> List[PenshotResult]:
        """异步批量处理多个剧本"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(script: str) -> PenshotResult:
            async with semaphore:
                task_id = self.breakdown_script_async(script, language=language)
                return await self.wait_for_result_async(task_id)

        tasks = [process_one(script) for script in scripts]
        return await asyncio.gather(*tasks)

    def _task_to_result(self, task_id: str, task: Dict) -> PenshotResult:
        """将任务字典转换为 PenshotResult"""
        status = task.get("status")
        is_success = status == "completed"

        processing_time_ms = None
        try:
            if task.get("completed_at") and task.get("created_at"):
                completed_at = datetime.fromisoformat(task["completed_at"]) if isinstance(task["completed_at"], str) else task["completed_at"]
                created_at = datetime.fromisoformat(task["created_at"]) if isinstance(task["created_at"], str) else task["created_at"]
                processing_time_ms = int((completed_at - created_at).total_seconds() * 1000)
        except Exception:
            pass

        data = None
        if task.get("result") and isinstance(task["result"], dict):
            data = task["result"].get("data")

        return PenshotResult(
            task_id=task_id,
            success=is_success,
            status=status,
            data=data,
            error=task.get("error"),
            processing_time_ms=processing_time_ms
        )

    def shutdown(self):
        """关闭后台事件循环"""
        if self._background_loop:
            self._background_loop.call_soon_threadsafe(self._background_loop.stop)
        if self._background_thread:
            self._background_thread.join(timeout=5)


if __name__ == '__main__':

    # 使用示例
    def create_penshot_agent(
            config: Optional[ShotConfig] = None,
            language: Language = Language.ZH
    ) -> PenshotFunction:
        """创建 Penshot 智能体实例"""
        return PenshotFunction(config=config, language=language)

    print(create_penshot_agent())
