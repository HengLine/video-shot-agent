"""
@FileName: mcp_server.py
@Description: MCP (Model Context Protocol) Server
支持 Claude、Cursor 等 MCP 兼容智能体调用
@Author: HiPeng
@Time: 2026/3/23 18:39
"""

import asyncio
import json
from typing import Dict, Any, Optional

from penshot.api.function_calls import PenshotFunction
from penshot.neopen.shot_language import Language
from penshot.neopen.task.task_manager import TaskManager


class PenshotMCPServer:
    """
    Penshot MCP Server

    复用现有的 TaskManager 进行任务管理
    """

    def __init__(self, task_manager: Optional[TaskManager] = None):
        """
        初始化 MCP Server

        Args:
            task_manager: 任务管理器（可选，复用现有实例）
        """
        self.task_manager = task_manager or TaskManager()
        self.penshot = PenshotFunction(
            task_manager=self.task_manager,
            language=Language.ZH
        )
        self._tools = {}
        self._register_tools()

    def _register_tools(self):
        """注册 MCP 工具"""

        self._tools["breakdown_script"] = {
            "description": "将剧本拆分为分镜序列",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "剧本文本"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["zh", "en"],
                        "description": "输出语言",
                        "default": "zh"
                    }
                },
                "required": ["script"]
            },
            "handler": self._handle_breakdown_script
        }

        self._tools["get_task_status"] = {
            "description": "获取分镜生成任务的状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "任务ID"
                    }
                },
                "required": ["task_id"]
            },
            "handler": self._handle_get_task_status
        }

        self._tools["get_task_result"] = {
            "description": "获取任务结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "任务ID"
                    }
                },
                "required": ["task_id"]
            },
            "handler": self._handle_get_task_result
        }

        self._tools["cancel_task"] = {
            "description": "取消正在执行的任务",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "任务ID"
                    }
                },
                "required": ["task_id"]
            },
            "handler": self._handle_cancel_task
        }

    def get_tools_list(self) -> list:
        """获取工具列表（MCP协议要求）"""
        return [
            {
                "name": name,
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for name, tool in self._tools.items()
        ]

    async def call_tool(self, tool_name: str, arguments: Dict) -> Dict[str, Any]:
        """调用工具"""
        if tool_name not in self._tools:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self._tools.keys())
            }

        tool = self._tools[tool_name]
        try:
            result = await tool["handler"](arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_breakdown_script(self, arguments: Dict) -> Dict:
        """处理剧本分镜拆分"""
        script = arguments.get("script")
        language = arguments.get("language", "zh")

        if not script:
            raise ValueError("script is required")

        lang = Language.ZH if language == "zh" else Language.EN

        # 异步提交任务
        task_id = self.penshot.breakdown_script_async(
            script_text=script,
            language=lang
        )

        return {
            "task_id": task_id,
            "status": "submitted",
            "message": "任务已提交，请使用 get_task_status 查询进度"
        }

    async def _handle_get_task_status(self, arguments: Dict) -> Dict:
        """获取任务状态"""
        task_id = arguments.get("task_id")

        if not task_id:
            raise ValueError("task_id is required")

        task = self.task_manager.get_task(task_id)

        if not task:
            return {"task_id": task_id, "status": "not_found"}

        return {
            "task_id": task_id,
            "status": task.get("status"),
            "stage": task.get("stage"),
            "progress": task.get("progress"),
            "created_at": task.get("created_at"),
            "updated_at": task.get("updated_at")
        }

    async def _handle_get_task_result(self, arguments: Dict) -> Dict:
        """获取任务结果"""
        task_id = arguments.get("task_id")

        if not task_id:
            raise ValueError("task_id is required")

        result = self.penshot.get_task_result(task_id)

        if not result:
            return {"task_id": task_id, "status": "not_found"}

        return {
            "task_id": result.task_id,
            "success": result.success,
            "status": result.status,
            "data": result.data,
            "error": result.error,
            "processing_time_ms": result.processing_time_ms
        }

    async def _handle_cancel_task(self, arguments: Dict) -> Dict:
        """取消任务"""
        task_id = arguments.get("task_id")

        if not task_id:
            raise ValueError("task_id is required")

        success = self.penshot.cancel_task(task_id)

        return {
            "task_id": task_id,
            "cancelled": success,
            "message": "任务已取消" if success else "任务不存在或无法取消"
        }

    def create_stdio_server(self):
        """创建标准输入输出服务器"""
        import sys

        async def handle_request(request: Dict) -> Dict:
            method = request.get("method")
            params = request.get("params", {})

            if method == "tools/list":
                return {"tools": self.get_tools_list()}
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                return await self.call_tool(tool_name, arguments)
            else:
                return {"error": f"Unknown method: {method}"}

        async def main():
            while True:
                line = sys.stdin.readline()
                if not line:
                    break

                try:
                    request = json.loads(line)
                    response = await handle_request(request)
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
                except Exception as e:
                    sys.stderr.write(f"Error: {str(e)}\n")

        asyncio.run(main())


def run_mcp_server():
    """启动 MCP 服务器（命令行入口）"""
    server = PenshotMCPServer()
    server.create_stdio_server()
