"""Python execution tool with explicit approval and a workspace boundary."""

from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt
from pydantic import BaseModel

from app.tools.filesystem import WorkspacePathError, resolve_workspace_path


MAX_RUN_PYTHON_ATTEMPTS_PER_TURN = 2


class ToolExecutionResult(BaseModel):
    ok: bool
    message: str
    error_type: str | None = None
    retry_strategy: str = "never"
    side_effect_status: str = "none"
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""


class RunPythonRetryGuardMiddleware(AgentMiddleware):
    """Block repeated Python runs in one turn unless there is new evidence."""

    @staticmethod
    def _completed_tool_calls_in_current_turn(messages: list[Any]) -> list[dict[str, Any]]:
        last_user_index = max(
            (
                index
                for index, message in enumerate(messages)
                if type(message).__name__ == "HumanMessage"
            ),
            default=-1,
        )

        calls_by_id = {}
        completed_calls = []
        for index, message in enumerate(messages[last_user_index + 1 :], start=last_user_index + 1):
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls or []:
                    calls_by_id[tool_call["id"]] = {
                        "name": tool_call["name"],
                        "args": tool_call.get("args") or {},
                    }
            elif isinstance(message, ToolMessage):
                tool_call = calls_by_id.get(message.tool_call_id)
                if tool_call:
                    completed_calls.append(
                        {**tool_call, "content": message.content, "message_index": index}
                    )
        return completed_calls

    @staticmethod
    def _blocked_result(
        request: ToolCallRequest,
        message: str,
        retry_strategy: str = "never",
    ) -> ToolMessage:
        return ToolMessage(
            content=ToolExecutionResult(
                ok=False,
                message=message,
                error_type="retry_blocked",
                retry_strategy=retry_strategy,
                side_effect_status="unknown",
            ).model_dump_json(),
            tool_call_id=request.tool_call["id"],
        )

    def _should_block_run(
        self,
        request: ToolCallRequest,
        file_path: str,
        completed_calls: list[dict[str, Any]],
    ) -> ToolMessage | None:
        previous_runs = [
            item
            for item in completed_calls
            if item["name"] == "run_python_file"
            and str(item["args"].get("file_path", "")) == file_path
        ]
        if len(previous_runs) >= MAX_RUN_PYTHON_ATTEMPTS_PER_TURN:
            return self._blocked_result(
                request,
                "本轮中该文件已达到最多两次运行的上限，必须停止并报告失败。",
            )
        if not previous_runs:
            return None

        latest_run = previous_runs[-1]
        try:
            latest_result = json.loads(str(latest_run["content"]))
        except json.JSONDecodeError:
            return None

        retry_strategy = latest_result.get("retry_strategy", "never")
        if latest_result.get("ok") or retry_strategy == "never":
            return self._blocked_result(request, "该文件本轮无需再次运行；禁止重复执行。")
        if retry_strategy == "after_fix_input":
            return self._blocked_result(
                request,
                "文件路径或输入未改变，不能原样重复运行。请先修正输入。",
                retry_strategy="after_fix_input",
            )
        if retry_strategy == "after_state_verification":
            has_verified_file = any(
                item["message_index"] > latest_run["message_index"]
                and item["name"] in {"read_file", "read_file_lines"}
                and str(item["args"].get("file_path", "")) == file_path
                for item in completed_calls
            )
            if not has_verified_file:
                return self._blocked_result(
                    request,
                    "上次运行可能已产生部分影响。请先读取同一文件确认现场，再决定是否重试。",
                    retry_strategy="after_state_verification",
                )
        return None

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        if request.tool_call["name"] != "run_python_file":
            return handler(request)
        file_path = str(request.tool_call["args"].get("file_path", ""))
        blocked = self._should_block_run(
            request,
            file_path,
            self._completed_tool_calls_in_current_turn(request.state["messages"]),
        )
        if blocked is not None:
            return blocked
        return handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        if request.tool_call["name"] != "run_python_file":
            return await handler(request)
        file_path = str(request.tool_call["args"].get("file_path", ""))
        blocked = self._should_block_run(
            request,
            file_path,
            self._completed_tool_calls_in_current_turn(request.state["messages"]),
        )
        if blocked is not None:
            return blocked
        return await handler(request)


def _execution_environment() -> dict[str, str]:
    """Do not pass common provider secrets to code started by the agent."""
    blocked = {"DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LANGSMITH_API_KEY"}
    return {key: value for key, value in os.environ.items() if key.upper() not in blocked}


def create_execution_tool(workspace_dir: Path, timeout_seconds: int):
    """Create a tool that can run only workspace-relative Python files after approval."""
    workspace = workspace_dir.resolve()

    @tool
    def run_python_file(file_path: str) -> str:
        """运行工作目录内的 .py 文件。每次运行前必须由用户人工审批。"""
        try:
            target = resolve_workspace_path(workspace, file_path)
        except WorkspacePathError as exc:
            return f"运行失败：{exc}"
        if target.suffix.lower() != ".py":
            return "只允许运行 .py 文件。"
        if not target.is_file():
            return f"文件不存在：{target}"

        decision = interrupt(
            {"action": "run_python_file", "file_path": file_path, "target": str(target)}
        )
        if decision is not True:
            return ToolExecutionResult(
                ok=False,
                message="用户取消了运行。",
                error_type="cancelled",
                retry_strategy="never",
            ).model_dump_json()

        try:
            result = subprocess.run(
                [sys.executable, str(target)],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=_execution_environment(),
            )
            if result.returncode == 0:
                return ToolExecutionResult(
                    ok=True,
                    message="Python 文件运行成功。",
                    side_effect_status="completed",
                    exit_code=0,
                    stdout=result.stdout,
                    stderr=result.stderr,
                ).model_dump_json()
            return ToolExecutionResult(
                ok=False,
                message="Python 文件运行失败。",
                error_type="process_exit_nonzero",
                retry_strategy="after_state_verification",
                side_effect_status="unknown",
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            ).model_dump_json()
        except subprocess.TimeoutExpired:
            return ToolExecutionResult(
                ok=False,
                message=f"运行超过 {timeout_seconds} 秒，已停止等待。",
                error_type="timeout",
                retry_strategy="after_state_verification",
                side_effect_status="unknown",
            ).model_dump_json()
        except OSError as exc:
            return ToolExecutionResult(
                ok=False,
                message=f"运行 Python 文件时发生错误：{exc}",
                error_type="tool_internal_error",
                retry_strategy="after_state_verification",
                side_effect_status="unknown",
            ).model_dump_json()

    return run_python_file
