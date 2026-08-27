"""Python execution tool with explicit approval and a workspace boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from langchain.tools import tool
from langgraph.types import interrupt

from app.tools.filesystem import WorkspacePathError, resolve_workspace_path


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
            return "用户取消了运行。"

        try:
            result = subprocess.run(
                [sys.executable, str(target)],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=_execution_environment(),
            )
            return (
                f"运行完成。\n退出码：{result.returncode}\n"
                f"[标准输出]\n{result.stdout or '(无)'}\n"
                f"[错误输出]\n{result.stderr or '(无)'}"
            )
        except subprocess.TimeoutExpired:
            return f"运行超过 {timeout_seconds} 秒，已停止等待。"
        except OSError as exc:
            return f"运行 Python 文件时发生错误：{exc}"

    return run_python_file
