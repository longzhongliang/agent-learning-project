"""Workspace-restricted file tools and their testable implementation helpers."""

from __future__ import annotations

import os
from pathlib import Path

from langchain.tools import tool
from langgraph.types import interrupt


class WorkspacePathError(ValueError):
    """Raised when a requested path escapes the configured workspace."""


def resolve_workspace_path(workspace_dir: Path, file_path: str) -> Path:
    """Resolve a relative path and reject paths outside ``workspace_dir``."""
    if not isinstance(file_path, str) or not file_path.strip():
        raise WorkspacePathError("文件路径必须是非空字符串。")

    workspace = workspace_dir.resolve()
    requested = Path(file_path)
    if requested.is_absolute():
        raise WorkspacePathError("只允许使用工作目录内的相对路径。")

    target = (workspace / requested).resolve()
    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise WorkspacePathError(f"拒绝访问工作目录以外的路径：{file_path}") from exc
    return target


def write_text_file(workspace_dir: Path, file_path: str, content: str) -> str:
    """Write UTF-8 text after the caller has obtained human approval."""
    if not isinstance(content, str):
        return "写入失败：content 必须是字符串。"

    target = resolve_workspace_path(workspace_dir, file_path)
    if target.exists() and not target.is_file():
        return f"写入失败：目标不是普通文件：{target}"
    if not target.parent.is_dir():
        return f"写入失败：父目录不存在：{target.parent}"

    try:
        target.write_text(content, encoding="utf-8")
        if target.read_text(encoding="utf-8") != content:
            return f"写入后校验失败：{target}"
        return f"写入成功并已校验：{target}（{len(content)} 个字符）"
    except OSError as exc:
        return f"写入失败：{exc}"


def read_text_file(workspace_dir: Path, file_path: str, max_length: int = 4000) -> str:
    """Read at most ``max_length`` UTF-8 characters from a workspace file."""
    if not isinstance(max_length, int) or max_length <= 0:
        return "读取失败：max_length 必须是大于 0 的整数。"

    try:
        target = resolve_workspace_path(workspace_dir, file_path)
        content = target.read_text(encoding="utf-8")
        if len(content) > max_length:
            return (
                content[:max_length]
                + f"\n\n[内容过长，已截断。原始长度：{len(content)}，输出长度：{max_length}]"
            )
        return content
    except FileNotFoundError:
        return "文件未找到。"
    except UnicodeDecodeError:
        return "文件不是 UTF-8 文本，暂时无法读取。"
    except WorkspacePathError as exc:
        return f"读取失败：{exc}"
    except OSError as exc:
        return f"读取文件时发生错误：{exc}"


def read_text_file_lines(workspace_dir: Path, file_path: str, start_line: int, end_line: int) -> str:
    """Read an inclusive line range from a UTF-8 workspace text file."""
    if not isinstance(start_line, int) or not isinstance(end_line, int):
        return "行号必须是整数。"
    if start_line < 1 or end_line < 1:
        return "行号必须从 1 开始。"

    try:
        target = resolve_workspace_path(workspace_dir, file_path)
        lines = target.read_text(encoding="utf-8").splitlines()
        end = min(len(lines), end_line)
        if start_line > end:
            return f"行号范围无效。文件总行数：{len(lines)}"
        return "\n".join(f"{number}: {lines[number - 1]}" for number in range(start_line, end + 1))
    except FileNotFoundError:
        return "文件未找到。"
    except UnicodeDecodeError:
        return "文件不是 UTF-8 文本，暂时无法读取。"
    except WorkspacePathError as exc:
        return f"读取失败：{exc}"
    except OSError as exc:
        return f"按行读取文件时发生错误：{exc}"


def search_text_in_file(workspace_dir: Path, file_path: str, keywords: str, max_result: int = 20) -> str:
    """Find matching lines in a UTF-8 workspace file."""
    if not isinstance(max_result, int) or max_result <= 0:
        return "搜索失败：max_result 必须是大于 0 的整数。"
    if not isinstance(keywords, str) or not keywords:
        return "请提供要搜索的关键词。"

    try:
        target = resolve_workspace_path(workspace_dir, file_path)
        matches: list[str] = []
        for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
            if keywords in line:
                matches.append(f"{number}: {line}")
                if len(matches) >= max_result:
                    break
        return "\n".join(matches) if matches else f"未找到包含关键词「{keywords}」的内容。"
    except FileNotFoundError:
        return "文件未找到。"
    except UnicodeDecodeError:
        return "文件不是 UTF-8 文本，暂时无法读取。"
    except WorkspacePathError as exc:
        return f"搜索失败：{exc}"
    except OSError as exc:
        return f"搜索文件时发生错误：{exc}"


def list_workspace_files(workspace_dir: Path, directory_path: str = ".") -> str:
    """List the immediate contents of a workspace directory."""
    try:
        directory = resolve_workspace_path(workspace_dir, directory_path)
        if not directory.is_dir():
            return "这不是目录。"
        entries = sorted(directory.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower()))
        if not entries:
            return "目录为空。"
        return "\n".join(
            f"[目录] {entry.name}" if entry.is_dir() else f"[文件] {entry.name}" for entry in entries
        )
    except FileNotFoundError:
        return "目录未找到。"
    except WorkspacePathError as exc:
        return f"列目录失败：{exc}"
    except OSError as exc:
        return f"读取目录时发生错误：{exc}"


def create_filesystem_tools(workspace_dir: Path):
    """Create LangChain tools bound to one workspace directory."""
    workspace = workspace_dir.resolve()

    @tool
    def write_file(file_path: str, content: str) -> str:
        """在工作目录内写入 UTF-8 文本。每次写入前必须由用户人工审批。"""
        try:
            target = resolve_workspace_path(workspace, file_path)
        except WorkspacePathError as exc:
            return f"写入失败：{exc}"
        decision = interrupt(
            {
                "action": "write_file",
                "file_path": file_path,
                "target": str(target),
                "content_length": len(content),
                "preview": content[:300],
            }
        )
        if decision is not True:
            return "用户取消了写入。"
        return write_text_file(workspace, file_path, content)

    @tool
    def read_file(file_path: str, max_length: int = 4000) -> str:
        """读取工作目录内的 UTF-8 文本文件，最多返回 max_length 个字符。"""
        return read_text_file(workspace, file_path, max_length)

    @tool
    def read_file_lines(file_path: str, start_line: int, end_line: int) -> str:
        """按行读取工作目录内的 UTF-8 文本文件，行号从 1 开始。"""
        return read_text_file_lines(workspace, file_path, start_line, end_line)

    @tool
    def search_text(file_path: str, keywords: str, max_result: int = 20) -> str:
        """在工作目录内的 UTF-8 文本文件中搜索关键词，返回匹配行。"""
        return search_text_in_file(workspace, file_path, keywords, max_result)

    @tool
    def list_files(directory_path: str = ".") -> str:
        """列出工作目录内指定目录的直接子项。"""
        return list_workspace_files(workspace, directory_path)

    return [write_file, read_file, read_file_lines, search_text, list_files]
