"""Optional MCP tool loading for external read-only context."""

from __future__ import annotations

import asyncio
import sys
from typing import Any


WANTED_MCP_TOOLS = {
    "search_repositories",
    "search_code",
    "search_issues",
    "search_users",
    "get_file_contents",
    "list_commits",
    "get_issue",
    "get_pull_request",
    "list_pull_requests",
    "get_pull_request_files",
    "fetch",
}


async def _load_mcp_tools() -> list[Any]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "transport": "stdio",
            },
            "fetch": {
                "command": sys.executable,
                "args": ["-m", "mcp_server_fetch", "--ignore-robots-txt"],
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    return [tool for tool in tools if tool.name in WANTED_MCP_TOOLS]


def load_mcp_tools(enabled: bool) -> list[Any]:
    """Load filtered MCP tools when enabled, otherwise return no tools."""
    if not enabled:
        return []
    try:
        tools = asyncio.run(_load_mcp_tools())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            tools = loop.run_until_complete(_load_mcp_tools())
        finally:
            loop.close()
    except Exception as exc:
        print(f"[MCP] 外部工具接入失败，已跳过：{type(exc).__name__} - {exc}")
        return []

    print(f"[MCP] 已接入 {len(tools)} 个外部工具（按需过滤后）。")
    return tools
