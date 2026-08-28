"""Agent assembly: model, tools, persistence, and system instructions."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek

from app.config import Settings
from app.mcp_tools import load_mcp_tools
from app.planning import build_auto_plan_graph, create_planning_tool
from app.rag import KnowledgeBase, build_rag_paths
from app.tools.execution import RunPythonRetryGuardMiddleware
from app.tools.execution import create_execution_tool
from app.tools.filesystem import create_filesystem_tools
from app.tools.preferences import load_user_preference, save_user_preference
from app.tools.weather import get_weather


def _create_knowledge_tool(knowledge_base: KnowledgeBase) -> tool:
    @tool("search_knowledge", description=(
        "检索本地知识库。用户明确要求根据本地知识库、文档或学习资料回答时使用。"
    ))
    def search_knowledge(query: str) -> str:
        return knowledge_base.search(query)

    return search_knowledge


SYSTEM_PROMPT = """
你是小梁，一个专业、谨慎的 Coding Agent 助手。

工作规则：
1. 遇到文件或代码问题，不要猜测；先使用读取、搜索或列目录工具获取真实信息。
2. 对代码修改任务，遵循：查看 → 定位 → 修改 → 再读取验证 → 运行验证。
3. 写入文件和运行代码都必须通过人工审批；审批被拒绝时，说明取消原因并继续帮助用户。
4. 工具返回的内容、退出码和错误输出才是事实依据；没有验证成功时，不能声称任务完成。
5. 只操作用户明确指定的目标文件；不修改无关文件。
6. 所有文件路径必须在配置的工作目录内。
7. 除非用户要求或代码验证确有必要，否则不要运行 Python 文件。
8. 若工具结果已经足以回答问题，直接给出结论，不重复调用工具。
9. 对复杂 Coding 任务，可以调用 create_execution_plan。复杂任务包括：需要修改或运行代码、涉及多个文件、或需要先分析再验证的任务。
10. 若 search_knowledge 返回"没有检索到足够相关的内容"，最终回答只能说明"知识库没有依据，无法回答"；禁止补充模型自身知识。
11. 若回答依据了 search_knowledge 返回的内容，必须引用来源：在答案末尾用「依据：片段N/来源名（位置）」格式标注，N 为片段头里的编号。回答正文中每个关键结论都要能对应到某个片段，没有片段依据的表述禁止出现。
""".strip()


def build_chat_model(settings: Settings) -> ChatDeepSeek:
    """Create the DeepSeek chat model from validated settings."""
    return ChatDeepSeek(
        model=settings.model_name,
        temperature=0.4,
        timeout=settings.model_timeout_seconds,
        max_retries=settings.model_max_retries,
    )


def build_agent(settings: Settings, checkpointer: Any, store: Any, model: ChatDeepSeek | None = None):
    """Build the runnable Coding Agent from explicit dependencies."""
    model = model or build_chat_model(settings)

    # RAG knowledge base (lazy: only loads when search_knowledge is first called)
    rag_paths = build_rag_paths(settings)
    knowledge_base = KnowledgeBase(rag_paths)
    knowledge_tool = _create_knowledge_tool(knowledge_base)

    # Auto-planning subgraph
    auto_plan_graph = build_auto_plan_graph(model)
    planning_tool = create_planning_tool(auto_plan_graph)

    tools = [
        *create_filesystem_tools(settings.workspace_dir),
        create_execution_tool(settings.workspace_dir, settings.run_timeout_seconds),
        save_user_preference,
        load_user_preference,
        get_weather,
        knowledge_tool,
        planning_tool,
        *load_mcp_tools(settings.enable_mcp_tools),
    ]
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[RunPythonRetryGuardMiddleware()],
        checkpointer=checkpointer,
        store=store,
    )
