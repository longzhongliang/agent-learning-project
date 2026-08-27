"""Interactive command-line interface for the Coding Agent."""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from app.agent import build_agent
from app.config import ConfigurationError, load_settings
from app.persistence import SessionRepository, create_store
from app.planning import build_auto_plan_graph


def _generate_title(model: Any, first_message: str) -> str:
    prompt = (
        "请用一句不超过 10 个字的简体中文，概括下面这句用户消息的主题，"
        "作为对话标题。只输出标题本身，不要引号、不要标点、不要解释。\n\n"
        f"用户消息：{first_message}"
    )
    try:
        title = str(model.invoke([{"role": "user", "content": prompt}]).content).strip().strip('"')
        return title or "未命名会话"
    except Exception as exc:
        print(f"[标题生成失败，将使用默认标题：{exc}]")
        return "未命名会话"


def _print_trace(messages: list[Any]) -> None:
    print("\n========== 本轮运行过程 ==========")
    for message in messages:
        if isinstance(message, HumanMessage):
            print(f"\n用户输入：{message.content}")
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                print("\n模型决定调用工具：")
                for tool_call in message.tool_calls:
                    print(f"- 工具名：{tool_call.get('name')}")
                    print(f"  参数：{tool_call.get('args')}")
            elif message.content:
                print(f"\n最终回答：{message.content}")
        elif isinstance(message, ToolMessage):
            content = str(message.content or "")
            preview = content[:300].replace("\n", " ")
            print(f"\n工具返回：\n{preview}")
            if len(content) > 300:
                print("...工具结果较长，已截断显示。")
    print("\n========== 本轮运行过程结束 ==========")


def _pending_tool_calls(agent: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    state = agent.get_state(config)
    for message in reversed(state.values.get("messages", [])):
        if isinstance(message, AIMessage) and message.tool_calls:
            return message.tool_calls
    return []


def _resume_approval_interrupts(agent: Any, config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Resolve every interrupt, including rejection, before accepting a new user turn."""
    while agent.get_state(config).next:
        calls = _pending_tool_calls(agent, config)
        print("\n>>> Agent 已暂停，等待人工审批。")
        for call in calls:
            print(f"    工具：{call.get('name')}  参数：{call.get('args')}")
        answer = input("是否批准执行？(yes/no)：").strip().lower()
        approved = answer == "yes"
        if not approved:
            print(">>> 已拒绝，正在把拒绝结果返回给 Agent。")
        # Even a rejection must resume the saved graph; otherwise the thread stays paused.
        result = agent.invoke(Command(resume=approved), config=config)
    return result


def _choose_session(repository: SessionRepository) -> tuple[str, bool, str | None]:
    while True:
        sessions = repository.list_sessions()
        if not sessions:
            print("还没有任何会话，为你创建新会话。")
            return str(uuid.uuid4()), True, None

        print("已有会话：")
        for index, session in enumerate(sessions, start=1):
            print(f"  [{index}] {session.title} （{session.created_at}）")
        print("  [n] 新建会话")
        print("  [d] 删除会话")
        choice = input("输入序号选择已有会话，n 新建，或 d 删除：").strip().lower()

        if choice == "n":
            return str(uuid.uuid4()), True, None
        if choice == "d":
            deletion = input("输入要删除的会话序号：").strip()
            try:
                session = sessions[int(deletion) - 1]
            except (ValueError, IndexError):
                print("无效的序号。")
                continue
            if input(f"确认删除「{session.title}」？(yes/no)：").strip().lower() == "yes":
                repository.delete_session(session.thread_id)
                print(f"已删除会话「{session.title}」。")
            else:
                print("已取消删除。")
            continue

        try:
            session = sessions[int(choice) - 1]
            return session.thread_id, False, session.title
        except (ValueError, IndexError):
            print("请输入有效序号、n 或 d。")


def run_cli() -> None:
    settings = load_settings()
    repository = SessionRepository(settings.database_path)
    try:
        with create_store(settings.database_path) as store:
            agent = build_agent(settings, repository.checkpointer, store)
            auto_plan_graph = build_auto_plan_graph(agent)
            thread_id, is_new, title = _choose_session(repository)
            config = {"configurable": {"thread_id": thread_id}}

            while True:
                user_input = input("用户：").strip()
                if user_input in {"退出", "exit", "quit"}:
                    print("退出对话。")
                    return
                if not user_input:
                    continue

                # /plan <task>：用户强制计划模式
                if user_input == "/plan" or user_input.startswith("/plan "):
                    task = user_input.removeprefix("/plan").strip()
                    if not task:
                        print("用法：/plan 你的复杂任务")
                        continue
                    try:
                        plan_result = auto_plan_graph.invoke({"task": task})
                        plan = plan_result.get("execution_plan", "")
                    except Exception as exc:
                        print(f"生成计划失败：{type(exc).__name__} - {exc}")
                        continue
                    print("\n========== 用户强制计划 ==========")
                    print(plan)
                    print("===================================")
                    if input("是否批准按此计划继续执行？(yes/no)：").strip().lower() != "yes":
                        print("已取消本轮任务，Agent 不会执行。")
                        continue
                    user_input = (
                        f"用户原始任务：\n{task}\n\n"
                        f"【用户强制计划：已确认】\n{plan}\n\n"
                        "请直接按这个计划执行。计划不适用时，先说明原因。"
                    )

                if is_new and title is None:
                    title = _generate_title(agent, user_input)
                    repository.create_session(thread_id, title)
                    is_new = False
                    print(f"[新会话已创建] 标题：{title}")

                before_count = len(agent.get_state(config).values.get("messages", []))
                result = agent.invoke({"messages": HumanMessage(content=user_input)}, config=config)
                result = _resume_approval_interrupts(agent, config, result)
                messages = result["messages"][before_count:]
                final_message = result["messages"][-1]
                print(f"模型：{final_message.content}")
                _print_trace(messages)
    finally:
        repository.close()


def main() -> None:
    try:
        run_cli()
    except ConfigurationError as exc:
        print(f"配置错误：{exc}")
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
