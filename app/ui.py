"""Gradio interface for the local Coding Agent."""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from app.agent import build_agent, build_chat_model
from app.config import load_settings
from app.cli import _generate_title, _recover_interrupted_tool_calls
from app.persistence import SessionRepository, create_store
from app.planning import build_auto_plan_graph, build_execution_plan


def _new_ui_state(thread_id: str | None = None, title: str | None = None, is_new: bool = True):
    return {
        "thread_id": thread_id or str(uuid.uuid4()),
        "title": title,
        "is_new": is_new,
        "history": [],
        "pending": None,
        "trace": "等待任务。",
    }


def _history_from_messages(messages: list[Any]) -> list[dict[str, str]]:
    history = []
    for message in messages:
        if isinstance(message, HumanMessage) and message.content:
            history.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, AIMessage) and not (message.tool_calls or []) and message.content:
            history.append({"role": "assistant", "content": str(message.content)})
    return history


def _trace_from_messages(messages: list[Any]) -> str:
    items = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                items.append(f"准备调用：{tool_call.get('name')}({tool_call.get('args')})")
        elif isinstance(message, ToolMessage):
            preview = str(message.content or "").replace("\n", " ")[:500]
            items.append(f"工具返回：{preview}")
    return "\n\n".join(items) or "本轮未调用工具。"


def run_gradio_app() -> None:
    """Start the Gradio UI. Requires the optional gradio dependency."""
    try:
        import gradio as gr
    except ModuleNotFoundError as error:
        raise RuntimeError("未安装 Gradio。请先运行：python -m pip install -e .") from error

    settings = load_settings()
    repository = SessionRepository(settings.database_path)
    store_context = create_store(settings.database_path)
    store = store_context.__enter__()

    try:
        model = build_chat_model(settings)
        agent = build_agent(settings, repository.checkpointer, store, model=model)
        auto_plan_graph = build_auto_plan_graph(model)

        def session_choices():
            return [
                (f"{session.title}  ·  {session.created_at}", session.thread_id)
                for session in repository.list_sessions()
            ]

        def pending_tool_text(config):
            tool_calls = []
            for message in reversed(agent.get_state(config).values.get("messages", [])):
                if isinstance(message, AIMessage) and message.tool_calls:
                    tool_calls = message.tool_calls
                    break
            if not tool_calls:
                return "Agent 正在等待确认，但没有读取到工具详情。"
            details = "\n".join(
                f"- `{tool_call.get('name')}`：`{tool_call.get('args')}`"
                for tool_call in tool_calls
            )
            return f"### Agent 请求执行操作\n\n{details}"

        def ui_response(state, status, plan_text="", show_plan=False, tool_text="", show_tool=False):
            return (
                state["history"],
                state,
                status,
                gr.update(value=plan_text, visible=show_plan),
                gr.update(visible=show_plan),
                gr.update(visible=show_plan),
                gr.update(value=tool_text, visible=show_tool),
                gr.update(visible=show_tool),
                gr.update(visible=show_tool),
                state["trace"],
            )

        def ensure_session(state, task):
            if not state["is_new"] or state["title"] is not None:
                return
            try:
                title = _generate_title(model, task)
            except Exception:
                title = "未命名会话"
            repository.create_session(state["thread_id"], title)
            state["title"] = title
            state["is_new"] = False

        def add_final_messages(state, messages):
            for message in messages:
                if isinstance(message, AIMessage) and not (message.tool_calls or []) and message.content:
                    state["history"].append({"role": "assistant", "content": str(message.content)})

        def execute_task(state, task, execution_plan=None, plan_source=None):
            ensure_session(state, task)
            config = {"configurable": {"thread_id": state["thread_id"]}}
            before_count = len(agent.get_state(config).values.get("messages", []))
            if execution_plan:
                message_content = (
                    f"用户原始任务：\n{task}\n\n"
                    f"【{plan_source}：已确认】\n{execution_plan}\n\n"
                    "请直接按这个计划执行。计划不适用时，先说明原因。"
                )
            else:
                message_content = task

            result = agent.invoke({"messages": HumanMessage(content=message_content)}, config=config)
            run_messages = result["messages"][before_count:]
            state["trace"] = _trace_from_messages(run_messages)
            if agent.get_state(config).next:
                state["pending"] = {"kind": "tool"}
                return ui_response(
                    state,
                    "Agent 已暂停，等待你决定是否批准工具操作。",
                    tool_text=pending_tool_text(config),
                    show_tool=True,
                )

            state["pending"] = None
            add_final_messages(state, run_messages)
            return ui_response(state, "本轮任务已结束。")

        def submit_message(message, state):
            message = (message or "").strip()
            if not message:
                return ("",) + ui_response(state, "请输入任务后再发送。")
            if state.get("pending"):
                return ("",) + ui_response(state, "请先处理当前的计划或工具审批。")

            task = message
            execution_plan = None
            plan_source = None
            state["history"].append({"role": "user", "content": task})

            if message == "/plan" or message.startswith("/plan "):
                task = message.removeprefix("/plan").strip()
                if not task:
                    state["history"].append({"role": "assistant", "content": "用法：`/plan 你的复杂任务`"})
                    return ("",) + ui_response(state, "计划任务为空。")
                try:
                    execution_plan = build_execution_plan(model, task)
                    plan_source = "用户强制计划"
                except Exception as error:
                    state["history"].append(
                        {"role": "assistant", "content": f"生成计划失败：{type(error).__name__} - {error}"}
                    )
                    return ("",) + ui_response(state, "生成计划失败。")
            else:
                try:
                    auto_plan_result = auto_plan_graph.invoke({"task": task})
                    if auto_plan_result.get("should_plan"):
                        execution_plan = auto_plan_result["execution_plan"]
                        plan_source = "自动计划"
                except Exception as error:
                    state["trace"] = f"自动计划流程失败，将按普通模式执行：{type(error).__name__} - {error}"

            if execution_plan:
                state["pending"] = {
                    "kind": "plan",
                    "task": task,
                    "execution_plan": execution_plan,
                    "plan_source": plan_source,
                }
                state["history"].append({"role": "assistant", "content": "我已生成执行计划，请确认后再开始。"})
                return ("",) + ui_response(
                    state,
                    f"{plan_source}已生成，等待确认。",
                    plan_text=f"### {plan_source}\n\n{execution_plan}",
                    show_plan=True,
                )

            return ("",) + execute_task(state, task)

        def approve_plan(state):
            pending = state.get("pending") or {}
            if pending.get("kind") != "plan":
                return ui_response(state, "当前没有待确认的计划。")
            state["pending"] = None
            state["history"].append({"role": "assistant", "content": "已批准计划，开始执行。"})
            return execute_task(
                state,
                pending["task"],
                pending["execution_plan"],
                pending["plan_source"],
            )

        def reject_plan(state):
            if (state.get("pending") or {}).get("kind") == "plan":
                state["pending"] = None
                state["history"].append({"role": "assistant", "content": "已取消本轮计划，Agent 未执行任何操作。"})
            return ui_response(state, "计划已取消。")

        def resume_tool(state, approved):
            if (state.get("pending") or {}).get("kind") != "tool":
                return ui_response(state, "当前没有待确认的工具操作。")

            config = {"configurable": {"thread_id": state["thread_id"]}}
            before_count = len(agent.get_state(config).values.get("messages", []))
            result = agent.invoke(Command(resume=approved), config=config)
            run_messages = result["messages"][before_count:]
            state["trace"] = _trace_from_messages(run_messages)
            if agent.get_state(config).next:
                return ui_response(
                    state,
                    "Agent 又请求了一项工具操作，请继续确认。",
                    tool_text=pending_tool_text(config),
                    show_tool=True,
                )

            state["pending"] = None
            add_final_messages(state, run_messages)
            return ui_response(
                state,
                "本轮任务已结束。" if approved else "已拒绝工具操作，Agent 已给出后续结论。",
            )

        def load_session(thread_id):
            if not thread_id:
                return ui_response(_new_ui_state(), "已准备新会话。")

            session = next(
                (item for item in repository.list_sessions() if item.thread_id == thread_id),
                None,
            )
            if session is None:
                return ui_response(_new_ui_state(), "该会话不存在，已准备新会话。")

            state = _new_ui_state(thread_id=session.thread_id, title=session.title, is_new=False)
            config = {"configurable": {"thread_id": session.thread_id}}
            _recover_interrupted_tool_calls(agent, config)
            state["history"] = _history_from_messages(agent.get_state(config).values.get("messages", []))
            return ui_response(state, f"已载入会话：{session.title}。")

        def create_new_session():
            state = _new_ui_state()
            return (gr.update(choices=session_choices(), value=None),) + ui_response(
                state,
                "已准备新会话；发送第一条消息后才会保存。",
            )

        css = """
        :root { --ink: #edf2f7; --muted: #9aa6b2; --panel: #151c27; --line: #28364b; --cyan: #5eead4; --orange: #fb923c; }
        .gradio-container { background: radial-gradient(circle at 8% 2%, #21374b 0, #0c1119 42%, #070a10 100%); color: var(--ink); min-height: 100vh; }
        #title { letter-spacing: .08em; font-weight: 700; color: var(--cyan); }
        #subhead { color: var(--muted); }
        .panel { border: 1px solid var(--line); border-radius: 14px; background: rgba(21, 28, 39, .88); box-shadow: 0 18px 50px rgba(0,0,0,.24); }
        .status { border-left: 3px solid var(--cyan); padding-left: 12px; color: var(--muted); }
        .danger { border-left: 3px solid var(--orange); padding-left: 12px; }
        """

        with gr.Blocks(title="小梁 · 本地 Agent", css=css, theme=gr.themes.Base()) as demo:
            ui_state = gr.State(_new_ui_state())
            gr.Markdown("# 小梁 · 本地 Agent", elem_id="title")
            gr.Markdown("本地会话、知识库检索、计划与人工审批都在这里完成。", elem_id="subhead")

            with gr.Row():
                with gr.Column(scale=1, min_width=260, elem_classes="panel"):
                    session_dropdown = gr.Dropdown(
                        choices=session_choices(),
                        label="历史会话",
                        info="选择一个会话继续，或新建会话。",
                    )
                    new_session_button = gr.Button("＋ 新建会话", variant="secondary")
                    status = gr.Markdown("准备就绪。", elem_classes="status")
                    trace = gr.Textbox(
                        label="本轮工具记录",
                        value="等待任务。",
                        lines=14,
                        max_lines=18,
                        interactive=False,
                    )

                with gr.Column(scale=3, elem_classes="panel"):
                    chatbot = gr.Chatbot(
                        label="对话",
                        type="messages",
                        height=560,
                        placeholder="从一个任务开始，例如：读取 README 并总结。",
                    )
                    plan_box = gr.Markdown(visible=False, elem_classes="danger")
                    with gr.Row():
                        approve_plan_button = gr.Button("批准计划", variant="primary")
                        reject_plan_button = gr.Button("取消计划", variant="stop")
                    tool_box = gr.Markdown(visible=False, elem_classes="danger")
                    with gr.Row():
                        approve_tool_button = gr.Button("批准操作", variant="primary")
                        reject_tool_button = gr.Button("拒绝操作", variant="stop")
                    with gr.Row():
                        user_input = gr.Textbox(
                            placeholder="输入任务；使用 /plan 可以强制先生成计划",
                            label="你的任务",
                            scale=8,
                            autofocus=True,
                        )
                        send_button = gr.Button("发送", variant="primary", scale=1)

            response_outputs = [
                chatbot,
                ui_state,
                status,
                plan_box,
                approve_plan_button,
                reject_plan_button,
                tool_box,
                approve_tool_button,
                reject_tool_button,
                trace,
            ]
            send_button.click(
                submit_message,
                inputs=[user_input, ui_state],
                outputs=[user_input] + response_outputs,
            )
            user_input.submit(
                submit_message,
                inputs=[user_input, ui_state],
                outputs=[user_input] + response_outputs,
            )
            approve_plan_button.click(approve_plan, inputs=ui_state, outputs=response_outputs)
            reject_plan_button.click(reject_plan, inputs=ui_state, outputs=response_outputs)
            approve_tool_button.click(
                lambda state: resume_tool(state, True),
                inputs=ui_state,
                outputs=response_outputs,
            )
            reject_tool_button.click(
                lambda state: resume_tool(state, False),
                inputs=ui_state,
                outputs=response_outputs,
            )
            session_dropdown.change(load_session, inputs=session_dropdown, outputs=response_outputs)
            new_session_button.click(
                create_new_session,
                outputs=[session_dropdown] + response_outputs,
            )

        demo.queue(default_concurrency_limit=1).launch(
            server_name="127.0.0.1",
            inbrowser=True,
            show_error=True,
        )
    finally:
        store_context.__exit__(None, None, None)
        repository.close()
