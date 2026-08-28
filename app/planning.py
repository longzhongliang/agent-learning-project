"""Auto-planning: a small StateGraph that decides whether a task needs a plan."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class PlanningDecision(BaseModel):
    should_plan: bool = Field(description="是否应该先制订计划")
    reason: str = Field(description="判断理由")


class AutoPlanState(TypedDict, total=False):
    """Shared state for the auto-planning subgraph."""

    task: str
    should_plan: bool
    reason: str
    execution_plan: str


ROUTER_PROMPT = """
你是 Coding Agent 的任务路由器，只判断任务是否需要计划。

should_plan = true 的条件：
- 有四个及以上独立操作；
- 需要修改或运行代码；
- 涉及多个文件；
- 需要先分析再验证；
- 任务描述有歧义。

简单问答、单文件只读查询、天气查询等，should_plan = false。
""".strip()

PLANNER_PROMPT = """
你是一个谨慎的 Coding Agent 规划器，只负责制订计划，不执行工具。

请为任务输出 2 到 5 个按顺序执行的步骤。每个步骤要说明：
- 做什么；
- 为什么做；
- 是否会写文件或运行代码；
- 最后如何验证任务完成。

不要执行操作，不要声称任务已经完成。
""".strip()


def build_execution_plan(model: ChatDeepSeek, task: str) -> str:
    """Generate an execution plan directly with the planner prompt."""
    response = model.invoke(
        [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": f"请为下面任务制订执行计划：\n{task}"},
        ]
    )
    return str(response.content)


def build_auto_plan_graph(model: ChatDeepSeek):
    """Build the compiled auto-planning graph with the given model.

    The graph has two nodes:
      decide     -> asks the router model whether the task needs a plan
      build_plan -> generates the execution plan
    """

    def decide_planning_mode(task: str) -> PlanningDecision:
        router_model = model.with_structured_output(PlanningDecision)
        return router_model.invoke(
            [
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": f"任务：{task}"},
            ]
        )

    def decide_auto_plan_node(state: AutoPlanState):
        decision = decide_planning_mode(state["task"])
        return {"should_plan": decision.should_plan, "reason": decision.reason}

    def build_auto_plan_node(state: AutoPlanState):
        execution_plan = build_execution_plan(model, state["task"])
        return {"execution_plan": execution_plan}

    def route_after_auto_plan_decision(state: AutoPlanState) -> str:
        if state["should_plan"]:
            return "build_plan"
        return "finish"

    builder = StateGraph(AutoPlanState)
    builder.add_node("decide", decide_auto_plan_node)
    builder.add_node("build_plan", build_auto_plan_node)
    builder.add_edge(START, "decide")
    builder.add_conditional_edges(
        "decide",
        route_after_auto_plan_decision,
        {"build_plan": "build_plan", "finish": END},
    )
    builder.add_edge("build_plan", END)
    return builder.compile()


def create_planning_tool(agent: Any) -> tool:
    """Wrap the built planning graph as a tool callable by the main agent."""

    @tool("create_execution_plan", description=(
        "为复杂 Coding 任务制订计划，只返回计划，不执行任何操作。"
        "当任务包含四个及以上独立步骤、需要修改或运行代码、涉及多个文件，"
        "或需要先分析再验证时，可以调用此工具。"
    ))
    def create_execution_plan(task: str) -> str:
        result = agent.invoke({"task": task})
        return result.get("execution_plan", "")

    return create_execution_plan
