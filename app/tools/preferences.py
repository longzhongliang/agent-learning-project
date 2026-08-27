"""Long-term user preference tools."""

from __future__ import annotations

from typing import Annotated, Any

from langchain.tools import InjectedStore
from langchain_core.tools import tool


@tool
def save_user_preference(key: str, value: str, store: Annotated[Any, InjectedStore()]) -> str:
    """把一条用户长期偏好保存到记忆中。"""
    store.put(("user_preferences",), key, {"value": value})
    return f"已记住：{key} = {value}"


@tool
def load_user_preference(key: str, store: Annotated[Any, InjectedStore()]) -> str:
    """读取一条已保存的用户长期偏好。"""
    item = store.get(("user_preferences",), key)
    if item is None:
        return f"没有找到名为「{key}」的长期记忆。"
    return f"{key} = {item.value['value']}"
