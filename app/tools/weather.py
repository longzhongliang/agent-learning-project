"""Read-only weather tool."""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request

from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气。city 支持中文或英文城市名。"""
    try:
        encoded_city = urllib.parse.quote(city)
        request = urllib.request.Request(
            f"https://wttr.in/{encoded_city}?format=%l:+%c+%t+%w+%h",
            headers={"User-Agent": "Coding-Agent/1.0"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            text = response.read().decode("utf-8").strip()
        if not text or "Unknown location" in text:
            return f"未找到城市「{city}」的天气信息，请确认城市名。"
        return f"{city} 当前天气：{text}"
    except urllib.error.HTTPError as exc:
        return f"天气查询失败：HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return f"天气查询失败：网络错误（{exc.reason}）"
    except Exception as exc:  # External services can fail in provider-specific ways.
        return f"天气查询失败：{type(exc).__name__} - {exc}"
