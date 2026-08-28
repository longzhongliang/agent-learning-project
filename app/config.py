"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Raised when a required application setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    model_name: str
    model_timeout_seconds: int
    model_max_retries: int
    workspace_dir: Path
    database_path: Path
    run_timeout_seconds: int
    enable_mcp_tools: bool


def _resolve_path(value: str) -> Path:
    """Resolve paths relative to the directory from which the app is started."""
    return Path(value).expanduser().resolve()


def load_settings() -> Settings:
    """Load and validate settings, creating only the required local directories."""
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "缺少 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 后填写密钥。"
        )

    timeout_text = os.getenv("AGENT_RUN_TIMEOUT_SECONDS", "20")
    try:
        timeout = int(timeout_text)
    except ValueError as exc:
        raise ConfigurationError("AGENT_RUN_TIMEOUT_SECONDS 必须是整数。") from exc
    if timeout < 1:
        raise ConfigurationError("AGENT_RUN_TIMEOUT_SECONDS 必须大于 0。")

    model_timeout_text = os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "90")
    try:
        model_timeout = int(model_timeout_text)
    except ValueError as exc:
        raise ConfigurationError("DEEPSEEK_TIMEOUT_SECONDS 必须是整数。") from exc
    if model_timeout < 1:
        raise ConfigurationError("DEEPSEEK_TIMEOUT_SECONDS 必须大于 0。")

    model_retries_text = os.getenv("DEEPSEEK_MAX_RETRIES", "8")
    try:
        model_retries = int(model_retries_text)
    except ValueError as exc:
        raise ConfigurationError("DEEPSEEK_MAX_RETRIES 必须是整数。") from exc
    if model_retries < 0:
        raise ConfigurationError("DEEPSEEK_MAX_RETRIES 不能小于 0。")

    workspace_dir = _resolve_path(os.getenv("AGENT_WORKSPACE_DIR", "./workspace"))
    database_path = _resolve_path(os.getenv("AGENT_DB_PATH", "./data/agent.db"))
    workspace_dir.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["DEEPSEEK_API_KEY"] = api_key

    return Settings(
        deepseek_api_key=api_key,
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        model_timeout_seconds=model_timeout,
        model_max_retries=model_retries,
        workspace_dir=workspace_dir,
        database_path=database_path,
        run_timeout_seconds=timeout,
        enable_mcp_tools=os.getenv("AGENT_ENABLE_MCP_TOOLS", "false").strip().lower()
        in {"1", "true", "yes", "on"},
    )
