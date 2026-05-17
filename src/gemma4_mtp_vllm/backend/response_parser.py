from __future__ import annotations

from typing import Any


def visible_text_for_history(text: str) -> str:
    return text


def finish_reason_from_openai(choice: dict[str, Any]) -> str:
    reason = choice.get("finish_reason") or "stop"
    return str(reason)


def usage_from_openai(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }
