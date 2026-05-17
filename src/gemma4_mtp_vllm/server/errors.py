from __future__ import annotations

from fastapi.responses import JSONResponse


_DEFAULT_OPENAI_TYPE = "invalid_request_error"


def protocol_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    protocol: str = "openai",
) -> JSONResponse:
    if protocol == "anthropic":
        body = {
            "type": "error",
            "error": {"type": code, "message": message},
        }
    else:
        body = {
            "error": {
                "code": code,
                "message": message,
                "type": _DEFAULT_OPENAI_TYPE,
            }
        }
    return JSONResponse(status_code=status_code, content=body)
