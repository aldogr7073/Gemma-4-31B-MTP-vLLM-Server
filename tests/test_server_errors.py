import json

from gemma4_mtp_vllm.server.errors import protocol_error_response


def test_openai_error_shape():
    response = protocol_error_response(
        status_code=400,
        code="invalid_request",
        message="boom",
    )
    assert response.status_code == 400
    body = json.loads(response.body)
    assert body == {
        "error": {
            "code": "invalid_request",
            "message": "boom",
            "type": "invalid_request_error",
        }
    }


def test_anthropic_error_shape():
    response = protocol_error_response(
        status_code=429,
        code="rate_limited",
        message="slow down",
        protocol="anthropic",
    )
    assert response.status_code == 429
    body = json.loads(response.body)
    assert body == {
        "type": "error",
        "error": {
            "type": "rate_limited",
            "message": "slow down",
        },
    }
