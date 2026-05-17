import httpx
from fastapi.testclient import TestClient

from gemma4_mtp_vllm.server.app import create_app


def _client():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    app = create_app(
        api_key="secret",
        vllm_base_url="http://vllm.local:8000",
        vllm_transport=httpx.MockTransport(handler),
    )
    return TestClient(app)


def test_metrics_requires_auth():
    client = _client()
    unauthorized = client.get("/metrics")
    authorized = client.get("/metrics", headers={"x-api-key": "secret"})

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    body = authorized.text
    assert "gemma4_mtp_active_requests" in body
    assert "gemma4_mtp_generation_tokens_total" in body
    assert "gemma4_mtp_backend_errors" in body
    assert authorized.headers["content-type"].startswith("text/plain")
