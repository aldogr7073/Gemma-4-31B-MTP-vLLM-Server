import pytest

from gemma4_mtp_vllm.server.bind_policy import bind_host_requires_api_key


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "LOCALHOST"])
def test_loopback_hosts_do_not_require_api_key(host):
    assert bind_host_requires_api_key(host) is False


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "192.168.1.50", "10.0.0.5", "example.com"],
)
def test_non_loopback_hosts_require_api_key(host):
    assert bind_host_requires_api_key(host) is True
