from __future__ import annotations


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def bind_host_requires_api_key(host: str) -> bool:
    normalized = host.strip().casefold()
    return normalized not in _LOOPBACK_HOSTS
