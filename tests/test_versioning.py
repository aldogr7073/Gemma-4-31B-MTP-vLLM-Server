from __future__ import annotations

from gemma4_mtp_vllm.versioning import version_at_least


def test_version_at_least_accepts_equal_version() -> None:
    assert version_at_least("0.21.0", "0.21.0")


def test_version_at_least_accepts_newer_patch() -> None:
    assert version_at_least("0.21.1", "0.21.0")


def test_version_at_least_rejects_older_minor() -> None:
    assert not version_at_least("0.20.2", "0.21.0")


def test_version_at_least_ignores_suffix() -> None:
    assert version_at_least("0.21.0+cu129", "0.21.0")


def test_version_at_least_rejects_missing_value() -> None:
    assert not version_at_least(None, "0.21.0")


def test_version_at_least_rejects_unparsable_value() -> None:
    assert not version_at_least("not-a-version", "0.21.0")
