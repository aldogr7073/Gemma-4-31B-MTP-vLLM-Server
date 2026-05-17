from __future__ import annotations

import re

_SEMVER_PREFIX_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


def version_at_least(value: str | None, minimum: str) -> bool:
    parsed_value = _parse_semver_prefix(value)
    parsed_minimum = _parse_semver_prefix(minimum)
    if parsed_value is None or parsed_minimum is None:
        return False
    return parsed_value >= parsed_minimum


def _parse_semver_prefix(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None

    match = _SEMVER_PREFIX_RE.match(value)
    if match is None:
        return None

    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)
