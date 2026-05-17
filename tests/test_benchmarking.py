from __future__ import annotations

import json

import pytest

from gemma4_mtp_vllm.benchmarking import (
    BenchmarkObservation,
    BenchmarkSummary,
    deterministic_parity,
    median_optional,
    speedup,
)


def test_speedup_returns_ratio_when_both_positive():
    assert speedup(10.0, 20.0) == pytest.approx(2.0)


def test_speedup_returns_none_when_baseline_missing():
    assert speedup(0.0, 10.0) is None
    assert speedup(None, 10.0) is None
    assert speedup(10.0, None) is None


@pytest.mark.parametrize(
    "no_draft, mtp, expected",
    [
        ("hello", "hello", True),
        ("hello", "world", False),
    ],
)
def test_deterministic_parity_greedy(no_draft, mtp, expected):
    assert deterministic_parity(no_draft, mtp, temperature=0.0, top_p=1.0) is expected


def test_deterministic_parity_returns_none_when_sampling():
    assert deterministic_parity("a", "b", temperature=0.7, top_p=1.0) is None


def test_median_optional_returns_value():
    assert median_optional([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert median_optional([]) is None
    assert median_optional([None]) is None


def test_benchmark_summary_to_json_roundtrip():
    summary = BenchmarkSummary(
        profile="safe80",
        prompt_name="default",
        prompt="Hello",
        num_speculative_tokens=4,
        observations=[
            BenchmarkObservation(
                index=1,
                no_draft_generation_tps=10.0,
                mtp_generation_tps=20.0,
                speedup=2.0,
                deterministic_parity=True,
            )
        ],
        median_no_draft_generation_tps=10.0,
        median_mtp_generation_tps=20.0,
        median_speedup=2.0,
    )
    body = json.loads(json.dumps(summary.to_dict()))
    assert body["profile"] == "safe80"
    assert body["observations"][0]["speedup"] == 2.0
