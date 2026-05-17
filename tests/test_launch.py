import json
import shlex

from gemma4_mtp_vllm.launch import build_vllm_serve_args
from gemma4_mtp_vllm.profiles import load_profiles, resolve_profile


def _profile():
    return resolve_profile("safe80", load_profiles())


def test_build_args_includes_target_and_speculative_config():
    profile = _profile()
    args = build_vllm_serve_args(profile=profile, host="127.0.0.1", port=8000)
    assert args[0] == "vllm"
    assert "serve" in args
    assert "google/gemma-4-31B-it" in args
    spec_idx = args.index("--speculative-config")
    spec = json.loads(args[spec_idx + 1])
    assert spec == {
        "method": "mtp",
        "model": profile.drafter,
        "num_speculative_tokens": profile.num_speculative_tokens,
    }
    assert "--tensor-parallel-size" in args
    assert "--max-model-len" in args
    assert "--gpu-memory-utilization" in args
    assert "--host" in args and "127.0.0.1" in args
    assert "--port" in args and "8000" in args


def test_build_args_enable_gemma4_reasoning_parser():
    args = build_vllm_serve_args(profile=_profile())
    parser_idx = args.index("--reasoning-parser")
    assert args[parser_idx + 1] == "gemma4"


def test_build_args_can_set_served_model_name():
    args = build_vllm_serve_args(
        profile=_profile(),
        served_model_name="gemma-4-31b-mtp",
    )
    served_idx = args.index("--served-model-name")
    assert args[served_idx + 1] == "gemma-4-31b-mtp"


def test_build_args_round_trip_through_shell_join():
    args = build_vllm_serve_args(profile=_profile())
    assert shlex.split(shlex.join(args)) == args


def test_build_args_can_disable_mtp_for_baseline():
    args = build_vllm_serve_args(
        profile=_profile(),
        host="127.0.0.1",
        port=8000,
        enable_mtp=False,
    )
    assert "--speculative-config" not in args
