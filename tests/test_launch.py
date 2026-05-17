from gemma4_mtp_vllm.launch import build_vllm_serve_args
from gemma4_mtp_vllm.profiles import load_profiles, resolve_profile


def _profile():
    return resolve_profile("safe80", load_profiles())


def test_build_args_includes_target_and_speculative_config():
    args = build_vllm_serve_args(profile=_profile(), host="127.0.0.1", port=8000)
    assert args[0] == "vllm"
    assert "serve" in args
    assert "google/gemma-4-31B-it" in args
    spec_idx = args.index("--speculative-config")
    assert "google/gemma-4-31B-it-assistant" in args[spec_idx + 1]
    assert "\"num_speculative_tokens\": 4" in args[spec_idx + 1]
    assert "--tensor-parallel-size" in args
    assert "--max-model-len" in args
    assert "--gpu-memory-utilization" in args
    assert "--host" in args and "127.0.0.1" in args
    assert "--port" in args and "8000" in args


def test_build_args_can_disable_mtp_for_baseline():
    args = build_vllm_serve_args(
        profile=_profile(),
        host="127.0.0.1",
        port=8000,
        enable_mtp=False,
    )
    assert "--speculative-config" not in args
