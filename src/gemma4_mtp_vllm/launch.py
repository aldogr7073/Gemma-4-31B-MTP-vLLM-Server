from __future__ import annotations

import json

from gemma4_mtp_vllm.profiles import ModelProfile


def build_vllm_serve_args(
    *,
    profile: ModelProfile,
    host: str = "127.0.0.1",
    port: int = 8000,
    enable_mtp: bool = True,
) -> list[str]:
    args: list[str] = [
        "vllm",
        "serve",
        profile.target,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(profile.tensor_parallel_size),
        "--max-model-len",
        str(profile.max_model_len),
        "--gpu-memory-utilization",
        f"{profile.gpu_memory_utilization:.2f}",
    ]
    if enable_mtp:
        spec = {
            "model": profile.drafter,
            "num_speculative_tokens": profile.num_speculative_tokens,
        }
        args.extend(["--speculative-config", json.dumps(spec)])
    return args
