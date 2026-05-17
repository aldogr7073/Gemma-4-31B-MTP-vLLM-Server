from __future__ import annotations

import json

from gemma4_mtp_vllm.profiles import ModelProfile


def build_vllm_serve_args(
    *,
    profile: ModelProfile,
    host: str = "127.0.0.1",
    port: int = 8000,
    enable_mtp: bool = True,
    served_model_name: str | None = None,
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
        "--reasoning-parser",
        "gemma4",
    ]
    if served_model_name:
        args.extend(["--served-model-name", served_model_name])
    if enable_mtp:
        spec = {
            "method": "mtp",
            "model": profile.drafter,
            "num_speculative_tokens": profile.num_speculative_tokens,
        }
        args.extend(["--speculative-config", json.dumps(spec, separators=(",", ":"))])
    return args
