from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Optional

import httpx
import typer

from gemma4_mtp_vllm import __version__
from gemma4_mtp_vllm.doctor import build_report
from gemma4_mtp_vllm.launch import build_vllm_serve_args
from gemma4_mtp_vllm.profiles import (
    ProfileSet,
    load_profiles,
    resolve_profile,
)
from gemma4_mtp_vllm.server.bind_policy import bind_host_requires_api_key
from gemma4_mtp_vllm.server.limits import ServerLimits

app = typer.Typer(add_completion=False, help="Gemma 4 31B MTP vLLM sidecar gateway")


def _profile_set() -> ProfileSet:
    return load_profiles()


def _mock_transport():
    """Test-only hook overridden in tests when VLLM_MTP_TRANSPORT_MOCK=1."""
    return None


def _build_transport() -> httpx.BaseTransport | None:
    if os.environ.get("VLLM_MTP_TRANSPORT_MOCK") == "1":
        return _mock_transport()
    return None


@app.command()
def doctor(
    profile: str = typer.Option("safe80", "--profile"),
    vllm_base_url: str = typer.Option(
        "http://127.0.0.1:8000", "--vllm-base-url"
    ),
) -> None:
    profile_set = _profile_set()
    selected = resolve_profile(profile, profile_set)
    transport = _build_transport()
    report = asyncio.run(
        build_report(
            profile=selected,
            vllm_base_url=vllm_base_url,
            transport=transport,
        )
    )
    # Emit single-line JSON so the test seam (splitlines()[-1]) yields a
    # parseable payload; multi-line indented output would leave the test
    # parsing just the closing brace.
    typer.echo(json.dumps(report))


@app.command()
def launch(
    profile: str = typer.Option("safe80", "--profile"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    print_only: bool = typer.Option(False, "--print-only"),
    no_mtp: bool = typer.Option(False, "--no-mtp"),
) -> None:
    selected = resolve_profile(profile, _profile_set())
    args = build_vllm_serve_args(
        profile=selected,
        host=host,
        port=port,
        enable_mtp=not no_mtp,
    )
    if print_only:
        typer.echo(" ".join(args))
        return
    os.execvp(args[0], args)


@app.command()
def serve(
    profile: str = typer.Option("safe80", "--profile"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    max_body_mb: float = typer.Option(2.0, "--max-body-mb"),
    max_output_tokens: int = typer.Option(4096, "--max-output-tokens"),
    max_queue_size: int = typer.Option(8, "--max-queue-size"),
    rate_limit_rpm: int = typer.Option(30, "--rate-limit-rpm"),
    vllm_base_url: str = typer.Option(
        "http://127.0.0.1:8000", "--vllm-base-url"
    ),
    cors_origin: list[str] = typer.Option([], "--cors-origin"),
) -> None:
    if bind_host_requires_api_key(host) and not api_key:
        typer.echo(f"host {host} requires --api-key", err=True)
        raise typer.Exit(code=1)

    import uvicorn

    from gemma4_mtp_vllm.server.app import create_app

    limits = ServerLimits(
        max_body_bytes=int(max_body_mb * 1024 * 1024),
        max_output_tokens=max_output_tokens,
        max_queue_size=max_queue_size,
        rate_limit_rpm=rate_limit_rpm,
        cors_origins=tuple(cors_origin),
    )
    selected_profile_name = profile
    fastapi_app = create_app(
        profile_name=selected_profile_name,
        bind_host=host,
        api_key=api_key,
        limits=limits,
        vllm_base_url=vllm_base_url,
    )
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command()
def generate(
    prompt: str = typer.Argument(...),
    profile: str = typer.Option("safe80", "--profile"),
    max_tokens: int = typer.Option(64, "--max-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    top_p: float = typer.Option(1.0, "--top-p"),
    vllm_base_url: str = typer.Option(
        "http://127.0.0.1:8000", "--vllm-base-url"
    ),
    no_mtp: bool = typer.Option(False, "--no-mtp", help="Disabled in v0.1: requires separate vLLM launch."),
) -> None:
    """One-shot generation via the configured vLLM server."""
    if no_mtp:
        typer.echo(
            "--no-mtp requires launching a separate vLLM process without "
            "--speculative-config; see `vllm-mtp bench` for paired runs.",
            err=True,
        )
        raise typer.Exit(code=2)

    selected = resolve_profile(profile, _profile_set())

    async def run() -> dict:
        async with httpx.AsyncClient(base_url=vllm_base_url, timeout=120.0) as http:
            response = await http.post(
                "/v1/chat/completions",
                json={
                    "model": selected.target,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            )
            response.raise_for_status()
            return response.json()

    payload = asyncio.run(run())
    text = payload["choices"][0]["message"]["content"]
    typer.echo(text)


@app.command()
def bench(
    prompt: str = typer.Option(..., "--prompt"),
    profile: str = typer.Option("safe80", "--profile"),
    max_tokens: int = typer.Option(64, "--max-tokens"),
    mtp_url: str = typer.Option(..., "--mtp-url"),
    baseline_url: str = typer.Option(..., "--baseline-url"),
    runs: int = typer.Option(3, "--runs"),
    warmup_runs: int = typer.Option(1, "--warmup-runs"),
    json_output: Optional[str] = typer.Option(None, "--json-output"),
) -> None:
    """Filled in Task 18."""
    typer.echo("bench: pending Task 18", err=True)
    raise typer.Exit(code=2)


@app.command("bench-matrix")
def bench_matrix(
    profile: str = typer.Option("safe80", "--profile"),
    mtp_url: str = typer.Option(..., "--mtp-url"),
    baseline_url: str = typer.Option(..., "--baseline-url"),
    prompt: list[str] = typer.Option([], "--prompt"),
    num_speculative_tokens: list[int] = typer.Option(
        [], "--num-speculative-tokens"
    ),
    runs: int = typer.Option(3, "--runs"),
    warmup_runs: int = typer.Option(1, "--warmup-runs"),
    json_output: Optional[str] = typer.Option(None, "--json-output"),
) -> None:
    """Filled in Task 18."""
    typer.echo("bench-matrix: pending Task 18", err=True)
    raise typer.Exit(code=2)
