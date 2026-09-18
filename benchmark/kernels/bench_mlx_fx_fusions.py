"""Compare fused/unfused real exported blocks, including the Torch/MLX bridge.

Example on Apple Silicon:
    python benchmark/kernels/bench_mlx_fx_fusions.py --tokens 1 128 1024

JSON lines include raw synchronized samples, graph-build and first-call cost,
and MLX allocator peaks. This is a block microbenchmark, not serving throughput.
"""

import argparse
import importlib.metadata
import json
import platform
import statistics
import time

import mlx.core as mx
import torch
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
)

from sglang.srt.hardware_backend.mlx.fx_lowering import (
    MlxFxLoweringRegistry,
    build_mlx_fx_plan,
    fuse_mlx_fx_plan,
    make_mlx_fx_executor,
)


class Rotary(torch.nn.Module):
    def forward(self, query, key, cosine, sine):
        return apply_rotary_pos_emb(query, key, cosine, sine)


def synchronize():
    # The bridge enqueues a Torch wait for the MLX completion event. Draining
    # that consumer stream includes MLX work without an extra MLX host wait.
    torch.mps.synchronize()


def timed_call(executor, inputs):
    synchronize()
    start = time.perf_counter()
    result = executor(*inputs)
    synchronize()
    return (time.perf_counter() - start) * 1000, result


def run_case(kind, tokens, dtype, args):
    torch.manual_seed(42)
    if kind == "rms_norm":
        module = Qwen3RMSNorm(args.width, eps=1e-6).to("mps", dtype).eval()
        inputs = (torch.randn(tokens, args.width, device="mps", dtype=dtype),)
    else:
        module = Rotary().eval()
        inputs = (
            torch.randn(
                1, args.heads, tokens, args.head_dim, device="mps", dtype=dtype
            ),
            torch.randn(
                1, args.heads, tokens, args.head_dim, device="mps", dtype=dtype
            ),
            torch.randn(1, tokens, args.head_dim, device="mps", dtype=dtype),
            torch.randn(1, tokens, args.head_dim, device="mps", dtype=dtype),
        )
    graph = torch.export.export(module, inputs).module(check_guards=False)
    plan = build_mlx_fx_plan(graph, MlxFxLoweringRegistry.standard_export_decoder())
    fused_plan = fuse_mlx_fx_plan(plan)
    fusion_count = sum(
        n.lowering.startswith("fused_") for n in fused_plan.nodes if n.lowering
    )
    assert fusion_count == (1 if kind == "rms_norm" else 2)
    with torch.no_grad():
        expected = module(*inputs)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    tolerance = 2e-6 if dtype == torch.float32 else 0.04
    rows, executors = {}, {}
    for enabled in (False, True):
        start = time.perf_counter()
        executor = make_mlx_fx_executor(plan, list(inputs), fuse_patterns=enabled)
        build_ms = (time.perf_counter() - start) * 1000
        first_ms, outputs = timed_call(executor, inputs)
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(
                actual.cpu(), reference.cpu(), atol=tolerance, rtol=tolerance
            )
        for _ in range(args.warmup):
            _, outputs = timed_call(executor, inputs)
        synchronize()
        executors[enabled] = executor
        rows[enabled] = {
            "enabled": enabled,
            "build_ms": build_ms,
            "first_call_ms": first_ms,
            "samples_ms": [],
            "peak_mlx_bytes": [],
            "active_before_bytes": [],
        }
    for repeat in range(args.repeats):
        for enabled in (False, True) if repeat % 2 == 0 else (True, False):
            synchronize()
            active = mx.get_active_memory()
            mx.reset_peak_memory()
            timings = []
            for _ in range(args.iterations):
                elapsed, outputs = timed_call(executors[enabled], inputs)
                timings.append(elapsed)
            rows[enabled]["samples_ms"].append(statistics.mean(timings))
            rows[enabled]["peak_mlx_bytes"].append(mx.get_peak_memory())
            rows[enabled]["active_before_bytes"].append(active)
    for row in rows.values():
        row["median_ms"] = statistics.median(row["samples_ms"])
    print(
        json.dumps(
            {
                "kind": kind,
                "tokens": tokens,
                "dtype": str(dtype),
                "width": args.width,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "fusion_count": fusion_count,
                "original_nodes": len(plan.nodes),
                "fused_nodes": len(fused_plan.nodes),
                "warmup": args.warmup,
                "repeats": args.repeats,
                "iterations": args.iterations,
                "unfused": rows[False],
                "fused": rows[True],
                "speedup": rows[False]["median_ms"] / rows[True]["median_ms"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 128, 1024])
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        nargs="+",
        default=["float32", "bfloat16"],
    )
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "environment": {
                    "torch": torch.__version__,
                    "mlx": importlib.metadata.version("mlx"),
                    "transformers": importlib.metadata.version("transformers"),
                    "platform": platform.platform(),
                }
            }
        ),
        flush=True,
    )
    with torch.no_grad():
        for name in args.dtype:
            for kind in ("rms_norm", "rotary"):
                for tokens in args.tokens:
                    run_case(kind, tokens, getattr(torch, name), args)
