"""Structural and runtime qualification of opt-in exported-graph fusions."""

import copy
import importlib.util

import pytest
import torch

from sglang.srt.hardware_backend.mlx.fx_lowering import (
    MlxFxLoweringRegistry,
    build_mlx_fx_plan,
    fuse_mlx_fx_plan,
    make_mlx_fx_executor,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_mlx_ci(est_time=10, suite="stage-a-unit-test-mlx")

_METAL = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and importlib.util.find_spec("mlx")),
    reason="requires MLX on Apple Silicon",
)


def _plan(module, inputs):
    graph = torch.export.export(module, inputs).module(check_guards=False)
    return build_mlx_fx_plan(graph, MlxFxLoweringRegistry.standard_export_decoder())


def _rms(architecture, dtype):
    if architecture == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        norm = Qwen3RMSNorm(32, eps=1e-6)
    else:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        norm = LlamaRMSNorm(32, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-1.5, 1.5, 32))
    return norm.to(dtype).eval()


@pytest.mark.parametrize("architecture", ["qwen3", "llama"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_real_exported_rms_norm_fuses_without_mutating_graph(architecture, dtype):
    value = torch.randn(2, 5, 32, dtype=dtype)
    plan = _plan(_rms(architecture, dtype), (value,))
    code = plan.graph_module.code
    fused = fuse_mlx_fx_plan(plan)
    assert sum(n.lowering == "fused_rms_norm" for n in fused.nodes) == 1
    assert len(fused.nodes) < len(plan.nodes)
    assert plan.graph_module.code == code
    assert fused.graph_module.weight is plan.graph_module.weight
    torch.testing.assert_close(fused.graph_module(value), plan.graph_module(value))


@pytest.mark.parametrize(
    "change", ["power", "axis", "all_axes", "epsilon", "keepdim", "dtype"]
)
def test_rms_near_matches_are_left_unfused(change):
    value = torch.randn(2, 32)
    plan = _plan(_rms("qwen3", value.dtype), (value,))
    graph = copy.deepcopy(plan.graph_module.graph)
    for node in graph.nodes:
        if change == "power" and node.target == torch.ops.aten.pow.Tensor_Scalar:
            node.args = (node.args[0], 3)
        elif change == "axis" and node.target == torch.ops.aten.mean.dim:
            node.args = (node.args[0], [0], True)
        elif change == "all_axes" and node.target == torch.ops.aten.mean.dim:
            node.args = (node.args[0], None, True)
        elif change == "epsilon" and node.target == torch.ops.aten.add.Tensor:
            node.args = (node.args[0], -1e-6)
        elif change == "keepdim" and node.target == torch.ops.aten.mean.dim:
            node.args = (node.args[0], [-1], False)
        elif change == "dtype" and node.target == torch.ops.aten.to.dtype:
            node.args = (node.args[0], torch.float16)
    modified = build_mlx_fx_plan(
        torch.fx.GraphModule(plan.graph_module, graph),
        MlxFxLoweringRegistry.standard_export_decoder(),
    )
    fused = fuse_mlx_fx_plan(modified)
    assert all(n.lowering != "fused_rms_norm" for n in fused.nodes)


def test_missing_metadata_leaves_pattern_unfused():
    value = torch.randn(2, 32)
    plan = _plan(_rms("qwen3", value.dtype), (value,))
    for node in plan.graph_module.graph.nodes:
        node.meta.pop("val", None)
    assert all(n.lowering != "fused_rms_norm" for n in fuse_mlx_fx_plan(plan).nodes)


@_METAL
@pytest.mark.parametrize("architecture", ["qwen3", "llama"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_fused_and_unfused_executors_match_torch(architecture, dtype):
    torch.manual_seed(314)
    value = torch.randn(2, 17, 32, device="mps", dtype=dtype)
    module = _rms(architecture, dtype).to("mps")
    plan = _plan(module, (value,))
    executors = [
        make_mlx_fx_executor(plan, [value], fuse_patterns=enabled)
        for enabled in (False, True)
    ]
    tolerance = {torch.float32: 2e-6, torch.float16: 0.003, torch.bfloat16: 0.025}[
        dtype
    ]
    for scale in (1.0, 0.0, 100.0):
        current = value * scale
        expected = module(current)
        for executor in executors:
            actual = executor(current)[0]
            torch.mps.synchronize()
            assert actual.dtype == dtype
            torch.testing.assert_close(
                actual.cpu(), expected.cpu(), atol=tolerance, rtol=tolerance
            )
