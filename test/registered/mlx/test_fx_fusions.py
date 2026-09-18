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


def test_fusion_preserves_custom_registry_semantics():
    value = torch.randn(2, 32)
    original = _plan(_rms("qwen3", value.dtype), (value,))
    registry = MlxFxLoweringRegistry.standard_export_decoder()
    registry.register_function(torch.ops.aten.mul.Tensor, "custom_multiply")
    plan = build_mlx_fx_plan(original.graph_module, registry)
    fused = fuse_mlx_fx_plan(plan)
    assert not any(node.lowering == "fused_rms_norm" for node in fused.nodes)
    assert any(node.lowering == "custom_multiply" for node in fused.nodes)


def test_fusion_does_not_erase_unrelated_unused_custom_calls():
    calls = []

    def side_effect(value):
        calls.append(1)
        return value

    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    graph.call_function(side_effect, (value,))
    graph.output(value)
    registry = MlxFxLoweringRegistry.standard_export_decoder()
    registry.register_function(side_effect, "custom_side_effect")
    plan = build_mlx_fx_plan(torch.fx.GraphModule({}, graph), registry)
    fused = fuse_mlx_fx_plan(plan)
    fused.graph_module(torch.ones(1))
    assert calls == [1]
    assert len(fused.nodes) == len(plan.nodes)


def test_fusion_only_prunes_metadata_checks_on_matched_intermediates():
    value = torch.randn(2, 32)
    module = torch.nn.Sequential(_rms("qwen3", value.dtype), _rms("llama", value.dtype))
    plan = _plan(module, (value,))
    graph = plan.graph_module.graph
    checks = set()
    unrelated = None
    for node in list(graph.nodes):
        if node.op == "placeholder" or node.target == torch.ops.aten.pow.Tensor_Scalar:
            with graph.inserting_after(node):
                check = graph.call_function(
                    torch.ops.aten._assert_tensor_metadata.default, (node,)
                )
            if node.op == "placeholder":
                unrelated = check.name
            else:
                checks.add(check.name)
    assert len(checks) == 2
    plan.graph_module.recompile()
    plan = build_mlx_fx_plan(
        plan.graph_module, MlxFxLoweringRegistry.standard_export_decoder()
    )
    fused = fuse_mlx_fx_plan(plan)
    names = {node.node_name for node in fused.nodes}
    assert sum(node.lowering == "fused_rms_norm" for node in fused.nodes) == 2
    assert unrelated in names
    assert checks.isdisjoint(names)
    torch.testing.assert_close(fused.graph_module(value), plan.graph_module(value))


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


class _Rotary(torch.nn.Module):
    def __init__(self, architecture="qwen3", unsqueeze_dim=1):
        super().__init__()
        if architecture == "qwen3":
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        else:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        self.apply_rotary = apply_rotary_pos_emb
        self.unsqueeze_dim = unsqueeze_dim

    def forward(self, query, key, cosine, sine):
        return self.apply_rotary(query, key, cosine, sine, self.unsqueeze_dim)


@pytest.mark.parametrize("architecture", ["qwen3", "llama"])
@pytest.mark.parametrize("unsqueeze_dim", [1, 2])
def test_real_rope_exports_fuse_both_q_and_k(architecture, unsqueeze_dim):
    shape = (2, 4, 7, 32) if unsqueeze_dim == 1 else (2, 7, 4, 32)
    inputs = (
        torch.randn(shape),
        torch.randn(shape),
        torch.randn(2, 7, 32),
        torch.randn(2, 7, 32),
    )
    plan = _plan(_Rotary(architecture, unsqueeze_dim), inputs)
    code = plan.graph_module.code
    fused = fuse_mlx_fx_plan(plan)
    assert sum(n.lowering == "fused_rotary_embedding" for n in fused.nodes) == 2
    assert plan.graph_module.code == code
    torch.testing.assert_close(fused.graph_module(*inputs), plan.graph_module(*inputs))


@pytest.mark.parametrize("change", ["reverse", "axis", "slice", "alpha", "table_dtype"])
def test_rope_near_matches_are_not_fused(change):
    inputs = (torch.randn(2, 4, 7, 32),) * 2 + (torch.randn(2, 7, 32),) * 2
    plan = _plan(_Rotary(), inputs)
    for node in plan.graph_module.graph.nodes:
        if change == "reverse" and node.target == torch.ops.aten.cat.default:
            node.args = (list(reversed(node.args[0])), -1)
        elif change == "axis" and node.target == torch.ops.aten.cat.default:
            node.args = (node.args[0], 2)
        elif change == "slice" and node.target == torch.ops.aten.slice.Tensor:
            node.args = (*node.args[:4], 2)
        elif change == "alpha" and node.target == torch.ops.aten.add.Tensor:
            node.kwargs = {"alpha": 2}
        elif (
            change == "table_dtype" and node.target == torch.ops.aten.unsqueeze.default
        ):
            node.meta["val"] = node.meta["val"].to(torch.float16)
    assert all(
        n.lowering != "fused_rotary_embedding" for n in fuse_mlx_fx_plan(plan).nodes
    )


@_METAL
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["batch", "shared", "transposed"])
def test_cached_rope_fused_unfused_and_torch_parity(dtype, layout):
    torch.manual_seed(321)
    unsqueeze_dim = 2 if layout == "transposed" else 1
    query = torch.randn(2, 3, 7, 32, device="mps", dtype=dtype)
    key = torch.randn_like(query)
    if layout == "transposed":
        query, key = query.transpose(1, 2), key.transpose(1, 2)
    batch = 1 if layout == "shared" else 2
    # Arbitrary, independent tables deliberately do not describe frequencies.
    cosine = torch.randn(batch, 7, 32, device="mps", dtype=dtype)
    sine = torch.randn_like(cosine)
    module = _Rotary(unsqueeze_dim=unsqueeze_dim)
    inputs = (query, key, cosine, sine)
    plan = _plan(module, inputs)
    fused = fuse_mlx_fx_plan(plan)
    assert sum(n.lowering == "fused_rotary_embedding" for n in fused.nodes) == 2
    executors = [
        make_mlx_fx_executor(plan, list(inputs), fuse_patterns=v) for v in (False, True)
    ]
    tolerance = {torch.float32: 2e-6, torch.float16: 0.004, torch.bfloat16: 0.04}[dtype]
    for factor in (1.0, -0.5):
        current = (query, key, cosine * factor, sine - factor)
        expected = module(*current)
        for executor in executors:
            actual = executor(*current)
            torch.mps.synchronize()
            for result, reference in zip(actual, expected):
                assert result.dtype == dtype
                torch.testing.assert_close(
                    result.cpu(), reference.cpu(), atol=tolerance, rtol=tolerance
                )


@_METAL
@pytest.mark.parametrize("explicit_plan", [False, True])
def test_fused_executor_keeps_original_attribute_owner(explicit_plan):
    value = torch.randn(2, 32, device="mps")
    plan = _plan(_rms("qwen3", value.dtype).to("mps"), (value,))
    source = plan.graph_module
    if explicit_plan:
        plan = fuse_mlx_fx_plan(plan)
    executor = make_mlx_fx_executor(plan, [value], fuse_patterns=not explicit_plan)
    executor(value)
    for replace in (False, True):
        if replace:
            source.weight = torch.nn.Parameter(torch.randn_like(source.weight))
        else:
            with torch.no_grad():
                source.weight.add_(0.5)
        actual = executor(value)[0]
        expected = source(value)
        torch.mps.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-6, rtol=2e-6)


@_METAL
@pytest.mark.parametrize("changed", ["shape", "dtype", "weight_dtype"])
def test_fusion_rejects_changed_runtime_signature(changed):
    value = torch.randn(2, 32, device="mps")
    plan = _plan(_rms("qwen3", value.dtype).to("mps"), (value,))
    executor = make_mlx_fx_executor(plan, [value], fuse_patterns=True)
    if changed == "shape":
        value = value[:1]
    elif changed == "dtype":
        value = value.half()
    else:
        plan.graph_module.weight = torch.nn.Parameter(plan.graph_module.weight.half())
    with pytest.raises(RuntimeError, match="shape/dtype changed"):
        executor(value)


def test_fusion_keeps_shared_rms_intermediate_alive():
    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(32))

        def forward(self, value):
            promoted = value.float()
            variance = promoted.pow(2).mean(-1, keepdim=True)
            normalized = promoted * torch.rsqrt(variance + 1e-6)
            return normalized.to(value.dtype) * self.weight, variance

    value = torch.randn(2, 32)
    plan = _plan(Shared(), (value,))
    fused = fuse_mlx_fx_plan(plan)
    assert any(node.lowering == "fused_rms_norm" for node in fused.nodes)
    assert any(node.lowering == "mean" for node in fused.nodes)
    torch.testing.assert_close(fused.graph_module(value), plan.graph_module(value))
