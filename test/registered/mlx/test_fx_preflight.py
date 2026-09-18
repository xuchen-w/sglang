"""Admission checks run before the generic FX executor borrows any tensors."""

import operator
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from packaging.version import Version

from sglang.srt.hardware_backend.mlx.fx_lowering import (
    MlxFxCaptureBackend,
    MlxFxLoweringRegistry,
    UnsupportedMlxFxGraphError,
    build_mlx_fx_plan,
    make_mlx_fx_executor,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")

_REGISTRY = MlxFxLoweringRegistry.standard_export_decoder()
_HAS_RUNTIME = torch.backends.mps.is_available() and Version(
    torch.__version__
) >= Version("2.13")


@pytest.fixture(autouse=True)
def _preserve_random_state():
    with torch.random.fork_rng(devices=[]):
        yield


def _graph(target, inputs, *, constants=(), kwargs=None):
    graph = torch.fx.Graph()
    nodes = tuple(graph.placeholder(f"input_{index}") for index in range(len(inputs)))
    result = graph.call_function(target, (*nodes, *constants), kwargs or {})
    graph.output(result)
    return torch.fx.GraphModule({}, graph)


def _sdpa_graph(inputs, **kwargs):
    return _graph(
        torch.ops.aten.scaled_dot_product_attention.default, inputs, kwargs=kwargs
    )


def _qkv(q=(2, 4, 3, 8), k=(2, 4, 5, 8), v=(2, 4, 5, 6), dtype=torch.float32):
    generator = torch.Generator().manual_seed(42)
    return [torch.randn(shape, dtype=dtype, generator=generator) for shape in (q, k, v)]


@pytest.mark.parametrize(
    "inputs,kwargs,reason",
    [
        (_qkv(), {"dropout_p": 0.1}, "dropout"),
        (
            _qkv(),
            {"is_causal": True, "attn_mask": torch.ones(3, 5, dtype=torch.bool)},
            "causal",
        ),
        (_qkv(), {"attn_mask": torch.ones(3, 5, dtype=torch.int32)}, "mask dtype"),
        (_qkv(dtype=torch.float64), {}, "floating-point"),
        ([_qkv()[0], _qkv()[1].half(), _qkv()[2]], {}, "matching"),
        (_qkv(q=(8,)), {}, "matrix axes"),
        (_qkv(k=(2, 4, 5, 7)), {}, "feature dimensions"),
        (_qkv(v=(2, 4, 6, 6)), {}, "sequence lengths"),
        (_qkv(v=(3, 4, 5, 6)), {}, "batch/head"),
        (_qkv(), {"attn_mask": torch.ones(4, 5)}, "broadcastable"),
        (_qkv(), {"attn_mask": torch.ones(1, 2, 4, 3, 5)}, "broadcastable"),
        (
            _qkv(q=(1, 4, 3, 8), k=(1, 4, 5, 8), v=(1, 4, 5, 6)),
            {"attn_mask": torch.ones(2, 4, 3, 5)},
            "broadcastable",
        ),
        (_qkv(k=(2, 3, 5, 8)), {"enable_gqa": True}, "divide"),
        (_qkv(k=(2, 0, 5, 8)), {"enable_gqa": True}, "divide"),
    ],
)
def test_sdpa_rejects_provably_invalid_metadata(inputs, kwargs, reason):
    graph = _sdpa_graph(inputs, **kwargs)
    plan = build_mlx_fx_plan(graph, _REGISTRY, example_inputs=inputs)
    assert not plan.fully_supported
    assert len(plan.unsupported) == 1
    assert reason in plan.unsupported[0].rejection_reason
    with pytest.raises(UnsupportedMlxFxGraphError, match=reason):
        plan.require_fully_supported()


@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sdpa_admits_broadcast_and_independent_gqa_heads(gqa, dtype):
    inputs = _qkv(
        k=(1, 1 if gqa else 4, 5, 8), v=(2, 2 if gqa else 4, 5, 6), dtype=dtype
    )
    graph = _sdpa_graph(
        inputs, attn_mask=torch.ones(1, 1, 3, 5, dtype=torch.bool), enable_gqa=gqa
    )
    assert build_mlx_fx_plan(graph, _REGISTRY, example_inputs=inputs).fully_supported


@pytest.mark.parametrize(
    "target", [torch.ops.aten.rms_norm.default, torch.ops.aten.layer_norm.default]
)
@pytest.mark.parametrize(
    "shape,weight,reason",
    [
        ((3, 8), torch.ones(3, 8), "single last axis"),
        ((7,), torch.ones(7), "last input axis"),
        ((8,), torch.ones(7), "parameter shape"),
    ],
)
def test_norm_rejects_unsupported_axes_and_parameter_shapes(
    target, shape, weight, reason
):
    inputs = [torch.empty(2, 3, 8)]
    graph = _graph(target, inputs, constants=(shape,), kwargs={"weight": weight})
    with pytest.raises(UnsupportedMlxFxGraphError, match=reason):
        build_mlx_fx_plan(
            graph, _REGISTRY, example_inputs=inputs
        ).require_fully_supported()


def test_rms_norm_rejects_missing_weight_before_executor_factory():
    inputs = [torch.randn(2, 8)]
    graph = _graph(torch.ops.aten.rms_norm.default, inputs, constants=([8],))
    factory = mock.Mock()
    backend = MlxFxCaptureBackend(
        _REGISTRY, executor_factory=factory, fallback_to_torch=True
    )
    execute = backend(graph, inputs)
    torch.testing.assert_close(execute(*inputs), graph(*inputs))
    factory.assert_not_called()


@pytest.mark.parametrize(
    "target", [torch.ops.aten.rms_norm.default, torch.ops.aten.layer_norm.default]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_norm_rejects_mixed_parameter_dtype(target, dtype):
    inputs = [torch.ones(2, 8, dtype=dtype)]
    graph = _graph(target, inputs, constants=([8],), kwargs={"weight": torch.ones(8)})
    with pytest.raises(
        UnsupportedMlxFxGraphError, match="matching input and parameter dtypes"
    ):
        build_mlx_fx_plan(
            graph, _REGISTRY, example_inputs=inputs
        ).require_fully_supported()


def test_layer_norm_rejects_mixed_bias_dtype():
    inputs = [torch.ones(2, 8, dtype=torch.float16)]
    graph = _graph(
        torch.ops.aten.layer_norm.default,
        inputs,
        constants=([8],),
        kwargs={"bias": torch.ones(8)},
    )
    assert not build_mlx_fx_plan(
        graph, _REGISTRY, example_inputs=inputs
    ).fully_supported


def test_used_scalar_input_is_rejected_before_attribute_borrow():
    root = torch.nn.Module()
    root.register_buffer("weight", torch.ones(8))
    graph = torch.fx.Graph()
    value, factor = [graph.placeholder(name) for name in ("value", "factor")]
    weight = graph.get_attr("weight")
    value = graph.call_function(torch.ops.aten.add.Tensor, (value, weight))
    value = graph.call_function(torch.ops.aten.mul.Tensor, (value, factor))
    graph.output(value)
    module = torch.fx.GraphModule(root, graph)
    plan = build_mlx_fx_plan(module, _REGISTRY)
    with mock.patch("sglang.srt.utils.tensor_bridge.MlxTensorView") as borrow:
        with pytest.raises(UnsupportedMlxFxGraphError, match="used symbolic scalar"):
            make_mlx_fx_executor(plan, [torch.ones(8), 2.0])
        borrow.assert_not_called()


@pytest.mark.parametrize(
    "target,constants,kwargs",
    [
        (torch.ops.aten.to.dtype, (torch.float64,), {}),
        (torch.ops.aten.mean.dim, ([0],), {"dtype": torch.float64}),
        (torch.ops.aten.empty_like.default, (), {"dtype": torch.float64}),
    ],
)
def test_unsupported_dtype_is_rejected_before_borrowing(target, constants, kwargs):
    inputs = [torch.ones(2, 8)]
    graph = _graph(target, inputs, constants=constants, kwargs=kwargs)
    # Name-only planning can still discover constant argument incompatibilities.
    plan = build_mlx_fx_plan(graph, _REGISTRY)
    with mock.patch("sglang.srt.utils.tensor_bridge.MlxTensorView") as borrow:
        with pytest.raises(UnsupportedMlxFxGraphError, match="dtype"):
            make_mlx_fx_executor(plan, inputs)
        borrow.assert_not_called()


def test_static_plan_reads_metadata_without_requiring_example_inputs():
    inputs = _qkv(k=(2, 4, 5, 7))
    graph = _sdpa_graph(inputs)
    for node, tensor in zip(graph.graph.nodes, inputs):
        node.meta["val"] = tensor.to("meta")
    plan = build_mlx_fx_plan(graph, _REGISTRY)
    assert "feature dimensions" in plan.unsupported[0].rejection_reason


def test_missing_and_symbolic_metadata_do_not_create_false_rejections():
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    graph = _sdpa_graph(_qkv())
    assert build_mlx_fx_plan(graph, _REGISTRY).fully_supported
    symbolic = ShapeEnv().create_unbacked_symint()
    for node in list(graph.graph.nodes)[:3]:
        node.meta["tensor_meta"] = SimpleNamespace(
            shape=(2, 4, symbolic, 8), dtype=torch.float32
        )
    assert build_mlx_fx_plan(graph, _REGISTRY).fully_supported


def test_fresh_inference_ignores_stale_intermediate_metadata():
    graph = torch.fx.Graph()
    q, k, v = [graph.placeholder(name) for name in ("q", "k", "v")]
    q = graph.call_function(torch.ops.aten.alias.default, (q,))
    q.meta["val"] = torch.empty(2, 4, 3, 8, device="meta")
    output = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default, (q, k, v)
    )
    graph.output(output)
    module = torch.fx.GraphModule({}, graph)
    plan = build_mlx_fx_plan(module, _REGISTRY)
    assert plan.fully_supported
    bad_inputs = _qkv(q=(2, 4, 3, 7))
    assert (
        "feature dimensions"
        in plan.for_inputs(bad_inputs).unsupported[0].rejection_reason
    )
    assert q.meta["val"].shape[-1] == 8


def test_custom_registered_function_is_not_executed_for_metadata():
    calls = []

    def custom(value):
        calls.append(value)
        return value

    registry = MlxFxLoweringRegistry.standard_export_decoder()
    registry.register_function(custom, "sdpa")
    # A consumer-owned semantic op is not the built-in generic SDPA schema.
    graph = _graph(custom, [torch.ones(2)])
    plan = build_mlx_fx_plan(graph, registry, example_inputs=[torch.ones(2)])
    assert plan.fully_supported
    assert not calls


def test_runtime_fallback_rechecks_metadata_before_calling_executor():
    inputs = [torch.randn(2, 8)]
    graph = _graph(torch.ops.aten.layer_norm.default, inputs, constants=([8],))
    executed = []

    def factory(plan, examples):
        def execute(*args):
            executed.append(args)
            return graph(*args)

        return execute

    backend = MlxFxCaptureBackend(
        _REGISTRY, executor_factory=factory, fallback_to_torch=True
    )
    execute = backend(graph, inputs)
    torch.testing.assert_close(execute(*inputs), graph(*inputs))
    # float64 is valid Torch LayerNorm but outside MLX Metal's dtype subset.
    new_input = inputs[0].double()
    torch.testing.assert_close(execute(new_input), graph(new_input))
    assert len(executed) == 1


def test_runtime_fallback_uses_changed_captured_constant_before_executor():
    root = torch.nn.Module()
    root.normalized_shape = [8]
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    shape = graph.get_attr("normalized_shape")
    output = graph.call_function(torch.ops.aten.layer_norm.default, (value, shape))
    graph.output(output)
    module = torch.fx.GraphModule(root, graph)
    called = mock.Mock(side_effect=module.forward)
    backend = MlxFxCaptureBackend(
        _REGISTRY,
        executor_factory=lambda plan, examples: called,
        fallback_to_torch=True,
    )
    value = torch.ones(2, 8)
    execute = backend(module, [value])
    execute(value)
    called.assert_called_once()
    module.normalized_shape[0] = 4
    updated = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    torch.testing.assert_close(execute(updated), module(updated))
    called.assert_called_once()


def test_nested_tensor_constants_never_execute_real_tensor_operations():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class RequireMeta(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert all(
                value.device.type == "meta"
                for value in tree_leaves((args, kwargs))
                if isinstance(value, torch.Tensor)
            )
            return func(*args, **(kwargs or {}))

    root = torch.nn.Module()
    root.constants = {"value": torch.ones(2)}
    graph = torch.fx.Graph()
    value = graph.get_attr("constants")
    value = graph.call_function(operator.getitem, (value, "value"))
    value = graph.call_function(torch.ops.aten.add.Tensor, (value, value))
    graph.output(value)
    module = torch.fx.GraphModule(root, graph)
    with RequireMeta():
        assert build_mlx_fx_plan(module, _REGISTRY, example_inputs=[]).fully_supported


def test_explicit_factory_device_never_allocates_real_storage():
    from torch.utils._python_dispatch import TorchDispatchMode

    class RequireMetaFactory(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if "device" in kwargs:
                assert kwargs["device"].type == "meta"
            return func(*args, **kwargs)

    inputs = [torch.ones(2)]
    graph = _graph(
        torch.ops.aten.empty_like.default,
        inputs,
        kwargs={"device": torch.device("cpu")},
    )
    with RequireMetaFactory():
        assert build_mlx_fx_plan(
            graph, _REGISTRY, example_inputs=inputs
        ).fully_supported


@pytest.mark.skipif(not _HAS_RUNTIME, reason="requires Torch 2.13 and MLX on MPS")
def test_direct_executor_rejects_changed_constant_before_launch():
    from sglang.srt.utils.tensor_bridge import mlx_call_multi

    root = torch.nn.Module()
    root.normalized_shape = [8]
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    shape = graph.get_attr("normalized_shape")
    output = graph.call_function(torch.ops.aten.layer_norm.default, (value, shape))
    graph.output(output)
    module = torch.fx.GraphModule(root, graph)
    inputs = [torch.ones(2, 8, device="mps")]
    plan = build_mlx_fx_plan(module, _REGISTRY)
    with mock.patch(
        "sglang.srt.utils.tensor_bridge.mlx_call_multi", wraps=mlx_call_multi
    ) as call:
        execute = make_mlx_fx_executor(plan, inputs)
        execute(*inputs)
        call.reset_mock()
        module.normalized_shape[0] = 4
        with pytest.raises(UnsupportedMlxFxGraphError, match="captured constant"):
            execute(torch.ones(2, 4, device="mps"))
        call.assert_not_called()


@pytest.mark.skipif(not _HAS_RUNTIME, reason="requires Torch 2.13 and MLX on MPS")
def test_runtime_change_is_rejected_before_mlx_call_and_valid_sizes_are_reusable():
    from sglang.srt.utils.tensor_bridge import mlx_call_multi

    inputs = [tensor.to("mps") for tensor in _qkv()]
    graph = _sdpa_graph(inputs)
    plan = build_mlx_fx_plan(graph, _REGISTRY)
    with mock.patch(
        "sglang.srt.utils.tensor_bridge.mlx_call_multi", wraps=mlx_call_multi
    ) as call:
        execute = make_mlx_fx_executor(plan, inputs)
        for values in (inputs, [tensor.to("mps") for tensor in _qkv(q=(2, 4, 7, 8))]):
            (actual,) = execute(*values)
            expected = graph(*(tensor.cpu() for tensor in values))
            # Same tolerance as the existing exported SDPA parity suite.
            torch.testing.assert_close(actual.cpu(), expected, atol=5e-3, rtol=5e-3)
        assert call.call_count == 2
        call.reset_mock()
        bad = [tensor.to("mps") for tensor in _qkv(q=(2, 4, 7, 9))]
        with pytest.raises(UnsupportedMlxFxGraphError, match="feature dimensions"):
            execute(*bad)
        call.assert_not_called()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
