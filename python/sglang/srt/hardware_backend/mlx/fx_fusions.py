"""Conservative, opt-in pattern fusion for exported ATen graphs.

Only static shapes with proven dtypes are fused. The copied graph shares the
caller's parameters/buffers; it never copies weights or changes the input graph.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node

_ATEN = torch.ops.aten
_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def fused_rms_norm(value, weight, epsilon):
    """Torch reference preserving the cast-before-weight RMSNorm variant."""
    normalized = value.float()
    normalized = normalized * torch.rsqrt(
        normalized.pow(2).mean(-1, keepdim=True) + epsilon
    )
    return normalized.to(value.dtype) * weight


@dataclass(frozen=True)
class MlxFxFusionResult:
    graph_module: GraphModule
    replacements: dict[str, str]


def _is_op(node, target):
    return (
        isinstance(node, Node) and node.op == "call_function" and node.target == target
    )


def _tensor(node):
    value = node.meta.get("val") if isinstance(node, Node) else None
    if not isinstance(value, torch.Tensor):
        return None
    if not all(type(size) is int and size > 0 for size in value.shape):
        return None
    return value


def _plain(node, target, nargs):
    return _is_op(node, target) and len(node.args) == nargs and not node.kwargs


def _rms_norm_match(root):
    if not _plain(root, _ATEN.mul.Tensor, 2):
        return None
    for weight, cast in (root.args, root.args[::-1]):
        if not _plain(cast, _ATEN.to.dtype, 2):
            continue
        multiply, output_dtype = cast.args
        if not _plain(multiply, _ATEN.mul.Tensor, 2):
            continue
        for promoted, rsqrt in (multiply.args, multiply.args[::-1]):
            if not _plain(promoted, _ATEN.to.dtype, 2):
                continue
            value, compute_dtype = promoted.args
            x, w = _tensor(value), _tensor(weight)
            if (
                x is None
                or w is None
                or x.ndim < 1
                or x.dtype not in _FLOAT_DTYPES
                or w.dtype != x.dtype
                or tuple(w.shape) != (x.shape[-1],)
                or output_dtype != x.dtype
                or compute_dtype != torch.float32
                or not _plain(rsqrt, _ATEN.rsqrt.default, 1)
            ):
                continue
            add = rsqrt.args[0]
            if not _plain(add, _ATEN.add.Tensor, 2):
                continue
            mean, epsilon = add.args
            if (
                type(epsilon) not in (float, int)
                or not math.isfinite(epsilon)
                or epsilon <= 0
                or not _plain(mean, _ATEN.mean.dim, 3)
            ):
                continue
            power, axes, keepdim = mean.args
            if (
                not isinstance(axes, (tuple, list))
                or list(axes) not in ([-1], [x.ndim - 1])
                or keepdim is not True
                or not _plain(power, _ATEN.pow.Tensor_Scalar, 2)
                or power.args != (promoted, 2)
            ):
                continue
            return (value, weight, epsilon), {
                cast,
                multiply,
                promoted,
                rsqrt,
                add,
                mean,
                power,
            }
    return None


def fuse_mlx_fx_graph(graph_module: GraphModule) -> MlxFxFusionResult:
    """Return a graph copy with only semantically proven patterns replaced."""
    graph = copy.deepcopy(graph_module.graph)
    replacements = {}
    for root in list(graph.nodes):
        match = _rms_norm_match(root)
        if match is None:
            continue
        args, intermediates = match
        # Export inserts metadata checks around dtype casts. The original
        # executor treats them as no-ops; retain every other outside user.
        for node in list(graph.nodes):
            if (
                _is_op(node, _ATEN._assert_tensor_metadata.default)
                and node.args[0] in intermediates
                and not node.users
            ):
                graph.erase_node(node)
        root.target = fused_rms_norm
        root.args = args
        replacements[root.name] = "fused_rms_norm"
    module = GraphModule(graph_module, graph)
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    return MlxFxFusionResult(module, replacements)
