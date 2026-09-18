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


def fused_rotary_embedding(value, cosine, sine):
    """Split-half RoPE with arbitrary cached tables, without frequency assumptions."""
    first, second = value.chunk(2, dim=-1)
    return value * cosine + torch.cat((-second, first), dim=-1) * sine


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


def _rope_match(root):
    if not _plain(root, _ATEN.add.Tensor, 2):
        return None
    for direct, rotated in (root.args, root.args[::-1]):
        if not _plain(direct, _ATEN.mul.Tensor, 2) or not _plain(
            rotated, _ATEN.mul.Tensor, 2
        ):
            continue
        for value, cosine in (direct.args, direct.args[::-1]):
            x, c = _tensor(value), _tensor(cosine)
            if x is None or c is None or x.ndim != 4 or x.shape[-1] % 2:
                continue
            for cat, sine in (rotated.args, rotated.args[::-1]):
                s = _tensor(sine)
                if (
                    s is None
                    or x.dtype not in _FLOAT_DTYPES
                    or c.dtype != x.dtype
                    or s.dtype != x.dtype
                    or c.ndim != 4
                    or s.ndim != 4
                    or any(a not in (1, b) for a, b in zip(c.shape, x.shape))
                    or any(a not in (1, b) for a, b in zip(s.shape, x.shape))
                    or c.shape[-1] != x.shape[-1]
                    or s.shape[-1] != x.shape[-1]
                    or not _plain(cat, _ATEN.cat.default, 2)
                    or cat.args[1] not in (-1, 3)
                    or not isinstance(cat.args[0], (list, tuple))
                    or len(cat.args[0]) != 2
                ):
                    continue
                neg, first = cat.args[0]
                if not _plain(neg, _ATEN.neg.default, 1):
                    continue
                second = neg.args[0]
                if not all(_is_op(n, _ATEN.slice.Tensor) for n in (first, second)):
                    continue
                half = x.shape[-1] // 2
                slices = ((first, 0, half), (second, half, x.shape[-1]))
                if any(
                    n.kwargs
                    or len(n.args) not in (4, 5)
                    or n.args[0] is not value
                    or n.args[1] not in (-1, 3)
                    or n.args[2] != start
                    or type(n.args[3]) is not int
                    or min(n.args[3], x.shape[-1]) != end
                    or (len(n.args) == 5 and n.args[4] != 1)
                    for n, start, end in slices
                ):
                    continue
                return (value, cosine, sine), {direct, rotated, cat, neg, first, second}
    return None


def fuse_mlx_fx_graph(graph_module: GraphModule) -> MlxFxFusionResult:
    """Return a graph copy with only semantically proven patterns replaced."""
    graph = copy.deepcopy(graph_module.graph)
    replacements = {}
    for root in list(graph.nodes):
        target, lowering = fused_rms_norm, "fused_rms_norm"
        match = _rms_norm_match(root)
        if match is None:
            target, lowering = fused_rotary_embedding, "fused_rotary_embedding"
            match = _rope_match(root)
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
        root.target = target
        root.args = args
        replacements[root.name] = lowering
    module = GraphModule(graph_module, graph)
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    return MlxFxFusionResult(module, replacements)
