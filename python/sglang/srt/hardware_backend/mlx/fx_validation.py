"""Metadata-only admissibility checks for selected generic MLX lowerings.

These checks prove known incompatibilities, not whole-graph equivalence. Unknown
or symbolic metadata is left undecided; custom serving ops remain the consumer's
responsibility. Runtime inference uses only Meta tensors and known Torch targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from typing import Any

import torch

MLX_DTYPE_NAMES = {
    torch.bool: "bool_",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_UNKNOWN = object()


@dataclass(frozen=True)
class _TensorMetadata:
    shape: tuple[Any, ...]
    dtype: torch.dtype


def _metadata(value):
    if hasattr(value, "shape") and isinstance(
        getattr(value, "dtype", None), torch.dtype
    ):
        return _TensorMetadata(tuple(value.shape), value.dtype)
    return None


def _known_int(value):
    # Do not specialize SymInts or introduce guards while inspecting a graph.
    return type(value) is int


def _different(left, right):
    return _known_int(left) and _known_int(right) and left != right


def _broadcast(shapes):
    result = ()
    for shape in shapes:
        dimensions = []
        for left, right in zip_longest(reversed(result), reversed(shape), fillvalue=1):
            if _known_int(left) and left == 1:
                dimensions.append(right)
            elif _known_int(right) and right == 1:
                dimensions.append(left)
            elif _different(left, right):
                return None
            else:
                dimensions.append(left if _known_int(left) else None)
        result = tuple(reversed(dimensions))
    return result


def _arg(args, kwargs, index, name, default=None):
    return args[index] if len(args) > index else kwargs.get(name, default)


def _validate_sdpa(args, kwargs):
    if len(args) < 3:
        return "SDPA lowering requires positional query, key and value"
    query, key, value = (_metadata(tensor) for tensor in args[:3])
    mask = _arg(args, kwargs, 3, "attn_mask")
    dropout = _arg(args, kwargs, 4, "dropout_p", 0.0)
    causal = _arg(args, kwargs, 5, "is_causal", False)
    gqa = _arg(args, kwargs, 7, "enable_gqa", False)
    if isinstance(dropout, (int, float)) and dropout != 0:
        return "SDPA dropout is unsupported"
    if causal is True and mask is not None and mask is not _UNKNOWN:
        return "SDPA causal attention with an explicit mask is unsupported"
    tensors = [tensor for tensor in (query, key, value) if tensor is not None]
    if (
        any(tensor.dtype not in _FLOAT_DTYPES for tensor in tensors)
        or len({tensor.dtype for tensor in tensors}) > 1
    ):
        return "SDPA requires matching floating-point Q/K/V dtypes"
    if any(len(tensor.shape) < (3 if gqa is True else 2) for tensor in tensors):
        return "SDPA requires Q/K/V matrix axes and a head axis for GQA"
    mask_meta = _metadata(mask)
    if (
        mask_meta is not None
        and query is not None
        and mask_meta.dtype
        not in (
            torch.bool,
            torch.float32,
            query.dtype,
        )
    ):
        return "unsupported SDPA mask dtype"
    if any(tensor is None for tensor in (query, key, value)):
        return None
    if _different(query.shape[-1], key.shape[-1]):
        return "SDPA query/key feature dimensions differ"
    if _known_int(query.shape[-1]) and query.shape[-1] == 0:
        return "SDPA zero-width query features are unsupported"
    if _different(key.shape[-2], value.shape[-2]):
        return "SDPA key/value sequence lengths differ"
    if gqa is True:
        query_heads = query.shape[-3]
        for tensor in (key, value):
            heads = tensor.shape[-3]
            if _known_int(heads) and (
                heads == 0 or (_known_int(query_heads) and query_heads % heads)
            ):
                return "SDPA GQA key/value head counts must divide query heads"
        batch = _broadcast([tensor.shape[:-3] for tensor in tensors])
        if batch is not None:
            batch = (*batch, query_heads)
    else:
        batch = _broadcast([tensor.shape[:-2] for tensor in tensors])
    if batch is None:
        return "SDPA batch/head dimensions are not broadcastable"
    score_shape = (*batch, query.shape[-2], key.shape[-2])
    if mask_meta is not None:
        expanded = _broadcast([score_shape, mask_meta.shape])
        if (
            expanded is None
            or len(expanded) != len(score_shape)
            or any(
                _different(left, right) for left, right in zip(expanded, score_shape)
            )
        ):
            return "SDPA mask is not broadcastable to attention scores"
    return None


def _validate_norm(lowering, args, kwargs):
    if len(args) < 2:
        return "MLX normalization requires positional input and normalized_shape"
    value = _metadata(args[0])
    shape = _arg(args, kwargs, 1, "normalized_shape")
    weight = _arg(args, kwargs, 2, "weight")
    if lowering == "rms_norm" and weight is None:
        return "MLX RMSNorm requires an explicit weight"
    parameters = [weight]
    if lowering == "layer_norm":
        parameters.append(_arg(args, kwargs, 3, "bias"))
    for parameter in parameters:
        metadata = _metadata(parameter)
        if metadata is not None and value is not None and metadata.dtype != value.dtype:
            return "MLX normalization requires matching input and parameter dtypes"
    if isinstance(shape, (tuple, list)):
        if len(shape) != 1:
            return "MLX normalization only supports a single last axis"
        if value is not None and (
            not value.shape or _different(shape[0], value.shape[-1])
        ):
            return "normalized_shape does not match the last input axis"
        for parameter in parameters:
            metadata = _metadata(parameter)
            if metadata is not None and (
                len(metadata.shape) != 1 or _different(metadata.shape[0], shape[0])
            ):
                return "normalization parameter shape does not match normalized_shape"
    if value is not None and value.dtype not in _FLOAT_DTYPES:
        return "MLX normalization requires float16, bfloat16 or float32 inputs"
    return None


def _validate_node(lowering, args, kwargs):
    if lowering == "sdpa":
        return _validate_sdpa(args, kwargs)
    if lowering in {"rms_norm", "layer_norm"}:
        return _validate_norm(lowering, args, kwargs)
    if lowering in {"to_dtype", "mean", "empty_like"}:
        if lowering == "to_dtype" and len(args) < 2:
            return "MLX dtype conversion requires a positional dtype"
        dtype = (
            _arg(args, kwargs, 1, "dtype")
            if lowering == "to_dtype"
            else kwargs.get("dtype")
        )
        if isinstance(dtype, torch.dtype) and dtype not in MLX_DTYPE_NAMES:
            return f"Torch dtype has no MLX Metal lowering: {dtype}"
    return None


def _as_meta(value):
    if isinstance(value, torch.Tensor):
        if value.device.type == "meta":
            return value
        if all(_known_int(size) for size in value.shape):
            return torch.empty(tuple(value.shape), dtype=value.dtype, device="meta")
        return _metadata(value)
    if isinstance(value, (tuple, list)):
        return type(value)(_as_meta(item) for item in value)
    if isinstance(value, dict):
        return {key: _as_meta(item) for key, item in value.items()}
    if isinstance(value, slice):
        return slice(_as_meta(value.start), _as_meta(value.stop), _as_meta(value.step))
    if (
        value is None
        or type(value) in (bool, int, float, complex, str)
        or isinstance(
            value, (torch.dtype, torch.device, torch.layout, torch.memory_format)
        )
    ):
        return value
    return _UNKNOWN


def _has_unknown(value):
    if value is _UNKNOWN or isinstance(value, _TensorMetadata):
        return True
    if isinstance(value, (tuple, list)):
        return any(_has_unknown(item) for item in value)
    if isinstance(value, dict):
        return any(_has_unknown(item) for item in value.values())
    if isinstance(value, slice):
        return any(_has_unknown(item) for item in (value.start, value.stop, value.step))
    if isinstance(value, torch.Tensor):
        return value.device.type != "meta"
    return False


def _captured_metadata(node):
    return node.meta.get(
        "val", node.meta.get("example_value", node.meta.get("tensor_meta", _UNKNOWN))
    )


def validate_graph(
    graph_module, lowerings, specs, resolve_value, resolve_attr, example_inputs=None
):
    """Return node-specific rejection reasons without importing MLX or borrowing data.

    Without examples, inspect capture metadata. With examples, infer fresh Meta
    outputs; never reuse potentially stale intermediate capture metadata. Missing
    Meta kernels and custom targets yield unknown metadata, not false rejections.
    """
    values = {}
    errors = {}
    placeholders = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    if example_inputs is not None and len(example_inputs) != len(placeholders):
        return {
            (
                placeholders[0]
                if placeholders
                else next(iter(graph_module.graph.nodes))
            ): "FX placeholder and example-input counts do not match"
        }
    examples = dict(zip(placeholders, example_inputs or ()))
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            values[node] = (
                _as_meta(examples[node])
                if example_inputs is not None
                else _captured_metadata(node)
            )
            continue
        if node.op == "get_attr":
            values[node] = _as_meta(resolve_attr(graph_module, str(node.target)))
            continue
        args = resolve_value(node.args, values)
        kwargs = resolve_value(node.kwargs, values)
        lowering = lowerings[node]
        spec = specs.get(lowering)
        known_target = spec is not None and (
            (
                node.op == "call_function"
                and node.target in (*spec.aten, *spec.functions)
            )
            or (node.op == "call_method" and node.target in spec.methods)
        )
        reason = _validate_node(lowering, args, kwargs) if known_target else None
        if reason is not None:
            errors[node] = reason
        if example_inputs is None:
            values[node] = _captured_metadata(node)
            continue
        values[node] = _UNKNOWN
        args, kwargs = _as_meta(args), _as_meta(kwargs)
        if reason or not known_target or _has_unknown((args, kwargs)):
            continue
        if "device" in kwargs:
            kwargs = {**kwargs, "device": torch.device("meta")}
        # The registry may contain custom Python functions or semantic serving
        # ops. Only known built-in lowering targets may run on Meta tensors.
        try:
            if node.op == "call_function":
                values[node] = node.target(*args, **kwargs)
            else:
                values[node] = getattr(args[0], node.target)(*args[1:], **kwargs)
        except (RuntimeError, TypeError, ValueError, NotImplementedError):
            pass
    return errors
