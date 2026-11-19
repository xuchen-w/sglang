"""Admission contracts for complete Torch causal-LM decode regions."""

from __future__ import annotations

import inspect
from enum import Enum

import msgspec
import torch

from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode


class UnsupportedMlxRegion(ValueError):
    pass


# One sequence mixer per decoder block: full attention or a linear-attention
# layer. Hybrid models use both kinds in the same stack.
STATEFUL_LAYERS = (RadixAttention, RadixLinearAttention)


class DecodeRegion(msgspec.Struct, frozen=True, kw_only=True):
    layers: tuple[RadixAttention, ...]
    linear_layers: tuple[RadixLinearAttention, ...]
    position_arg: str
    pass_pp_proxy: bool


def _check_gdn(layer: RadixLinearAttention) -> None:
    """GDN geometry the compiled graph can express, or an unsupported region."""
    weight = layer.conv_weights
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 2
        or weight.shape[0] != 2 * layer.k_dim + layer.v_dim
        or weight.shape[1] < 2
        or layer.num_q_heads != layer.num_k_heads
        or layer.head_q_dim != layer.head_k_dim
        or layer.head_k_dim % 32
        or layer.head_v_dim % 4
        or layer.num_v_heads % layer.num_k_heads
        or layer.activation not in ("silu", "swish")
        or layer.lower_bound is not None
    ):
        raise UnsupportedMlxRegion("Unsupported MLX graph GDN geometry")


def discover_decode_region(model: torch.nn.Module) -> DecodeRegion:
    modules = tuple(model.named_modules(remove_duplicate=False))
    stacks = [
        (name, module)
        for name, module in modules
        if isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict))
        and any(isinstance(child, STATEFUL_LAYERS) for child in module.modules())
    ]
    stacks = [
        (name, module)
        for name, module in stacks
        if not any(other.startswith(name + ".") for other, _ in stacks)
    ]
    if len(stacks) != 1:
        raise UnsupportedMlxRegion("MLX requires exactly one decoder stack")
    mixers = tuple(
        module for _, module in modules if isinstance(module, STATEFUL_LAYERS)
    )
    stack_mixers = tuple(
        module
        for _, module in stacks[0][1].named_modules(remove_duplicate=False)
        if isinstance(module, STATEFUL_LAYERS)
    )
    if mixers != stack_mixers or len({id(x) for x in mixers}) != len(mixers):
        raise UnsupportedMlxRegion("MLX requires unique mixer modules in one stack")
    if len({x.layer_id for x in mixers}) != len(mixers):
        raise UnsupportedMlxRegion("MLX requires unique attention layer IDs")
    for block in stacks[0][1].children():
        if sum(isinstance(x, STATEFUL_LAYERS) for x in block.modules()) != 1:
            raise UnsupportedMlxRegion(
                "MLX requires one attention invocation per block"
            )
    layers = tuple(x for x in mixers if isinstance(x, RadixAttention))
    linear_layers = tuple(x for x in mixers if isinstance(x, RadixLinearAttention))
    geometry = {(x.tp_k_head_num, x.head_dim) for x in layers}
    if len(geometry) > 1:
        raise UnsupportedMlxRegion("MLX requires homogeneous KV geometry")
    for layer in linear_layers:
        _check_gdn(layer)
    for layer in layers:
        if (
            layer.head_dim not in (64, 128, 256)
            or layer.qk_head_dim != layer.head_dim
            or layer.v_head_dim != layer.head_dim
            or layer.tp_q_head_num <= 0
            or layer.tp_k_head_num <= 0
            or layer.tp_q_head_num % layer.tp_k_head_num
            or layer.sliding_window_size > 0
            or layer.logit_cap
            or layer.is_cross_attention
            or layer.attn_type != AttentionType.DECODER
            or layer.pos_encoding_mode != "NONE"
            or layer.xai_temperature_len > 0
            or layer.quant_method is not None
        ):
            raise UnsupportedMlxRegion("Unsupported MLX graph attention geometry")
    # Model families do not share a base class declaring quant_config.
    if any(vars(module).get("quant_config") is not None for _, module in modules):
        raise UnsupportedMlxRegion("MLX regions require unquantized weights")
    parameters = inspect.signature(model.forward).parameters
    positions = [name for name in ("positions", "position_ids") if name in parameters]
    if len(positions) != 1 or not {"input_ids", "forward_batch"} <= parameters.keys():
        raise UnsupportedMlxRegion(
            "MLX requires token, position and ForwardBatch inputs"
        )
    supplied = {"input_ids", "forward_batch", positions[0], "pp_proxy_tensors"}
    for name, parameter in parameters.items():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        ) or (
            parameter.kind is not inspect.Parameter.VAR_KEYWORD
            and name not in supplied
            and parameter.default is inspect.Parameter.empty
        ):
            raise UnsupportedMlxRegion(f"Unsupported MLX forward argument: {name}")
    return DecodeRegion(
        layers=layers,
        linear_layers=linear_layers,
        position_arg=positions[0],
        pass_pp_proxy="pp_proxy_tensors" in parameters,
    )


class RegionBatch(ForwardBatch):
    """Record static field reads so replay cannot silently bake in batch metadata."""

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        state = super().__getattribute__("__dict__")
        if "_static_reads" in state and name in ForwardBatch.__dataclass_fields__:
            if name in (
                "input_ids",
                "positions",
                "mrope_positions",
                "req_pool_indices",
                "seq_lens",
                "out_cache_loc",
                "batch_size",
            ):
                return value
            if state["forward_mode"] == ForwardMode.EXTEND and name in (
                "extend_seq_lens",
                "extend_prefix_lens",
                "extend_start_loc",
            ):
                return value
            if name == "residual_stream" and state["_residual_written"]:
                return value
            if (
                state["forward_mode"] == ForwardMode.EXTEND
                and name
                in (
                    "extend_seq_lens_cpu",
                    "extend_prefix_lens_cpu",
                    "extend_logprob_start_lens_cpu",
                )
                and isinstance(value, list)
                and all(type(x) is int or isinstance(x, torch.SymInt) for x in value)
            ):
                state["_static_reads"].setdefault(name, list(value))
                return value
            # A text-only multimodal model reads its per-request inputs, which
            # stay None for every text batch.
            if name == "mm_inputs" and isinstance(value, list):
                if all(item is None for item in value):
                    return value
                raise UnsupportedMlxRegion("MLX cannot replay multimodal inputs")
            if value is not None and not isinstance(
                value, (bool, int, float, str, Enum)
            ):
                if state["forward_mode"] != ForwardMode.EXTEND or not isinstance(
                    value, torch.SymInt
                ):
                    raise UnsupportedMlxRegion(
                        f"Unsupported MLX batch metadata: {name}"
                    )
            state["_static_reads"][name] = value
        return value

    def __setattr__(self, name, value):
        if (
            "_static_reads" in self.__dict__
            and name in ForwardBatch.__dataclass_fields__
        ):
            if name != "residual_stream":
                raise UnsupportedMlxRegion(f"MLX cannot mutate batch metadata: {name}")
            self.__dict__["_residual_written"] = True
        super().__setattr__(name, value)

    def track_reads(self, reads):
        self._static_reads = reads
        self._residual_written = False

    def _validate_reads(self):
        for name, expected in self._static_reads.items():
            if isinstance(expected, list) and self.__dict__[name] != expected:
                raise UnsupportedMlxRegion(f"MLX cannot mutate batch metadata: {name}")


def static_metadata_matches(batch: ForwardBatch, reads: dict) -> bool:
    for name, expected in reads.items():
        actual = getattr(batch, name)
        if type(actual) is not type(expected) or actual != expected:
            return False
    return True
