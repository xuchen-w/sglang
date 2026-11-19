"""Hybrid GDN serving state owned by the compiled MLX runner."""

import msgspec
import torch

from sglang.kernels.ops.attention.mlx.gated_delta import (
    gdn_decode as mlx_gdn_decode,
)
from sglang.kernels.ops.attention.mlx.gated_delta_export import gdn_decode
from sglang.kernels.ops.attention.mlx.radix_attention import (
    radix_decode as mlx_radix_decode,
)
from sglang.kernels.ops.attention.mlx.radix_attention_export import radix_decode
from sglang.srt.configs.hybrid_arch import hybrid_gdn_config
from sglang.srt.hardware_backend.mps.compiled_graph import (
    StatefulOp,
    radix_stateful_op,
)
from sglang.srt.hardware_backend.mps.compiled_region import (
    DecodeRegion,
    UnsupportedMlxRegion,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
)


class GdnState(msgspec.Struct, frozen=True, kw_only=True):
    """The request-scoped recurrent state one compiled graph reads per layer.

    ``table`` maps request rows to state slots, ``conv`` is
    [layers, slots, channels, kernel - 1] and ``ssm`` is
    [layers, slots, v_heads, head_v_dim, head_k_dim]. Pool row j belongs to
    the region's j-th linear layer.
    """

    table: torch.Tensor
    conv: torch.Tensor
    ssm: torch.Tensor


def gdn_state_contract(model_runner, region) -> GdnState | str:
    """The state a hybrid graph reads, or why this runner stays on eager MPS."""
    if hybrid_gdn_config(model_runner.model_config) is None:
        return "linear attention is not Gated DeltaNet"
    pool = model_runner.token_to_kv_pool
    req_pool = model_runner.req_to_token_pool
    if not isinstance(pool, HybridLinearKVPool) or not isinstance(
        req_pool, HybridReqToTokenPool
    ):
        return "GDN state is not a hybrid MHA/state pool pair"
    if not isinstance(pool.full_kv_pool, MHATokenToKVPool) or pool.use_mla:
        return "the full-attention pool is not a standard MHA KV pool"
    # Unified memory hands out virtual state slot ids, which the graph cannot
    # translate; a static table's slots index the pool as they are.
    if model_runner.runner_ids.state_ids(req_pool.state_sub_pool).write is not None:
        return "unified-memory state slots"
    mamba = req_pool.mamba_pool
    if len(mamba.mamba_cache.conv) != 1:
        return "GDN state spans several conv pools"
    if req_pool.short_conv_pool.enabled or req_pool.ngram_pool.enabled:
        return "extra PLE state pools are enabled"
    conv = mamba.mamba_cache.conv[0]
    ssm = mamba.mamba_cache.temporal
    if not (conv.is_contiguous() and ssm.is_contiguous()):
        return "GDN state pools are not contiguous"
    rows = tuple(
        req_pool.mamba_map.get(layer.layer_id) for layer in region.linear_layers
    )
    if rows != tuple(range(conv.shape[0])):
        return "GDN state pool is not ordered like the decoder stack"
    return GdnState(
        table=req_pool.req_index_to_mamba_index_mapping,
        conv=conv,
        ssm=ssm,
    )


def commit_gdn_state(state: GdnState, *, slots, conv_rows, ssm_rows):
    """Write the batch's new rows into the state pool.

    ``conv_rows`` is [layers, batch, channels, kernel - 1]; ``ssm_rows`` is one
    [batch, v_heads, head_v_dim, head_k_dim] tensor per layer. One index put
    covers conv, while each layer's state is copied on its own row.
    """
    state.conv[:, slots] = conv_rows
    for layer, rows in enumerate(ssm_rows):
        state.ssm[layer].index_copy_(0, slots, rows)


def stateful_ops(region: DecodeRegion, *, prefill: bool = False):
    """The named tail layout of every stateful op in ``region``.

    The graph outputs logits, then the stacked K/V of the full-attention
    layers, then one conv-row stack and one SSM-row tensor per linear layer;
    these are also the lookahead tail inputs, in that order.
    """
    ops = {}
    if prefill:
        return ops
    if region.layers:
        ops[radix_decode._opoverload] = radix_stateful_op(
            mlx_radix_decode, len(region.layers)
        )
    if region.linear_layers:
        tail = 2 if region.layers else 0
        ops[gdn_decode._opoverload] = StatefulOp(
            fn=mlx_gdn_decode,
            tail_args=("tail_conv", "tail_ssm"),
            tails=tuple(
                ((tail, index), (tail + 1 + index, None))
                for index in range(len(region.linear_layers))
            ),
            tail_width=tail + 1 + len(region.linear_layers),
        )
    return ops


def conv_weight_path(model: torch.nn.Module, layer) -> str:
    """The parameter path whose storage backs ``layer.conv_weights``.

    ``conv_weights`` is a view cached at construction, so binding it directly
    would freeze the value; the owning Parameter keeps weight replacement
    visible to the graph, as for every other weight.
    """
    storage = layer.conv_weights.untyped_storage().data_ptr()
    for path, parameter in model.named_parameters(remove_duplicate=False):
        if parameter.untyped_storage().data_ptr() == storage:
            return path
    raise UnsupportedMlxRegion("GDN conv weights have no owning parameter")


def conv_weight(model: torch.nn.Module, path: str, shape) -> torch.Tensor:
    """The live convolution weight, viewed as [channels, kernel]."""
    return model.get_parameter(path).view(shape)
