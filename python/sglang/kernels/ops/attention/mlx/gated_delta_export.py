"""Torch export contracts and references for read-only MLX gated delta layers.

Both operators follow SGLang's GDN semantics: a depthwise causal Conv1D with
fused SiLU over ``mixed_qkv = [q | k | v]``, the gate
``g = exp(-exp(A_log) * softplus(a + dt_bias))``, ``beta = sigmoid(b)``,
per-head L2-normalized q/k with q scaled by ``head_k_dim**-0.5``, and an fp32
recurrent state laid out as ``[v_heads, head_v_dim, head_k_dim]``. Neither
operator writes a pool: they return the next Conv1D window and recurrent state
for the caller to commit.
"""

import torch
import torch.nn.functional as F

# Matches the FLA/Triton GDN kernels: x / sqrt(sum(x * x) + eps).
L2NORM_EPS = 1e-6


def _validate_shapes(mixed_qkv, a, b, conv_weight, conv_bias, A_log, dt_bias, dims):
    num_k_heads, head_k_dim, head_v_dim = dims
    if mixed_qkv.ndim != 2 or a.ndim != 2 or conv_weight.ndim != 2:
        raise ValueError("Unsupported MLX gated delta tensor rank")
    tokens, channels = mixed_qkv.shape
    v_heads = a.shape[1]
    if (
        tokens < 1
        or min(num_k_heads, head_k_dim, head_v_dim) < 1
        or v_heads % num_k_heads
        or channels != 2 * num_k_heads * head_k_dim + v_heads * head_v_dim
        or mixed_qkv.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or a.shape[0] != tokens
        or b.shape != a.shape
        or conv_weight.shape[0] != channels
        or conv_weight.shape[1] < 2
        or (conv_bias is not None and conv_bias.shape != (channels,))
        or A_log.shape != (v_heads,)
        or dt_bias.shape != (v_heads,)
    ):
        raise ValueError("Unsupported MLX gated delta shape or dtype")


def _validate_decode_state(
    mixed_qkv, a, conv_pool, ssm_pool, state_table, requests, dims
):
    _, head_k_dim, head_v_dim = dims
    batch, channels = mixed_qkv.shape
    if (
        conv_pool.ndim != 4
        or ssm_pool.ndim != 5
        or conv_pool.shape[2] != channels
        or conv_pool.shape[3] < 1
        or ssm_pool.shape[2:] != (a.shape[1], head_v_dim, head_k_dim)
        or ssm_pool.shape[1] != conv_pool.shape[1]
        or state_table.ndim != 1
        or state_table.dtype != torch.int32
        or requests.shape != (batch,)
        or requests.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Unsupported MLX gated delta state pool shape or dtype")


def _validate_tails(batch, conv_pool, ssm_pool, tail_conv, tail_ssm):
    if (tail_conv is None) != (tail_ssm is None) or (
        tail_conv is not None
        and any(
            tail.shape != (batch, *pool.shape[2:])
            or tail.dtype != pool.dtype
            or tail.device != pool.device
            for tail, pool in ((tail_conv, conv_pool), (tail_ssm, ssm_pool))
        )
    ):
        raise ValueError(
            "Pending Conv1D/SSM rows must both match one pool row per request"
        )


def _conv_silu(windows, weight, bias):
    """Depthwise causal Conv1D over windows [..., channels, kernel], in fp32."""
    out = (windows.float() * weight.float()).sum(-1)
    if bias is not None:
        out = out + bias.float()
    return F.silu(out)


def _l2norm(x):
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + L2NORM_EPS)


def _gated_delta(qkv, a, b, A_log, dt_bias, state, dims):
    """Run the recurrence on qkv [N, T, channels] from state [N, Hv, Dv, Dk]."""
    num_k_heads, head_k_dim, head_v_dim = dims
    count, tokens, _ = qkv.shape
    v_heads = a.shape[-1]
    groups = v_heads // num_k_heads
    key_dim = num_k_heads * head_k_dim
    q, k, v = qkv.float().split([key_dim, key_dim, v_heads * head_v_dim], -1)
    q, k = (
        _l2norm(x.reshape(count, tokens, num_k_heads, head_k_dim)).repeat_interleave(
            groups, dim=2
        )
        for x in (q, k)
    )
    q = q * head_k_dim**-0.5
    v = v.reshape(count, tokens, v_heads, head_v_dim)
    decay = torch.exp(
        -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float())
    )
    beta = torch.sigmoid(b.float())
    state = state.float()
    outputs = []
    for t in range(tokens):
        state = state * decay[:, t, :, None, None]
        memory = (state * k[:, t, :, None, :]).sum(-1)
        delta = (v[:, t] - memory) * beta[:, t, :, None]
        state = state + delta[..., None] * k[:, t, :, None, :]
        outputs.append((state * q[:, t, :, None, :]).sum(-1))
    return torch.stack(outputs, dim=1), state


@torch.library.custom_op("sglang::mlx_gdn_decode", mutates_args=())
def gdn_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_pool: torch.Tensor,
    ssm_pool: torch.Tensor,
    state_table: torch.Tensor,
    requests: torch.Tensor,
    tail_conv: torch.Tensor | None = None,
    tail_ssm: torch.Tensor | None = None,
    *,
    layer: int,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance one token per request from its pool row without modifying a pool.

    mixed_qkv is [batch, channels]; a/b are [batch, v_heads]; conv_weight is
    [channels, kernel]. The pools keep every layer: conv_pool is
    [layers, slots, channels, kernel - 1] and ssm_pool is
    [layers, slots, v_heads, head_v_dim, head_k_dim], with ``layer`` selecting
    the layer and ``state_table[requests]`` each request's slot. Serving must
    validate those slots before executing the exported graph. Optional
    tail_conv/tail_ssm replace the pool rows with the uncommitted rows of the
    preceding token. Returns the output [batch, v_heads, head_v_dim] in the
    activation dtype and each request's next Conv1D window and state, in the
    pool dtypes.
    """
    dims = (num_k_heads, head_k_dim, head_v_dim)
    _validate_shapes(mixed_qkv, a, b, conv_weight, conv_bias, A_log, dt_bias, dims)
    _validate_decode_state(
        mixed_qkv, a, conv_pool, ssm_pool, state_table, requests, dims
    )
    _validate_tails(mixed_qkv.shape[0], conv_pool, ssm_pool, tail_conv, tail_ssm)
    if not 0 <= layer < conv_pool.shape[0]:
        raise ValueError("MLX gated delta layer index is outside the state pool")
    if tail_conv is None:
        rows = state_table[requests.long()].long()
        if bool(((rows < 0) | (rows >= conv_pool.shape[1])).any()):
            raise ValueError("MLX gated delta state slot is outside the pool")
        windows, states = conv_pool[layer, rows], ssm_pool[layer, rows]
    else:
        windows, states = tail_conv, tail_ssm
    windows = torch.cat((windows, mixed_qkv.unsqueeze(-1)), -1)
    qkv = _conv_silu(windows, conv_weight, conv_bias).to(mixed_qkv.dtype)
    out, states = _gated_delta(
        qkv.unsqueeze(1), a.unsqueeze(1), b.unsqueeze(1), A_log, dt_bias, states, dims
    )
    return (
        out.squeeze(1).to(mixed_qkv.dtype),
        windows[..., 1:].contiguous().to(conv_pool.dtype),
        states.to(ssm_pool.dtype),
    )


@gdn_decode.register_fake
def _decode_fake(
    mixed_qkv,
    a,
    b,
    conv_weight,
    conv_bias,
    A_log,
    dt_bias,
    conv_pool,
    ssm_pool,
    state_table,
    requests,
    tail_conv=None,
    tail_ssm=None,
    *,
    layer,
    num_k_heads,
    head_k_dim,
    head_v_dim,
):
    dims = (num_k_heads, head_k_dim, head_v_dim)
    _validate_shapes(mixed_qkv, a, b, conv_weight, conv_bias, A_log, dt_bias, dims)
    _validate_decode_state(
        mixed_qkv, a, conv_pool, ssm_pool, state_table, requests, dims
    )
    _validate_tails(mixed_qkv.shape[0], conv_pool, ssm_pool, tail_conv, tail_ssm)
    batch = mixed_qkv.shape[0]
    return (
        mixed_qkv.new_empty(batch, a.shape[1], head_v_dim),
        conv_pool.new_empty(batch, *conv_pool.shape[2:]),
        ssm_pool.new_empty(batch, *ssm_pool.shape[2:]),
    )


@torch.library.custom_op("sglang::mlx_gdn_prefill", mutates_args=())
def gdn_prefill(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_dtype: torch.dtype,
    ssm_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one uncached sequence of [tokens, channels] from a zero state.

    Returns the output [tokens, v_heads, head_v_dim], the final Conv1D window
    [1, channels, kernel - 1] in ``conv_dtype`` and the final state
    [1, v_heads, head_v_dim, head_k_dim] in ``ssm_dtype``.
    """
    dims = (num_k_heads, head_k_dim, head_v_dim)
    _validate_shapes(mixed_qkv, a, b, conv_weight, conv_bias, A_log, dt_bias, dims)
    if any(
        dtype not in (torch.float32, torch.float16, torch.bfloat16)
        for dtype in (conv_dtype, ssm_dtype)
    ):
        raise ValueError("Unsupported MLX gated delta state dtype")
    tokens, channels = mixed_qkv.shape
    width = conv_weight.shape[1]
    inputs = torch.cat((mixed_qkv.new_zeros(width - 1, channels), mixed_qkv))
    qkv = _conv_silu(inputs.unfold(0, width, 1), conv_weight, conv_bias)
    state = mixed_qkv.new_zeros(
        1, a.shape[1], head_v_dim, head_k_dim, dtype=torch.float32
    )
    out, state = _gated_delta(
        qkv.to(mixed_qkv.dtype).unsqueeze(0),
        a.unsqueeze(0),
        b.unsqueeze(0),
        A_log,
        dt_bias,
        state,
        dims,
    )
    return (
        out.squeeze(0).to(mixed_qkv.dtype),
        inputs[tokens:].T.unsqueeze(0).contiguous().to(conv_dtype),
        state.to(ssm_dtype),
    )


@gdn_prefill.register_fake
def _prefill_fake(
    mixed_qkv,
    a,
    b,
    conv_weight,
    conv_bias,
    A_log,
    dt_bias,
    *,
    num_k_heads,
    head_k_dim,
    head_v_dim,
    conv_dtype,
    ssm_dtype,
):
    dims = (num_k_heads, head_k_dim, head_v_dim)
    _validate_shapes(mixed_qkv, a, b, conv_weight, conv_bias, A_log, dt_bias, dims)
    tokens, channels = mixed_qkv.shape
    v_heads = a.shape[1]
    return (
        mixed_qkv.new_empty(tokens, v_heads, head_v_dim),
        mixed_qkv.new_empty(1, channels, conv_weight.shape[1] - 1, dtype=conv_dtype),
        mixed_qkv.new_empty(1, v_heads, head_v_dim, head_k_dim, dtype=ssm_dtype),
    )
