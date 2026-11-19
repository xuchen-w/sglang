"""MLX execution of the read-only gated delta export contracts."""

import mlx.core as mx
import torch

# Matches FLA/Triton, not F.normalize: epsilon is added to the squared norm.
L2NORM_EPS = 1e-6
SOFTPLUS_THRESHOLD = 20.0


_DTYPES = {
    torch.float16: mx.float16,
    torch.bfloat16: mx.bfloat16,
    torch.float32: mx.float32,
}


def _dtype(value):
    """The MLX dtype of a state-pool dtype the export contract carries as Torch."""
    return value if isinstance(value, mx.Dtype) else _DTYPES[value]


def _activated(conv, conv_bias, dtype):
    """SiLU of a depthwise convolution result, rounded to the activation dtype."""
    if conv_bias is not None:
        conv = conv + conv_bias.astype(mx.float32)
    return (conv * mx.sigmoid(conv)).astype(dtype)


def _conv_silu(window, conv_weight, conv_bias, dtype):
    """Depthwise causal Conv1D at each window's last [..., channels, K] position."""
    conv = mx.sum(window.astype(mx.float32) * conv_weight.astype(mx.float32), axis=-1)
    return _activated(conv, conv_bias, dtype)


def _qkv_gates(
    qkv, a, b, A_log, dt_bias, num_k_heads, head_k_dim, num_v_heads, head_v_dim
):
    """Split post-conv qkv, normalize q/k and build the gates in fp32."""
    key_dim = num_k_heads * head_k_dim
    qkv = qkv.astype(mx.float32)
    prefix = qkv.shape[:-1]
    q = qkv[..., :key_dim].reshape(*prefix, num_k_heads, head_k_dim)
    k = qkv[..., key_dim : 2 * key_dim].reshape(*prefix, num_k_heads, head_k_dim)
    v = qkv[..., 2 * key_dim :].reshape(*prefix, num_v_heads, head_v_dim)
    q = q * mx.rsqrt(mx.sum(q * q, axis=-1, keepdims=True) + L2NORM_EPS)
    k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True) + L2NORM_EPS)
    q = q * head_k_dim**-0.5
    shifted = a.astype(mx.float32) + dt_bias.astype(mx.float32)
    softplus = mx.where(
        shifted > SOFTPLUS_THRESHOLD,
        shifted,
        mx.log1p(mx.exp(mx.minimum(shifted, SOFTPLUS_THRESHOLD))),
    )
    decay = mx.exp(-mx.exp(A_log.astype(mx.float32)) * softplus)
    return q, k, v, decay, mx.sigmoid(b.astype(mx.float32))


def _recurrence(q, k, v, decay, beta, state):
    """Recurrence over q/k [B, T, Hk, Dk], v [B, T, Hv, Dv], state [B, Hv, Dv, Dk]."""
    # MLX 0.32.4 rounds the decay to the activation dtype for bf16 inputs
    # (ml-explore/mlx#4760), so the recurrence always runs in fp32; drop this
    # once SGLang requires an MLX release that contains the fix.
    return mx.fast.gated_delta_update(
        q.astype(mx.float32),
        k.astype(mx.float32),
        v.astype(mx.float32),
        decay,
        beta,
        state.astype(mx.float32),
    )


def _state_rows(conv_pool, ssm_pool, state_table, requests, layer):
    slots = mx.take(state_table, requests, axis=0).astype(mx.int32)
    # Fused index: rows only, never a whole layer slice.
    return conv_pool[layer, slots], ssm_pool[layer, slots]


def gdn_decode(
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
    *,
    layer,
    num_k_heads,
    head_k_dim,
    head_v_dim,
    tails=None,
):
    """One GDN step per request over the full state pools, or over pending rows."""
    if tails is None:
        conv_state, ssm_state = _state_rows(
            conv_pool, ssm_pool, state_table, requests, layer
        )
    else:
        conv_state, ssm_state = tails
    window = mx.concatenate((conv_state, mixed_qkv[:, :, None]), axis=-1)
    qkv = _conv_silu(window, conv_weight, conv_bias, mixed_qkv.dtype)
    q, k, v, decay, beta = _qkv_gates(
        qkv, a, b, A_log, dt_bias, num_k_heads, head_k_dim, a.shape[1], head_v_dim
    )
    out, state = _recurrence(
        q[:, None], k[:, None], v[:, None], decay[:, None], beta[:, None], ssm_state
    )
    return (
        out[:, 0].astype(mixed_qkv.dtype),
        window[:, :, 1:].astype(conv_pool.dtype),
        state.astype(ssm_pool.dtype),
    )


def gdn_prefill(
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
    """One uncached sequence from a zero state: output, final window and state."""
    tokens, channels = mixed_qkv.shape
    width = conv_weight.shape[1]
    padded = mx.concatenate(
        (mx.zeros((width - 1, channels), dtype=mixed_qkv.dtype), mixed_qkv)
    )
    convolved = sum(
        padded[offset : offset + tokens].astype(mx.float32)
        * conv_weight[:, offset].astype(mx.float32)
        for offset in range(width)
    )
    qkv = _activated(convolved, conv_bias, mixed_qkv.dtype)
    q, k, v, decay, beta = _qkv_gates(
        qkv, a, b, A_log, dt_bias, num_k_heads, head_k_dim, a.shape[1], head_v_dim
    )
    state = mx.zeros((1, a.shape[1], head_v_dim, head_k_dim), dtype=mx.float32)
    out, state = _recurrence(q[None], k[None], v[None], decay[None], beta[None], state)
    window = mx.transpose(padded[tokens:], (1, 0))[None]
    return (
        out[0].astype(mixed_qkv.dtype),
        window.astype(_dtype(conv_dtype)),
        state.astype(_dtype(ssm_dtype)),
    )
