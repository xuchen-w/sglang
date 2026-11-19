"""Mixed-precision convolution history in the compiled MLX GDN decode op."""

import pytest
import torch

from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.kernels import gdn_reference as ref

register_mps_ci(est_time=10, suite="extra-unit-test-mps")

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple MPS"
)
DEV = "mps"


def _randn(*shape, gen, dtype, scale=1.0):
    return (torch.randn(*shape, generator=gen) * scale).to(dtype)


def test_mlx_decode_mixed_dtype_window():
    import mlx.core as mx

    from sglang.kernels.ops.attention.mlx.gated_delta import (
        gdn_decode as mlx_gdn_decode,
    )
    from sglang.kernels.ops.attention.mlx.gated_delta_export import (
        gdn_decode as torch_gdn_decode,
    )
    from sglang.srt.utils.tensor_bridge import mlx_to_torch, torch_to_mlx

    gen = torch.Generator().manual_seed(31)
    channels, width, slots = 64, 4, 8
    weight = _randn(channels, width, gen=gen, dtype=torch.bfloat16, scale=0.2)
    bias = _randn(channels, gen=gen, dtype=torch.bfloat16, scale=0.1)
    indices = torch.tensor([0, -1, 2], dtype=torch.int32)
    pool = _randn(slots, channels, width - 1, gen=gen, dtype=torch.bfloat16)

    x = _randn(3, channels, gen=gen, dtype=torch.float32)
    requests = torch.arange(3, dtype=torch.int64)
    head_v_dim = 1
    small_channels = head_v_dim + 2
    expected_mlx = pool[:, :small_channels].clone()
    expected_mlx[0] = expected_mlx[0].roll(-1, dims=-1)
    expected_mlx[0, :, -1] = x[0, :small_channels].to(torch.bfloat16)
    expected_mlx[2] = expected_mlx[2].roll(-1, dims=-1)
    expected_mlx[2, :, -1] = x[2, :small_channels].to(torch.bfloat16)
    expected_mlx = expected_mlx.index_select(0, indices.clamp_min(0))
    expected_mlx[1, :, -1] = x[1, :small_channels].to(torch.bfloat16)
    ssm_pool = torch.zeros(1, slots, 1, head_v_dim, 1, dtype=torch.float32)
    torch_args = (
        x[:, :small_channels],
        torch.full((3, 1), -100.0),
        torch.full((3, 1), 100.0),
        weight[:small_channels].to(torch.float32),
        bias[:small_channels].to(torch.float32),
        torch.full((1,), -10.0, dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
        pool[:, :small_channels][None],
        ssm_pool.cpu(),
        indices.clamp_min(0),
        requests,
    )
    torch_y, torch_window, _ = torch_gdn_decode(
        *map(lambda t: t.to(DEV), torch_args),
        layer=0,
        num_k_heads=1,
        head_k_dim=1,
        head_v_dim=head_v_dim,
    )
    torch_x = torch_args[0].clone()
    torch_x[:, 0] += 1.23456789
    sensitive_args = (torch_x, *torch_args[1:])
    sensitive_y, _, _ = torch_gdn_decode(
        *map(lambda t: t.to(DEV), sensitive_args),
        layer=0,
        num_k_heads=1,
        head_k_dim=1,
        head_v_dim=head_v_dim,
    )
    mlx_y, mlx_window, _ = mlx_gdn_decode(
        *map(lambda t: torch_to_mlx(t.to(DEV)), torch_args),
        layer=0,
        num_k_heads=1,
        head_k_dim=1,
        head_v_dim=head_v_dim,
    )
    mx.eval(mlx_y, mlx_window)
    expected_out = torch.stack(
        [
            ref._conv_window_silu(
                torch.cat(
                    (pool[slot, :small_channels], row[:small_channels, None]), -1
                ),
                weight[:small_channels].float(),
                bias[:small_channels].float(),
            )[2].view(1, 1)
            for slot, row in zip(indices.clamp_min(0).tolist(), x)
        ]
    )
    torch.testing.assert_close(
        -mlx_to_torch(mlx_y).cpu(), expected_out.cpu(), atol=1e-3, rtol=1e-3
    )
    torch.testing.assert_close(-torch_y.cpu(), expected_out.cpu(), atol=1e-3, rtol=1e-3)
    sensitive_window = torch.cat(
        (pool[0, :small_channels], torch_x[0, :small_channels, None]), -1
    )
    sensitive_expected = ref._conv_window_silu(
        sensitive_window, weight[:small_channels].float(), bias[:small_channels].float()
    )[2]
    torch.testing.assert_close(
        -sensitive_y[0, 0, 0].cpu(), sensitive_expected.cpu(), atol=1e-5, rtol=1e-5
    )
    for got, want in ((mlx_window, expected_mlx), (torch_window, expected_mlx)):
        got = mlx_to_torch(got) if isinstance(got, mx.array) else got
        torch.testing.assert_close(got.cpu(), want, atol=0, rtol=0)


def test_mlx_decode_convolves_fp32_input():
    """MLX decode must convolve the fp32 input; only the stored window is bf16."""
    import mlx.core as mx

    from sglang.kernels.ops.attention.mlx.gated_delta import (
        gdn_decode as mlx_gdn_decode,
    )
    from sglang.kernels.ops.attention.mlx.gated_delta_export import (
        gdn_decode as torch_gdn_decode,
    )
    from sglang.srt.utils.tensor_bridge import mlx_to_torch, torch_to_mlx

    inputs = dict(
        # bf16 rounds these to 1 and 2; the last tap (1.0) carries that to the output.
        mixed_qkv=torch.tensor([[1.003, 2.007, 1.003], [2.007, 1.003, 2.007]]),
        a=torch.zeros(2, 1),
        b=torch.zeros(2, 1),
        conv_weight=torch.tensor([[0.1, -0.2, 0.3, 1.0]]).repeat(3, 1),
        conv_bias=torch.full((3,), 0.05),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        conv_pool=torch.linspace(-1, 1, 27).view(1, 3, 3, 3).to(torch.bfloat16),
        ssm_pool=torch.full((1, 3, 1, 1, 1), 0.25),
        state_table=torch.tensor([0, 2, 1], dtype=torch.int32),
        requests=torch.tensor([1, 2]),
    )
    dims = dict(layer=0, num_k_heads=1, head_k_dim=1, head_v_dim=1)
    want_y, want_window, _ = torch_gdn_decode(
        **{name: t.to(DEV) for name, t in inputs.items()}, **dims
    )
    got_y, got_window, _ = mlx_gdn_decode(
        **{name: torch_to_mlx(t.to(DEV)) for name, t in inputs.items()}, **dims
    )
    mx.eval(got_y, got_window)
    # fp32 math agrees to ~1e-7; rounding the input first moves the output >1e-3.
    torch.testing.assert_close(mlx_to_torch(got_y), want_y, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mlx_to_torch(got_window), want_window, atol=0, rtol=0)
