"""Single-pass split-half RoPE for cached, possibly broadcast cos/sin tables."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="sglang_cached_split_half_rope",
        input_names=["x", "cosine", "sine"],
        output_names=["out"],
        header="#pragma clang fp contract(off)\n",
        source="""
            uint i = thread_position_in_grid.x;
            if (i >= N) return;
            uint d = i % D;
            uint l = (i / D) % L;
            uint h = (i / (D * L)) % H;
            uint b = i / (D * L * H);
            uint ci = (((C0 == 1 ? 0 : b) * C1 + (C1 == 1 ? 0 : h))
                       * C2 + (C2 == 1 ? 0 : l)) * D + d;
            uint si = (((S0 == 1 ? 0 : b) * S1 + (S1 == 1 ? 0 : h))
                       * S2 + (S2 == 1 ? 0 : l)) * D + d;
            uint other = d < D / 2 ? i + D / 2 : i - D / 2;
            T rotated = d < D / 2 ? -x[other] : x[other];
            // Match the two dtype-rounded products before the final sum.
            T left = T(float(x[i]) * float(cosine[ci]));
            T right = T(float(rotated) * float(sine[si]));
            out[i] = T(float(left) + float(right));
        """,
    )


def cached_rotary_embedding(value, cosine, sine):
    """Evaluate one matched pattern; shape constants never depend on table data."""
    if (
        value.ndim != 4
        or value.shape[-1] % 2
        or any(
            table.ndim != 4
            or table.dtype != value.dtype
            or table.shape[-1] != value.shape[-1]
            or any(a not in (1, b) for a, b in zip(table.shape, value.shape))
            for table in (cosine, sine)
        )
    ):
        raise ValueError("cached RoPE fusion requires matching 4D broadcast tables")
    _, heads, length, width = value.shape
    template = [("T", value.dtype), ("N", value.size), ("D", width)]
    template.extend((("H", heads), ("L", length)))
    for prefix, table in (("C", cosine), ("S", sine)):
        template.extend((f"{prefix}{axis}", table.shape[axis]) for axis in range(3))
    return _kernel()(
        inputs=[value, cosine, sine],
        template=template,
        grid=(value.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[value.shape],
        output_dtypes=[value.dtype],
    )[0]
