# Experimental QKV RoPE / FP8 preparation

The public APIs `flashinfer.qkv_rope_quantize.qkv_rope_quantize_fp8` and
`flashinfer.qkv_rope_quantize.qkv_rope_amax_partial` use Triton exclusively
(`backend="triton"`). Calling either opts into experimental JIT-only support.

Supported: SM100/SM120; nonempty BF16/FP16 THD Q/K/V with identical shapes and
strides, contiguous features, eight-byte aligned input bases and token/head
strides, and D divisible by eight up to 512. This includes packed MHA QKV
views for Pixtral/Nemotron D64 and Qwen D64/72/80. GQA, independent
QKV strides, empty inputs, and oversized logical index ranges
are rejected. Input addressing uses int64; logical loop counters use guarded
int32 ranges. See API docstrings for the full contract.

The caller supplies contiguous FP32/BF16/FP16 [max_position,D] cosine/sine
rows and int32/int64 positions. Coefficient storage dtype is independent of
Q/K/V dtype; coefficients are converted to FP32 in registers, without a
separate conversion allocation or kernel. Full-head split-half RoPE uses FP32
arithmetic without FMA contraction and no intermediate BF16/FP16 rounding. Q/K/V have independent GPU FP32
dequantization scales: payload = saturating E4M3(rotated input / max(scale,
FLT_MIN)), where FLT_MIN = 2**-126 is the smallest positive normal FP32 value.
Scale division uses reciprocal multiplication, permitting an adjacent FP8
code near rounding boundaries. Scales are clamped to FLT_MIN before
inversion, keeping their reciprocals finite.
Scales must be positive and finite; positions must index valid cache rows.
Values are not copied to the CPU for validation.

Alternatively, pass a cosine table as `cos_sin_cache` and a separate sine
table through `sin=`. Each table may have width D/2 or D, contiguous features,
independent positive row strides, and independent FP32/BF16/FP16 storage dtype.
Only the first D/2 columns are read. With `positions=None`, coefficient rows
are `token_index % table_rows`, allowing a table to repeat over flattened
batches without allocating a position vector. Both layouts read current
coefficient values during graph replay; no caller tensors are cached.

Conversion can pad D72 to D80 with zero tails in the same launch. Output
width must be divisible by four and between D and 512. Partials are FP32
[P,3] maxima (1 <= P <= 1024). The default `reduction="absmax"` uses
absolute values and zero unused partitions; `reduction="max"` uses signed
values and negative-infinity unused partitions. Reduction mode is selected
at compilation and included in the compilation cache key.
The partial count is inferred from caller-provided output storage when
`num_partials` is omitted, or defaults to 1024 without output storage.

History, final reduction and scale updates remain caller-owned. Outputs must
not overlap inputs or one another. Warm up before CUDA graph capture;
replay reads current device scales and launches on the current stream.
Direct full-graph torch.compile support is not claimed.

Input and coefficient dtypes select separate Triton specializations; no
device-side dtype branch or input conversion kernel is introduced. The public
API selects launch geometry from architecture and shape metadata. Triton caches
compiled kernels; prepared calls retain launch metadata and read current tensor
contents on the current CUDA stream.

## Static calibration scales

`qkv_rope_quantize_fp8` is already the static-scale fused operation. Supply the
three calibrated GPU FP32 descales directly; it rotates Q/K, converts all Q/K/V,
and zero-pads any output tails in one kernel launch. Do not call
`qkv_rope_amax_partial`, reduce maxima, or maintain history for static scaling.
For example, using the Q/K/V views and coefficient tables from the example below:

```python
q_scale, k_scale, v_scale = [
    torch.full((), value, device=q.device, dtype=torch.float32)
    for value in (0.03, 0.07, 0.09)  # replace with calibrated descales
]
q_fp8, k_fp8, v_fp8 = qkv_rope_quantize_fp8(
    q,
    k,
    v,
    cos,
    None,
    q_scale,
    k_scale,
    v_scale,
    sin=sin,
    output_head_dim=((q.shape[-1] + 15) // 16) * 16,
)
```

These tensors are not modified by the operation. Reloaded scale values are read
on the next invocation or CUDA graph replay. Encoders without RoPE can use a
single implicit coefficient row with cosine one and sine zero, using the same
API and one conversion launch.

## Example

This example uses current-input absolute maxima. Applications may instead
combine partials with their own scale-history policy.

```python
import torch
from flashinfer.qkv_rope_quantize import qkv_rope_amax_partial, qkv_rope_quantize_fp8

qkv = torch.randn(17, 3, 16, 64, device="cuda", dtype=torch.bfloat16)
q, k, v = qkv.unbind(1)
angles = torch.randn(17, 32, device="cuda", dtype=torch.float32)
cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(q.dtype)
positions = torch.arange(17, device="cuda", dtype=torch.int32)
partials = qkv_rope_amax_partial(q, k, v, cache, positions, num_partials=128)
scales = (partials.amax(dim=0) / 448).clamp_min(torch.finfo(torch.float32).tiny)
q_fp8, k_fp8, v_fp8 = qkv_rope_quantize_fp8(
    q, k, v, cache, positions, scales[0:1], scales[1:2], scales[2:3]
)
```

For existing separate tables, no packing or position construction is needed:

```python
cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
partials = qkv_rope_amax_partial(q, k, v, cos, None, sin=sin)
scales = (partials.amax(dim=0) / 448).clamp_min(torch.finfo(torch.float32).tiny)
q_fp8, k_fp8, v_fp8 = qkv_rope_quantize_fp8(
    q, k, v, cos, None, *scales.unbind(), sin=sin
)
```

## Validation

Run `.venv/bin/python -m pytest -q tests/experimental/test_qkv_rope_quantize_triton.py`.
The suite covers BF16/FP16 inputs, FP32/BF16/FP16 coefficients, graph replay with
changing scales and coefficients, padding, int32/int64 positions,
absolute/signed partial reductions, subnormal values, mixed-dtype rejection,
fresh tensor pointers, token-count changes and minimally aligned offset views.
Singleton dimensions with arbitrary valid contiguous strides are covered.
End-to-end serving performance is not established by these tests.
