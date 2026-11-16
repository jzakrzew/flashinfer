# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental cache-free RoPE and per-tensor FP8 QKV preparation."""

from typing import Optional, Tuple

import torch

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def qkv_rope_quantize_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: Optional[torch.Tensor],
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    *,
    output_head_dim: Optional[int] = None,
    backend: str = "triton",
    sin: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate Q/K and quantize Q/K/V using independent GPU-resident scales.

    Experimental, JIT-only Triton API for SM100 and SM120. Inputs are
    nonempty BF16/FP16 THD tensors with identical shapes and strides (MHA), a
    contiguous feature dimension, and positive D divisible by eight, D <= 512.
    Input bases and token/head strides must be aligned to eight bytes.
    Packed QKV projection views are supported. GQA is unsupported.
    Logical element counts must fit the backend's guarded int32 range;
    input address offsets use int64.

    `cos_sin_cache` is contiguous FP32/BF16/FP16 [max_position, D], with
    half-width cosine followed by half-width sine, using the full-head
    split-half convention. Coefficients are converted to FP32 in registers;
    their storage dtype may differ from Q/K/V.
    `positions` is contiguous CUDA int32/int64 [T]; values must index the cache.
    When `sin` is supplied, `cos_sin_cache` instead holds a separate cosine
    table. Separate tables have matching nonzero row counts, contiguous
    features, and width D/2 or D; only their first D/2 columns are read. Their
    dtypes and row strides may differ. With `positions=None`, row indices are
    `token_index % table_rows`, supporting repeated tables across batches.
    This implicit mapping also works with packed tables. Coefficient values
    are read on every invocation and CUDA graph replay.
    RoPE uses FP32 arithmetic without rounding its result through the input dtype.

    Each scale is a contiguous one-element CUDA FP32 *dequantization* tensor:
    payload = FP8_E4M3(rotated_input / max(scale, FLT_MIN)), with saturating
    conversion. `FLT_MIN` is the smallest positive normal FP32 value (2**-126).
    Division uses reciprocal multiplication and can differ by one FP8 code
    near rounding boundaries. Scales are clamped before inversion to keep
    reciprocals finite.
    Scales must be finite and positive. They are read on device on every call,
    including CUDA graph replay, without host synchronization.

    `out`, when provided, contains contiguous FP8 E4M3 tensors matching Q/K/V.
    If `output_head_dim` is set, output feature dimensions use that size (at
    least D, at most 512, divisible by four), and tails are zeroed in the conversion launch. When
    omitted, the dimension is inferred from `out` or defaults to D.
    Outputs must not overlap inputs or one another. Warm up once before capture.
    No KV cache, history state, RMSNorm or scale update is involved.
    For static quantization, pass calibrated scales directly to this function;
    it performs only the fused RoPE/conversion launch. The partial-maximum API
    is optional and is needed only when the caller estimates scales at runtime.
    """
    from .experimental.qkv_rope_quantize.backend import quantize

    return quantize(
        q,
        k,
        v,
        cos_sin_cache,
        positions,
        q_scale,
        k_scale,
        v_scale,
        out,
        output_head_dim,
        backend,
        sin,
    )


@flashinfer_experimental_api
def qkv_rope_amax_partial(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: Optional[torch.Tensor],
    out: Optional[torch.Tensor] = None,
    num_partials: Optional[int] = None,
    reduction: str = "absmax",
    *,
    backend: str = "triton",
    sin: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute bounded partial maxima of FP32 rotated Q/K and raw V.

    The input and rotary contracts match `qkv_rope_quantize_fp8`. Returns a
    contiguous CUDA FP32 [num_partials, 3] tensor, with columns Q/K/V. Its
    columnwise maximum is the global absolute maximum for each operand
    with `reduction="absmax"` (default), or the signed maximum with
    `reduction="max"`.
    All entries are written, including unused partitions: zero for absmax
    and negative infinity for signed maxima. Inputs must be
    nonempty. `num_partials` must be in [1, 1024]. When omitted, it is inferred from
    `out`, or defaults to 1024 when `out` is absent. `out` allows storage reuse.

    This stage intentionally does not update scale history or quantize data.
    A caller can reduce these partials together with its scale-history policy,
    then call `qkv_rope_quantize_fp8` using the resulting device scales. This
    preserves the current-input scale dependency without materializing RoPE.
    """
    from .experimental.qkv_rope_quantize.backend import amax_partial

    return amax_partial(
        q, k, v, cos_sin_cache, positions, out, num_partials, reduction, backend, sin
    )
