# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Triton RoPE and per-tensor FP8 QKV preparation."""

import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _values(
    Q,
    K,
    V,
    Cos,
    Sin,
    Pos,
    row,
    col,
    mask,
    H: tl.constexpr,
    D: tl.constexpr,
    TS: tl.constexpr,
    HS: tl.constexpr,
    PERIOD: tl.constexpr,
    CS: tl.constexpr,
    SS: tl.constexpr,
    EXPLICIT: tl.constexpr,
    SEPARATE: tl.constexpr,
):
    token = row // H
    head = row % H
    if EXPLICIT:
        position = tl.load(Pos + token, mask, 0).to(tl.int64)
    elif PERIOD == 0:
        position = token.to(tl.int64)
    else:
        position = (token % PERIOD).to(tl.int64)
    offset = token.to(tl.int64) * TS + head.to(tl.int64) * HS + col
    co = tl.load(Cos + position * CS + col, mask, 0).to(tl.float32)
    if SEPARATE:
        sine_offset = position * SS + col
    else:
        sine_offset = position * CS + D // 2 + col
    si = tl.load(Sin + sine_offset, mask, 0).to(tl.float32)
    qa = tl.load(Q + offset, mask, 0).to(tl.float32)
    qb = tl.load(Q + offset + D // 2, mask, 0).to(tl.float32)
    ka = tl.load(K + offset, mask, 0).to(tl.float32)
    kb = tl.load(K + offset + D // 2, mask, 0).to(tl.float32)
    va = tl.load(V + offset, mask, 0).to(tl.float32)
    vb = tl.load(V + offset + D // 2, mask, 0).to(tl.float32)
    return (
        qa * co - qb * si,
        qb * co + qa * si,
        ka * co - kb * si,
        kb * co + ka * si,
        va,
        vb,
    )


@triton.jit
def reciprocal(scale):
    return tl.inline_asm_elementwise(
        "rcp.approx.f32 $0,$1;",
        constraints="=f,f",
        args=[tl.maximum(scale, 1.1754943508222875e-38)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit(do_not_specialize=["T"], do_not_specialize_on_alignment=["QS", "KS", "VS"])
def convert(
    Q,
    K,
    V,
    Cos,
    Sin,
    Pos,
    QS,
    KS,
    VS,
    OQ,
    OK,
    OV,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    OD: tl.constexpr,
    TS: tl.constexpr,
    HS: tl.constexpr,
    PERIOD: tl.constexpr,
    CS: tl.constexpr,
    SS: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    EXPLICIT: tl.constexpr,
    SEPARATE: tl.constexpr,
):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC)
    mask = (row[:, None] < T * H) & (col[None, :] < D // 2)
    rq = reciprocal(tl.load(QS))
    rk = reciprocal(tl.load(KS))
    rv = reciprocal(tl.load(VS))
    q0, q1, k0, k1, v0, v1 = _values(
        Q,
        K,
        V,
        Cos,
        Sin,
        Pos,
        row[:, None],
        col[None, :],
        mask,
        H,
        D,
        TS,
        HS,
        PERIOD,
        CS,
        SS,
        EXPLICIT,
        SEPARATE,
    )
    base = row.to(tl.int64)[:, None] * OD + col[None, :]
    tl.store(OQ + base, tl.clamp(q0 * rq, -448.0, 448.0), mask)
    tl.store(OQ + base + D // 2, tl.clamp(q1 * rq, -448.0, 448.0), mask)
    tl.store(OK + base, tl.clamp(k0 * rk, -448.0, 448.0), mask)
    tl.store(OK + base + D // 2, tl.clamp(k1 * rk, -448.0, 448.0), mask)
    tl.store(OV + base, tl.clamp(v0 * rv, -448.0, 448.0), mask)
    tl.store(OV + base + D // 2, tl.clamp(v1 * rv, -448.0, 448.0), mask)
    if OD > D:
        pad_col = tl.arange(0, triton.next_power_of_2(OD - D))
        pad = (row[:, None] < T * H) & (pad_col[None, :] < OD - D)
        dst = row.to(tl.int64)[:, None] * OD + D + pad_col[None, :]
        tl.store(OQ + dst, 0.0, pad)
        tl.store(OK + dst, 0.0, pad)
        tl.store(OV + dst, 0.0, pad)


@triton.jit(do_not_specialize=["T"])
def partial(
    Q,
    K,
    V,
    Cos,
    Sin,
    Pos,
    Out,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    TS: tl.constexpr,
    HS: tl.constexpr,
    PERIOD: tl.constexpr,
    CS: tl.constexpr,
    SS: tl.constexpr,
    P: tl.constexpr,
    B: tl.constexpr,
    EXPLICIT: tl.constexpr,
    SEPARATE: tl.constexpr,
    ABSMAX: tl.constexpr,
):
    pid = tl.program_id(0)
    count = T * H * (D // 2)
    steps = tl.cdiv(count, P * B)
    offsets = pid * steps * B + tl.arange(0, B)
    identity = 0.0 if ABSMAX else -float("inf")
    mq = tl.full((B,), identity, tl.float32)
    mk = tl.full((B,), identity, tl.float32)
    mv = tl.full((B,), identity, tl.float32)
    for i in range(steps):
        pos = offsets + i * B
        row = pos // (D // 2)
        col = pos % (D // 2)
        valid = pos < count
        q0, q1, k0, k1, v0, v1 = _values(
            Q,
            K,
            V,
            Cos,
            Sin,
            Pos,
            row,
            col,
            valid,
            H,
            D,
            TS,
            HS,
            PERIOD,
            CS,
            SS,
            EXPLICIT,
            SEPARATE,
        )
        if ABSMAX:
            q0, q1 = tl.abs(q0), tl.abs(q1)
            k0, k1 = tl.abs(k0), tl.abs(k1)
            v0, v1 = tl.abs(v0), tl.abs(v1)
        mq = tl.maximum(mq, tl.where(valid, tl.maximum(q0, q1), identity))
        mk = tl.maximum(mk, tl.where(valid, tl.maximum(k0, k1), identity))
        mv = tl.maximum(mv, tl.where(valid, tl.maximum(v0, v1), identity))
    tl.store(Out + pid * 3, tl.max(mq, 0))
    tl.store(Out + pid * 3 + 1, tl.max(mk, 0))
    tl.store(Out + pid * 3 + 2, tl.max(mv, 0))


def _validate_inputs(q, k, v, cache, pos, sine=None):
    if q.ndim != 3 or any(size <= 0 for size in q.shape):
        raise ValueError("Triton backend requires nonempty THD tensors")
    t, h, d = q.shape
    if (
        q.dtype not in (torch.bfloat16, torch.float16)
        or any(x.dtype != q.dtype for x in (k, v))
        or d % 8
        or d > 512
    ):
        raise ValueError(
            "Triton backend requires matching BF16/FP16 dtypes and positive D divisible by 8, D<=512"
        )
    if any(x.shape != q.shape or x.stride() != q.stride() for x in (k, v)):
        raise ValueError("Triton backend requires shared MHA QKV shape/strides")
    if q.stride(2) != 1 or any(st < 0 or st % 4 for st in q.stride()[:2]):
        raise ValueError(
            "Triton backend requires contiguous features and 8-byte aligned strides"
        )
    if any(x.data_ptr() % 8 for x in (q, k, v)):
        raise ValueError("Triton backend requires 8-byte aligned inputs")
    if q.device.type != "cuda" or any(
        x.device != q.device for x in (k, v, cache, pos, sine) if x is not None
    ):
        raise ValueError("Triton backend tensors must share one CUDA device")
    tables = (cache,) if sine is None else (cache, sine)
    for table in tables:
        if (
            table.dtype not in (torch.float32, torch.bfloat16, torch.float16)
            or table.ndim != 2
            or table.shape[0] == 0
            or table.stride(1) != 1
            or (table.shape[0] > 1 and table.stride(0) <= 0)
            or (sine is None and (table.shape[1] != d or not table.is_contiguous()))
            or (sine is not None and table.shape[1] not in (d // 2, d))
        ):
            raise ValueError(
                "Triton backend requires FP32/BF16/FP16 rotary tables with contiguous features; packed [rows,D] or separate [rows,D/2] / [rows,D]"
            )
    if sine is not None and sine.shape[0] != cache.shape[0]:
        raise ValueError("Separate cosine/sine tables must have equal row counts")
    if pos is not None and (
        pos.dtype not in (torch.int32, torch.int64)
        or pos.ndim != 1
        or pos.numel() != t
        or not pos.is_contiguous()
    ):
        raise ValueError("Triton backend requires contiguous int32/int64 positions[T]")
    if _device_capability(q.device)[0] not in (10, 12):
        raise ValueError("Triton backend requires SM100 or SM120")
    return t, h, d


@functools.cache
def _device_capability(device):
    return torch.cuda.get_device_capability(device)


def _conversion_config(arch, d, elements):
    if arch[0] == 10:
        if d >= 256:
            return 8, 4
        if d not in (64, 128):
            return 8, 4
        return (64 if elements >= 2**26 else 32), 4
    if d == 64:
        return (8, 8) if elements >= 2**26 else (16, 4) if elements >= 2**23 else (8, 4)
    return (8, 8) if d <= 128 else (8, 4)


def prepare(q, k, v, cache, pos, scales, out, *, sin=None):
    """Validate inputs and return a prepared, allocation-free conversion call."""
    t, h, d = _validate_inputs(q, k, v, cache, pos, sin)
    if len(out) != 3 or out[0].ndim != 3:
        raise ValueError("Triton backend requires three contiguous FP8 outputs")
    od = out[0].shape[-1]
    if od < d or od > 512 or od % 4:
        raise ValueError(
            "Triton backend padded D must be divisible by four, D<=padded D<=512"
        )
    if any(
        x.shape != (t, h, od)
        or x.dtype != torch.float8_e4m3fn
        or x.device != q.device
        or not x.is_contiguous()
        or x.data_ptr() % 4
        for x in out
    ):
        raise ValueError(
            "Triton backend requires aligned contiguous FP8 outputs[T,H,padded_D]"
        )
    if len(scales) != 3 or any(
        not isinstance(x, torch.Tensor)
        or x.dtype != torch.float32
        or x.numel() != 1
        or x.device != q.device
        or not x.is_contiguous()
        for x in scales
    ):
        raise ValueError("Triton backend requires three CUDA float32 scalar scales")
    rows, warps = _conversion_config(_device_capability(q.device), d, t * h * od)
    if t * h * od + rows > 2**31 - 1:
        raise ValueError(
            "Triton backend conversion exceeds its safe int32 indexing range"
        )
    sine = cache if sin is None else sin
    positions = cache if pos is None else pos
    args = (
        q,
        k,
        v,
        cache,
        sine,
        positions,
        *scales,
        *out,
        t,
        h,
        d,
        od,
        q.stride(0),
        q.stride(1),
        cache.shape[0] if pos is None and cache.shape[0] < t else 0,
        cache.stride(0),
        sine.stride(0),
        rows,
        triton.next_power_of_2(d // 2),
        pos is not None,
        sin is not None,
    )
    grid = (triton.cdiv(t * h, rows),)
    return lambda: convert[grid](*args, num_warps=warps, enable_fp_fusion=False)


def prepare_partials(q, k, v, cache, pos, out, reduction="absmax", *, sin=None):
    """Prepare signed or absolute partial maxima without updating caller scales."""
    t, h, d = _validate_inputs(q, k, v, cache, pos, sin)
    if reduction not in ("max", "absmax"):
        raise ValueError("reduction must be absmax or max")
    if (
        out.ndim != 2
        or out.shape[1] != 3
        or not 1 <= out.shape[0] <= 1024
        or out.dtype != torch.float32
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "Triton backend requires contiguous float32 partials[P,3], 1<=P<=1024"
        )
    sine = cache if sin is None else sin
    positions = cache if pos is None else pos
    arch = _device_capability(q.device)
    small = t * h * d <= 8192 * out.shape[0]
    block, warps = (
        ((512, 8) if small else (1024, 4))
        if arch[0] == 10
        else ((256, 4) if small and d == 64 else (1024, 8) if not small else (512, 4))
    )
    if t * h * d + out.shape[0] * block > 2**31 - 1:
        raise ValueError(
            "Triton backend partial reduction exceeds its safe int32 indexing range"
        )
    args = (
        q,
        k,
        v,
        cache,
        sine,
        positions,
        out,
        t,
        h,
        d,
        q.stride(0),
        q.stride(1),
        cache.shape[0] if pos is None and cache.shape[0] < t else 0,
        cache.stride(0),
        sine.stride(0),
        out.shape[0],
        block,
        pos is not None,
        sin is not None,
        reduction == "absmax",
    )
    grid = (out.shape[0],)
    return lambda: partial[grid](*args, num_warps=warps, enable_fp_fusion=False)
