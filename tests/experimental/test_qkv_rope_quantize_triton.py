# SPDX-License-Identifier: Apache-2.0
"""Public Triton QKV FP8 correctness against independent PyTorch arithmetic."""

import pytest
import torch
from flashinfer.qkv_rope_quantize import qkv_rope_quantize_fp8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 12),
    reason="requires Blackwell",
)


@pytest.mark.parametrize(
    "d,od", [(8, 8), (64, 64), (72, 80), (80, 80), (128, 144), (512, 512)]
)
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("coeff_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_triton_graph_reads_updated_scales_and_zeroes_padding(
    d, od, pos_dtype, dtype, coeff_dtype
):
    torch.manual_seed(37)
    t, h = 17, 3
    raw = torch.randn((t, 3, h, d), device="cuda", dtype=dtype)
    q, k, v = raw.unbind(1)
    cache = torch.randn((31, d), device="cuda", dtype=coeff_dtype)
    pos = torch.arange(t, device="cuda", dtype=pos_dtype)
    scales = [torch.tensor(x, device="cuda") for x in (0.03, 0.07, 0.09)]
    out = tuple(
        torch.empty((t, h, od), device="cuda", dtype=torch.float8_e4m3fn)
        for _ in range(3)
    )

    def fn():
        return qkv_rope_quantize_fp8(q, k, v, cache, pos, *scales, out=out)

    fn()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        fn()
    for s in scales:
        s.mul_(1.3)
    cache.mul_(0.7)
    graph.replay()
    co, si = cache[pos].chunk(2, -1)
    rotated = []
    for x in (q, k):
        left, right = x.float().chunk(2, -1)
        rotated.append(
            torch.cat(
                (
                    left * co[:, None] - right * si[:, None],
                    right * co[:, None] + left * si[:, None],
                ),
                -1,
            )
        )
    expected = [
        torch.nn.functional.pad((x / scale).clamp(-448, 448), (0, od - d)).to(
            torch.float8_e4m3fn
        )
        for x, scale in zip((*rotated, v.float()), scales, strict=True)
    ]
    for x, y in zip(out, expected, strict=False):
        torch.testing.assert_close(x.float(), y.float(), atol=0.25, rtol=0.13)
        if od > d:
            assert torch.count_nonzero(x.view(torch.uint8)[..., d:]) == 0


def test_triton_rejects_gqa_layout():
    q = torch.empty((3, 4, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((3, 2, 64), device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros((3, 64), device="cuda")
    pos = torch.arange(3, device="cuda")
    scales = [torch.ones((), device="cuda") for _ in range(3)]
    out = [torch.empty_like(q, dtype=torch.float8_e4m3fn) for _ in range(3)]
    with pytest.raises(ValueError, match="shared MHA"):
        qkv_rope_quantize_fp8(q, k, k, cache, pos, *scales, out=out)


@pytest.mark.parametrize("d", [64, 72])
@pytest.mark.parametrize("p", [1, 65])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("coeff_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_triton_partials_absmax_and_unused_partitions(d, p, dtype, coeff_dtype):
    from flashinfer.qkv_rope_quantize import qkv_rope_amax_partial

    raw = -torch.rand((17, 3, 3, d), device="cuda", dtype=dtype)
    q, k, v = raw.unbind(1)
    cache = torch.randn((19, d), device="cuda", dtype=coeff_dtype)
    pos = torch.arange(17, device="cuda")
    out = torch.empty((p, 3), device="cuda")
    qkv_rope_amax_partial(q, k, v, cache, pos, out=out, num_partials=p)
    co, si = cache[pos].chunk(2, -1)
    maxima = []
    for x in (q, k):
        left, right = x.float().chunk(2, -1)
        maxima.append(
            torch.maximum(
                (left * co[:, None] - right * si[:, None]).abs().amax(),
                (right * co[:, None] + left * si[:, None]).abs().amax(),
            )
        )
    expected = torch.stack((*maxima, v.float().abs().amax()))
    torch.testing.assert_close(out.amax(0), expected, atol=1e-4, rtol=1e-5)
    assert torch.all(out >= 0)
    if p == 65:
        assert torch.count_nonzero(out[-1]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("coeff_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_identity_rope_preserves_input_dtype_values(dtype, coeff_dtype):
    """Packed decoding must preserve signs, small values and saturation inputs."""
    tiny = torch.finfo(dtype).tiny
    values = torch.tensor(
        [0, -1, 1, -448, 448, 600, tiny, tiny / 128], dtype=dtype, device="cuda"
    )
    q = values.reshape(1, 1, 8)
    cache = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device="cuda", dtype=coeff_dtype)
    positions = torch.zeros(1, device="cuda", dtype=torch.int32)
    scales = [torch.tensor(s, device="cuda") for s in (1.0, 0.5, 2.0**-16)]
    actual = qkv_rope_quantize_fp8(q, q, q, cache, positions, *scales)
    for out, scale in zip(actual, scales, strict=True):
        expected = (q.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        torch.testing.assert_close(out.float(), expected.float(), atol=0, rtol=0)


def test_triton_rejects_mixed_input_dtypes():
    q = torch.ones((1, 1, 8), device="cuda", dtype=torch.float16)
    cache = torch.ones((1, 8), device="cuda")
    positions = torch.zeros(1, device="cuda", dtype=torch.int32)
    scale = torch.ones(1, device="cuda")
    with pytest.raises(ValueError, match="matching BF16/FP16"):
        qkv_rope_quantize_fp8(q, q.bfloat16(), q, cache, positions, scale, scale, scale)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("d", [64, 72])
@pytest.mark.parametrize("reduction", ["absmax", "max"])
@pytest.mark.parametrize("coeff_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_partial_reduction_mode_survives_graph_replay(dtype, d, reduction, coeff_dtype):
    """Signed maxima stay negative, unused partitions retain their identity."""
    from flashinfer.qkv_rope_quantize import qkv_rope_amax_partial

    raw = -torch.rand((17, 3, 3, d), device="cuda", dtype=dtype) - 1
    q, k, v = raw.unbind(1)
    cache = torch.cat(
        (
            torch.ones((17, d // 2), device="cuda", dtype=coeff_dtype),
            torch.zeros((17, d // 2), device="cuda", dtype=coeff_dtype),
        ),
        dim=-1,
    )
    pos = torch.arange(17, device="cuda", dtype=torch.int32)
    out = torch.empty((65, 3), device="cuda")

    def run():
        qkv_rope_amax_partial(
            q, k, v, cache, pos, out=out, num_partials=65, reduction=reduction
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    raw.mul_(0.5)
    graph.replay()
    expected = torch.stack(
        [
            (x.float().abs() if reduction == "absmax" else x.float()).amax()
            for x in (q, k, v)
        ]
    )
    torch.testing.assert_close(out.amax(0), expected, atol=0, rtol=0)
    if reduction == "absmax":
        assert torch.equal(out[-1], torch.zeros_like(out[-1]))
    else:
        assert torch.isneginf(out[-1]).all()
        assert (out.amax(0) < 0).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("d", [64, 72])
def test_cached_launch_accepts_fresh_pointers_sizes_and_minimal_alignment(dtype, d):
    """A cached binding must read new tensors, including scalar and offset views."""
    from flashinfer.qkv_rope_quantize import qkv_rope_amax_partial

    h, od = 3, 80 if d == 72 else d
    for t, factor in ((17, 1), (23, 2)):
        backing = torch.randint(1, 9, (t * 3 * h * (d + 4) + 4,), device="cuda").to(
            dtype
        )
        raw = backing[4:].view(t, 3, h, d + 4)[..., :d]
        q, k, v = raw.unbind(1)
        cache_storage = torch.zeros(t * d + 1, device="cuda", dtype=dtype)
        cache = cache_storage[1:].view(t, d)
        cache[:, : d // 2] = factor
        pos = torch.arange(t, device="cuda", dtype=torch.int64)
        scales = [
            torch.full(shape, value * factor, device="cuda")
            for shape, value in (((), 1.0), ((1,), 2.0), ((1, 1), 4.0))
        ]
        out = tuple(
            torch.empty(t * h * od + 4, device="cuda", dtype=torch.float8_e4m3fn)[
                4:
            ].view(t, h, od)
            for _ in range(3)
        )
        actual = qkv_rope_quantize_fp8(q, k, v, cache, pos, *scales, out=out)
        for result, x, scale, multiplier in zip(
            actual, (q, k, v), scales, (factor, factor, 1), strict=True
        ):
            expected = torch.nn.functional.pad(
                x.float() * multiplier / scale.reshape(()), (0, od - d)
            ).to(torch.float8_e4m3fn)
            torch.testing.assert_close(result.float(), expected.float(), atol=0, rtol=0)
        partials = qkv_rope_amax_partial(q, k, v, cache, pos, num_partials=7)
        expected_max = torch.stack(
            (q.float().amax() * factor, k.float().amax() * factor, v.float().amax())
        )
        torch.testing.assert_close(partials.amax(0), expected_max, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_singleton_strides_do_not_narrow_contiguous_tensor_contract(dtype):
    """Unit dimensions may have arbitrary strides despite being contiguous."""
    from flashinfer.qkv_rope_quantize import qkv_rope_amax_partial

    q = torch.full((1, 1, 8), 2.0, device="cuda", dtype=dtype)
    cache = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device="cuda", dtype=dtype)
    cache = cache.as_strided((1, 8), (0, 1))
    pos = torch.zeros(1, device="cuda", dtype=torch.int32).as_strided((1,), (0,))
    scales = [torch.ones(1, device="cuda").as_strided((1, 1), (0, 0)) for _ in range(3)]
    out = tuple(
        torch.empty(12, device="cuda", dtype=torch.float8_e4m3fn)[4:].as_strided(
            (1, 1, 8), (1, 1, 1)
        )
        for _ in range(3)
    )
    result = qkv_rope_quantize_fp8(q, q, q, cache, pos, *scales, out=out)
    for value in result:
        torch.testing.assert_close(value.float(), q.float(), atol=0, rtol=0)
    partial = torch.empty(3, device="cuda").as_strided((1, 3), (0, 1))
    qkv_rope_amax_partial(q, q, q, cache, pos, out=partial, num_partials=1)
    torch.testing.assert_close(partial, torch.full_like(partial, 2.0), atol=0, rtol=0)


@pytest.mark.parametrize("t,d,od", [(4960, 64, 64), (19840, 72, 80), (124000, 64, 64)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("coeff_dtype", [torch.float32, torch.bfloat16])
def test_shape_tuned_conversion_covers_updated_graph_inputs(
    t, d, od, dtype, coeff_dtype
):
    """Larger conversion tiles must preserve coverage and device reads on replay."""
    from flashinfer.experimental.qkv_rope_quantize.triton_backend import prepare

    torch.manual_seed(83)
    raw = torch.randn((t, 3, 16, d), device="cuda", dtype=dtype)
    q, k, v = raw.unbind(1)
    cache = torch.randn((4960, d), device="cuda", dtype=coeff_dtype)
    positions = torch.arange(t, device="cuda", dtype=torch.int32) % 4960
    scales = [torch.tensor(s, device="cuda") for s in (0.03, 0.07, 0.09)]
    out = tuple(
        torch.empty((t, 16, od), device="cuda", dtype=torch.float8_e4m3fn)
        for _ in range(3)
    )
    expected = tuple(torch.empty_like(x) for x in out)
    baseline = prepare(q, k, v, cache, positions, scales, expected)

    def run():
        qkv_rope_quantize_fp8(q, k, v, cache, positions, *scales, out=out)

    run()
    baseline()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    raw.mul_(-0.7)
    cache.mul_(0.9)
    positions.add_(7).remainder_(4960)
    for scale in scales:
        scale.mul_(1.3)
    graph.replay()
    baseline()
    for actual, reference in zip(out, expected, strict=True):
        torch.testing.assert_close(
            actual.view(torch.uint8), reference.view(torch.uint8), atol=0, rtol=0
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("coeff_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("d", [64, 72])
@pytest.mark.parametrize("explicit", [False, True])
def test_separate_rotary_tables_read_mutations_on_graph_replay(
    dtype, coeff_dtype, d, explicit
):
    """Separate strided tables and implicit rows match freshly packed coefficients."""
    from flashinfer.qkv_rope_quantize import qkv_rope_amax_partial

    torch.manual_seed(101)
    t, h = 21, 3
    rows = t if d == 64 else 7
    q, k, v = torch.randn((t, 3, h, d), device="cuda", dtype=dtype).unbind(1)
    # Full-width cosine and half-width sine, independent dtype/row strides.
    co = torch.randn((rows, d + 6), device="cuda", dtype=coeff_dtype)[:, 1 : d + 1]
    si = torch.randn((rows, d // 2 + 5), device="cuda", dtype=torch.float16)[
        :, 1 : d // 2 + 1
    ]
    positions = (
        torch.randint(rows, (t,), device="cuda", dtype=torch.int64)
        if explicit
        else None
    )
    scales = [torch.full((), x, device="cuda") for x in (0.03, 0.07, 0.09)]
    od = (d + 15) // 16 * 16
    outputs = tuple(
        torch.empty((t, h, od), device="cuda", dtype=torch.float8_e4m3fn)
        for _ in range(3)
    )
    partials = torch.empty((13, 3), device="cuda")

    def run():
        qkv_rope_amax_partial(
            q,
            k,
            v,
            co,
            positions,
            out=partials,
            num_partials=13,
            reduction="max",
            sin=si,
        )
        qkv_rope_quantize_fp8(q, k, v, co, positions, *scales, out=outputs, sin=si)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    co.mul_(0.7)
    si.add_(0.125)
    for scale in scales:
        scale.mul_(1.5)
    if explicit:
        positions.add_(1).remainder_(rows)
    graph.replay()
    packed = torch.cat((co[:, : d // 2].float(), si.float()), dim=-1)
    indices = positions if explicit else torch.arange(t, device="cuda") % rows
    reference_partials = qkv_rope_amax_partial(
        q, k, v, packed, indices, num_partials=13, reduction="max"
    )
    reference_outputs = qkv_rope_quantize_fp8(
        q, k, v, packed, indices, *scales, output_head_dim=od
    )
    torch.testing.assert_close(partials, reference_partials, atol=0, rtol=0)
    for actual, expected in zip(outputs, reference_outputs, strict=True):
        torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)


def test_separate_rotary_tables_reject_mismatched_rows():
    q = torch.empty((3, 2, 64), device="cuda", dtype=torch.bfloat16)
    co = torch.zeros((3, 32), device="cuda")
    si = torch.zeros((4, 32), device="cuda")
    scale = torch.ones((), device="cuda")
    with pytest.raises(ValueError, match="equal row counts"):
        qkv_rope_quantize_fp8(q, q, q, co, None, scale, scale, scale, sin=si)


def test_separate_rotary_tables_reuses_launch_with_fresh_rows_and_strides():
    """Cached code reads the current table shape, strides and pointers."""
    from flashinfer.qkv_rope_quantize import qkv_rope_amax_partial

    for t, rows, extra in ((3, 5, 0), (11, 5, 3), (7, 3, 1)):
        q = torch.randn((t, 2, 8), device="cuda", dtype=torch.bfloat16)
        co = torch.randn((rows, 4 + extra), device="cuda")[:, :4]
        si = torch.randn((rows, 4 + 2 * extra), device="cuda")[:, :4]
        packed = torch.cat((co, si), dim=-1)
        positions = torch.arange(t, device="cuda", dtype=torch.int32) % rows
        scale = torch.full((), 0.03, device="cuda")
        actual = qkv_rope_quantize_fp8(q, q, q, co, None, scale, scale, scale, sin=si)
        expected = qkv_rope_quantize_fp8(
            q, q, q, packed, positions, scale, scale, scale
        )
        for x, y in zip(actual, expected, strict=True):
            torch.testing.assert_close(x.float(), y.float(), atol=0, rtol=0)
        actual_partials = qkv_rope_amax_partial(
            q, q, q, co, None, num_partials=3, sin=si
        )
        expected_partials = qkv_rope_amax_partial(
            q, q, q, packed, positions, num_partials=3
        )
        torch.testing.assert_close(actual_partials, expected_partials, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale", [1e-40, torch.finfo(torch.float32).tiny, 3.0, 1e38])
def test_triton_scales_handle_extreme_values_and_rounding_boundaries(dtype, scale):
    # Float64 division is an independent oracle, including FP8 rounding ties.
    q = torch.ones((1, 1, 8), device="cuda", dtype=dtype)
    q[..., 0] = 0
    if scale < 1e-38 and dtype == torch.bfloat16:
        q[..., 1] = scale
    cache = torch.zeros((1, 8), device="cuda")
    coefficient = torch.nextafter(torch.tensor(0.890625), torch.tensor(0.0))
    if scale > 1e30:
        coefficient = torch.tensor(1e38)
    cache[:, :4] = coefficient.item()
    scales = [torch.tensor(scale, device="cuda") for _ in range(3)]
    out = qkv_rope_quantize_fp8(q, q, q, cache, None, *scales)
    for index, actual in enumerate(out):
        value = q.cpu().float()
        if index < 2:
            value = value * coefficient
        effective_scale = (
            scales[index].cpu().double().clamp_min(torch.finfo(torch.float32).tiny)
        )
        expected = (value.double() / effective_scale).float()
        expected = expected.clamp(-448, 448).to(torch.float8_e4m3fn)
        code_distance = (
            actual.cpu().view(torch.uint8).to(torch.int16)
            - expected.view(torch.uint8).to(torch.int16)
        ).abs()
        assert torch.all(code_distance <= (0 if scale < 1e-38 else 1))


def test_triton_rejects_python_float_scales_clearly():
    q = torch.zeros((1, 1, 8), device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros((1, 8), device="cuda")
    with pytest.raises(ValueError, match="float32 scalar scales"):
        qkv_rope_quantize_fp8(q, q, q, cache, None, 1.0, 1.0, 1.0)


def test_triton_partials_infers_partition_count_from_output():
    from flashinfer.qkv_rope_quantize import qkv_rope_amax_partial

    q = torch.ones((1, 1, 8), device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros((1, 8), device="cuda")
    cache[:, :4] = 1
    out = torch.empty((128, 3), device="cuda")
    result = qkv_rope_amax_partial(q, q, q, cache, None, out=out)
    assert result is out
    torch.testing.assert_close(result.amax(0), torch.ones(3, device="cuda"))
    with pytest.raises(ValueError, match="shape"):
        qkv_rope_amax_partial(q, q, q, cache, None, out=out, num_partials=64)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("d,od", [(64, 64), (72, 80), (128, 128), (256, 256)])
@pytest.mark.parametrize("identity", [False, True])
def test_static_calibration_reads_scales_without_amax_on_replay(
    monkeypatch, dtype, d, od, identity
):
    """Static preparation needs no reduction/history and preserves device scales."""
    from flashinfer.experimental.qkv_rope_quantize import triton_backend

    def unexpected_partials(*args, **kwargs):
        raise AssertionError("static quantization must not compute partial maxima")

    monkeypatch.setattr(triton_backend, "prepare_partials", unexpected_partials)
    torch.manual_seed(163)
    t, h = 21, 3
    raw = torch.randint(-16, 17, (t, 3, h, d), device="cuda").to(dtype)
    q, k, v = raw.unbind(1)
    table_rows = 1 if identity else 7
    cos = torch.ones((table_rows, d // 2), device="cuda", dtype=dtype)
    sin = torch.full_like(cos, 0.0 if identity else 0.5)
    scales = [torch.full((), x, device="cuda") for x in (0.125, 0.25, 0.5)]
    out = tuple(
        torch.empty((t, h, od), device="cuda", dtype=torch.float8_e4m3fn)
        for _ in range(3)
    )

    def run():
        return qkv_rope_quantize_fp8(q, k, v, cos, None, *scales, out=out, sin=sin)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    raw.add_(2)
    for scale, factor in zip(scales, (2.0, 0.5, 4.0), strict=True):
        scale.mul_(factor)
    expected_scales = [x.clone() for x in scales]
    graph.replay()
    rows = torch.arange(t, device="cuda") % table_rows
    co, si = cos[rows, None].float(), sin[rows, None].float()
    rotated = []
    for x in (q, k):
        left, right = x.float().chunk(2, dim=-1)
        rotated.append(torch.cat((left * co - right * si, right * co + left * si), -1))
    for actual, x, scale in zip(out, (*rotated, v.float()), scales, strict=True):
        expected = torch.nn.functional.pad(
            (x / scale).clamp(-448, 448), (0, od - d)
        ).to(torch.float8_e4m3fn)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)
    for actual, expected in zip(scales, expected_scales, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("d,od", [(8, 16), (64, 68), (64, 512)])
def test_triton_zeroes_entire_arbitrary_padding_tail(d, od):
    """Padding can exceed one half-head and need not be a multiple of 16."""
    q = torch.ones((5, 3, d), device="cuda", dtype=torch.bfloat16)
    cache = torch.cat((torch.ones((1, d // 2)), torch.zeros((1, d // 2))), -1).cuda()
    scale = torch.ones((), device="cuda")
    out = tuple(
        torch.empty((5, 3, od), device="cuda", dtype=torch.float8_e4m3fn)
        for _ in range(3)
    )
    for x in out:
        x.view(torch.uint8).fill_(127)
    qkv_rope_quantize_fp8(q, q, q, cache, None, scale, scale, scale, out=out)
    for x in out:
        torch.testing.assert_close(x.float()[..., :d], q.float(), rtol=0, atol=0)
        assert torch.count_nonzero(x.view(torch.uint8)[..., d:]) == 0


@pytest.mark.parametrize("explicit", [False, True])
def test_full_rotary_tables_reuse_compilation_across_token_counts(
    monkeypatch, explicit
):
    """Full tables and explicit positions must not specialize on table row count."""
    from flashinfer.experimental.qkv_rope_quantize import triton_backend
    from flashinfer.qkv_rope_quantize import qkv_rope_amax_partial

    def inputs(t):
        raw = torch.ones((t, 3, 3, 64), device="cuda", dtype=torch.bfloat16)
        q, k, v = raw.unbind(1)
        rows = t + 3 if explicit else t
        cos = torch.ones((rows, 32), device="cuda")
        sin = torch.zeros_like(cos)
        pos = torch.arange(t, device="cuda", dtype=torch.int32) if explicit else None
        return q, k, v, cos, sin, pos

    scale = torch.ones((), device="cuda")
    for t in (43, 57):
        q, k, v, cos, sin, pos = inputs(t)
        outputs = qkv_rope_quantize_fp8(q, k, v, cos, pos, scale, scale, scale, sin=sin)
        partials = qkv_rope_amax_partial(q, k, v, cos, pos, num_partials=7, sin=sin)
        for output in outputs:
            torch.testing.assert_close(output.float(), q.float(), rtol=0, atol=0)
        torch.testing.assert_close(
            partials.amax(0), torch.ones(3, device="cuda"), rtol=0, atol=0
        )
        if t == 43:

            def unexpected_compile(*args, **kwargs):
                pytest.fail(
                    "Changing T/table rows triggered a new Triton specialization"
                )

            for kernel in (triton_backend.convert, triton_backend.partial):
                monkeypatch.setattr(kernel, "compile", unexpected_compile)
