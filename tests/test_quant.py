"""The Python and C++ quantizers must agree byte for byte, and the SIMD kernels must
agree with their scalar reference.

Byte-exactness is what makes the PyTorch twin a faithful stand-in for the engine, so
these tests guard the foundation of every later acceptance measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from specdraft import _engine as cpp
from specdraft import quant as pyq

FORMATS = ["q4", "q8", "a8"]
LENGTHS = [32, 64, 1024, 4096]
MAGNITUDES = [1e-6, 1e-3, 1.0, 100.0]

_PY_QUANTIZE = {"q4": pyq.quantize_q4, "q8": pyq.quantize_q8, "a8": pyq.quantize_a8}
_CPP_QUANTIZE = {"q4": cpp.quantize_q4, "q8": cpp.quantize_q8, "a8": cpp.quantize_a8}
_PY_DEQUANTIZE = {"q4": pyq.dequantize_q4, "q8": pyq.dequantize_q8, "a8": pyq.dequantize_a8}
_CPP_DEQUANTIZE = {"q4": cpp.dequantize_q4, "q8": cpp.dequantize_q8, "a8": cpp.dequantize_a8}


def sample(kind: str, n: int, magnitude: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "normal":
        x = rng.standard_normal(n)
    elif kind == "uniform":
        x = rng.uniform(-1.0, 1.0, n)
    elif kind == "spiky":  # a few outliers per block stress the shared scale
        x = rng.standard_normal(n)
        x[::7] *= 50.0
    elif kind == "ties":  # values that land exactly halfway between quantization levels
        x = rng.integers(-16, 17, n) / 2.0
    else:
        raise ValueError(kind)
    return (x * magnitude).astype(np.float32)


def test_block_layout_matches():
    assert cpp.QK == pyq.QK == 32
    assert cpp.BLOCK_SIZES == pyq.BLOCK_BYTES


def test_cpu_features_reported():
    features = cpp.cpu_features()
    assert set(features) == {"avx2", "fma", "f16c", "avx_vnni", "avx512f"}
    assert features["avx2"] and features["fma"] and features["f16c"], "AVX2/FMA/F16C is the baseline"
    assert cpp.kernel_path() in {"vnni", "scalar"}


@pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 0.1, 65504.0, 1e-5, 1e-8, 123456.0, -3.14159])
def test_fp16_conversion_matches_numpy(value):
    with np.errstate(over="ignore"):  # 123456.0 overflows to inf in fp16, on both sides
        as_fp16 = np.float16(np.float32(value))
    assert cpp.fp32_to_fp16(value) == int(as_fp16.view(np.uint16))
    assert cpp.fp16_to_fp32(int(as_fp16.view(np.uint16))) == pytest.approx(float(as_fp16), abs=0.0)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("kind", ["normal", "uniform", "spiky", "ties"])
@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_quantize_bytes_identical(fmt, kind, n, magnitude):
    x = sample(kind, n, magnitude, seed=n + int(magnitude * 7))
    assert _PY_QUANTIZE[fmt](x) == _CPP_QUANTIZE[fmt](x)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize(
    "x",
    [
        np.zeros(64, dtype=np.float32),
        np.full(64, 3.5, dtype=np.float32),
        np.full(32, -7.0, dtype=np.float32),
        np.tile([1.0, -1.0], 16).astype(np.float32),
        np.linspace(-1.0, 1.0, 96, dtype=np.float32),
        np.array([1e-8] * 31 + [1e-7], dtype=np.float32),  # scale underflows fp16
        np.array([65504.0] + [0.0] * 31, dtype=np.float32),  # largest fp16 scale
    ],
)
def test_quantize_bytes_identical_edge_cases(fmt, x):
    assert _PY_QUANTIZE[fmt](x) == _CPP_QUANTIZE[fmt](x)


@pytest.mark.parametrize("fmt", FORMATS)
def test_dequantize_identical(fmt):
    x = sample("normal", 2048, 1.0, seed=1)
    blob = _PY_QUANTIZE[fmt](x)
    assert np.array_equal(_PY_DEQUANTIZE[fmt](blob), _CPP_DEQUANTIZE[fmt](blob))


@pytest.mark.parametrize("fmt,max_relative_rms", [("q4", 0.12), ("q8", 0.01), ("a8", 0.01)])
def test_roundtrip_error_is_bounded(fmt, max_relative_rms):
    x = sample("normal", 32768, 1.0, seed=2)
    error = _PY_DEQUANTIZE[fmt](_PY_QUANTIZE[fmt](x)) - x
    assert np.sqrt(np.mean(error**2)) / np.std(x) < max_relative_rms


def test_q4_preserves_the_largest_value_in_a_block():
    x = sample("spiky", 32, 1.0, seed=3)
    recovered = _PY_DEQUANTIZE["q4"](_PY_QUANTIZE["q4"](x))
    peak = int(np.abs(x).argmax())
    # The peak maps to nibble 0, so only the fp16 rounding of the scale costs anything.
    assert recovered[peak] == pytest.approx(x[peak], rel=1e-3)


@pytest.mark.parametrize("weight_fmt", ["q4", "q8"])
@pytest.mark.parametrize("n", [32, 1024, 4096])
def test_dot_matches_exact_reference(weight_fmt, n):
    w = sample("normal", n, 1.0, seed=10)
    x = sample("normal", n, 0.5, seed=11)
    w_blob = _PY_QUANTIZE[weight_fmt](w)
    x_blob = pyq.quantize_a8(x)

    # Both kernels multiply exactly these dequantized values, so a float64 dot product of
    # them is the exact answer; only the float32 summation order differs.
    wq = _PY_DEQUANTIZE[weight_fmt](w_blob).astype(np.float64)
    xq = pyq.dequantize_a8(x_blob).astype(np.float64)
    exact = float(wq @ xq)
    tolerance = 1e-6 * float(np.abs(wq * xq).sum())

    dispatched = getattr(cpp, f"dot_{weight_fmt}_a8")(w_blob, x_blob)
    scalar = getattr(cpp, f"dot_{weight_fmt}_a8_scalar")(w_blob, x_blob)
    assert abs(dispatched - exact) <= tolerance
    assert abs(scalar - exact) <= tolerance


@pytest.mark.parametrize("weight_fmt", ["q4", "q8"])
def test_dot_zero_activations_is_exactly_zero(weight_fmt):
    w_blob = _PY_QUANTIZE[weight_fmt](sample("normal", 256, 1.0, seed=12))
    x_blob = pyq.quantize_a8(np.zeros(256, dtype=np.float32))
    assert getattr(cpp, f"dot_{weight_fmt}_a8")(w_blob, x_blob) == 0.0


@pytest.mark.parametrize("fmt", FORMATS)
def test_bad_lengths_raise(fmt):
    with pytest.raises(ValueError):
        _PY_QUANTIZE[fmt](np.zeros(33, dtype=np.float32))
    with pytest.raises(Exception):
        _CPP_QUANTIZE[fmt](np.zeros(33, dtype=np.float32))


def test_fake_quantize_keeps_shape():
    x = sample("normal", 4 * 64, 1.0, seed=13).reshape(4, 64)
    out = pyq.fake_quantize(x, "q4")
    assert out.shape == x.shape and out.dtype == np.float32
    # Blocks run along the last axis, so each row is quantized independently.
    assert np.array_equal(out[1], pyq.fake_quantize(x[1], "q4"))


# ------------------------------------------------- the k-token kernel (plan §10.1)


@pytest.mark.parametrize("weight_fmt", ["q4", "q8"])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 8, 13])
@pytest.mark.parametrize("n", [32, 1024])
def test_multi_token_kernel_is_bit_exact_against_one_token_at_a_time(weight_fmt, tokens, n):
    """This is what lets verification share one pass over the weights without changing a
    single output bit, which is what bit-exact greedy speculative decoding needs."""
    w_blob = _PY_QUANTIZE[weight_fmt](sample("normal", n, 1.0, seed=20))
    vectors = [sample("normal", n, 0.5, seed=30 + i) for i in range(tokens)]
    blobs = [pyq.quantize_a8(v) for v in vectors]

    together = getattr(cpp, f"dot_{weight_fmt}_a8_multi")(w_blob, b"".join(blobs))
    separately = [getattr(cpp, f"dot_{weight_fmt}_a8")(w_blob, blob) for blob in blobs]

    assert together.shape == (tokens,)
    assert np.array_equal(together, np.array(separately, dtype=np.float32))


@pytest.mark.parametrize("weight_fmt", ["q4", "q8"])
def test_multi_token_kernel_rejects_ragged_input(weight_fmt):
    w_blob = _PY_QUANTIZE[weight_fmt](sample("normal", 64, 1.0, seed=21))
    short = pyq.quantize_a8(sample("normal", 32, 1.0, seed=22))
    with pytest.raises(Exception):
        getattr(cpp, f"dot_{weight_fmt}_a8_multi")(w_blob, short)


# --------------------------------------------------------------------- the kernel microbenchmark
#
# bench_dot exists to time the kernel away from the engine, so what matters is that the work it
# reports is the work it actually did -- a benchmark that miscounts is worse than no benchmark.


def test_bench_dot_reports_the_work_it_did():
    result = cpp.bench_dot(rows=4, n_in=64, tokens=3, iters=2, format="q4")
    assert result["macs"] == 4 * 64 * 3 * 2
    assert result["weight_bytes"] == 4 * (64 // 32) * 18
    assert result["weight_bytes_read"] == result["weight_bytes"] * 2
    assert result["activation_bytes"] == 3 * (64 // 32) * 36
    assert result["seconds"] > 0.0
    assert np.isfinite(result["checksum"])


def test_bench_dot_q8_blocks_are_larger():
    result = cpp.bench_dot(rows=4, n_in=64, tokens=1, iters=1, format="q8")
    assert result["weight_bytes"] == 4 * (64 // 32) * 34


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 0, "n_in": 32, "tokens": 1},
        {"rows": 1, "n_in": 0, "tokens": 1},
        {"rows": 1, "n_in": 32, "tokens": 0},
        {"rows": 1, "n_in": 32, "tokens": 1, "iters": 0},
        {"rows": 1, "n_in": 48, "tokens": 1},  # not a whole number of blocks
        {"rows": 1, "n_in": 32, "tokens": 1, "format": "q3"},
    ],
)
def test_bench_dot_rejects_nonsense(kwargs):
    with pytest.raises(ValueError):
        cpp.bench_dot(**kwargs)


# ------------------------------------------------------------------------------ the SoA prototype
#
# Two things in it are easy to get subtly wrong and impossible to notice from a benchmark: the
# offset trick, which replaces the sign trick with a per-lane bias, and the eight-block reduction,
# whose lane order has to come out in block order or every scale lands on the wrong block.


@pytest.mark.parametrize("n", [32, 64, 256, 288, 2560])  # 288 is nine blocks, exercising the tail
@pytest.mark.parametrize("tokens", [1, 3])
def test_soa_kernel_matches_the_exact_reference(n, tokens):
    w = sample("normal", n, 1.0, seed=40)
    vectors = [sample("normal", n, 0.5, seed=41 + i) for i in range(tokens)]
    w_blob = pyq.quantize_q4(w)
    blobs = [pyq.quantize_a8(v) for v in vectors]

    wq = pyq.dequantize_q4(w_blob).astype(np.float64)
    exact = np.array([wq @ pyq.dequantize_a8(b).astype(np.float64) for b in blobs])
    tolerance = 1e-6 * float(np.abs(wq).sum()) * 0.5

    got = cpp.dot_q4_a8_soa(w_blob, b"".join(blobs))
    assert got.shape == (tokens,)
    assert np.abs(got - exact).max() <= tolerance


def test_soa_kernel_handles_zero_activations():
    w_blob = pyq.quantize_q4(sample("normal", 256, 1.0, seed=43))
    x_blob = pyq.quantize_a8(np.zeros(256, dtype=np.float32))
    # The bias is zero when every activation is zero, so this also checks the offset correction
    # is not adding a constant of its own.
    assert cpp.dot_q4_a8_soa(w_blob, x_blob)[0] == 0.0


def test_soa_kernel_rejects_ragged_input():
    w_blob = pyq.quantize_q4(sample("normal", 64, 1.0, seed=44))
    short = pyq.quantize_a8(sample("normal", 32, 1.0, seed=45))
    with pytest.raises(ValueError):
        cpp.dot_q4_a8_soa(w_blob, short)


def test_bench_dot_soa_reports_the_same_work_as_the_interleaved_one():
    soa = cpp.bench_dot_soa(rows=4, n_in=64, tokens=3, iters=2)
    aos = cpp.bench_dot(rows=4, n_in=64, tokens=3, iters=2, format="q4")
    assert soa["macs"] == aos["macs"]
    assert soa["weight_bytes"] == aos["weight_bytes"]  # the same 18 bytes a block, in two arrays
    assert soa["seconds"] > 0.0


# ------------------------------------------------- activations in the layout the engine's kernels read
#
# quantize_a8_soa is a second implementation of quantize_a8's arithmetic writing to a different
# shape. If the two ever disagree the engine stops matching the PyTorch twin, and the twin is what
# every offline acceptance number is scored through, so pin them together.


def split_a8(blob: bytes):
    """The scales and the quantized bytes of an interleaved a8 blob."""
    raw = np.frombuffer(blob, dtype=np.uint8).reshape(-1, 36)
    scales = raw[:, :4].copy().view(np.float32).ravel()
    qs = raw[:, 4:].copy().view(np.int8).ravel()
    return scales, qs


def unpair_a8(qs: np.ndarray) -> np.ndarray:
    """Undo the paired activation order, returning one row of 32 values per block.

    The q4 kernel reads two blocks with one 32-byte weight load, so a pair of blocks is stored as
    x[b][0:16], x[b+1][0:16], x[b][16:32], x[b+1][16:32]. Only whole groups of eight blocks are
    paired; the rest keep the plain order. This mirrors `paired_index` in quant.cpp -- if the two ever
    disagree, the tests below are the ones that notice.
    """
    nblocks = qs.size // 32
    pairs_end = nblocks - nblocks % 8
    out = np.empty((nblocks, 32), dtype=np.int8)
    for b in range(nblocks):
        for i in range(32):
            if b < pairs_end:
                out[b, i] = qs[(b // 2) * 64 + ((i // 16) * 2 + (b & 1)) * 16 + (i % 16)]
            else:
                out[b, i] = qs[b * 32 + i]
    return out


@pytest.mark.parametrize("kind", ["normal", "uniform", "spiky", "ties"])
@pytest.mark.parametrize("n", [32, 256, 2560])
@pytest.mark.parametrize("paired", [False, True])
def test_soa_activation_quantizer_agrees_byte_for_byte(kind, n, paired):
    x = sample(kind, n, 1.0, seed=50)
    want_scales, want_qs = split_a8(pyq.quantize_a8(x))
    got = cpp.quantize_a8_soa(x, 8, paired)

    # The same bytes either way; `paired` only decides where each one sits.
    laid_out = unpair_a8(got["qs"]) if paired else got["qs"].reshape(-1, 32)
    assert np.array_equal(laid_out.ravel(), want_qs)
    assert np.array_equal(got["scales"], want_scales)


@pytest.mark.parametrize("zero_point", [8, 128])
@pytest.mark.parametrize("paired", [False, True])
def test_soa_activation_offset_is_the_block_sum(zero_point, paired):
    x = sample("normal", 256, 1.0, seed=51)
    got = cpp.quantize_a8_soa(x, zero_point, paired)
    # One integer a block, not one a lane. The kernel reduces its accumulators before scaling, so the
    # correction for the weights being stored `zero_point` too large can be subtracted from the
    # reduced vector -- which lets eight blocks share one 32-byte load. The sum is over a block's own
    # values, so the paired order has to be undone first or neighbouring blocks get mixed.
    laid_out = unpair_a8(got["qs"]) if paired else got["qs"].reshape(-1, 32)
    blocks = laid_out.astype(np.int32).sum(axis=1)
    assert got["offsets"].shape == (x.size // 32,)
    assert np.array_equal(got["offsets"], zero_point * blocks)


@pytest.mark.parametrize("n", [32, 288, 2560])
@pytest.mark.parametrize("tokens", [1, 3])
def test_soa_q8_kernel_matches_the_exact_reference(n, tokens):
    w = sample("normal", n, 1.0, seed=52)
    vectors = [sample("normal", n, 0.5, seed=53 + i) for i in range(tokens)]
    w_blob = pyq.quantize_q8(w)
    blobs = [pyq.quantize_a8(v) for v in vectors]

    wq = pyq.dequantize_q8(w_blob).astype(np.float64)
    exact = np.array([wq @ pyq.dequantize_a8(b).astype(np.float64) for b in blobs])
    tolerance = 1e-6 * float(np.abs(wq).sum()) * 0.5

    got = cpp.dot_q8_a8_soa(w_blob, b"".join(blobs))
    assert got.shape == (tokens,)
    assert np.abs(got - exact).max() <= tolerance


@pytest.mark.parametrize("weight_fmt", ["q4", "q8"])
def test_the_two_layouts_agree(weight_fmt):
    """The split and interleaved kernels are independent implementations of one computation."""
    n = 2560
    w = sample("normal", n, 1.0, seed=54)
    w_blob = _PY_QUANTIZE[weight_fmt](w)
    vectors = [sample("normal", n, 0.5, seed=55 + i) for i in range(4)]
    blobs = [pyq.quantize_a8(v) for v in vectors]
    joined = b"".join(blobs)

    split = getattr(cpp, f"dot_{weight_fmt}_a8_soa")(w_blob, joined)
    interleaved = getattr(cpp, f"dot_{weight_fmt}_a8_multi")(w_blob, joined)

    # Not bit-identical: the split kernel groups eight blocks before adding in float, a different
    # summation order -- and a shorter one, so if anything the more accurate of the two. The tolerance
    # has to scale with the terms being summed rather than with the answer, which for a dot product of
    # independent signs is far smaller than the terms and tells you nothing about the rounding.
    wq = _PY_DEQUANTIZE[weight_fmt](w_blob).astype(np.float64)
    terms = float(np.abs(wq).sum()) * 0.5
    assert np.abs(split - interleaved).max() <= 1e-6 * terms


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("chunks", [1, 2, 3, 6, 7, 80])
def test_splitting_the_activation_quantizer_changes_nothing(paired, chunks):
    """The engine spreads this over its workers, so a split must be byte-identical.

    Every block's scale comes from its own 32 values, so this holds by construction -- but the failure
    it guards against is nasty: the engine would disagree with the PyTorch twin only at whichever block
    boundary the thread count happened to land on, and only on the machines with that many cores.
    """
    x = sample("normal", 2560, 1.0, seed=60)
    whole = cpp.quantize_a8_soa(x, 8, paired)
    split = cpp.quantize_a8_soa_chunked(x, 8, paired, chunks)

    assert np.array_equal(split["qs"], whole["qs"])
    assert np.array_equal(split["scales"], whole["scales"])
    assert np.array_equal(split["offsets"], whole["offsets"])
