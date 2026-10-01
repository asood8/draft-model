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
