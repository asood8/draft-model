"""The torch, NumPy and C++ quantizers must agree bit for bit.

Three implementations exist for good reasons (the engine runs C++, the twin runs torch on a
GPU, the export script uses NumPy), and the whole evaluation strategy rests on them being
the same arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from specdraft import _engine as cpp  # noqa: E402
from specdraft import quant as pyq  # noqa: E402
from specdraft import quant_torch as tq  # noqa: E402

FORMATS = ["q4", "q8", "a8"]
_CPP_QUANTIZE = {"q4": cpp.quantize_q4, "q8": cpp.quantize_q8, "a8": cpp.quantize_a8}
_CPP_DEQUANTIZE = {"q4": cpp.dequantize_q4, "q8": cpp.dequantize_q8, "a8": cpp.dequantize_a8}


def sample(n: int, magnitude: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x[::11] *= 30.0  # outliers stress the shared block scale
    return (x * magnitude).astype(np.float32)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("magnitude", [1e-6, 1e-3, 1.0, 100.0])
@pytest.mark.parametrize("n", [32, 1024, 4096])
def test_torch_matches_numpy_exactly(fmt, magnitude, n):
    x = sample(n, magnitude, seed=n)
    from_numpy = pyq.fake_quantize(x, fmt)
    from_torch = tq.fake_quantize(torch.from_numpy(x), fmt).numpy()
    assert np.array_equal(from_torch, from_numpy)


@pytest.mark.parametrize("fmt", FORMATS)
def test_torch_matches_the_engine_exactly(fmt):
    x = sample(2048, 1.0, seed=7)
    from_engine = _CPP_DEQUANTIZE[fmt](_CPP_QUANTIZE[fmt](x))
    from_torch = tq.fake_quantize(torch.from_numpy(x), fmt).numpy()
    assert np.array_equal(from_torch, from_engine)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize(
    "special",
    [
        np.zeros(64, dtype=np.float32),
        np.full(32, -2.5, dtype=np.float32),
        np.array([1e-8] * 31 + [1e-7], dtype=np.float32),  # the scale underflows fp16
        np.array([65504.0] + [0.0] * 31, dtype=np.float32),  # the largest fp16 scale
    ],
)
def test_edge_cases_match(fmt, special):
    from_torch = tq.fake_quantize(torch.from_numpy(special), fmt).numpy()
    assert np.array_equal(from_torch, pyq.fake_quantize(special, fmt))


@pytest.mark.parametrize("fmt", FORMATS)
def test_quantizing_twice_changes_nothing(fmt):
    """Values already on the grid must survive, or the twin would drift every layer."""
    once = tq.fake_quantize(torch.from_numpy(sample(1024, 1.0, seed=8)), fmt)
    assert torch.equal(tq.fake_quantize(once, fmt), once)


@pytest.mark.parametrize("fmt", FORMATS)
def test_blocks_run_along_the_last_axis(fmt):
    x = torch.from_numpy(sample(6 * 64, 1.0, seed=9)).reshape(6, 64)
    out = tq.fake_quantize(x, fmt)
    assert out.shape == x.shape
    assert torch.equal(out[3], tq.fake_quantize(x[3], fmt))  # rows are independent


def test_rejects_bad_shapes_and_names():
    with pytest.raises(ValueError):
        tq.fake_quantize(torch.zeros(33), "q4")
    with pytest.raises(ValueError):
        tq.fake_quantize(torch.zeros(32), "q3")


@pytest.mark.parametrize("fmt,max_relative_rms", [("q4", 0.12), ("q8", 0.01), ("a8", 0.01)])
def test_error_is_bounded(fmt, max_relative_rms):
    x = torch.randn(32768, generator=torch.Generator().manual_seed(10))
    error = tq.fake_quantize(x, fmt) - x
    assert (error.pow(2).mean().sqrt() / x.std()).item() < max_relative_rms
