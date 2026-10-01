"""The scalar kernels, which are the reference the SIMD ones were written against.

On a CPU with AVX-VNNI the scalar path never runs, so a bug in it would be invisible on the machine
this project is developed on — and it is exactly the path a machine without AVX-VNNI would take, CI
included. Selecting it at runtime is how it stays honest.

The two paths are not expected to agree bit for bit: the SIMD kernel accumulates in eight lanes and
the scalar one in a single running sum. What must hold is that they agree to float rounding, and
that the whole engine reaches the same decisions either way.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from specdraft import _engine as cpp  # noqa: E402
from specdraft import quant as pyq  # noqa: E402
from specdraft.export import write_model  # noqa: E402
from specdraft.reference import TINY_CONFIG, perturbed_copy, random_reference  # noqa: E402


@pytest.fixture(autouse=True)
def restore_kernel_path():
    """Never leave the override set: it is process-wide."""
    yield
    cpp.set_force_scalar(False)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("scalar")
    target = random_reference(TINY_CONFIG, seed=0)
    draft = perturbed_copy(target, sigma=0.02, seed=1)
    out = {}
    for name, model in (("target", target), ("draft", draft)):
        for fmt in ("q4", "q8"):
            path = directory / f"{name}-{fmt}.sdm"
            write_model(path, model.state_dict(), model.config, weight_format=fmt)
            out[(name, fmt)] = str(path)
    return out


def tokens(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, TINY_CONFIG.vocab_size, size=n, dtype=np.int32)


def test_the_override_switches_the_reported_path():
    has_vnni = cpp.cpu_features()["avx_vnni"]
    assert cpp.kernel_path() == ("vnni" if has_vnni else "scalar")
    cpp.set_force_scalar(True)
    assert cpp.force_scalar() and cpp.kernel_path() == "scalar"
    cpp.set_force_scalar(False)
    assert not cpp.force_scalar()


@pytest.mark.parametrize("weight_fmt", ["q4", "q8"])
@pytest.mark.parametrize("n", [32, 1024, 4096])
def test_both_paths_agree_on_a_dot_product(weight_fmt, n):
    w = pyq.quantize_q4(np.random.default_rng(1).standard_normal(n).astype(np.float32)) \
        if weight_fmt == "q4" else \
        pyq.quantize_q8(np.random.default_rng(1).standard_normal(n).astype(np.float32))
    x = pyq.quantize_a8((np.random.default_rng(2).standard_normal(n) * 0.5).astype(np.float32))

    dispatched = getattr(cpp, f"dot_{weight_fmt}_a8")
    cpp.set_force_scalar(False)
    fast = dispatched(w, x)
    cpp.set_force_scalar(True)
    reference = dispatched(w, x)

    # Both multiply the same integers; only the order of the float accumulation differs.
    assert abs(fast - reference) <= 1e-5 * max(1.0, abs(reference))
    assert reference == getattr(cpp, f"dot_{weight_fmt}_a8_scalar")(w, x)


@pytest.mark.parametrize("weight_fmt", ["q4", "q8"])
@pytest.mark.parametrize("count", [1, 3, 4, 7])
def test_the_multi_token_kernel_has_a_scalar_path_too(weight_fmt, count):
    quantize = pyq.quantize_q4 if weight_fmt == "q4" else pyq.quantize_q8
    w = quantize(np.random.default_rng(3).standard_normal(256).astype(np.float32))
    blobs = [
        pyq.quantize_a8((np.random.default_rng(10 + i).standard_normal(256) * 0.5).astype(np.float32))
        for i in range(count)
    ]
    joined = b"".join(blobs)
    multi = getattr(cpp, f"dot_{weight_fmt}_a8_multi")
    single = getattr(cpp, f"dot_{weight_fmt}_a8")

    cpp.set_force_scalar(True)
    together = multi(w, joined)
    separately = np.array([single(w, blob) for blob in blobs], dtype=np.float32)
    # Scalar or SIMD, sharing one pass over the weights must change nothing at all.
    assert np.array_equal(together, separately)


@pytest.mark.parametrize("fmt", ["q4", "q8"])
def test_the_engine_decides_the_same_way_on_either_path(files, fmt):
    ids = tokens(9, seed=4)

    cpp.set_force_scalar(False)
    fast = cpp.Model(files[("target", fmt)], max_positions=32).forward(ids, all_logits=True)
    cpp.set_force_scalar(True)
    scalar = cpp.Model(files[("target", fmt)], max_positions=32).forward(ids, all_logits=True)

    assert np.abs(fast - scalar).max() < 1e-3
    assert np.array_equal(fast.argmax(-1), scalar.argmax(-1)), "the two paths chose different tokens"


@pytest.mark.parametrize("fmt", ["q4", "q8"])
def test_speculative_decoding_works_on_the_scalar_path(files, fmt):
    """What a machine without AVX-VNNI would run, including CI."""
    cpp.set_force_scalar(True)
    target = cpp.Model(files[("target", fmt)], max_positions=96, max_batch=5)
    draft = cpp.Model(files[("draft", fmt)], max_positions=96, max_batch=5)
    prompt = tokens(5, seed=5).tolist()

    expected, _ = cpp.generate_plain(target, prompt, max_new_tokens=20)
    got, stats = cpp.generate_speculative(target, draft, prompt, max_new_tokens=20, gamma=4)

    assert got == expected
    assert stats["tokens_per_target_forward"] > 1.0
