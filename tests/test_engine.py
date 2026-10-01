"""The C++ engine must compute the same thing as the PyTorch side (plan §8).

Three comparisons, in order of strictness:

* an fp32 weights file against the plain reference, which isolates the arithmetic
  (RMSNorm, the per-head q/k norms, RoPE, grouped-query attention, SwiGLU) from anything
  to do with quantization;
* a quantized file against the twin at the same formats, which is what the engine will
  actually run;
* a k-token pass against k single-token passes, which has to be bit-exact, because that
  is what makes greedy speculative decoding bit-exact later.

All of this runs on a tiny random model, so it needs no weights on disk.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from specdraft import _engine as cpp  # noqa: E402
from specdraft.export import write_model  # noqa: E402
from specdraft.reference import TINY_CONFIG, random_reference  # noqa: E402
from specdraft.twin import QuantizedTwin  # noqa: E402

MAX_POSITIONS = 64


@pytest.fixture(scope="module")
def reference():
    return random_reference(TINY_CONFIG, seed=0)


@pytest.fixture(scope="module")
def files(tmp_path_factory, reference):
    """One weights file per format."""
    directory = tmp_path_factory.mktemp("engine")
    paths = {}
    for fmt in ("fp32", "q4", "q8"):
        path = directory / f"tiny-{fmt}.sdm"
        write_model(path, reference.state_dict(), reference.config, weight_format=fmt)
        paths[fmt] = str(path)
    return paths


def engine_for(files, fmt: str) -> "cpp.Model":
    return cpp.Model(files[fmt], max_positions=MAX_POSITIONS)


def twin_for(reference, fmt: str) -> QuantizedTwin:
    """The twin configured exactly as the engine runs.

    The engine's "fp32" refers to its *weights and activations*: its KV cache is always
    fp16, so the twin keeps an fp16 cache in every configuration.
    """
    return QuantizedTwin.from_reference(
        reference,
        weight_format=None if fmt == "fp32" else fmt,
        activation_format=None if fmt == "fp32" else "a8",
        kv_dtype=torch.float16,
    )


def tokens(n: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(0, TINY_CONFIG.vocab_size, size=n, dtype=np.int32)


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def torch_logits(model, ids: np.ndarray, vocab_limit: int) -> np.ndarray:
    """What the PyTorch side gets, using the same fp16 cache the engine keeps."""
    with torch.no_grad():
        out = model.forward(
            torch.from_numpy(ids).long(), cache=model.new_cache(MAX_POSITIONS), only_last_logits=False
        )
    return out[:, :vocab_limit].numpy()


# ------------------------------------------------------------------- the file and config


def test_config_survives_the_round_trip(files):
    config = cpp.Model(files["q4"], max_positions=8).config
    assert config["hidden_size"] == TINY_CONFIG.hidden_size
    assert config["head_dim"] == TINY_CONFIG.head_dim
    assert config["num_key_value_heads"] == TINY_CONFIG.num_key_value_heads
    assert config["rms_norm_eps"] == pytest.approx(TINY_CONFIG.rms_norm_eps)
    assert config["rope_theta"] == pytest.approx(TINY_CONFIG.rope_theta)
    assert config["tie_word_embeddings"] is True


def test_file_info_lists_the_directory(files):
    info = cpp.model_file_info(files["q8"])
    assert info["tensors"]["blk.0.qkv"][0] == "q8"
    assert info["tensors"]["blk.0.attn_norm"][0] == "fp32"
    assert info["size_bytes"] > 0


def test_missing_file_raises():
    with pytest.raises(Exception):
        cpp.Model("no/such/file.sdm")


# --------------------------------------------------------------- fp32: pure arithmetic


def test_fp32_engine_matches_the_reference(files, reference):
    """No quantization on either side: only float rounding may differ."""
    ids = tokens(12, seed=1)
    mine = engine_for(files, "fp32").forward(ids, all_logits=True)
    theirs = torch_logits(twin_for(reference, "fp32"), ids, mine.shape[1])
    assert relative_error(mine, theirs) < 1e-5


def test_fp32_engine_matches_layer_by_layer(files, reference):
    ids = tokens(10, seed=2)
    mine = engine_for(files, "fp32").forward_capture(ids)

    captured: list[torch.Tensor] = []
    model = twin_for(reference, "fp32")
    with torch.no_grad():
        model.forward(torch.from_numpy(ids).long(), cache=model.new_cache(MAX_POSITIONS), capture=captured)

    for layer, theirs in enumerate(captured):
        error = relative_error(mine[layer], theirs.numpy())
        assert error < 1e-5, f"layer {layer} differs by {error:.2e}"


# ------------------------------------------------------- quantized: what the engine runs


@pytest.mark.parametrize("fmt", ["q8", "q4"])
def test_quantized_engine_matches_the_twin(files, reference, fmt):
    ids = tokens(12, seed=3)
    mine = engine_for(files, fmt).forward(ids, all_logits=True)
    theirs = torch_logits(twin_for(reference, fmt), ids, mine.shape[1])
    error = relative_error(mine, theirs)
    assert error < 1e-4, f"{fmt}: relative error {error:.2e}"


@pytest.mark.parametrize("fmt", ["q8", "q4"])
def test_quantized_engine_matches_layer_by_layer(files, reference, fmt):
    ids = tokens(10, seed=4)
    mine = engine_for(files, fmt).forward_capture(ids)

    captured: list[torch.Tensor] = []
    model = twin_for(reference, fmt)
    with torch.no_grad():
        model.forward(torch.from_numpy(ids).long(), cache=model.new_cache(MAX_POSITIONS), capture=captured)

    for layer, theirs in enumerate(captured):
        error = relative_error(mine[layer], theirs.numpy())
        assert error < 1e-4, f"{fmt} layer {layer} differs by {error:.2e}"


# ------------------------------------------------------------------ engine-side invariants


@pytest.mark.parametrize("fmt", ["fp32", "q4"])
def test_k_tokens_equal_k_single_tokens_bit_for_bit(files, fmt):
    """The property bit-exact greedy speculative decoding rests on."""
    ids = tokens(9, seed=5)
    together = engine_for(files, fmt).forward(ids, all_logits=True)

    one_at_a_time = engine_for(files, fmt)
    rows = [one_at_a_time.forward(ids[i : i + 1], all_logits=True)[0] for i in range(len(ids))]
    assert np.array_equal(together, np.stack(rows))


def test_only_last_logits_matches_the_last_row(files):
    ids = tokens(6, seed=6)
    everything = engine_for(files, "q4").forward(ids, all_logits=True)
    last = engine_for(files, "q4").forward(ids, all_logits=False)
    assert last.shape == (1, everything.shape[1])
    assert np.array_equal(last[0], everything[-1])


def test_position_counter_advances_and_rolls_back(files):
    engine = engine_for(files, "q4")
    assert engine.pos == 0
    engine.forward(tokens(5, seed=7))
    assert engine.pos == 5
    engine.set_pos(2)
    assert engine.pos == 2
    engine.reset()
    assert engine.pos == 0


def test_rolling_back_discards_rejected_tokens(files):
    """Rewinding has to leave no trace, the way a rejected draft block must not."""
    accepted, rejected, replacement = tokens(6, seed=8), tokens(4, seed=9), tokens(4, seed=10)

    engine = engine_for(files, "q4")
    engine.forward(accepted)
    engine.forward(rejected)
    engine.set_pos(len(accepted))
    after_rewind = engine.forward(replacement, all_logits=True)

    clean = engine_for(files, "q4")
    clean.forward(accepted)
    expected = clean.forward(replacement, all_logits=True)

    assert np.array_equal(after_rewind, expected)


def test_logits_stop_at_the_vocabulary_limit(tmp_path, reference):
    """The padded embedding rows must never be computed or sampled."""
    limit = TINY_CONFIG.vocab_size - 5
    path = tmp_path / "limited.sdm"
    write_model(path, reference.state_dict(), reference.config, weight_format="q8", vocab_limit=limit)
    engine = cpp.Model(str(path), max_positions=8)
    assert engine.config["vocab_limit"] == limit
    assert engine.forward(tokens(3, seed=11), all_logits=True).shape == (3, limit)


def test_out_of_range_inputs_raise(files):
    engine = engine_for(files, "q4")
    with pytest.raises(Exception):
        engine.forward(np.array([TINY_CONFIG.vocab_size + 1], dtype=np.int32))
    with pytest.raises(Exception):
        engine.forward(np.array([], dtype=np.int32))
    with pytest.raises(Exception):
        engine.forward(tokens(MAX_POSITIONS + 1, seed=12))
    with pytest.raises(Exception):
        engine.set_pos(MAX_POSITIONS + 1)
