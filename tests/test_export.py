"""The exported file must hold exactly what the twin computes with.

If these two ever disagree, the engine and the PyTorch side are running different models and
every later comparison is meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from specdraft.export import ALIGNMENT, MAGIC, read_model, write_model  # noqa: E402
from specdraft.reference import TINY_CONFIG, random_reference  # noqa: E402
from specdraft.twin import QuantizedTwin  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return random_reference(TINY_CONFIG, seed=0)


@pytest.fixture
def written(tmp_path, model):
    path = tmp_path / "tiny-q4.sdm"
    write_model(path, model.state_dict(), model.config, weight_format="q4", vocab_limit=60)
    return read_model(path)


def test_header_and_directory(written, model):
    assert written.path.read_bytes()[:4] == MAGIC
    assert written.metadata["vocab_limit"] == 60
    assert written.metadata["weight_format"] == "q4"
    assert written.config == model.config

    expected = {"token_embd", "output_norm"} | {
        f"blk.{i}.{part}"
        for i in range(TINY_CONFIG.num_hidden_layers)
        for part in ("attn_norm", "qkv", "q_norm", "k_norm", "attn_out", "ffn_norm", "gate_up", "ffn_down")
    }
    assert set(written.entries) == expected


def test_every_tensor_is_aligned_and_inside_the_file(written):
    size = written.path.stat().st_size
    for entry in written.entries.values():
        assert entry.offset % ALIGNMENT == 0, f"{entry.name} is misaligned"
        assert entry.offset + entry.nbytes <= size


def test_shapes_are_fused_as_the_engine_expects(written):
    cfg = TINY_CONFIG
    assert written.entries["blk.0.qkv"].shape == (cfg.q_dim + 2 * cfg.kv_dim, cfg.hidden_size)
    assert written.entries["blk.0.gate_up"].shape == (2 * cfg.intermediate_size, cfg.hidden_size)
    assert written.entries["blk.0.attn_out"].shape == (cfg.hidden_size, cfg.q_dim)
    assert written.entries["token_embd"].shape == (cfg.vocab_size, cfg.hidden_size)
    # Tied embeddings: one matrix serves the lookup and the output layer.
    assert "output" not in written.entries


def test_norms_stay_fp32_and_matrices_are_quantized(written):
    assert written.entries["blk.0.attn_norm"].format == "fp32"
    assert written.entries["blk.0.q_norm"].format == "fp32"
    assert written.entries["output_norm"].format == "fp32"
    assert written.entries["blk.0.qkv"].format == "q4"
    assert written.entries["token_embd"].format == "q4"


def test_values_match_the_twin_exactly(written, model):
    """The engine's weights and the twin's weights must be the same numbers."""
    twin = QuantizedTwin.from_reference(model, weight_format="q4")
    layer, twin_layer = written, twin.layers[0]

    assert np.array_equal(layer.tensor("token_embd"), twin.embed_tokens.numpy())
    assert np.array_equal(layer.tensor("blk.0.attn_out"), twin_layer.o_proj.numpy())
    assert np.array_equal(layer.tensor("blk.0.ffn_down"), twin_layer.down_proj.numpy())
    assert np.array_equal(layer.tensor("output_norm"), model.final_norm.numpy())

    fused_qkv = np.concatenate(
        [twin_layer.q_proj.numpy(), twin_layer.k_proj.numpy(), twin_layer.v_proj.numpy()]
    )
    assert np.array_equal(layer.tensor("blk.0.qkv"), fused_qkv)
    fused_gate_up = np.concatenate([twin_layer.gate_proj.numpy(), twin_layer.up_proj.numpy()])
    assert np.array_equal(layer.tensor("blk.0.gate_up"), fused_gate_up)


def test_mixed_precision_output_layer(tmp_path, model):
    path = tmp_path / "tiny-mixed.sdm"
    write_model(path, model.state_dict(), model.config, weight_format="q4", output_format="q8")
    written = read_model(path)
    assert written.entries["token_embd"].format == "q8"
    assert written.entries["blk.0.qkv"].format == "q4"

    twin = QuantizedTwin.from_reference(model, weight_format="q4", output_format="q8")
    assert np.array_equal(written.tensor("token_embd"), twin.embed_tokens.numpy())


def test_fp32_export_is_lossless(tmp_path, model):
    """The format the engine is brought up on, before any quantized kernel exists."""
    path = tmp_path / "tiny-fp32.sdm"
    write_model(path, model.state_dict(), model.config, weight_format="fp32")
    written = read_model(path)
    assert np.array_equal(written.tensor("blk.1.attn_out"), model.layers[1].o_proj.numpy())


def test_untied_models_get_a_separate_output_matrix(tmp_path):
    import dataclasses

    config = dataclasses.replace(TINY_CONFIG, tie_word_embeddings=False)
    model = random_reference(config, seed=1)
    path = tmp_path / "untied.sdm"
    write_model(path, model.state_dict(), config, weight_format="q8")
    written = read_model(path)
    assert written.entries["output"].shape == (config.vocab_size, config.hidden_size)


def test_bad_inputs_are_rejected(tmp_path, model):
    with pytest.raises(ValueError):
        write_model(tmp_path / "x.sdm", model.state_dict(), model.config, weight_format="q3")
    (tmp_path / "junk.sdm").write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(ValueError):
        read_model(tmp_path / "junk.sdm")
