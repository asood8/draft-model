"""The training entry point, end to end (plan §11.5).

Runs ``scripts/train_draft.py`` against a tiny model that carries the *real* tokenizer, so the
templating, the response boundary, the loop, checkpointing and the saved output are all exercised
on the same code path a Kaggle run would take. Skipped when the tokenizer has not been fetched.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from specdraft.data import PromptRecord, write_records  # noqa: E402
from specdraft.reference import Qwen3Config, Qwen3Reference, random_reference  # noqa: E402

MODEL_DIR = Path(os.environ.get("SPECDRAFT_DRAFT_MODEL", "models/Qwen3-0.6B"))

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer.json").is_file(), reason=f"no tokenizer at {MODEL_DIR}"
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# Small everywhere except the vocabulary, which has to match the real tokenizer.
TINY_VOCAB = 151_680  # a multiple of 32, so the quantized formats accept it


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """A miniature Qwen3 in Hugging Face layout, with the project's real tokenizer."""
    from safetensors.torch import save_file

    directory = tmp_path_factory.mktemp("tiny_qwen3")
    config = Qwen3Config(
        vocab_size=TINY_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        tie_word_embeddings=True,
    )
    model = random_reference(config, seed=0)
    save_file(
        {name: tensor.contiguous() for name, tensor in model.state_dict().items()},
        str(directory / "model.safetensors"),
    )
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                **config.__dict__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "chat_template.jinja"):
        source = MODEL_DIR / name
        if source.is_file():
            shutil.copy2(source, directory / name)
    return directory


@pytest.fixture(scope="module")
def data_file(tmp_path_factory):
    directory = tmp_path_factory.mktemp("train_data")
    path = directory / "records.jsonl"
    records = [
        PromptRecord(f"Question number {i} about arithmetic: what is {i} plus {i}?",
                     f"The answer is {2 * i}. " * 6, "test")
        for i in range(12)
    ]
    write_records(path, records)
    return path


def run_train_script(arguments: list[str]) -> None:
    sys.path.insert(0, str(SCRIPTS))
    try:
        import train_draft

        original = sys.argv
        sys.argv = ["train_draft.py", *arguments]
        try:
            train_draft.main()
        finally:
            sys.argv = original
    finally:
        sys.path.remove(str(SCRIPTS))


def test_training_runs_and_saves_a_loadable_draft(tiny_model_dir, data_file, tmp_path):
    out = tmp_path / "run"
    run_train_script(
        [
            "--student", str(tiny_model_dir),
            "--teacher", str(tiny_model_dir),
            "--data", str(data_file),
            "--out", str(out),
            "--loss", "tvd",
            "--max-tokens", "600",
            "--tokens-per-step", "200",
            "--learning-rate", "1e-3",
            "--validation", "2",
            "--loss-chunk", "64",
        ]
    )

    # The checkpoint a resumed session would pick up.
    assert (out / "checkpoint.pt").is_file()
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    assert state["step"] >= 1
    assert state["response_tokens"] >= 600
    assert state["history"]

    # The saved draft: loadable by the project, and by Hugging Face.
    draft = out / "draft"
    assert (draft / "model.safetensors").is_file()
    assert (draft / "config.json").is_file()
    assert (draft / "tokenizer.json").is_file(), "the tokenizer must travel with the weights"

    reloaded = Qwen3Reference.from_pretrained(draft)
    assert reloaded.config.num_hidden_layers == 2
    logits = reloaded.forward(torch.tensor([5, 9, 11]))
    assert logits.shape == (3, TINY_VOCAB)
    assert torch.isfinite(logits).all()

    training = json.loads((draft / "training.json").read_text(encoding="utf-8"))
    assert training["loss"] == "tvd"
    assert "before" in training and "after" in training


def test_the_draft_actually_changed(tiny_model_dir, data_file, tmp_path):
    """A run that reports progress but leaves the weights alone would be worse than useless."""
    out = tmp_path / "run2"
    run_train_script(
        [
            "--student", str(tiny_model_dir),
            "--teacher", str(tiny_model_dir),
            "--data", str(data_file),
            "--out", str(out),
            "--loss", "fkl",
            "--max-tokens", "400",
            "--tokens-per-step", "200",
            "--learning-rate", "5e-3",
            "--validation", "2",
        ]
    )
    original = Qwen3Reference.from_pretrained(tiny_model_dir)
    trained = Qwen3Reference.from_pretrained(out / "draft")
    assert not torch.equal(original.layers[0].q_proj, trained.layers[0].q_proj)
    # The teacher here is the student's own starting point, so the loss had almost nothing to
    # close; what matters is that the pipeline wrote back a genuinely updated model.
    assert torch.isfinite(trained.layers[0].q_proj).all()


def test_exported_draft_feeds_the_engine(tiny_model_dir, data_file, tmp_path):
    """The point of the whole exercise: a trained draft the C++ engine can run."""
    from specdraft import _engine as cpp
    from specdraft.export import write_model

    out = tmp_path / "run3"
    run_train_script(
        [
            "--student", str(tiny_model_dir),
            "--teacher", str(tiny_model_dir),
            "--data", str(data_file),
            "--out", str(out),
            "--loss", "rkl",
            "--max-tokens", "200",
            "--tokens-per-step", "200",
            "--validation", "2",
        ]
    )
    trained = Qwen3Reference.from_pretrained(out / "draft")
    weights_file = tmp_path / "trained-q4.sdm"
    write_model(weights_file, trained.state_dict(), trained.config, weight_format="q4",
                vocab_limit=151_669)

    engine = cpp.Model(str(weights_file), max_positions=32, max_batch=4)
    logits = engine.forward([3, 5, 7], all_logits=True)
    assert logits.shape == (3, 151_669)


def test_quantization_aware_run_saves_a_loadable_draft(tiny_model_dir, data_file, tmp_path):
    """The --quantization-aware path trains through the rounding the engine will apply."""
    out = tmp_path / "run_qat"
    run_train_script(
        [
            "--student", str(tiny_model_dir),
            "--teacher", str(tiny_model_dir),
            "--data", str(data_file),
            "--out", str(out),
            "--loss", "tvd",
            "--max-tokens", "300",
            "--tokens-per-step", "150",
            "--learning-rate", "1e-3",
            "--validation", "2",
            "--quantization-aware",
            "--student-format", "q4",
        ]
    )
    training = json.loads((out / "draft" / "training.json").read_text(encoding="utf-8"))
    assert training["quantization_aware"] is True
    assert training["student_format"] == "q4"

    # Saved at full precision, so exporting lands it on the same grid it was trained against.
    reloaded = Qwen3Reference.from_pretrained(out / "draft")
    logits = reloaded.forward(torch.tensor([2, 4, 6]))
    assert torch.isfinite(logits).all()
