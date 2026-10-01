"""The distillation loop (plan §11.5).

The draft learns to agree with the target on the positions it will have to predict during
decoding. The teacher is a callable rather than a model, so the same loop serves every variant
the plan compares: the full-precision target, the 4-bit twin that matches the engine, or logits
computed earlier and read back from disk. On two GPUs the caller runs the teacher on one device
and the student on the other; the loop here does not care.

Memory is the constraint that shapes everything. One sequence's logits over a 151,669-token
vocabulary run to hundreds of megabytes, so the loss applies the output layer a slice at a time
and rebuilds it during the backward pass, and sequences are trained one at a time with
gradients accumulated to a sensible optimizer step.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor

from .losses import LOSSES, chunked_distill_loss, per_position_loss
from .trainable import TrainableDraft

# Given a token sequence, return the teacher's logits for it: [len(tokens), vocab].
TeacherFn = Callable[[Tensor], Tensor]


@dataclass
class TrainedSequence:
    """One training example: the tokens, and where the response starts.

    The loss applies at the positions whose *next* token belongs to the response, which is rows
    ``response_start - 1`` through ``len(tokens) - 2``, the same convention the offline metrics
    use.
    """

    tokens: list[int]
    response_start: int
    source: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.response_start < len(self.tokens):
            raise ValueError("response_start must be inside the sequence")

    @property
    def response_tokens(self) -> int:
        return len(self.tokens) - self.response_start


@dataclass
class TrainConfig:
    loss: str = "fkl"
    temperature: float = 1.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_fraction: float = 0.02
    grad_clip: float = 1.0
    # Tokens of response to train on before stopping, and how many to accumulate per step.
    max_response_tokens: int = 1_000_000
    tokens_per_step: int = 65_536
    loss_chunk: int = 256
    vocab_limit: int | None = None
    device: str = "cpu"
    amp: bool = False  # fp16 autocast with a gradient scaler, for T4s
    eight_bit_adam: bool = False
    seed: int = 0
    log_every_steps: int = 20
    checkpoint_every_seconds: float = 1800.0
    max_sequence_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.loss not in LOSSES:
            raise ValueError(f"unknown loss {self.loss!r}; expected one of {LOSSES}")
        if self.tokens_per_step <= 0 or self.max_response_tokens <= 0:
            raise ValueError("token budgets must be positive")


@dataclass
class TrainState:
    step: int = 0
    response_tokens: int = 0
    sequences: int = 0
    history: list[dict] = field(default_factory=list)
    skipped_steps: int = 0  # non-finite losses, which fp16 can produce


def build_optimizer(model: TrainableDraft, config: TrainConfig):
    parameters = model.trainable_parameters()
    if config.eight_bit_adam:
        # bitsandbytes keeps the optimizer state in 8 bits, which is what makes a 0.6B fit
        # alongside its activations on a 16 GB card.
        import bitsandbytes

        return bitsandbytes.optim.AdamW8bit(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    return torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )


def learning_rate_at(step: int, total_steps: int, config: TrainConfig) -> float:
    """Linear warmup, then cosine decay."""
    warmup = max(1, int(total_steps * config.warmup_fraction))
    if step < warmup:
        return config.learning_rate * (step + 1) / warmup
    if total_steps <= warmup:
        return config.learning_rate
    progress = (step - warmup) / max(1, total_steps - warmup)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def validate(
    student: TrainableDraft,
    teacher: TeacherFn,
    sequences: Sequence[TrainedSequence],
    config: TrainConfig,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Offline acceptance on held-out text: the cheap number that tracks the real one.

    Returns the greedy top-1 agreement and the mean 1 − TVD, which is the expected acceptance
    rate when sampling (plan §11.6). One forward pass per model, no decoding.
    """
    matches = 0
    positions = 0
    acceptance = 0.0
    for sequence in sequences:
        tokens = torch.tensor(sequence.tokens, dtype=torch.long, device=config.device)
        rows = slice(sequence.response_start - 1, len(sequence.tokens) - 1)
        teacher_logits = teacher(tokens)[rows]
        student_logits = student.logits(tokens)[rows]
        if config.vocab_limit is not None:
            teacher_logits = teacher_logits[..., : config.vocab_limit]
            student_logits = student_logits[..., : config.vocab_limit]

        matches += int((teacher_logits.argmax(-1) == student_logits.argmax(-1)).sum())
        p = torch.softmax(teacher_logits.float() / temperature, dim=-1)
        q = torch.softmax(student_logits.float() / temperature, dim=-1)
        acceptance += float(torch.minimum(p, q).sum(-1).sum())
        positions += teacher_logits.shape[0]

    if positions == 0:
        return {"positions": 0}
    return {
        "positions": positions,
        "greedy_top1_match": matches / positions,
        "sampling_acceptance": acceptance / positions,
    }


def save_checkpoint(path: Path, student: TrainableDraft, optimizer, state: TrainState,
                    config: TrainConfig, scaler=None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weights": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "step": state.step,
            "response_tokens": state.response_tokens,
            "sequences": state.sequences,
            "skipped_steps": state.skipped_steps,
        },
        path / "checkpoint.pt",
    )
    (path / "state.json").write_text(
        json.dumps(
            {
                "step": state.step,
                "response_tokens": state.response_tokens,
                "sequences": state.sequences,
                "skipped_steps": state.skipped_steps,
                "config": config.__dict__,
                "history": state.history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_checkpoint(path: Path, student: TrainableDraft, optimizer=None, scaler=None) -> TrainState:
    blob = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    student.load_state_dict(blob["weights"])
    if optimizer is not None and blob.get("optimizer") is not None:
        optimizer.load_state_dict(blob["optimizer"])
    if scaler is not None and blob.get("scaler") is not None:
        scaler.load_state_dict(blob["scaler"])
    state = TrainState(
        step=blob["step"],
        response_tokens=blob["response_tokens"],
        sequences=blob["sequences"],
        skipped_steps=blob.get("skipped_steps", 0),
    )
    state_file = path / "state.json"
    if state_file.is_file():
        state.history = json.loads(state_file.read_text(encoding="utf-8")).get("history", [])
    return state


def train(
    student: TrainableDraft,
    teacher: TeacherFn,
    sequences: Iterable[TrainedSequence],
    config: TrainConfig,
    validation: Sequence[TrainedSequence] | None = None,
    checkpoint_dir: Path | None = None,
    optimizer=None,
    state: TrainState | None = None,
    on_log: Callable[[dict], None] | None = None,
) -> TrainState:
    """Run until the response-token budget is spent, or the sequences run out."""
    torch.manual_seed(config.seed)
    optimizer = optimizer or build_optimizer(student, config)
    state = state or TrainState()
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)

    total_steps = max(1, config.max_response_tokens // config.tokens_per_step)
    tokens_in_step = 0
    loss_in_step = 0.0
    last_checkpoint = time.perf_counter()
    student.train()

    for sequence in sequences:
        if state.response_tokens >= config.max_response_tokens:
            break
        if len(sequence.tokens) > config.max_sequence_tokens:
            sequence = TrainedSequence(
                sequence.tokens[: config.max_sequence_tokens], sequence.response_start,
                sequence.source,
            )
            if sequence.response_start >= len(sequence.tokens):
                continue

        tokens = torch.tensor(sequence.tokens, dtype=torch.long, device=config.device)
        rows = slice(sequence.response_start - 1, len(sequence.tokens) - 1)
        labels = tokens[sequence.response_start :]

        with torch.no_grad():
            teacher_logits = teacher(tokens)[rows]
        if teacher_logits.device != tokens.device:
            teacher_logits = teacher_logits.to(tokens.device)

        with torch.autocast("cuda", dtype=torch.float16, enabled=config.amp):
            hidden = student.hidden_states(tokens)[rows]
        loss = chunked_distill_loss(
            hidden.float(),
            student.output_weight,
            teacher_logits,
            kind=config.loss,
            labels=labels,
            temperature=config.temperature,
            vocab_limit=config.vocab_limit,
            chunk=config.loss_chunk,
        )

        # Each sequence contributes in proportion to its length, so the optimizer step is an
        # average over tokens rather than over sequences.
        weight = sequence.response_tokens / config.tokens_per_step
        if not torch.isfinite(loss):
            # fp16 can overflow on models trained in bfloat16; drop the sequence, keep going.
            state.skipped_steps += 1
            optimizer.zero_grad(set_to_none=True)
            tokens_in_step = 0
            loss_in_step = 0.0
            continue

        scaler.scale(loss * weight).backward()
        loss_in_step += float(loss.detach()) * sequence.response_tokens
        tokens_in_step += sequence.response_tokens
        state.response_tokens += sequence.response_tokens
        state.sequences += 1

        if tokens_in_step >= config.tokens_per_step:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(state.step, total_steps, config)
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.trainable_parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            state.step += 1

            entry = {
                "step": state.step,
                "loss": loss_in_step / max(1, tokens_in_step),
                "response_tokens": state.response_tokens,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if validation and state.step % max(1, config.log_every_steps) == 0:
                entry.update(validate(student, teacher, validation, config))
                student.train()
            state.history.append(entry)
            if on_log is not None:
                on_log(entry)

            tokens_in_step = 0
            loss_in_step = 0.0

            if checkpoint_dir is not None and (
                time.perf_counter() - last_checkpoint > config.checkpoint_every_seconds
            ):
                save_checkpoint(Path(checkpoint_dir), student, optimizer, state, config, scaler)
                last_checkpoint = time.perf_counter()

    if checkpoint_dir is not None:
        save_checkpoint(Path(checkpoint_dir), student, optimizer, state, config, scaler)
    return state


def teacher_from_model(model, device: str | None = None) -> TeacherFn:
    """Wrap a reference model or quantization twin as a teacher."""

    @torch.no_grad()
    def teacher(tokens: Tensor) -> Tensor:
        logits = model.forward(tokens.to(model.device))
        return logits if device is None else logits.to(device)

    return teacher


def save_draft(
    path: str | Path,
    student: TrainableDraft,
    source_model_dir: str | Path | None = None,
    extra: dict | None = None,
) -> Path:
    """Write the trained draft where both the Hub and the export script can read it.

    The architecture is unchanged, so the source model's config and tokenizer are copied across
    and only the fields that training can alter are overridden. That keeps the result loadable by
    ``AutoModelForCausalLM`` and by ``scripts/export_model.py`` without special cases.
    """
    import shutil

    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    weights = {
        name: tensor.detach().to(torch.float32).contiguous()
        for name, tensor in student.reference_state_dict().items()
    }
    save_file(weights, str(path / "model.safetensors"))

    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        **{k: v for k, v in student.config.__dict__.items()},
    }
    if source_model_dir is not None:
        source = Path(source_model_dir)
        original = json.loads((source / "config.json").read_text(encoding="utf-8"))
        original.update(
            {
                "num_hidden_layers": student.config.num_hidden_layers,
                "tie_word_embeddings": student.config.tie_word_embeddings,
                "torch_dtype": "float32",
            }
        )
        config = original
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                     "chat_template.jinja", "generation_config.json"):
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, path / name)

    (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if extra:
        (path / "training.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    return path
