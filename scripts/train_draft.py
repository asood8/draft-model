"""Distil a draft model toward a target (plan §11).

One run of the grid, start to finish::

    python scripts/train_draft.py \
        --student models/Qwen3-0.6B --teacher models/Qwen3-4B \
        --data data/target_generated.jsonl --loss tvd \
        --max-tokens 10000000 --out runs/tvd-target

On Kaggle's two T4s, put the teacher and the student on different devices and switch on fp16::

    --teacher-device cuda:0 --student-device cuda:1 --amp

``--teacher-format q4`` trains against the quantization twin instead of the full-precision
target, which is what the engine actually verifies against. The plan compares both.

Checkpoints are written periodically and the run resumes from them, because Kaggle sessions end
on their own schedule.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from specdraft.data import read_records, read_sequences, tokenize_records
from specdraft.reference import Qwen3Config, Qwen3Reference
from specdraft.train import (
    TrainConfig,
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
    save_draft,
    teacher_from_model,
    train,
    validate,
)
from specdraft.trainable import QuantizationAwareDraft, TrainableDraft
from specdraft.twin import QuantizedTwin


def load_sequences(path: Path, tokenizer, max_length: int):
    """Accepts either tokenized sequences or raw prompt/response records."""
    if path.suffix == ".jsonl":
        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        if "tokens" in first:
            return read_sequences(path)
    if tokenizer is None:
        raise SystemExit("raw records need a tokenizer; pass --student so one can be loaded")
    return tokenize_records(tokenizer, read_records(path), max_length=max_length)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True, help="the draft to start from")
    parser.add_argument("--teacher", type=Path, required=True, help="the target to match")
    parser.add_argument("--data", type=Path, required=True, help="jsonl of records or sequences")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--loss", default="fkl", help="sft, fkl, rkl, tvd or jsd")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-tokens", type=int, default=1_000_000, help="response tokens")
    parser.add_argument("--tokens-per-step", type=int, default=65_536)
    parser.add_argument("--max-sequence-tokens", type=int, default=2048)
    parser.add_argument("--loss-chunk", type=int, default=256)
    parser.add_argument("--validation", type=int, default=32, help="sequences held back")
    parser.add_argument("--teacher-format", default=None, choices=[None, "q4", "q8"],
                        help="train against the quantization twin the engine runs")
    parser.add_argument("--teacher-device", default="cpu")
    parser.add_argument("--student-device", default="cpu")
    parser.add_argument("--amp", action="store_true", help="fp16 autocast with a gradient scaler")
    parser.add_argument("--eight-bit-adam", action="store_true")
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--quantization-aware", action="store_true",
                        help="train through the draft's own rounding, since the engine runs it "
                             "quantized (plan section 11.1)")
    parser.add_argument("--student-format", default="q4", choices=["q4", "q8"],
                        help="the format the draft will be run in, for --quantization-aware")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    vocab_limit = len(tokenizer)

    sequences = load_sequences(args.data, tokenizer, args.max_sequence_tokens)
    random.Random(args.seed).shuffle(sequences)
    held_out = sequences[: args.validation]
    training = sequences[args.validation :]
    response_tokens = sum(s.response_tokens for s in training)
    print(f"{len(training)} training sequences ({response_tokens / 1e6:.2f}M response tokens), "
          f"{len(held_out)} held out")

    if args.quantization_aware:
        from specdraft.reference import Qwen3Config, load_safetensors

        student = QuantizationAwareDraft(
            Qwen3Config.from_pretrained(args.student),
            load_safetensors(args.student),
            weight_format=args.student_format,
            freeze_embeddings=args.freeze_embeddings,
        ).to(args.student_device)
    else:
        student = TrainableDraft.from_pretrained(
            args.student, freeze_embeddings=args.freeze_embeddings
        ).to(args.student_device)
    trainable, total = student.parameter_count()
    print(f"student: {trainable / 1e6:.0f}M trainable of {total / 1e6:.0f}M"
          + (f", trained through {args.student_format} rounding" if args.quantization_aware else ""))

    if args.teacher_format is None:
        teacher_model = Qwen3Reference.from_pretrained(args.teacher, device=args.teacher_device)
        print(f"teacher: {args.teacher.name} at full precision")
    else:
        teacher_model = QuantizedTwin.from_pretrained(
            args.teacher, device=args.teacher_device, weight_format=args.teacher_format
        )
        print(f"teacher: {args.teacher.name} as the engine runs it ({teacher_model.describe()})")
    teacher = teacher_from_model(teacher_model, device=args.student_device)

    config = TrainConfig(
        loss=args.loss,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        max_response_tokens=args.max_tokens,
        tokens_per_step=args.tokens_per_step,
        loss_chunk=args.loss_chunk,
        vocab_limit=vocab_limit,
        device=args.student_device,
        amp=args.amp,
        eight_bit_adam=args.eight_bit_adam,
        max_sequence_tokens=args.max_sequence_tokens,
        seed=args.seed,
    )
    optimizer = build_optimizer(student, config)
    state = None
    if args.resume and (args.out / "checkpoint.pt").is_file():
        state = load_checkpoint(args.out, student, optimizer)
        print(f"resumed at step {state.step}, {state.response_tokens / 1e6:.2f}M tokens in")

    before = validate(student, teacher, held_out, config)
    print(f"before: top-1 {before.get('greedy_top1_match', float('nan')):.4f}  "
          f"acceptance {before.get('sampling_acceptance', float('nan')):.4f}")

    def report(entry: dict) -> None:
        line = f"step {entry['step']:>5}  loss {entry['loss']:.4f}  " \
               f"{entry['response_tokens'] / 1e6:.2f}M tokens  lr {entry['learning_rate']:.2e}"
        if "sampling_acceptance" in entry:
            line += f"  acceptance {entry['sampling_acceptance']:.4f}"
        print(line, flush=True)

    # Repeat the data if the budget asks for more tokens than one pass provides.
    def stream():
        while True:
            for sequence in training:
                yield sequence

    state = train(
        student, teacher, stream(), config, validation=held_out, checkpoint_dir=args.out,
        optimizer=optimizer, state=state, on_log=report,
    )

    after = validate(student, teacher, held_out, config)
    print(f"after:  top-1 {after.get('greedy_top1_match', float('nan')):.4f}  "
          f"acceptance {after.get('sampling_acceptance', float('nan')):.4f}")

    save_checkpoint(args.out, student, optimizer, state, config)
    save_draft(
        args.out / "draft",
        student,
        source_model_dir=args.student,
        extra={
            "loss": args.loss,
            "quantization_aware": args.quantization_aware,
            "student_format": args.student_format if args.quantization_aware else None,
            "teacher": str(args.teacher),
            "teacher_format": args.teacher_format or "fp32",
            "response_tokens": state.response_tokens,
            "steps": state.step,
            "skipped_steps": state.skipped_steps,
            "before": before,
            "after": after,
            "data": str(args.data),
        },
    )
    print(f"\nwrote {args.out / 'draft'}  ({state.step} steps, "
          f"{state.response_tokens / 1e6:.2f}M response tokens, {state.skipped_steps} skipped)")


if __name__ == "__main__":
    main()
