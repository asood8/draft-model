"""Speculative decoding in Python: the round loop the C++ engine will mirror.

The cache invariant is the one from plan §6.1/§10.2. ``seq`` is the whole token sequence,
each model's cache holds some prefix of it, and before a forward pass a model is fed
exactly the tokens its cache is missing. After verification, both caches are rewound to
"everything but the newest token", which is one token for the target and one or two for
the draft: after a round where every guess was accepted, the draft never saw its own last
guess or the target's bonus token.

Anything with ``forward(tokens, cache, only_last_logits)`` and ``new_cache(n)`` can be
driven by this loop, so the same code and the same tests apply to the PyTorch reference
and later to the C++ engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import torch
from torch import Tensor

from .sampling import GREEDY, SamplingConfig, accept_or_resample, sample, warp_probs


class ModelLike(Protocol):
    device: Any

    def new_cache(self, max_positions: int) -> Any: ...

    def forward(
        self, tokens: Tensor, cache: Any = None, only_last_logits: bool = False
    ) -> Tensor: ...


@dataclass
class DecodeStats:
    """Everything needed for the speedup model of plan §3."""

    emitted: int = 0
    rounds: int = 0
    accepted: int = 0  # accepted draft tokens
    rejections: int = 0  # rounds that ended in a rejection
    target_forwards: int = 0  # verification passes, prefill excluded
    draft_forwards: int = 0
    accepted_lengths: list[int] = field(default_factory=list)

    @property
    def tokens_per_target_forward(self) -> float:
        """τ: the number this project is trying to raise."""
        return self.emitted / self.target_forwards if self.target_forwards else float("nan")

    @property
    def alpha(self) -> float:
        """Per-token acceptance rate.

        Each round shows accepts until the first rejection, so the maximum-likelihood
        estimate for a truncated geometric is accepts / (accepts + rejections).
        """
        trials = self.accepted + self.rejections
        return self.accepted / trials if trials else float("nan")


def _tokens(values: Sequence[int], model: ModelLike) -> Tensor:
    return torch.tensor(list(values), dtype=torch.long, device=model.device)


def _trim(logits: Tensor, vocab_limit: int | None) -> Tensor:
    return logits if vocab_limit is None else logits[..., :vocab_limit]


def plain_generate(
    model: ModelLike,
    prompt: Sequence[int],
    max_new_tokens: int,
    config: SamplingConfig = GREEDY,
    vocab_limit: int | None = None,
    stop: set[int] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[list[int], DecodeStats]:
    """Ordinary token-at-a-time decoding: the baseline and the correctness reference."""
    seq = [int(t) for t in prompt]
    stats = DecodeStats()
    cache = model.new_cache(len(seq) + max_new_tokens + 1)
    stop = set(stop or ())

    if len(seq) > 1:  # prefill everything but the newest token
        model.forward(_tokens(seq[:-1], model), cache=cache)

    for _ in range(max_new_tokens):
        logits = model.forward(_tokens(seq[cache.pos :], model), cache=cache, only_last_logits=True)
        stats.target_forwards += 1
        row = _trim(logits[-1], vocab_limit)
        token = int(row.argmax()) if config.greedy else sample(warp_probs(row, config), generator)
        seq.append(token)
        if token in stop:
            break

    generated = seq[len(prompt) :]
    stats.emitted = len(generated)
    return generated, stats


def speculative_generate(
    target: ModelLike,
    draft: ModelLike,
    prompt: Sequence[int],
    max_new_tokens: int,
    gamma: int = 4,
    config: SamplingConfig = GREEDY,
    vocab_limit: int | None = None,
    stop: set[int] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[list[int], DecodeStats]:
    """Draft γ tokens, verify them in one target pass, repeat.

    The output has exactly the distribution plain decoding from ``target`` would have, so
    ``config=GREEDY`` must reproduce ``plain_generate`` token for token.
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")

    seq = [int(t) for t in prompt]
    prompt_len = len(seq)
    capacity = prompt_len + max_new_tokens + gamma + 2
    target_cache = target.new_cache(capacity)
    draft_cache = draft.new_cache(capacity)
    stats = DecodeStats()
    stop = set(stop or ())

    if prompt_len > 1:  # after this, every round has the same shapes
        prefill = _tokens(seq[:-1], target)
        target.forward(prefill, cache=target_cache)
        draft.forward(_tokens(seq[:-1], draft), cache=draft_cache)

    finished = False
    while len(seq) - prompt_len < max_new_tokens and not finished:
        # -- draft γ tokens, one at a time -------------------------------------------
        guesses: list[int] = []
        q_rows: list[Tensor] = []
        pending = seq[draft_cache.pos :]  # 1 token, or 2 after an all-accepted round
        logits = draft.forward(_tokens(pending, draft), cache=draft_cache, only_last_logits=True)
        stats.draft_forwards += 1
        for j in range(gamma):
            row = _trim(logits[-1], vocab_limit)
            if config.greedy:
                q_rows.append(row)
                token = int(row.argmax())
            else:
                probs = warp_probs(row, config)
                q_rows.append(probs)
                token = sample(probs, generator)
            guesses.append(token)
            if j + 1 < gamma:
                logits = draft.forward(
                    _tokens([token], draft), cache=draft_cache, only_last_logits=True
                )
                stats.draft_forwards += 1

        # -- verify all of them in one target pass ------------------------------------
        target_logits = target.forward(_tokens([seq[-1]] + guesses, target), cache=target_cache)
        stats.target_forwards += 1
        rows = _trim(target_logits, vocab_limit)
        p = rows if config.greedy else warp_probs(rows, config)

        accepted, next_token = accept_or_resample(
            p,
            torch.stack(q_rows),
            _tokens(guesses, target),
            greedy=config.greedy,
            generator=generator,
        )
        n = int(accepted.shape[0])
        stats.rounds += 1
        stats.accepted += n
        stats.accepted_lengths.append(n)
        if n < gamma:
            stats.rejections += 1

        # -- commit, then rewind both caches to "everything but the newest token" ------
        for token in guesses[:n] + [next_token]:
            if len(seq) - prompt_len >= max_new_tokens:
                break
            seq.append(token)
            if token in stop:
                finished = True
                break
        target_cache.rewind_to(len(seq) - 1)
        draft_cache.rewind_to(min(draft_cache.pos, len(seq) - 1))

    generated = seq[prompt_len:]
    stats.emitted = len(generated)
    return generated, stats
