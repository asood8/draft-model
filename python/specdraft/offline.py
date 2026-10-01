"""Offline acceptance metrics and the round simulator (plan §11.6).

Decoding on a laptop CPU runs at roughly 15-30 tokens a second, so measuring acceptance
by actually decoding every draft × decoding mode × γ × task would take days. Instead each
model is run *once* over text the target generated, and the rounds are simulated from the
per-position numbers. That is not an approximation:

* **Greedy.** The speculative output is the target's greedy text, and the draft's context
  at every position is the accepted prefix of that text, so teacher-forced top-1 matches
  determine every round exactly.
* **Sampling.** In one speculative step, the chance that the emitted token x came from an
  accepted guess is ``min(p(x), q(x)) / p(x)``. So if the reference text was sampled from
  the target with the same settings, drawing an independent Bernoulli with that
  probability at each drafted position reproduces the joint distribution of outputs and
  round boundaries. Averaged over x ~ p, that probability is exactly 1 − TVD(p, q).

Everything here works on NumPy arrays of per-position numbers, so it is also what scores
training checkpoints and sweeps γ for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .sampling import GREEDY, SamplingConfig, expected_acceptance, warp_probs


@dataclass
class PositionMetrics:
    """One value per predicted token."""

    greedy_match: np.ndarray  # the two models' top-1 tokens agree: acceptance when greedy
    one_minus_tvd: np.ndarray  # Σ min(p, q): expected acceptance when sampling
    coupled_accept: np.ndarray  # min(1, q(x)/p(x)) for the reference token x
    draft_confidence: np.ndarray  # max softmax(draft logits), for early stopping
    target_match: np.ndarray  # target's top-1 == the reference token, a sanity check
    tokens: np.ndarray  # the reference tokens these refer to

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def is_greedy_reference(self) -> bool:
        """Whether this text really is the target's greedy output.

        The greedy simulation is only exact on such text, because the draft's context
        during decoding is the accepted prefix of exactly that continuation.
        """
        return bool(self.target_match.all())

    @property
    def greedy_alpha(self) -> float:
        return float(self.greedy_match.mean())

    @property
    def sampling_alpha(self) -> float:
        return float(self.one_minus_tvd.mean())

    def accept_prob(self, greedy: bool) -> np.ndarray:
        """What the simulator needs: 1/0 matches for greedy, coupled draws for sampling."""
        return self.greedy_match.astype(np.float64) if greedy else self.coupled_accept


def _row_logits(model, tokens: Tensor, rows: np.ndarray, chunk: int):
    """Yield (begin, end, logits) for the requested rows, in chunks.

    Models that can hand back hidden states (the reference and the twin) get their output
    layer applied chunk by chunk, so a full [T, 151936] float32 tensor never exists. The C++
    engine cannot separate its output layer from its forward pass, so it is asked for all
    logits at once instead.
    """
    try:
        hidden = model.forward(tokens, hidden_only=True)
    except NotImplementedError:
        logits = model.forward(tokens, cache=model.new_cache(len(tokens)), only_last_logits=False)
        for begin in range(0, len(rows), chunk):
            end = min(begin + chunk, len(rows))
            yield begin, end, logits[torch.as_tensor(rows[begin:end])]
        return

    for begin in range(0, len(rows), chunk):
        end = min(begin + chunk, len(rows))
        index = torch.as_tensor(rows[begin:end], device=hidden.device)
        yield begin, end, model.logits_from_hidden(hidden[index])


def concatenate(parts: Sequence[PositionMetrics]) -> PositionMetrics:
    """Pool the scores from several sequences into one set of per-position numbers."""
    if not parts:
        raise ValueError("nothing to concatenate")
    return PositionMetrics(
        greedy_match=np.concatenate([part.greedy_match for part in parts]),
        one_minus_tvd=np.concatenate([part.one_minus_tvd for part in parts]),
        coupled_accept=np.concatenate([part.coupled_accept for part in parts]),
        draft_confidence=np.concatenate([part.draft_confidence for part in parts]),
        target_match=np.concatenate([part.target_match for part in parts]),
        tokens=np.concatenate([part.tokens for part in parts]),
    )


@torch.no_grad()
def score_sequence(
    target,
    draft,
    tokens: Tensor,
    response_start: int,
    config: SamplingConfig = GREEDY,
    vocab_limit: int | None = None,
    chunk: int = 256,
) -> PositionMetrics:
    """Run both models over one reference sequence and score each response position.

    ``tokens`` is prompt + response; ``response_start`` is the index of the first response
    token. Logits row t - 1 predicts token t, so rows response_start-1 .. len-2 are scored.
    The output layer is applied in chunks, because a full [T, 151936] float32 tensor would
    be gigabytes.
    """
    if not 1 <= response_start < len(tokens):
        raise ValueError("response_start must be inside the sequence")

    rows = np.arange(response_start - 1, len(tokens) - 1)  # logits rows to score
    predicted = np.asarray(tokens[response_start:].tolist(), dtype=np.int64)

    greedy_match = np.zeros(len(rows), dtype=bool)
    one_minus_tvd = np.zeros(len(rows), dtype=np.float64)
    coupled = np.zeros(len(rows), dtype=np.float64)
    confidence = np.zeros(len(rows), dtype=np.float64)
    target_match = np.zeros(len(rows), dtype=bool)

    for (begin, end, p_logits), (_, _, q_logits) in zip(
        _row_logits(target, tokens, rows, chunk), _row_logits(draft, tokens, rows, chunk)
    ):
        if vocab_limit is not None:
            p_logits = p_logits[..., :vocab_limit]
            q_logits = q_logits[..., :vocab_limit]
        wanted = torch.as_tensor(predicted[begin:end], device=p_logits.device)

        target_top = p_logits.argmax(-1)
        greedy_match[begin:end] = (q_logits.argmax(-1) == target_top).cpu().numpy()
        target_match[begin:end] = (target_top == wanted).cpu().numpy()

        # Confidence always comes from the unwarped draft distribution, so it means the
        # same thing under greedy decoding and under sampling.
        confidence[begin:end] = (
            torch.softmax(q_logits.to(torch.float32), dim=-1).max(-1).values.cpu().numpy()
        )

        if config.greedy:
            # Acceptance is deterministic: it depends only on whether the top tokens agree.
            match = greedy_match[begin:end].astype(np.float64)
            one_minus_tvd[begin:end] = match
            coupled[begin:end] = match
            continue

        p = warp_probs(p_logits, config)
        q = warp_probs(q_logits, config)
        one_minus_tvd[begin:end] = expected_acceptance(p, q).cpu().numpy()
        p_x = p.gather(-1, wanted[:, None]).squeeze(-1)
        q_x = q.gather(-1, wanted[:, None]).squeeze(-1)
        ratio = torch.where(
            p_x > 0, q_x / p_x.clamp_min(torch.finfo(p_x.dtype).tiny), torch.zeros_like(p_x)
        )
        coupled[begin:end] = ratio.clamp(max=1.0).cpu().numpy()

    return PositionMetrics(
        greedy_match=greedy_match,
        one_minus_tvd=one_minus_tvd,
        coupled_accept=coupled,
        draft_confidence=confidence,
        target_match=target_match,
        tokens=predicted,
    )


# --------------------------------------------------------------------- the simulator


def _run_lengths(flags: np.ndarray, gamma: int) -> np.ndarray:
    """For each position, how many consecutive flags are True from there, capped at γ."""
    n = len(flags)
    position = np.arange(n)
    falses = np.flatnonzero(~flags)
    if len(falses) == 0:  # every guess accepted everywhere
        return np.minimum(n - position, gamma)
    following = np.searchsorted(falses, position, side="left")
    clamped = np.minimum(following, len(falses) - 1)
    next_false = np.where(following < len(falses), falses[clamped], n)
    return np.minimum(next_false - position, gamma)


def simulate(
    accept_prob: np.ndarray,
    gamma: int,
    draws: int = 32,
    seed: int = 0,
    draft_confidence: np.ndarray | None = None,
    confidence_threshold: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Replay the rounds over a reference sequence.

    Returns tokens per target forward pass, and the distribution of accepted-prefix
    lengths (an array of γ+1 counts).

    With ``draft_confidence`` and a threshold, drafting stops at a position the draft is
    unsure about (plan §8.3). The token count stays exact; only the number of wasted draft
    steps after a rejection cannot be seen offline.
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    accept_prob = np.asarray(accept_prob, dtype=np.float64)
    if accept_prob.size == 0:
        raise ValueError("need at least one scored position")

    deterministic = bool(np.all((accept_prob == 0.0) | (accept_prob == 1.0)))
    if deterministic:
        draws = 1  # greedy: nothing random is left

    confident = (
        np.ones(len(accept_prob), dtype=bool)
        if draft_confidence is None
        else np.asarray(draft_confidence) >= confidence_threshold
    )

    rng = np.random.default_rng(seed)
    total_tokens = 0
    total_steps = 0
    lengths = np.zeros(gamma + 1, dtype=np.int64)

    for _ in range(draws):
        flags = confident & (rng.random(len(accept_prob)) < accept_prob)
        runs = _run_lengths(flags, gamma)
        i = 0
        while i < len(flags):
            n = int(runs[i])
            if i + n > len(flags):  # the reference text ran out mid-round
                n = len(flags) - i
            lengths[n] += 1
            i += n + 1  # n accepted guesses, plus one token from the target
            total_steps += 1
        total_tokens += len(flags)

    return total_tokens / total_steps, lengths


def simulate_tokens_per_step(accept_prob: np.ndarray, gamma: int, **kwargs) -> float:
    return simulate(accept_prob, gamma, **kwargs)[0]


# ------------------------------------------------------------------ the speedup model


def tau_from_alpha(alpha: float, gamma: int) -> float:
    """Expected tokens per round if acceptance were i.i.d.: (1 − α^(γ+1)) / (1 − α)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if alpha == 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def predicted_speedup(tau: float, gamma: int, c: float, v: float = 1.0, o: float = 0.0) -> float:
    """τ / (γ·c + v + o), the model of plan §3.

    c is one draft step over one target step, v is the cost of verifying γ+1 tokens in the
    same units, and o is per-round overhead such as sampling over a 152k vocabulary.
    """
    return tau / (gamma * c + v + o)


# ----------------------------------------------------- what kinds of token get rejected
#
# Plan §9.3: acceptance is an average, and averages hide the interesting part. Grouping
# positions by what kind of token the target wanted shows where a draft is actually failing,
# which is what makes the write-up worth reading. Numbers, names and sentence openings are the
# usual suspects, and the classes below are chosen to separate them.

TOKEN_CLASSES = (
    "whitespace",
    "digit",
    "punctuation",
    "cjk",
    "word_start_capital",
    "word_start",
    "word_continuation",
    "special",
    "other",
)


def classify_token(text: str, is_special: bool = False) -> str:
    """Put one token's surface form into a class. The order below is the priority."""
    if is_special:
        return "special"
    if not text:
        return "other"
    if text.strip() == "":
        return "whitespace"
    if any(character.isdigit() for character in text):
        return "digit"
    if any("\u4e00" <= character <= "\u9fff" or "\u3040" <= character <= "\u30ff" for character in text):
        return "cjk"
    stripped = text.strip()
    if all(not character.isalnum() for character in stripped):
        return "punctuation"

    leading_space = text[0].isspace()
    first_letter = next((character for character in stripped if character.isalpha()), "")
    if leading_space:
        return "word_start_capital" if first_letter.isupper() else "word_start"
    return "word_continuation"


def token_class_names(tokenizer, token_ids: np.ndarray) -> list[str]:
    """One class per token id, using the tokenizer's own surface forms."""
    special = set(getattr(tokenizer, "all_special_ids", None) or [])
    classes = []
    for token_id in token_ids.tolist():
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        classes.append(classify_token(text, is_special=int(token_id) in special))
    return classes


def acceptance_by_class(
    metrics: PositionMetrics, classes: Sequence[str], greedy: bool = True
) -> dict[str, dict[str, float]]:
    """Acceptance per token class, with how much of the text each class accounts for.

    A class that is both frequent and poorly accepted is where a distillation run should be
    judged; a rare one with low acceptance is a curiosity.
    """
    accept = metrics.accept_prob(greedy=greedy)
    totals: dict[str, list[float]] = {}
    for name, value in zip(classes, accept):
        totals.setdefault(name, []).append(float(value))

    positions = len(accept)
    return {
        name: {
            "positions": len(values),
            "share": len(values) / positions if positions else 0.0,
            "acceptance": sum(values) / len(values),
        }
        for name, values in sorted(totals.items(), key=lambda item: -len(item[1]))
    }


def gamma_sweep(
    metrics: PositionMetrics,
    gammas: Sequence[int],
    greedy: bool = True,
    draws: int = 32,
    seed: int = 0,
    confidence_threshold: float = 0.0,
) -> dict[int, float]:
    """Tokens per target pass for each γ, from the one pass over the text.

    This is what makes a γ sweep nearly free: the same per-position numbers answer every γ, so
    the expensive part happens once.
    """
    accept = metrics.accept_prob(greedy=greedy)
    confidence = metrics.draft_confidence if confidence_threshold > 0 else None
    return {
        gamma: simulate_tokens_per_step(
            accept,
            gamma,
            draws=draws,
            seed=seed,
            draft_confidence=confidence,
            confidence_threshold=confidence_threshold,
        )
        for gamma in gammas
    }


def best_gamma(
    tokens_per_step: dict[int, float],
    c: float,
    v: dict[int, float] | float = 1.0,
    o: float = 0.0,
) -> tuple[int, float]:
    """The γ with the best predicted speedup, and that speedup.

    ``v`` may be a single number or the measured v(k) curve, in which case v(γ+1) is used for
    each γ — which is the whole reason the curve gets measured.
    """
    best = (0, 0.0)
    for gamma, tau in tokens_per_step.items():
        verification = v if isinstance(v, (int, float)) else v.get(gamma + 1, 1.0)
        speedup = predicted_speedup(tau, gamma, c, float(verification), o)
        if speedup > best[1]:
            best = (gamma, speedup)
    return best
