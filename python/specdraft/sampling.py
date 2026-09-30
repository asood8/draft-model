"""Sampling warps and the speculative acceptance rule.

The rule that makes speculative decoding exact: accept a guess with probability
min(1, p/q), resample from the renormalized ``max(0, p - q)`` at the first rejection, and
take a bonus token from the target when every guess is accepted.

Two conditions have to hold for the output to follow the target exactly:

* ``p`` is the *warped* target distribution, after temperature, top-k and top-p;
* ``q`` is *exactly* the distribution the draft sampled from.

The draft's warps may differ from the target's without breaking correctness; they only
change how often guesses are accepted. This module is also the oracle the C++
implementation is tested against (plan §10.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingConfig:
    """temperature = 0 means greedy; top_k = 0 and top_p = 1 mean no filtering."""

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")


GREEDY = SamplingConfig(temperature=0.0)


def warp_probs(logits: Tensor, config: SamplingConfig) -> Tensor:
    """Turn logits into the distribution that is actually sampled from.

    Accepts [V] or [N, V] and returns the same shape. Computed in float32, because the
    acceptance test compares small probabilities.
    """
    if config.greedy:
        raise ValueError("greedy decoding has no distribution; compare argmax instead")

    logits = logits.to(torch.float32)
    single = logits.ndim == 1
    if single:
        logits = logits[None, :]
    logits = logits / config.temperature

    if config.top_k and config.top_k < logits.shape[-1]:
        kth = logits.topk(config.top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = torch.softmax(logits, dim=-1)

    if config.top_p < 1.0:
        ordered, order = probs.sort(dim=-1, descending=True)
        # Keep the shortest prefix whose mass reaches top_p: drop a token only if the mass
        # before it already got there. The most likely token is always kept.
        mass_before = ordered.cumsum(dim=-1) - ordered
        keep_sorted = mass_before < config.top_p
        keep = torch.zeros_like(keep_sorted).scatter_(-1, order, keep_sorted)
        probs = probs * keep
        probs = probs / probs.sum(dim=-1, keepdim=True)

    return probs[0] if single else probs


def sample(probs: Tensor, generator: torch.Generator | None = None) -> int:
    """One token from a [V] distribution."""
    return int(torch.multinomial(probs, 1, generator=generator).item())


def residual_probs(p: Tensor, q: Tensor) -> Tensor:
    """The renormalized max(0, p - q) a rejection resamples from."""
    residual = (p - q).clamp_min(0)
    total = residual.sum()
    # Mathematically total > 0 whenever a rejection can happen: rejection needs
    # p(x) < q(x) for some x, and both sum to 1. The guard is only for float round-off.
    return residual / total if total > 0 else p


def accept_or_resample(
    p: Tensor,
    q: Tensor,
    draft_tokens: Tensor,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, int]:
    """Verify γ guesses in one shot.

    p: [γ+1, V] warped target probabilities (greedy: any monotone scores, e.g. logits).
    q: [γ, V] the exact distributions the draft sampled from.
    draft_tokens: [γ].

    Returns the accepted prefix of the guesses and one token from the target, which is a
    correction after a rejection or a bonus when everything was accepted.
    """
    gamma = int(draft_tokens.shape[0])
    if p.shape[0] != gamma + 1 or q.shape[0] != gamma:
        raise ValueError(f"expected p [{gamma + 1}, V] and q [{gamma}, V]")

    index = torch.arange(gamma, device=p.device)
    if greedy:
        accepted = draft_tokens == p[:gamma].argmax(-1)
    else:
        u = torch.rand(gamma, generator=generator, device=p.device)
        # u < min(1, p/q), written without a division so q = 0 cannot blow up.
        accepted = u * q[index, draft_tokens] < p[index, draft_tokens]

    n = int(accepted.long().cumprod(0).sum())

    if greedy:
        next_token = int(p[n].argmax())
    elif n < gamma:
        next_token = sample(residual_probs(p[n], q[n]), generator)
    else:
        next_token = sample(p[gamma], generator)
    return draft_tokens[:n], next_token


def expected_acceptance(p: Tensor, q: Tensor) -> Tensor:
    """Σ min(p, q) = 1 − TVD(p, q): the chance a guess at this position is accepted."""
    return torch.minimum(p, q).sum(-1)
