"""Distillation losses (plan §11.4).

The draft is trained to agree with the target, and which notion of "agree" matters: TVD is
exactly one minus the expected acceptance rate when sampling, so minimizing it optimizes the
thing the project measures. The KLs are the standard alternatives, and plain cross-entropy on
the target's own text is the baseline that answers whether the target's full distributions are
worth the trouble at all.

======  ======================================================  =============================
Name    Optimizes                                               Why it might win
======  ======================================================  =============================
``sft`` −log q(x) on the text                                   No teacher logits needed
``fkl`` KL(p‖q): mass-covering                                  Covers everything p might say
``rkl`` KL(q‖p): mode-seeking                                   Concentrates where p is sure
``tvd`` ½Σ|p−q| = 1 − expected acceptance                       Directly what sampling needs
``jsd`` symmetric, bounded                                      Stable when p and q disagree
======  ======================================================  =============================

With a 151,669-token vocabulary, one sequence's logits in float32 run to hundreds of
megabytes, so ``chunked_distill_loss`` applies the output layer a slice at a time and
recomputes it during the backward pass rather than keeping it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

LOSSES = ("sft", "fkl", "rkl", "tvd", "jsd")


def per_position_loss(
    draft_logits: Tensor,
    target_logits: Tensor | None = None,
    kind: str = "fkl",
    labels: Tensor | None = None,
    temperature: float = 1.0,
) -> Tensor:
    """One loss value per position. Shapes are [N, V] in and [N] out.

    Everything is computed in float32 even when the models run in half precision, because the
    acceptance rate depends on small probabilities.
    """
    if kind not in LOSSES:
        raise ValueError(f"unknown loss {kind!r}; expected one of {LOSSES}")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    q_log = F.log_softmax(draft_logits.float() / temperature, dim=-1)

    if kind == "sft":
        if labels is None:
            raise ValueError("sft needs labels: the tokens that actually came next")
        return F.nll_loss(q_log, labels, reduction="none")

    if target_logits is None:
        raise ValueError(f"{kind} needs the target's logits")
    p_log = F.log_softmax(target_logits.float() / temperature, dim=-1)
    p = p_log.exp()
    q = q_log.exp()

    if kind == "fkl":
        return (p * (p_log - q_log)).sum(-1)
    if kind == "rkl":
        return (q * (q_log - p_log)).sum(-1)
    if kind == "tvd":
        return 0.5 * (p - q).abs().sum(-1)
    # jsd: the average of both KLs against the mixture, computed from log-space to stay stable.
    m_log = torch.logaddexp(p_log, q_log) - math.log(2.0)
    return 0.5 * (p * (p_log - m_log)).sum(-1) + 0.5 * (q * (q_log - m_log)).sum(-1)


def distill_loss(
    draft_logits: Tensor,
    target_logits: Tensor | None = None,
    kind: str = "fkl",
    labels: Tensor | None = None,
    temperature: float = 1.0,
) -> Tensor:
    """The mean over positions, which is what gets optimized."""
    return per_position_loss(draft_logits, target_logits, kind, labels, temperature).mean()


def chunked_distill_loss(
    draft_hidden: Tensor,
    draft_output_weight: Tensor,
    target_logits: Tensor | None = None,
    kind: str = "fkl",
    labels: Tensor | None = None,
    temperature: float = 1.0,
    vocab_limit: int | None = None,
    chunk: int = 256,
    recompute: bool = True,
) -> Tensor:
    """The same loss, without ever holding the whole [N, V] logits tensor.

    draft_hidden: [N, H] hidden states from the student, carrying gradients.
    draft_output_weight: [V, H] the student's output matrix.
    target_logits: [N, V] from the teacher, no gradients; sliced to vocab_limit here.

    With ``recompute`` the student's logits for each slice are rebuilt during the backward pass
    instead of being stored, which is what makes a 151,669-token vocabulary fit.
    """
    positions = draft_hidden.shape[0]
    if positions == 0:
        raise ValueError("need at least one position")
    if target_logits is not None and target_logits.shape[0] != positions:
        raise ValueError("teacher and student disagree about how many positions there are")

    def slice_loss(hidden: Tensor, begin: int, end: int) -> Tensor:
        logits = hidden @ draft_output_weight.transpose(0, 1)
        if vocab_limit is not None:
            logits = logits[..., :vocab_limit]
        teacher = None
        if target_logits is not None:
            teacher = target_logits[begin:end]
            if vocab_limit is not None:
                teacher = teacher[..., :vocab_limit]
        wanted = None if labels is None else labels[begin:end]
        return per_position_loss(logits, teacher, kind, wanted, temperature).sum()

    total = draft_hidden.new_zeros((), dtype=torch.float32)
    for begin in range(0, positions, chunk):
        end = min(begin + chunk, positions)
        hidden = draft_hidden[begin:end]
        if recompute and torch.is_grad_enabled() and hidden.requires_grad:
            total = total + checkpoint(slice_loss, hidden, begin, end, use_reentrant=False)
        else:
            total = total + slice_loss(hidden, begin, end)
    return total / positions
