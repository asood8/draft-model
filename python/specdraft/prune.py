"""Making the draft cheaper by removing layers (plan §8.1).

Speedup depends on acceptance *and* on what a draft step costs, and on a CPU the cost is mostly
bytes: fewer layers means fewer weights to stream and fewer points where threads wait for each
other. The trade is acceptance, which distillation is then used to win back — that pair of moves
is what produces the α-versus-c curve the write-up is built around.

Which layers to drop is decided by block influence (ShortGPT): a layer that barely changes the
residual stream is doing little, so compare each layer's input and output and score it by how far
apart they point. The last layer is kept regardless, because removing it tends to cost far more
than its influence score suggests.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from .reference import Qwen3Config, Qwen3Reference


@torch.no_grad()
def block_influence(model: Qwen3Reference, sequences: list[Tensor]) -> list[float]:
    """One score per layer: 1 − cosine similarity between its input and its output.

    Averaged over every position of every calibration sequence. A score near zero means the layer
    leaves the residual stream pointing where it already pointed.
    """
    if not sequences:
        raise ValueError("block influence needs at least one calibration sequence")

    layers = model.config.num_hidden_layers
    totals = [0.0] * layers
    positions = 0

    for tokens in sequences:
        tokens = tokens.to(model.device)
        captured: list[Tensor] = []
        model.forward(tokens, capture=captured)
        # The input to layer 0 is the embedding; after that, each layer's input is the previous
        # layer's output.
        states = [model.embed_tokens[tokens]] + captured
        for index in range(layers):
            similarity = F.cosine_similarity(
                states[index].float(), states[index + 1].float(), dim=-1
            )
            totals[index] += float((1.0 - similarity).sum())
        positions += tokens.shape[0]

    return [total / positions for total in totals]


def choose_layers_to_keep(
    scores: list[float], keep: int, protect_last: bool = True
) -> list[int]:
    """The `keep` most influential layers, in their original order.

    ``protect_last`` keeps the final layer whatever it scores: in practice it feeds the output
    layer directly and dropping it costs more than the score implies.
    """
    total = len(scores)
    if not 0 < keep <= total:
        raise ValueError(f"keep must be between 1 and {total}")
    if keep == total:
        return list(range(total))

    protected = {total - 1} if protect_last else set()
    candidates = [index for index in range(total) if index not in protected]
    ranked = sorted(candidates, key=lambda index: scores[index], reverse=True)
    kept = set(list(protected) + ranked[: keep - len(protected)])
    return sorted(kept)


def prune_layers(model: Qwen3Reference, keep_indices: list[int]) -> Qwen3Reference:
    """A copy with only the named layers, renumbered so nothing downstream has to care."""
    if not keep_indices:
        raise ValueError("at least one layer must be kept")
    if sorted(set(keep_indices)) != list(keep_indices):
        raise ValueError("keep_indices must be sorted and free of duplicates")
    if keep_indices[-1] >= model.config.num_hidden_layers:
        raise ValueError("keep_indices refers to a layer that does not exist")

    state = model.state_dict()
    pruned: dict[str, Tensor] = {
        name: tensor
        for name, tensor in state.items()
        if not name.startswith("model.layers.")
    }
    for new_index, old_index in enumerate(keep_indices):
        old_prefix = f"model.layers.{old_index}."
        new_prefix = f"model.layers.{new_index}."
        for name, tensor in state.items():
            if name.startswith(old_prefix):
                pruned[new_prefix + name[len(old_prefix) :]] = tensor

    config = replace(model.config, num_hidden_layers=len(keep_indices))
    return Qwen3Reference(config, pruned)


def prune_to_size(
    model: Qwen3Reference,
    keep: int,
    sequences: list[Tensor],
    protect_last: bool = True,
) -> tuple[Qwen3Reference, list[int], list[float]]:
    """Score, choose and prune in one step. Returns the model, the kept layers and the scores."""
    scores = block_influence(model, sequences)
    keep_indices = choose_layers_to_keep(scores, keep, protect_last)
    return prune_layers(model, keep_indices), keep_indices, scores


def bytes_per_token(config: Qwen3Config, bits_per_weight: float = 4.5) -> int:
    """Roughly what one decode step has to read, which is what sets c on a CPU.

    Counts the matrices a step streams: the per-layer projections plus the output layer. The
    embedding lookup touches a single row, so it does not count.
    """
    per_layer = (
        config.hidden_size * (config.q_dim + 2 * config.kv_dim)  # fused qkv
        + config.q_dim * config.hidden_size  # attention output
        + 3 * config.hidden_size * config.intermediate_size  # gate, up, down
    )
    output_layer = config.vocab_size * config.hidden_size
    return int((per_layer * config.num_hidden_layers + output_layer) * bits_per_weight / 8)


def describe_pruning(original: Qwen3Config, pruned: Qwen3Config, bits_per_weight: float = 4.5) -> dict:
    """What the pruning bought, in the units the speedup formula uses."""
    before = bytes_per_token(original, bits_per_weight)
    after = bytes_per_token(pruned, bits_per_weight)
    return {
        "layers_before": original.num_hidden_layers,
        "layers_after": pruned.num_hidden_layers,
        "bytes_per_token_before": before,
        "bytes_per_token_after": after,
        "bytes_ratio": after / before,
    }


def save_pruned(
    path: str | Path,
    model: Qwen3Reference,
    source_model_dir: str | Path | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a pruned model where the trainer and the export script can pick it up."""
    from .train import save_draft
    from .trainable import TrainableDraft

    student = TrainableDraft(model.config, model.state_dict())
    return save_draft(path, student, source_model_dir=source_model_dir, extra=extra)
