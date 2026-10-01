"""A trainable wrapper around the reference model.

``Qwen3Reference`` keeps its weights as plain tensors and does its arithmetic functionally, so
turning it into something an optimizer can update needs nothing more than swapping those
tensors for parameters: autograd then flows through the same forward pass that is already
checked against Hugging Face. Training the draft therefore exercises exactly the model the
engine will run, rather than a second implementation that might drift from it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .reference import Qwen3Config, Qwen3Reference, load_safetensors


class TrainableDraft(nn.Module):
    """The draft model, with its weights registered as parameters.

    ``freeze_embeddings`` leaves the tied embedding and output matrix alone, which is about a
    quarter of the 0.6B's parameters and the easiest thing to give up when memory is tight
    (plan §11.5).
    """

    def __init__(
        self,
        config: Qwen3Config,
        state_dict: dict[str, Tensor],
        freeze_embeddings: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.config = config
        self.freeze_embeddings = freeze_embeddings

        # ParameterDict keys cannot contain dots, so the Hugging Face names are kept in a map.
        self._names = {name: name.replace(".", "|") for name in state_dict}
        self.weights = nn.ParameterDict(
            {
                self._names[name]: nn.Parameter(
                    tensor.detach().clone().to(dtype),
                    requires_grad=not (freeze_embeddings and _is_embedding(name)),
                )
                for name, tensor in state_dict.items()
            }
        )
        # The same parameter objects, under their original names, so the reference's forward
        # pass reads and differentiates through them.
        self.model = Qwen3Reference(config, {name: self.weights[key] for name, key in self._names.items()})

    @classmethod
    def from_pretrained(
        cls, path: str | Path, dtype: torch.dtype = torch.float32, **kwargs
    ) -> "TrainableDraft":
        path = Path(path)
        config = Qwen3Config.from_pretrained(path)
        return cls(config, load_safetensors(path), dtype=dtype, **kwargs)

    @classmethod
    def from_reference(cls, model: Qwen3Reference, **kwargs) -> "TrainableDraft":
        return cls(model.config, model.state_dict(), **kwargs)

    # -- what the trainer needs ----------------------------------------------------

    def hidden_states(self, tokens: Tensor) -> Tensor:
        """[N, hidden] for the whole sequence, with gradients."""
        return self.model.forward(tokens, hidden_only=True)

    @property
    def output_weight(self) -> Tensor:
        return self.model.lm_head

    def logits(self, tokens: Tensor) -> Tensor:
        return self.model.forward(tokens)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_count(self) -> tuple[int, int]:
        """(trainable, total)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        return trainable, total

    def reference_state_dict(self) -> dict[str, Tensor]:
        """Hugging Face names again, detached, ready for export or saving."""
        return {name: self.weights[key].detach() for name, key in self._names.items()}

    def as_reference(self) -> Qwen3Reference:
        """A read-only copy for evaluation, sharing no autograd history."""
        return Qwen3Reference(self.config, self.reference_state_dict())


def _is_embedding(name: str) -> bool:
    return name in ("model.embed_tokens.weight", "lm_head.weight")
