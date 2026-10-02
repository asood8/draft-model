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
    (plan section 11.5).
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


class QuantizationAwareDraft(TrainableDraft):
    """A draft trained with the rounding it will be run under (plan section 11.1, stretch).

    The engine runs the draft quantized, and 4-bit costs real quality on a model this small:
    WikiText-2 perplexity went from 28.5 to 32.2 on the 0.6B. Training against full-precision
    weights and quantizing afterwards asks the optimizer to find a point that happens to survive
    rounding; training *through* the rounding asks for one that is good after rounding, which is a
    different and easier request.

    The rounding is a step function with no useful derivative, so the backward pass uses a
    straight-through estimator: the forward value is quantized and the gradient reaches the
    underlying weight as though it had not been. The optimizer's own copy stays full precision;
    only what the forward pass multiplies is rounded, and it is re-rounded every step, since the
    weights move between them.
    """

    def __init__(
        self,
        config: Qwen3Config,
        state_dict: dict[str, Tensor],
        weight_format: str = "q4",
        output_format: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, state_dict, **kwargs)
        self.weight_format = weight_format
        self.output_format = weight_format if output_format is None else output_format
        # The reference calls this for every matrix multiply, which is exactly where the engine
        # reads a quantized weight, so the rounding goes here rather than into a stale copy.
        self.model.matmul = self._quantized_matmul  # type: ignore[method-assign]
        self.model.embed = self._quantized_embed  # type: ignore[method-assign]

    def _format_for(self, name: str) -> str | None:
        if name == "lm_head":
            return self.output_format
        # Norm weights are not quantized, in training or in the engine.
        return self.weight_format if name.endswith("_proj") else None

    def _quantized_matmul(self, x: Tensor, weight: Tensor, name: str) -> Tensor:
        fmt = self._format_for(name)
        if fmt is not None:
            weight = _StraightThroughQuantize.apply(weight, fmt)
        return x @ weight.transpose(0, 1)

    def _quantized_embed(self, tokens: Tensor) -> Tensor:
        # Rounding the gathered rows is the same thing as gathering from the rounded matrix,
        # because the blocks run along the hidden dimension, and it costs a great deal less.
        rows = self.weights[self._names["model.embed_tokens.weight"]][tokens]
        return _StraightThroughQuantize.apply(rows, self.output_format)

    def quantized_reference(self) -> Qwen3Reference:
        """What the engine will actually run: the weights on their quantization grid."""
        from .quant_torch import fake_quantize

        state = {}
        for name, key in self._names.items():
            tensor = self.weights[key].detach()
            if name in ("model.embed_tokens.weight", "lm_head.weight"):
                state[name] = fake_quantize(tensor, self.output_format)
            elif name.endswith("_proj.weight"):
                state[name] = fake_quantize(tensor, self.weight_format)
            else:
                state[name] = tensor
        return Qwen3Reference(self.config, state)


class _StraightThroughQuantize(torch.autograd.Function):
    """Quantize going forward, pass the gradient straight through coming back."""

    @staticmethod
    def forward(ctx, weight: Tensor, fmt: str) -> Tensor:
        from .quant_torch import fake_quantize

        return fake_quantize(weight, fmt)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        # The derivative of rounding is zero almost everywhere and undefined at the steps, so the
        # estimator pretends it was the identity. Without that, nothing would learn at all.
        return gradient, None
