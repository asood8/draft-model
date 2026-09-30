"""The reference model with the engine's rounding applied: the twin (plan §7.2).

The engine does three things PyTorch normally does not: it stores weights in 4-bit or
8-bit blocks, it quantizes every activation vector to 8-bit before each matrix multiply,
and it keeps the KV cache in fp16. The twin does all three, so:

* perplexity numbers describe what the engine actually runs, not an idealized version;
* acceptance measured on a GPU predicts what the engine will accept, which is what makes
  the cheap offline evaluation trustworthy;
* the C++ forward pass has something exact to be compared against.

What the twin does *not* reproduce is summation order. The engine accumulates integer
products per block and scales them; the twin multiplies dequantized floats. The values
being multiplied are identical, so the two agree to float rounding, which is the level the
layer-by-layer comparison in Milestone 2 uses.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from .quant_torch import fake_quantize
from .reference import Qwen3Config, Qwen3Reference, load_safetensors

# Everything else (RMSNorm weights, including the per-head q/k norms) stays in fp32, as in
# the engine.
_QUANTIZED_SUFFIXES = ("_proj.weight",)
_OUTPUT_LAYER_NAMES = ("model.embed_tokens.weight", "lm_head.weight")


class QuantizedTwin(Qwen3Reference):
    def __init__(
        self,
        config: Qwen3Config,
        state_dict: dict[str, Tensor],
        weight_format: str | None = "q4",
        output_format: str | None = None,
        activation_format: str | None = "a8",
        kv_dtype: torch.dtype | None = torch.float16,
    ) -> None:
        """
        weight_format: format for every projection matrix, or None to leave them alone.
        output_format: format for the embedding/output matrix; defaults to weight_format.
            Keeping this at "q8" while the body is "q4" is the middle option of plan §5.4,
            since this one matrix is about a quarter of the 0.6B's weights.
        activation_format: applied to the input of every matrix multiply, or None.
        kv_dtype: the cache dtype; the engine uses fp16.
        """
        self.weight_format = weight_format
        self.output_format = weight_format if output_format is None else output_format
        self.activation_format = activation_format
        self.kv_dtype = kv_dtype
        super().__init__(config, self._quantize_weights(state_dict))

    def _quantize_weights(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for name, tensor in state_dict.items():
            if name in _OUTPUT_LAYER_NAMES:
                fmt = self.output_format
            elif name.endswith(_QUANTIZED_SUFFIXES):
                fmt = self.weight_format
            else:
                fmt = None
            out[name] = tensor if fmt is None else fake_quantize(tensor, fmt)
        return out

    # -- the engine's arithmetic ----------------------------------------------------

    def matmul(self, x: Tensor, weight: Tensor, name: str) -> Tensor:
        """Quantize the activation vector, then multiply, as every engine kernel does."""
        if self.activation_format is not None:
            x = fake_quantize(x, self.activation_format)
        return x @ weight.transpose(0, 1)

    def new_cache(self, max_positions: int, dtype: torch.dtype | None = None):
        return super().new_cache(max_positions, dtype=dtype or self.kv_dtype)

    # -- constructors ---------------------------------------------------------------

    @classmethod
    def from_reference(cls, model: Qwen3Reference, **kwargs) -> "QuantizedTwin":
        return cls(model.config, model.state_dict(), **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> "QuantizedTwin":
        path = Path(path)
        config = Qwen3Config.from_pretrained(path)
        state_dict = {
            name: tensor.to(dtype=dtype, device=device)
            for name, tensor in load_safetensors(path).items()
        }
        return cls(config, state_dict, **kwargs)

    def describe(self) -> str:
        return (
            f"weights={self.weight_format or 'fp32'} output={self.output_format or 'fp32'} "
            f"activations={self.activation_format or 'fp32'} "
            f"kv={str(self.kv_dtype).replace('torch.', '') if self.kv_dtype else 'fp32'}"
        )
