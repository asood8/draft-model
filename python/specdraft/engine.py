"""Drive the C++ engine with the Python decoder.

``speculative.py`` and ``offline.py`` only need ``forward(tokens, cache, ...)`` and
``new_cache(n)``, so wrapping the engine in that shape means the same round loop, the same
acceptance rule and the same metrics work on the engine as on the PyTorch reference. The
tests for greedy equivalence and the sampling distribution then apply to the engine too.

The engine owns one KV cache per model, so the cache object here is only a handle: it
reports the engine's position and rewinds it. Two models means two ``EngineModel``
instances, which is what speculative decoding wants anyway.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from . import _engine


class EngineCache:
    """A handle with the same surface as ``reference.KVCache``; the engine holds the data."""

    def __init__(self, model: "EngineModel", max_positions: int) -> None:
        self._model = model
        self.max_positions = max_positions

    @property
    def pos(self) -> int:
        return self._model.raw.pos

    def rewind_to(self, position: int) -> None:
        self._model.raw.set_pos(max(0, min(self.pos, position)))

    def reset(self) -> None:
        self._model.raw.reset()


class EngineModel:
    """The C++ engine behind the interface the Python decoder expects."""

    def __init__(self, path: str | Path, max_positions: int = 2048) -> None:
        self.raw = _engine.Model(str(path), max_positions=max_positions)
        self.path = Path(path)
        self.device = torch.device("cpu")

    # -- what the decoder calls ----------------------------------------------------

    def new_cache(self, max_positions: int | None = None, dtype: torch.dtype | None = None) -> EngineCache:
        """Start a fresh sequence. The engine keeps its KV cache in fp16 regardless."""
        wanted = self.raw.max_positions if max_positions is None else max_positions
        if wanted > self.raw.max_positions:
            raise ValueError(
                f"{self.path.name} was opened for {self.raw.max_positions} positions, needs {wanted}"
            )
        self.raw.reset()
        return EngineCache(self, wanted)

    def forward(
        self,
        tokens: Tensor,
        cache: EngineCache | None = None,
        capture: list[Tensor] | None = None,
        only_last_logits: bool = False,
        hidden_only: bool = False,
    ) -> Tensor:
        """Run k tokens and return logits [k or 1, vocab_limit].

        ``cache`` is accepted for interface compatibility; the engine always uses its own.
        ``hidden_only`` is not available, since the engine's output layer is not separable
        from its forward pass the way the reference's is.
        """
        if hidden_only:
            raise NotImplementedError("the engine cannot return hidden states instead of logits")
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1-D sequence")
        if tokens.numel() == 0:
            # Usually this means one EngineModel is being used as both target and draft: they
            # would share the engine's single KV cache, so each thinks the other's tokens are
            # already in it. Open a second EngineModel on the same file instead.
            raise ValueError(
                "no tokens to feed; two roles cannot share one EngineModel, since it owns one KV cache"
            )
        ids = tokens.detach().to("cpu", torch.int32).numpy()

        if capture is not None:
            captured = self.raw.forward_capture(ids)
            for layer in range(captured.shape[0]):
                capture.append(torch.from_numpy(captured[layer]))
            # forward_capture computes no logits, so run again for them; only tests do this.
            self.raw.set_pos(self.raw.pos - len(ids))
        logits = self.raw.forward(ids, all_logits=not only_last_logits)
        return torch.from_numpy(logits)

    # -- information ---------------------------------------------------------------

    @property
    def config(self) -> dict:
        return self.raw.config

    @property
    def vocab_limit(self) -> int:
        return int(self.raw.config["vocab_limit"])

    @property
    def pos(self) -> int:
        return self.raw.pos

    def __repr__(self) -> str:
        config = self.raw.config
        return (
            f"EngineModel({self.path.name}, layers={config['num_hidden_layers']}, "
            f"hidden={config['hidden_size']}, vocab_limit={config['vocab_limit']})"
        )
