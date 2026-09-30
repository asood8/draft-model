"""Qwen3 written from scratch in PyTorch (plan §7.1).

This is the oracle for everything else: the C++ engine is tested layer by layer against
it, the quantization twin is this model with the engine's rounding applied, and the
offline acceptance metrics run on it.

Two deliberate choices keep it close to the engine:

* One sequence at a time, no batch dimension, exactly like decoding at batch size 1.
* ``forward(tokens, cache)`` takes *k* tokens at once. Normal decoding passes one token,
  speculative verification passes γ+1, and prompt processing passes many.

Only ``torch`` and ``safetensors`` are needed, never ``transformers``; that keeps the
oracle independent of the library it is checked against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

# ---------------------------------------------------------------------------- config


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def group_size(self) -> int:
        """Query heads per key/value head (2 for the 0.6B, 4 for the 4B)."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, raw: dict) -> "Qwen3Config":
        # head_dim is given in the config and is NOT hidden_size // num_attention_heads
        # for any Qwen3 model (128 vs 64 for the 0.6B, 128 vs 80 for the 4B).
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"]),
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "Qwen3Config":
        return cls.from_dict(json.loads((Path(path) / "config.json").read_text(encoding="utf-8")))


# ----------------------------------------------------------------------------- cache


class KVCache:
    """Keys and values for every layer, plus one position counter.

    Laid out as ``[layer, position, kv_head, head_dim]``, which mirrors the engine's
    per-layer ``[kv_head][position][head_dim]`` closely enough for parity testing. As in
    the engine, rolling back rejected draft tokens only moves ``pos``: attention reads up
    to ``pos``, and stale entries beyond it get overwritten later.
    """

    def __init__(
        self,
        config: Qwen3Config,
        max_positions: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        shape = (
            config.num_hidden_layers,
            max_positions,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.keys = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(shape, dtype=dtype, device=device)
        self.max_positions = max_positions
        self.pos = 0

    def write(self, layer: int, start: int, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        """Store this forward pass's keys/values and return everything up to their end."""
        end = start + keys.shape[0]
        if end > self.max_positions:
            raise ValueError(f"cache holds {self.max_positions} positions, needed {end}")
        self.keys[layer, start:end] = keys.to(self.keys.dtype)
        self.values[layer, start:end] = values.to(self.values.dtype)
        return self.keys[layer, :end], self.values[layer, :end]

    def rewind_to(self, pos: int) -> None:
        self.pos = max(0, min(self.pos, pos))

    def reset(self) -> None:
        self.pos = 0


# ------------------------------------------------------------------------- the model


@dataclass
class LayerWeights:
    input_norm: Tensor
    q_proj: Tensor
    k_proj: Tensor
    v_proj: Tensor
    o_proj: Tensor
    q_norm: Tensor
    k_norm: Tensor
    post_attn_norm: Tensor
    gate_proj: Tensor
    up_proj: Tensor
    down_proj: Tensor


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMSNorm with the reduction in float32, as Qwen3 does it."""
    dtype = x.dtype
    x32 = x.to(torch.float32)
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the two halves of each head, which is how Hugging Face lays out RoPE."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: [k, heads, head_dim]; cos/sin: [k, head_dim]."""
    return x * cos[:, None, :] + _rotate_half(x) * sin[:, None, :]


class Qwen3Reference:
    def __init__(self, config: Qwen3Config, state_dict: dict[str, Tensor]) -> None:
        self.config = config
        get = state_dict.__getitem__

        self.embed_tokens = get("model.embed_tokens.weight")
        self.final_norm = get("model.norm.weight")
        self.lm_head = (
            self.embed_tokens if config.tie_word_embeddings else get("lm_head.weight")
        )

        self.layers: list[LayerWeights] = []
        for i in range(config.num_hidden_layers):
            p = f"model.layers.{i}."
            self.layers.append(
                LayerWeights(
                    input_norm=get(p + "input_layernorm.weight"),
                    q_proj=get(p + "self_attn.q_proj.weight"),
                    k_proj=get(p + "self_attn.k_proj.weight"),
                    v_proj=get(p + "self_attn.v_proj.weight"),
                    o_proj=get(p + "self_attn.o_proj.weight"),
                    q_norm=get(p + "self_attn.q_norm.weight"),
                    k_norm=get(p + "self_attn.k_norm.weight"),
                    post_attn_norm=get(p + "post_attention_layernorm.weight"),
                    gate_proj=get(p + "mlp.gate_proj.weight"),
                    up_proj=get(p + "mlp.up_proj.weight"),
                    down_proj=get(p + "mlp.down_proj.weight"),
                )
            )

    # -- loading ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> "Qwen3Reference":
        path = Path(path)
        config = Qwen3Config.from_pretrained(path)
        state_dict = {
            name: tensor.to(dtype=dtype, device=device)
            for name, tensor in load_safetensors(path).items()
        }
        return cls(config, state_dict)

    def state_dict(self) -> dict[str, Tensor]:
        """Hugging Face weight names again, for export and for making modified copies."""
        state = {
            "model.embed_tokens.weight": self.embed_tokens,
            "model.norm.weight": self.final_norm,
        }
        if not self.config.tie_word_embeddings:
            state["lm_head.weight"] = self.lm_head
        for i, layer in enumerate(self.layers):
            p = f"model.layers.{i}."
            state.update(
                {
                    p + "input_layernorm.weight": layer.input_norm,
                    p + "post_attention_layernorm.weight": layer.post_attn_norm,
                    p + "self_attn.q_proj.weight": layer.q_proj,
                    p + "self_attn.k_proj.weight": layer.k_proj,
                    p + "self_attn.v_proj.weight": layer.v_proj,
                    p + "self_attn.o_proj.weight": layer.o_proj,
                    p + "self_attn.q_norm.weight": layer.q_norm,
                    p + "self_attn.k_norm.weight": layer.k_norm,
                    p + "mlp.gate_proj.weight": layer.gate_proj,
                    p + "mlp.up_proj.weight": layer.up_proj,
                    p + "mlp.down_proj.weight": layer.down_proj,
                }
            )
        return state

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.dtype

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.device

    def new_cache(self, max_positions: int, dtype: torch.dtype | None = None) -> KVCache:
        """A cache for this model. Pass dtype=torch.float16 for the engine's fp16 cache."""
        return KVCache(
            self.config,
            max_positions,
            dtype=self.dtype if dtype is None else dtype,
            device=self.device,
        )

    # -- hooks the twin overrides -------------------------------------------------

    def quantize_activation(self, x: Tensor, name: str) -> Tensor:
        """Identity here. The twin rounds activations exactly as the engine does."""
        return x

    def matmul(self, x: Tensor, weight: Tensor, name: str) -> Tensor:
        """One linear layer, x @ weight.T. The twin quantizes both sides first."""
        return x @ weight.transpose(0, 1)

    # -- forward ------------------------------------------------------------------

    def _rope(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        exponent = torch.arange(0, cfg.head_dim, 2, dtype=torch.float32, device=positions.device)
        inv_freq = 1.0 / (cfg.rope_theta ** (exponent / cfg.head_dim))
        freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def _attention(
        self,
        layer: LayerWeights,
        h: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Tensor | None,
        cache: KVCache | None,
        layer_idx: int,
        start: int,
    ) -> Tensor:
        cfg = self.config
        k = h.shape[0]

        q = self.matmul(h, layer.q_proj, "q_proj").view(k, cfg.num_attention_heads, cfg.head_dim)
        keys = self.matmul(h, layer.k_proj, "k_proj").view(k, cfg.num_key_value_heads, cfg.head_dim)
        values = self.matmul(h, layer.v_proj, "v_proj").view(
            k, cfg.num_key_value_heads, cfg.head_dim
        )

        # Qwen3 normalizes queries and keys per head, before RoPE.
        q = rms_norm(q, layer.q_norm, cfg.rms_norm_eps)
        keys = rms_norm(keys, layer.k_norm, cfg.rms_norm_eps)
        q = apply_rope(q, cos, sin)
        keys = apply_rope(keys, cos, sin)

        if cache is not None:
            keys, values = cache.write(layer_idx, start, keys, values)
        keys = keys.to(q.dtype)
        values = values.to(q.dtype)

        # [heads, k, head_dim] against [heads, total, head_dim], with each kv head shared
        # by group_size query heads.
        q = q.transpose(0, 1)
        keys = keys.transpose(0, 1).repeat_interleave(cfg.group_size, dim=0)
        values = values.transpose(0, 1).repeat_interleave(cfg.group_size, dim=0)

        scores = (q @ keys.transpose(1, 2)) * (cfg.head_dim**-0.5)
        if mask is not None:
            scores = scores + mask
        probs = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)
        context = (probs @ values).transpose(0, 1).reshape(k, cfg.q_dim)
        return self.matmul(context, layer.o_proj, "o_proj")

    def _mlp(self, layer: LayerWeights, h: Tensor) -> Tensor:
        gate = self.matmul(h, layer.gate_proj, "gate_proj")
        up = self.matmul(h, layer.up_proj, "up_proj")
        return self.matmul(torch.nn.functional.silu(gate) * up, layer.down_proj, "down_proj")

    @torch.no_grad()
    def forward(
        self,
        tokens: Tensor,
        cache: KVCache | None = None,
        capture: list[Tensor] | None = None,
        only_last_logits: bool = False,
        hidden_only: bool = False,
    ) -> Tensor:
        """Run k tokens and return logits of shape [k, vocab_size] (or [1, vocab_size]).

        ``capture`` collects the hidden state after every layer, which is how the C++
        engine gets compared layer by layer. ``hidden_only`` returns the final hidden
        states instead of logits, so callers can apply the output layer in chunks rather
        than materializing a [k, 151936] tensor.
        """
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1-D sequence; this reference has no batch dim")
        cfg = self.config
        k = tokens.shape[0]
        start = cache.pos if cache is not None else 0

        x = self.embed_tokens[tokens.to(self.device)]
        positions = torch.arange(start, start + k, device=self.device)
        cos, sin = self._rope(positions)
        mask = self._causal_mask(k, start)

        for i, layer in enumerate(self.layers):
            h = rms_norm(x, layer.input_norm, cfg.rms_norm_eps)
            h = self.quantize_activation(h, "attn_input")
            x = x + self._attention(layer, h, cos, sin, mask, cache, i, start)
            h = rms_norm(x, layer.post_attn_norm, cfg.rms_norm_eps)
            h = self.quantize_activation(h, "mlp_input")
            x = x + self._mlp(layer, h)
            if capture is not None:
                capture.append(x.clone())

        if cache is not None:
            cache.pos = start + k

        x = rms_norm(x, self.final_norm, cfg.rms_norm_eps)
        if only_last_logits:
            x = x[-1:]
        return x if hidden_only else self.logits_from_hidden(x)

    def logits_from_hidden(self, hidden: Tensor) -> Tensor:
        """The output layer on its own, so logits can be computed in chunks."""
        return self.matmul(self.quantize_activation(hidden, "lm_head_input"), self.lm_head, "lm_head")

    __call__ = forward

    def _causal_mask(self, k: int, start: int) -> Tensor | None:
        if k == 1:
            return None  # a single query attends to the whole cache
        total = start + k
        query_pos = torch.arange(start, total, device=self.device)[:, None]
        key_pos = torch.arange(total, device=self.device)[None, :]
        return torch.where(
            key_pos <= query_pos,
            torch.zeros((), dtype=self.dtype, device=self.device),
            torch.full((), float("-inf"), dtype=self.dtype, device=self.device),
        )

    # -- convenience --------------------------------------------------------------

    @torch.no_grad()
    def greedy_generate(self, prompt: Tensor, max_new_tokens: int, stop: set[int] | None = None) -> list[int]:
        """Plain greedy decoding, the baseline the speculative decoder must reproduce."""
        cache = self.new_cache(prompt.shape[0] + max_new_tokens)
        logits = self.forward(prompt, cache=cache, only_last_logits=True)
        out: list[int] = []
        for _ in range(max_new_tokens):
            token = int(trim_vocab(logits[-1], self.config.vocab_size).argmax())
            out.append(token)
            if stop and token in stop:
                break
            logits = self.forward(
                torch.tensor([token], device=self.device), cache=cache, only_last_logits=True
            )
        return out


# --------------------------------------------------------------------------- helpers


def load_safetensors(path: str | Path) -> dict[str, Tensor]:
    """Load one model directory's tensors, sharded or not."""
    from safetensors.torch import load_file

    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]

    tensors: dict[str, Tensor] = {}
    for shard in shards:
        tensors.update(load_file(path / shard))
    return tensors


def trim_vocab(logits: Tensor, vocab_limit: int) -> Tensor:
    """Drop the padded embedding rows.

    Qwen3 pads the embedding to 151,936 rows while the tokenizer defines about 151,669
    tokens. The extra rows hold junk that must never receive probability mass, and both
    models must be sliced the same way before their distributions are compared.
    """
    return logits[..., :vocab_limit]


def random_reference(
    config: Qwen3Config, seed: int = 0, scale: float = 0.02, dtype: torch.dtype = torch.float32
) -> Qwen3Reference:
    """A tiny randomly initialized model, for tests that must not download weights."""
    generator = torch.Generator().manual_seed(seed)

    def normal(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator, dtype=dtype) * scale

    def norm_weight(size: int) -> Tensor:
        return 1.0 + normal(size)

    state: dict[str, Tensor] = {
        "model.embed_tokens.weight": normal(config.vocab_size, config.hidden_size),
        "model.norm.weight": norm_weight(config.hidden_size),
    }
    if not config.tie_word_embeddings:
        state["lm_head.weight"] = normal(config.vocab_size, config.hidden_size)
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}."
        state.update(
            {
                p + "input_layernorm.weight": norm_weight(config.hidden_size),
                p + "post_attention_layernorm.weight": norm_weight(config.hidden_size),
                p + "self_attn.q_proj.weight": normal(config.q_dim, config.hidden_size),
                p + "self_attn.k_proj.weight": normal(config.kv_dim, config.hidden_size),
                p + "self_attn.v_proj.weight": normal(config.kv_dim, config.hidden_size),
                p + "self_attn.o_proj.weight": normal(config.hidden_size, config.q_dim),
                p + "self_attn.q_norm.weight": norm_weight(config.head_dim),
                p + "self_attn.k_norm.weight": norm_weight(config.head_dim),
                p + "mlp.gate_proj.weight": normal(config.intermediate_size, config.hidden_size),
                p + "mlp.up_proj.weight": normal(config.intermediate_size, config.hidden_size),
                p + "mlp.down_proj.weight": normal(config.hidden_size, config.intermediate_size),
            }
        )
    return Qwen3Reference(config, state)


# Every dimension is a multiple of 32 so the quantization twin can use it too, and
# head_dim deliberately differs from hidden_size // num_attention_heads, as in real Qwen3.
def perturbed_copy(model: Qwen3Reference, sigma: float = 0.01, seed: int = 0) -> Qwen3Reference:
    """A copy of a model with Gaussian noise added to every weight.

    Used as a stand-in draft in tests: it agrees with the original often but not always, so
    both the accept and the reject paths of the speculative decoder get exercised.
    """
    generator = torch.Generator().manual_seed(seed)
    state = {
        name: tensor + torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype) * sigma
        for name, tensor in model.state_dict().items()
    }
    return Qwen3Reference(model.config, state)


TINY_CONFIG = Qwen3Config(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
    tie_word_embeddings=True,
)
