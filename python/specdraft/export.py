"""Write the single weights file the C++ engine memory-maps (plan §7.3).

Layout::

    offset 0   magic "SDM1" | uint32 version | uint32 json_bytes | uint32 reserved
    offset 16  JSON metadata: config, vocab limit, and a tensor directory
    then       tensor data, each at a 64-byte aligned offset

Two things happen here beyond quantizing:

* **Matrices are fused.** Q, K and V become one matrix and gate and up become another.
  Each output row is still an independent dot product, so the arithmetic is unchanged, but
  the engine gets fewer places where threads have to wait for each other.
* **The embedding is stored once.** Qwen3 ties it to the output layer, and the engine uses
  the same quantized matrix for the token lookup and for the final projection.

Norm weights, including the per-head q/k norms, stay fp32, as in the engine.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import quant
from .reference import Qwen3Config

MAGIC = b"SDM1"
VERSION = 1
HEADER_BYTES = 16
ALIGNMENT = 64

# Formats a tensor may be stored in. "fp32" is for bring-up and debugging.
WEIGHT_FORMATS = ("q4", "q8", "fp32")


@dataclass(frozen=True)
class TensorEntry:
    name: str
    format: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    def as_json(self) -> dict:
        return {
            "name": self.name,
            "format": self.format,
            "shape": list(self.shape),
            "offset": self.offset,
            "nbytes": self.nbytes,
        }


def _encode(array: np.ndarray, fmt: str) -> bytes:
    array = np.ascontiguousarray(array, dtype=np.float32)
    if fmt == "fp32":
        return array.tobytes()
    if fmt == "q4":
        return quant.quantize_q4(array)
    if fmt == "q8":
        return quant.quantize_q8(array)
    raise ValueError(f"unknown format {fmt!r}; expected one of {WEIGHT_FORMATS}")


def decode(blob: bytes, fmt: str, shape: tuple[int, ...]) -> np.ndarray:
    """Inverse of ``_encode``, for tests and tooling."""
    if fmt == "fp32":
        values = np.frombuffer(blob, dtype=np.float32)
    elif fmt == "q4":
        values = quant.dequantize_q4(blob)
    elif fmt == "q8":
        values = quant.dequantize_q8(blob)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    return values.reshape(shape)


def _to_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):  # a torch tensor
        return tensor.detach().to("cpu").float().numpy()
    return np.ascontiguousarray(tensor, dtype=np.float32)


def plan_tensors(
    state_dict: dict,
    config: Qwen3Config,
    weight_format: str = "q4",
    output_format: str | None = None,
) -> list[tuple[str, np.ndarray, str]]:
    """The tensors to write, in file order: (name, values, format)."""
    np_ = _to_numpy
    output_format = weight_format if output_format is None else output_format
    items: list[tuple[str, np.ndarray, str]] = [
        ("token_embd", np_(state_dict["model.embed_tokens.weight"]), output_format),
    ]
    if not config.tie_word_embeddings:
        items.append(("output", np_(state_dict["lm_head.weight"]), output_format))

    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}."
        qkv = np.concatenate(
            [
                np_(state_dict[p + "self_attn.q_proj.weight"]),
                np_(state_dict[p + "self_attn.k_proj.weight"]),
                np_(state_dict[p + "self_attn.v_proj.weight"]),
            ],
            axis=0,
        )
        gate_up = np.concatenate(
            [np_(state_dict[p + "mlp.gate_proj.weight"]), np_(state_dict[p + "mlp.up_proj.weight"])],
            axis=0,
        )
        items += [
            (f"blk.{i}.attn_norm", np_(state_dict[p + "input_layernorm.weight"]), "fp32"),
            (f"blk.{i}.qkv", qkv, weight_format),
            (f"blk.{i}.q_norm", np_(state_dict[p + "self_attn.q_norm.weight"]), "fp32"),
            (f"blk.{i}.k_norm", np_(state_dict[p + "self_attn.k_norm.weight"]), "fp32"),
            (f"blk.{i}.attn_out", np_(state_dict[p + "self_attn.o_proj.weight"]), weight_format),
            (f"blk.{i}.ffn_norm", np_(state_dict[p + "post_attention_layernorm.weight"]), "fp32"),
            (f"blk.{i}.gate_up", gate_up, weight_format),
            (f"blk.{i}.ffn_down", np_(state_dict[p + "mlp.down_proj.weight"]), weight_format),
        ]
    items.append(("output_norm", np_(state_dict["model.norm.weight"]), "fp32"))
    return items


def write_model(
    path: str | Path,
    state_dict: dict,
    config: Qwen3Config,
    weight_format: str = "q4",
    output_format: str | None = None,
    vocab_limit: int | None = None,
) -> list[TensorEntry]:
    """Quantize, fuse and write the file. Returns the tensor directory."""
    if weight_format not in WEIGHT_FORMATS:
        raise ValueError(f"unknown format {weight_format!r}")

    items = plan_tensors(state_dict, config, weight_format, output_format)
    encoded = [(name, _encode(values, fmt), fmt, values.shape) for name, values, fmt in items]

    # Two passes: the directory has to know each offset, and the offsets depend on how
    # long the JSON is, so lay the tensors out first and then place them after the header.
    entries: list[TensorEntry] = []
    cursor = 0
    for name, blob, fmt, shape in encoded:
        cursor = (cursor + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        entries.append(TensorEntry(name, fmt, tuple(shape), cursor, len(blob)))
        cursor += len(blob)

    metadata = {
        "config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "tie_word_embeddings": config.tie_word_embeddings,
        },
        "vocab_limit": vocab_limit if vocab_limit is not None else config.vocab_size,
        "weight_format": weight_format,
        "output_format": output_format or weight_format,
        "tensors": [e.as_json() for e in entries],
    }
    # The offsets live in the JSON, and the JSON's length decides where the data starts, so
    # settle both together. Writing longer offsets can push the start out by one block,
    # which changes the offsets again; two or three rounds always converge.
    relative = entries
    data_start = 0
    while True:
        shifted = [
            TensorEntry(e.name, e.format, e.shape, e.offset + data_start, e.nbytes)
            for e in relative
        ]
        metadata["tensors"] = [e.as_json() for e in shifted]
        json_bytes = json.dumps(metadata).encode("utf-8")
        needed = (HEADER_BYTES + len(json_bytes) + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        if needed == data_start:
            entries = shifted
            break
        data_start = needed

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, len(json_bytes), 0))
        f.write(json_bytes)
        for entry, (_, blob, _, _) in zip(entries, encoded):
            f.write(b"\0" * (entry.offset - f.tell()))
            f.write(blob)
    return entries


# ------------------------------------------------------------------------------ reading


@dataclass
class ModelFile:
    """A written file, read back: metadata plus lazily decoded tensors."""

    path: Path
    metadata: dict
    entries: dict[str, TensorEntry]

    @property
    def config(self) -> Qwen3Config:
        return Qwen3Config.from_dict(self.metadata["config"])

    def raw(self, name: str) -> bytes:
        entry = self.entries[name]
        with self.path.open("rb") as f:
            f.seek(entry.offset)
            return f.read(entry.nbytes)

    def tensor(self, name: str) -> np.ndarray:
        entry = self.entries[name]
        return decode(self.raw(name), entry.format, entry.shape)


def read_model(path: str | Path) -> ModelFile:
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(HEADER_BYTES)
        if header[:4] != MAGIC:
            raise ValueError(f"{path} is not a specdraft model file")
        version, json_bytes, _ = struct.unpack("<III", header[4:16])
        if version != VERSION:
            raise ValueError(f"unsupported model file version {version}")
        metadata = json.loads(f.read(json_bytes).decode("utf-8"))

    entries = {
        t["name"]: TensorEntry(t["name"], t["format"], tuple(t["shape"]), t["offset"], t["nbytes"])
        for t in metadata["tensors"]
    }
    return ModelFile(path=path, metadata=metadata, entries=entries)
