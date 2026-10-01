"""Write the single weights file the C++ engine memory-maps (plan §7.3).

The layout is fixed-size binary so the engine can read it with ``memcpy`` and no parser::

    0    magic "SDM2" | uint32 version | uint32 tensor_count | uint32 data_start
    16   config: 9 x uint32 then 2 x float64 (see CONFIG_FIELDS)
    128  directory: tensor_count entries of 72 bytes
         char name[40] | uint32 format | uint32 ndim | uint32 dim0 | uint32 dim1
                       | uint64 offset | uint64 nbytes
    ...  tensor payloads, each at a 64-byte aligned offset

Two things happen here beyond quantizing:

* **Matrices are fused.** Q, K and V become one matrix and gate and up become another. Each
  output row is still an independent dot product, so the arithmetic is unchanged, but the
  engine gets fewer places where threads have to wait for each other.
* **The embedding is stored once.** Qwen3 ties it to the output layer, and the engine uses
  the same quantized matrix for the token lookup and for the final projection.

Norm weights, including the per-head q/k norms, stay fp32, as in the engine.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import quant
from .reference import Qwen3Config

MAGIC = b"SDM2"
VERSION = 2
HEADER_BYTES = 128
DIRECTORY_ENTRY_BYTES = 72
NAME_BYTES = 40
ALIGNMENT = 64

WEIGHT_FORMATS = ("q4", "q8", "fp32")
FORMAT_CODES = {"fp32": 0, "q4": 1, "q8": 2}
FORMAT_NAMES = {code: name for name, code in FORMAT_CODES.items()}

# The engine reads these in this order; keep both sides in step.
CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_limit",
    "tie_word_embeddings",
)


@dataclass(frozen=True)
class TensorEntry:
    name: str
    format: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


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
    output_format = weight_format if output_format is None else output_format
    items: list[tuple[str, np.ndarray, str]] = [
        ("token_embd", _to_numpy(state_dict["model.embed_tokens.weight"]), output_format),
    ]
    if not config.tie_word_embeddings:
        items.append(("output", _to_numpy(state_dict["lm_head.weight"]), output_format))

    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}."
        qkv = np.concatenate(
            [
                _to_numpy(state_dict[p + "self_attn.q_proj.weight"]),
                _to_numpy(state_dict[p + "self_attn.k_proj.weight"]),
                _to_numpy(state_dict[p + "self_attn.v_proj.weight"]),
            ],
            axis=0,
        )
        gate_up = np.concatenate(
            [
                _to_numpy(state_dict[p + "mlp.gate_proj.weight"]),
                _to_numpy(state_dict[p + "mlp.up_proj.weight"]),
            ],
            axis=0,
        )
        items += [
            (f"blk.{i}.attn_norm", _to_numpy(state_dict[p + "input_layernorm.weight"]), "fp32"),
            (f"blk.{i}.qkv", qkv, weight_format),
            (f"blk.{i}.q_norm", _to_numpy(state_dict[p + "self_attn.q_norm.weight"]), "fp32"),
            (f"blk.{i}.k_norm", _to_numpy(state_dict[p + "self_attn.k_norm.weight"]), "fp32"),
            (f"blk.{i}.attn_out", _to_numpy(state_dict[p + "self_attn.o_proj.weight"]), weight_format),
            (
                f"blk.{i}.ffn_norm",
                _to_numpy(state_dict[p + "post_attention_layernorm.weight"]),
                "fp32",
            ),
            (f"blk.{i}.gate_up", gate_up, weight_format),
            (f"blk.{i}.ffn_down", _to_numpy(state_dict[p + "mlp.down_proj.weight"]), weight_format),
        ]
    items.append(("output_norm", _to_numpy(state_dict["model.norm.weight"]), "fp32"))
    return items


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


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
        raise ValueError(f"unknown format {weight_format!r}; expected one of {WEIGHT_FORMATS}")

    items = plan_tensors(state_dict, config, weight_format, output_format)
    payloads = [(name, _encode(values, fmt), fmt, tuple(values.shape)) for name, values, fmt in items]

    data_start = _align(HEADER_BYTES + DIRECTORY_ENTRY_BYTES * len(payloads))
    entries: list[TensorEntry] = []
    cursor = data_start
    for name, blob, fmt, shape in payloads:
        if len(name.encode("utf-8")) >= NAME_BYTES:
            raise ValueError(f"tensor name too long: {name}")
        if not 1 <= len(shape) <= 2:
            raise ValueError(f"{name}: only vectors and matrices are supported")
        cursor = _align(cursor)
        entries.append(TensorEntry(name, fmt, shape, cursor, len(blob)))
        cursor += len(blob)

    values = {
        **{field: getattr(config, field, None) for field in CONFIG_FIELDS},
        "vocab_limit": vocab_limit if vocab_limit is not None else config.vocab_size,
        "tie_word_embeddings": int(config.tie_word_embeddings),
    }
    header = bytearray(HEADER_BYTES)
    header[0:4] = MAGIC
    struct.pack_into("<III", header, 4, VERSION, len(entries), data_start)
    struct.pack_into(
        "<9I2d",
        header,
        16,
        *[int(values[field]) for field in CONFIG_FIELDS],
        float(config.rms_norm_eps),
        float(config.rope_theta),
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        for entry in entries:
            shape = entry.shape + (1,) if len(entry.shape) == 1 else entry.shape
            f.write(
                struct.pack(
                    f"<{NAME_BYTES}sIIIIQQ",
                    entry.name.encode("utf-8"),
                    FORMAT_CODES[entry.format],
                    len(entry.shape),
                    shape[0],
                    shape[1],
                    entry.offset,
                    entry.nbytes,
                )
            )
        for entry, (_, blob, _, _) in zip(entries, payloads):
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
        if len(header) < HEADER_BYTES or header[:4] != MAGIC:
            raise ValueError(f"{path} is not a specdraft model file")
        version, count, data_start = struct.unpack_from("<III", header, 4)
        if version != VERSION:
            raise ValueError(f"unsupported model file version {version}")
        fields = struct.unpack_from("<9I2d", header, 16)
        directory = f.read(DIRECTORY_ENTRY_BYTES * count)

    config = dict(zip(CONFIG_FIELDS, fields[:9]))
    config["rms_norm_eps"], config["rope_theta"] = fields[9], fields[10]
    config["tie_word_embeddings"] = bool(config["tie_word_embeddings"])
    vocab_limit = config.pop("vocab_limit")

    entries: dict[str, TensorEntry] = {}
    for i in range(count):
        raw_name, code, ndim, dim0, dim1, offset, nbytes = struct.unpack_from(
            f"<{NAME_BYTES}sIIIIQQ", directory, i * DIRECTORY_ENTRY_BYTES
        )
        name = raw_name.split(b"\0", 1)[0].decode("utf-8")
        shape = (dim0,) if ndim == 1 else (dim0, dim1)
        entries[name] = TensorEntry(name, FORMAT_NAMES[code], shape, offset, nbytes)

    metadata = {
        "version": version,
        "config": config,
        "vocab_limit": vocab_limit,
        "data_start": data_start,
        # Not stored separately: the formats are visible in the directory itself.
        "weight_format": entries["blk.0.qkv"].format if "blk.0.qkv" in entries else None,
        "output_format": entries["token_embd"].format if "token_embd" in entries else None,
    }
    return ModelFile(path=path, metadata=metadata, entries=entries)
