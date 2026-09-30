"""The engine's number formats, mirrored in NumPy.

This module and ``engine/src/quant.cpp`` must produce byte-identical output;
``tests/test_quant.py`` checks that on random and adversarial inputs. Keeping a Python
mirror is what lets the PyTorch reference model act as a twin of the engine (plan §7.2),
so acceptance rates measured on a GPU predict what the engine will do.

Block layouts (32 values per block, one scale per block):

===========  =================================  =========  ====================
Format       Layout                             Bytes      Used for
===========  =================================  =========  ====================
``q4``       fp16 scale + 16 packed bytes       18         4-bit weights
``q8``       fp16 scale + 32 int8               34         8-bit weights
``a8``       fp32 scale + 32 int8               36         8-bit activations
===========  =================================  =========  ====================

In ``q4``, byte *i* holds weight *i* in its low nibble and weight *i + 16* in its high
nibble, and a stored nibble ``q`` means the value ``(q - 8) * scale``.
"""

from __future__ import annotations

import numpy as np

QK = 32
BLOCK_BYTES = {"q4": 18, "q8": 34, "a8": 36}

_F32 = np.float32


def _as_blocks(x) -> np.ndarray:
    """Flatten to float32 and reshape into (nblocks, QK)."""
    flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    if flat.size == 0 or flat.size % QK:
        raise ValueError("input length must be a positive multiple of 32")
    return flat.reshape(-1, QK)


def _reciprocal(scale: np.ndarray) -> np.ndarray:
    """1/scale, and 0 where the scale is 0 (an all-zero block)."""
    inv = np.zeros_like(scale, dtype=np.float32)
    np.divide(_F32(1.0), scale, out=inv, where=scale != 0)
    return inv


def _scale_bytes(scale: np.ndarray, dtype: str) -> np.ndarray:
    """Little-endian raw bytes of one scale per block, shaped (nblocks, itemsize)."""
    return np.ascontiguousarray(scale, dtype=dtype).view(np.uint8).reshape(len(scale), -1)


def quantize_q4(x) -> bytes:
    """4-bit weights: scale = fp16(v_max / -8), q = clamp(floor(x/scale + 8.5), 0, 15).

    v_max is the value of largest magnitude in the block, with its sign, so that value
    maps to nibble 0 and is reproduced almost exactly.
    """
    blocks = _as_blocks(x)
    largest = np.abs(blocks).argmax(axis=1)
    v_max = blocks[np.arange(blocks.shape[0]), largest]
    scale16 = (v_max / _F32(-8.0)).astype(np.float16)
    inv = _reciprocal(scale16.astype(np.float32))

    q = np.floor(blocks * inv[:, None] + _F32(8.5))
    q = np.clip(q, 0, 15).astype(np.uint8)
    packed = q[:, :16] | (q[:, 16:] << 4)

    out = np.empty((blocks.shape[0], BLOCK_BYTES["q4"]), dtype=np.uint8)
    out[:, :2] = _scale_bytes(scale16, "<f2")
    out[:, 2:] = packed
    return out.tobytes()


def quantize_q8(x) -> bytes:
    """8-bit weights: scale = fp16(max|x| / 127), q = clamp(rint(x/scale), -127, 127)."""
    blocks = _as_blocks(x)
    scale16 = (np.abs(blocks).max(axis=1) / _F32(127.0)).astype(np.float16)
    inv = _reciprocal(scale16.astype(np.float32))

    q = np.clip(np.rint(blocks * inv[:, None]), -127, 127).astype(np.int8)

    out = np.empty((blocks.shape[0], BLOCK_BYTES["q8"]), dtype=np.uint8)
    out[:, :2] = _scale_bytes(scale16, "<f2")
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def quantize_a8(x) -> bytes:
    """8-bit activations: the same as q8 but the scale stays fp32."""
    blocks = _as_blocks(x)
    scale = (np.abs(blocks).max(axis=1) / _F32(127.0)).astype(np.float32)
    inv = _reciprocal(scale)

    q = np.clip(np.rint(blocks * inv[:, None]), -127, 127).astype(np.int8)

    out = np.empty((blocks.shape[0], BLOCK_BYTES["a8"]), dtype=np.uint8)
    out[:, :4] = _scale_bytes(scale, "<f4")
    out[:, 4:] = q.view(np.uint8)
    return out.tobytes()


def _raw_blocks(blob: bytes, fmt: str) -> np.ndarray:
    size = BLOCK_BYTES[fmt]
    raw = np.frombuffer(blob, dtype=np.uint8)
    if raw.size == 0 or raw.size % size:
        raise ValueError(f"blob size is not a whole number of {fmt} blocks")
    return raw.reshape(-1, size)


def dequantize_q4(blob: bytes) -> np.ndarray:
    raw = _raw_blocks(blob, "q4")
    scale = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    nibbles = raw[:, 2:]
    low = (nibbles & 0x0F).astype(np.int16) - 8  # weights 0..15
    high = (nibbles >> 4).astype(np.int16) - 8  # weights 16..31
    q = np.concatenate([low, high], axis=1).astype(np.float32)
    return (q * scale[:, None]).reshape(-1)


def dequantize_q8(blob: bytes) -> np.ndarray:
    raw = _raw_blocks(blob, "q8")
    scale = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    q = raw[:, 2:].copy().view(np.int8).astype(np.float32)
    return (q * scale[:, None]).reshape(-1)


def dequantize_a8(blob: bytes) -> np.ndarray:
    raw = _raw_blocks(blob, "a8")
    scale = raw[:, :4].copy().view(np.float32).reshape(-1)
    q = raw[:, 4:].copy().view(np.int8).astype(np.float32)
    return (q * scale[:, None]).reshape(-1)


_QUANTIZE = {"q4": quantize_q4, "q8": quantize_q8, "a8": quantize_a8}
_DEQUANTIZE = {"q4": dequantize_q4, "q8": dequantize_q8, "a8": dequantize_a8}


def fake_quantize(x, fmt: str = "q4") -> np.ndarray:
    """Quantize and immediately dequantize along the last axis, keeping the shape.

    This is what the twin uses: blocks run along the last axis, which for a weight
    matrix ``[out_features, in_features]`` means along the input dimension, exactly as
    the engine's kernels read it.
    """
    array = np.ascontiguousarray(x, dtype=np.float32)
    if array.shape[-1] % QK:
        raise ValueError("last axis must be a multiple of 32")
    flat = _DEQUANTIZE[fmt](_QUANTIZE[fmt](array))
    return flat.reshape(array.shape)
