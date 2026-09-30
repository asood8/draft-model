"""The engine's number formats again, in PyTorch.

``specdraft.quant`` is the NumPy definition and ``engine/src/quant.cpp`` is the C++ one.
This module exists because the twin applies quantization on every forward pass, and
because distillation against a 4-bit teacher has to run on a GPU. It produces bitwise the
same values as the other two, which ``tests/test_quant_torch.py`` checks.

Only fake quantization lives here: quantize and immediately dequantize. Packing into
bytes is the export script's job.
"""

from __future__ import annotations

import torch
from torch import Tensor

QK = 32
FORMATS = ("q4", "q8", "a8")


def _as_blocks(x: Tensor) -> tuple[Tensor, torch.Size]:
    if x.shape[-1] % QK:
        raise ValueError(f"last axis must be a multiple of {QK}, got {x.shape[-1]}")
    return x.reshape(*x.shape[:-1], x.shape[-1] // QK, QK).to(torch.float32), x.shape


def _reciprocal(scale: Tensor) -> Tensor:
    """1/scale, and 0 for an all-zero block.

    The substitution must not touch the sign: q4 scales are negative whenever the block's
    largest value is positive, so clamping the denominator would flip them.
    """
    nonzero = scale != 0
    safe = torch.where(nonzero, scale, torch.ones_like(scale))
    return torch.where(nonzero, 1.0 / safe, torch.zeros_like(scale))


def fake_quantize_q4(x: Tensor) -> Tensor:
    """4-bit weights: scale = fp16(v_max / -8), q = clamp(floor(x/scale + 8.5), 0, 15)."""
    blocks, shape = _as_blocks(x)
    largest = blocks.abs().argmax(dim=-1, keepdim=True)
    v_max = blocks.gather(-1, largest)
    scale = (v_max / -8.0).to(torch.float16).to(torch.float32)  # stored as fp16
    q = torch.floor(blocks * _reciprocal(scale) + 8.5).clamp_(0, 15)
    return ((q - 8.0) * scale).reshape(shape).to(x.dtype)


def _fake_quantize_int8(x: Tensor, fp16_scale: bool) -> Tensor:
    blocks, shape = _as_blocks(x)
    scale = blocks.abs().amax(dim=-1, keepdim=True) / 127.0
    if fp16_scale:
        scale = scale.to(torch.float16).to(torch.float32)
    q = torch.round(blocks * _reciprocal(scale)).clamp_(-127, 127)
    return (q * scale).reshape(shape).to(x.dtype)


def fake_quantize_q8(x: Tensor) -> Tensor:
    """8-bit weights: an fp16 scale per block."""
    return _fake_quantize_int8(x, fp16_scale=True)


def fake_quantize_a8(x: Tensor) -> Tensor:
    """8-bit activations: the scale stays fp32, as it does in the engine."""
    return _fake_quantize_int8(x, fp16_scale=False)


_BY_NAME = {"q4": fake_quantize_q4, "q8": fake_quantize_q8, "a8": fake_quantize_a8}


def fake_quantize(x: Tensor, fmt: str) -> Tensor:
    """Round ``x`` to one of the engine's formats along its last axis.

    For a weight matrix [out_features, in_features] that means along the input dimension,
    exactly how the engine's kernels read it.
    """
    try:
        return _BY_NAME[fmt](x)
    except KeyError:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}") from None
