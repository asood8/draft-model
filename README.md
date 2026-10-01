# draft-model (in progress)

Speculative decoding for **Qwen3** on a laptop CPU, built from scratch: a C++ inference engine,
the speculative round loop inside it, and a distilled draft model trained to feed it.

The question the project exists to answer is what speculative decoding costs on a CPU. The usual
speedup formula assumes that checking γ+1 guesses costs about as much as producing one token,
which holds on a GPU. On a CPU it does not — the weights are read once however many tokens share
the pass, but the arithmetic grows with each extra token — so the formula gets a measured term:

```
speedup(γ) = τ(γ) / (γ·c + v(γ+1)),    τ(γ) = (1 − α^(γ+1)) / (1 − α)
```

where α is how often guesses are accepted, c is a draft step over a target step, and **v(k) is
measured rather than assumed**. See [PLAN.md](PLAN.md) for the full plan and the open questions.

## What works now

| Piece | State |
|---|---|
| Qwen3 written from scratch in PyTorch | Matches Hugging Face layer by layer and token for token |
| 4-bit and 8-bit block formats | Three implementations (C++, NumPy, torch), **byte-identical** |
| C++ engine | Memory-mapped weights, AVX-VNNI kernels, thread pool, fp16 KV cache |
| k-token verification kernel | One pass over the weights for several tokens, **bit-identical** to one token at a time |
| Speculative decoding | Inside the engine; greedy output matches plain decoding exactly |
| Prompt lookup drafting | Copies from earlier text, zero model work, output still exact |
| Distillation | Five losses, a trainable draft, the training loop, and the data pipeline |
| 600+ tests | Statistical, bit-exactness, and end-to-end |

## Measured so far

On an i7-13620H (6 performance cores + 4 efficiency cores, 16 GB DDR4-3200) with Qwen3-0.6B at
4 bits. These are rehearsal numbers: the real target, Qwen3-4B, is not in place yet, and the
machine is a noisy instrument (see plan §13).

- **Read bandwidth 37–39 GB/s**, about 75% of the theoretical 51.2, which sets the speed ceiling.
- **Decode went from 4.4 to about 30–50 tokens/second**, roughly half the weights-only ceiling,
  via a thread pool, SIMD attention, and inlining the fp16 conversions.
- **Efficiency cores hurt.** Six performance cores beat all ten, because a fixed split leaves the
  fast cores waiting; dynamic chunks recover most of the loss but do not overtake.
- **4-bit costs real quality on a 0.6B**: WikiText-2 perplexity 28.5 → 32.2, while 8-bit is free
  and an 8-bit output layer buys back a fifth of the loss.
- **The quantization twin predicts the engine** to 98.9% top-1 agreement and 0.023 mean TVD,
  which is what lets training recipes be ranked on a GPU.

## Layout

```
engine/      C++: weights file, forward pass, kernels, thread pool, sampling, the round loop
python/      reference model, quantization twin, export, offline metrics, distillation
scripts/     fetch, export, benchmark, measure v(k), generate data, train
tests/       600+ tests, all runnable on a CPU
results/     measurements, as JSON
```

## Building

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[torch,dev]"   # builds the C++ extension
.venv/Scripts/python -m pytest -q -m "not slow"                       # the fast suite
```

Needs a C++20 compiler with AVX2/FMA/F16C. AVX-VNNI is used when present and detected at runtime.

## Trying it

```bash
python scripts/fetch_model.py Qwen/Qwen3-0.6B
python scripts/export_model.py models/Qwen3-0.6B --format q4
python scripts/bench_engine.py models/Qwen3-0.6B-q4.sdm --quick
python scripts/measure_vk.py models/Qwen3-0.6B-q4.sdm
```
