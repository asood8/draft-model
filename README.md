# draft-model (in progress)

Speculative decoding for Qwen3 on a laptop CPU. The repository holds a C++ inference engine written
from scratch, with the speculative round loop implemented inside it. The distilled draft model the
repository is named after has not been trained yet; see [Not done yet](#not-done-yet).

The project measures what speculative decoding costs on a CPU. The usual speedup formula assumes
that checking γ+1 guesses costs about as much as producing one token, which holds on a GPU. It does
not hold on a CPU, where the weights are read once however many tokens share the pass while the
arithmetic grows with each additional token. The formula therefore needs a measured term:

```
speedup(γ) = τ(γ) / (γ·c + v(γ+1)),    τ(γ) = (1 − α^(γ+1)) / (1 − α)
```

α is how often guesses are accepted, c is the cost of a draft step relative to a target step, and
v(k) is the cost of verifying k tokens in a single pass. The standard formula sets v(k) = 1; this
project measures it. See [PLAN.md](PLAN.md) for the full plan and the open questions.

## What works now

| Piece | State |
|---|---|
| Qwen3 written from scratch in PyTorch | Matches Hugging Face layer by layer and token for token |
| 4-bit and 8-bit block formats | Three implementations (C++, NumPy, torch), byte-identical |
| C++ engine | Memory-mapped weights, AVX-VNNI kernels, thread pool, fp16 KV cache |
| k-token verification kernel | One pass over the weights for several tokens, bit-identical to running one token at a time |
| Speculative decoding | Inside the engine; greedy output matches plain decoding token for token |
| Prompt lookup drafting | Copies from earlier text, needs no model work, output still exact |
| Distillation pipeline | Five losses, a trainable draft, the training loop and the data pipeline, tested end to end on a miniature model |
| 877 tests | Statistical, bit-exactness and end-to-end |

## Measured so far

Qwen3-4B at 4 bits as the target, Qwen3-0.6B at 4 bits as the draft, on an i7-13620H (6 performance
cores, 4 efficiency cores, 16 GB DDR4-3200). The figures are from a quiet-machine run on
2026-10-04; section 13 of the plan explains why measurement on this machine needs care. Attention
has since been rewritten and has not yet been re-measured.

- Read bandwidth is 37–39 GB/s against a theoretical 51.2, which sets the ceiling on decoding speed.
- Qwen3-4B decodes at 64.1 ms a token, or 15.6 tokens a second, which is 88.7% of that ceiling. The
  k-token kernel reaches 95–105% of the AVX-VNNI issue peak for k ≥ 4 at six threads.
- Speculative decoding is 1.20× on decoding, measured on Spec-Bench at γ=1 against the same engine
  without it, at acceptance α = 0.736 and 1.72 tokens a round. Wall-clock speedup is 1.07–1.11× on
  short prompts and 0.82× on long ones, since the prompt pass is 45–92% of the clock.
- The γ sweep gives 1.20×, 1.01×, 0.95× and 0.88× for γ = 1 to 4, so one guess a round is the best
  choice available. Assuming v(k) = 1 instead predicts 1.69× at γ=3, where the engine measures
  0.95×, which is slower than not speculating at all.
- v(k) and c are functions of context. Between context 128 and 2048, v(2) rises from 1.12 to 1.46
  and c from 0.186 to 0.268. Measured at the context actually being used, the cost model predicts
  the engine to within 2–7%.
- Efficiency cores lower throughput. Six performance cores beat all ten, because a fixed split
  leaves the fast cores waiting; dynamic chunking recovers most of the loss without overtaking.
- 4-bit quantization costs accuracy on a 0.6B, taking WikiText-2 perplexity from 28.5 to 32.2.
  8-bit is roughly free, and an 8-bit output layer recovers a fifth of the 4-bit loss.
- The quantization twin agrees with the engine on 98.9% of top-1 tokens at 0.023 mean TVD, close
  enough to rank training recipes on a GPU.

## Not done yet

- **A distilled draft model**, the part the repository is named after. The losses, trainable draft,
  training loop and data pipeline are written and tested, but no draft has been trained. The scope
  is budgeted in section 11.7 of the plan. Recomputed at the context the engine is evaluated at,
  distillation is worth about +0.06× rather than the +0.25× estimated earlier, and it is not
  expected to change the best γ.
- **A llama.cpp baseline**, which needs a decision on how to obtain matching weights there.
- **The write-up**, section 12 of the plan.

## Layout

```
engine/      C++: weights file, forward pass, kernels, thread pool, sampling, the round loop
python/      reference model, quantization twin, export, offline metrics, distillation
scripts/     fetch, export, benchmark, measure v(k), generate data, train
tests/       877 tests, all runnable on a CPU
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
