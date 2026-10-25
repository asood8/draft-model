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
| 900 tests | Statistical, bit-exactness and end-to-end |

## Measured so far

Qwen3-4B at 4 bits as the target, Qwen3-0.6B at 4 bits as the draft, on an i7-13620H (6 performance
cores, 4 efficiency cores, 16 GB DDR4-3200). The figures are from **2026-10-25**, the first
session in which every number came from one machine state: read bandwidth drifted 0.12% across
eight v(k) runs, eight Spec-Bench runs, two per-stage sweeps and 900 tests. Section 13 of the plan
explains why measurement on this machine needs care, and `results/README.md` records which stored
runs are comparable with which.

Two things that session changed are worth naming up front. The gamma sweep stored from 2026-10-05
had been run ascending and back to back, so every larger gamma inherited the heat of the runs
before it; re-measured with the visit order cancelled, three of its four entries moved by 11-19%.
And the decode step at short context is genuinely 13% slower than it was on 2026-10-04 -- which is
not the machine, since bandwidth is the highest ever recorded here and long context got *faster*,
and which is not yet explained.

- Read bandwidth on the six performance cores is 40.19 GB/s against a theoretical 51.2, which sets
  the ceiling on decoding speed at 17.8 tokens a second. Earlier sessions read 39.0–39.8, and an
  early run on a worse machine state read 29.5, which is the spread section 13 is about rather than
  a second figure.
- Qwen3-4B decodes at 72.6 ms a token, or 13.8 tokens a second, which is 77.5% of that ceiling. On
  2026-10-04 the same measurement read 64.1 ms, 15.6 tokens a second and 88.7%, corroborated there
  by two independent harnesses agreeing to 0.05%, so the 13% is real rather than an artefact. It is
  not bandwidth, which is higher now; not the clock, since the per-round overhead `o` is an absolute
  cost of non-streaming work and did not move; and not the code, since rebuilding the baseline
  commit and alternating it against HEAD inside one process put head/base at 0.975. It is spread
  across the weight-streaming stages with no single culprit, and it is open. The k-token kernel
  reaches 95–105% of the AVX-VNNI issue peak for k ≥ 4 at six threads.
- Speculative decoding is 1.19× on decoding and 1.07× on wall clock, measured on Spec-Bench at γ=1
  against the same engine without it, at acceptance α = 0.768 and 1.76 tokens a target forward pass,
  over four categories whose prompt pass is 27% of the clock. On the short- and long-prompt sets
  measured 2026-10-05, wall clock is 1.07× short and 0.82× long, since the prompt pass is 15–92% of
  the clock depending on the category and about half of it in aggregate. By category the
  short-prompt wall figure runs 0.94× (multi-turn) to 1.12× (math), so two of the four are at or
  below parity even there.
- The γ sweep, with each γ visited in two different orders so that drift cannot masquerade as the
  effect, gives **1.19×, 1.20×, 1.06× and 0.98× on decoding** and **1.07×, 0.99×, 0.95× and 0.84× on
  wall clock** for γ = 1 to 4. One guess a round is the best choice available, and on wall clock the
  decline is monotonic — but on decoding γ=1 and γ=2 are tied and the decline does not start until
  γ=3. Assuming v(k) = 1 instead predicts 1.79× at γ=3, where the engine measures 1.06× on decoding
  and 0.95× on the clock; the error from treating verification as free grows from 1.23× at γ=1 to
  1.83× at γ=4.
- Confidence-based early stopping is what makes a larger γ survivable. Plain speculation drops below
  parity on wall clock as soon as γ=2 (0.99×), where stopping early holds 1.05× at γ=2 and 1.01× at
  γ=3, and posts the best decoding figure of the sweep at γ=1 (1.20×). The best single wall-clock
  number is still plain speculation at γ=1.
- v(k) and c are functions of context. Between context 128 and 2048, v(2) rises from 1.13 to 1.25
  and c from 0.182 to 0.245. Measured at the context actually being used, the cost model predicts
  the order-cancelled γ sweep to 5.0% mean error at context 512 and 5.6% at 2048. The direction is
  solid — c is monotonic in every pass ever taken — but the magnitude of v(2)'s context dependence
  is not pinned: two shuffled passes on the same quiet night read spreads of 1.04× and 1.18× across
  the same four contexts, so long-context v(k) is the noisy half.
- Efficiency cores lower throughput. Six performance cores beat all ten, because a fixed split
  leaves the fast cores waiting; dynamic chunking recovers most of the loss without overtaking.
- 4-bit quantization costs accuracy on a 0.6B, taking WikiText-2 perplexity from 28.5 to 32.2.
  8-bit is roughly free, and an 8-bit output layer recovers a fifth of the 4-bit loss.
- The quantization twin agrees with the engine on 98.9% of top-1 tokens at 0.023 mean TVD, close
  enough to rank training recipes on a GPU.
- Attention, after the rewrite, costs 9.91 ms a token at context 2048 in the kernel benchmark,
  against a floor of 8.88 ms for the access pattern it needs, so 12% remains and the stage is
  finished. The shape it replaced costs 19.87 ms on the same measurement. Inside the running model
  it measures 12.49 ms against the 26.88 ms it replaced — **2.15× faster** — with the gain growing
  across context: 1.27× at 128, then 1.60×, 1.87× and 2.15× at 512, 1024 and 2048. The whole decode
  step follows at 1.04×, 0.97×, 0.95× and 0.86×, so the rewrite costs 4% at short context and buys
  14% at 2048.
- The vectorized softmax is 4.86x a scalar one at a 129-element row, 5.70x at 2049 and 6.00x over
  the 151,936-entry vocabulary. Since a token holds 1152 attention rows and one vocabulary row, that
  is 0.549 ms a token at context 128 and 8.894 ms at 2048. It is also the more accurate of the two:
  the scalar row's probabilities sum to 1.00010138 where this one reaches 0.99999996.
- The vectorized argmax is 9.16x the scalar scan over the vocabulary, 0.058 ms against 0.534 ms.

## Not done yet

- **A distilled draft model**, the part the repository is named after. The losses, trainable draft,
  training loop and data pipeline are written and tested, but no draft has been trained. The scope
  is budgeted in section 11.7 of the plan. Recomputed at the context the engine is evaluated at,
  distillation is worth about +0.08× on decoding rather than the +0.25× estimated earlier. It could
  now move the best γ on decoding, since the crossover where two guesses beat one fell to 0.818 at
  context 512 and 0.730 at 1024 against a measured α of 0.768 — but not on wall clock, where γ=2
  trails γ=1 by 9% for a reason no amount of α touches: the engine prefills the prompt into both
  models, and that cost does not care what γ is.
- **A llama.cpp baseline**, which needs a decision on how to obtain matching weights there.
- **The write-up**, section 12 of the plan.

## Layout

```
engine/      C++: weights file, forward pass, kernels, thread pool, sampling, the round loop
python/      reference model, quantization twin, export, offline metrics, distillation
scripts/     fetch, export, benchmark, measure v(k), generate data, train
tests/       900 tests, all runnable on a CPU
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
