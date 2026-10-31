# draft-model (in progress)

### 📄 [What Speculative Decoding Costs on a CPU: Measuring the Verification Term](paper/main.pdf)

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

## The paper

**[Read it here (PDF)](paper/main.pdf)** — the cost model, the measurements behind it, and what it
still cannot account for. The source is in [`paper/`](paper/), its figures are generated from the
stored results by `scripts/make_plots.py`, and every number it quotes in prose is checked against
`results/` by `tests/test_documented_numbers.py` rather than by rereading.

It is published here rather than at a preprint server, which keeps the text, the figures, the raw
measurements and the tests that tie them together at one revision.

## The result

Measured over the whole of Spec-Bench — 480 questions, 2240 turns, γ=1:

| | decoding | wall clock |
|---|---|---|
| Speculative decoding | **1.11×** | **0.98×** |
| Prompt lookup | 1.015× | **1.016×** |
| Speculative + early stopping | 1.12× | 0.99× |

Speculative decoding is **1.11× on decoding and 0.98× on wall clock**, so on the number a user
actually waits for, **it loses**. Prompt lookup — which drafts by copying from earlier text and runs
no model at all — is the only method measured above parity, at **1.016×**, despite losing on
decoding in every run ever taken here.

The reason is not the decode loop. The engine prefills the prompt into *both* models, and the
target's prompt pass alone is **44.5% of the clock**. Which method wins is decided by that share,
not by acceptance: the wall-clock winner is **monotonic in prefill share** across all six
categories, and a cost model with one term for the prompt pass and no new free parameter predicts
all 18 category-method pairs to **1.2% mean absolute error**.

That is a worse headline than this repository used to report, and the difference is the benchmark
rather than the engine — the 8-question subset it used before contained neither long-prompt
category. The paper says so in those words.

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
| The paper | Written, compiled, and published from this repository |
| 944 tests | Statistical, bit-exactness and end-to-end |

## Measured so far

Qwen3-4B at 4 bits as the target, Qwen3-0.6B at 4 bits as the draft, on an i7-13620H (6 performance
cores, 4 efficiency cores, 16 GB DDR4-3200).

The figures come from two sessions, and which one matters per figure. The **γ=1 speedups and the
wall-clock model** are from the full 480-question run of **2026-10-27**, 16 resumable chunks over 28
hours. Everything else — v(k), c, the γ sweep, the per-stage costs — is from the quiet session of
**2026-10-25**, the first in which every number came from one machine state: read bandwidth drifted
0.12% across eight v(k) runs, eight Spec-Bench runs, two per-stage sweeps and the whole suite.
Section 13 of the plan explains why measurement on this machine needs care, and
`results/README.md` records which stored runs are comparable with which.

Two findings about the measurements themselves are worth naming up front. The γ sweep stored from
2026-10-05 had been run ascending and back to back, so every larger γ inherited the heat of the runs
before it; re-measured with the visit order cancelled, three of its four entries moved by 11–19%.
And the decode step at short context is genuinely 13% slower than it was on 2026-10-04 — which is
not the machine, since bandwidth is the highest ever recorded here and long context got *faster*,
and which is not yet explained.

### Speculative decoding

- **Speculative decoding is 1.11× on decoding and 0.98× on wall clock** over the whole of
  Spec-Bench — 480 questions, 2240 turns, γ=1, at acceptance α = 0.717 and 1.71 tokens a target
  forward pass, with the target's prompt pass 44.5% of the clock. **On wall clock it loses.** Prompt
  lookup, at 1.016×, is the only method measured above parity.
- The difference from the figure this repository used to report is the benchmark subset rather than
  the engine. An 8-question subset read 1.19× and 1.07×, because it contained no `rag` or
  `summarization` — the two long-prompt categories, where the prompt pass is 62–66% of the clock and
  a second prefill cannot be repaid. The figures here supersede it.
- By category the wall-clock winner is **monotonic in prefill share**, which is what the cost model
  in section 3.1 of the plan predicts: a model draft wins at qa (f = 0.053) and math (0.136), prompt
  lookup wins at multi-turn (0.309), translation (0.489), summarization (0.617) and rag (0.659).
  Speculation ranges 1.14× on math down to 0.89× on summarization.
- The wall-clock model gets *sharper* on the full benchmark, not looser. Its one extra term is
  k = P_draft / P_target, measured at **+0.203** for a model draft and **+0.008** for prompt lookup
  — zero within noise, as a drafter that runs no model requires. One pooled k per method predicts
  all 18 category-method pairs to **1.20%**; six per-category values do *worse*, at 1.38%, which is
  the useful way round. It picks the winning method in 5 of the 6 categories.
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
- The γ sweep above, v(k), c and the per-stage costs are from the 8-question subset; only the γ=1
  figures have been re-measured on the full benchmark.
- v(k) and c are functions of context. Between context 128 and 2048, v(2) rises from 1.125 to 1.245
  and c from 0.182 to 0.245. Measured at the context actually being used, the cost model predicts
  the order-cancelled γ sweep to 5.0% mean error at context 512 and 5.6% at 2048. The direction is
  solid — c is monotonic in every pass ever taken — but the magnitude of v(2)'s context dependence
  is not pinned: two shuffled passes on the same quiet night read spreads of 1.04× and 1.18× across
  the same four contexts, so long-context v(k) is the noisy half.

### The engine

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
- Attention, after the rewrite, costs 9.91 ms a token at context 2048 in the kernel benchmark,
  against a floor of 8.88 ms for the access pattern it needs, so 12% remains and the stage is
  finished. The shape it replaced costs 19.87 ms on the same measurement. Inside the running model
  it measures 12.49 ms against the 26.88 ms it replaced — **2.15× faster** — with the gain growing
  across context: 1.27× at 128, then 1.60×, 1.87× and 2.15× at 512, 1024 and 2048. The whole decode
  step follows at 1.04×, 0.97×, 0.95× and 0.86×, so the rewrite costs 4% at short context and buys
  14% at 2048.
- Efficiency cores lower throughput. Six performance cores beat all ten, because a fixed split
  leaves the fast cores waiting; dynamic chunking recovers most of the loss without overtaking.
- The vectorized softmax is 4.86× a scalar one at a 129-element row, 5.70× at 2049 and 6.00× over
  the 151,936-entry vocabulary. Since a token holds 1152 attention rows and one vocabulary row, that
  is 0.549 ms a token at context 128 and 8.894 ms at 2048. It is also the more accurate of the two:
  the scalar row's probabilities sum to 1.00010138 where this one reaches 0.99999996.
- The vectorized argmax is 9.16× the scalar scan over the vocabulary, 0.058 ms against 0.534 ms.

### Quantization

- 4-bit quantization costs accuracy on a 0.6B, taking WikiText-2 perplexity from 28.5 to 32.2.
  8-bit is roughly free, and an 8-bit output layer recovers a fifth of the 4-bit loss.
- The quantization twin agrees with the engine on 98.9% of top-1 tokens at 0.023 mean TVD, close
  enough to rank training recipes on a GPU.

## Not done yet

- **A distilled draft model**, the part the repository is named after. The losses, trainable draft,
  training loop and data pipeline are written and tested, but no draft has been trained; the scope
  is budgeted in section 11.7 of the plan.

  The full-benchmark result raises its value. Speculation now loses on the clock, so a distilled
  draft is what would carry it back above parity rather than merely improving on something that
  already works. Recomputed at the context the engine is evaluated at, distillation is worth about
  **+0.08× on decoding** rather than the +0.25× estimated earlier — and inverting the wall-clock
  model says parity needs only **+0.04× on decoding** at the measured prefill share, so the prize
  clears the bar with about a factor of two to spare.

  It is worth far less on the clock than on decoding, roughly **+0.03×**, because 44.5% of the clock
  is a prompt pass that no amount of draft quality touches. It is nonetheless the only lever
  measured here that crosses parity at all. It could also move the best γ on decoding, since the
  crossover where two guesses beat one fell to 0.818 at context 512 and 0.730 at 1024, against the
  subset's measured acceptance of 0.768 — but not on wall clock, where γ=2 trails γ=1 by 9% for a
  reason no amount of acceptance touches: the engine prefills the prompt into both models, and that
  cost does not care what γ is.
- **A γ sweep on the full benchmark.** γ=1 is now measured on all 480 questions; γ=2, 3 and 4 are
  still the 8-question subset — the same subset whose composition this project shows to be
  misleading — so the claim that γ=1 is the best choice rests on it for the comparison, even though
  γ=1's own value is now solid.
- **A second machine**, which is the paper's one falsifiable prediction: a kernel that spends more
  arithmetic per token should show a *steeper* v(k). Testing it does not strictly need different
  hardware, since the engine selects its kernel at runtime and can be forced onto the slower path
  with memory traffic held fixed.
- **A llama.cpp baseline** — decided against, 2026-10-25. A relative speed figure is only as
  informative as the program it is relative to, and establishing that llama.cpp runs bit-identical
  weights (same model, same block quantization, same numbers) is difficult enough that a comparison
  that could not be fully defended would be worse than none. The absolute argument is also stronger:
  sitting within a few percent of what the hardware can do bounds how much faster *any*
  implementation could be, which no pairwise comparison does.

## Layout

```
engine/      C++: weights file, forward pass, kernels, thread pool, sampling, the round loop
python/      reference model, quantization twin, export, offline metrics, distillation
scripts/     fetch, export, benchmark, measure v(k), generate data, train
tests/       944 tests, all runnable on a CPU
results/     measurements, as JSON
paper/       the write-up, in LaTeX, with the compiled PDF
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

## License

[MIT](LICENSE). The Qwen3 weights are Alibaba's and carry their own licence; Spec-Bench is its
authors' and is fetched rather than vendored.
