# What speculative decoding costs on a CPU

*Draft, 2026-10-24. Numbers in square brackets are not yet measured; everything else carries the
context and the machine conditions it was taken under, because on this hardware a figure without
them is not a figure (see "Measuring on a laptop" below).*

## The claim this started from

Speculative decoding runs a small draft model γ tokens ahead, then checks all γ+1 guesses in a
single pass of the large model. Its speedup is usually written

```
speedup(γ) = τ(γ) / (γ·c + 1),    τ(γ) = (1 − α^(γ+1)) / (1 − α)
```

where α is how often a guess is accepted and c is the cost of a draft step relative to a target
step. The `1` in the denominator is the interesting part. It says that verifying γ+1 tokens costs
the same as generating one, and on a GPU that is close to true: generation is limited by reading the
weights, the weights are read once however many tokens share the pass, and the extra arithmetic
disappears into hardware that has arithmetic to spare.

A CPU does not have arithmetic to spare. The weights are still read once, but the integer
multiply-accumulates grow with every token added to the pass, and at six cores they are not free. So
the denominator needs a measured term:

```
speedup(γ) = τ(γ) / (γ·c + v(γ+1) + o),    v(k) = pass(k) / pass(1)
```

v(k) is the cost of verifying k tokens in one pass, relative to an ordinary single-token step, and
o is the per-round overhead of sampling and the acceptance test. This project is a from-scratch C++
inference engine built to measure v(k) on a laptop, and the finding is that **v(k) does not correct
the answer, it chooses it**.

## The engine

Qwen3 implemented twice: once in PyTorch, validated layer by layer and token for token against
Hugging Face, and once in C++ as the thing that actually runs.

- **Weights** are memory-mapped in a custom format, quantized to 4 or 8 bits in blocks with a scale
  and zero-point each. Three independent implementations of the format — C++, NumPy, PyTorch — agree
  byte for byte, which is what makes the PyTorch "twin" usable as a stand-in for the engine.
- **Kernels** are hand-written AVX2/FMA/F16C with AVX-VNNI detected at runtime and a scalar
  fallback that CI exercises for real, since the runners have no VNNI.
- **The k-token kernel** shares one pass over the weights across the tokens being verified, and is
  **bit-identical** to calling the single-token kernel k times, tested for k = 1 to 13 and across
  tile boundaries. That is the invariant greedy speculative decoding rests on: identical arithmetic
  means the speculative output is the same text the target would have produced alone, so the
  speedup is free of any quality question.
- **Attention** reads each cached key/value row once per *group* of query heads that share it rather
  than once per head, and is parallelized over (key/value head, chunk of 256 positions). The chunk
  size is a constant rather than a function of the sequence length, because the per-chunk partial
  sums are added in chunk order and fixed boundaries are what keep a k-token pass bit-identical to
  k single-token passes.
- **The decode loop lives in C++** — sampling warps, a seeded xoshiro generator, the acceptance
  rule, the round loop — so no Python overhead can land in a timing.
- 877 tests: bit-exactness, statistical (chi-square over two-token continuation distributions),
  golden tests against Hugging Face, and end-to-end runs that train a miniature model, export it and
  decode with it in C++.

Qwen3-4B at 4 bits is the target; Qwen3-0.6B at 4 bits is the draft.

## Measuring on a laptop

This matters more than it should, so it comes before the results rather than after.

The machine is an i7-13620H: six performance cores, four efficiency cores, 16 GB of DDR4-3200.
Repeated samples of the *same build* have ranged from 40 to 70 tokens a second, and medians taken
minutes apart have differed by 75%. Single-threaded runs are steady to about 10%; six-thread runs
are not, and saturating all six performance cores produces roughly ten times as many long stalls as
leaving one core free — the operating system needs a core, and a spinning worker that gets preempted
holds up the whole barrier.

Three rules came out of that, each from being burned:

1. **Compare variants inside one process, alternating.** Every kernel result here was taken that
   way. Two builds timed one after the other measure the gap between two machine states as much as
   between two kernels.
2. **Never sweep a parameter in a fixed order.** A sweep over context lengths taken in ascending
   order once reported v(2) = 0.97 — two tokens verified for less than the price of one, which is
   impossible. The machine was drifting monotonically and a monotonic drift is indistinguishable
   from the monotonic dependence on context that was being measured.
3. **Record the conditions with the number.** v(k) is a ratio, and a slower clock inflates pass(1)
   while leaving memory bandwidth alone, which flattens the curve for free. A v(k) table without its
   machine state attached cannot be compared with another one.

## Results

### The kernel is at the hardware's limit

At context 128, on a quiet machine (2026-10-04):

| | |
|---|---|
| One decode step, Qwen3-4B 4-bit | **64.10 ms, 15.6 tokens/second** |
| Weights read per token | 2262 MB |
| Measured read bandwidth | 39.0–39.8 GB/s, against 51.2 theoretical |
| Streaming floor for one step | 56.8 ms |
| **Share of the ceiling reached** | **88.7%** |

The k-token kernel reaches **95–105% of the AVX-VNNI issue-rate peak** for k ≥ 4 at six threads:
0.030 issue slots per multiply-accumulate where a pure `dpbusd` stream is 0.031. Getting there took
three changes measured by counting instructions — applying scales eight blocks at a time, replacing
a sign trick with `Σ(q−z)x = Σqx − zΣx` and one precomputed integer per block, and loading two
blocks in one 32-byte load so eight blocks need four accumulators instead of eight. 43 instructions
per eight blocks per row against 64, and the share of them doing arithmetic went from 9.9% to 23.6%.
Fitted, a verification pass went from `32 + 34k` ms to **`34.5 + 18.7k` ms**: the per-token slope
fell 1.82× while the intercept stayed put, which is the shape a kernel change should have and agrees
with the 1.65–1.77× an independent kernel sweep measures.

### v(k), and the context it was measured at

All from 2026-10-05, quiet machine, Qwen3-4B 4-bit, k up to 5, three interleaved samples:

| context | one step | v(2) | v(3) | v(4) | v(5) | c | best γ |
|---|---|---|---|---|---|---|---|
| 128 | 64.10 ms | 1.12 | 1.34 | 1.64 | 1.87 | 0.186 | 1 → 1.31× predicted |
| 512 | 79.93 ms | 1.28 | 1.67 | 2.14 | 2.52 | 0.205 | 1 → 1.17× predicted |
| 1024 | 86.14 ms | 1.35 | 1.78 | 2.36 | 2.74 | 0.244 | 1 → 1.09× predicted |
| 2048 | 98.36 ms | 1.46 | 1.97 | 2.49 | 3.00 | 0.268 | 1 → **1.01×** predicted |

o is 0.007 target steps and negligible — see "What I got wrong" for how it got there.

Two things to take from the table. **v(k) is a function of context**, because the KV cache is read
per token of a pass rather than shared across it: 147.5 KB per token of context for this model,
which is nothing beside 2.26 GB of weights at context 128 and 147 MB at context 1000, multiplied by
k. And **by 2k tokens of context, speculation stops paying at all** — both terms move against it.

### v(k) chooses γ; assuming v = 1 chooses wrong

Spec-Bench, greedy, each γ re-measuring its own baseline in the same process (2026-10-04, quiet):

| γ | predicted from v(k) at context 512 | measured, decoding | assuming v(k) = 1 |
|---|---|---|---|
| 1 | 1.17× | **1.20×** | 1.33× |
| 2 | 1.09× | 1.01× | 1.56× |
| 3 | 0.97× | 0.95× | **1.69×** |
| 4 | 0.89× | 0.88× | 1.77× |

The cost model predicts the engine **to within 2–7%** once v(k) and c are measured at the context
being run at. The textbook formula, which would pick γ = 3 and promise 1.69×, selects the
configuration where this engine is **5% slower than not speculating at all**. That is the result the
project exists for: the term is not a correction to the answer, it is what picks the configuration.

Acceptance, scored offline over 1,318 response positions from 24 Spec-Bench prompts: **α = 0.711**
off the shelf, by category 0.80 math, 0.73 multiturn, 0.72 rag, 0.72 translation, 0.69
summarization, 0.61 qa.

### Decode speedup and wall-clock speedup are different numbers

Worth separating, because quoting only the first would be the easy dishonesty here:

| | short prompts | long prompts |
|---|---|---|
| Decoding | **1.20×** (math 1.29×, τ = 1.72, α = 0.736) | **0.95×** (α falls to 0.44–0.60) |
| Wall clock | 1.07–1.11× | **0.82×** |

The gap is the prompt pass, which is **45–92% of the wall clock**. Speculative decoding does nothing
for it: the prompt is processed in one batched pass whether or not there is a draft model, so a
technique that only accelerates decoding is diluted by however much of the clock decoding is. On
long-prompt categories the engine is slower end to end than not speculating, and the honest headline
is the pair of numbers, not the better one.

### What the stages cost, and what moved

ms a token, context 128 → 2048, before the attention work (2026-10-04, quiet):

| stage | 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| step | 70.32 | 78.38 | 83.31 | 98.97 |
| attention | **1.50** | **7.17** | **13.32** | **26.88** |
| qkv | 11.84 | 13.74 | 13.35 | 13.76 |
| gate_up | 29.39 | 30.14 | 29.38 | 29.91 |
| ffn_down | 15.64 | 15.08 | 14.73 | 14.79 |
| output | 5.70 | 5.76 | 5.97 | 6.67 |

Attention is the only stage that grows with context, and at 2048 it is 27% of a step. It was never
memory-bound: a bare scan of the same cache reaches 35.5 GB/s where attention was managing 11.3. The
costs were a scratch buffer, converting each cached row once per query head instead of once per
group, and job imbalance at 8 key/value heads over 6 workers.

After the rewrite, measured in isolation at context 2048, in one process, best of three:

| shape | ms/token | GB/s |
|---|---|---|
| bare scan — the floor the access pattern allows | 8.88 | 34.0 |
| **what the engine runs** (grouped, 6 jobs a head) | **9.91** | 30.5 |
| the shape it replaced | 19.87 | 15.2 |

12% off the floor, so attention is finished: 1 ms a token is about 1% of a step, and collecting it
means touching the code that carries the chunk-order invariant.

The same table confirms the parallel split's value by a second, opposite-signed route. Cutting
`scan` finer makes it **worse** — 34.0 GB/s at one job a head, 19.5 at three — because pure
streaming has nothing to balance and pays the barriers anyway, while the grouped shape improves 18.1
to 30.5 GB/s over the same change, because 8 jobs over 6 workers cannot balance and splitting is the
only fix. One mechanism, two signs.

And the sampling path, vectorized against a scalar baseline kept for the purpose, alternating in one
process:

| row | vector | scalar | | scalar's probabilities sum to |
|---|---|---|---|---|
| 129 floats (an attention row at context 128) | 0.123 µs | 0.600 µs | 4.86× | 0.99999951 |
| 2049 floats (at context 2048) | 1.644 µs | 9.365 µs | 5.70× | 1.00000002 |
| 151,936 floats (one sampling warp) | 117.5 µs | 704.8 µs | **6.00×** | **1.00010138** |

A token holds **1152 attention rows** — 32 heads over 36 layers — and one vocabulary row, so this is
worth 0.549 ms a token at context 128 and 8.894 ms at 2048, plus 0.587 ms a round for the warp. The
change was made for the vocabulary, where the accuracy argument is; most of its value turned out to
be in attention, where there is none. There was also no trade to make: the version it replaced summed
151,936 float32 exponentials into a single float and reached 1.00010138, where accumulating in double
reaches 0.99999996. Six times faster and four orders of magnitude closer to 1.

### Quantization and the twin

4 bits costs real quality on a 0.6B — WikiText-2 perplexity 28.5 → 32.2 — while 8 bits is
approximately free, and an 8-bit output layer buys back about a fifth of the 4-bit loss. The PyTorch
twin of the engine's arithmetic agrees with it on **98.9% of top-1 tokens at 0.023 mean TVD**, which
is close enough to rank training recipes on a rented GPU and deploy the winner on a CPU, and not
close enough to quote acceptance from — so the chosen draft gets re-scored by the engine itself.

## What I got wrong

The project's measurements caught a long list of my own mistakes, and leaving them out would make
the rest less believable.

**Two bugs in my own benchmark.** Wall-clock tokens per second initially included the prompt pass
and reported 0.82×; fixing it the wrong way compared a decode rate against a wall rate and reported
9.91×. Neither was real. The baseline now gets measured exactly like the thing it is a baseline for.

**The argmax was the entire per-round overhead.** o was 8.71 ms a round and I had named thread
dispatch as the suspect. Dispatch is 2 µs a round. The scalar argmax over a 151,936-entry vocabulary,
run about 6.7 times a round, was 3.9 ms of it by benchmark and more like 8 ms in practice on a
freshly written 608 KB row. Vectorized, 9.16×, and o fell to 0.57 ms — 0.007 target steps, which is
why the third term of the formula can be ignored on this engine.

**Four optimization hypotheses killed by measurement**, including one where the fix would have been
slower than the bug: a prefill chunk size of 128 would have cost 42.6 ms a token against the buggy
40.6, and the right answer was 16. Another — that redundant per-head conversions were the problem —
was right about the cause and wrong about the cure, since an accumulator array indexed by a runtime
bound cannot live in registers, and templating the group size is what made the grouped shape fastest.
The arithmetic was never the problem; where the accumulators lived was.

**A pruning result that looked like a trade-off and wasn't.** Block influence correctly ranked layers
23–26 as least useful, but dropping 8 of 28 leaves a model emitting gibberish whose acceptance as a
draft collapses to **4%**. There is no α-against-c curve to explore until a training run heals it, so
every pruned variant has to be budgeted with training attached. The saving is also smaller than the
layer count suggests — 0.79× bytes a token, because the output layer over 151,936 rows is a fixed
cost pruning cannot touch — which is what makes vocabulary trimming its companion rather than its
alternative (87.5 MB → 18.9 MB, and 335 → 267 MB a step).

**A milestone talked out of its own budget.** The plan had distillation worth 1.42–1.5×. Those
numbers came from v(k) and c measured at context 128, and the evaluation prompts are four times
longer. Recomputed at context 512 the same formula gives 1.21–1.25× against 1.17× today, so the
prize is about **+0.06×**, and two guesses a round do not overtake one until α = 0.864, past where
distilling a 0.6B against a 4B plausibly lands. The distillation grid was cut from a 50M-token
program to 14–19 GPU-hours on that basis, with a success threshold fixed before any of it is spent.

**And three in one night, during the last re-measurement.** Blaming a probe's drift on heat when ten
calls amount to under 100 ms of spinning; blaming a 13 ms regression on the shared softmax, which
turned out to be saving 8.9 ms a token; and reading a jump at k = 5 as a kernel tiling boundary when
it was thermal and vanished on a cool run. The same run also overwrote three stored result files with
invalid numbers, which is why results are now named by the conditions they were taken under.

## What is not done

- **The distilled draft model**, which is the part the repository is named after. The losses,
  trainable draft, training loop and data pipeline are written and tested end to end on a miniature
  model, but no draft has been trained. Expected value at the deployment context: [+0.06×].
- **A llama.cpp baseline**, pending a decision on how to obtain matching weights there.
- **A re-measured v(k) table** after the attention and softmax work. The in-process comparisons above
  hold regardless of the machine, but the end-to-end figures are still from 2026-10-04 and the
  attempt to retake them on a busy machine was inconclusive — which is itself the third rule above,
  earning its place.

## The shortest version

A from-scratch C++ CPU inference engine for Qwen3 reaching 88.7% of its machine's memory-bandwidth
ceiling and 95–105% of the AVX-VNNI issue-rate peak, used to measure the term the standard
speculative-decoding speedup formula assumes away. Measured, that term picks one draft token a round
and delivers **1.20× on decoding, 1.07–1.11× on the wall clock, at context 512**. Assumed away, it
picks three and promises 1.69×, where the engine measures 0.95× — slower than not speculating. The
cost model predicts the engine to within 2–7% once v(k) and c are measured at the context being used,
which is the practical form of the result: **on a CPU, the configuration is chosen by the cost of
verification, not by the acceptance rate.**
