# Project Plan: A Distilled Draft Model for Speculative Decoding

Train a small draft model that makes speculative decoding of **Qwen3-4B** measurably faster on a single
**T4 GPU**. Along the way, build the decoder from scratch, study which distillation choices actually raise
acceptance, and check measured speedups against a cost model.

This is a living document. Section 14 tracks decisions and open questions.

---

## Contents

0. [TL;DR](#0-tldr)
1. [Goal, deliverables, success criteria](#1-goal-deliverables-success-criteria)
2. [Background: where the speedup comes from](#2-background-where-the-speedup-comes-from)
3. [Changes from the original idea](#3-changes-from-the-original-idea)
4. [Setup](#4-setup)
5. [Phase 0: Feasibility](#5-phase-0-feasibility-24-evenings)
6. [Phase 1: The decoder](#6-phase-1-the-decoder-12-weeks)
7. [Phase 2: Distillation](#7-phase-2-distillation-23-weeks)
8. [Phase 2b: Make the draft cheaper](#8-phase-2b-make-the-draft-cheaper-optional-1-week)
9. [Phase 3: Experiments](#9-phase-3-experiments-2-weeks)
10. [Phase 4: Release](#10-phase-4-release-a-few-days)
11. [Timeline and milestones](#11-timeline-and-milestones)
12. [Risks and mitigations](#12-risks-and-mitigations)
13. [Positioning and related work](#13-positioning-and-related-work)
14. [Decisions and open questions](#14-decisions-and-open-questions)

---

## 0. TL;DR

- **Speedup = tokens per round ÷ cost per round.** Distillation raises the first number by increasing the
  acceptance rate α. CUDA graphs, layer pruning and vocabulary trimming lower the second by reducing the draft
  cost ratio c. The project works on both.
- **Pair:** Qwen3-0.6B as the draft, Qwen3-4B as the target, non-thinking mode, float16 on a T4 (Kaggle 2×T4).
- **Main additions to the original idea:**
  1. **An offline round simulator.** It is exact for greedy decoding and exact in distribution for sampling. γ
     sweeps, the loss × data grid and checkpoint selection then cost one forward pass per model instead of full
     decoding runs, which is what makes the experiment grid fit in Kaggle's GPU quota.
  2. **A cost model with verification and overhead terms**, measured against a target baseline that is optimized
     just as much (static cache + CUDA graphs), so speedups aren't inflated.
  3. **A track for lowering c:** layer-pruned drafts recovered by distillation, a trimmed draft vocabulary, and
     confidence-based early stopping. Together they give an α-versus-c Pareto plot.
  4. **An SFT-on-target-text baseline loss**, to answer whether matching the target's full distributions is worth
     it compared with plain fine-tuning on the target's outputs.
  5. **Engineering scaffolding:** a compute budget, a dev set kept separate from Spec-Bench, tests that run on a
     CPU with tiny models, and a minimum viable project (MVP) cut line.
- **Timeline:** about 9–10 weeks part-time. The MVP is done at about week 6.

---

## 1. Goal, deliverables, success criteria

**Headline question:** How much can distillation, together with making the draft cheaper, speed up Qwen3-4B on
a T4? And which training-data and loss choices matter under greedy decoding versus sampling?

**Deliverables**

- A Python package (working name `specdraft`) containing the speculative decoder (dynamic- and static-cache
  backends), the distillation trainer, the offline evaluator and the benchmark harness, all with tests.
- One or more distilled drafts on the Hugging Face Hub, with model cards.
- A README and write-up built around a headline table and a handful of plots (listed in §9.4).
- Reproducible results: every number traces back to a config, a git commit and a hardware record.

**Success criteria** (replace the placeholders with real numbers after Phase 0)

- **Correct.** Greedy output is identical to plain greedy decoding, apart from documented float16 near-ties.
  The acceptance-rule and end-to-end distribution tests pass.
- **Consistent.** Offline simulated tokens per step match online decoding exactly for greedy, and within Monte
  Carlo error for sampling.
- **Better.** The distilled draft beats the off-the-shelf draft on tokens per target step in at least 5 of the 6
  Spec-Bench categories, and on wall-clock speedup on average.
- **Explained.** Predicted speedup is within about 10% of measured speedup for the main configurations, or the gap
  is explained.
- **Competitive.** The custom decoder is at least as fast as Hugging Face's assisted generation with the same
  draft and γ.

---

## 2. Background: where the speedup comes from

Each round, the draft proposes γ tokens autoregressively and the target scores all of them in one forward pass.
Each guess is accepted with probability min(1, p/q), where p and q are the target's and draft's probabilities.
At the first rejection, a replacement is sampled from max(0, p − q), renormalized. If every guess is accepted,
the target's final position supplies one bonus token. The output distribution is exactly the target's
(Leviathan et al., 2023; Chen et al., 2023), so quality cannot change. Only speed does.

**Tokens per round.** With an i.i.d. per-token acceptance rate α and γ guesses per round:

$$\tau(\gamma) = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

- **Sampling:** the acceptance probability at a position is Σₓ min(p(x), q(x)) = 1 − TVD(p, q).
- **Greedy:** a guess is accepted exactly when argmax p = argmax q.

**Cost per round.** Let t_T be one target decode step, t_D one draft step, t_V(k) a target forward pass over k
tokens, and t_O the per-round overhead (sampling, the acceptance test, cache bookkeeping, Python). With
c = t_D/t_T, v = t_V(γ+1)/t_T and o = t_O/t_T:

$$S(\gamma) = \frac{\tau(\gamma)}{\gamma c + v + o}$$

Leviathan et al.'s formula is the special case v = 1, o = 0. At batch size 1 on a T4, v should be close to 1
because the step is memory-bound, but not exactly 1. Kaggle's CPUs are slow, so o can matter. Measure both
rather than assuming them.

**Predicted speedup at the best γ** (v = 1, o = 0, γ ≤ 10; the best γ is in parentheses):

| α \ c | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.45 | 0.60 |
|---|---|---|---|---|---|---|---|
| 0.6 | 1.92× (4) | 1.67× (3) | 1.51× (2) | 1.40× (2) | 1.23× (1) | 1.10× (1) | 1.00× (1) |
| 0.7 | 2.35× (6) | 1.98× (4) | 1.75× (3) | 1.58× (3) | 1.37× (2) | 1.17× (1) | 1.06× (1) |
| 0.8 | 3.09× (8) | 2.47× (6) | 2.11× (5) | 1.87× (4) | 1.55× (3) | 1.28× (2) | 1.12× (1) |
| 0.9 | 4.57× (10) | 3.43× (10) | 2.78× (8) | 2.37× (7) | 1.87× (5) | 1.46× (3) | 1.23× (2) |

Two things to read from this table:

- At c ≈ 0.45, even a strong draft (α = 0.8) gives only about 1.3×. Cost has to come down before acceptance
  matters much.
- At c ≈ 0.15–0.2, cutting c in half is worth about as much as raising α from 0.7 to 0.8.

**Back-of-envelope for this pair on a T4.** These are predictions for Phase 0 to check.

| | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| Layers / hidden size | 28 / 1024 | 36 / 2560 |
| Parameters (lm_head is tied to the embedding) | ~0.60B (0.44B non-embedding) | ~4.0B (3.6B non-embedding) |
| Bytes read per decode step in fp16 | ~1.2 GB (the lm_head alone is ~0.31 GB) | ~8.0 GB |
| Memory-bound step at ~250 GB/s effective | ~5 ms | ~32 ms |

- **Memory bandwidth** alone gives c ≈ 0.15.
- **In eager PyTorch** each layer launches dozens of small kernels, so a draft step issues on the order of a
  thousand launches. That puts the draft at roughly 10–25 ms per step, bound by the CPU rather than the GPU,
  for an **eager c of about 0.4–0.7**, which is where speculative decoding stops paying off. The target's ~32 ms
  of GPU work hides most of its own launch overhead.
- **CUDA graphs** therefore mostly help the draft. Expect a **compiled c of about 0.15–0.25**.

---

## 3. Changes from the original idea

| Change | Why |
|---|---|
| Add an **offline round simulator** (§7.5) that is exact for greedy and exact in distribution for sampling | The original plan's online grid (drafts × losses × 3 decoding modes × γ 1–8 × 6 categories) would take hundreds of T4 GPU-hours. Offline, the whole grid costs minutes per draft, and online runs only confirm and time selected points. |
| **Refine the cost model** with a verification term v and an overhead term o | Makes the predicted-vs-measured plot explainable instead of just "close" or "off". |
| **Compare against an equally optimized target** (static cache + `torch.compile`) | Otherwise part of the reported "speculative" speedup is really a compile speedup. |
| Add **SFT on target text** (sequence-level KD) as a baseline loss, and JSD as an optional one | Answers "are the target's full distributions worth it?". JSD is one of DistillSpec's candidates. |
| **Trim the training grid**: 4 losses on target-generated data, then the best 2 losses on the other sources | Fits the Kaggle quota (§4.3) while keeping both comparisons. |
| Add a **cost-reduction track** (§8): layer pruning + distillation, vocab trimming, confidence-based early stopping | The 0.6B draft has 28 layers against the target's 36, so c is the binding constraint on a T4. This track also produces an α–c Pareto plot. |
| Keep a **dev set separate from Spec-Bench** | Model selection on Spec-Bench would contaminate the final numbers. |
| Add **on-policy distillation** as a stretch goal | Regenerating draft outputs during training (GKD/DistillSpec style) is the natural next step after fixed draft-generated data. |
| Make the **tests run on a CPU with tiny random Qwen3 configs** | Development happens on a laptop, and Kaggle is used only for GPU work. |
| Define an **MVP cut line** (§11) | Keeps a finished, publishable project if time runs short. |
| **Time on a single GPU** (both models on one T4) | Matches real deployment. Two GPUs are for training and generating data only. |

---

## 4. Setup

### 4.1 Models

- **Target:** `Qwen/Qwen3-4B`. **Draft initialization:** `Qwen/Qwen3-0.6B`. Both are Apache-2.0, share the
  tokenizer and chat template, and run in **non-thinking mode** (`enable_thinking=False`).
- **Why stay with Qwen3:** it is pure attention, so rolling a cache back is just a crop. Families that mix in
  linear-attention or recurrent layers can't be cropped and need state snapshots per drafted token, which is a
  different project.
- **Why not Qwen3-1.7B as the draft:** it has the same 28 layers and about three times the bytes per step, so its
  cost ratio would be far worse.
- **Variant to consider:** `Qwen3-4B-Instruct-2507`, a stronger target that has no thinking mode. There is no
  matching 0.6B update, so the draft–target mismatch is larger and distillation matters more. Its tokenizer
  should match, but its chat template differs, so verify both before using it. Decide in Phase 0 by measuring
  baseline acceptance (§14).

### 4.2 Day-one gotchas (checklist)

- [ ] **Vocabulary padding.** Both configs pad the embedding to 151,936 rows, while `len(tokenizer)` is about
      151,669. Slice logits to `len(tokenizer)` everywhere (decoding, losses, metrics) through **one shared
      helper**. The padded rows carry junk logits that would otherwise receive probability mass.
- [ ] **Thinking mode.** Pass `enable_thinking=False` for both models. The non-thinking template inserts an empty
      `<think></think>` block into the generation prompt, so build every prompt through the same
      `apply_chat_template` call.
- [ ] **Tokenizer and template identity.** Assert the same vocabulary, the same special tokens and byte-identical
      template output for a few conversations, including multi-turn ones.
- [ ] **End-of-sequence.** Stop on `<|im_end|>` (and `<|endoftext|>`). Handle an EOS that lands in the middle of
      an accepted draft block, and truncate at `max_new_tokens` partway through a round.
- [ ] **float16.** The T4 has no bfloat16, and Qwen3 was trained in bfloat16. Compare fp16 against fp32 logits for
      both models and look for inf/NaN and very large activations (§5).
- [ ] **Sampling warps.** If you use top-p or top-k, p must be the *warped* target distribution and q must be
      *exactly* the distribution the draft sampled from. Correctness only needs that; efficiency needs the two to
      be similar.
- [ ] **Tied embeddings.** Both models tie the input embedding to the lm_head. This matters for freezing
      embeddings during training and for vocab trimming (§8.2).

### 4.3 Hardware and compute budget

- **Kaggle, 2×T4 (16 GB each).** About 30 GPU-hours a week, with sessions capped at around 12 hours (check the
  current limits).
  - *Training:* GPU0 runs the target (inference only) and GPU1 trains the draft.
  - *Timing:* put both models on **one** T4, as in deployment. Together they take about 9.2 GB of weights.
- **Local Windows laptop:** CPU-only development and tests with tiny random models. No `torch.compile` is needed
  locally.
- **Fallbacks:** Colab (T4 or L4), or a few hours on a rented GPU. An L4 also supports bfloat16 if fp16 turns out
  to be broken (§12).

**Rough GPU budget.** Replace these estimates with Phase 0 measurements.

| Job | Estimate |
|---|---|
| Phase 0 measurements and dev-set references | ~3 h |
| Spec-Bench target references (greedy, T=0.7, T=1.0) | 2–5 h |
| Target-generated training data (~30k responses, ~12M tokens) | 3–8 h (vLLM if it runs on a T4 in your version; otherwise batched HF `generate` on both GPUs) |
| Draft-generated training data | 1–2 h |
| Pilot run plus 8 grid runs of ~10M response tokens each | 15–20 h |
| Scale-up run (~50M tokens) | 7–10 h |
| Phase 2b pruning runs (3 × ~15M tokens, on smaller models) | 5–8 h |
| Offline evaluation (one forward pass per model per reference set) | 3–5 h in total |
| Online timing runs | 12–20 h |
| **Total** | **~50–75 GPU-hours**, about 2–3 weeks of quota spread over the project |

Start generating target data during Phase 1. It only needs the GPU, not the decoder.

### 4.4 Software and workflow

- Python 3.11, PyTorch 2.x and `transformers`. **Pin exact versions** in `pyproject.toml`, because the cache API
  changes between releases. Other dependencies: `datasets`, `bitsandbytes` (8-bit AdamW), `scipy` (chi-square
  tests), `pytest` and `matplotlib`. Optional: `vllm` for fast data generation and `wandb` for logging.
- **Loop:** write code in this repo → a Kaggle notebook runs
  `pip install git+https://github.com/asood8/draft-model@<commit>` → checkpoints and large artifacts go to the
  HF Hub (the token lives in Kaggle Secrets) → small JSON results get committed to `results/`.
- **Every run writes a JSON record** containing the config, git commit, torch/transformers/CUDA versions, GPU name
  and seeds.

### 4.5 Proposed repo layout

```
draft-model/
├── PLAN.md
├── README.md
├── pyproject.toml
├── src/specdraft/
│   ├── models.py      # loading, vocab slicing, chat templates, fp16 checks
│   ├── sampling.py    # temperature/top-p/top-k warps, accept_or_resample
│   ├── decode.py      # speculative loop; DynamicCache and StaticCache backends
│   ├── baselines.py   # plain target, HF assisted generation, prompt lookup
│   ├── offline.py     # 1-TVD, top-1 match, round simulator, predicted speedup
│   ├── data.py        # prompt mixing, decontamination, response generation
│   ├── losses.py      # sft, fkl, rkl, tvd, jsd (chunked)
│   ├── train.py       # two-GPU distillation loop, checkpoint/resume
│   ├── prune.py       # layer pruning, vocab trimming
│   └── bench.py       # Spec-Bench runner, timing protocol
├── configs/           # one YAML file per experiment
├── scripts/           # CLI entry points
├── notebooks/         # thin Kaggle notebooks that install this repo
├── tests/             # CPU tests with tiny random Qwen3 models
└── results/           # JSON records and plots (small files only)
```

---

## 5. Phase 0: Feasibility (2–4 evenings)

**Goal:** know c, v, o, the baseline α and whether fp16 works before writing the real decoder.

1. **Sanity checks.**
   - Load both models in fp16 and run the §4.2 checklist.
   - On about 20 prompts, compare fp16 and fp32 logits: max absolute difference, top-1 agreement, and any inf or
     NaN.
   - Record the largest activation in each layer with forward hooks. The 4B model in fp32 won't fit on a T4, so
     use the CPU for its fp32 reference; a handful of prompts is enough.
2. **Microbenchmarks.**
   - Measure t_T, t_D and t_V(k) for k = 1…9, at context lengths of about 256 and 1024.
   - Run each in eager mode and with a static cache plus `torch.compile(mode="reduce-overhead")`.
   - Protocol: warm up first, call `torch.cuda.synchronize()` around the timed region, and take the median of at
     least 20 runs.
3. **End-to-end baselines** on a small dev subset (not Spec-Bench):
   - plain target, both eager and compiled;
   - HF assisted generation, once with a fixed γ = 4 (set `num_assistant_tokens`, a `"constant"` schedule, and
     disable `assistant_confidence_threshold` in the assistant's generation config) and once with the defaults;
   - prompt lookup decoding (`prompt_lookup_num_tokens`).
4. **Baseline acceptance.**
   - Generate target references (greedy and T=1.0) for about 50 dev prompts per category.
   - Compute the top-1 match rate, the mean 1 − TVD, and simulated τ(γ) for γ = 1…8.
   - Write the simulator now (§7.5). It is about 30 lines.
5. **Predict.** Plug the measurements into S(γ) and write a one-page `results/phase0.md`.

**Decision gates**

| Observation | Action |
|---|---|
| fp16 overflows in the 4B | Keep the offending modules in fp32. If that fails, move to a bfloat16-capable GPU (L4) and note it in the write-up. |
| Compiled c ≤ 0.2 | Proceed as planned. |
| Compiled c > 0.3 | Pull Phase 2b (pruning) forward. It is the main lever. |
| CUDA graphs don't work with HF's static cache in the pinned version | Try another `transformers` version, or capture the step manually with `torch.cuda.graphs`. If both fail, continue in eager mode and make pruning central. |
| Baseline α is already very high in a category | Expect little distillation gain there, and say so in the write-up rather than chasing it. |

---

## 6. Phase 1: The decoder (1–2 weeks)

### 6.1 Design

`seq` is the full token list (prompt plus generated tokens). **Invariant:** each model's KV cache holds a prefix of
`seq`, and before any forward pass you feed exactly the tokens that cache is missing.

```
prefill: target and draft each process prompt[:-1]      # later rounds have fixed shapes
loop:
    feed the draft the tokens its cache is missing        # 1 token, or 2 after an all-accepted round
    sample d_1 from q_1; for j = 2..γ: feed d_{j-1}, sample d_j from q_j
    feed the target [seq[-1], d_1, ..., d_γ]  →  p_1 .. p_{γ+1}
    n, y = accept_or_resample(p, q, d)
    seq += d_1..d_n, y                          # stop at EOS or max_new_tokens
    crop the target cache to len(seq) - 1
    crop the draft cache to min(draft_len, len(seq) - 1)
```

After an all-accepted round the draft cache is two tokens short, because d_γ was never fed back in. The "feed
what's missing" rule handles that case with no special code.

### 6.2 The acceptance rule

This version adds a greedy branch, avoids dividing by q, and guards the residual against floating-point
round-off:

```python
import torch

def accept_or_resample(p, q, draft_tokens, greedy=False):
    """p: [γ+1, V] target probs after the decoding warps (fp32).
    q: [γ, V] the exact distributions the draft sampled from. draft_tokens: [γ] long.
    Returns (accepted draft tokens, one token from the target)."""
    g = draft_tokens.shape[0]
    idx = torch.arange(g, device=p.device)
    if greedy:
        accepted = draft_tokens == p[:g].argmax(-1)
    else:
        u = torch.rand(g, device=p.device)
        accepted = u * q[idx, draft_tokens] < p[idx, draft_tokens]  # u < min(1, p/q)
    n = int(accepted.long().cumprod(0).sum())    # length of the accepted prefix
    if greedy:
        next_token = p[n].argmax(-1, keepdim=True)
    elif n < g:                                  # first rejection at position n
        residual = (p[n] - q[n]).clamp_min(0)
        total = residual.sum()                   # > 0 whenever a rejection happened
        next_token = torch.multinomial(residual / total if total > 0 else p[n], 1)
    else:                                        # all accepted: bonus token
        next_token = torch.multinomial(p[g], 1)
    return draft_tokens[:n], next_token
```

### 6.3 Two cache backends

- **`DynamicCache`** comes first, for correctness. Rollback is `crop()`.
- **`StaticCache` + `torch.compile(mode="reduce-overhead")`** comes second, for speed.
  - Rolling back just moves the write position (pass `cache_position` explicitly). Stale entries past that
    position are hidden by the causal mask. Test this against a fresh forward pass rather than assuming it.
  - Keep the number of shapes small, since each one is a separate graph capture. The draft always makes 1-token
    calls (do the 2-token catch-up as two calls). The target makes 1-token calls (baseline) and (γ+1)-token calls
    (verification). Run the prefill eagerly.
- **The target-alone baseline must use the same static cache and compile.**

### 6.4 Tests

These run on a CPU with `pytest`, using tiny random `Qwen3ForCausalLM` models (2 layers, hidden size 64, small
vocab). Make the draft a *noised copy* of the target so that acceptance is neither near 0 nor near 1 and both
the accept and reject paths get exercised.

1. **Acceptance-rule chi-square.** Draw random p and q over a vocab of 8–16 and run the rule about 100k times. The
   first emitted token must follow p (chi-square test at a significance level of 0.001). Also test the greedy
   branch.
2. **End-to-end distribution.** Use a vocab of 16 and continuations of length 2. Compute the exact target
   probabilities of all 256 sequences by enumeration, then chi-square-test about 20k speculative samples against
   them. This catches cache, bonus-token and warping bugs that test 1 can't see.
3. **Greedy equivalence.** With tiny fp32 models, output must be identical to plain greedy for γ = 1…8 over many
   prompts. On the GPU with the real fp16 models, run a dev subset and check the top-2 logit gap at any
   divergence, then document the near-ties.
4. **Cache consistency.** After every round, next-token logits from the cache must match a fresh full-sequence
   forward pass (within tolerance). Run this for both backends.
5. **Offline/online agreement.** Greedy τ from the simulator must equal the online accepted counts exactly.
   Sampling τ must agree within Monte Carlo error.
6. **Edge cases.** EOS inside an accepted block; `max_new_tokens` hit mid-round; γ = 1; a 1-token prompt; several
   all-accepted rounds in a row.

### 6.5 Instrumentation

Record, for each round: the number accepted; draft, verify and overhead times (CUDA events, switched off during
headline timing runs); and each rejected token with its context, for the failure analysis.

**Done when** all tests pass, the GPU greedy check is clean, the decoder is at least as fast as HF assisted
generation at the same γ, and online τ matches offline τ.

---

## 7. Phase 2: Distillation (2–3 weeks)

### 7.1 Data

- **Training prompts (~30–60k):**
  - chat: `HuggingFaceH4/ultrachat_200k`
  - code: e.g. `ise-uiuc/Magicoder-Evol-Instruct-110K`
  - math: the `openai/gsm8k` **train** split plus a subset of `meta-math/MetaMathQA`

  Keep the mix generic. Adding summarization and translation prompts from train splits would target Spec-Bench
  categories directly, so run that only as a labeled ablation.
- **Decontamination.** Normalize text, then drop any training prompt that shares a 13-gram with a Spec-Bench
  prompt. Also exact-match against the GSM8K test questions and the CNN/DM test articles.
- **Splits:**
  - *train*
  - *dev* (~300 prompts, about 50 per Spec-Bench-like category, drawn from held-out and validation data): used
    for model selection
  - *Spec-Bench* (480 prompts in 6 categories): used **only** for final numbers

### 7.2 Where training responses come from

| Source | How | Cost | Notes |
|---|---|---|---|
| Fixed text | The dataset's own responses | Free | Off-policy for both models (UltraChat responses came from ChatGPT) |
| Target-generated | Qwen3-4B samples at T=1.0 | Most expensive to generate | Closest to what the draft sees during decoding |
| Draft-generated | Qwen3-0.6B samples, scored by the target | About 5× cheaper | DistillSpec found it works well |
| *Stretch:* on-policy | Regenerate from the current draft every N steps | More moving parts | GKD/DistillSpec style |

In every case, run both models over the same text (teacher forcing) and compute the loss only at positions whose
next token is part of the response.

### 7.3 Losses

| Loss | What it optimizes |
|---|---|
| `sft` | Cross-entropy on the text. On target-generated text this is sequence-level KD, the "no logits needed" baseline. |
| `fkl`, KL(p‖q) | Mass-covering: the draft covers everything the target might say |
| `rkl`, KL(q‖p) | Mode-seeking: the draft concentrates on the target's likeliest tokens |
| `tvd` | Exactly 1 − expected acceptance when sampling |
| `jsd` *(optional)* | Symmetric and bounded; one of DistillSpec's candidates |

```python
import math
import torch
import torch.nn.functional as F

def distill_loss(draft_logits, target_logits, labels=None, kind="fkl", T=1.0):
    """[N, V] logits at positions whose next token is in the response, sliced to len(tokenizer).
    target_logits come from torch.no_grad(). labels: [N] next tokens, used only by 'sft'."""
    q_log = F.log_softmax(draft_logits.float() / T, dim=-1)
    if kind == "sft":
        return F.nll_loss(q_log, labels)
    p_log = F.log_softmax(target_logits.float() / T, dim=-1)
    p, q = p_log.exp(), q_log.exp()
    if kind == "fkl":
        per_pos = (p * (p_log - q_log)).sum(-1)
    elif kind == "rkl":
        per_pos = (q * (q_log - p_log)).sum(-1)
    elif kind == "tvd":
        per_pos = 0.5 * (p - q).abs().sum(-1)
    elif kind == "jsd":
        m_log = torch.logaddexp(p_log, q_log) - math.log(2)
        per_pos = 0.5 * (p * (p_log - m_log)).sum(-1) + 0.5 * (q * (q_log - m_log)).sum(-1)
    return per_pos.mean()
```

Apply the loss over **chunks of about 256 positions**, with each chunk's lm_head and loss wrapped in
`torch.utils.checkpoint`, so full `[N, 151,669]` fp32 tensors never exist at the same time. Distill at T=1 by
default. As an ablation, distill at the decoding temperature for the T=0.7 mode.

### 7.4 Training setup

- **Two-GPU pipeline.** GPU0 runs the target forward pass (`no_grad`, fp16) and sends response-position logits
  (fp16, sliced) to GPU1, which runs the draft's forward and backward passes. Launch the target pass for the
  *next* micro-batch before the draft step for the current one, so both GPUs stay busy.
- **Precision.** fp32 master weights, fp16 autocast and a `GradScaler`, with losses computed in fp32. If a loss is
  not finite, skip the step, log it, and watch the scaler's scale.
- **Memory.** 0.6B parameters with fp32 weights, gradients and 8-bit AdamW states come to about 6 GB before
  activations. To fit:
  - micro-batches of 1–2 sequences of at most about 2k tokens, with gradient accumulation to about 64–128k
    tokens per optimizer step;
  - gradient checkpointing;
  - `logits_to_keep` or a response-only lm_head, plus the chunked loss above.

  If memory is still tight, freeze the tied embedding (~155M of the 0.6B parameters) and treat that as an
  ablation.
- **Hyperparameters (starting point).** AdamW, lr 2e-5 (also try 1e-5 and 5e-5 in a 2M-token pilot), cosine
  decay, 2% warmup, no weight decay, gradient clipping at 1.0, one epoch.
- **Checkpointing.** Push to the HF Hub every 30–45 minutes: model, optimizer, scaler, data cursor and RNG
  state. Resume automatically, because Kaggle sessions die.
- **Throughput estimate.** About 5–8M response tokens per hour with the pipeline, so roughly 1.5–2 hours per
  10M-token run. Confirm this in the pilot.

### 7.5 Offline evaluation (the workhorse)

For each checkpoint, on dev references generated by the target (greedy, T=0.7 and T=1.0), compute:

- **top-1 match rate**, which is α for greedy decoding;
- **mean 1 − TVD** at T=0.7 and T=1.0, which is α for sampling;
- **simulated τ(γ)** for γ = 1…8, and from it the predicted speedup S(γ) using the measured c, v and o.

**Why the simulator is exact.**

- *Greedy.* The speculative output is the target's greedy text, and the draft's context at every position is the
  accepted prefix of that text. The teacher-forced top-1 matches therefore determine every round exactly.
- *Sampling.* In one speculative step, the probability that the emitted token x came from an accepted draft guess
  is min(p(x), q(x)) / p(x). So if the reference text is sampled from the target (with the same temperature and
  warps), drawing an independent Bernoulli with probability min(1, qᵢ(xᵢ)/pᵢ(xᵢ)) at each drafted position
  reproduces the joint distribution of outputs and round boundaries. Averaging over draws gives the expected τ.
  The mean of those probabilities over x ~ p is exactly 1 − TVD.

```python
import random

def simulate_tokens_per_step(accept_prob, gamma, draws=20, seed=0):
    """accept_prob[i]: chance the draft's guess for response token i is accepted, given the reference prefix.
    Greedy: 1.0 if the draft's top-1 equals the reference token, else 0.0 (one draw is enough).
    Sampling: min(1, q_i(x_i) / p_i(x_i)), where x was sampled from the target with the same settings.
    Returns the expected number of tokens emitted per target forward pass."""
    rng = random.Random(seed)
    T, steps = len(accept_prob), 0
    for _ in range(draws):
        i = 0
        while i < T:
            n = 0
            while n < gamma and i + n < T and rng.random() < accept_prob[i + n]:
                n += 1
            i += n + 1        # n accepted guesses plus one token from the target
            steps += 1
    return draws * T / steps
```

**Caveats**

- The reference must come from the target with the same decoding settings.
- Batched (padded) fp16 generation can differ slightly from batch-1 decoding. For the exact greedy agreement test,
  generate references at batch size 1.
- Rare fp16 near-ties still apply.

### 7.6 Phase 2 grid (budget-aware)

| Stage | Runs | Purpose |
|---|---|---|
| A. Pilot | 1 × ~2M tokens | lr sanity, NaN check, throughput → recalibrate the budget |
| B. Loss comparison | `sft`, `fkl`, `rkl`, `tvd` on target-generated data, ~10M tokens each | Which loss for which decoding mode |
| C. Data comparison | Best 2 losses from B on fixed text and draft-generated data | Which data source |
| D. Scale-up | Best (source, loss) pair to ~50M tokens, evaluating checkpoints along the way | Final draft, plus the acceptance-vs-training-tokens curve |
| *E. Stretch* | On-policy distillation; a code-only draft | Beyond fixed data; cross-domain transfer |

Evaluate every run offline on the dev set for greedy, T=0.7 and T=1.0, per category.

---

## 8. Phase 2b: Make the draft cheaper (optional, ~1 week)

Pull this phase forward if compiled c comes out above 0.3.

### 8.1 Layer pruning plus distillation

- **Score the layers.** Block influence (ShortGPT) is BIₗ = 1 − mean cos(hₗ_in, hₗ_out) on calibration text. Drop
  the lowest-scoring layers and always keep the last one.
- **Variants:** 28 layers (unpruned), 20, 14 and 10.
- **Recover** each variant with the best distillation recipe, using about 10–15M tokens. These runs are cheap
  because the models are smaller.
- **Bandwidth-bound c:** 14 layers ≈ 0.09 and 10 layers ≈ 0.08 before vocab trimming, and 10 layers ≈ 0.05 with
  a 32k vocab.
- **Plot:** compiled c against offline α for each variant, with iso-speedup contours from S(γ*). **This α–c Pareto
  plot is the centerpiece of the write-up.**

### 8.2 Draft vocab trimming (FR-Spec style)

- **How:** restrict the draft's lm_head to the K most frequent tokens in target-generated text (K = 16k or 32k),
  and set q = 0 everywhere else.
- **Why it stays exact:** the draft never proposes a trimmed token, and the residual max(0, p − q) is just p on
  those tokens.
- **What it saves:** the lm_head is about 26% of the draft's bytes per step, and a larger share after pruning.
- **What it costs:** α drops by the target probability mass that falls outside the kept set. Measure it.
- **Training:** train with the trimmed head, untied from the embedding.
- **Caveat:** a trimmed draft no longer matches the target's vocabulary, so it won't work in llama.cpp. Release
  the untrimmed version for GGUF.

### 8.3 Confidence-based early stopping (dynamic γ)

- **Rule:** stop drafting when the draft's top probability falls below a threshold h. Sweep h from 0.2 to 0.6 and
  compare with a fixed γ (see SpecDec++, and HF's `assistant_confidence_threshold`).
- **Offline:** the simulator still gives τ exactly, but the draft's cost only approximately. After a rejection
  the draft keeps going on its own wrong continuation, and teacher forcing can't observe how far. Confirm online.

---

## 9. Phase 3: Experiments (~2 weeks)

### 9.1 Methods

| Method | Role |
|---|---|
| Target alone, compiled (static cache + CUDA graphs) | **The reference for every speedup** |
| Target alone, eager | Shows how much compiling alone buys |
| HF assisted generation with the off-the-shelf 0.6B (defaults, and fixed γ) | Checks the custom decoder's speed |
| Prompt lookup decoding (mainly greedy) | Strong copy-based baseline; hard to beat on summarization and RAG |
| Custom decoder + off-the-shelf 0.6B | What distillation has to beat |
| Custom decoder + best distilled draft | Main result |
| Custom decoder + best pruned/trimmed draft | Result of the cost track, if Phase 2b ran |
| HF assisted generation + distilled draft | Shows the draft helps in stock HF too |

**Decoding modes:** greedy, T=0.7 and T=1.0 (pure temperature). Also run one realistic configuration for the best
draft: Qwen's recommended non-thinking settings (T=0.7, top-p 0.8, top-k 20).

### 9.2 Protocol

- **Offline, on all of Spec-Bench** (480 prompts; every draft × mode × γ = 1…8 × category).
  - Generate target references once per mode (greedy, T=0.7, T=1.0) and reuse them everywhere.
  - Outputs: τ, α and predicted speedup.
- **Online wall-clock, on selected configurations.**
  - Each method at its best γ, plus a full γ sweep for the off-the-shelf and best distilled drafts.
  - Use a stratified Spec-Bench subset of about 40 prompts per category, with `max_new_tokens=512`.
  - Both models on one T4.
- **Timing hygiene.**
  - Warm up first, including graph capture, and call `torch.cuda.synchronize()` before reading the clock.
  - **Interleave methods** (A, B, A, B, …) so thermal throttling and noisy neighbours don't bias any one method.
  - Run at least 3 repeats and report the median and interquartile range.
  - Log `nvidia-smi` clocks and temperature.
- **Metrics**
  - **τ:** tokens per target forward pass during decoding.
  - **α:** the MLE, accepted ÷ (accepted + rejections).
  - **c, v, o**, tokens per second, speedup over the compiled target, and predicted speedup.
  - Sampling runs use several seeds.

### 9.3 Failure analysis

This is computed from the offline passes, so it covers the whole evaluation set.

- **Where rejections happen:**
  - *By token class:* digits, capitalized or name-like tokens, sentence openings, punctuation, whitespace and
    newlines, code identifiers.
  - *By position:* the first few tokens of the response versus later ones.
  - *By category.*
- **Calibration:** plot the draft's top probability against the acceptance rate. This motivates §8.3.
- **What distillation fixed:** the change in α per token class, before versus after training.

### 9.4 Plots and tables for the write-up

1. **Headline table:** τ and speedup per method per Spec-Bench category, for each decoding mode.
2. **Predicted vs measured speedup:** a scatter against the y = x line, with gaps explained by v and o.
3. **Loss × data-source heatmaps**, one per decoding mode. Does TVD win for sampling and SFT/FKL for greedy?
4. **τ and speedup vs γ**, with the optimal γ per category.
5. **α–c Pareto frontier** with iso-speedup contours (Phase 2b).
6. **Acceptance vs training tokens.**
7. **Rejection rate by token class**, before and after distillation.

---

## 10. Phase 4: Release (a few days)

- **HF Hub.** Upload the best draft (plus the pruned variant, if any) with a model card covering:
  - intended use: as an assistant model for Qwen3-4B in non-thinking mode;
  - a usage snippet, `model.generate(..., assistant_model=draft)`;
  - the training data and recipe;
  - the evaluation table;
  - limitations: it is not a standalone chat model, and the timings are specific to the T4;
  - license: Apache-2.0.
- **README.** Lead with the headline table, the predicted-vs-measured plot and the Pareto plot, followed by a
  short method section and the commands to reproduce the results.
- **Write-up** (blog post or report) with the analyses from §9.
- **GGUF / llama.cpp.**
  - Convert the untrimmed draft with `convert_hf_to_gguf.py`. A pruned draft is still the Qwen3 architecture, so
    it converts fine.
  - Run it as a draft model (`llama-speculative`, or `llama-server` with `--model-draft`) on the laptop and report
    tokens per second.
- **Tag a release** and archive the result JSONs.

**Resume bullet (template):** "Implemented speculative decoding and KL/TVD distillation in PyTorch; trained a
distilled (and layer-pruned) Qwen3 draft that raised tokens per target step from X to Y and sped up Qwen3-4B
generation Z× on a T4 (Spec-Bench), with an offline simulator that predicts acceptance without decoding."

---

## 11. Timeline and milestones

These assume part-time work. GPU jobs such as data generation run on Kaggle while coding continues locally.

| Week | Work | Milestone |
|---|---|---|
| 0 | Setup; Phase 0 | **M0:** `results/phase0.md` with c, v, o, baseline α and a go/no-go decision |
| 1–2 | Phase 1 decoder, tests, simulator; start target data generation | **M1:** tests green; online τ = offline τ; at least as fast as HF assisted generation |
| 3 | Data pipeline (prompts, decontamination, dev set); training loop; pilot | **M2:** pilot improves dev α with no NaNs |
| 4–5 | Phase 2 stages B–D | **M3:** best recipe chosen; acceptance-vs-tokens curve |
| 6 | Phase 2b (optional) | **M4:** Pareto plot |
| 7–8 | Phase 3 | **M5:** headline table and all plots |
| 9 | Phase 4 | **M6:** Hub release, README, write-up |

**MVP cut line (about 6 weeks):**

- Phases 0 and 1;
- target-generated data with the `sft`, `fkl` and `tvd` losses;
- online evaluation of the best draft against the off-the-shelf draft, HF assisted generation and the target;
- a README with the headline table and the predicted-vs-measured plot.

Everything else is an extension.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| fp16 overflow in either model | Check on day one (§5). Keep the offending modules in fp32, or fall back to a bfloat16-capable GPU. |
| c stays high even when compiled | Layer pruning, vocab trimming and early stopping (§8); smaller γ. |
| `torch.compile` / CUDA graphs break with HF caches | Pin versions; capture graphs manually; worst case, work in eager mode and focus on pruning. |
| Kaggle quota or session timeouts | Budget (§4.3), small grid runs, Hub checkpoints every 30–45 minutes, automatic resume. |
| Distillation gains are small (same family, already well aligned) | Report per category, since gains concentrate where baseline α is low. The pruning track still pays off because distillation is what makes pruned drafts usable. |
| Noisy timing on shared hardware | Medians and IQR, interleaved runs, repeats, logged clocks and temperature. |
| HF API churn | Pin versions; keep a thin wrapper around cache operations. |
| Test-set contamination | 13-gram decontamination; Spec-Bench used only for final numbers. |
| Scope creep | Respect the MVP line; stretch items go last. |

---

## 13. Positioning and related work

Feature-level draft heads such as **EAGLE-2/3**, **Medusa** and **HASS** are the current state of the art. A
one-layer head that reads the target's hidden states gets both a lower c and a higher acceptance than a separate
small model. This project still studies a **standalone draft**, for three reasons:

1. It is a clean, controlled setting for studying distillation data and losses.
2. The result drops into any runtime that accepts a separate draft model, such as HF assisted generation or
   llama.cpp, with no custom code.
3. On a T4, the cost side of the trade-off is interesting in its own right.

Acknowledge EAGLE-style heads in the write-up. As a stretch goal, compare against a public EAGLE-3 head for
Qwen3-4B if one runs on your hardware.

**References**

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. arXiv:2211.17192
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023. arXiv:2302.01318
- Zhou et al. *DistillSpec: Improving Speculative Decoding via Knowledge Distillation.* ICLR 2024. arXiv:2310.08461
- Agarwal et al. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (GKD). ICLR 2024. arXiv:2306.13649
- Kim, Rush. *Sequence-Level Knowledge Distillation.* EMNLP 2016. arXiv:1606.07947
- Xia et al. *Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding* (Spec-Bench). ACL Findings 2024. arXiv:2401.07851
- Li et al. *EAGLE* (arXiv:2401.15077), *EAGLE-2* (arXiv:2406.16858), *EAGLE-3* (arXiv:2503.01840)
- Cai et al. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* 2024. arXiv:2401.10774
- Men et al. *ShortGPT: Layers in Large Language Models are More Redundant Than You Expect.* 2024. arXiv:2403.03853
- Gromov et al. *The Unreasonable Ineffectiveness of the Deeper Layers.* 2024. arXiv:2403.17887
- Zhao et al. *FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling.* 2025. arXiv:2502.14856
- Huang et al. *SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths.* 2024. arXiv:2405.19715
- Qwen Team. *Qwen3 Technical Report.* 2025. arXiv:2505.09388
- Saxena. *Prompt Lookup Decoding.* 2023. github.com/apoorvumang/prompt-lookup-decoding

---

## 14. Decisions and open questions

**Decided (2026-09-21)**

- Pair: Qwen3-0.6B → Qwen3-4B, in non-thinking mode, using fp16.
- Primary hardware: Kaggle 2×T4. Timing runs on a single T4.
- The offline simulator is the main evaluation tool. Online runs confirm results and measure wall-clock speed.
- Every speedup is measured against the compiled target.

**Open**

1. **Target variant:** `Qwen3-4B` (default) or `Qwen3-4B-Instruct-2507`. Decide after measuring baseline α for
   both in Phase 0.
2. **Hardware:** Kaggle only (default), or is a local NVIDIA GPU or cloud credit available?
3. **Is Phase 2b in scope?** The default is yes after the MVP, and earlier if compiled c > 0.3.
4. **Domain-targeted training prompts** (summarization and translation from train splits): the default is to run
   them only as a labeled ablation.
5. **Weekly time budget.** The timeline assumes part-time work; adjust the milestones once Phase 0 shows the real
   pace.
