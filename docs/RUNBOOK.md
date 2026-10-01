# Runbook: measuring the thing

The order to run the measurements in, what each one costs, and what it should produce. Written for
the laptop in this repo's plan (i7-13620H, 16 GB), with the toolchain and Qwen3-0.6B already in
place.

## Before any timing run

The plan's §13 rules exist because this machine is a noisy instrument: samples of one build have
ranged 40–70 tok/s, and medians taken minutes apart have differed by 75%.

- [ ] Plugged in.
- [x] **Power mode: nothing to do.** Checked on this machine: the only scheme Windows 11 exposes is
      "Balanced", but the *power mode overlay* on AC is already `ded574b5...`, which is Best
      performance. The classic High performance scheme is hidden, and switching schemes would change
      nothing. (`powercfg /list` shows the scheme; the overlay lives under
      `HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes` as
      `ActiveOverlayAcPowerScheme`.) What is left is thermal headroom on a 45 W chip and the OS
      preempting threads that are spin-waiting, neither of which is a setting.
- [ ] Other programs closed; OneDrive, Windows Update and search indexing paused.
- [ ] Nothing else of yours running: the engine pins threads to the performance cores and spins on
      them, so a background build will show up in the numbers.

Everything below writes JSON into `results/`, so a run is reproducible and comparable later.

## 1. Export the target (≈2 minutes, CPU-light)

```bash
python scripts/export_model.py models/Qwen3-4B --format q4
python scripts/export_model.py models/Qwen3-4B --format q4 --output-format q8   # optional variant
```

Expect about 2.3 GB at 4.50 bits per weight. The file format is version 3; anything exported
earlier must be re-exported.

## 2. The ceiling, and where the time goes (≈5 minutes)

```bash
python scripts/bench_engine.py models/Qwen3-4B-q4.sdm --context 128
```

Reports measured read bandwidth (the ceiling every speed is a fraction of), decode and prefill
speed for five thread configurations, and the per-stage breakdown. On the 0.6B this gave 37–39 GB/s
and six performance cores winning over all ten.

**What to look for:** the 4B reads about 2.3 GB per token, so the ceiling is roughly 16 tok/s. If
the engine lands near half of that, the thread pool and kernels are doing their job.

## 3. v(k): what verification actually costs (≈10 minutes)

This is the term the project adds to the speedup formula, and the number that decides the best γ.

```bash
python scripts/measure_vk.py models/Qwen3-4B-q4.sdm --draft models/Qwen3-0.6B-q4.sdm \
    --context 128 --max-k 12
```

Prints v(k) for k = 1…12, measures c, and predicts the speedup at each γ for several acceptance
rates. **This is the first real test of the project's premise:** on the 0.6B, v(5) came out near
3.1, because a small model's weights are modest next to its arithmetic. The 4B reads seven times
the weights per token, so its curve should be much flatter — if it is not, the honest finding is
that CPU verification is expensive and small γ wins.

## 4. Acceptance, without decoding (≈20 minutes)

The cheap grid. Generate the target's own continuations once, then score any draft against them.

```bash
# prompts to score on; a few hundred is plenty for acceptance
python scripts/generate_data.py prompts --out data/dev_prompts.jsonl \
    --mix ultrachat=200,code=50,gsm8k=50

python scripts/generate_references.py --model models/Qwen3-4B-q4.sdm \
    --tokenizer models/Qwen3-4B --prompts data/dev_prompts.jsonl \
    --mode greedy --max-new-tokens 128 --out data/references_greedy.jsonl

python scripts/offline_acceptance.py \
    --target models/Qwen3-4B-q4.sdm --draft models/Qwen3-0.6B-q4.sdm \
    --references data/references_greedy.jsonl --mode greedy \
    --tokenizer models/Qwen3-4B --c <c from step 3> --vk results/vk_Qwen3-4B-q4.json \
    --gammas 1 2 3 4 5 6 8 --out results/offline_baseline.json
```

Generating references is the slow part (the 4B at ~15 tok/s), so keep `--max-new-tokens` modest the
first time. The scoring itself is one forward pass per model.

**What to look for:** the baseline acceptance of the off-the-shelf 0.6B against the 4B. That single
number, with c and v(k), predicts the whole speedup — and tells you how much distillation has to
win to be worth the Kaggle hours.

## 5. End to end, on Spec-Bench (≈30 minutes for a subset)

```bash
python scripts/run_specbench.py \
    --target models/Qwen3-4B-q4.sdm --draft models/Qwen3-0.6B-q4.sdm \
    --tokenizer models/Qwen3-4B --questions data/spec_bench/question.jsonl \
    --gamma <best from step 3> --per-category 5 --max-new-tokens 128 \
    --out results/specbench_baseline.json
```

The question file is not bundled: take `data/spec_bench/question.jsonl` from
github.com/hemingkx/Spec-Bench, or any JSON-lines file with `category` and `turns`.

Runs target-alone, the draft, prompt lookup and early stopping, interleaved per question. Compare
the measured speedup against step 4's prediction — the gap between them is the interesting part, and
is what `o`, the per-round overhead, is for.

## 6. Sanity checks worth keeping

```bash
python scripts/engine_twin_agreement.py models/Qwen3-4B-q4.sdm models/Qwen3-4B
```

Confirms the PyTorch twin still predicts the engine (0.6B: 98.9% top-1, 0.023 TVD). Re-run whenever
the engine's arithmetic changes, since everything scored on a GPU depends on it.

## Then: the drafts worth training

Nothing above needs a GPU. These do, and they run on Kaggle:

```bash
# one cell of the grid
python scripts/generate_data.py prompts --out data/prompts.jsonl \
    --eval-prompts data/spec_bench/question.jsonl          # decontaminate, or the numbers mean nothing
python scripts/generate_data.py responses --prompts data/prompts.jsonl \
    --source target --model models/Qwen3-4B --out data/target_generated.jsonl
python scripts/train_draft.py --student models/Qwen3-0.6B --teacher models/Qwen3-4B \
    --data data/target_generated.jsonl --loss tvd --teacher-format q4 \
    --teacher-device cuda:0 --student-device cuda:1 --amp \
    --max-tokens 10000000 --out runs/tvd-target
```

Then bring the result back to step 4 and step 5 to see what it bought.

The cheaper draft variants, which trade acceptance for cost and need distillation to pay off:

```bash
python scripts/prune_draft.py --model models/Qwen3-0.6B --keep 20 --out models/Qwen3-0.6B-keep20
python scripts/trim_vocab.py --model models/Qwen3-0.6B --keep 32768 \
    --data data/target_generated.jsonl --out models/Qwen3-0.6B-trim32k-q4.sdm
```

Count the trimming frequencies on the target's **own generations**, not on convenient prose: a set
chosen from WikiText covered 100% of WikiText but missed 23% of the same model's chat output.
