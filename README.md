# draft-model

Distilling a small draft model to speed up **Qwen3-4B** with speculative decoding on a single T4 GPU.

The project covers:

- a from-scratch speculative decoder;
- a comparison of distillation data sources and losses (SFT, forward/reverse KL, TVD) under greedy decoding and
  sampling;
- cheaper drafts through layer pruning and vocabulary trimming;
- predicted-versus-measured speedup analysis on Spec-Bench.

**Status:** planning. See [PLAN.md](PLAN.md) for the full plan.
