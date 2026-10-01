# draft-model (in progress)

Speculative decoding for **Qwen3-4B** on a laptop CPU, built from scratch.

The project has four parts:

- a C++ inference engine with 4-bit and 8-bit AVX-VNNI kernels, a thread pool that uses both core types, and an
  fp16 KV cache;
- speculative decoding inside that engine, with bit-exact greedy verification;
- a Qwen3-0.6B draft distilled to match the engine's quantized target;
- predicted-versus-measured speedup analysis on Spec-Bench, including how much each extra guess costs to verify on
  a CPU.

**Status:** planning. See [PLAN.md](PLAN.md) for the full plan.
