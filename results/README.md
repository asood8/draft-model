# What is in here, and under what conditions it was taken

Absolute timings from this laptop are only comparable against runs taken under the same conditions
(plan section 13). A file name that says only what was measured is therefore not enough, so the
directories carry the date and the machine's state as well.

**Quote the `2026-10-25` files.** That session is the only one in this project's history in which
every number came from a single machine state, and it supersedes the stitched figures rather than
merely disagreeing with them. See "Why 2026-10-25 supersedes everything" below.

| Path | Conditions | Use it for |
|---|---|---|
| `conditions_2026-10-25_quiet_{start,end}.json` | 2026-10-25, WSL down, Chrome closed | **The session's conditions, measured at both ends.** Bandwidth drifted 0.12% between them. |
| `vk_2026-10-25_quiet_pass{A,B}_ctx*/` | 2026-10-25, two shuffled orders | **The v(k), c and o table quoted in the plan.** |
| `specbench_2026-10-25_quiet_pass{A,B}_gamma{1..4}.json` | 2026-10-25, each gamma in two orders, four methods | **The gamma sweep, and the three-way drafting comparison.** |
| `stages_2026-10-25_quiet.json` | 2026-10-25, two shuffled sweeps | **Per-stage cost of a decode step**, and the attention rewrite's value. |
| `bench_Qwen3-4B-q4.json` | 2026-10-04, quiet | The byte accounting (`weight_bytes_per_token`), and the 2026-10-04 15.59 tok/s for comparison only. |
| `vk_context{512,1024,2048}/`, `vk_Qwen3-4B-q4.json` | 2026-10-04/05, quiet | The previous v(k) table. **Superseded** by the 2026-10-25 runs. |
| `specbench_gamma{1,2,3,4}.json` | 2026-10-05, quiet, **ascending order** | **Do not quote.** Confounded by its own visit order -- see below. |
| `specbench_{short,long}.json` | 2026-10-05, quiet | The short/long prompt split and the prefill shares. Not re-measured since. |
| `vk_2026-10-24_quiet_pass{A,B}_ctx*/` | 2026-10-24, browser closed | **Superseded.** Kept because the night's reasoning is recorded against them. |
| `vk_2026-10-24_browser_open_*/` | 2026-10-24, browser open | **Superseded**, see below. |
| `bench_2026-10-24_quiet/` | 2026-10-24, browser closed | The conditions of 2026-10-24, for the record. |
| `bench_*.json` (others) | see each file's own fields | Bandwidth, dispatch, per-stage. |

## Why 2026-10-25 supersedes everything

The conditions were measured at both ends of a session that ran eight v(k) runs, eight Spec-Bench
runs, two per-stage sweeps and 900 tests in between:

| | start | end | drift |
|---|---|---|---|
| dispatch overhead, fresh process | 1.474 us | 1.192 us | better |
| read bandwidth, performance cores | 40.19 GB/s | 40.14 GB/s | **0.12%** |
| free memory | 8.28 GB | 8.25 GB | -- |
| page-ins | 0 | 0 | -- |

Two things follow. First, every figure from that session is comparable with every other figure from
it, which is what the plan's section 13 has been asking for since 2026-10-02. Second, the 15% spread
that made 2026-10-24 unusable did not recur: gamma=1's two passes agree to **0.2% on wall clock and
0.4% on decode**.

## The stored gamma sweep is confounded and must not be quoted

`specbench_gamma{1,2,3,4}.json` were written at **01:38:38, 01:41:31, 01:44:15 and 01:47:01** on
2026-10-05 -- strictly ascending gamma, back to back, nine minutes for four runs, no cooldowns. Every
larger gamma therefore inherited the heat of the runs before it. Its decode speedups decline
monotonically at 1.2011, 1.0073, 0.9514 and 0.8823, which is exactly the pattern this project banned
for v(k) sweeps on 2026-10-24, on the grounds that a monotonic drift cannot be separated from the
monotonic effect being measured. It survived here because the result agreed with the theory.

Re-measured on 2026-10-25 with each gamma visited in two different orders:

| gamma | stored decode | order-cancelled | stored wall | order-cancelled |
|---|---|---|---|---|
| 1 | 1.2011 | 1.1932 (-0.7%) | 1.0903 | 1.0731 (-1.6%) |
| 2 | 1.0073 | **1.1998 (+19.1%)** | 0.9114 | 0.9856 (+8.1%) |
| 3 | 0.9514 | **1.0618 (+11.6%)** | 0.8482 | 0.9512 (+12.1%) |
| 4 | 0.8823 | **0.9824 (+11.3%)** | 0.8175 | 0.8439 (+3.2%) |

Gamma=1 reproduces and every larger gamma reads 11-19% high. In each pass, whichever gamma ran first
won it, so **gamma=1 and gamma=2 are tied on decode**; gamma=1 wins on wall clock, by 8.9%, in both
passes.

## What reproduced, and the one thing that did not

The **ratios reproduce**. At context 128, against 2026-10-04: v(2) 1.130/1.120 against 1.122, v(4)
1.616/1.632 against 1.636, c 0.182 against 0.186, o 0.00707/0.00729 against 0.00712. So the v(k)
table this project is built on is sound, and the 2026-10-24 reading that the curve had *flattened*
was an artefact of an inflated pass(1) dividing it down -- which is what an earlier version of this
file suspected when it declined to quote those runs as absolutes.

The **absolute step time did not**, and the direction depends on context:

| context | 2026-10-04 | 2026-10-25 | |
|---|---|---|---|
| 128 | 64.10 ms | 72.62 / 73.06 | **+13.3%** |
| 512 | 79.93 | 85.97 / 84.31 | +6.5% |
| 1024 | 86.14 | 77.02 / 79.57 | **-9.0%** |
| 2048 | 98.36 | 91.00 / 90.58 | **-7.7%** |

A slower machine is slower everywhere, so this is not the machine -- and the long-context half is
the attention rewrite, which `stages_2026-10-25_quiet.json` measures at 1.27x, 1.60x, 1.87x and
2.15x faster at the four contexts.

The short-context 13.3% is **real and unexplained**. It is corroborated twice on each side:
2026-10-04 measured it with `bench_engine` at 15.59 tok/s = 64.13 ms and with `measure_vk` at 64.10
ms, agreeing to 0.05%, and 2026-10-25 measured it with `measure_vk` at 72.62/73.06 and the stage
timer at 73.29/74.14. An earlier reading of this -- that 64.10 was a cherry-picked low figure
because that day's stage timer said 70.32 ms -- was **wrong**, and was the fifth bad guess at this
gap: the stage timer is instrumented and reading high is what instrumentation does.

What it is not:

- **Not bandwidth.** 40.19 GB/s, the highest ever recorded on this machine, against 39.77 on
  2026-10-04. The decode loop went from 88.7% of that ceiling to 77.5% of a higher one.
- **Not the clock.** `o` is an absolute per-round cost of pure non-streaming work and did not move.
- **Not the code.** The 2026-10-04 bench ran on commit `e33a0b6`. Rebuilt from that commit with the
  same MSVC 19.51 and the same `/O2 /arch:AVX2 /fp:precise`, then alternated against HEAD across
  eight fresh processes with the order flipped each round, it gives head/base = **0.975** on decode
  and 1.6% on the stage total.

It is spread across the weight-streaming stages -- qkv 1.12x, output 1.10x, gate_up 1.05x, attn_out
1.05x, ffn_down 0.98x -- with no single culprit. Five guesses have been wrong, so it is left open.

## Why the 2026-10-24 runs are kept

They are superseded in full: the 2026-10-25 session measures the same quantities under conditions
that held, and the 2026-10-24 ones were taken on a machine whose unchanged baseline binary read
13.18, 11.49, 12.39 and 12.36 tok/s across four rounds -- a 15% range from one binary. They are kept
because the plan's 2026-10-24 log row and section 13 reason against them by name, and deleting the
evidence for an argument would leave the argument unsupported.

`vk_2026-10-24_browser_open_*` is kept for the same reason and is worse still: it was taken
immediately after ten calls of the dispatch-overhead probe, which runs thousands of all-core
barriers and heats the package, and it reported v(2) = 0.97 -- two tokens verified for less than the
price of one, which cannot happen.
