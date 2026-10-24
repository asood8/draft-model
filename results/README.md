# What is in here, and under what conditions it was taken

Absolute timings from this laptop are only comparable against runs taken under the same conditions
(plan section 13). A file name that says only what was measured is therefore not enough, so the
directories carry the date and the machine's state as well.

| Path | Conditions | Use it for |
|---|---|---|
| `vk_context{512,1024,2048}/` | 2026-10-05, machine deliberately quiet | **The v(k) table quoted in the plan.** |
| `vk_Qwen3-4B-q4.json`, `vk_Qwen3-0.6B-q4.json` | 2026-10-04, quiet | The headline v(k) and c at context 128. |
| `vk_2026-10-24_busy_*/` | 2026-10-24, browser open, 5.6 GB of 15.7 free | **Not comparable to the above.** Kept for the record only. |
| `specbench_gamma{1,2,3,4}.json` | 2026-10-04, quiet | The gamma sweep. |
| `bench_*.json` | see each file's own fields | Bandwidth, dispatch, per-stage. |

## Why `vk_2026-10-24_busy_*` is kept but not used

The run is real and was taken carefully -- cool, contexts descending, three samples -- but every
weight-streaming stage measured 1.05-1.25x above the 2026-10-05 figures while attention, the only
stage whose code had changed, measured 1.43 ms against 1.50. A uniform penalty on the stages that
stream weights and none on the stage that was rewritten is a slower machine, not a regression. Read
bandwidth was still 39.0 GB/s, so it was the clock and not memory.

Since v(k) is the ratio pass(k)/pass(1), an inflated pass(1) flattens the curve for free, and the
apparent finding in that run -- that v(k) no longer depends on context -- cannot be separated from
the machine's state. It needs retaking quiet before it can be believed or quoted.

An earlier sweep the same night, taken in ascending order, reported v(2) = 0.966: two tokens
verified for less than the price of one. It also **overwrote three of the 2026-10-05 files**, which
were recovered from git. That is the reason for this file: a results directory whose names do not
record the conditions invites exactly that mistake.
