# What is in here, and under what conditions it was taken

Absolute timings from this laptop are only comparable against runs taken under the same conditions
(plan section 13). A file name that says only what was measured is therefore not enough, so the
directories carry the date and the machine's state as well.

| Path | Conditions | Use it for |
|---|---|---|
| `vk_context{512,1024,2048}/` | 2026-10-05, machine deliberately quiet | **The v(k) table quoted in the plan.** |
| `vk_Qwen3-4B-q4.json`, `vk_Qwen3-0.6B-q4.json` | 2026-10-04, quiet | The headline v(k) and c at context 128. |
| `bench_Qwen3-4B-q4.json` | 2026-10-04, quiet | The headline 15.59 tok/s and the bandwidth ceiling. |
| `specbench_gamma{1,2,3,4}.json` | 2026-10-05, quiet | The gamma sweep. |
| `vk_2026-10-24_browser_open_*/` | 2026-10-24, browser open | **Superseded**, see below. |
| `vk_2026-10-24_quiet_pass{A,B}_ctx*/` | 2026-10-24, browser closed, two shuffled orders | v(k) *ratios* only, not absolute timings. |
| `bench_2026-10-24_quiet/` | 2026-10-24, browser closed | The conditions of 2026-10-24, for the record. |
| `bench_*.json` (others) | see each file's own fields | Bandwidth, dispatch, per-stage. |

## The 2026-10-04 conditions have not been reproducible since

On 2026-10-24 every absolute figure measured 20-25% worse than the 2026-10-04 run: pass(1) at
context 128 read 77.10 ms with a browser open, then 78.10 and 80.29 ms with it closed, against
**64.10 ms** on 2026-10-04. Closing the browser moved nothing, which disposes of the first
explanation -- recorded here in an earlier version of this file -- that the machine was merely busy.

**The engine is not the cause, and that was established rather than assumed.** The 2026-10-04 bench
ran on commit `e33a0b6`, before any attention change. Rebuilt from that commit with the same MSVC
19.51 and the same `/O2 /arch:AVX2 /fp:precise`, then alternated against HEAD across eight fresh
processes with the order flipped each round, it gives head/base = **0.975** on decode and 1.6% on the
stage total: the two builds are indistinguishable, and the old source reproduces 2026-10-24's figure
rather than its own.

What changed is the spread as much as the speed. That same unchanged baseline binary read 13.18,
11.49, 12.39 and 12.36 tok/s across four rounds -- a **15% range from one binary** -- where
2026-10-04's nine samples at six threads spread **0.8%**. Supporting conditions, all recorded in
`bench_2026-10-24_quiet/`: read bandwidth 37.4 GB/s over four fresh processes against 39.77, dispatch
overhead 0.876 us against 0.702, performance cores sustaining 2.75-2.92 GHz under all-core decode,
and memory and build flags unchanged. The mechanism is **not established**; three guesses at it have
been wrong, so it is left open.

So: **quote the 2026-10-04 and 2026-10-05 files for absolute speed**, and do not replace them with
2026-10-24 figures, which measure a worse-behaved machine and not a worse engine. To compare any
future number against them, rebuild that commit and measure it alongside rather than differencing
against the stored value.

## Why the 2026-10-24 v(k) runs are ratios only

v(k) = pass(k)/pass(1), and an inflated pass(1) flattens the curve for free, so these runs cannot be
quoted as the new v(k) table. They are still informative in one direction. Section 13 records that a
*slower* machine reports a **steeper** v(k), because the clock governs the arithmetic in the
numerator while the denominator is bandwidth-bound; 2026-10-01 measured exactly that, v(2) reading
1.45 cool and 1.63 hot. Tonight's runs read **flatter** at every context -- v(2) of 1.23 at context
2048 against 1.456 on 2026-10-05 -- so the flattening survives a bias working against it. The
direction is supported; the magnitude is not.

Two passes were taken in different context orders (1024-128-2048-512 and 512-2048-128-1024) because
a monotonic drift is otherwise indistinguishable from the monotonic dependence on context being
measured. They agree to 2.8% at context 128 and disagree by 10-16% at the long contexts, which is the
noise floor for anything measured there that night.

## Why `vk_2026-10-24_browser_open_*` is kept but superseded

The run is real and was taken carefully -- cool, contexts descending, three samples -- and its
numbers match the browser-closed runs that replaced it, which is itself the finding. Kept for that
comparison only; the directory was named `..._busy_*` when it was thought the browser was the cause.

An earlier sweep the same evening, taken in ascending order straight after ten calls of the dispatch
probe, reported v(2) = 0.966: two tokens verified for less than the price of one. It also
**overwrote three of the 2026-10-05 files**, which were recovered from git. That is the reason for
this file: a results directory whose names do not record the conditions invites exactly that mistake.
