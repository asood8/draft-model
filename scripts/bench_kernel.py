"""The k-token kernel on its own, separated from the memory system.

    python scripts/bench_kernel.py
    python scripts/bench_kernel.py --n-in 2560 --max-tokens 8

Measuring v(k) through the engine (scripts/measure_vk.py) gives the number the speedup model needs,
but it cannot say *why* the curve has the slope it has. Two quite different things are mixed in it:
how efficiently the kernel issues its multiply-accumulates, and how often it waits for weights to
arrive from memory. This script separates them by choosing how many weight rows to sweep:

* a working set inside L1 or L2 is a pure throughput figure -- every weight is already in cache, so
  what remains is the instruction cost of the inner loop;
* a working set far larger than L3 pays the real memory cost, which is what a decode step pays.

The gap between the two is how much of v(k) is memory and how much is arithmetic. The figure to
watch is operations per multiply-accumulate: one AVX-VNNI `dpbusd` does 32 of them, so a kernel
doing nothing but multiply-accumulates on two issue ports would reach 64 per cycle.
"""
from __future__ import annotations

import argparse

from specdraft import _engine as cpp

# Assumed only for the cycles-per-dpbusd column, which is a diagnostic, not a result.
ASSUMED_GHZ = 3.5
DPBUSD_MACS = 32  # 32 int8 products per instruction
DPBUSD_PORTS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-in", type=int, default=2560, help="input width; 2560 is the 4B hidden size")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--format", choices=("q4", "q8"), default="q4")
    parser.add_argument("--seconds", type=float, default=0.35, help="target time per measurement")
    parser.add_argument("--repeats", type=int, default=3, help="measurements per point; the best is kept")
    parser.add_argument("--layouts", nargs="*", default=["interleaved", "split-flat", "split"],
                        choices=("interleaved", "split-flat", "split"),
                        help="split is what the engine runs; the others are the kernels it replaced")
    parser.add_argument("--threads", type=int, default=1,
                        help="6 asks the question the engine cares about, where bandwidth binds")
    return parser.parse_args()


def run(rows: int, n_in: int, tokens: int, fmt: str, seconds: float, repeats: int,
        layout: str = "split", threads: int = 1) -> dict:
    """Time one point, choosing the iteration count so each measurement lasts about `seconds`.

    `layout` picks the kernel. "split" is the one the engine runs: scales and quantized bytes in
    separate arrays, two blocks read per 32-byte load. "split-flat" is the same layout read one block
    at a time, the kernel before pair-packing. "interleaved" is the original, each scale beside its own
    bytes. Only q4 has all three, that being the format the engine stores weights in.

    `threads` matters more than it looks. On one thread the memory system is nowhere near saturated, so
    a kernel that wins there need not win on six, where bandwidth is the constraint -- and comparing
    layouts at one thread answers the wrong question. All of them are timed in this one process, which
    on this machine is the only comparison worth making: three attempts to compare kernels across
    separate runs gave three different answers, because an unchanged baseline drifted by a third
    between them.
    """
    if layout in ("split", "split-flat"):
        if fmt != "q4":
            raise SystemExit("the split-layout kernels are only benchmarked for q4")
        variant = "flat" if layout == "split-flat" else "paired"

        def call(iters):
            return cpp.bench_dot_soa(rows=rows, n_in=n_in, tokens=tokens, iters=iters,
                                     threads=threads, row_major=False, variant=variant)
    else:

        def call(iters):
            return cpp.bench_dot(rows=rows, n_in=n_in, tokens=tokens, iters=iters, format=fmt,
                                 threads=threads)

    probe = call(1)
    iters = max(1, int(seconds / max(probe["seconds"], 1e-6)))
    best = None
    for _ in range(repeats):
        result = call(iters)
        rate = result["macs"] / result["seconds"]
        if best is None or rate > best[0]:
            best = (rate, result)
    return {"rate": best[0], "iters": iters, **best[1]}


def main() -> None:
    args = parse_args()
    features = cpp.cpu_features()
    print(f"kernel path: {cpp.kernel_path()}   avx_vnni={features['avx_vnni']}")
    if not features["avx_vnni"]:
        print("no AVX-VNNI on this machine; the numbers below are the scalar fallback")

    bytes_per_row = (args.n_in // 32) * (18 if args.format == "q4" else 34)
    # One working set per level of the hierarchy. The L1 point is deliberately tiny: at 8 tokens the
    # activations alone are 23 KB of the 48 KB L1, so the weights have to be smaller still.
    levels = [
        ("L1", max(1, 16 * 1024 // bytes_per_row)),
        ("L2", max(1, 384 * 1024 // bytes_per_row)),
        ("L3", max(1, 8 * 1024 * 1024 // bytes_per_row)),
        ("DRAM", max(1, 192 * 1024 * 1024 // bytes_per_row)),
    ]

    peak = DPBUSD_MACS * DPBUSD_PORTS * ASSUMED_GHZ  # GMAC/s on one core, at the assumed clock
    print(f"\n{args.format} weights, n_in {args.n_in}, {args.threads} thread(s), "
          f"best of {args.repeats}")
    print(f"a dpbusd is {DPBUSD_MACS} multiply-accumulates and two can issue per cycle, so one core")
    print(f"at an assumed {ASSUMED_GHZ} GHz could reach {peak:.0f} GMAC/s")

    for name, rows in levels:
        working_set = rows * bytes_per_row / 1024
        unit = "KB" if working_set < 1024 else "MB"
        shown = working_set if working_set < 1024 else working_set / 1024
        print(f"\n{name}: {rows} rows, {shown:.0f} {unit} of weights")
        for layout in args.layouts:
            label = {"split": "split, two blocks a load -- the engine's kernel",
                     "split-flat": "split, one block a load -- what it replaced",
                     "interleaved": "interleaved blocks -- the original"}[layout]
            print(f"  {label}")
            print(f"    {'k':>3} {'GMAC/s':>9} {'% peak':>7} {'ops/MAC':>8} {'ns/row':>8} "
                  f"{'GB/s':>7} {'per tok':>8} {'v(k)':>6}")
            baseline = None
            for tokens in range(1, args.max_tokens + 1):
                point = run(rows, args.n_in, tokens, args.format, args.seconds, args.repeats, layout,
                            args.threads)
                rate = point["rate"] / 1e9
                per_row = point["seconds"] / (point["iters"] * rows) * 1e9
                if baseline is None:
                    baseline = per_row
                # Operations per multiply-accumulate, with "operation" meaning one issue slot: the
                # inverse of how much work a cycle gets done, scaled so a pure dpbusd stream would
                # be 1/32.
                ops_per_mac = (ASSUMED_GHZ * 1e9 * DPBUSD_PORTS) / point["rate"]
                gbs = point["weight_bytes_read"] / point["seconds"] / 1e9
                print(f"    {tokens:>3} {rate:>9.1f} {100 * rate / peak:>6.1f}% "
                      f"{ops_per_mac:>8.3f} {per_row:>8.1f} {gbs:>7.1f} "
                      f"{per_row / tokens:>7.1f}ns {per_row / baseline:>6.2f}")

    print("\nops/MAC is issue slots per multiply-accumulate: 0.031 would be a pure dpbusd stream on")
    print("two ports. v(k) here is the kernel's own curve, free of everything the engine adds.")


if __name__ == "__main__":
    main()
