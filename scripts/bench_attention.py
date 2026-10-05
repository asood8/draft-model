"""Attention's inner loop on its own, separated from the rest of a forward pass.

    python scripts/bench_attention.py
    python scripts/bench_attention.py --positions 2048 --threads 6

At context 2048 attention is 27% of a decode step on the 4B and reads the KV cache at about
11 GB/s, on a machine that streams 39.8. That looks like a memory problem and is not one, which is
why this script exists: it runs the same cache with three different inner loops, back to back in one
process, so the comparison survives a machine that drifts.

* **scan** converts the cached rows and sums them, and nothing else. That is the floor the access
  pattern allows, and the number everything else is read against.
* **engine** is what `Model::attention` does: one position at a time, convert the key row, one dot
  product per query head in the group, then a second pass converting values and accumulating.
* **blocked** is the same arithmetic with positions taken in blocks, so several cache rows are
  converted before any of them is used.
* **fused** converts each cached row in registers as it is multiplied, with no scratch buffer at
  all. Same accumulator structure, so the same answer to the last bit.
* **grouped** converts each row once and feeds every query head that shares it, which is what the
  other three all fail to do: a key row belongs to a kv head and is read by every query head in its
  group, so converting it per head is `group` times the work for one row's worth of data.

`--jobs-per-head` cuts each key/value head's positions into that many parallel jobs. The engine uses
one, which gives 8 jobs for 6 workers and cannot balance; this says what that is worth.
"""
from __future__ import annotations

import argparse

from specdraft import _engine as cpp

SHAPES = ("scan", "engine", "blocked", "fused", "grouped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--positions", type=int, default=1024, help="context length to simulate")
    parser.add_argument("--layers", type=int, default=36, help="36 is the 4B; sizes the sweep past L3")
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--group", type=int, default=4, help="query heads per key/value head")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--jobs-per-head", type=int, nargs="*", default=[1, 3, 6])
    parser.add_argument("--block", type=int, default=8, help="positions per block, for `blocked`")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--cores", default="performance")
    parser.add_argument("--seconds", type=float, default=0.3, help="target time per measurement")
    parser.add_argument("--repeats", type=int, default=3, help="measurements per point; best kept")
    parser.add_argument("--shapes", nargs="*", default=list(SHAPES), choices=SHAPES)
    return parser.parse_args()


def run(args: argparse.Namespace, shape: str, jobs_per_head: int) -> dict:
    """Time one point, choosing the iteration count so the measurement lasts about --seconds."""
    def call(iters: int) -> dict:
        return cpp.bench_attention(
            positions=args.positions, layers=args.layers, kv_heads=args.kv_heads,
            group=args.group, head_dim=args.head_dim, jobs_per_head=jobs_per_head,
            block=args.block, iters=iters, threads=args.threads, cores=args.cores, shape=shape,
        )

    probe = call(1)
    iters = max(1, int(args.seconds / max(probe["seconds"], 1e-6)))
    best = None
    for _ in range(args.repeats):
        point = call(iters)
        if best is None or point["gb_per_second"] > best["gb_per_second"]:
            best = point
    return best


def main() -> None:
    args = parse_args()
    bytes_per_token = 2 * args.layers * args.kv_heads * args.positions * args.head_dim * 2
    print(f"KV cache for one token at context {args.positions}: {bytes_per_token / 1e6:.0f} MB "
          f"({args.layers} layers, {args.kv_heads} kv heads, {args.group} query heads each)")
    print(f"{args.threads} thread(s) on {args.cores} cores, best of {args.repeats}")
    print()
    print(f"{'shape':>9} {'jobs/head':>10} {'jobs':>5} {'ms/token':>10} {'GB/s':>8} {'vs scan':>8}")

    floor = {}
    for shape in args.shapes:
        for jobs_per_head in args.jobs_per_head:
            point = run(args, shape, jobs_per_head)
            per_token = point["seconds"] / max(1, point["cache_bytes_read"] / point["cache_bytes"])
            floor.setdefault(jobs_per_head, point["gb_per_second"] if shape == "scan" else None)
            reference = floor.get(jobs_per_head)
            share = f"{point['gb_per_second'] / reference:>7.2f}x" if reference else "       -"
            print(f"{shape:>9} {jobs_per_head:>10} {point['jobs']:>5} {per_token * 1e3:>9.2f}ms "
                  f"{point['gb_per_second']:>8.1f} {share:>8}")

    print()
    print("ms/token is what this would cost inside one decode step. `scan` is the floor the access")
    print("pattern allows; the distance from it is arithmetic and score writes, not memory.")


if __name__ == "__main__":
    main()
