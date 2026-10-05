"""Measure v(k): what it costs to verify k tokens in one pass (plan section 3).

    python scripts/measure_vk.py models/Qwen3-0.6B-q4.sdm
    python scripts/measure_vk.py models/Qwen3-4B-q4.sdm --draft models/Qwen3-0.6B-q4.sdm

The original speedup formula assumes checking gamma+1 tokens costs the same as one ordinary step.
On a CPU it does not: the weights are read once however many tokens share the pass, but the
integer arithmetic grows with every extra token, so v(k) starts near 1 and climbs. Where it
starts climbing decides the best number of guesses per round, and that is the term this
project adds to the formula:

    speedup(gamma) = tau(gamma) / (gamma*c + v(gamma+1)),   tau(gamma) = (1 - alpha^(gamma+1)) / (1 - alpha)

With a draft model given, c is measured too and the whole table is predicted.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from specdraft import _engine as cpp


def time_passes(jobs: list[tuple], context: int, samples: int, inner: int = 4) -> dict:
    """Median seconds per forward pass for each job, with the jobs interleaved.

    A job is ``(label, model, tokens)``. Measuring every sample of one k before moving to the next
    lets a drifting clock masquerade as a change in v(k), and on this machine it does: a sweep taken
    that way came out non-monotonic, k=4 cheaper than k=3, which is impossible for the same work plus
    one more token. Rotating through the jobs spreads a drift over all of them, which is plan
    section 13's rule about interleaving configurations.

    What interleaving does *not* buy is independence from the machine's state, and it is tempting to
    think it does, v(k) being a ratio. Throttling cuts the clock, which governs the arithmetic, and
    leaves the memory bandwidth alone; a one-token pass is bandwidth-bound while a k-token pass is
    compute-bound, so a hot machine reports a steeper curve. Take this measurement cool.
    """
    batches = {label: np.array([7] * tokens, dtype=np.int32) for label, _, tokens in jobs}
    prepared = set()
    for label, model, _ in jobs:
        if id(model) not in prepared:
            model.reset()
            if context:
                model.forward(np.array([5] * context, dtype=np.int32))
            prepared.add(id(model))
        for _ in range(2):  # warm up this shape
            model.set_pos(context)
            model.forward(batches[label])

    timings: dict = {label: [] for label, _, _ in jobs}
    for _ in range(samples):
        for label, model, _ in jobs:
            batch = batches[label]
            started = time.perf_counter()
            for _ in range(inner):
                model.set_pos(context)  # the cached prefix stays valid; only the tail is rewritten
                model.forward(batch)
            timings[label].append((time.perf_counter() - started) / inner)
    return {label: statistics.median(values) for label, values in timings.items()}


def tau(alpha: float, gamma: int) -> float:
    if alpha >= 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="the model being verified against")
    parser.add_argument("--draft", type=Path, default=None, help="to measure c as well")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--max-k", type=int, default=16)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.6, 0.7, 0.8])
    parser.add_argument("--cores", default="performance")
    parser.add_argument("--measure-overhead", action="store_true", default=True,
                        help="also measure o, the per-round overhead (needs --draft)")
    parser.add_argument("--no-measure-overhead", dest="measure_overhead", action="store_false")
    parser.add_argument("--overhead-gamma", type=int, default=4)
    parser.add_argument("--overhead-tokens", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    # Room for the longest thing either measurement does: the k-token sweep, or the overhead run
    # which generates tokens on top of the context.
    # The overhead measurement runs two budgets, the longer being twice the first.
    capacity = args.context + max(args.max_k, 2 * args.overhead_tokens + args.overhead_gamma) + 8

    def open_model(path: Path):
        # max_batch has to cover the widest pass being timed, and it is also the chunk the context
        # prefill is fed in, which wants at least 16: a prompt pushed through in max_k-token passes
        # costs 40.6 ms a token at 2 against 27.4 at 16 on this machine. It cannot change what is
        # measured here, since a k-token pass is one chunk either way.
        return cpp.Model(
            str(path), max_positions=capacity, cores=args.cores,
            max_batch=max(args.max_k, 16)
        )

    target = open_model(args.target)
    print(f"target {args.target.name}: {target.threads} threads on {target.core_selection} cores, "
          f"context {args.context}")

    # The draft joins the rotation rather than being timed afterwards, so c is a ratio taken under
    # the same conditions as v(k) -- and so that the draft's weights compete for cache with the
    # target's, which is what happens during real speculative decoding.
    draft = open_model(args.draft) if args.draft is not None else None
    jobs = [(k, target, k) for k in range(1, args.max_k + 1)]
    if draft is not None:
        jobs.append(("draft", draft, 1))
    measured = time_passes(jobs, args.context, args.samples)

    costs = {k: measured[k] for k in range(1, args.max_k + 1)}
    single = costs[1]

    print(f"\none target step: {single * 1e3:.2f} ms  ({1 / single:.1f} tok/s)")
    print(f"\n{'k':>3} {'pass (ms)':>10} {'v(k)':>7} {'per token':>11}")
    for k, seconds in costs.items():
        print(f"{k:>3} {seconds * 1e3:>10.2f} {seconds / single:>7.2f} "
              f"{seconds / k * 1e3:>9.2f} ms")

    record = {
        "target": str(args.target),
        "context": args.context,
        "cores": args.cores,
        "threads": target.threads,
        "target_step_seconds": single,
        "v": {str(k): seconds / single for k, seconds in costs.items()},
        "pass_seconds": {str(k): seconds for k, seconds in costs.items()},
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    draft_step = 0.0
    if draft is not None:
        draft_step = measured["draft"]
        del draft
        c = draft_step / single
        record["draft"] = str(args.draft)
        record["draft_step_seconds"] = draft_step
        record["c"] = c
        print(f"\none draft step: {draft_step * 1e3:.2f} ms   c = {c:.3f}")

        print(f"\npredicted speedup, using measured v(gamma+1) and c = {c:.3f}")
        header = "  gamma " + "".join(f"  alpha={a:<5.2f}" for a in args.alphas) + "  v(gamma+1)"
        print(header)
        best = {a: (0.0, 0) for a in args.alphas}
        rows = []
        for gamma in range(1, args.max_k):
            v = costs[gamma + 1] / single
            line = f"{gamma:>3} "
            entry = {"gamma": gamma, "v": v, "speedup": {}}
            for alpha in args.alphas:
                speedup = tau(alpha, gamma) / (gamma * c + v)
                entry["speedup"][str(alpha)] = speedup
                line += f"  {speedup:>6.2f} "
                if speedup > best[alpha][0]:
                    best[alpha] = (speedup, gamma)
            rows.append(entry)
            print(line + f"  {v:>6.2f}")
        record["predictions"] = rows
        record["best"] = {str(a): {"speedup": s, "gamma": g} for a, (s, g) in best.items()}
        print()
        for alpha in args.alphas:
            speedup, gamma = best[alpha]
            print(f"best at alpha={alpha:.2f}: {speedup:.2f}x with gamma={gamma}")

    if args.draft is not None and args.measure_overhead:
        # The third term in the formula. A round costs gamma draft steps plus one verification
        # pass; whatever is left over -- sampling over a 152k vocabulary, the acceptance test,
        # cache bookkeeping -- is o. Without it, any gap between predicted and measured speedup
        # has nowhere to go.
        #
        # Taken from the engine's own per-stage timers rather than by subtracting one run from
        # another: the models report how long they spent computing, and whatever the round loop
        # took beyond that is overhead. Differencing two runs was tried first and failed, because
        # this machine's run-to-run variance is larger than the quantity being measured.
        gamma = min(args.overhead_gamma, args.max_k - 1)
        # Reopened: the round loop needs both models timing-enabled, and closing them first keeps
        # one 4B in memory rather than two, with one thread pool rather than two spinning.
        del target
        target = open_model(args.target)
        draft = open_model(args.draft)
        for model in (target, draft):
            model.set_timing(True)
            model.reset_timings()

        _, stats = cpp.generate_speculative(
            target, draft, [7] * args.context, max_new_tokens=args.overhead_tokens, gamma=gamma
        )
        # The prompt pass is taken out of both sides. It is neither a round nor free, so charging its
        # share to the rounds would inflate o -- measured, it happens not to, because a prefill's wall
        # time equals its model time to the microsecond and the two corrections cancel exactly. The
        # formula is still the right one, and it stays right if that ever stops being true.
        model_seconds = target.timings()["total"] + draft.timings()["total"]
        rounds = max(1, stats["rounds"])
        round_wall = stats["rounds_seconds"]
        round_model = model_seconds - stats["prefill_model_seconds"]
        per_round = (round_wall - round_model) / rounds
        o = per_round / single

        # Where that overhead actually sits. The round loop times its own sections, so o can be
        # attributed rather than guessed at -- which it was, wrongly, three times. Everything here is
        # per round.
        #   dispatch:    forward calls minus what the models' stage timers claim, so thread wake-ups,
        #                barriers, and anything between stages
        #   draft work:  the draft's sampling: argmaxes over the vocabulary, one per guess
        #   accept:      the acceptance test, which is another argmax per guess when greedy
        #   bookkeeping: appending tokens, the stop check, rewinding both caches
        forwards = stats["draft_forward_seconds"] + stats["verify_seconds"]
        parts = {
            "dispatch": (forwards - round_model) / rounds,
            "draft sampling": stats["draft_sampling_seconds"] / rounds,
            "accept": stats["accept_seconds"] / rounds,
            "bookkeeping": stats["bookkeeping_seconds"] / rounds,
        }

        record["overhead"] = {
            "gamma": gamma,
            "rounds": stats["rounds"],
            "total_seconds": stats["seconds"],
            "model_seconds": model_seconds,
            "seconds_per_round": per_round,
            "o_per_round": o,
            "parts_seconds_per_round": parts,
            "tokens_per_target_forward": stats["tokens_per_target_forward"],
            "alpha": stats["alpha"],
        }
        print()
        print(f"per-round overhead at gamma={gamma}: {per_round * 1e3:.2f} ms, "
              f"o = {o:.3f} target steps")
        print(f"  ({round_wall * 1e3:.0f} ms in the rounds, of which "
              f"{round_model * 1e3:.0f} ms was the models computing, over {rounds} rounds)")
        print(f"  (the prompt pass, excluded from both: {stats['prefill_seconds'] * 1e3:.0f} ms wall, "
              f"{stats['prefill_model_seconds'] * 1e3:.0f} ms of it computing)")
        print(f"  that run: tau {stats['tokens_per_target_forward']:.2f}, "
              f"alpha {stats['alpha']:.3f}")
        # The four sections cover the round loop exactly, so this is a decomposition of o rather
        # than a list of suspects. It is not the prompt pass, which is excluded above, and not the
        # softmax, which greedy decoding never calls; see PLAN.md section 10.4.
        print("  where it goes, per round:")
        for name, seconds in sorted(parts.items(), key=lambda kv: -kv[1]):
            share = seconds / per_round if per_round > 0 else 0.0
            print(f"    {name:<15} {seconds * 1e3:>7.3f} ms  {100 * share:>5.1f}%  "
                  f"o = {seconds / single:.3f}")
        del target, draft

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"vk_{args.target.stem}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
