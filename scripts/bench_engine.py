"""Measure the engine: bandwidth ceiling, tokens per second, and where the time goes.

    python scripts/bench_engine.py models/Qwen3-0.6B-q4.sdm
    python scripts/bench_engine.py models/Qwen3-0.6B-q4.sdm --quick   # for iterating

Every speed is a percentage of this machine's measured read bandwidth, because single-token
decoding streams the weights once per token and little else. The thread configurations are
the four from plan §9.4, plus one thread for reference.

**Why each sample lasts seconds.** A laptop's clocks wander: short samples of the same build
on this machine ranged from 40 to 70 tok/s, a 51% spread, which is wider than most
optimizations are worth. So the harness follows plan §13: a sustained warm-up to reach a
steady thermal state, samples measured over whole seconds, configurations interleaved so
drift hits them equally, and the median reported with the spread beside it. A result whose
interquartile range overlaps another's is not a difference.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from specdraft import _engine as cpp

CONFIGURATIONS = [
    ("1 thread", {"threads": 1, "cores": "performance"}),
    ("performance cores", {"cores": "performance"}),
    ("all physical, fixed split", {"cores": "physical"}),
    ("all physical, dynamic chunks", {"cores": "physical", "dynamic_schedule": True}),
    ("every hyperthread", {"cores": "logical"}),
]

PREFILL_TOKENS = 64
STEPS_PER_BURST = 16


def decode_for(model, seconds: float, context: int) -> float:
    """Decode speed at a fixed context length.

    Attention cost grows with the context, so the context has to be held still or the number
    means nothing. Each burst refills the cache to `context` (untimed) and then times a fixed
    number of single-token steps.
    """
    one = np.array([3], dtype=np.int32)
    model.reset()
    if context:
        model.forward(np.array([5] * context, dtype=np.int32))  # filled once, not per burst

    tokens = 0
    elapsed = 0.0
    while elapsed < seconds:
        # Rewinding is enough: the cache still holds the first `context` tokens, and the
        # entries past that point get overwritten. This is the same operation the
        # speculative decoder performs when it discards rejected guesses.
        model.set_pos(context)
        started = time.perf_counter()
        for _ in range(STEPS_PER_BURST):
            model.forward(one)
        elapsed += time.perf_counter() - started
        tokens += STEPS_PER_BURST
    return tokens / elapsed


def prefill_for(model, seconds: float) -> float:
    batch = np.array([5] * PREFILL_TOKENS, dtype=np.int32)
    tokens = 0
    elapsed = 0.0
    while elapsed < seconds:
        model.reset()
        started = time.perf_counter()
        model.forward(batch)
        elapsed += time.perf_counter() - started
        tokens += PREFILL_TOKENS
    return tokens / elapsed


def power_scheme() -> str:
    try:
        out = subprocess.run(
            ["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip()
    except Exception:  # not Windows, or powercfg unavailable
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--seconds", type=float, default=1.5, help="per sample")
    parser.add_argument("--repeats", type=int, default=9, help="samples per configuration")
    parser.add_argument("--warmup", type=float, default=20.0, help="sustained load first")
    parser.add_argument("--context", type=int, default=128, help="context length to decode at")
    parser.add_argument("--quick", action="store_true", help="short and noisy, for iterating")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.quick:
        args.seconds, args.repeats, args.warmup = 1.0, 3, 2.0

    topology = cpp.core_topology()
    physical = len({core["core_index"] for core in topology})
    print(f"CPU: {len(topology)} logical processors, {physical} physical cores, "
          f"kernel path {cpp.kernel_path()}")
    print(f"power scheme: {power_scheme()}")

    print("\nread bandwidth (the ceiling):")
    bandwidth = {}
    for selection in ("performance", "physical", "logical"):
        value = cpp.measure_read_bandwidth(1 << 30, 0, selection, 5)
        bandwidth[selection] = value
        print(f"  {selection:<12} {value:6.1f} GB/s  ({len(cpp.cores_for(selection))} threads)")
    ceiling = max(bandwidth.values())
    dispatch = cpp.measure_dispatch_overhead(3000, 0, "performance")
    print(f"dispatch overhead: {dispatch * 1e6:.2f} us per parallel job")

    # Only as much cache as the measurement needs: a big one costs a quarter of a gigabyte of
    # zeroing per model, and a model is opened for every sample.
    max_positions = args.context + STEPS_PER_BURST + 8

    def open_model(options: dict):
        return cpp.Model(str(args.model), max_positions=max_positions, **options)

    # One model at a time. Each one owns a thread pool whose idle workers spin, so keeping
    # several alive would have them stealing cycles from whichever is being measured.
    probe = open_model({"cores": "performance"})
    weight_bytes = probe.weight_bytes_per_token
    kv_bytes = probe.kv_bytes_per_token
    print(f"\nweights read per token: {weight_bytes / 1e6:.1f} MB"
          f"   KV cache per token of context: {kv_bytes / 1e3:.1f} KB")
    print(f"ceiling at {ceiling:.1f} GB/s: {ceiling * 1e9 / weight_bytes:.1f} tok/s "
          "(weights only, short context)")

    if args.warmup > 0:  # reach a steady thermal state before any number is recorded
        print(f"\nwarming up for {args.warmup:.0f}s ...")
        warmup_started = time.perf_counter()
        while time.perf_counter() - warmup_started < args.warmup:
            decode_for(probe, min(2.0, args.warmup), args.context)
    del probe

    print(f"\nmeasuring: {args.repeats} samples of {args.seconds:.0f}s per configuration at "
          f"context {args.context}, interleaved")
    decode: dict[str, list[float]] = {name: [] for name, _ in CONFIGURATIONS}
    prefill: dict[str, list[float]] = {name: [] for name, _ in CONFIGURATIONS}
    threads: dict[str, int] = {}
    for _ in range(args.repeats):
        for name, options in CONFIGURATIONS:
            model = open_model(options)
            threads[name] = model.threads
            decode[name].append(decode_for(model, args.seconds, args.context))
            prefill[name].append(prefill_for(model, args.seconds))
            del model  # joins its threads before the next configuration starts

    rows = []
    print(f"\n{'configuration':<30} {'thr':>4} {'decode':>11} {'IQR':>7} {'% ceiling':>10} {'prefill':>10}")
    for name, _options in CONFIGURATIONS:
        speed = statistics.median(decode[name])
        ordered = sorted(decode[name])
        low = statistics.median(ordered[: len(ordered) // 2])
        high = statistics.median(ordered[(len(ordered) + 1) // 2 :])
        spread = (high - low) / speed if speed else 0.0
        share = speed * weight_bytes / (ceiling * 1e9)
        prefill_speed = statistics.median(prefill[name])
        print(f"{name:<30} {threads[name]:>4} {speed:>8.1f} t/s {spread:>6.0%} "
              f"{share:>9.1%} {prefill_speed:>7.1f} t/s")
        rows.append(
            {
                "configuration": name,
                "threads": threads[name],
                "decode_tokens_per_second": speed,
                "decode_spread": spread,
                "decode_samples": decode[name],
                "prefill_tokens_per_second": prefill_speed,
                "prefill_samples": prefill[name],
                "share_of_ceiling": share,
            }
        )

    best = max(rows, key=lambda row: row["decode_tokens_per_second"])["configuration"]
    best_options = dict(CONFIGURATIONS)[best]
    model = open_model(best_options)
    model.set_timing(True)
    model.reset_timings()
    decode_for(model, args.seconds, args.context)
    timings = model.timings()
    model.set_timing(False)
    print(f"\nwhere the time goes ({best}, {timings['tokens']} tokens):")
    for stage, seconds in sorted(timings.items(), key=lambda item: -float(item[1])):
        if stage in {"total", "tokens"} or float(seconds) <= 0:
            continue
        print(f"  {stage:<12} {float(seconds) / timings['tokens'] * 1e3:7.2f} ms/token"
              f"  {float(seconds) / timings['total']:6.1%}")

    record = {
        "model_file": str(args.model),
        "cpu": platform.processor(),
        "power_scheme": power_scheme(),
        "logical_processors": len(topology),
        "physical_cores": physical,
        "kernel_path": cpp.kernel_path(),
        "read_bandwidth_gb_s": bandwidth,
        "dispatch_overhead_seconds": dispatch,
        "weight_bytes_per_token": weight_bytes,
        "kv_bytes_per_token": kv_bytes,
        "context": args.context,
        "sample_seconds": args.seconds,
        "repeats": args.repeats,
        "warmup_seconds": args.warmup,
        "rows": rows,
        "timings": {k: float(v) for k, v in timings.items()},
        "timing_configuration": best,
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"bench_{args.model.stem}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
