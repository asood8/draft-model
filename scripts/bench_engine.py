"""Measure the engine: bandwidth ceiling, tokens per second, and where the time goes.

    python scripts/bench_engine.py models/Qwen3-0.6B-q4.sdm

Every speed is reported as a percentage of this machine's measured read bandwidth, because
single-token decoding streams the weights once per token and little else. The thread
configurations compared are the four from plan §9.4: performance cores only, all physical
cores with a fixed split, all physical cores with dynamic chunks, and every hyperthread.

Benchmarking rules from plan §13 apply: plug in, fix the power mode, close other programs.
Methods are interleaved and medians reported, so thermal drift does not favour whichever ran
first.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
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


def decode_speed(model, tokens: int, warmup: int = 3) -> float:
    """Tokens per second for single-token decoding, the number that matters."""
    ids = np.array([3] * (tokens + warmup), dtype=np.int32)
    model.reset()
    for i in range(warmup):
        model.forward(ids[i : i + 1])
    started = time.perf_counter()
    for i in range(warmup, warmup + tokens):
        model.forward(ids[i : i + 1])
    return tokens / (time.perf_counter() - started)


def prefill_speed(model, tokens: int) -> float:
    ids = np.array([5] * tokens, dtype=np.int32)
    model.reset()
    started = time.perf_counter()
    model.forward(ids)
    return tokens / (time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--decode-tokens", type=int, default=24)
    parser.add_argument("--prefill-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    topology = cpp.core_topology()
    classes = sorted({core["efficiency_class"] for core in topology})
    physical = len({core["core_index"] for core in topology})
    print(f"CPU: {len(topology)} logical processors, {physical} physical cores, "
          f"efficiency classes {classes}")
    print(f"kernel path: {cpp.kernel_path()}")

    print("\nread bandwidth (the ceiling):")
    bandwidth = {}
    for name, selection in (("performance", "performance"), ("physical", "physical"),
                            ("logical", "logical")):
        value = cpp.measure_read_bandwidth(1 << 30, 0, selection, 3)
        bandwidth[name] = value
        print(f"  {name:<12} {value:6.1f} GB/s  ({len(cpp.cores_for(selection))} threads)")
    ceiling = max(bandwidth.values())

    probe = cpp.Model(str(args.model), max_positions=64)
    weight_bytes = probe.weight_bytes_per_token
    kv_bytes = probe.kv_bytes_per_token
    print(f"\nweights read per token: {weight_bytes / 1e6:.1f} MB"
          f"   KV cache per token of context: {kv_bytes / 1e3:.1f} KB")
    print(f"ceiling at {ceiling:.1f} GB/s: {ceiling * 1e9 / weight_bytes:.1f} tok/s "
          "(weights only, short context)")
    del probe

    rows = []
    models = {}
    for name, options in CONFIGURATIONS:
        models[name] = cpp.Model(
            str(args.model), max_positions=args.decode_tokens + args.prefill_tokens + 8, **options
        )

    print(f"\n{'configuration':<30} {'threads':>7} {'decode':>12} {'% ceiling':>10} {'prefill':>10}")
    samples: dict[str, list[float]] = {name: [] for name, _ in CONFIGURATIONS}
    prefills: dict[str, list[float]] = {name: [] for name, _ in CONFIGURATIONS}
    for _ in range(args.repeats):  # interleaved, so thermal drift hits every method equally
        for name, _options in CONFIGURATIONS:
            samples[name].append(decode_speed(models[name], args.decode_tokens))
            prefills[name].append(prefill_speed(models[name], args.prefill_tokens))

    for name, _options in CONFIGURATIONS:
        speed = statistics.median(samples[name])
        prefill = statistics.median(prefills[name])
        share = speed * weight_bytes / (ceiling * 1e9)
        print(f"{name:<30} {models[name].threads:>7} {speed:>9.2f} t/s {share:>9.1%} "
              f"{prefill:>7.2f} t/s")
        rows.append(
            {
                "configuration": name,
                "threads": models[name].threads,
                "decode_tokens_per_second": speed,
                "decode_samples": samples[name],
                "prefill_tokens_per_second": prefill,
                "share_of_ceiling": share,
            }
        )

    # Where the time goes, for the fastest configuration.
    best = max(rows, key=lambda row: row["decode_tokens_per_second"])["configuration"]
    model = models[best]
    model.set_timing(True)
    model.reset_timings()
    decode_speed(model, args.decode_tokens)
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
        "logical_processors": len(topology),
        "physical_cores": physical,
        "kernel_path": cpp.kernel_path(),
        "read_bandwidth_gb_s": bandwidth,
        "weight_bytes_per_token": weight_bytes,
        "kv_bytes_per_token": kv_bytes,
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
