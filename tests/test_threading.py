"""Threads must change the speed and nothing else.

Each output row is computed start to finish by one worker, so neither the thread count nor
the schedule changes the summation order. That makes the engine's results reproducible and
keeps the bit-exact k-token property intact no matter how the pool is configured — asserted
here as exact equality, not closeness.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from specdraft import _engine as cpp  # noqa: E402
from specdraft.export import write_model  # noqa: E402
from specdraft.reference import TINY_CONFIG, random_reference  # noqa: E402

CONFIGURATIONS = [
    {"threads": 1, "cores": "performance"},
    {"cores": "performance"},
    {"cores": "physical"},
    {"cores": "physical", "dynamic_schedule": True},
    {"cores": "logical"},
    {"cores": "any"},
    {"threads": 3, "cores": "logical", "dynamic_schedule": True},
]


@pytest.fixture(scope="module")
def weights_file(tmp_path_factory):
    model = random_reference(TINY_CONFIG, seed=0)
    path = tmp_path_factory.mktemp("threading") / "tiny-q4.sdm"
    write_model(path, model.state_dict(), model.config, weight_format="q4")
    return str(path)


def tokens(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, TINY_CONFIG.vocab_size, size=n, dtype=np.int32)


# ------------------------------------------------------------------------ topology


def test_topology_is_reported():
    cores = cpp.core_topology()
    assert cores, "expected at least one logical processor"
    assert all(set(core) == {"logical_index", "core_index", "efficiency_class", "primary"} for core in cores)
    assert sum(1 for core in cores if core["primary"]) == len({c["core_index"] for c in cores})


def test_selections_are_nested():
    performance = set(cpp.cores_for("performance"))
    physical = set(cpp.cores_for("physical"))
    logical = set(cpp.cores_for("logical"))
    assert performance <= physical <= logical
    assert performance, "there must be at least one performance core"


def test_unknown_selection_is_rejected(weights_file):
    with pytest.raises(Exception):
        cpp.cores_for("efficiency")
    with pytest.raises(Exception):
        cpp.Model(weights_file, max_positions=8, cores="turbo")


def test_read_bandwidth_is_plausible():
    speed = cpp.measure_read_bandwidth(1 << 24, 0, "performance", 1)
    assert 0.5 < speed < 2000.0, f"{speed} GB/s is not a believable figure"


# -------------------------------------------------------------- results are identical


@pytest.mark.parametrize("options", CONFIGURATIONS[1:])
def test_every_thread_configuration_gives_identical_logits(weights_file, options):
    ids = tokens(9, seed=1)
    single = cpp.Model(weights_file, max_positions=32, threads=1, cores="performance")
    other = cpp.Model(weights_file, max_positions=32, **options)
    assert np.array_equal(
        single.forward(ids, all_logits=True), other.forward(ids, all_logits=True)
    ), f"{options} changed the result"


@pytest.mark.parametrize("options", CONFIGURATIONS)
def test_k_tokens_stay_bit_exact_under_threading(weights_file, options):
    ids = tokens(7, seed=2)
    together = cpp.Model(weights_file, max_positions=32, **options).forward(ids, all_logits=True)

    stepwise = cpp.Model(weights_file, max_positions=32, **options)
    rows = [stepwise.forward(ids[i : i + 1], all_logits=True)[0] for i in range(len(ids))]
    assert np.array_equal(together, np.stack(rows))


def test_thread_count_is_reported(weights_file):
    one = cpp.Model(weights_file, max_positions=8, threads=1)
    assert one.threads == 1
    assert one.core_selection == "performance"
    assert not one.dynamic_schedule

    many = cpp.Model(weights_file, max_positions=8, cores="logical", dynamic_schedule=True)
    assert many.threads == len(cpp.cores_for("logical"))
    assert many.core_selection == "logical" and many.dynamic_schedule


# ----------------------------------------------------------------------- timings


def test_timings_are_off_until_asked_for(weights_file):
    model = cpp.Model(weights_file, max_positions=16)
    model.forward(tokens(3, seed=3))
    assert model.timings()["total"] == 0.0
    assert model.timings()["tokens"] == 0

    model.set_timing(True)
    model.forward(tokens(3, seed=4))
    timings = model.timings()
    assert timings["tokens"] == 3
    assert timings["total"] > 0.0
    assert {"qkv", "attention", "attn_out", "gate_up", "ffn_down", "output", "embed"} <= set(timings)
    assert timings["qkv"] > 0.0

    model.reset_timings()
    assert model.timings()["total"] == 0.0


def test_byte_counts_match_the_file(weights_file):
    model = cpp.Model(weights_file, max_positions=8)
    config = model.config
    # 4-bit blocks are 18 bytes per 32 weights, and the output layer reads vocab_limit rows.
    per_row = config["hidden_size"] // 32 * 18
    assert model.weight_bytes_per_token > per_row * config["vocab_limit"]
    expected_kv = (
        config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"] * 2 * 2
    )
    assert model.kv_bytes_per_token == expected_kv
