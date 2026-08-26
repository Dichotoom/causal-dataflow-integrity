#!/usr/bin/env python3
"""Verify the frozen data and reported acquisition and continuation results."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_HASHES = {
    "data/fpit_train.jsonl": "863dbe27047f13d1d1805d2fb959e00dac165af93a36cd979a860466e15dc9df",
    "data/fpit_test.jsonl": "c5a485192c7bc1a007145cee805e647d37d505b91168debc1a4f84e052638b86",
}

ACQUISITION_HASHES = {
    "results/acquisition/protocol.json": "fd4b7de680b7fb1b95495bfa2be317a54c62b63c0e15f2daa7a32bda34a7610a",
    "results/acquisition/endpoint_eval_summary.json": "7a8fd69462d9292e9c5ce7b57f503ee64c09b2aa42ed2ce12882638d1d3a99cc",
    "results/acquisition/path_eval_summary.json": "c0ca0a6b64ed558e32849d3c3870997ea0a33a2ff93e9ba8f954e891b797ff87",
}

RESULT_HASHES = {
    3407: "35cd18a0e9270c16750e6d1d8e5c541fd8550595069668030647d4ebe3b2ac50",
    9173: "d92c89cf6fe9008542ec29c540daec47b38b81624f90205fa5dfd6537ecb9251",
    17011: "292bdbc113df62b4bd218cb3e68b4cd7f67dcd33d825d8d7e3223877e6257fa7",
}

PRIMARY = {
    3407: {"A": 9, "B": 16, "B_only": 7, "A_only": 0, "p": 0.015625},
    9173: {"A": 12, "B": 11, "B_only": 4, "A_only": 5, "p": 1.0},
    17011: {"A": 9, "B": 12, "B_only": 5, "A_only": 2, "p": 0.453125},
}

EXPECTED_MEAN_SD = {
    "base_accuracy": ((26.5, 2.0), (36.7, 3.5)),
    "new_accuracy": ((44.2, 1.2), (36.7, 7.4)),
    "pair_accuracy": ((20.4, 3.5), (26.5, 5.4)),
    "factor_all_correct": ((55.1, 4.1), (49.0, 6.1)),
    "path_full_accuracy": ((19.0, 4.7), (24.5, 5.4)),
}

CALCULATORS = {
    "cha2ds2_vasc": (13, [(3, 7), (5, 6), (3, 5)]),
    "sirs": (13, [(5, 8), (6, 5), (5, 6)]),
    "gcs": (5, [(1, 1), (1, 0), (1, 1)]),
    "fena": (18, [(0, 0), (0, 0), (0, 0)]),
}

SEEDS = (3407, 9173, 17011)

METRIC_LABELS = {
    "base_accuracy": "BASE endpoint accuracy",
    "new_accuracy": "NEW endpoint accuracy",
    "pair_accuracy": "Joint endpoint accuracy",
    "factor_all_correct": "Complete factor correctness",
    "path_full_accuracy": "CDI rate",
}

CALCULATOR_LABELS = {
    "cha2ds2_vasc": "CHA2DS2-VASc",
    "sirs": "SIRS",
    "gcs": "GCS",
    "fena": "FENa",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def arm_map(obj: dict) -> dict[str, dict]:
    return {arm["arm"]: arm for arm in obj["arms"]}


def primary_summary(arm: dict) -> dict:
    return arm["primary"]["summary"]["structured_primary"]


def per_calculator(arm: dict) -> dict:
    return arm["primary"]["summary"]["per_calculator"]


def prompt_hashes(arm: dict) -> tuple[str, ...]:
    rows = arm["primary"]["summary"]["case_level"]
    return tuple(sorted(row["prompt_hash"] for row in rows))


def rounded_mean_sd(values: list[float]) -> tuple[float, float]:
    return round(statistics.mean(values) * 100, 1), round(statistics.stdev(values) * 100, 1)


def count_from_metric(block: dict, metric: str = "pair_accuracy") -> int:
    return round(block[metric] * block["n_cases"])


def verify_acquisition() -> None:
    for rel, expected in ACQUISITION_HASHES.items():
        require(sha256(ROOT / rel) == expected, f"Hash mismatch for {rel}")

    protocol = load_json("results/acquisition/protocol.json")
    corpus = protocol["corpus"]
    require(protocol["seed"] == 3407 and protocol["epochs"] == 2 and protocol["K_train"] == 8, "Acquisition protocol mismatch")
    require(corpus["train_sha256"] == DATA_HASHES["data/fpit_train.jsonl"], "Protocol train hash mismatch")
    require(corpus["test_sha256"] == DATA_HASHES["data/fpit_test.jsonl"], "Protocol test hash mismatch")
    require(corpus["smoke_exposed_test_cases"] == 12, "Smoke-exposure count mismatch")
    require(corpus["confirmatory_test_cases"] == 80, "Confirmatory test count mismatch")
    require(corpus["confirmatory_structured_cases"] == 49, "Structured primary count mismatch")

    rows = {
        "Endpoint-only": load_json("results/acquisition/endpoint_eval_summary.json")["greedy"]["structured_primary"],
        "FPIT": load_json("results/acquisition/path_eval_summary.json")["greedy"]["structured_primary"],
    }
    expected = {
        "Endpoint-only": (17, 10, 6, 9, 1),
        "FPIT": (14, 18, 7, 26, 7),
    }

    print("Acquisition (Table 1)")
    for name, block in rows.items():
        require(block["n_cases"] == 49 and block["n_rollouts"] == 49, f"Acquisition n mismatch for {name}")
        got = (
            count_from_metric(block, "base_accuracy"),
            count_from_metric(block, "new_accuracy"),
            count_from_metric(block, "pair_accuracy"),
            count_from_metric(block, "factor_all_correct"),
            count_from_metric(block, "path_full_accuracy"),
        )
        require(got == expected[name], f"Acquisition mismatch for {name}: {got}")
        print(
            f"  {name}: BASE {got[0]}/49, NEW {got[1]}/49, joint {got[2]}/49, "
            f"factors {got[3]}/49, CDI {got[4]}/49"
        )


def verify_continuation() -> None:
    reference = None
    loaded = {}

    print("\nContinuation primary outcome")
    for seed in SEEDS:
        path = ROOT / f"results/continuation/seed_{seed}.json"
        require(sha256(path) == RESULT_HASHES[seed], f"Hash mismatch for {path.name}")
        obj = json.loads(path.read_text(encoding="utf-8"))
        loaded[seed] = obj
        arms = arm_map(obj)
        a, b = arms["A"], arms["B"]
        exp = PRIMARY[seed]

        require(a["primary"]["n"] == b["primary"]["n"] == 49, f"n mismatch for seed {seed}")
        require(a["primary"]["pair_correct"] == exp["A"], f"Arm A mismatch for seed {seed}")
        require(b["primary"]["pair_correct"] == exp["B"], f"Arm B mismatch for seed {seed}")

        paired = obj["paired_primary"]
        require(paired["A_wrong_B_right"] == exp["B_only"], f"B-only mismatch for seed {seed}")
        require(paired["A_right_B_wrong"] == exp["A_only"], f"A-only mismatch for seed {seed}")
        require(abs(paired["mcnemar_exact_p"] - exp["p"]) < 1e-12, f"McNemar mismatch for seed {seed}")

        ah, bh = prompt_hashes(a), prompt_hashes(b)
        require(ah == bh and len(ah) == 49, f"A/B primary-set mismatch for seed {seed}")
        if reference is None:
            reference = ah
        else:
            require(ah == reference, f"Primary set differs across seeds at seed {seed}")

        print(
            f"  {seed}: A={exp['A']}/49, B={exp['B']}/49, "
            f"B-only={exp['B_only']}, A-only={exp['A_only']}, p={exp['p']:.6g}"
        )

    print("\nDescriptive across-seed metrics (mean ± sample SD, %)")
    for metric, expected in EXPECTED_MEAN_SD.items():
        vals_a, vals_b = [], []
        for seed in SEEDS:
            arms = arm_map(loaded[seed])
            vals_a.append(primary_summary(arms["A"])[metric])
            vals_b.append(primary_summary(arms["B"])[metric])
        got_a, got_b = rounded_mean_sd(vals_a), rounded_mean_sd(vals_b)
        require(got_a == expected[0], f"Across-seed A mismatch for {metric}: {got_a}")
        require(got_b == expected[1], f"Across-seed B mismatch for {metric}: {got_b}")
        label = METRIC_LABELS[metric]
        print(f"  {label}: A={got_a[0]:.1f} ± {got_a[1]:.1f}, B={got_b[0]:.1f} ± {got_b[1]:.1f}")

    print("\nCalculator-level joint endpoint counts (A -> B)")
    for slug, (expected_n, expected_pairs) in CALCULATORS.items():
        got_pairs = []
        for seed in SEEDS:
            arms = arm_map(loaded[seed])
            a_block = per_calculator(arms["A"])[slug]
            b_block = per_calculator(arms["B"])[slug]
            require(a_block["n_cases"] == b_block["n_cases"] == expected_n, f"n mismatch for {slug}, seed {seed}")
            got_pairs.append((count_from_metric(a_block), count_from_metric(b_block)))
        require(got_pairs == expected_pairs, f"Calculator mismatch for {slug}: {got_pairs}")
        label = CALCULATOR_LABELS[slug]
        print(f"  {label} (n={expected_n}): " + ", ".join(f"{a}->{b}" for a, b in got_pairs))


def main() -> None:
    for rel, expected in DATA_HASHES.items():
        got = sha256(ROOT / rel)
        require(got == expected, f"Hash mismatch for {rel}: {got}")

    verify_acquisition()
    verify_continuation()
    print("\nPASS: frozen data and reported results verified.")


if __name__ == "__main__":
    main()
