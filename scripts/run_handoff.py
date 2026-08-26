#!/usr/bin/env python3
"""Run the matched FPIT continuation experiment from the frozen checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


# =============================================================================
# Frozen experiment constants
# =============================================================================

EXPERIMENT_ID = "CDI_HANDOFF"

RESULTS_ARCHIVE_SHA256 = (
    "cce8a7f703c5ac86bebeb563eb528d1ceee8a822ff2dc085431f51e0a3b10df0"
)
DATA_ARCHIVE_SHA256 = (
    "8c215ca24fcd29a0a79ce3381925537be1f0b17503c9d6f5aecbe72c9368d458"
)

ARCHIVED_SOURCE_REL = Path("fpit_path_rl_final_v1_6.py")
ARCHIVED_PROTOCOL_REL = Path("fpit_path_rl_final_v1_6/protocol.json")
ARCHIVED_PATH_ADAPTER_REL = Path("fpit_path_rl_final_v1_6/path/adapter")
ARCHIVED_PATH_SUMMARY_REL = Path(
    "fpit_path_rl_final_v1_6/path/eval/path_eval_summary.json"
)

FROZEN_DATA_DIR_REL = Path("fpit_frozen_v1_3")

ARCHIVED_SOURCE_SHA256 = (
    "0cf904943596f1aa2be48a5a5e327eaa25a4691cc08c16fff3567f6817cb6613"
)
ARCHIVED_ADAPTER_MODEL_SHA256 = (
    "25cd4ab15520e7e6cf1453935db020c54f98e4c3a6420bfca522a0aae74bfb51"
)
ARCHIVED_ADAPTER_CONFIG_SHA256 = (
    "d935bf146de8969ba965583634e5824631745732ee119ca8c8a0350037c867fa"
)
ARCHIVED_PATH_SUMMARY_SHA256 = (
    "c0ca0a6b64ed558e32849d3c3870997ea0a33a2ff93e9ba8f954e891b797ff87"
)
ARCHIVED_PROTOCOL_SHA256 = (
    "fd4b7de680b7fb1b95495bfa2be317a54c62b63c0e15f2daa7a32bda34a7610a"
)

FROZEN_TRAIN_SHA256 = (
    "863dbe27047f13d1d1805d2fb959e00dac165af93a36cd979a860466e15dc9df"
)
FROZEN_TEST_SHA256 = (
    "c5a485192c7bc1a007145cee805e647d37d505b91168debc1a4f84e052638b86"
)
FROZEN_TRAIN_N = 1376
FROZEN_TEST_N = 92

ARCHIVED_PHASE1_SEED = 3407
SEED = 3407  # overwritten from --seed in main()
EXPECTED_PRIMARY_N = 49
EXPECTED_START_PAIR_CORRECT = 7
EXPECTED_START_PAIR_ACCURACY = EXPECTED_START_PAIR_CORRECT / EXPECTED_PRIMARY_N

EXPECTED_ENV = {
    "python": "3.12",
    "torch": "2.5.0+cu124",
    "cuda": "12.4",
    "trl": "0.24.0",
    "transformers": "5.5.0",
    "unsloth": "2026.8.12",
    "xformers": "0.0.28.post2",
    "datasets": "4.3.0",
    "accelerate": "1.14.0",
    "peft": "0.20.0",
    "triton": "3.1.0",
}

EXPECTED_CORE = {
    "epochs_phase1": 2,
    "K_train": 8,
    "learning_rate": 5e-6,
    "beta": 0.001,
    "loss_type": "dapo",
    "scale_rewards": "none",
    "max_prompt_length": 768,
    "max_completion_length": 192,
    "max_seq_length": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
}

DEFAULT_CHECKPOINT_STEPS = 688
EXPECTED_TRAINABLE_PARAMS = 40_370_176

DEFAULT_VENV = Path("/workspace/venvs/fpit-handoff-ab")


# =============================================================================
# Small utilities
# =============================================================================

def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                fail(f"{path}:{line_no}: expected a JSON object")
            rows.append(obj)
    return rows


def set_all_seeds(torch_mod, seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch_mod.manual_seed(seed)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)


def cleanup_gpu(torch_mod) -> None:
    gc.collect()
    if torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()


def import_archived_source(path: Path, tag: str):
    name = f"_fpit_archived_{tag}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        fail(f"Could not import archived FPIT source: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def safe_extract_tar_gz(archive: Path, destination: Path) -> None:
    """Extract a tar archive after checking for links and path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for m in members:
            if m.issym() or m.islnk():
                fail(f"Refusing link member in frozen archive: {m.name}")
            member_path = Path(m.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                fail(f"Unsafe archive member: {m.name}")
            resolved = (destination / member_path).resolve()
            if root not in resolved.parents and resolved != root:
                fail(f"Archive path escapes destination: {m.name}")
        tf.extractall(destination, filter="data")


def verify_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        fail(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        fail(
            f"{label} SHA256 mismatch\n"
            f"  expected {expected}\n"
            f"  actual   {actual}\n"
            f"  path     {path}"
        )


# =============================================================================
# Fresh-instance environment bootstrap
# =============================================================================

def current_env_snapshot() -> dict[str, str | None]:
    out: dict[str, str | None] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    try:
        import importlib.metadata as im
        out.update({
            "trl": im.version("trl"),
            "transformers": im.version("transformers"),
            "unsloth": im.version("unsloth"),
            "xformers": im.version("xformers"),
            "datasets": im.version("datasets"),
            "accelerate": im.version("accelerate"),
            "peft": im.version("peft"),
            "triton": im.version("triton"),
        })
    except Exception:
        for k in ("trl", "transformers", "unsloth", "xformers",
                  "datasets", "accelerate", "peft", "triton"):
            out.setdefault(k, None)

    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
    except Exception:
        out["torch"] = None
        out["cuda"] = None
    return out


def env_matches_exactly() -> bool:
    observed = current_env_snapshot()
    return all(str(observed.get(k)) == str(v) for k, v in EXPECTED_ENV.items())


def find_python312() -> str:
    candidates = [
        os.environ.get("PYTHON312"),
        shutil.which("python3.12"),
        "/usr/bin/python3.12",
        "/usr/local/bin/python3.12",
    ]
    for c in candidates:
        if not c:
            continue
        try:
            r = subprocess.run(
                [str(c), "-c", "import sys\nprint(sys.version_info[:2])"],
                check=True,
                capture_output=True,
                text=True,
            )
            if "(3, 12)" in r.stdout:
                return str(c)
        except Exception:
            pass
    fail(
        "Python 3.12 is required to recreate the frozen environment, but "
        "python3.12 was not found. Use a fresh GPU image with Python 3.12."
    )


def cheap_instance_guard(work_dir: Path) -> None:
    """Check GPU and disk capacity before installing packages."""
    if shutil.which("nvidia-smi") is None:
        fail(
            "nvidia-smi is not available. Start a CUDA GPU instance before the full run."
        )
    probe = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        fail("nvidia-smi could not query the GPU.")
    print("Fresh-instance GPU probe:", probe.stdout.strip().splitlines()[0])

    free = shutil.disk_usage(work_dir).free / (1024 ** 3)
    if free < 30:
        fail(
            f"Only {free:.1f} GiB free under {work_dir}. Allocate at least 30 GiB "
            "free for the frozen environment, model cache, trainer checkpoints, "
            "and final adapters."
        )
    print(f"Free disk under work dir: {free:.1f} GiB")


def bootstrap_and_reexec(args, script_path: Path) -> None:
    """Create the recorded Python environment and re-run this script."""
    if env_matches_exactly():
        return

    if not args.bootstrap:
        observed = current_env_snapshot()
        fail(
            "Current environment does not match the frozen FPIT stack.\n"
            f"Observed: {json.dumps(observed, indent=2)}\n"
            "Re-run with --bootstrap, or activate an exact matching environment."
        )

    venv = Path(args.venv)
    marker = venv / ".fpit_ab_env_complete"

    if not marker.exists():
        py312 = find_python312()
        print(f"\nCreating frozen venv at {venv} using {py312}")
        print("Bootstrap: exact torch cu124 wheel + PyPI dependency resolution + torchao compatibility cleanup")
        venv.parent.mkdir(parents=True, exist_ok=True)
        if venv.exists():
            shutil.rmtree(venv)
        subprocess.run([py312, "-m", "venv", str(venv)], check=True)

        pip = venv / "bin" / "pip"
        subprocess.run(
            [str(pip), "install", "--upgrade", "pip", "setuptools", "wheel"],
            check=True,
        )

        torch_wheel = (
            "https://download.pytorch.org/whl/cu124/"
            "torch-2.5.0%2Bcu124-cp312-cp312-linux_x86_64.whl"
        )
        subprocess.run(
            [str(pip), "install", torch_wheel],
            check=True,
        )

        subprocess.run(
            [str(pip), "install", "xformers==0.0.28.post2"],
            check=True,
        )

        subprocess.run(
            [
                str(pip), "install",
                "unsloth==2026.8.12",
                "trl==0.24.0",
                "transformers==5.5.0",
                "datasets==4.3.0",
                "accelerate==1.14.0",
                "peft==0.20.0",
            ],
            check=True,
        )

        subprocess.run(
            [
                str(pip), "install", "--no-deps", "--force-reinstall",
                "trl==0.24.0",
                "transformers==5.5.0",
                "datasets==4.3.0",
                "accelerate==1.14.0",
                "peft==0.20.0",
                "xformers==0.0.28.post2",
            ],
            check=True,
        )
        subprocess.run(
            [
                str(pip), "install", "--no-deps", "--force-reinstall",
                torch_wheel,
            ],
            check=True,
        )
        subprocess.run([str(pip), "check"], check=True)

        # torchao is incompatible with the recorded torch version.
        subprocess.run(
            [str(pip), "uninstall", "-y", "torchao"],
            check=False,
        )
        marker.write_text("complete\n", encoding="utf-8")

    python = venv / "bin" / "python"
    if not python.exists():
        fail(f"Bootstrapped venv has no interpreter: {python}")

    env = os.environ.copy()
    env["FPIT_AB_REEXEC"] = "1"
    print("\nRe-executing inside the frozen environment...")
    os.execve(
        str(python),
        [str(python), str(script_path), *sys.argv[1:]],
        env,
    )


def exact_runtime_guard(src) -> dict[str, Any]:
    """Check the recorded runtime and return environment metadata."""
    src.load_training_stack()
    detected = src.runtime_guard(False)
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    if py != EXPECTED_ENV["python"]:
        fail(f"Python={py}, expected {EXPECTED_ENV['python']}")

    import torch
    if not torch.cuda.is_available():
        fail("CUDA is not available to PyTorch.")

    props = torch.cuda.get_device_properties(0)
    gpu = {
        "name": torch.cuda.get_device_name(0),
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gib": props.total_memory / (1024**3),
    }
    print("GPU:", gpu)
    if gpu["total_memory_gib"] < 22.0:
        fail(
            f"GPU has only {gpu['total_memory_gib']:.1f} GiB VRAM. The frozen "
            "Qwen2.5-7B GRPO run was validated on a 24-GiB RTX 4090. Use a "
            "Use a GPU with at least ~24 GiB VRAM for this run."
        )

    if props.major >= 12:
        fail(
            "This GPU appears to be Blackwell-class (compute capability >= 12), "
            "but the frozen torch 2.5.0+cu124 stack predates Blackwell support. "
            "Use RTX 4090/A100/H100-class hardware for this experiment."
        )

    return {"python": py, "packages": detected, "gpu": gpu}


# =============================================================================
# Frozen archive/data verification
# =============================================================================

def prepare_frozen_artifacts(args) -> dict[str, Path]:
    results_archive = Path(args.results_archive).resolve()
    data_archive = Path(args.data_archive).resolve()
    work = Path(args.work_dir).resolve()
    extract_root = work / "frozen"

    if not results_archive.is_file():
        fail(f"Missing FPIT final-results archive: {results_archive}")
    if not data_archive.is_file():
        fail(f"Missing FPIT frozen-data archive: {data_archive}")

    # Internal file hashes define the frozen artifacts.
    actual_results_archive_sha256 = sha256_file(results_archive)
    actual_data_archive_sha256 = sha256_file(data_archive)

    def report_outer_hash(label: str, actual: str, originally_inspected: str):
        if actual == originally_inspected:
            print(f"{label} outer SHA256: PASS ({actual})")
        else:
            print(
                f"{label} outer SHA256 differs from the copy originally inspected:\n"
                f"  originally inspected: {originally_inspected}\n"
                f"  current archive:       {actual}\n"
                "  This is allowed because all required internal files are "
                "verified byte-for-byte after extraction."
            )

    report_outer_hash(
        "FPIT final-results archive",
        actual_results_archive_sha256,
        RESULTS_ARCHIVE_SHA256,
    )
    report_outer_hash(
        "FPIT frozen-data archive",
        actual_data_archive_sha256,
        DATA_ARCHIVE_SHA256,
    )

    extraction_manifest = extract_root / ".extraction_manifest.json"
    desired_manifest = {
        "results_archive_path": str(results_archive),
        "results_archive_sha256": actual_results_archive_sha256,
        "data_archive_path": str(data_archive),
        "data_archive_sha256": actual_data_archive_sha256,
    }
    existing_manifest = None
    if extraction_manifest.is_file():
        try:
            existing_manifest = json.loads(
                extraction_manifest.read_text(encoding="utf-8")
            )
        except Exception:
            existing_manifest = None

    if existing_manifest != desired_manifest:
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        print("Extracting frozen archives...")
        safe_extract_tar_gz(results_archive, extract_root)
        safe_extract_tar_gz(data_archive, extract_root)
        extraction_manifest.write_text(
            json.dumps(desired_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        print("Reusing extraction tree: archive hashes match extraction manifest.")

    source = extract_root / ARCHIVED_SOURCE_REL
    protocol = extract_root / ARCHIVED_PROTOCOL_REL
    adapter = extract_root / ARCHIVED_PATH_ADAPTER_REL
    archived_path_summary = extract_root / ARCHIVED_PATH_SUMMARY_REL
    data_dir = extract_root / FROZEN_DATA_DIR_REL
    train_file = data_dir / "fpit_train.jsonl"
    test_file = data_dir / "fpit_test.jsonl"

    verify_file_hash(source, ARCHIVED_SOURCE_SHA256, "archived FPIT source")
    verify_file_hash(protocol, ARCHIVED_PROTOCOL_SHA256, "archived FPIT protocol")
    verify_file_hash(
        adapter / "adapter_model.safetensors",
        ARCHIVED_ADAPTER_MODEL_SHA256,
        "frozen FPIT path adapter weights",
    )
    verify_file_hash(
        adapter / "adapter_config.json",
        ARCHIVED_ADAPTER_CONFIG_SHA256,
        "frozen FPIT path adapter config",
    )
    verify_file_hash(
        archived_path_summary,
        ARCHIVED_PATH_SUMMARY_SHA256,
        "archived FPIT path evaluation summary",
    )
    verify_file_hash(train_file, FROZEN_TRAIN_SHA256, "frozen train JSONL")
    verify_file_hash(test_file, FROZEN_TEST_SHA256, "frozen test JSONL")

    if len(load_jsonl(train_file)) != FROZEN_TRAIN_N:
        fail("Frozen train row count mismatch.")
    if len(load_jsonl(test_file)) != FROZEN_TEST_N:
        fail("Frozen test row count mismatch.")

    return {
        "work": work,
        "extract_root": extract_root,
        "results_archive_sha256_actual": actual_results_archive_sha256,
        "data_archive_sha256_actual": actual_data_archive_sha256,
        "source": source,
        "protocol": protocol,
        "adapter": adapter,
        "archived_path_summary": archived_path_summary,
        "data_dir": data_dir,
        "train_file": train_file,
        "test_file": test_file,
    }


def verify_archived_science(artifacts: dict[str, Path]):
    src = import_archived_source(artifacts["source"], "cpu_preflight")

    src.parser_reward_self_test()

    train_rows, test_rows, corpus_audit = src.corpus_integrity_audit(
        artifacts["data_dir"]
    )

    checks = {
        "SEED": (src.SEED, ARCHIVED_PHASE1_SEED),
        "EPOCHS": (src.EPOCHS, EXPECTED_CORE["epochs_phase1"]),
        "K_TRAIN": (src.K_TRAIN, EXPECTED_CORE["K_train"]),
        "LR": (src.LR, EXPECTED_CORE["learning_rate"]),
        "BETA": (src.BETA, EXPECTED_CORE["beta"]),
        "MAX_PROMPT_LENGTH": (
            src.MAX_PROMPT_LENGTH, EXPECTED_CORE["max_prompt_length"]
        ),
        "MAX_COMPLETION_LENGTH": (
            src.MAX_COMPLETION_LENGTH, EXPECTED_CORE["max_completion_length"]
        ),
        "MAX_SEQ_LENGTH": (
            src.MAX_SEQ_LENGTH, EXPECTED_CORE["max_seq_length"]
        ),
        "LORA_R": (src.LORA_R, EXPECTED_CORE["lora_r"]),
        "LORA_ALPHA": (src.LORA_ALPHA, EXPECTED_CORE["lora_alpha"]),
    }
    bad = [f"{k}: {a!r} != {e!r}" for k, (a, e) in checks.items() if a != e]
    if bad:
        fail("Archived source constant mismatch:\n  " + "\n  ".join(bad))

    protocol = json.loads(artifacts["protocol"].read_text(encoding="utf-8"))
    if protocol["loss_type"] != EXPECTED_CORE["loss_type"]:
        fail("Archived loss_type mismatch.")
    if protocol["scale_rewards"] != EXPECTED_CORE["scale_rewards"]:
        fail("Archived scale_rewards mismatch.")

    archived_summary = json.loads(
        artifacts["archived_path_summary"].read_text(encoding="utf-8")
    )
    pri = archived_summary["greedy"]["structured_primary"]
    archived_n = int(pri["n_cases"])
    archived_pair = float(pri["pair_accuracy"])
    archived_correct = round(archived_pair * archived_n)
    if (
        archived_n != EXPECTED_PRIMARY_N
        or archived_correct != EXPECTED_START_PAIR_CORRECT
        or abs(archived_pair - EXPECTED_START_PAIR_ACCURACY) > 1e-12
    ):
        fail(
            "Archived path result does not match the frozen starting point: "
            f"{archived_correct}/{archived_n}={archived_pair}"
        )

    # Check strict-pair scoring with the archived verifier.
    row = train_rows[0]
    fields = row["expected_fields"]
    targets = row["target_fields"]

    def completion(overrides: dict[str, float] | None = None) -> str:
        vals = dict(targets)
        if overrides:
            vals.update(overrides)
        return "\n".join(f"{f}: {vals[f]}" for f in fields)

    gold = src.score_text_against_row(completion(), row)
    base_wrong = src.score_text_against_row(
        completion({"BASE_ANSWER": float(targets["BASE_ANSWER"]) + 99}), row
    )
    new_wrong = src.score_text_against_row(
        completion({"NEW_ANSWER": float(targets["NEW_ANSWER"]) + 99}), row
    )
    both_wrong = src.score_text_against_row(
        completion({
            "BASE_ANSWER": float(targets["BASE_ANSWER"]) + 99,
            "NEW_ANSWER": float(targets["NEW_ANSWER"]) + 99,
        }),
        row,
    )
    strict_vector = [
        float(gold["pair_correct"]),
        float(base_wrong["pair_correct"]),
        float(new_wrong["pair_correct"]),
        float(both_wrong["pair_correct"]),
    ]
    if strict_vector != [1.0, 0.0, 0.0, 0.0]:
        fail(f"Strict pair verifier unit test failed: {strict_vector}")

    print("\nCPU/DATA/PARSER PREFLIGHT: PASS")
    print(
        f"  starting archived primary: "
        f"{EXPECTED_START_PAIR_CORRECT}/{EXPECTED_PRIMARY_N} "
        f"= {EXPECTED_START_PAIR_ACCURACY:.4%}"
    )
    print("  strict reward truth table: [both, base-only, new-only, neither]")
    print("                           ->", strict_vector)

    return src, train_rows, test_rows, corpus_audit


# =============================================================================
# Trainable checkpoint loading
# =============================================================================

def load_frozen_fpit_policy(src, adapter_dir: Path, trainable: bool):
    """Load the frozen FPIT LoRA as a trainable Unsloth PEFT model."""
    src.set_all_seeds(SEED)

    adapter_cfg = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    base_id = adapter_cfg.get("base_model_name_or_path")
    if not base_id:
        fail("adapter_config.json has no base_model_name_or_path")

    model, tokenizer = src.FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=src.MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
        use_gradient_checkpointing="unsloth",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if trainable:
        src.FastLanguageModel.for_training(model)
    else:
        src.FastLanguageModel.for_inference(model)

    # The frozen LoRA must be the active trainable adapter.
    if not getattr(model, "peft_config", None):
        fail(
            "Unsloth did not recognize the frozen checkpoint as a PEFT model. "
            "Refusing continuation."
        )

    trainable_named = [
        (n, p) for n, p in model.named_parameters() if p.requires_grad
    ]
    trainable_names = [n for n, _ in trainable_named]
    trainable_param_count = sum(int(p.numel()) for _, p in trainable_named)
    if trainable:
        if not trainable_names:
            fail("Loaded frozen FPIT adapter has zero trainable parameters.")
        unexpected = [
            n for n in trainable_names
            if "lora_" not in n.lower()
            and "modules_to_save" not in n.lower()
        ]
        if unexpected:
            fail(
                "Unexpected non-LoRA trainable parameters:\n  "
                + "\n  ".join(unexpected[:30])
            )
        if trainable_param_count != EXPECTED_TRAINABLE_PARAMS:
            fail(
                "Frozen FPIT adapter trainable-parameter count mismatch: "
                f"got {trainable_param_count:,}, expected "
                f"{EXPECTED_TRAINABLE_PARAMS:,}."
            )

    return model, tokenizer, {
        "adapter_directory": str(adapter_dir),
        "base_model_name_or_path": base_id,
        "trainable_parameter_names": trainable_names,
        "n_trainable_parameter_tensors": len(trainable_names),
        "n_trainable_parameters": trainable_param_count,
    }


# =============================================================================
# Rewards
# =============================================================================

class StrictPairOracle:
    """Strict joint endpoint reward used by Arm B."""
    def __init__(self, src, audit_path: Path):
        self.src = src
        self.__name__ = "strict_pair_reward"
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        if self.audit_path.exists():
            self.audit_path.unlink()
        self.calls = 0
        self.history: list[dict[str, Any]] = []

    def __call__(self, completions, **kwargs):
        self.calls += 1
        n = len(completions)

        required = [
            "calculator_slug", "row_number", "prompt_hash", "source_state_hash",
            "expected_fields_json", "target_fields_json",
            "derived_changed_fields_json", "derived_invariant_fields_json",
        ]
        for col in required:
            if col not in kwargs:
                fail(f"StrictPairOracle missing dataset column: {col}")
            if len(kwargs[col]) != n:
                fail(f"StrictPairOracle length mismatch for {col}")

        prompt_hashes = [self.src.clean_text(x) for x in kwargs["prompt_hash"]]
        if len(set(prompt_hashes)) != 1:
            fail(
                "StrictPairOracle expected one prompt group per reward call. "
                f"got {len(set(prompt_hashes))}"
            )
        if n != self.src.K_TRAIN:
            fail(
                f"StrictPairOracle expected K={self.src.K_TRAIN}, got {n}"
            )

        rewards: list[float] = []
        records: list[dict[str, Any]] = []

        for i, completion in enumerate(completions):
            slug = self.src.clean_text(kwargs["calculator_slug"][i])
            row = {
                "calculator_slug": slug,
                "expected_fields": json.loads(kwargs["expected_fields_json"][i]),
                "target_fields": json.loads(kwargs["target_fields_json"][i]),
                "derived_changed_fields": json.loads(
                    kwargs["derived_changed_fields_json"][i]
                ),
                "derived_invariant_fields": json.loads(
                    kwargs["derived_invariant_fields_json"][i]
                ),
            }
            text = self.src.completion_to_text(completion)
            scored = self.src.score_text_against_row(text, row)
            reward = float(bool(scored["pair_correct"]))
            if reward not in (0.0, 1.0):
                fail(f"Non-binary strict reward encountered: {reward}")
            rewards.append(reward)
            records.append({
                "call": self.calls,
                "sample_index": i,
                "arm": "strict_handoff",
                "calculator_slug": slug,
                "row_number": self.src.clean_text(kwargs["row_number"][i]),
                "prompt_hash": prompt_hashes[i],
                "source_state_hash": self.src.clean_text(
                    kwargs["source_state_hash"][i]
                ),
                "completion": text,
                "reward": reward,
                **scored,
            })

        with self.audit_path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        mean_r = float(statistics.mean(rewards))
        std_r = (
            float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0
        )
        hist = {
            "call": self.calls,
            "arm": "strict_handoff",
            "prompt_hash": prompt_hashes[0],
            "calculator_slug": records[0]["calculator_slug"],
            "row_number": records[0]["row_number"],
            "n": len(rewards),
            "mean_reward": mean_r,
            "std_reward": std_r,
            "zero_variance": bool(max(rewards) - min(rewards) <= 1e-12),
            "pair_correct_rate": mean_r,
            "rewards": rewards,
        }
        self.history.append(hist)
        return rewards


# =============================================================================
# Matched config
# =============================================================================

def continuation_config(
    src,
    arm_dir: Path,
    continuation_epochs: float,
    checkpoint_steps: int,
    smoke_max_steps: int | None = None,
):
    cfg = src.grpo_config(arm_dir, smoke_test=False)

    expected = {
        "learning_rate": EXPECTED_CORE["learning_rate"],
        "beta": EXPECTED_CORE["beta"],
        "num_generations": EXPECTED_CORE["K_train"],
        "max_prompt_length": EXPECTED_CORE["max_prompt_length"],
        "max_completion_length": EXPECTED_CORE["max_completion_length"],
        "loss_type": EXPECTED_CORE["loss_type"],
    }
    for key, exp in expected.items():
        got = getattr(cfg, key, None)
        if got != exp:
            fail(
                f"Archived GRPO config drift: {key}={got!r}, expected {exp!r}"
            )

    scale = getattr(cfg, "scale_rewards", None)
    if not (scale is False or str(scale).lower() == "none"):
        fail(f"Archived scale_rewards drift: {scale!r}")

    cfg.num_train_epochs = float(continuation_epochs)
    cfg.max_steps = int(smoke_max_steps) if smoke_max_steps is not None else -1
    cfg.output_dir = str(arm_dir / "trainer")

    # Checkpoints are for fault tolerance only.
    if smoke_max_steps is not None:
        cfg.save_strategy = "no"
    else:
        cfg.save_strategy = "steps"
        cfg.save_steps = int(checkpoint_steps)
        cfg.save_total_limit = 2

    cfg.seed = SEED
    cfg.data_seed = SEED
    return cfg


def config_for_comparison(cfg) -> dict[str, Any]:
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(vars(cfg))
    for k in ("output_dir", "run_name", "logging_dir"):
        d.pop(k, None)
    return d


# =============================================================================
# Greedy held-out evaluation
# =============================================================================

def primary_rows(src, test_rows):
    rows = [
        r for r in test_rows
        if r["calculator_slug"] in src.STRUCTURED_SLUGS
        and r["prompt_hash"] not in src.SMOKE_EXPOSED_PROMPT_HASHES
    ]
    if len(rows) != EXPECTED_PRIMARY_N:
        fail(
            f"Primary-set reconstruction gave {len(rows)} cases, "
            f"expected {EXPECTED_PRIMARY_N}."
        )
    return rows


def greedy_eval_rows(
    src,
    torch_mod,
    model,
    tokenizer,
    rows,
    policy_name: str,
    output_jsonl: Path,
) -> dict[str, Any]:
    src.FastLanguageModel.for_inference(model)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, row in enumerate(rows, start=1):
        torch_mod.manual_seed(SEED + 700000 + idx)
        torch_mod.cuda.manual_seed_all(SEED + 700000 + idx)
        with torch_mod.inference_mode():
            texts = src.generate_for_case(
                model, tokenizer, row,
                n=1,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        text = texts[0]
        scored = src.score_text_against_row(text, row)
        records.append({
            "policy": policy_name,
            "mode": "greedy",
            "calculator_slug": row["calculator_slug"],
            "calculator_name": row["calculator_name"],
            "row_number": src.clean_text(row["row_number"]),
            "prompt_hash": row["prompt_hash"],
            "source_state_hash": row["source_state_hash"],
            "sample_no": 1,
            "completion": text,
            **scored,
        })
        if idx % 10 == 0 or idx == len(rows):
            pairs = sum(int(r["pair_correct"]) for r in records)
            print(f"  greedy {idx:2d}/{len(rows)} | pair_correct={pairs}")

    with output_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = src.summarize_eval_records(records)
    pri = summary["structured_primary"]
    return {
        "records": records,
        "summary": summary,
        "pair_correct": sum(int(r["pair_correct"]) for r in records),
        "n": len(records),
        "pair_accuracy": float(pri["pair_accuracy"]),
    }


def baseline_gpu_guard(
    src,
    torch_mod,
    model,
    tokenizer,
    test_rows,
    out_dir: Path,
) -> dict[str, Any]:
    print("\nGPU BASELINE GUARD: regenerating frozen FPIT primary...")
    rows = primary_rows(src, test_rows)
    result = greedy_eval_rows(
        src, torch_mod, model, tokenizer, rows,
        "frozen_fpit_start",
        out_dir / "starting_fpit_primary_greedy.jsonl",
    )
    write_json(
        out_dir / "starting_fpit_primary_summary.json",
        {k: v for k, v in result.items() if k != "records"},
    )

    if (
        result["n"] != EXPECTED_PRIMARY_N
        or result["pair_correct"] != EXPECTED_START_PAIR_CORRECT
        or abs(result["pair_accuracy"] - EXPECTED_START_PAIR_ACCURACY) > 1e-12
    ):
        fail(
            "FROZEN CHECKPOINT REPRODUCTION FAILED.\n"
            f"Expected {EXPECTED_START_PAIR_CORRECT}/{EXPECTED_PRIMARY_N} "
            f"= {EXPECTED_START_PAIR_ACCURACY:.6f}\n"
            f"Observed {result['pair_correct']}/{result['n']} "
            f"= {result['pair_accuracy']:.6f}\n"
            "Do not spend GPU on continuation until this is resolved."
        )

    print(
        "GPU BASELINE GUARD: PASS — "
        f"{result['pair_correct']}/{result['n']} "
        f"= {result['pair_accuracy']:.4%}"
    )
    return result


# =============================================================================
# Arm B smoke test
# =============================================================================

def one_step_strict_smoke(
    src,
    torch_mod,
    model,
    tokenizer,
    train_rows,
    smoke_dir: Path,
):
    print("\nONE-STEP STRICT-HANDOFF TRAINING SMOKE...")
    src.FastLanguageModel.for_training(model)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    # Smoke test uses training rows only.
    tiny = train_rows[:16]
    train_ds = src.make_train_dataset(tiny)
    oracle = StrictPairOracle(src, smoke_dir / "strict_smoke_rollouts.jsonl")
    cfg = continuation_config(
        src,
        smoke_dir,
        continuation_epochs=1.0,
        checkpoint_steps=DEFAULT_CHECKPOINT_STEPS,
        smoke_max_steps=1,
    )

    trainer = src.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=oracle,
        args=cfg,
        train_dataset=train_ds,
    )
    result = trainer.train()

    if trainer.state.global_step != 1:
        fail(
            f"Strict handoff smoke expected global_step=1, "
            f"got {trainer.state.global_step}"
        )
    if oracle.calls != 1:
        fail(f"Strict handoff smoke expected 1 reward call, got {oracle.calls}")
    if not oracle.history:
        fail("Strict handoff smoke produced no reward history.")

    write_json(
        smoke_dir / "strict_smoke_summary.json",
        {
            "global_step": trainer.state.global_step,
            "reward_calls": oracle.calls,
            "history": oracle.history,
            "metrics": result.metrics,
        },
    )
    print("ONE-STEP STRICT-HANDOFF TRAINING SMOKE: PASS")
    del trainer


# =============================================================================
# Full matched arms
# =============================================================================

def summarize_training_oracle(oracle) -> dict[str, Any]:
    history = oracle.history
    if not history:
        return {
            "n_reward_calls": 0,
            "zero_variance_groups": 0,
            "zero_variance_rate": None,
            "mean_training_reward": None,
        }
    return {
        "n_reward_calls": len(history),
        "zero_variance_groups": sum(bool(h["zero_variance"]) for h in history),
        "zero_variance_rate": (
            sum(bool(h["zero_variance"]) for h in history) / len(history)
        ),
        "mean_training_reward": statistics.mean(
            float(h["mean_reward"]) for h in history
        ),
    }


def train_arm(
    arm: str,
    args,
    artifacts,
    src,
    train_rows,
    test_rows,
    runtime_info,
) -> dict[str, Any]:
    import torch

    if arm not in {"A", "B"}:
        fail(f"Unknown arm {arm}")

    arm_name = "A_fpit_continue" if arm == "A" else "B_strict_handoff"
    arm_dir = Path(args.work_dir) / "results" / arm_name
    complete_path = arm_dir / "ARM_COMPLETE.json"

    if complete_path.exists() and not args.force_arm:
        print(f"\nSkipping completed {arm_name}: {complete_path}")
        return json.loads(complete_path.read_text(encoding="utf-8"))

    if arm_dir.exists() and any(arm_dir.iterdir()) and not args.force_arm:
        fail(
            f"{arm_dir} already contains a partial run.\n"
            "Inspect it before resuming or deleting it. "
            "Use --force-arm only if you intentionally want to restart this arm "
            "from the frozen FPIT checkpoint."
        )
    if args.force_arm and arm_dir.exists():
        shutil.rmtree(arm_dir)
    arm_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"Training: {arm_name}")
    print("=" * 100)

    # Reload the frozen FPIT adapter for each arm.
    model, tokenizer, load_meta = load_frozen_fpit_policy(
        src, artifacts["adapter"], trainable=True
    )
    set_all_seeds(torch, SEED)

    train_ds = src.make_train_dataset(train_rows)

    if arm == "A":
        oracle = src.RewardOracle(
            "path", arm_dir / "training_rollouts.jsonl"
        )
        reward_definition = "archived RewardOracle('path') unchanged"
    else:
        oracle = StrictPairOracle(
            src, arm_dir / "training_rollouts.jsonl"
        )
        reward_definition = "R = 1[BASE correct AND NEW correct]"

    cfg = continuation_config(
        src,
        arm_dir,
        continuation_epochs=args.continuation_epochs,
        checkpoint_steps=args.checkpoint_steps,
        smoke_max_steps=None,
    )

    pretrain = {
        "experiment_id": EXPERIMENT_ID,
        "arm": arm,
        "arm_name": arm_name,
        "reward_definition": reward_definition,
        "start_adapter_model_sha256": ARCHIVED_ADAPTER_MODEL_SHA256,
        "source_sha256": ARCHIVED_SOURCE_SHA256,
        "train_sha256": FROZEN_TRAIN_SHA256,
        "test_sha256": FROZEN_TEST_SHA256,
        "seed": SEED,
        "continuation_epochs": args.continuation_epochs,
        "checkpoint_steps": args.checkpoint_steps,
        "runtime": runtime_info,
        "load_meta": load_meta,
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg),
    }
    write_json(arm_dir / "pretrain_manifest.json", pretrain)

    trainer = src.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=oracle,
        args=cfg,
        train_dataset=train_ds,
    )

    start = time.perf_counter()
    result = trainer.train()  # fresh optimizer/scheduler state
    train_seconds = time.perf_counter() - start

    final_adapter = arm_dir / "adapter_final"
    model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)

    write_json(arm_dir / "reward_history.json", oracle.history)

    training_summary = {
        "arm": arm,
        "arm_name": arm_name,
        "reward_definition": reward_definition,
        "train_seconds": train_seconds,
        "trainer_global_step": trainer.state.global_step,
        "trainer_epoch": trainer.state.epoch,
        "train_result_metrics": result.metrics,
        "oracle": summarize_training_oracle(oracle),
        "log_history": trainer.state.log_history,
        "final_adapter_model_sha256": (
            sha256_file(final_adapter / "adapter_model.safetensors")
            if (final_adapter / "adapter_model.safetensors").exists()
            else None
        ),
    }
    write_json(arm_dir / "train_meta.json", training_summary)

    # Primary evaluation uses the final checkpoint with greedy decoding.
    print(f"\nFinal primary evaluation: {arm_name}")
    pri_rows = primary_rows(src, test_rows)
    eval_result = greedy_eval_rows(
        src,
        torch,
        model,
        tokenizer,
        pri_rows,
        arm_name,
        arm_dir / "eval" / "primary_greedy.jsonl",
    )
    eval_summary = {k: v for k, v in eval_result.items() if k != "records"}
    write_json(arm_dir / "eval" / "primary_summary.json", eval_summary)

    complete = {
        "experiment_id": EXPERIMENT_ID,
        "arm": arm,
        "arm_name": arm_name,
        "reward_definition": reward_definition,
        "primary_checkpoint": "final checkpoint only",
        "primary": eval_summary,
        "training": {
            "train_seconds": train_seconds,
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
        },
        "adapter_final": str(final_adapter),
    }
    write_json(complete_path, complete)

    del trainer, model, tokenizer
    cleanup_gpu(torch)
    return complete


# =============================================================================
# Final comparison / packaging
# =============================================================================

def paired_case_comparison(a_path: Path, b_path: Path) -> dict[str, Any]:
    a = {r["prompt_hash"]: r for r in load_jsonl(a_path)}
    b = {r["prompt_hash"]: r for r in load_jsonl(b_path)}
    if set(a) != set(b):
        fail("A/B primary evaluation case sets differ.")

    hashes = sorted(a)
    n01 = sum(
        (not bool(a[h]["pair_correct"])) and bool(b[h]["pair_correct"])
        for h in hashes
    )
    n10 = sum(
        bool(a[h]["pair_correct"]) and (not bool(b[h]["pair_correct"]))
        for h in hashes
    )
    n = n01 + n10
    if n == 0:
        p = None
    else:
        k = min(n01, n10)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)

    a_acc = statistics.mean(float(a[h]["pair_correct"]) for h in hashes)
    b_acc = statistics.mean(float(b[h]["pair_correct"]) for h in hashes)
    return {
        "n": len(hashes),
        "A_pair_accuracy": a_acc,
        "B_pair_accuracy": b_acc,
        "delta_B_minus_A": b_acc - a_acc,
        "A_wrong_B_right": n01,
        "A_right_B_wrong": n10,
        "mcnemar_exact_p": p,
    }


def package_final_results(args, artifacts, comparison_path: Path) -> tuple[Path, Path]:
    """Package the final continuation results and adapters."""
    results_root = Path(args.work_dir) / "results"
    bundle_root = Path(args.work_dir) / "final_bundle"
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(Path(__file__).resolve(), bundle_root / Path(__file__).name)
    shutil.copy2(
        artifacts["source"], bundle_root / "archived_fpit_source.py"
    )
    shutil.copy2(
        artifacts["protocol"], bundle_root / "archived_fpit_protocol.json"
    )
    shutil.copy2(
        artifacts["train_file"], bundle_root / "fpit_train.jsonl"
    )
    shutil.copy2(
        artifacts["test_file"], bundle_root / "fpit_test.jsonl"
    )
    shutil.copy2(comparison_path, bundle_root / "comparison_final.json")

    for arm_name in ("A_fpit_continue", "B_strict_handoff"):
        src_dir = results_root / arm_name
        dst_dir = bundle_root / arm_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        for rel in (
            Path("pretrain_manifest.json"),
            Path("train_meta.json"),
            Path("reward_history.json"),
            Path("training_rollouts.jsonl"),
            Path("ARM_COMPLETE.json"),
            Path("eval/primary_greedy.jsonl"),
            Path("eval/primary_summary.json"),
        ):
            s = src_dir / rel
            if s.exists():
                d = dst_dir / rel
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, d)

        adapter_src = src_dir / "adapter_final"
        adapter_dst = dst_dir / "adapter_final"
        if adapter_src.exists():
            shutil.copytree(adapter_src, adapter_dst)

    out_tar = Path(args.work_dir) / "FPIT_HANDOFF_AB_FINAL_RESULTS.tar.gz"
    with tarfile.open(out_tar, "w:gz") as tf:
        tf.add(bundle_root, arcname="FPIT_HANDOFF_AB_FINAL_RESULTS")

    checksum = Path(str(out_tar) + ".sha256")
    digest = sha256_file(out_tar)
    checksum.write_text(f"{digest}  {out_tar.name}\n", encoding="utf-8")
    return out_tar, checksum


# =============================================================================
# Frozen data archive
# =============================================================================

def build_data_archive_from_repo(data_dir: Path, output: Path) -> Path:
    """Package the committed frozen JSONLs in the archive layout expected below."""
    train = data_dir / "fpit_train.jsonl"
    test = data_dir / "fpit_test.jsonl"
    if not train.is_file() or not test.is_file():
        fail(f"Expected fpit_train.jsonl and fpit_test.jsonl in {data_dir}")
    if sha256_file(train) != FROZEN_TRAIN_SHA256:
        fail("Committed fpit_train.jsonl does not match the frozen training data hash.")
    if sha256_file(test) != FROZEN_TEST_SHA256:
        fail("Committed fpit_test.jsonl does not match the frozen test data hash.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tf:
        tf.add(train, arcname="fpit_frozen_v1_3/fpit_train.jsonl")
        tf.add(test, arcname="fpit_frozen_v1_3/fpit_test.jsonl")
    return output

# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument(
        "--results-archive",
        required=True,
        help="Frozen acquisition archive used by the original continuation run.",
    )
    ap.add_argument(
        "--data-archive",
        default=None,
        help="Optional frozen-data tar.gz. If omitted, it is built from repo data/.",
    )
    ap.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Directory containing fpit_train.jsonl and fpit_test.jsonl.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Continuation seed. Reported paper seeds: 3407, 9173, 17011.",
    )
    ap.add_argument(
        "--work-dir",
        default="/workspace/fpit_handoff_ab",
    )
    ap.add_argument(
        "--arm",
        choices=["A", "B", "all"],
        default="all",
    )
    ap.add_argument(
        "--continuation-epochs",
        type=float,
        default=4.0,
        help="Matched continuation epochs per arm. Acquisition used 2 epochs.",
    )
    ap.add_argument(
        "--checkpoint-steps",
        type=int,
        default=DEFAULT_CHECKPOINT_STEPS,
        help="Fault-tolerance checkpoint cadence. Not used for model selection.",
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help="CPU, archive, parser, and data validation only. No GPU stack install.",
    )
    ap.add_argument(
        "--bootstrap",
        action="store_true",
        help="Create/reuse exact Python-3.12 frozen venv if current env differs.",
    )
    ap.add_argument(
        "--venv",
        default=str(DEFAULT_VENV),
    )
    ap.add_argument(
        "--force-arm",
        action="store_true",
        help="Delete and restart an existing partial arm.",
    )
    args = ap.parse_args()

    global SEED, EXPERIMENT_ID
    SEED = int(args.seed)
    EXPERIMENT_ID = f"CDI_HANDOFF_SEED_{SEED}"

    if args.continuation_epochs <= 0:
        fail("--continuation-epochs must be > 0")
    if args.checkpoint_steps <= 0:
        fail("--checkpoint-steps must be > 0")

    script_path = Path(__file__).resolve()
    work_path = Path(args.work_dir).resolve()
    work_path.mkdir(parents=True, exist_ok=True)

    if args.data_archive is None:
        data_archive = work_path / "frozen_data.tar.gz"
        build_data_archive_from_repo(Path(args.data_dir).resolve(), data_archive)
        args.data_archive = str(data_archive)

    os.environ.setdefault("HF_HOME", str(work_path / "hf_cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    artifacts = prepare_frozen_artifacts(args)
    src_cpu, train_rows_cpu, test_rows_cpu, corpus_audit = (
        verify_archived_science(artifacts)
    )

    if args.preflight_only:
        print("\nPreflight complete. No GPU packages were imported or installed.")
        return 0

    cheap_instance_guard(work_path)

    if not env_matches_exactly():
        bootstrap_and_reexec(args, script_path)

    src = import_archived_source(artifacts["source"], "gpu_run")
    src.parser_reward_self_test()
    train_rows, test_rows, corpus_audit2 = src.corpus_integrity_audit(
        artifacts["data_dir"]
    )
    if corpus_audit2 != corpus_audit:
        fail("CPU/GPU corpus audits disagree.")

    runtime_info = exact_runtime_guard(src)
    import torch

    cfg_A = continuation_config(
        src,
        Path(args.work_dir) / "_config_A",
        args.continuation_epochs,
        args.checkpoint_steps,
    )
    cfg_B = continuation_config(
        src,
        Path(args.work_dir) / "_config_B",
        args.continuation_epochs,
        args.checkpoint_steps,
    )
    if config_for_comparison(cfg_A) != config_for_comparison(cfg_B):
        fail("Arm A/B GRPO configs differ before reward assignment.")

    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "question": (
            "After FPIT makes successful joint endpoint trajectories reachable, "
            "does switching to strict pair-only outcome RL improve final greedy "
            "joint pair accuracy relative to continuing FPIT?"
        ),
        "starting_policy": "frozen FPIT path adapter from 2026-08-14 archive",
        "arms": {
            "A": "continue archived RewardOracle('path') unchanged",
            "B": "strict reward = 1[BASE correct AND NEW correct]",
        },
        "primary_endpoint": (
            "final-checkpoint greedy pair accuracy on the 49 confirmatory "
            "structured held-out cases"
        ),
        "primary_checkpoint_selection": (
            "final checkpoint only. Intermediate checkpoints cannot be selected"
        ),
        "continuation_epochs_per_arm": args.continuation_epochs,
        "checkpoint_steps_fault_tolerance_only": args.checkpoint_steps,
        "seed": SEED,
        "start_expected_pair": {
            "correct": EXPECTED_START_PAIR_CORRECT,
            "n": EXPECTED_PRIMARY_N,
            "accuracy": EXPECTED_START_PAIR_ACCURACY,
        },
        "frozen_artifacts": {
            "results_archive_sha256_current": artifacts[
                "results_archive_sha256_actual"
            ],
            "results_archive_sha256_originally_inspected": RESULTS_ARCHIVE_SHA256,
            "data_archive_sha256_current": artifacts[
                "data_archive_sha256_actual"
            ],
            "data_archive_sha256_originally_inspected": DATA_ARCHIVE_SHA256,
            "source_sha256": ARCHIVED_SOURCE_SHA256,
            "start_adapter_model_sha256": ARCHIVED_ADAPTER_MODEL_SHA256,
            "train_sha256": FROZEN_TRAIN_SHA256,
            "test_sha256": FROZEN_TEST_SHA256,
        },
        "environment": runtime_info,
        "corpus_audit": corpus_audit,
        "matched_config": config_for_comparison(cfg_A),
    }
    protocol_path = Path(args.work_dir) / "results" / "protocol.json"
    write_json(protocol_path, protocol)

    print("\n" + "=" * 100)
    print(EXPERIMENT_ID)
    print("=" * 100)
    print("Scientific arms:")
    print("  A: continue original FPIT path reward")
    print("  B: R = 1[BASE AND NEW correct], no partial/process credit")
    print(f"Continuation epochs/arm: {args.continuation_epochs}")
    print(f"Primary: final greedy pair accuracy on n={EXPECTED_PRIMARY_N}")

    guard_dir = Path(args.work_dir) / "results" / "_guards"
    guard_model, guard_tokenizer, guard_load_meta = load_frozen_fpit_policy(
        src, artifacts["adapter"], trainable=True
    )
    write_json(guard_dir / "load_meta.json", guard_load_meta)

    baseline = baseline_gpu_guard(
        src, torch, guard_model, guard_tokenizer, test_rows, guard_dir
    )

    # Run one discarded Arm B step before full training.
    one_step_strict_smoke(
        src, torch, guard_model, guard_tokenizer, train_rows,
        guard_dir / "strict_one_step_smoke",
    )
    del guard_model, guard_tokenizer
    cleanup_gpu(torch)

    arms = ["A", "B"] if args.arm == "all" else [args.arm]
    results = []
    for arm in arms:
        results.append(
            train_arm(
                arm=arm,
                args=args,
                artifacts=artifacts,
                src=src,
                train_rows=train_rows,
                test_rows=test_rows,
                runtime_info=runtime_info,
            )
        )

    # A completed Arm A result may be reused when running Arm B alone.
    if args.arm == "B":
        a_complete = (
            Path(args.work_dir) / "results" / "A_fpit_continue" / "ARM_COMPLETE.json"
        )
        if a_complete.exists():
            a_result = json.loads(a_complete.read_text(encoding="utf-8"))
            if a_result.get("arm") != "A":
                fail(f"Unexpected completed A metadata: {a_complete}")
            results.insert(0, a_result)
            print(
                "\nRecovered completed Arm A from prior instance for final paired comparison: "
                f"{a_result['primary']['pair_correct']}/{a_result['primary']['n']}"
            )

    comparison = {
        "experiment_id": EXPERIMENT_ID,
        "starting_fpit": {
            "pair_correct": baseline["pair_correct"],
            "n": baseline["n"],
            "pair_accuracy": baseline["pair_accuracy"],
        },
        "arms": results,
    }

    if {r["arm"] for r in results} == {"A", "B"}:
        a = next(r for r in results if r["arm"] == "A")
        b = next(r for r in results if r["arm"] == "B")
        paired = paired_case_comparison(
            Path(args.work_dir) / "results" / "A_fpit_continue"
            / "eval" / "primary_greedy.jsonl",
            Path(args.work_dir) / "results" / "B_strict_handoff"
            / "eval" / "primary_greedy.jsonl",
        )
        comparison["paired_primary"] = paired
        comparison["delta_B_minus_A"] = (
            b["primary"]["pair_accuracy"] - a["primary"]["pair_accuracy"]
        )

    comparison_path = Path(args.work_dir) / "results" / "comparison_final.json"
    write_json(comparison_path, comparison)

    print("\n" + "=" * 100)
    print("Final primary evaluation")
    print("=" * 100)
    print(
        f"START: {baseline['pair_correct']}/{baseline['n']} "
        f"= {baseline['pair_accuracy']:.4%}"
    )
    for r in results:
        p = r["primary"]
        print(
            f"{r['arm_name']}: {p['pair_correct']}/{p['n']} "
            f"= {p['pair_accuracy']:.4%}"
        )
    if "paired_primary" in comparison:
        p = comparison["paired_primary"]
        print(
            f"B - A = {p['delta_B_minus_A']:+.4%}. "
            f"A-wrong/B-right={p['A_wrong_B_right']}, "
            f"A-right/B-wrong={p['A_right_B_wrong']}, "
            f"McNemar exact p={p['mcnemar_exact_p']}"
        )

        out_tar, checksum = package_final_results(
            args, artifacts, comparison_path
        )
        print("\nSaved final artifacts:")
        print(f"  {out_tar}")
        print(f"  {checksum}")
        print(f"  SHA256: {sha256_file(out_tar)}")

    print("\nRun complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
