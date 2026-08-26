# Causal Dataflow Integrity

Anonymous review repository for *Right Reasoning, Wrong Answer: Factor Acquisition and Consolidation in Language Models for Clinical Calculation*.

## Verify the reported results

```bash
python verify.py
```

The script checks the frozen data and result hashes and reproduces the acquisition and continuation metrics reported in the paper. It also checks the seed-level McNemar results, across-seed summaries, and calculator-level joint endpoint counts. No GPU is required.

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── verify.py
├── data/
│   ├── fpit_train.jsonl
│   └── fpit_test.jsonl
├── scripts/
│   └── run_handoff.py
└── results/
    ├── acquisition/
    │   ├── protocol.json
    │   ├── endpoint_eval_summary.json
    │   └── path_eval_summary.json
    └── continuation/
        ├── seed_3407.json
        ├── seed_9173.json
        └── seed_17011.json
```

## Data

The frozen corpus contains 1,376 training pairs and 92 held-out pairs. Twelve held-out pairs exposed during engineering smoke checks were excluded from confirmatory evaluation. This leaves 80 confirmatory pairs, including the 49 structured pairs used for the primary evaluation.

The corpus was compiled from **MedCalc-Bench Verified v1.0.7**, commit `88ed8fd`, based on MedCalc-Bench. The JSONL files are derived from MedCalc-Bench and are redistributed under CC BY-SA 4.0.

`results/acquisition/protocol.json` records the corpus hashes, excluded prompt hashes, training configuration, reward definitions, and software environment.

## Acquisition results

`endpoint_eval_summary.json` is the original endpoint-only acquisition evaluation summary. `path_eval_summary.json` is the original FPIT acquisition evaluation summary. These are the frozen evaluation files used to report Table 1.

The result files retain a few internal metric names:

- `pair_accuracy` means joint endpoint accuracy
- `factor_all_correct` means complete factor correctness
- `path_full_accuracy` means CDI rate

## Matched continuation

Both continuation arms start from the same frozen FPIT checkpoint. Arm A continues FPIT. Arm B uses a strict reward that is 1 only when both BASE and NEW endpoints are correct.

The primary outcome is final-checkpoint greedy joint endpoint accuracy on the 49 structured confirmatory cases. The continuation result files cover seeds `3407`, `9173`, and `17011`.

`scripts/run_handoff.py` is the continuation script used in the study. Exact retraining is not self-contained in this anonymous review repository because the frozen acquisition archive used as the starting point is not included.

## Environment

The reported experiments used Python 3.12 on a single NVIDIA RTX 4090. The recorded package versions are listed in `requirements.txt`.
