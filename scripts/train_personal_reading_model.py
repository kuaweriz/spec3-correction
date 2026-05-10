#!/usr/bin/env python3
"""Train the local personal READ/LIVE detector from recorded samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_managers.personal_reading_model import PersonalReadingLearner


def main() -> int:
    parser = argparse.ArgumentParser(description="Train personal reading detector")
    parser.add_argument(
        "--support-dir",
        type=Path,
        default=None,
        help="Directory containing reading_samples.csv and output reading_model.json",
    )
    args = parser.parse_args()

    learner = PersonalReadingLearner(support_dir=args.support_dir)
    result = learner.train()
    print(result.message)
    print(f"samples={result.samples} read={result.positives} non_read={result.negatives}")
    if result.success:
        print(f"model={learner.model_path}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
