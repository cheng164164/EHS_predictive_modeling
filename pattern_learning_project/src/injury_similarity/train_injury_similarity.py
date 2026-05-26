"""Run temporal validation, then fit the final all-record model."""
from __future__ import annotations
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from injury_similarity.core import run_training_workflow

def train() -> dict:
    return run_training_workflow(include_predictions=False)

if __name__ == "__main__":
    train()
    print("Training workflow complete")
