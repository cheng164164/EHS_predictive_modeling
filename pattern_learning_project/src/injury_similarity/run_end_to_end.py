"""Run validation, final training, and prediction with no arguments."""
from __future__ import annotations
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from injury_similarity.core import run_training_workflow

if __name__ == "__main__":
    run_training_workflow(include_predictions=True)
    print("End-to-end injury similarity workflow complete")
