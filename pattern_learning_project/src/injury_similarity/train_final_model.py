"""Fit the final all-record production model only."""
from __future__ import annotations
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from injury_similarity.core import train_final_model

if __name__ == "__main__":
    train_final_model()
    print("Final model training complete")
