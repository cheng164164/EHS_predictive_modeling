"""Score near-miss/hazard candidates against the final model."""
from __future__ import annotations
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from injury_similarity.core import predict_injury_similarity

if __name__ == "__main__":
    predict_injury_similarity()
    print("Prediction complete")
