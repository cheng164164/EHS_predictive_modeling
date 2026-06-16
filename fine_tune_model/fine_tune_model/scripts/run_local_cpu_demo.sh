#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python src/01_prepare_data.py --config configs/config.yaml
python src/03_train_qlora.py --config configs/config.yaml
python src/04_evaluate.py --config configs/config.yaml
