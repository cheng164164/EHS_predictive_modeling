"""Build site/department/month any-injury risk datasets without training a model.

Run directly from the project root:

    python src/injury_risk_classification/build_classification_dataset.py

All paths and tunable parameters are configured in:

    src/injury_risk_classification/config.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from injury_risk_classification import config
from injury_risk_classification.feature_engineering import build_classification_dataset
from injury_risk_classification.utils import save_json


def _clustering_requested() -> bool:
    return config.FEATURE_SET in {"with_clusters", "both", "experiments", "all"}


def main() -> None:
    clustered_path = Path(config.CLUSTERED_PATTERN_RECORDS_PATH)
    clustering_requested = _clustering_requested()
    if clustering_requested and not clustered_path.exists() and config.REQUIRE_CLUSTERED_RECORDS_FOR_CLUSTER_FEATURES:
        raise FileNotFoundError(
            f"Pattern features were requested, but the clustered/theme pattern file does not exist: {clustered_path}\n"
            "Run the unsupervised pipeline first: "
            "python src/pattern_learning_unsupervised/train_pattern_clusters_hdbscan.py"
        )

    clustered_records = clustered_path if clustered_path.exists() and clustering_requested else None
    experiments = config.get_pattern_feature_experiments() if config.FEATURE_SET in {"experiments", "all"} else []
    bundle = build_classification_dataset(
        input_dir=config.INPUT_DIR,
        output_dir=config.OUTPUT_DIR,
        clustered_records_path=clustered_records,
        horizon_months=config.HORIZON_MONTHS,
        target_type=getattr(config, "TARGET_TYPE", "any_injury"),
        rolling_windows=list(config.ROLLING_WINDOWS),
        top_n_clusters=config.TOP_N_CLUSTERS,
        min_history_months=config.MIN_HISTORY_MONTHS,
        reference_date=config.REFERENCE_DATE,
        write_outputs=True,
        pattern_feature_config=config.get_pattern_feature_config(),
        pattern_feature_experiments=experiments,
    )

    feature_dir = Path(config.FEATURE_OUTPUT_DIR)
    save_json(bundle.metadata, feature_dir / "classification_dataset_metadata.json")
    print("Saved feature datasets under:", feature_dir)
    print("Target:", bundle.metadata.get("target_column"))
    print("Baseline modeling rows:", len(bundle.baseline_dataset))
    print("Baseline scoring rows:", 0 if bundle.baseline_scoring_dataset is None else len(bundle.baseline_scoring_dataset))
    print("Default pattern modeling rows:", 0 if bundle.with_cluster_dataset is None else len(bundle.with_cluster_dataset))
    print("Default pattern scoring rows:", 0 if bundle.with_cluster_scoring_dataset is None else len(bundle.with_cluster_scoring_dataset))
    if bundle.pattern_datasets:
        print("Experiment datasets:")
        for name, df in bundle.pattern_datasets.items():
            score_rows = 0 if not bundle.pattern_scoring_datasets or name not in bundle.pattern_scoring_datasets else len(bundle.pattern_scoring_datasets[name])
            print(f"  {name}: modeling_rows={len(df):,}, scoring_rows={score_rows:,}")
    if not clustering_requested:
        print("FEATURE_SET is baseline, so pattern features were not requested.")
    elif clustered_records is None:
        print("No clustered/theme records file was found. Baseline dataset was created; pattern datasets were skipped.")


if __name__ == "__main__":
    main()
