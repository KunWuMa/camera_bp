from pathlib import Path

import pandas as pd


RESULTS = Path(__file__).resolve().parents[1] / "artifacts" / "results"
FILES = [
    "bspc_added_deep_predictions.csv",
    "bspc_model_suite_predictions.csv",
    "bspc_model_suite_seed_metrics.csv",
    "bspc_model_suite_seed_summary.csv",
    "bspc_model_suite_leakage_statistics.csv",
    "bspc_model_suite_architecture_statistics.csv",
    "bspc_model_suite_complexity.csv",
]


def main() -> None:
    for name in FILES:
        frame = pd.read_csv(RESULTS / name, low_memory=False)
        print(f"{name}: shape={frame.shape}, missing={int(frame.isna().sum().sum())}")

    predictions = pd.read_csv(RESULTS / FILES[0], low_memory=False)
    print("new_model_rows:", predictions["model"].value_counts().sort_index().to_dict())
    folds = predictions.groupby(
        ["cohort", "protocol", "model", "target", "seed"]
    )["fold"].nunique()
    print("fold_count_distribution:", folds.value_counts().sort_index().to_dict())
    print("fold_counts_by_cohort_protocol:")
    print(folds.groupby(level=[0, 1]).agg(["min", "max"]).to_string())
    print("prediction_columns:", predictions.columns.tolist())
    id_column = "sample_id" if "sample_id" in predictions.columns else "unit_id"
    keys = ["cohort", "protocol", "model", "target", "seed", id_column]
    print("duplicate_prediction_keys:", int(predictions.duplicated(keys).sum()))
    print("\ncomplexity:")
    print(pd.read_csv(RESULTS / FILES[-1]).to_string(index=False))

    public = pd.read_csv(RESULTS / "bspc_model_suite_predictions.csv", low_memory=False)
    public = public[public["cohort"].eq("public_ukl")]
    subject_counts = public.groupby(["model", "protocol", "target"])["subject_id"].nunique()
    print("\npublic_subject_counts:")
    print(subject_counts.to_string())


if __name__ == "__main__":
    main()
