from __future__ import annotations

"""Quantitative audit analyses requested during BSPC manuscript revision.

The script deliberately reuses the frozen fold-generation code and saved
out-of-fold predictions; it does not refit any BP model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from common import ensure_dirs, load_config
from run_bspc_extension_experiments import private_protocols, private_split
from train_deep import load_arrays


BOOTSTRAP_REPEATS = 5000
SEED = 20260907


def variance_components(y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Return one-way between/within decomposition and ICC(1,1).

    The unbalanced-repeat correction n0 follows the standard one-way random
    effects definition: (N - sum(n_i^2)/N)/(k-1).
    """
    frame = pd.DataFrame({"y": np.asarray(y, float), "group": np.asarray(groups)})
    frame = frame.dropna()
    grand = float(frame["y"].mean())
    grouped = frame.groupby("group", sort=False)["y"]
    counts = grouped.size().astype(float)
    means = grouped.mean()
    ss_between = float(np.sum(counts * (means - grand) ** 2))
    centered = frame["y"] - frame["group"].map(means)
    ss_within = float(np.sum(centered**2))
    total = ss_between + ss_within
    k, n = len(counts), len(frame)
    ms_between = ss_between / (k - 1) if k > 1 else np.nan
    ms_within = ss_within / (n - k) if n > k else np.nan
    n0 = (n - float(np.sum(counts**2)) / n) / (k - 1) if k > 1 else np.nan
    denominator = ms_between + (n0 - 1) * ms_within
    icc = (ms_between - ms_within) / denominator if denominator != 0 else np.nan
    return {
        "n_sessions": int(n),
        "n_subjects": int(k),
        "ss_between": ss_between,
        "ss_within": ss_within,
        "eta_squared": ss_between / total if total else np.nan,
        "icc_1_1": float(icc),
        "n0": float(n0),
    }


def bootstrap_components(
    frame: pd.DataFrame, target: str, repeats: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    subjects = frame["subject_id"].drop_duplicates().to_numpy()
    eta, icc = np.empty(repeats), np.empty(repeats)
    subject_frames = {key: part.copy() for key, part in frame.groupby("subject_id", sort=False)}
    for repeat in range(repeats):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        pieces = []
        for draw, subject in enumerate(sampled):
            part = subject_frames[subject].copy()
            # A subject sampled twice must remain two independent bootstrap clusters.
            part["bootstrap_subject"] = draw
            pieces.append(part)
        sample = pd.concat(pieces, ignore_index=True)
        values = variance_components(sample[target].to_numpy(), sample["bootstrap_subject"].to_numpy())
        eta[repeat], icc[repeat] = values["eta_squared"], values["icc_1_1"]
    return eta, icc


def identity_variance(frame: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for target in ("sbp", "dbp", "hr"):
        estimate = variance_components(frame[target].to_numpy(), frame["subject_id"].to_numpy())
        eta_boot, icc_boot = bootstrap_components(frame, target, BOOTSTRAP_REPEATS, rng)
        rows.append(
            {
                "target": target.upper(),
                **estimate,
                "eta_squared_ci_low": float(np.nanquantile(eta_boot, 0.025)),
                "eta_squared_ci_high": float(np.nanquantile(eta_boot, 0.975)),
                "icc_1_1_ci_low": float(np.nanquantile(icc_boot, 0.025)),
                "icc_1_1_ci_high": float(np.nanquantile(icc_boot, 0.975)),
                "bootstrap_unit": "subject",
                "bootstrap_repeats": BOOTSTRAP_REPEATS,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "identity_shortcut_rigorous.csv", index=False, encoding="utf-8-sig")
    return result


def identity_classifier(frame: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    feature_columns = [
        column
        for column in frame.columns
        if column.startswith("face_") and pd.api.types.is_numeric_dtype(frame[column])
    ]
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["subject_id"])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.full(len(frame), -1, dtype=int)
    rows: list[dict] = []
    acquisition_column = "session_id" if "session_id" in frame.columns else None
    for fold, (train, test) in enumerate(cv.split(frame[feature_columns], labels)):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(C=10, kernel="rbf"),
        )
        model.fit(frame.iloc[train][feature_columns], labels[train])
        predicted = model.predict(frame.iloc[test][feature_columns])
        oof[test] = predicted
        overlap = 0
        if acquisition_column:
            overlap = len(
                set(frame.iloc[train][acquisition_column])
                & set(frame.iloc[test][acquisition_column])
            )
        rows.append(
            {
                "row_type": "fold",
                "fold": fold,
                "train_acquisitions": len(train),
                "test_acquisitions": len(test),
                "train_test_acquisition_overlap": overlap,
                "accuracy": accuracy_score(labels[test], predicted),
                "balanced_accuracy": balanced_accuracy_score(labels[test], predicted),
                "chance_accuracy": 1 / len(encoder.classes_),
                "n_subjects": len(encoder.classes_),
                "n_features": len(feature_columns),
                "split_unit": "acquisition",
            }
        )

    # Cluster bootstrap of the fixed out-of-fold predictions.
    rng = np.random.default_rng(SEED + 1)
    subjects = frame["subject_id"].drop_duplicates().to_numpy()
    indices = {subject: np.flatnonzero(frame["subject_id"].to_numpy() == subject) for subject in subjects}
    boot_accuracy = np.empty(BOOTSTRAP_REPEATS)
    for repeat in range(BOOTSTRAP_REPEATS):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        selected = np.concatenate([indices[subject] for subject in sampled])
        boot_accuracy[repeat] = accuracy_score(labels[selected], oof[selected])
    rows.append(
        {
            "row_type": "summary",
            "fold": "all",
            "train_acquisitions": np.nan,
            "test_acquisitions": len(frame),
            "train_test_acquisition_overlap": 0,
            "accuracy": accuracy_score(labels, oof),
            "balanced_accuracy": balanced_accuracy_score(labels, oof),
            "chance_accuracy": 1 / len(encoder.classes_),
            "n_subjects": len(encoder.classes_),
            "n_features": len(feature_columns),
            "split_unit": "acquisition",
            "accuracy_ci_low": float(np.quantile(boot_accuracy, 0.025)),
            "accuracy_ci_high": float(np.quantile(boot_accuracy, 0.975)),
            "bootstrap_unit": "subject",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
        }
    )
    result = pd.DataFrame(rows)
    result.to_csv(
        out_dir / "identity_classifier_acquisition_disjoint.csv", index=False, encoding="utf-8-sig"
    )
    return result


def overlapping_neighbor_rate(
    test_idx: np.ndarray, train_idx: np.ndarray, sessions: np.ndarray
) -> float:
    train = set(np.asarray(train_idx, int).tolist())
    has_neighbor = []
    by_session: dict[object, list[int]] = {}
    for index, session in enumerate(sessions):
        by_session.setdefault(session, []).append(index)
    positions = {
        index: (ordered, position)
        for ordered in by_session.values()
        for position, index in enumerate(ordered)
    }
    for index in np.asarray(test_idx, int):
        ordered, position = positions[index]
        neighbors = []
        if position > 0:
            neighbors.append(ordered[position - 1])
        if position + 1 < len(ordered):
            neighbors.append(ordered[position + 1])
        has_neighbor.append(any(neighbor in train for neighbor in neighbors))
    return float(np.mean(has_neighbor)) if has_neighbor else np.nan


def split_overlap(data: dict, cfg: dict, out_dir: Path) -> pd.DataFrame:
    fold_rows: list[dict] = []
    for cohort in ("hospital", "lab"):
        base = np.flatnonzero(data["source"] == cohort)
        protocols = private_protocols(data, cohort, base, cfg["seed"])
        for protocol, assignment in protocols.items():
            for fold in sorted(np.unique(assignment[base]).astype(int)):
                train_idx, _, test_idx = private_split(
                    data, base, cohort, protocol, assignment, fold, cfg["seed"]
                )
                train_subjects, test_subjects = set(data["subject"][train_idx]), set(data["subject"][test_idx])
                train_sessions, test_sessions = set(data["session"][train_idx]), set(data["session"][test_idx])
                fold_rows.append(
                    {
                        "cohort": cohort,
                        "protocol": protocol,
                        "fold": fold,
                        "test_windows": len(test_idx),
                        "test_subjects": len(test_subjects),
                        "test_acquisitions": len(test_sessions),
                        "subject_overlap_rate": len(train_subjects & test_subjects) / len(test_subjects),
                        "acquisition_overlap_rate": len(train_sessions & test_sessions) / len(test_sessions),
                        "overlapping_neighbor_rate": overlapping_neighbor_rate(
                            test_idx, train_idx, data["session"]
                        ),
                    }
                )

    public_audit = pd.read_csv(out_dir / "bspc_extension_split_audit.csv")
    public_audit = public_audit[public_audit["cohort"].eq("public_ukl")]
    for row in public_audit.itertuples(index=False):
        fold_rows.append(
            {
                "cohort": row.cohort,
                "protocol": row.protocol,
                "fold": row.fold,
                "test_windows": row.test_windows,
                "test_subjects": row.test_subjects,
                "test_acquisitions": np.nan,
                "subject_overlap_rate": row.train_test_subject_overlap / row.test_subjects,
                "acquisition_overlap_rate": np.nan,
                "overlapping_neighbor_rate": np.nan,
            }
        )

    folds = pd.DataFrame(fold_rows)
    rows = []
    for (cohort, protocol), part in folds.groupby(["cohort", "protocol"], sort=False):
        row = {
            "cohort": cohort,
            "protocol": protocol,
            "folds": len(part),
            "total_test_windows_across_folds": int(part["test_windows"].sum()),
        }
        for metric in ("subject_overlap_rate", "acquisition_overlap_rate", "overlapping_neighbor_rate"):
            row[f"mean_{metric}"] = part[metric].mean()
            row[f"min_{metric}"] = part[metric].min()
            row[f"max_{metric}"] = part[metric].max()
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "split_overlap_quantitative.csv", index=False, encoding="utf-8-sig")
    return result


def weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def agreement_statistics(out_dir: Path) -> pd.DataFrame:
    # The extension file is the canonical ANN/SNN prediction table.  The wider
    # suite table also contains appended architecture runs and is not needed here.
    frame = pd.read_csv(out_dir / "bspc_extension_predictions.csv", low_memory=False)
    frame = frame[
        frame["protocol"].eq("Subject-disjoint") & frame["model"].eq("Face-ANN")
    ]
    ensemble = frame.groupby(
        ["cohort", "target", "subject_id", "unit_id", "unit_type"], as_index=False
    ).agg(actual=("actual", "first"), prediction=("prediction", "mean"))
    rows = []
    for (cohort, target), part in ensemble.groupby(["cohort", "target"], sort=False):
        counts = part.groupby("subject_id")["unit_id"].transform("count").to_numpy(float)
        weights = 1 / counts
        errors = part["prediction"].to_numpy() - part["actual"].to_numpy()
        bias = float(np.average(errors, weights=weights))
        error_sd = weighted_sd(errors, weights)
        slope, intercept, correlation, _, _ = stats.linregress(part["actual"], part["prediction"])
        rows.append(
            {
                "cohort": cohort,
                "target": target.upper(),
                "model": "Face-ANN seed ensemble",
                "n_units": len(part),
                "n_subjects": part["subject_id"].nunique(),
                "subject_macro_mae": part.assign(abs_error=np.abs(errors)).groupby("subject_id")["abs_error"].mean().mean(),
                "subject_balanced_bias": bias,
                "subject_balanced_error_sd": error_sd,
                "loa_lower": bias - 1.96 * error_sd,
                "loa_upper": bias + 1.96 * error_sd,
                "pooled_pearson_r": correlation,
                "pooled_r_squared": correlation**2,
                "prediction_reference_slope": slope,
                "prediction_reference_intercept": intercept,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "strict_agreement_statistics.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    cfg = load_config()
    root = ensure_dirs(cfg)
    out_dir = root / "results"
    features = pd.read_csv(root / "features" / "session_features.csv")
    laboratory = features[features["source"].eq("lab")].reset_index(drop=True)
    data = load_arrays(root / "features" / "windows.h5")

    variance = identity_variance(laboratory, out_dir)
    classifier = identity_classifier(laboratory, out_dir)
    overlap = split_overlap(data, cfg, out_dir)
    agreement = agreement_statistics(out_dir)

    print("IDENTITY VARIANCE")
    print(variance.to_string(index=False))
    print("\nIDENTITY CLASSIFIER")
    print(classifier.tail(1).to_string(index=False))
    print("\nSPLIT OVERLAP")
    print(overlap.to_string(index=False))
    print("\nSTRICT AGREEMENT")
    print(agreement.to_string(index=False))


if __name__ == "__main__":
    main()
