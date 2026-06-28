from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import build_feature_matrix, split_feature_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an early dropout detection model.")
    parser.add_argument("--data-dir", default=".", help="Folder containing the source CSV files.")
    parser.add_argument("--model-dir", default="models", help="Folder for the trained model artifact.")
    parser.add_argument("--report-dir", default="reports", help="Folder for metrics and report CSVs.")
    parser.add_argument("--cutoff-day", type=int, default=30, help="Use behavior up to this course day.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation share.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="Rows per chunk for studentVle.csv.")
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.55,
        help="Minimum validation precision allowed when choosing a recall-first threshold.",
    )
    return parser.parse_args()


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def candidate_models(random_state: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def best_f1_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = probabilities >= threshold
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def threshold_sweep(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = probabilities >= threshold
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
                "alerts": int(np.sum(predictions)),
                "false_positives": int(confusion_matrix(y_true, predictions).ravel()[1]),
                "false_negatives": int(confusion_matrix(y_true, predictions).ravel()[2]),
            }
        )
    return pd.DataFrame(rows)


def recall_first_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    min_precision: float,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    sweep = threshold_sweep(y_true, probabilities)
    candidates = sweep[sweep["precision"].ge(min_precision)].copy()
    if candidates.empty:
        candidates = sweep.copy()
        strategy = (
            f"No threshold reached precision >= {min_precision:.2f}; "
            "fallback selected the best F2 threshold."
        )
        selected = candidates.sort_values(["f2", "recall"], ascending=False).iloc[0]
    else:
        strategy = (
            f"Recall-first threshold: maximize recall while keeping precision >= {min_precision:.2f}."
        )
        selected = candidates.sort_values(["recall", "f2", "threshold"], ascending=[False, False, False]).iloc[0]

    details = {
        "strategy": strategy,
        "min_precision": float(min_precision),
        "selected_precision": float(selected["precision"]),
        "selected_recall": float(selected["recall"]),
        "selected_f2": float(selected["f2"]),
    }
    return float(selected["threshold"]), details, sweep


def evaluate_predictions(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = probabilities >= threshold
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            target_names=["not_dropout", "dropout"],
            zero_division=0,
            output_dict=True,
        ),
    }


def evaluate_binary_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def rule_based_baseline(features: pd.DataFrame) -> np.ndarray:
    """A transparent baseline an interviewer can understand in 20 seconds."""
    due = features.get("assessments_due_by_cutoff", pd.Series(0, index=features.index)).fillna(0)
    submitted_due = features.get("submitted_due_assessments_by_cutoff", pd.Series(0, index=features.index)).fillna(0)
    mean_score = features.get("mean_score_by_cutoff", pd.Series(0, index=features.index)).fillna(0)
    total_clicks = features.get("total_clicks", pd.Series(0, index=features.index)).fillna(0)
    active_days = features.get("active_days", pd.Series(0, index=features.index)).fillna(0)
    days_since_last = features.get("days_since_last_activity", pd.Series(0, index=features.index)).fillna(0)
    late_submissions = features.get("late_submissions_by_cutoff", pd.Series(0, index=features.index)).fillna(0)

    submission_rate = (submitted_due / due.replace(0, np.nan)).fillna(1)
    risk_points = (
        total_clicks.lt(120).astype(int)
        + active_days.lt(6).astype(int)
        + days_since_last.ge(7).astype(int)
        + (due.sub(submitted_due).gt(0)).astype(int)
        + submission_rate.lt(0.75).astype(int)
        + ((mean_score.gt(0)) & mean_score.lt(55)).astype(int)
        + late_submissions.gt(0).astype(int)
    )
    return risk_points.ge(2).to_numpy()


def flatten_metrics(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    matrix = metrics.get("confusion_matrix", [[None, None], [None, None]])
    return {
        "model": name,
        "threshold": metrics.get("threshold", ""),
        "roc_auc": metrics.get("roc_auc", ""),
        "average_precision": metrics.get("average_precision", ""),
        "accuracy": metrics.get("accuracy", ""),
        "balanced_accuracy": metrics.get("balanced_accuracy", ""),
        "precision": metrics.get("precision", ""),
        "recall": metrics.get("recall", ""),
        "f1": metrics.get("f1", ""),
        "f2": metrics.get("f2", ""),
        "true_negatives": matrix[0][0],
        "false_positives": matrix[0][1],
        "false_negatives": matrix[1][0],
        "true_positives": matrix[1][1],
    }


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    clean_names = [
        name.replace("num__", "").replace("cat__", "").replace("onehot__", "")
        for name in feature_names
    ]

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        frame = pd.DataFrame({"feature": clean_names, "importance": values})
        return frame.sort_values("importance", ascending=False)

    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_[0]
        frame = pd.DataFrame(
            {
                "feature": clean_names,
                "coefficient": coefficients,
                "importance": np.abs(coefficients),
            }
        )
        return frame.sort_values("importance", ascending=False)

    return pd.DataFrame({"feature": clean_names})


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    features, target, metadata = build_feature_matrix(
        args.data_dir,
        cutoff_day=args.cutoff_day,
        chunksize=args.chunksize,
    )
    numeric_features, categorical_features = split_feature_types(features)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_index, valid_index = next(splitter.split(features, target, groups=metadata["id_student"]))
    x_train, x_valid = features.iloc[train_index], features.iloc[valid_index]
    y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
    valid_metadata = metadata.iloc[valid_index].reset_index(drop=True)

    results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Pipeline] = {}
    for name, classifier in candidate_models(args.random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_valid)[:, 1]
        threshold, threshold_details, _ = recall_first_threshold(
            y_valid,
            probabilities,
            min_precision=args.min_precision,
        )
        f1_threshold, _ = best_f1_threshold(y_valid, probabilities)
        metrics = evaluate_predictions(y_valid, probabilities, threshold)
        metrics["model_name"] = name
        metrics["threshold_strategy"] = threshold_details
        metrics["best_f1_threshold"] = f1_threshold
        metrics["best_f1_threshold_metrics"] = evaluate_predictions(y_valid, probabilities, f1_threshold)
        results[name] = metrics
        fitted_models[name] = pipeline

    best_name = max(results, key=lambda name: results[name]["average_precision"])
    best_pipeline = fitted_models[best_name]
    best_threshold = results[best_name]["threshold"]
    best_probabilities = best_pipeline.predict_proba(x_valid)[:, 1]
    best_predictions = best_probabilities >= best_threshold
    _, _, best_sweep = recall_first_threshold(y_valid, best_probabilities, min_precision=args.min_precision)

    baseline_predictions = rule_based_baseline(x_valid)
    baseline_metrics = evaluate_binary_predictions(y_valid, baseline_predictions)

    artifact = {
        "model": best_pipeline,
        "threshold": best_threshold,
        "cutoff_day": args.cutoff_day,
        "feature_columns": features.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_name": best_name,
        "metrics": results[best_name],
        "threshold_strategy": results[best_name]["threshold_strategy"],
    }
    model_path = model_dir / "dropout_model.joblib"
    joblib.dump(artifact, model_path)

    metrics_path = report_dir / "metrics.json"
    metrics_payload = {
        "best_model": best_name,
        "selection_metric": "average_precision",
        "positive_class": "dropout / Withdrawn",
        "business_objective": "Prioritize recall because missing a true dropout is more costly than a false alert.",
        "baseline": baseline_metrics,
        "models": results,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    feature_columns_path = report_dir / "feature_columns.csv"
    pd.DataFrame({"feature": features.columns}).to_csv(feature_columns_path, index=False)

    importance_path = report_dir / "feature_importance.csv"
    get_feature_importance(best_pipeline).head(60).to_csv(importance_path, index=False)

    best_sweep.to_csv(report_dir / "threshold_tradeoffs.csv", index=False)

    comparison_rows = [flatten_metrics("rule_based_baseline", baseline_metrics)]
    comparison_rows.extend(flatten_metrics(name, metrics) for name, metrics in results.items())
    pd.DataFrame(comparison_rows).to_csv(report_dir / "baseline_vs_model_report.csv", index=False)

    validation_predictions = valid_metadata.copy()
    validation_predictions["actual_dropout"] = y_valid.reset_index(drop=True)
    validation_predictions["dropout_probability"] = best_probabilities
    validation_predictions["predicted_dropout"] = best_predictions.astype(int)
    validation_predictions["baseline_predicted_dropout"] = baseline_predictions.astype(int)
    validation_predictions.to_csv(report_dir / "validation_predictions.csv", index=False)

    model_card = f"""# Early Dropout Detection Model Card

## Objective

Predict whether a learner will withdraw from an online course using only early behavior up to day {args.cutoff_day}.

## Positive class

`final_result == "Withdrawn"` is treated as dropout.

## Decision policy

This is an early-warning system, so recall matters more than raw accuracy. The chosen threshold is not the default 0.50. It uses this policy:

{results[best_name]["threshold_strategy"]["strategy"]}

Selected threshold: `{best_threshold:.3f}`

## Baseline comparison

The model is compared against a transparent rule baseline that flags learners when multiple simple risk signals appear: low clicks, few active days, recent inactivity, missing assessments, low score, or late submissions.

See `baseline_vs_model_report.csv` for exact metrics.

## Best model

Best model by validation average precision: `{best_name}`.

Validation metrics at the recall-first threshold:

- Precision: `{results[best_name]["precision"]:.3f}`
- Recall: `{results[best_name]["recall"]:.3f}`
- F2: `{results[best_name]["f2"]:.3f}`
- ROC AUC: `{results[best_name]["roc_auc"]:.3f}`
- Average precision: `{results[best_name]["average_precision"]:.3f}`

## Explainability

The app returns local feature explanations for each prediction using SHAP when available, with a fallback to model feature importance. Positive contributions push the learner toward dropout risk; negative contributions push toward persistence.
"""
    (report_dir / "model_card.md").write_text(model_card, encoding="utf-8")

    print(f"Built feature matrix: {features.shape[0]} rows x {features.shape[1]} features")
    print(f"Best model: {best_name}")
    print(f"ROC AUC: {results[best_name]['roc_auc']:.4f}")
    print(f"Recall: {results[best_name]['recall']:.4f}")
    print(f"Precision: {results[best_name]['precision']:.4f}")
    print(f"F2: {results[best_name]['f2']:.4f}")
    print(f"Threshold strategy: {results[best_name]['threshold_strategy']['strategy']}")
    print(f"Saved model: {model_path}")
    print(f"Saved reports: {report_dir}")


if __name__ == "__main__":
    main()
