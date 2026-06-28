from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import build_feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score dropout risk for student-course records.")
    parser.add_argument("--data-dir", default=".", help="Folder containing source CSV files.")
    parser.add_argument("--model-path", default="models/dropout_model.joblib", help="Trained model artifact.")
    parser.add_argument("--output", default="reports/dropout_risk_predictions.csv", help="Prediction CSV path.")
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="Rows per chunk for studentVle.csv.")
    return parser.parse_args()


def risk_band(probability: float) -> str:
    if probability >= 0.75:
        return "high"
    if probability >= 0.5:
        return "medium"
    if probability >= 0.25:
        return "watch"
    return "low"


def align_features(features: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    aligned = features.copy()
    for column in expected_columns:
        if column not in aligned:
            aligned[column] = 0
    return aligned[expected_columns]


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.model_path)
    cutoff_day = int(artifact["cutoff_day"])

    features, target, metadata = build_feature_matrix(
        args.data_dir,
        cutoff_day=cutoff_day,
        chunksize=args.chunksize,
    )
    features = align_features(features, artifact["feature_columns"])

    probabilities = artifact["model"].predict_proba(features)[:, 1]
    predictions = probabilities >= artifact["threshold"]

    output = metadata.copy()
    output["actual_dropout"] = target
    output["dropout_probability"] = np.round(probabilities, 6)
    output["predicted_dropout"] = predictions.astype(int)
    output["risk_band"] = [risk_band(value) for value in probabilities]
    output = output.sort_values("dropout_probability", ascending=False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
