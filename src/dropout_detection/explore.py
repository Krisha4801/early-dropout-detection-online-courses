from __future__ import annotations

import argparse

import pandas as pd

from .data import available_csvs, read_csv
from .features import build_feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quickly inspect the dropout detection data.")
    parser.add_argument("--data-dir", default=".", help="Folder containing source CSV files.")
    parser.add_argument("--cutoff-day", type=int, default=30, help="Feature cutoff day.")
    parser.add_argument("--skip-features", action="store_true", help="Only print raw table summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Resolved CSV files:")
    for key, path in available_csvs(args.data_dir).items():
        print(f"  {key}: {path}")

    info = read_csv(args.data_dir, "student_info")
    print("\nFinal result distribution:")
    print(info["final_result"].value_counts().to_string())

    print("\nRaw table shapes:")
    for key in ["courses", "assessments", "vle", "student_info", "student_registration", "student_assessment"]:
        frame = read_csv(args.data_dir, key)
        print(f"  {key}: {frame.shape[0]:,} rows x {frame.shape[1]} columns")

    if args.skip_features:
        return

    features, target, _ = build_feature_matrix(args.data_dir, cutoff_day=args.cutoff_day)
    print("\nFeature matrix:")
    print(f"  rows: {features.shape[0]:,}")
    print(f"  features: {features.shape[1]:,}")
    print(f"  dropout rate: {target.mean():.2%}")
    print("\nFeature preview:")
    with pd.option_context("display.max_columns", 12, "display.width", 140):
        print(features.head().to_string(index=False))


if __name__ == "__main__":
    main()
