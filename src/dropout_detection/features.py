from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import read_csv, resolve_csv, validate_required_files


KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]
COURSE_COLUMNS = ["code_module", "code_presentation"]


def clean_column_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def _concat_or_empty(parts: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def _prefix_feature_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = {
        col: f"{prefix}_{clean_column_name(col)}"
        for col in df.columns
        if col not in KEY_COLUMNS and col not in COURSE_COLUMNS
    }
    return df.rename(columns=renamed)


def build_base_table(data_dir: str | Path, cutoff_day: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    info = read_csv(data_dir, "student_info")
    registration = read_csv(data_dir, "student_registration")
    courses = read_csv(data_dir, "courses")

    base = (
        info.merge(registration, on=KEY_COLUMNS, how="left")
        .merge(courses, on=COURSE_COLUMNS, how="left")
        .copy()
    )

    y = base["final_result"].eq("Withdrawn").astype(int)
    metadata = base[KEY_COLUMNS + ["final_result"]].copy()

    base["date_registration_missing"] = base["date_registration"].isna().astype(int)
    base["date_registration"] = base["date_registration"].fillna(0)
    base["days_registered_before_start"] = (-base["date_registration"]).clip(lower=0)
    base["registered_after_course_start"] = base["date_registration"].gt(0).astype(int)
    base["course_progress_at_cutoff"] = cutoff_day / base["module_presentation_length"]

    # Avoid leakage: final_result and date_unregistration are outcomes, not input signals.
    feature_columns = [
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
        "num_of_prev_attempts",
        "studied_credits",
        "module_presentation_length",
        "date_registration",
        "date_registration_missing",
        "days_registered_before_start",
        "registered_after_course_start",
        "course_progress_at_cutoff",
    ]
    return base[KEY_COLUMNS + feature_columns].copy(), y, metadata


def aggregate_vle_features(
    data_dir: str | Path,
    cutoff_day: int,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    vle_path = resolve_csv(data_dir, "studentVle.csv")
    vle_lookup = read_csv(data_dir, "vle")[
        ["id_site", "code_module", "code_presentation", "activity_type"]
    ].drop_duplicates()

    summary_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    activity_parts: list[pd.DataFrame] = []
    period_parts: list[pd.DataFrame] = []
    site_parts: list[pd.DataFrame] = []

    usecols = KEY_COLUMNS + ["id_site", "date", "sum_click"]
    for chunk in pd.read_csv(vle_path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["date"].le(cutoff_day)].copy()
        if chunk.empty:
            continue

        chunk = chunk.merge(vle_lookup, on=["id_site", "code_module", "code_presentation"], how="left")
        chunk["activity_type"] = chunk["activity_type"].fillna("unknown")

        summary_parts.append(
            chunk.groupby(KEY_COLUMNS, observed=True)
            .agg(
                total_clicks=("sum_click", "sum"),
                vle_events=("sum_click", "size"),
                first_activity_date=("date", "min"),
                last_activity_date=("date", "max"),
            )
            .reset_index()
        )

        daily_parts.append(
            chunk.groupby(KEY_COLUMNS + ["date"], observed=True)["sum_click"].sum().reset_index()
        )

        activity_parts.append(
            chunk.groupby(KEY_COLUMNS + ["activity_type"], observed=True)["sum_click"].sum().reset_index()
        )

        period = np.select(
            [
                chunk["date"].lt(0),
                chunk["date"].between(0, 7),
                chunk["date"].between(8, 14),
                chunk["date"].between(15, cutoff_day),
            ],
            ["pre_course", "days_00_07", "days_08_14", "days_15_cutoff"],
            default="after_cutoff",
        )
        period_chunk = chunk.assign(activity_period=period)
        period_parts.append(
            period_chunk.groupby(KEY_COLUMNS + ["activity_period"], observed=True)["sum_click"]
            .sum()
            .reset_index()
        )

        site_parts.append(chunk[KEY_COLUMNS + ["id_site"]].drop_duplicates())

    summary = _concat_or_empty(
        summary_parts,
        KEY_COLUMNS + ["total_clicks", "vle_events", "first_activity_date", "last_activity_date"],
    )
    if not summary.empty:
        summary = (
            summary.groupby(KEY_COLUMNS, observed=True)
            .agg(
                total_clicks=("total_clicks", "sum"),
                vle_events=("vle_events", "sum"),
                first_activity_date=("first_activity_date", "min"),
                last_activity_date=("last_activity_date", "max"),
            )
            .reset_index()
        )

    daily = _concat_or_empty(daily_parts, KEY_COLUMNS + ["date", "sum_click"])
    if not daily.empty:
        daily = daily.groupby(KEY_COLUMNS + ["date"], observed=True)["sum_click"].sum().reset_index()
        daily_summary = (
            daily.groupby(KEY_COLUMNS, observed=True)
            .agg(
                active_days=("date", "nunique"),
                max_clicks_in_day=("sum_click", "max"),
                mean_clicks_per_active_day=("sum_click", "mean"),
                std_clicks_per_active_day=("sum_click", "std"),
            )
            .reset_index()
        )
        summary = summary.merge(daily_summary, on=KEY_COLUMNS, how="outer")

    site = _concat_or_empty(site_parts, KEY_COLUMNS + ["id_site"])
    if not site.empty:
        site_diversity = (
            site.drop_duplicates()
            .groupby(KEY_COLUMNS, observed=True)["id_site"]
            .nunique()
            .reset_index(name="unique_sites_visited")
        )
        summary = summary.merge(site_diversity, on=KEY_COLUMNS, how="outer")

    activity = _concat_or_empty(activity_parts, KEY_COLUMNS + ["activity_type", "sum_click"])
    if not activity.empty:
        activity = (
            activity.groupby(KEY_COLUMNS + ["activity_type"], observed=True)["sum_click"]
            .sum()
            .reset_index()
        )
        activity_pivot = activity.pivot_table(
            index=KEY_COLUMNS,
            columns="activity_type",
            values="sum_click",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        activity_pivot = _prefix_feature_columns(activity_pivot, "clicks_activity")
        summary = summary.merge(activity_pivot, on=KEY_COLUMNS, how="outer")

    period_df = _concat_or_empty(period_parts, KEY_COLUMNS + ["activity_period", "sum_click"])
    if not period_df.empty:
        period_df = (
            period_df.groupby(KEY_COLUMNS + ["activity_period"], observed=True)["sum_click"]
            .sum()
            .reset_index()
        )
        period_pivot = period_df.pivot_table(
            index=KEY_COLUMNS,
            columns="activity_period",
            values="sum_click",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        period_pivot = _prefix_feature_columns(period_pivot, "clicks_period")
        summary = summary.merge(period_pivot, on=KEY_COLUMNS, how="outer")

    if summary.empty:
        return pd.DataFrame(columns=KEY_COLUMNS)

    summary["days_since_last_activity"] = cutoff_day - summary["last_activity_date"]
    summary["days_to_first_activity"] = summary["first_activity_date"]
    summary["no_vle_activity_before_cutoff"] = summary["total_clicks"].isna().astype(int)
    summary = summary.drop(columns=["first_activity_date", "last_activity_date"])
    return summary


def aggregate_assessment_features(data_dir: str | Path, cutoff_day: int, base_keys: pd.DataFrame) -> pd.DataFrame:
    assessments = read_csv(data_dir, "assessments")
    student_assessment = read_csv(data_dir, "student_assessment")

    merged = student_assessment.merge(assessments, on="id_assessment", how="left")
    submitted = merged[merged["date_submitted"].le(cutoff_day)].copy()
    submitted["score"] = pd.to_numeric(submitted["score"], errors="coerce")
    submitted["weighted_score_component"] = submitted["score"] * submitted["weight"]
    submitted["submitted_due_by_cutoff"] = submitted["date"].notna() & submitted["date"].le(cutoff_day)
    submitted["late_submission"] = submitted["date"].notna() & submitted["date_submitted"].gt(submitted["date"])
    submitted["on_time_submission"] = submitted["date"].notna() & submitted["date_submitted"].le(submitted["date"])

    if submitted.empty:
        student_features = pd.DataFrame(columns=KEY_COLUMNS)
    else:
        student_features = (
            submitted.groupby(KEY_COLUMNS, observed=True)
            .agg(
                assessment_submissions_by_cutoff=("id_assessment", "nunique"),
                submitted_due_assessments_by_cutoff=("submitted_due_by_cutoff", "sum"),
                late_submissions_by_cutoff=("late_submission", "sum"),
                on_time_submissions_by_cutoff=("on_time_submission", "sum"),
                banked_assessments_by_cutoff=("is_banked", "sum"),
                mean_score_by_cutoff=("score", "mean"),
                min_score_by_cutoff=("score", "min"),
                max_score_by_cutoff=("score", "max"),
                submitted_weight_by_cutoff=("weight", "sum"),
                weighted_score_sum_by_cutoff=("weighted_score_component", "sum"),
            )
            .reset_index()
        )
        denominator = student_features["submitted_weight_by_cutoff"].replace(0, np.nan)
        student_features["weighted_score_by_cutoff"] = (
            student_features["weighted_score_sum_by_cutoff"] / denominator
        )
        student_features = student_features.drop(columns=["weighted_score_sum_by_cutoff"])

        type_counts = (
            submitted.groupby(KEY_COLUMNS + ["assessment_type"], observed=True)["id_assessment"]
            .nunique()
            .reset_index(name="submitted_count")
            .pivot_table(
                index=KEY_COLUMNS,
                columns="assessment_type",
                values="submitted_count",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )
        type_counts = _prefix_feature_columns(type_counts, "submitted_assessment_type")
        student_features = student_features.merge(type_counts, on=KEY_COLUMNS, how="left")

    due = assessments[assessments["date"].notna() & assessments["date"].le(cutoff_day)].copy()
    if due.empty:
        course_due = base_keys[COURSE_COLUMNS].drop_duplicates().copy()
        course_due["assessments_due_by_cutoff"] = 0
        course_due["assessment_weight_due_by_cutoff"] = 0.0
    else:
        course_due = (
            due.groupby(COURSE_COLUMNS, observed=True)
            .agg(
                assessments_due_by_cutoff=("id_assessment", "nunique"),
                assessment_weight_due_by_cutoff=("weight", "sum"),
            )
            .reset_index()
        )
        due_type_counts = (
            due.groupby(COURSE_COLUMNS + ["assessment_type"], observed=True)["id_assessment"]
            .nunique()
            .reset_index(name="due_count")
            .pivot_table(
                index=COURSE_COLUMNS,
                columns="assessment_type",
                values="due_count",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )
        due_type_counts = _prefix_feature_columns(due_type_counts, "due_assessment_type")
        course_due = course_due.merge(due_type_counts, on=COURSE_COLUMNS, how="left")

    features = base_keys[KEY_COLUMNS].merge(course_due, on=COURSE_COLUMNS, how="left")
    features = features.merge(student_features, on=KEY_COLUMNS, how="left")

    for column in [
        "assessment_submissions_by_cutoff",
        "submitted_due_assessments_by_cutoff",
        "late_submissions_by_cutoff",
        "on_time_submissions_by_cutoff",
        "banked_assessments_by_cutoff",
    ]:
        if column not in features:
            features[column] = 0

    features["has_assessment_submission_by_cutoff"] = (
        features["assessment_submissions_by_cutoff"].fillna(0).gt(0).astype(int)
    )
    features["missing_due_assessments_by_cutoff"] = (
        features["assessments_due_by_cutoff"].fillna(0)
        - features["submitted_due_assessments_by_cutoff"].fillna(0)
    ).clip(lower=0)
    due_count = features["assessments_due_by_cutoff"].replace(0, np.nan)
    features["assessment_submission_rate_by_cutoff"] = (
        features["submitted_due_assessments_by_cutoff"] / due_count
    ).fillna(0)

    return features


def build_feature_matrix(
    data_dir: str | Path,
    cutoff_day: int = 30,
    chunksize: int = 1_000_000,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    validate_required_files(data_dir)

    base_features, y, metadata = build_base_table(data_dir, cutoff_day)
    base_keys = base_features[KEY_COLUMNS].copy()
    vle_features = aggregate_vle_features(data_dir, cutoff_day, chunksize=chunksize)
    assessment_features = aggregate_assessment_features(data_dir, cutoff_day, base_keys)

    features = (
        base_features.merge(vle_features, on=KEY_COLUMNS, how="left")
        .merge(assessment_features, on=KEY_COLUMNS, how="left")
        .drop(columns=KEY_COLUMNS)
    )

    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns
    features[numeric_columns] = features[numeric_columns].fillna(0)
    features = features.replace([np.inf, -np.inf], np.nan)
    features[numeric_columns] = features[numeric_columns].fillna(0)

    return features, y, metadata


def split_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    return numeric, categorical
