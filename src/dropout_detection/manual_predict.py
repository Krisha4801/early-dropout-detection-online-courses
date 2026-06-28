from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

try:
    import shap
except ImportError:  # pragma: no cover - handled at runtime for clearer API output.
    shap = None


DEFAULT_CATEGORICAL = {
    "code_module": "AAA",
    "code_presentation": "2014J",
    "gender": "M",
    "region": "East Anglian Region",
    "highest_education": "A Level or Equivalent",
    "imd_band": "50-60%",
    "age_band": "0-35",
    "disability": "N",
}

ALIASES = {
    "resource_clicks": "clicks_activity_resource",
    "forum_clicks": "clicks_activity_forumng",
    "homepage_clicks": "clicks_activity_homepage",
    "content_clicks": "clicks_activity_oucontent",
    "subpage_clicks": "clicks_activity_subpage",
    "url_clicks": "clicks_activity_url",
    "quiz_clicks": "clicks_activity_quiz",
    "clicks_pre_course": "clicks_period_pre_course",
    "clicks_days_00_07": "clicks_period_days_00_07",
    "clicks_days_08_14": "clicks_period_days_08_14",
    "clicks_days_15_cutoff": "clicks_period_days_15_cutoff",
    "due_assessments": "assessments_due_by_cutoff",
    "due_weight": "assessment_weight_due_by_cutoff",
    "assessment_submissions": "assessment_submissions_by_cutoff",
    "submitted_due_assessments": "submitted_due_assessments_by_cutoff",
    "late_submissions": "late_submissions_by_cutoff",
    "on_time_submissions": "on_time_submissions_by_cutoff",
    "banked_assessments": "banked_assessments_by_cutoff",
    "mean_score": "mean_score_by_cutoff",
    "min_score": "min_score_by_cutoff",
    "max_score": "max_score_by_cutoff",
    "submitted_weight": "submitted_weight_by_cutoff",
    "weighted_score": "weighted_score_by_cutoff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict dropout risk from one JSON payload.")
    parser.add_argument("--model-path", default="models/dropout_model.joblib")
    parser.add_argument("--input", help="Input JSON file. If omitted, stdin is used.")
    return parser.parse_args()


def as_number(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def as_int(value: Any, default: int = 0) -> int:
    return int(round(as_number(value, default)))


def risk_band(probability: float) -> str:
    if probability >= 0.75:
        return "high"
    if probability >= 0.5:
        return "medium"
    if probability >= 0.25:
        return "watch"
    return "low"


def recommendations(features: dict[str, Any], probability: float) -> list[str]:
    notes: list[str] = []
    if features.get("missing_due_assessments_by_cutoff", 0) > 0:
        notes.append("Prioritize missed due assessments before adding new coursework.")
    if features.get("active_days", 0) < 5:
        notes.append("Increase weekly learning touchpoints; activity is sparse for this window.")
    if features.get("days_since_last_activity", 0) >= 7:
        notes.append("Reach out quickly; the learner has been inactive for a week or more.")
    if features.get("mean_score_by_cutoff", 0) and features.get("mean_score_by_cutoff", 0) < 50:
        notes.append("Offer assessment support; early scores are below a comfortable passing range.")
    if probability >= 0.75 and not notes:
        notes.append("Schedule a direct support check-in and review the engagement pattern.")
    if not notes:
        notes.append("Maintain regular nudges and keep watching the next assessment milestone.")
    return notes[:3]


def display_feature_name(feature: str) -> str:
    names = {
        "total_clicks": "total clicks",
        "active_days": "active days",
        "days_since_last_activity": "days since last activity",
        "missing_due_assessments_by_cutoff": "missing due assessments",
        "assessment_submission_rate_by_cutoff": "assessment submission rate",
        "mean_score_by_cutoff": "average score",
        "late_submissions_by_cutoff": "late submissions",
        "studied_credits": "studied credits",
        "unique_sites_visited": "unique sites visited",
        "mean_clicks_per_active_day": "mean clicks per active day",
        "clicks_period_days_15_cutoff": "clicks during days 15-30",
        "submitted_due_assessments_by_cutoff": "submitted due assessments",
    }
    return names.get(feature, feature.replace("_", " "))


def transformed_to_original_feature(name: str, categorical_features: list[str]) -> str:
    clean = name.replace("num__", "").replace("cat__", "")
    for feature in sorted(categorical_features, key=len, reverse=True):
        if clean == feature or clean.startswith(f"{feature}_"):
            return feature
    return clean


def shap_class_one_values(raw_values: Any) -> np.ndarray:
    values = np.asarray(raw_values)
    if isinstance(raw_values, list):
        return np.asarray(raw_values[1])[0]
    if values.ndim == 3:
        return values[0, :, 1]
    if values.ndim == 2:
        return values[0]
    return values


def local_explanations(features: pd.DataFrame, artifact: dict[str, Any], top_n: int = 5) -> list[dict[str, Any]]:
    pipeline = artifact["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    categorical_features = artifact.get("categorical_features", [])

    try:
        transformed = preprocessor.transform(features)
        if sparse.issparse(transformed):
            transformed = transformed.toarray()

        transformed_names = preprocessor.get_feature_names_out()

        if shap is not None and artifact.get("model_name") == "random_forest":
            explainer = shap.TreeExplainer(classifier)
            raw_values = explainer.shap_values(transformed, check_additivity=False)
            contributions = shap_class_one_values(raw_values)
        elif hasattr(classifier, "coef_"):
            contributions = transformed[0] * classifier.coef_[0]
        else:
            contributions = np.zeros(len(transformed_names))

        grouped: dict[str, float] = {}
        for transformed_name, contribution in zip(transformed_names, contributions):
            original_name = transformed_to_original_feature(transformed_name, categorical_features)
            grouped[original_name] = grouped.get(original_name, 0.0) + float(contribution)

        feature_values = features.iloc[0].to_dict()
        rows = []
        for feature, contribution in grouped.items():
            if feature not in feature_values:
                continue
            rows.append(
                {
                    "feature": feature,
                    "label": display_feature_name(feature),
                    "value": feature_values[feature],
                    "contribution": round(contribution, 6),
                    "direction": "increases risk" if contribution > 0 else "lowers risk",
                }
            )

        rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return rows[:top_n]
    except Exception as error:
        return [
            {
                "feature": "explanation_unavailable",
                "label": "explanation unavailable",
                "value": "",
                "contribution": 0,
                "direction": "neutral",
                "note": str(error),
            }
        ]


def normalize_payload(payload: dict[str, Any], artifact: dict[str, Any]) -> pd.DataFrame:
    feature_columns = artifact["feature_columns"]
    categorical_features = set(artifact.get("categorical_features", []))
    cutoff_day = as_number(payload.get("cutoff_day"), artifact.get("cutoff_day", 30))

    features: dict[str, Any] = {}
    for column in feature_columns:
        features[column] = DEFAULT_CATEGORICAL.get(column, "Unknown") if column in categorical_features else 0.0

    normalized_payload = dict(payload)
    for source, target in ALIASES.items():
        if source in normalized_payload and target not in normalized_payload:
            normalized_payload[target] = normalized_payload[source]

    for column in feature_columns:
        if column in normalized_payload:
            if column in categorical_features:
                value = normalized_payload[column]
                features[column] = DEFAULT_CATEGORICAL.get(column, "Unknown") if value in (None, "") else str(value)
            else:
                features[column] = as_number(normalized_payload[column])

    module_length = max(as_number(features.get("module_presentation_length"), 268), 1)
    features["course_progress_at_cutoff"] = cutoff_day / module_length

    if "days_registered_before_start" in normalized_payload and "date_registration" not in normalized_payload:
        days_before = as_number(normalized_payload["days_registered_before_start"])
        features["date_registration"] = -days_before

    date_registration = as_number(features.get("date_registration"))
    features["date_registration_missing"] = 0
    features["days_registered_before_start"] = max(-date_registration, 0)
    features["registered_after_course_start"] = int(date_registration > 0)

    total_clicks = as_number(features.get("total_clicks"))
    active_days = as_number(features.get("active_days"))
    if active_days > 0 and as_number(features.get("mean_clicks_per_active_day")) == 0:
        features["mean_clicks_per_active_day"] = total_clicks / active_days
    if active_days > 0 and as_number(features.get("max_clicks_in_day")) == 0:
        features["max_clicks_in_day"] = max(total_clicks / active_days, 0)
    if as_number(features.get("vle_events")) == 0 and total_clicks > 0:
        features["vle_events"] = max(active_days, round(total_clicks / 4))

    if "last_activity_day" in normalized_payload:
        features["days_since_last_activity"] = max(cutoff_day - as_number(normalized_payload["last_activity_day"]), 0)
    if "first_activity_day" in normalized_payload:
        features["days_to_first_activity"] = as_number(normalized_payload["first_activity_day"])
    features["no_vle_activity_before_cutoff"] = int(total_clicks <= 0 and active_days <= 0)

    assessment_due = as_number(features.get("assessments_due_by_cutoff"))
    submitted_due = as_number(features.get("submitted_due_assessments_by_cutoff"))
    submissions = as_number(features.get("assessment_submissions_by_cutoff"))
    if submissions == 0 and submitted_due > 0:
        features["assessment_submissions_by_cutoff"] = submitted_due
        submissions = submitted_due
    features["has_assessment_submission_by_cutoff"] = int(submissions > 0)
    features["missing_due_assessments_by_cutoff"] = max(assessment_due - submitted_due, 0)
    features["assessment_submission_rate_by_cutoff"] = submitted_due / assessment_due if assessment_due else 0

    mean_score = as_number(features.get("mean_score_by_cutoff"))
    if mean_score:
        if as_number(features.get("min_score_by_cutoff")) == 0:
            features["min_score_by_cutoff"] = mean_score
        if as_number(features.get("max_score_by_cutoff")) == 0:
            features["max_score_by_cutoff"] = mean_score
        if as_number(features.get("weighted_score_by_cutoff")) == 0:
            features["weighted_score_by_cutoff"] = mean_score

    if assessment_due and as_number(features.get("due_assessment_type_tma")) == 0:
        features["due_assessment_type_tma"] = assessment_due
    if submissions and as_number(features.get("submitted_assessment_type_tma")) == 0:
        features["submitted_assessment_type_tma"] = submissions

    frame = pd.DataFrame([{column: features[column] for column in feature_columns}])
    return frame


def predict(payload: dict[str, Any], model_path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(model_path)
    features = normalize_payload(payload, artifact)
    probability = float(artifact["model"].predict_proba(features)[:, 1][0])
    threshold = float(artifact["threshold"])
    predicted = probability >= threshold
    feature_dict = features.iloc[0].to_dict()
    explanations = local_explanations(features, artifact)
    return {
        "dropoutProbability": round(probability, 6),
        "threshold": round(threshold, 6),
        "predictedDropout": bool(predicted),
        "riskBand": risk_band(probability),
        "modelName": artifact.get("model_name", "model"),
        "cutoffDay": int(artifact.get("cutoff_day", 30)),
        "thresholdStrategy": artifact.get("threshold_strategy", {}),
        "explanations": explanations,
        "recommendations": recommendations(feature_dict, probability),
    }


def main() -> None:
    args = parse_args()
    if args.input:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    else:
        payload = json.loads(sys.stdin.read())
    result = predict(payload, args.model_path)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
