import glob
import os
import re
from functools import lru_cache
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


MODEL_FILES = {
    "rf_model": "rf_model.pkl",
    "kmeans_model": "kmeans_model.pkl",
    "scaler": "scaler.pkl",
    "encoders": "encoders.pkl",
    "latest_bundle": "crime_pipeline_latest.joblib",
}
BUNDLE_PREFIX = "crime_pipeline_"
VERSION_ARCHIVE_DIRNAME = "versions"


class ModelArtifactsError(RuntimeError):
    pass


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _require_file(path: str) -> None:
    if not os.path.exists(path):
        raise ModelArtifactsError(
            f"Missing model artifact: {path}. Run /api/train first."
        )


def _versioned_bundle_paths(model_dir: str) -> List[str]:
    bundle_paths: List[str] = []
    for candidate_dir in _candidate_model_dirs(model_dir):
        pattern = os.path.join(candidate_dir, f"{BUNDLE_PREFIX}*.joblib")
        bundle_paths.extend(
            [
                path
                for path in glob.glob(pattern)
                if os.path.basename(path) != MODEL_FILES["latest_bundle"]
            ]
        )
    return sorted(set(bundle_paths), key=os.path.getmtime, reverse=True)


def _candidate_model_dirs(model_dir: str) -> List[str]:
    candidates = [model_dir, os.path.join(model_dir, VERSION_ARCHIVE_DIRNAME)]
    unique_candidates: List[str] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _normalize_zone_label(value: object) -> str:
    normalized = _normalize_key(value)
    if any(token in normalized for token in ["critical", "veryhigh", "high"]):
        return "high"
    if any(token in normalized for token in ["moderate", "medium"]):
        return "medium"
    if "low" in normalized:
        return "low"
    return "medium"


def _normalize_case_closed_prediction(value: object) -> str:
    normalized = _normalize_key(value)
    if normalized in {"yes", "closed", "solved", "true", "1"}:
        return "Closed"
    if normalized in {"no", "open", "unsolved", "notsolved", "false", "0"}:
        return "Open"
    return str(value)


def _artifact_file_status(model_dir: str) -> Dict[str, bool]:
    return {
        MODEL_FILES["latest_bundle"]: any(
            os.path.exists(os.path.join(candidate_dir, MODEL_FILES["latest_bundle"]))
            for candidate_dir in _candidate_model_dirs(model_dir)
        ),
        MODEL_FILES["rf_model"]: any(
            os.path.exists(os.path.join(candidate_dir, MODEL_FILES["rf_model"]))
            for candidate_dir in _candidate_model_dirs(model_dir)
        ),
        MODEL_FILES["kmeans_model"]: any(
            os.path.exists(os.path.join(candidate_dir, MODEL_FILES["kmeans_model"]))
            for candidate_dir in _candidate_model_dirs(model_dir)
        ),
        MODEL_FILES["scaler"]: any(
            os.path.exists(os.path.join(candidate_dir, MODEL_FILES["scaler"]))
            for candidate_dir in _candidate_model_dirs(model_dir)
        ),
        MODEL_FILES["encoders"]: any(
            os.path.exists(os.path.join(candidate_dir, MODEL_FILES["encoders"]))
            for candidate_dir in _candidate_model_dirs(model_dir)
        ),
    }


def _list_model_versions(model_dir: str) -> List[Dict[str, object]]:
    versions: List[Dict[str, object]] = []
    for path in _versioned_bundle_paths(model_dir):
        file_name = os.path.basename(path)
        version = file_name.replace(BUNDLE_PREFIX, "").replace(".joblib", "")
        versions.append(
            {
                "version": version,
                "file_name": file_name,
                "updated_at": os.path.getmtime(path),
            }
        )
    return versions


def _build_model_overview(encoders: Dict[str, object]) -> Dict[str, object]:
    kmeans_metadata = encoders.get("kmeans_metadata", {})
    raw_zone_distribution = kmeans_metadata.get("zone_distribution", {})
    zone_distribution = {"high": 0, "medium": 0, "low": 0}
    for key, value in raw_zone_distribution.items():
        zone_distribution[_normalize_zone_label(key)] += int(value)

    cluster_distribution = {
        str(key): int(value)
        for key, value in kmeans_metadata.get("cluster_distribution", {}).items()
    }
    cluster_summary = [
        {
            "cluster_id": int(item.get("cluster_id", 0)),
            "records": int(item.get("records", 0)),
            "crime_zone": _normalize_zone_label(item.get("crime_zone", "medium")),
            "open_case_rate": round(float(item.get("open_case_rate", 0.0)), 4),
        }
        for item in kmeans_metadata.get("cluster_summary", [])
    ]
    feature_importance = [
        {
            "feature": str(item.get("feature")),
            "importance": round(float(item.get("importance", 0.0)), 4),
        }
        for item in encoders.get("feature_importance", [])
    ]

    return {
        "model_version": encoders.get("model_version", "legacy"),
        "trained_at": encoders.get("trained_at"),
        "rows_used_for_training": int(encoders.get("dataset_profile", {}).get("rows", 0)),
        "feature_count": len(encoders.get("feature_columns", [])),
        "selected_k": kmeans_metadata.get("selected_k"),
        "zone_distribution": zone_distribution,
        "cluster_distribution": cluster_distribution,
        "cluster_summary": cluster_summary,
        "metrics": encoders.get("random_forest_metrics", {}),
        "feature_importance": feature_importance,
    }


def get_model_status(model_dir: str = "models") -> Dict[str, object]:
    os.makedirs(model_dir, exist_ok=True)
    artifact_files = _artifact_file_status(model_dir)
    available_versions = _list_model_versions(model_dir)
    bundle_ready = artifact_files.get(MODEL_FILES["latest_bundle"], False) or bool(available_versions)
    legacy_ready = all(
        artifact_files.get(file_name, False)
        for file_name in [
            MODEL_FILES["rf_model"],
            MODEL_FILES["kmeans_model"],
            MODEL_FILES["scaler"],
            MODEL_FILES["encoders"],
        ]
    )
    models_ready = bundle_ready or legacy_ready

    response: Dict[str, object] = {
        "models_ready": models_ready,
        "artifact_files": artifact_files,
        "available_versions": available_versions,
    }
    if not models_ready:
        return response

    try:
        _, _, _, encoders = load_artifacts(model_dir)
        response["model_overview"] = _build_model_overview(encoders)
    except Exception as exc:  # pragma: no cover
        response["models_ready"] = False
        response["load_error"] = str(exc)
    return response


def _prepare_rf_model_for_inference(rf_model):
    if hasattr(rf_model, "n_jobs"):
        rf_model.n_jobs = 1
    return rf_model


@lru_cache(maxsize=4)
def load_artifacts(model_dir: str) -> Tuple[object, object, object, Dict[str, object]]:
    for candidate_dir in _candidate_model_dirs(model_dir):
        latest_bundle_path = os.path.join(candidate_dir, MODEL_FILES["latest_bundle"])
        if os.path.exists(latest_bundle_path):
            bundle = joblib.load(latest_bundle_path)
            metadata = bundle.get("metadata") or bundle.get("encoders")
            if not metadata:
                raise ModelArtifactsError("Latest model bundle is missing metadata.")
            rf_model = _prepare_rf_model_for_inference(bundle["rf_model"])
            return rf_model, bundle["kmeans_model"], bundle["scaler"], metadata

    versioned_bundles = _versioned_bundle_paths(model_dir)
    if versioned_bundles:
        bundle = joblib.load(versioned_bundles[0])
        metadata = bundle.get("metadata") or bundle.get("encoders")
        if not metadata:
            raise ModelArtifactsError("Versioned model bundle is missing metadata.")
        rf_model = _prepare_rf_model_for_inference(bundle["rf_model"])
        return rf_model, bundle["kmeans_model"], bundle["scaler"], metadata

    for candidate_dir in _candidate_model_dirs(model_dir):
        rf_model_path = os.path.join(candidate_dir, MODEL_FILES["rf_model"])
        kmeans_path = os.path.join(candidate_dir, MODEL_FILES["kmeans_model"])
        scaler_path = os.path.join(candidate_dir, MODEL_FILES["scaler"])
        encoders_path = os.path.join(candidate_dir, MODEL_FILES["encoders"])

        if all(os.path.exists(path) for path in [rf_model_path, kmeans_path, scaler_path, encoders_path]):
            rf_model = _prepare_rf_model_for_inference(joblib.load(rf_model_path))
            kmeans_model = joblib.load(kmeans_path)
            scaler = joblib.load(scaler_path)
            encoders = joblib.load(encoders_path)
            return rf_model, kmeans_model, scaler, encoders

    rf_model_path = os.path.join(model_dir, MODEL_FILES["rf_model"])
    _require_file(rf_model_path)
    raise ModelArtifactsError(
        f"Missing companion model artifacts in {model_dir}. Run /api/train first."
    )


def _build_input_frame(payload: Dict[str, object], encoders: Dict[str, object]) -> pd.DataFrame:
    feature_columns = encoders["feature_columns"]
    categorical_columns = set(encoders["categorical_columns"])
    numerical_columns = set(encoders["numerical_columns"])
    fill_values = encoders["fill_values"]

    normalized_payload = {_normalize_key(key): value for key, value in payload.items()}
    row: Dict[str, object] = {}

    for col in feature_columns:
        normalized_col = _normalize_key(col)
        default_value = fill_values.get(col, "Unknown" if col in categorical_columns else 0.0)
        value = normalized_payload.get(normalized_col, default_value)

        if col in categorical_columns:
            value = str(value).strip() if value is not None else "Unknown"
            if value == "" or value.lower() in {"nan", "none"}:
                value = "Unknown"
        elif col in numerical_columns:
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float(default_value)

        row[col] = value

    return pd.DataFrame([row], columns=feature_columns)


def _encode_and_scale(input_frame: pd.DataFrame, scaler, encoders: Dict[str, object]) -> np.ndarray:
    categorical_encoders = encoders["categorical_encoders"]
    categorical_columns = encoders["categorical_columns"]
    numerical_columns = encoders["numerical_columns"]
    feature_columns = encoders["feature_columns"]
    fill_values = encoders["fill_values"]

    encoded = input_frame.copy()
    for col in categorical_columns:
        encoder = categorical_encoders[col]
        value = str(encoded.at[0, col])
        classes = set(encoder.classes_.tolist())
        # Unknown category values are safely mapped to "Unknown" if available.
        if value not in classes:
            value = "Unknown" if "Unknown" in classes else str(encoder.classes_[0])
        encoded.at[0, col] = int(encoder.transform([value])[0])

    for col in numerical_columns:
        encoded[col] = pd.to_numeric(encoded[col], errors="coerce").fillna(fill_values[col])

    X = encoded[feature_columns].astype(float).values
    return scaler.transform(X)


def predict_case(payload: Dict[str, object], model_dir: str = "models") -> Dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Prediction input must be a JSON object.")

    rf_model, kmeans_model, scaler, encoders = load_artifacts(model_dir)
    input_frame = _build_input_frame(payload, encoders)
    X_scaled = _encode_and_scale(input_frame, scaler, encoders)

    solved_prediction_idx = int(rf_model.predict(X_scaled)[0])
    target_encoder = encoders["target_encoder"]
    solved_prediction_label = str(
        target_encoder.inverse_transform([solved_prediction_idx])[0]
    )

    confidence = None
    if hasattr(rf_model, "predict_proba"):
        probabilities = rf_model.predict_proba(X_scaled)[0]
        confidence = float(np.max(probabilities))

    cluster_id = int(kmeans_model.predict(X_scaled)[0])
    kmeans_metadata = encoders.get("kmeans_metadata", {})
    zone_map = kmeans_metadata.get("cluster_zone_map", {})
    raw_zone = zone_map.get(str(cluster_id), zone_map.get(cluster_id, "medium"))
    crime_zone = _normalize_zone_label(raw_zone)
    case_closed_prediction = _normalize_case_closed_prediction(solved_prediction_label)

    return {
        "crime_zone": crime_zone,
        "crime_zone_display": f"{crime_zone.title()} Crime Zone",
        "cluster_id": cluster_id,
        "case_closed_prediction": case_closed_prediction,
        "probability_score": round(confidence, 4) if confidence is not None else None,
        "model_version": encoders.get("model_version", "legacy"),
        "trained_at": encoders.get("trained_at"),
        "raw_prediction_label": solved_prediction_label,
        "input_features": input_frame.to_dict(orient="records")[0],
    }
