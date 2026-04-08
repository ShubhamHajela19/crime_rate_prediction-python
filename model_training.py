import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "Case Closed"

RAW_COLUMNS: List[str] = [
    "Date Reported",
    "Date of Occurrence",
    "Time of Occurrence",
    "City",
    "Crime Code",
    "Crime Description",
    "Victim Age",
    "Victim Gender",
    "Weapon Used",
    "Crime Domain",
    "Police Deployed",
    TARGET_COLUMN,
]

CATEGORICAL_COLUMNS: List[str] = [
    "City",
    "Crime Description",
    "Victim Gender",
    "Weapon Used",
    "Crime Domain",
]

NUMERICAL_COLUMNS: List[str] = [
    "Crime Code",
    "Victim Age",
    "Police Deployed",
    "Occurrence Year",
    "Occurrence Month",
    "Occurrence Hour",
    "Report Delay Hours",
]

FEATURE_COLUMNS: List[str] = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
REQUIRED_COLUMNS: List[str] = RAW_COLUMNS

MODEL_FILENAMES = {
    "rf_model": "rf_model.pkl",
    "kmeans_model": "kmeans_model.pkl",
    "scaler": "scaler.pkl",
    "encoders": "encoders.pkl",
    "latest_bundle": "crime_pipeline_latest.joblib",
}
VERSIONED_BUNDLE_TEMPLATE = "crime_pipeline_{version}.joblib"
VERSION_ARCHIVE_DIRNAME = "versions"


def _clean_categorical(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("Unknown").astype(str).str.strip()
    cleaned = cleaned.replace({"": "Unknown", "nan": "Unknown", "NaN": "Unknown", "None": "Unknown"})
    return cleaned


def _coerce_numeric(series: pd.Series) -> Tuple[pd.Series, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    median_value = float(numeric.median()) if not numeric.dropna().empty else 0.0
    numeric = numeric.fillna(median_value)
    return numeric, median_value


def _parse_mixed_datetime(series: pd.Series) -> pd.Series:
    clean = series.astype(str).str.strip()
    clean = clean.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan})

    parsed = pd.to_datetime(clean, errors="coerce", dayfirst=True)
    missing_mask = parsed.isna() & clean.notna()
    if missing_mask.any():
        parsed_alt = pd.to_datetime(clean[missing_mask], errors="coerce", dayfirst=False)
        parsed.loc[missing_mask] = parsed_alt
    return parsed


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))


def _load_dataset(dataset_path: str, max_rows: Optional[int]) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path, usecols=REQUIRED_COLUMNS, low_memory=False)
    if df.empty:
        raise ValueError("Dataset is empty.")

    if max_rows and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)

    _validate_columns(df)
    return df


def _derive_time_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    occurrence_dt = _parse_mixed_datetime(prepared["Date of Occurrence"])
    reported_dt = _parse_mixed_datetime(prepared["Date Reported"])
    occurrence_time_dt = _parse_mixed_datetime(prepared["Time of Occurrence"])

    best_occurrence_dt = occurrence_dt.fillna(occurrence_time_dt)
    fallback_dt = reported_dt.fillna(best_occurrence_dt)

    prepared["Occurrence Year"] = best_occurrence_dt.dt.year.fillna(fallback_dt.dt.year)
    prepared["Occurrence Month"] = best_occurrence_dt.dt.month.fillna(fallback_dt.dt.month)
    prepared["Occurrence Hour"] = occurrence_time_dt.dt.hour.fillna(best_occurrence_dt.dt.hour).fillna(0)

    report_delay_hours = (reported_dt - best_occurrence_dt).dt.total_seconds() / 3600.0
    report_delay_hours = report_delay_hours.clip(lower=0)
    prepared["Report Delay Hours"] = report_delay_hours
    return prepared


def _prepare_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    prepared = _derive_time_features(df)
    fill_values: Dict[str, object] = {}

    for col in CATEGORICAL_COLUMNS + [TARGET_COLUMN]:
        prepared[col] = _clean_categorical(prepared[col])
        fill_values[col] = "Unknown"

    for col in NUMERICAL_COLUMNS:
        prepared[col], median_value = _coerce_numeric(prepared[col])
        fill_values[col] = median_value

    prepared["Crime Code"] = prepared["Crime Code"].clip(lower=0)
    prepared["Victim Age"] = prepared["Victim Age"].clip(lower=0, upper=120)
    prepared["Police Deployed"] = prepared["Police Deployed"].clip(lower=0, upper=500)
    prepared["Occurrence Year"] = prepared["Occurrence Year"].clip(lower=2000, upper=2100)
    prepared["Occurrence Month"] = prepared["Occurrence Month"].clip(lower=1, upper=12)
    prepared["Occurrence Hour"] = prepared["Occurrence Hour"].clip(lower=0, upper=23)
    prepared["Report Delay Hours"] = prepared["Report Delay Hours"].clip(lower=0, upper=24 * 365 * 5)
    return prepared, fill_values


def _fit_label_encoders(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], LabelEncoder]:
    encoded = df.copy()
    categorical_encoders: Dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        values = encoded[col].astype(str)
        if "Unknown" not in set(values):
            values = pd.concat([values, pd.Series(["Unknown"])], ignore_index=True)
        encoder.fit(values)
        encoded[col] = encoder.transform(encoded[col].astype(str))
        categorical_encoders[col] = encoder

    target_encoder = LabelEncoder()
    target_encoder.fit(encoded[TARGET_COLUMN].astype(str))
    return encoded, categorical_encoders, target_encoder


def _estimate_elbow_k(inertia_by_k: Dict[int, float]) -> int:
    k_values = np.array(sorted(inertia_by_k.keys()), dtype=int)
    inertias = np.array([inertia_by_k[k] for k in k_values], dtype=float)
    if len(k_values) < 3:
        return int(k_values[0])

    first_derivative = np.diff(inertias)
    second_derivative = np.diff(first_derivative)
    elbow_index = int(np.argmax(np.abs(second_derivative))) + 1
    return int(k_values[elbow_index])


def _kmeans_search(
    X_scaled: np.ndarray, min_k: int = 2, max_k: int = 10
) -> Tuple[Dict[int, float], Dict[int, float], int, int]:
    n_samples = X_scaled.shape[0]
    if n_samples < 3:
        raise ValueError("At least 3 records are required for clustering.")

    sample_size = min(12000, n_samples)
    if sample_size < n_samples:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n_samples, size=sample_size, replace=False)
        x_eval = X_scaled[idx]
    else:
        x_eval = X_scaled

    max_k = min(max_k, max(min_k, sample_size - 1))
    k_values = list(range(min_k, max_k + 1))
    inertia_by_k: Dict[int, float] = {}
    silhouette_by_k: Dict[int, float] = {}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(x_eval)
        inertia_by_k[k] = float(kmeans.inertia_)
        if len(np.unique(labels)) > 1:
            silhouette_by_k[k] = float(silhouette_score(x_eval, labels))
        else:
            silhouette_by_k[k] = -1.0

    best_silhouette_k = max(silhouette_by_k, key=silhouette_by_k.get)
    elbow_k = _estimate_elbow_k(inertia_by_k)
    return inertia_by_k, silhouette_by_k, int(best_silhouette_k), int(elbow_k)


def _generate_model_version() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:6]}"


def _build_zone_map(sorted_clusters: List[int]) -> Dict[str, str]:
    if not sorted_clusters:
        return {}
    if len(sorted_clusters) == 1:
        return {str(sorted_clusters[0]): "medium"}
    if len(sorted_clusters) == 2:
        return {
            str(sorted_clusters[0]): "high",
            str(sorted_clusters[1]): "low",
        }

    zone_map: Dict[str, str] = {}
    grouped_clusters = np.array_split(np.array(sorted_clusters, dtype=int), 3)
    zone_order = ["high", "medium", "low"]
    for zone_name, cluster_group in zip(zone_order, grouped_clusters):
        for cluster_id in cluster_group.tolist():
            zone_map[str(int(cluster_id))] = zone_name
    return zone_map


def _serialize_float_map(values: Dict[int, float]) -> Dict[str, float]:
    return {str(key): round(float(value), 4) for key, value in values.items()}


def _top_feature_importance(
    rf_model: RandomForestClassifier,
    feature_names: List[str],
    top_n: int = 6,
) -> List[Dict[str, float]]:
    if not hasattr(rf_model, "feature_importances_"):
        return []

    ranked = sorted(
        zip(feature_names, rf_model.feature_importances_),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [
        {
            "feature": str(feature_name),
            "importance": round(float(importance), 4),
        }
        for feature_name, importance in ranked[:top_n]
    ]


def train_and_save_models(
    dataset_path: str,
    model_dir: str = "models",
    max_rows: Optional[int] = None,
) -> Dict[str, object]:
    df = _load_dataset(dataset_path, max_rows=max_rows)
    prepared_df, fill_values = _prepare_dataframe(df)

    if prepared_df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Target column has fewer than 2 classes.")

    encoded_df, categorical_encoders, target_encoder = _fit_label_encoders(prepared_df)
    X = encoded_df[FEATURE_COLUMNS].astype(float).values
    y = target_encoder.transform(encoded_df[TARGET_COLUMN].astype(str).values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertia_by_k, silhouette_by_k, best_silhouette_k, elbow_k = _kmeans_search(X_scaled)
    selected_k = best_silhouette_k

    kmeans_model = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=10)
    cluster_ids = kmeans_model.fit_predict(X_scaled)

    target_as_text = prepared_df[TARGET_COLUMN].astype(str).str.lower().str.strip()
    closed_tokens = {"yes", "closed", "true", "1"}
    open_case_mask = ~target_as_text.isin(closed_tokens)

    cluster_open_rate: Dict[int, float] = {}
    for cluster in sorted(np.unique(cluster_ids)):
        cluster_mask = cluster_ids == cluster
        cluster_open_rate[int(cluster)] = float(open_case_mask[cluster_mask].mean())

    sorted_clusters = sorted(
        cluster_open_rate.keys(),
        key=lambda cluster_id: cluster_open_rate[cluster_id],
        reverse=True,
    )
    cluster_zone_map = _build_zone_map(sorted_clusters)
    cluster_distribution = {
        str(int(cluster_id)): int((cluster_ids == cluster_id).sum())
        for cluster_id in sorted(np.unique(cluster_ids))
    }
    zone_distribution = {"high": 0, "medium": 0, "low": 0}
    for cluster_id in cluster_ids:
        zone_name = cluster_zone_map.get(str(int(cluster_id)), "medium")
        zone_distribution[zone_name] = zone_distribution.get(zone_name, 0) + 1

    class_counts = pd.Series(y).value_counts()
    stratify_target = y if len(np.unique(y)) > 1 and int(class_counts.min()) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify_target,
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=1,
        class_weight="balanced_subsample",
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    model_version = _generate_model_version()
    trained_at = datetime.now(timezone.utc).isoformat()
    feature_importance = _top_feature_importance(rf_model, FEATURE_COLUMNS)
    cluster_summary = [
        {
            "cluster_id": int(cluster_id),
            "records": cluster_distribution[str(int(cluster_id))],
            "crime_zone": cluster_zone_map.get(str(int(cluster_id)), "medium"),
            "open_case_rate": round(cluster_open_rate[int(cluster_id)], 4),
        }
        for cluster_id in sorted(np.unique(cluster_ids))
    ]

    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(rf_model, os.path.join(model_dir, MODEL_FILENAMES["rf_model"]))
    joblib.dump(kmeans_model, os.path.join(model_dir, MODEL_FILENAMES["kmeans_model"]))
    joblib.dump(scaler, os.path.join(model_dir, MODEL_FILENAMES["scaler"]))
    encoder_payload = {
        "model_version": model_version,
        "trained_at": trained_at,
        "categorical_encoders": categorical_encoders,
        "target_encoder": target_encoder,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numerical_columns": NUMERICAL_COLUMNS,
        "fill_values": fill_values,
        "target_column": TARGET_COLUMN,
        "dataset_profile": {
            "rows": int(len(prepared_df)),
            "raw_columns": RAW_COLUMNS,
            "dataset_path": dataset_path,
        },
        "random_forest_metrics": {
            "accuracy": round(accuracy, 4),
            "precision_weighted": round(float(precision), 4),
            "recall_weighted": round(float(recall), 4),
            "f1_weighted": round(float(f1_score), 4),
        },
        "feature_importance": feature_importance,
        "kmeans_metadata": {
            "selected_k": selected_k,
            "elbow_k": elbow_k,
            "best_silhouette_k": best_silhouette_k,
            "inertia_by_k": _serialize_float_map(inertia_by_k),
            "silhouette_by_k": _serialize_float_map(silhouette_by_k),
            "cluster_zone_map": cluster_zone_map,
            "cluster_open_case_rate": _serialize_float_map(cluster_open_rate),
            "cluster_distribution": cluster_distribution,
            "zone_distribution": zone_distribution,
            "cluster_summary": cluster_summary,
        },
    }
    joblib.dump(encoder_payload, os.path.join(model_dir, MODEL_FILENAMES["encoders"]))

    versions_dir = os.path.join(model_dir, VERSION_ARCHIVE_DIRNAME)
    os.makedirs(versions_dir, exist_ok=True)

    latest_bundle_path = os.path.join(model_dir, MODEL_FILENAMES["latest_bundle"])
    versioned_bundle_name = VERSIONED_BUNDLE_TEMPLATE.format(version=model_version)
    versioned_bundle_path = os.path.join(versions_dir, versioned_bundle_name)
    bundle_payload = {
        "model_version": model_version,
        "trained_at": trained_at,
        "rf_model": rf_model,
        "kmeans_model": kmeans_model,
        "scaler": scaler,
        "metadata": encoder_payload,
    }
    joblib.dump(bundle_payload, latest_bundle_path)
    joblib.dump(bundle_payload, versioned_bundle_path)

    return {
        "model_version": model_version,
        "trained_at": trained_at,
        "rows_used_for_training": int(len(prepared_df)),
        "feature_columns": FEATURE_COLUMNS,
        "target_classes": target_encoder.classes_.tolist(),
        "random_forest_metrics": encoder_payload["random_forest_metrics"],
        "feature_importance": feature_importance,
        "artifacts": {
            "latest_bundle": MODEL_FILENAMES["latest_bundle"],
            "versioned_bundle": versioned_bundle_name,
            "model_directory": model_dir,
        },
        "kmeans_summary": {
            "selected_k": selected_k,
            "elbow_k": elbow_k,
            "best_silhouette_k": best_silhouette_k,
            "silhouette_by_k": _serialize_float_map(silhouette_by_k),
            "inertia_by_k": _serialize_float_map(inertia_by_k),
            "cluster_zone_map": cluster_zone_map,
            "cluster_distribution": cluster_distribution,
            "zone_distribution": zone_distribution,
            "cluster_open_case_rate": _serialize_float_map(cluster_open_rate),
            "cluster_summary": cluster_summary,
        },
    }
