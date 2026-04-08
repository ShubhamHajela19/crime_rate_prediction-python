import os
from typing import Any, Optional

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from model_training import train_and_save_models
from predict import ModelArtifactsError, get_model_status, load_artifacts, predict_case


APP_TITLE = "Crime Rate Prediction Dashboard"
SERVICE_NAME = "Crime Rate Prediction API"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_FILENAME = "crime_dataset_india.csv"


def _resolve_project_path(path_value: str) -> str:
    return path_value if os.path.isabs(path_value) else os.path.join(PROJECT_ROOT, path_value)


def _default_dataset_path() -> str:
    return _resolve_project_path(os.getenv("DATASET_PATH", DEFAULT_DATASET_FILENAME))


def _parse_optional_int(value: Optional[Any]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value = str(value).strip()
    if not value:
        return None
    return int(value)


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    app.config.update(
        APP_TITLE=APP_TITLE,
        DATASET_PATH=_default_dataset_path(),
        MODEL_DIR=_resolve_project_path(os.getenv("MODEL_DIR", "models")),
        TRAIN_MAX_ROWS=os.getenv("TRAIN_MAX_ROWS"),
    )

    os.makedirs(app.config["MODEL_DIR"], exist_ok=True)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            app_title=app.config["APP_TITLE"],
            default_dataset_path=app.config["DATASET_PATH"],
        )

    @app.get("/api/status")
    @app.get("/health")
    def api_status():
        model_dir = app.config["MODEL_DIR"]
        dataset_path = app.config["DATASET_PATH"]
        status_payload = get_model_status(model_dir)
        return jsonify(
            {
                "status": "ok",
                "service": SERVICE_NAME,
                "project": "Crime Rate Prediction using K-Means Clustering and Random Forest",
                "dataset": {
                    "path": dataset_path,
                    "exists": os.path.exists(dataset_path),
                },
                **status_payload,
            }
        )

    @app.post("/api/train")
    @app.post("/train")
    def api_train():
        try:
            request_data = request.get_json(silent=True) or {}
            dataset_path_input = str(
                request_data.get("dataset_path") or app.config["DATASET_PATH"]
            ).strip()
            dataset_path = _resolve_project_path(dataset_path_input)
            max_rows = _parse_optional_int(
                request_data.get("max_rows", app.config["TRAIN_MAX_ROWS"])
            )

            training_details = train_and_save_models(
                dataset_path=dataset_path,
                model_dir=app.config["MODEL_DIR"],
                max_rows=max_rows,
            )
            load_artifacts.cache_clear()
            latest_status = get_model_status(app.config["MODEL_DIR"])

            return jsonify(
                {
                    "status": "success",
                    "message": "Model training completed successfully.",
                    "details": training_details,
                    "model_overview": latest_status.get("model_overview", {}),
                    "available_versions": latest_status.get("available_versions", []),
                }
            )
        except FileNotFoundError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception as exc:  # pragma: no cover
            return jsonify({"status": "error", "message": f"Training failed: {exc}"}), 500

    @app.post("/api/predict")
    @app.post("/predict")
    def api_predict():
        try:
            payload = request.get_json(silent=True)
            if not payload:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "A JSON body is required for prediction.",
                        }
                    ),
                    400,
                )

            prediction = predict_case(payload, model_dir=app.config["MODEL_DIR"])
            return jsonify({"status": "success", "prediction": prediction})
        except ModelArtifactsError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception as exc:  # pragma: no cover
            return jsonify({"status": "error", "message": f"Prediction failed: {exc}"}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=os.getenv("FLASK_DEBUG", "1").strip().lower() not in {"0", "false", "no"},
    )
