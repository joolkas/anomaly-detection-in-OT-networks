from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    mode: str  # "prep" | "train-forecasting" | "train-classification" | "train-all" | "online"


@dataclass(frozen=True)
class Paths:
    """All filesystem paths live here (avoid hardcoding elsewhere)."""

    project_root: Path
    raw_data_dir: Path
    device_csv_name: str
    processed_dir: Path
    artifacts_dir: Path

    @property
    def raw_csv_path(self) -> Path:
        return self.raw_data_dir / self.device_csv_name

    @property
    def processed_forecasting_csv(self) -> Path:
        return self.processed_dir / "forecasting_features.csv"

    @property
    def processed_classification_csv(self) -> Path:
        return self.processed_dir / "classification_features.csv"

    @property
    def forecasting_artifacts_dir(self) -> Path:
        return self.artifacts_dir / "forecasting"

    @property
    def classification_artifacts_dir(self) -> Path:
        return self.artifacts_dir / "classification"


@dataclass(frozen=True)
class PreprocessingConfig:
    resample_freq: str
    forecasting_include_keywords: tuple[str, ...]
    forecasting_exclude_keywords: tuple[str, ...]
    classification_include_keywords: tuple[str, ...]
    classification_exclude_keywords: tuple[str, ...]
    outlier_z_threshold: float


@dataclass(frozen=True)
class ForecastingModelConfig:
    context_length: int
    prediction_horizon: int
    lstm_units: tuple[int, int, int]
    dense_units: int
    activation: str
    dropout_rate: float
    epochs: int
    batch_size: int
    validation_split: float
    learning_rate: float
    initial_train_minutes: int


@dataclass(frozen=True)
class ClassificationModelConfig:
    window_size: int
    traffic_ratio_threshold: float
    start_bit: int
    epsilon: float
    conv1_units: int
    conv2_units: int
    dense_units: int
    dropout_rate: float
    epochs: int
    batch_size: int
    validation_split: float


@dataclass(frozen=True)
class DashboardConfig:
    host: str
    port: int
    max_points: int
    update_interval_ms: int


@dataclass(frozen=True)
class AppConfig:
    run: RunConfig
    paths: Paths
    preprocessing: PreprocessingConfig
    forecasting: ForecastingModelConfig
    classification: ClassificationModelConfig
    dashboard: DashboardConfig


def _as_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing config key: {key}")
    return d[key]


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load JSON config.

    No CLI parsing; by default reads config.json next to this file.
    Optionally supports INDUSTRIAL_NETWORK_V2_CONFIG env var.
    """
    if config_path is None:
        env = os.environ.get("INDUSTRIAL_NETWORK_V2_CONFIG")
        if env:
            config_path = Path(env)
        else:
            config_path = Path(__file__).resolve().parent / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config JSON not found: {config_path}")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.resolve().parent

    run_raw = _require(raw, "run")
    paths_raw = _require(raw, "paths")
    pre_raw = _require(raw, "preprocessing")
    f_raw = _require(raw, "forecasting")
    c_raw = _require(raw, "classification")
    d_raw = _require(raw, "dashboard")

    paths = Paths(
        project_root=project_root,
        raw_data_dir=_as_path(project_root, _require(paths_raw, "raw_data_dir")),
        device_csv_name=str(_require(paths_raw, "device_csv_name")),
        processed_dir=_as_path(project_root, _require(paths_raw, "processed_dir")),
        artifacts_dir=_as_path(project_root, _require(paths_raw, "artifacts_dir")),
    )

    preprocessing = PreprocessingConfig(
        resample_freq=str(_require(pre_raw, "resample_freq")),
        forecasting_include_keywords=tuple(_require(pre_raw, "forecasting_include_keywords")),
        forecasting_exclude_keywords=tuple(_require(pre_raw, "forecasting_exclude_keywords")),
        classification_include_keywords=tuple(_require(pre_raw, "classification_include_keywords")),
        classification_exclude_keywords=tuple(_require(pre_raw, "classification_exclude_keywords")),
        outlier_z_threshold=float(_require(pre_raw, "outlier_z_threshold")),
    )

    forecasting = ForecastingModelConfig(
        context_length=int(_require(f_raw, "context_length")),
        prediction_horizon=int(_require(f_raw, "prediction_horizon")),
        lstm_units=tuple(int(x) for x in _require(f_raw, "lstm_units")),
        dense_units=int(_require(f_raw, "dense_units")),
        activation=str(_require(f_raw, "activation")),
        dropout_rate=float(_require(f_raw, "dropout_rate")),
        epochs=int(_require(f_raw, "epochs")),
        batch_size=int(_require(f_raw, "batch_size")),
        validation_split=float(_require(f_raw, "validation_split")),
        learning_rate=float(_require(f_raw, "learning_rate")),
        initial_train_minutes=int(_require(f_raw, "initial_train_minutes")),
    )

    classification = ClassificationModelConfig(
        window_size=int(_require(c_raw, "window_size")),
        traffic_ratio_threshold=float(_require(c_raw, "traffic_ratio_threshold")),
        start_bit=int(_require(c_raw, "start_bit")),
        epsilon=float(_require(c_raw, "epsilon")),
        conv1_units=int(_require(c_raw, "conv1_units")),
        conv2_units=int(_require(c_raw, "conv2_units")),
        dense_units=int(_require(c_raw, "dense_units")),
        dropout_rate=float(_require(c_raw, "dropout_rate")),
        epochs=int(_require(c_raw, "epochs")),
        batch_size=int(_require(c_raw, "batch_size")),
        validation_split=float(_require(c_raw, "validation_split")),
    )

    dashboard = DashboardConfig(
        host=str(_require(d_raw, "host")),
        port=int(_require(d_raw, "port")),
        max_points=int(_require(d_raw, "max_points")),
        update_interval_ms=int(_require(d_raw, "update_interval_ms")),
    )

    run = RunConfig(mode=str(_require(run_raw, "mode")))

    return AppConfig(
        run=run,
        paths=paths,
        preprocessing=preprocessing,
        forecasting=forecasting,
        classification=classification,
        dashboard=dashboard,
    )
