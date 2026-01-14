from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tensorflow import keras

from .config import AppConfig


def _build_multistep_lstm(cfg: AppConfig, num_features: int) -> keras.Model:
    fcfg = cfg.forecasting
    output_size = fcfg.prediction_horizon * num_features

    u1, u2, u3 = fcfg.lstm_units

    model = keras.models.Sequential(
        [
            keras.layers.LSTM(u1, return_sequences=True, input_shape=(fcfg.context_length, num_features)),
            keras.layers.Dropout(fcfg.dropout_rate * 0.5),
            keras.layers.LSTM(u2, return_sequences=True),
            keras.layers.Dropout(fcfg.dropout_rate * 0.5),
            keras.layers.LSTM(u3, return_sequences=False),
            keras.layers.Dropout(fcfg.dropout_rate * 0.7),
            keras.layers.Dense(fcfg.dense_units, activation=fcfg.activation),
            keras.layers.Dropout(fcfg.dropout_rate),
            keras.layers.Dense(fcfg.dense_units // 2, activation=fcfg.activation),
            keras.layers.Dropout(fcfg.dropout_rate * 0.5),
            keras.layers.Dense(output_size, activation="linear"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=fcfg.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _make_multistep_training_arrays(
    df_scaled: pd.DataFrame,
    context_length: int,
    prediction_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(context_length, len(df_scaled) - prediction_horizon + 1):
        X.append(df_scaled.iloc[i - context_length : i].values)
        future = []
        for step in range(prediction_horizon):
            future.extend(df_scaled.iloc[i + step].values)
        y.append(future)

    return np.array(X), np.array(y)


def train_and_save(cfg: AppConfig) -> None:
    paths = cfg.paths
    fcfg = cfg.forecasting

    if not paths.processed_forecasting_csv.exists():
        raise FileNotFoundError(
            f"Processed forecasting CSV not found: {paths.processed_forecasting_csv}. Run preprocessing first."
        )

    df = pd.read_csv(paths.processed_forecasting_csv, index_col=0, parse_dates=True)
    df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    # Differencing (same as v1)
    df_diff = df.diff().dropna()

    # Split into initial training and online portion
    initial_n = min(fcfg.initial_train_minutes, len(df_diff))
    df_initial = df_diff.iloc[:initial_n].copy()

    variables = df_initial.columns.tolist()

    # Per-variable scalers
    scalers: dict[str, StandardScaler] = {}
    scaled = np.zeros_like(df_initial.values)

    for i, var in enumerate(variables):
        scaler = StandardScaler()
        scaled[:, i] = scaler.fit_transform(df_initial[[var]]).flatten()
        scalers[var] = scaler

    df_scaled = pd.DataFrame(scaled, columns=variables, index=df_initial.index)

    X_train, y_train = _make_multistep_training_arrays(
        df_scaled,
        context_length=fcfg.context_length,
        prediction_horizon=fcfg.prediction_horizon,
    )

    model = _build_multistep_lstm(cfg, num_features=len(variables))

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7),
    ]

    print(f"Training forecasting model: X={X_train.shape}, y={y_train.shape}")

    history = model.fit(
        X_train,
        y_train,
        epochs=fcfg.epochs,
        batch_size=fcfg.batch_size,
        validation_split=fcfg.validation_split,
        verbose=1,
        callbacks=callbacks,
    )

    out_dir = paths.forecasting_artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "forecasting_model.h5"
    model.save(model_path)

    with open(out_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    (out_dir / "variables.txt").write_text("\n".join(variables), encoding="utf-8")

    np.save(out_dir / "context_length.npy", fcfg.context_length)
    np.save(out_dir / "prediction_horizon.npy", fcfg.prediction_horizon)

    print(f"Saved forecasting model: {model_path}")
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")


def load_artifacts(cfg: AppConfig):
    paths = cfg.paths
    out_dir = paths.forecasting_artifacts_dir

    model = keras.models.load_model(out_dir / "forecasting_model.h5", compile = False)
    with open(out_dir / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    variables = [line.strip() for line in (out_dir / "variables.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    context_length = int(np.load(out_dir / "context_length.npy").item())
    prediction_horizon = int(np.load(out_dir / "prediction_horizon.npy").item())

    return model, scalers, variables, context_length, prediction_horizon
