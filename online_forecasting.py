from __future__ import annotations

import time
import numpy as np
import pandas as pd

from .config import AppConfig
from .dashboard import DashRealTimePlotter
from .train_forecasting import load_artifacts as load_forecasting
from .train_classification import load_artifacts as load_classification


def _predict_multistep_direct(model, context: np.ndarray, n_features: int, prediction_horizon: int) -> list[np.ndarray]:
    ctx = context.reshape(1, context.shape[0], n_features)
    out = model.predict(ctx, verbose=0)

    preds: list[np.ndarray] = []
    for step in range(prediction_horizon):
        start = step * n_features
        end = (step + 1) * n_features
        preds.append(out[0, start:end])
    return preds


def run_online(cfg: AppConfig) -> None:
    # Load forecasting artifacts
    f_model, f_scalers, variables, context_length, prediction_horizon = load_forecasting(cfg)

    # Load classification artifacts
    c_model, c_prep = load_classification(cfg)

    # Load processed data
    df_forecasting = pd.read_csv(cfg.paths.processed_forecasting_csv, index_col=0, parse_dates=True)
    df_forecasting = df_forecasting[variables].copy()

    df_classification = pd.read_csv(cfg.paths.processed_classification_csv, index_col=0, parse_dates=True)

    # Start dashboard
    plotter = DashRealTimePlotter(max_points=cfg.dashboard.max_points, update_interval_ms=cfg.dashboard.update_interval_ms)
    plotter.set_label_to_name(c_prep.get("label_to_name", {}))
    plotter.start_server(host=cfg.dashboard.host, port=cfg.dashboard.port)

    print(f"Open http://{cfg.dashboard.host}:{cfg.dashboard.port} to view dashboard")
    time.sleep(2)

    # Forecasting pipeline expects differenced+scaled data
    df_diff = df_forecasting.diff().dropna()

    scaled = np.zeros_like(df_diff.values)
    for i, var in enumerate(variables):
        scaler = f_scalers[var]
        scaled[:, i] = scaler.transform(df_diff[[var]]).flatten()

    total_steps = len(scaled) - context_length - prediction_horizon + 1
    plotter.set_total_steps(total_steps)

    # Classification inputs
    c_feature_cols: list[str] = c_prep["feature_columns"]
    c_scaler = c_prep["scaler"]
    c_window = int(c_prep["window_size"])

    # Main loop
    context = scaled[:context_length].copy()

    for t in range(context_length, len(scaled) - prediction_horizon + 1):
        current_step = t - context_length
        ts = df_diff.index[t]

        # Forecast: scaled diffs
        preds_scaled = _predict_multistep_direct(f_model, context, n_features=len(variables), prediction_horizon=prediction_horizon)

        # Back to diff scale
        preds_diff = []
        for step_pred in preds_scaled:
            row = []
            for i, var in enumerate(variables):
                row.append(float(f_scalers[var].inverse_transform([[step_pred[i]]])[0, 0]))
            preds_diff.append(np.array(row))

        # Actual diff (t+1)
        actual_diff_t1 = df_diff.iloc[t + 1][variables].to_numpy(dtype=float)

        # Inverse differencing (use last actual values from raw)
        base_idx = min(t, len(df_forecasting) - 1)
        last_actual = df_forecasting.iloc[base_idx][variables].to_numpy(dtype=float)

        preds_actual = []
        cur = last_actual.copy()
        for step_diff in preds_diff:
            cur = cur + step_diff
            preds_actual.append(cur.copy())

        actual_actual_t1 = last_actual + actual_diff_t1

        # Classification on most recent window (raw, not differenced)
        cls_result = None
        try:
            # Align by timestamp index if possible; fallback to positional
            if ts in df_classification.index:
                end_loc = df_classification.index.get_loc(ts)
            else:
                end_loc = min(t, len(df_classification) - 1)

            start_loc = max(0, end_loc - c_window + 1)
            window_df = df_classification.iloc[start_loc : end_loc + 1]

            window_df = window_df.reindex(columns=c_feature_cols, fill_value=0)
            X = window_df.to_numpy(dtype=float)
            Xs = c_scaler.transform(X)

            # Pad if too short
            if Xs.shape[0] < c_window:
                pad = np.zeros((c_window - Xs.shape[0], Xs.shape[1]), dtype=float)
                Xs = np.vstack([pad, Xs])

            X_in = Xs.reshape(1, c_window, Xs.shape[1])
            probs = c_model.predict(X_in, verbose=0)[0]
            idx = int(np.argmax(probs))
            conf = float(np.max(probs))
            name = c_prep.get("label_to_name", {}).get(idx, f"class_{idx}")

            cls_result = {
                "timestamp": str(ts),
                "classification": name,
                "confidence": conf,
                "class_index": idx,
            }
        except Exception as e:
            print(f"Classification error: {type(e).__name__}: {e}")

        # Push to dashboard (t+1 actual + horizon predictions)
        plotter.add_step(
            timestamp=ts,
            variable_names=variables,
            actual_row=actual_actual_t1,
            predictions_horizon=preds_actual,
            current_step=current_step,
            classification_result=cls_result,
        )

        # Update context with current observed scaled diff
        context = np.vstack([context[1:], scaled[t, :]])

        # Real-time pacing (optional)
        time.sleep(0.0)

    print("Online run completed")
