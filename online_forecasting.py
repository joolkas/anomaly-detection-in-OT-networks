from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import time

import numpy as np
import pandas as pd

from tensorflow import keras

from .config import AppConfig
from .dashboard import DashRealTimePlotter
from .train_forecasting import load_artifacts as load_forecasting
from .train_classification import load_artifacts as load_classification


@dataclass
class _StepRecord:
    timestamp: pd.Timestamp
    actual_t1: np.ndarray
    pred_t1: np.ndarray
    classification: str | None
    confidence: float | None


def _quit_requested() -> bool:
    """Best-effort non-blocking key check for quitting.

    On Windows consoles, this supports immediate single-key quit using msvcrt.
    """
    try:
        import msvcrt  # type: ignore

        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            return ch in ("q", "Q")
        return False
    except Exception:
        return False


def _sleep_with_quit_check(duration_s: float) -> bool:
    """Sleep up to duration_s, returning True if quit was requested."""
    if duration_s <= 0:
        return _quit_requested()

    deadline = time.perf_counter() + duration_s
    while True:
        if _quit_requested():
            return True
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return False
        time.sleep(min(0.2, remaining))


def _compute_summary(
    records: list[_StepRecord],
    variable_names: list[str],
    exceptions: list[dict],
) -> dict:
    if not records:
        return {
            "steps_processed": 0,
            "window": {"hours": 24, "available_steps": 0},
            "forecasting": {},
            "classification": {},
            "errors": {"count": len(exceptions), "last": exceptions[-10:]},
        }

    last_ts = records[-1].timestamp
    cutoff = last_ts - pd.Timedelta(hours=24)
    window = [r for r in records if r.timestamp >= cutoff]
    if not window:
        window = records

    actual = np.stack([r.actual_t1 for r in window], axis=0)
    pred = np.stack([r.pred_t1 for r in window], axis=0)
    err = pred - actual

    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    bias = np.mean(err, axis=0)

    per_var: dict[str, dict] = {}
    for i, v in enumerate(variable_names):
        per_var[v] = {
            "mae": float(mae[i]),
            "rmse": float(rmse[i]),
            "bias": float(bias[i]),
        }

    overall = {
        "mae_mean": float(np.mean(mae)),
        "rmse_mean": float(np.mean(rmse)),
        "bias_mean": float(np.mean(bias)),
    }

    cls_names = [r.classification for r in window if r.classification]
    cls_conf = [r.confidence for r in window if r.confidence is not None]
    cls_counts: dict[str, int] = {}
    for name in cls_names:
        cls_counts[str(name)] = cls_counts.get(str(name), 0) + 1

    storm_like = sum(1 for n in cls_names if "storm" in str(n).lower() or "anomaly" in str(n).lower())
    normal_like = sum(1 for n in cls_names if "normal" in str(n).lower())

    return {
        "steps_processed": len(records),
        "window": {
            "hours": 24,
            "available_steps": len(window),
            "start": str(window[0].timestamp),
            "end": str(window[-1].timestamp),
        },
        "forecasting": {
            "overall": overall,
            "per_variable": per_var,
        },
        "classification": {
            "available_steps": len(cls_names),
            "counts": cls_counts,
            "storm_like": int(storm_like),
            "normal_like": int(normal_like),
            "avg_confidence": float(np.mean(cls_conf)) if cls_conf else None,
        },
        "errors": {
            "count": len(exceptions),
            "last": exceptions[-10:],
        },
    }


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

    # Optional slicing to start from point N (same as preprocessing slice).
    start = int(getattr(cfg.preprocessing, "slice_start_row", 0) or 0)
    end = getattr(cfg.preprocessing, "slice_end_row", None)
    if end is not None:
        end = int(end)
    if start or end is not None:
        df_forecasting = df_forecasting.iloc[start:end].copy()
        df_classification = df_classification.iloc[start:end].copy()
        print(f"[online] Applied data slice rows [{start}:{end}]")

    # Start dashboard
    plotter = DashRealTimePlotter(max_points=cfg.dashboard.max_points, update_interval_ms=cfg.dashboard.update_interval_ms)
    plotter.set_label_to_name(c_prep.get("label_to_name", {}))
    plotter.start_server(host=cfg.dashboard.host, port=cfg.dashboard.port)

    print(f"Open http://{cfg.dashboard.host}:{cfg.dashboard.port} to view dashboard")
    time.sleep(2)

    # Pace the online loop to match the dashboard refresh cadence.
    step_interval_s = max(0.0, float(cfg.dashboard.update_interval_ms) / 1000.0)

    print("Press 'q' to stop and print a run summary.")

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

    records: list[_StepRecord] = []
    exceptions: list[dict] = []

    # Online monitoring + fine-tuning buffers
    fcfg = cfg.forecasting
    err_mae_window = deque(maxlen=int(fcfg.online_error_window))
    err_rmse_window = deque(maxlen=int(fcfg.online_error_window))
    train_X = deque(maxlen=int(fcfg.online_finetune_train_samples))
    train_y = deque(maxlen=int(fcfg.online_finetune_train_samples))
    last_finetune_step = -10**9

    def _maybe_print_status(ts_now, step: int, mae_t1: float, rmse_t1: float) -> None:
        every = int(getattr(fcfg, "online_status_every_steps", 1) or 1)
        if step % every != 0:
            return

        w = int(fcfg.online_error_window)
        if len(err_mae_window) >= w:
            mae_w = float(np.mean(err_mae_window))
            rmse_w = float(np.mean(err_rmse_window))
            print(
                f"[online] step={step} ts={ts_now} "
                f"mae_t+1={mae_t1:.4f} rmse_t+1={rmse_t1:.4f} "
                f"mae{w}={mae_w:.4f} rmse{w}={rmse_w:.4f}"
            )
        else:
            print(
                f"[online] step={step} ts={ts_now} "
                f"mae_t+1={mae_t1:.4f} rmse_t+1={rmse_t1:.4f} "
                f"(warming up {len(err_mae_window)}/{w})"
            )

    quit_requested = False

    try:
        for t in range(context_length, len(scaled) - prediction_horizon + 1):
            step_started = time.perf_counter()

            if _quit_requested():
                quit_requested = True
                break

            current_step = t - context_length

            # The context contains diffs up to index (t-1). So "now" is df_diff.index[t-1],
            # and the forecast horizon corresponds to df_diff.index[t ... t+h-1].
            ts_now = df_diff.index[t - 1]
            horizon_timestamps = list(df_diff.index[t : t + prediction_horizon])
            ts_t1 = horizon_timestamps[0]

            # Forecast: scaled diffs
            preds_scaled = _predict_multistep_direct(
                f_model,
                context,
                n_features=len(variables),
                prediction_horizon=prediction_horizon,
            )

            # Online metrics in scaled-diff space for t+1
            # (unitless; comparable across variables)
            true_t1_scaled = scaled[t, :]
            err_t1_scaled = preds_scaled[0] - true_t1_scaled
            mae_t1_scaled = float(np.mean(np.abs(err_t1_scaled)))
            rmse_t1_scaled = float(np.sqrt(np.mean(err_t1_scaled**2)))
            err_mae_window.append(mae_t1_scaled)
            err_rmse_window.append(rmse_t1_scaled)

            _maybe_print_status(ts_now, current_step, mae_t1_scaled, rmse_t1_scaled)

            # Training sample for potential fine-tune (multistep)
            y_true_scaled = scaled[t : t + prediction_horizon, :].reshape(-1)
            train_X.append(context.copy())
            train_y.append(y_true_scaled.copy())

            # Back to diff scale
            preds_diff = []
            for step_pred in preds_scaled:
                row = []
                for i, var in enumerate(variables):
                    row.append(float(f_scalers[var].inverse_transform([[step_pred[i]]])[0, 0]))
                preds_diff.append(np.array(row))

            # Base actual at "now" timestamp
            last_actual = df_forecasting.loc[ts_now, variables].to_numpy(dtype=float)

            preds_actual = []
            cur = last_actual.copy()
            for step_diff in preds_diff:
                cur = cur + step_diff
                preds_actual.append(cur.copy())

            # Metrics + plotting alignment (forecast-style):
            # prediction made at ts_now for ts_t1 is plotted at ts_t1 and scored against actual(ts_t1).
            actual_t1 = df_forecasting.loc[ts_t1, variables].to_numpy(dtype=float)

            # Trigger small fine-tune if rolling error is high
            if (
                fcfg.online_finetune_enabled
                and len(err_mae_window) >= int(fcfg.online_error_window)
                and (current_step - last_finetune_step) >= int(fcfg.online_finetune_cooldown_steps)
            ):
                rolling_mae = float(np.mean(err_mae_window))
                rolling_rmse = float(np.mean(err_rmse_window))

                if rolling_mae > float(fcfg.online_error_mae_threshold):
                    try:
                        X_ft = np.asarray(train_X)
                        y_ft = np.asarray(train_y)
                        if X_ft.size and y_ft.size:
                            ft_started = time.perf_counter()
                            f_model.compile(
                                optimizer=keras.optimizers.Adam(learning_rate=float(fcfg.online_finetune_learning_rate)),
                                loss="mse",
                            )
                            print(
                                f"[online-finetune] step={current_step} ts={ts_now} "
                                f"mae{int(fcfg.online_error_window)}={rolling_mae:.4f} (thr {float(fcfg.online_error_mae_threshold):.4f}) "
                                f"samples={len(train_X)} epochs={int(fcfg.online_finetune_epochs)} lr={float(fcfg.online_finetune_learning_rate):.2e}"
                            )
                            history = f_model.fit(
                                X_ft,
                                y_ft,
                                epochs=int(fcfg.online_finetune_epochs),
                                batch_size=int(fcfg.online_finetune_batch_size),
                                verbose=0,
                                shuffle=False,
                            )
                            ft_s = time.perf_counter() - ft_started
                            try:
                                loss = float(history.history.get("loss", [None])[-1])
                            except Exception:
                                loss = float("nan")
                            print(f"[online-finetune] done in {ft_s:.2f}s loss={loss:.6f}")
                            last_finetune_step = current_step
                    except Exception as e:
                        msg = f"{type(e).__name__}: {e}"
                        print(f"Online fine-tune error: {msg}")
                        exceptions.append({"where": "online_finetune", "timestamp": str(ts_now), "message": msg})

            # Classification on most recent window (raw, not differenced)
            cls_result = None
            try:
                # Align by timestamp index if possible; fallback to positional
                if ts_now in df_classification.index:
                    end_loc = df_classification.index.get_loc(ts_now)
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
                    "timestamp": str(ts_now),
                    "classification": name,
                    "confidence": conf,
                    "class_index": idx,
                }
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                print(f"Classification error: {msg}")
                exceptions.append({"where": "classification", "timestamp": str(ts_now), "message": msg})

        # Push to dashboard:
        # - actual at ts_now (blue)
        # - pred(t+1) at ts_t1 (green)
        # - full horizon pred(t+1..t+6) displayed on horizon_timestamps (red)
            plotter.add_step(
                timestamp=ts_now,
                variable_names=variables,
                actual_row=last_actual,
                pred1_timestamp=ts_t1,
                pred1_row=preds_actual[0],
                horizon_timestamps=horizon_timestamps,
                predictions_horizon=preds_actual,
                current_step=current_step,
                classification_result=cls_result,
            )

            records.append(
                _StepRecord(
                    timestamp=pd.Timestamp(ts_t1),
                    actual_t1=np.array(actual_t1, dtype=float),
                    pred_t1=np.array(preds_actual[0], dtype=float),
                    classification=(cls_result or {}).get("classification") if cls_result else None,
                    confidence=(cls_result or {}).get("confidence") if cls_result else None,
                )
            )

        # Update context with current observed scaled diff
            context = np.vstack([context[1:], scaled[t, :]])

        # Real-time pacing (optional): run one step per interval.
            if step_interval_s > 0:
                elapsed = time.perf_counter() - step_started
                if _sleep_with_quit_check(max(0.0, step_interval_s - elapsed)):
                    quit_requested = True
                    break

    except KeyboardInterrupt:
        quit_requested = True

    if quit_requested:
        print("Quit requested. Generating summary...")

    summary = _compute_summary(records=records, variable_names=variables, exceptions=exceptions)

    summary["meta"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "step_interval_s": float(step_interval_s),
        "dashboard_url": f"http://{cfg.dashboard.host}:{cfg.dashboard.port}",
    }

    out_dir = cfg.paths.artifacts_dir / "online"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = out_dir / f"summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Summary saved to: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("Online run completed")
