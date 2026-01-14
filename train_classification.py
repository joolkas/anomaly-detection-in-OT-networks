from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from tensorflow import keras

from .config import AppConfig
from .utils import create_windows


def _extract_port_id(col: str) -> str | None:
    # Try to extract something like 1/0/1
    m = re.search(r"(\d+/\d+/\d+)", col)
    if m:
        return m.group(1)
    # fallback: last integer sequence
    nums = re.findall(r"\d+", col)
    if nums:
        return nums[-1]
    return None


def _find_bits_pairs(columns: list[str]) -> dict[str, tuple[str, str]]:
    """Return mapping port_id -> (sent_col, recv_col)"""
    sent_cols = [c for c in columns if "bits" in c.lower() and "sent" in c.lower()]
    recv_cols = [c for c in columns if "bits" in c.lower() and ("recv" in c.lower() or "received" in c.lower())]

    pairs: dict[str, tuple[str, str]] = {}
    for s in sent_cols:
        pid = _extract_port_id(s)
        if not pid:
            continue
        # find best matching recv with same pid
        match = None
        for r in recv_cols:
            if pid in r:
                match = r
                break
        if match is None:
            continue
        pairs[pid] = (s, match)

    return pairs


def _encode_labels_bitmask(df: pd.DataFrame, pairs: dict[str, tuple[str, str]], cfg: AppConfig) -> tuple[pd.Series, list[str]]:
    ccfg = cfg.classification

    ports = sorted(pairs.keys())
    label = pd.Series(0, index=df.index, dtype=object)

    for i, port in enumerate(ports):
        sent_col, recv_col = pairs[port]
        sent = pd.to_numeric(df[sent_col], errors="coerce").fillna(0).astype(float)
        recv = pd.to_numeric(df[recv_col], errors="coerce").fillna(0).astype(float)
        ratio = recv / (sent + float(ccfg.epsilon))

        bit_pos = ccfg.start_bit + i
        label = label + (ratio > ccfg.traffic_ratio_threshold).astype(object) * (1 << bit_pos)

    return label, ports


def _build_cnn(window_size: int, num_features: int, num_classes: int, cfg: AppConfig) -> keras.Model:
    ccfg = cfg.classification

    model = keras.models.Sequential(
        [
            keras.layers.Conv1D(ccfg.conv1_units, kernel_size=3, activation="relu", input_shape=(window_size, num_features)),
            keras.layers.BatchNormalization(),
            keras.layers.Conv1D(ccfg.conv2_units, kernel_size=3, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dropout(ccfg.dropout_rate),
            keras.layers.Dense(ccfg.dense_units, activation="relu"),
            keras.layers.Dropout(ccfg.dropout_rate),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_save(cfg: AppConfig) -> None:
    paths = cfg.paths
    ccfg = cfg.classification

    if not paths.processed_classification_csv.exists():
        raise FileNotFoundError(
            f"Processed classification CSV not found: {paths.processed_classification_csv}. Run preprocessing first."
        )

    df = pd.read_csv(paths.processed_classification_csv, index_col=0, parse_dates=True)
    df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    pairs = _find_bits_pairs(list(df.columns))
    if not pairs:
        raise RuntimeError(
            "No Bits Sent/Received column pairs found for labeling. "
            "Ensure classification features include bits sent + bits received columns."
        )

    labels_bitmask, ports = _encode_labels_bitmask(df, pairs, cfg)

    # Build label set, map to indices
    unique_labels = sorted(pd.unique(labels_bitmask))
    label_to_index = {int(lbl): i for i, lbl in enumerate(unique_labels)}
    y = labels_bitmask.map(lambda v: label_to_index[int(v)]).to_numpy(dtype=int)

    # Human-readable class names (storm if any bit set)
    index_to_label = {i: int(lbl) for lbl, i in label_to_index.items()}
    label_to_name: dict[int, str] = {}
    for idx, raw_label in index_to_label.items():
        if raw_label == 0:
            label_to_name[idx] = "normal"
        else:
            label_to_name[idx] = "storm"  # simple; can be expanded per-port later

    feature_cols = list(df.columns)
    X_raw = df[feature_cols].to_numpy(dtype=float)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_seq, y_seq = create_windows(X_scaled, y, window_size=ccfg.window_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_seq,
        y_seq,
        test_size=ccfg.validation_split,
        random_state=42,
        stratify=y_seq if len(np.unique(y_seq)) > 1 else None,
    )

    model = _build_cnn(ccfg.window_size, num_features=X_seq.shape[2], num_classes=len(unique_labels), cfg=cfg)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ]

    print(f"Training classification model: X={X_train.shape}, classes={len(unique_labels)}")

    model.fit(
        X_train,
        y_train,
        epochs=ccfg.epochs,
        batch_size=ccfg.batch_size,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=callbacks,
    )

    out_dir = paths.classification_artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "classification_model.h5"
    model.save(model_path)

    preprocessing = {
        "scaler": scaler,
        "feature_columns": feature_cols,
        "window_size": ccfg.window_size,
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "label_to_name": label_to_name,
        "ports": ports,
    }

    with open(out_dir / "preprocessing.pkl", "wb") as f:
        pickle.dump(preprocessing, f)

    print(f"Saved classification model: {model_path}")
    print(f"Saved preprocessing objects: {out_dir / 'preprocessing.pkl'}")


def load_artifacts(cfg: AppConfig):
    paths = cfg.paths
    out_dir = paths.classification_artifacts_dir

    model = keras.models.load_model(out_dir / "classification_model.h5", compile = False)
    with open(out_dir / "preprocessing.pkl", "rb") as f:
        preprocessing = pickle.load(f)

    return model, preprocessing
