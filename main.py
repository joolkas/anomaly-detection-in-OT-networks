from __future__ import annotations

from .config import load_config
from .data_pipeline import prepare_and_save
from .online_forecasting import run_online
from .train_forecasting import train_and_save as train_forecasting
from .train_classification import train_and_save as train_classification
from .utils import safe_print_exception


def main() -> int:
    cfg = load_config()

    try:
        mode = cfg.run.mode

        if mode == "prep":
            prepare_and_save(cfg)
        elif mode == "train-forecasting":
            train_forecasting(cfg)
        elif mode == "train-classification":
            train_classification(cfg)
        elif mode == "train-all":
            prepare_and_save(cfg)
            train_forecasting(cfg)
            train_classification(cfg)
        elif mode == "online":
            run_online(cfg)
        else:
            raise ValueError(
                f"Unknown run.mode '{mode}'. Allowed: prep, train-forecasting, train-classification, train-all, online"
            )

        return 0

    except Exception as e:
        safe_print_exception("Fatal error", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())