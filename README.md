# anomaly-detection-in-OT-networks
The repository contains code used for Master Thesis "Hybrid LSTM-CNN Framework for Predictive Anomaly Detection in  Operational Technology Networks"

# 1. Install all requirements from requirements.txt, inside venv:

pip install -r requirements.txt

# 2. Prepare configuration for the program by adjusting config.json
    # 1) Pay attention to the following parameters:

        - `paths.raw_data_dir`: folder containing your device CSV
        - `paths.device_csv_name`: your CSV filename (e.g. `A1-SW-A-245.csv`)

    # ! The raw CSV is expected in the v1 “long format”:
        - columns: `name,timestamp,value`

    # 2) Available modes:

        `run.mode`:
            - `prep`
            - `train-forecasting`
            - `train-classification`
            - `train-all`
            - `online`

    # 3) 

# 3. Run the program:
    A. Data preprocessing:
        - set:

            run.mode = "prep"

        - run:

            ```bash
            python -m anomaly-detection-in-OT-networks.main
            ```

    B. Train the forecasting model:
        - set:

            run.mode = "train-forecasting"

        - run:

            ```bash
            python -m anomaly-detection-in-OT-networks.main
            ```

    C. Train the classification model:
        - set:

            run.mode = "train-classification"

        - run:

            ```bash
            python -m anomaly-detection-in-OT-networks.main
            ```

    D. Run the program, online forecasting with classifiction + dashboard display:
        - set:

            run.mode = "online"

        - run:

            ```bash
            python -m anomaly-detection-in-OT-networks.main
            ```

        - see the live results:

            open "http://127.0.0.1:8050", configured in config.json (`dashboard.host`, dashboard.port`)

        - refresh/step rate:

            `dashboard.update_interval_ms` controls both the dashboard refresh and the online processing step interval.

        - graceful quit:

            press `q` to stop; the program prints a summary (last 24h of processed steps, or what is available) and saves it under `artifacts/online/summary_*.json`.

    # IMPORTANT: You can run everything in one command by using:

        run.mode = "train-all"