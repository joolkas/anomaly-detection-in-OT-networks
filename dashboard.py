from __future__ import annotations

from collections import deque
import threading

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.subplots as sp


class DashRealTimePlotter:
    def __init__(self, max_points: int = 60, update_interval_ms: int = 60_000):
        self.app = dash.Dash(__name__)
        self.max_points = max_points
        self.update_interval_ms = update_interval_ms

        self._lock = threading.Lock()

        self.timestamps = deque(maxlen=max_points)
        self.actual_values: dict[str, deque] = {}

        # Stored one-step-ahead predictions (used for metrics, shown historically)
        self.pred1_timestamps = deque(maxlen=max_points)
        self.pred1_values: dict[str, deque] = {}

        # Current horizon predictions (shown as a future projection; not stored historically)
        self.horizon_timestamps: list = []
        self.horizon_values: dict[str, list[float]] = {}

        self.variable_names: list[str] = []

        self.classification_results = deque(maxlen=50)
        self.label_to_name_dict: dict[int, str] = {}

        self.current_step = 0
        self.total_steps = 0

        self._setup_layout()
        self._setup_callbacks()

    def start_server(self, host: str = "127.0.0.1", port: int = 8050) -> None:
        def run():
            self.app.run(host=host, port=port, debug=False)

        t = threading.Thread(target=run, daemon=True)
        t.start()

    def set_total_steps(self, total_steps: int) -> None:
        with self._lock:
            self.total_steps = int(total_steps)

    def set_label_to_name(self, label_to_name: dict[int, str]) -> None:
        with self._lock:
            self.label_to_name_dict = dict(label_to_name)

    def add_classification_result(self, result: dict) -> None:
        with self._lock:
            self.classification_results.append(result)

    def add_step(
        self,
        timestamp,
        variable_names: list[str],
        actual_row,
        pred1_timestamp,
        pred1_row,
        horizon_timestamps: list,
        predictions_horizon,
        current_step: int,
        classification_result: dict | None,
    ) -> None:
        with self._lock:
            self.current_step = current_step
            self.variable_names = variable_names

            self.timestamps.append(timestamp)
            self.pred1_timestamps.append(pred1_timestamp)

            self.horizon_timestamps = list(horizon_timestamps)

            # Init deques
            for v in variable_names:
                self.actual_values.setdefault(v, deque(maxlen=self.max_points))
                self.pred1_values.setdefault(v, deque(maxlen=self.max_points))
                self.horizon_values.setdefault(v, [])

            for i, v in enumerate(variable_names):
                self.actual_values[v].append(float(actual_row[i]))
                self.pred1_values[v].append(float(pred1_row[i]))

            # Store current horizon prediction for display.
            for i, v in enumerate(variable_names):
                self.horizon_values[v] = [float(step[i]) for step in predictions_horizon]

            if classification_result is not None:
                self.classification_results.append(classification_result)

    def _setup_layout(self):
        self.app.layout = html.Div(
            [
                html.Div(
                    [
                        html.H2("Classification", style={"textAlign": "center"}),
                        html.Div(id="classification-results", style={"textAlign": "center"}),
                    ]
                ),
                html.H1("OT Network Forecasting Dashboard", style={"textAlign": "center"}),
                html.Div(id="status-info", style={"textAlign": "center"}),
                html.Div(id="graphs"),
                dcc.Interval(id="interval", interval=self.update_interval_ms, n_intervals=0),
            ]
        )

    def _setup_callbacks(self):
        @self.app.callback(
            [Output("classification-results", "children"), Output("status-info", "children"), Output("graphs", "children")],
            [Input("interval", "n_intervals")],
        )
        def _update(_):
            return self._render_classification(), self._render_status(), self._render_graphs()

    def _render_status(self):
        with self._lock:
            return f"Step: {self.current_step}/{self.total_steps}  |  Variables: {len(self.variable_names)}"

    def _render_classification(self):
        with self._lock:
            if self.label_to_name_dict:
                mapping = " | ".join([f"{k}:{v}" for k, v in sorted(self.label_to_name_dict.items())])
            else:
                mapping = "(no label mapping loaded)"

            last = self.classification_results[-1] if self.classification_results else None

        if last is None:
            return html.Div([html.Div(f"Labels: {mapping}"), html.Div("No classification yet")])

        cls = last.get("classification", "unknown")
        conf = last.get("confidence", 0.0)
        ts = last.get("timestamp", "")
        color = "#e74c3c" if "storm" in str(cls).lower() else "#27ae60"

        return html.Div(
            [
                html.Div(f"Labels: {mapping}"),
                html.Div([html.Span(f"{ts}  "), html.Span(f"{cls}", style={"color": color, "fontWeight": "bold"}), html.Span(f"  (conf {conf:.2f})")]),
            ]
        )

    def _render_graphs(self):
        with self._lock:
            variable_names = list(self.variable_names)
            timestamps = list(self.timestamps)
            actual_values = {k: list(v) for k, v in self.actual_values.items()}
            pred1_timestamps = list(self.pred1_timestamps)
            pred1_values = {k: list(v) for k, v in self.pred1_values.items()}
            horizon_timestamps = list(self.horizon_timestamps)
            horizon_values = {k: list(v) for k, v in self.horizon_values.items()}

        if not variable_names or not timestamps:
            return html.Div("Waiting for data...")

        cols = 2
        n = len(variable_names)
        rows = (n + cols - 1) // cols
        fig = sp.make_subplots(rows=rows, cols=cols, subplot_titles=variable_names)

        ts = timestamps
        for i, v in enumerate(variable_names):
            row = i // cols + 1
            col = i % cols + 1
            fig.add_trace(
                go.Scatter(x=ts, y=list(actual_values.get(v, [])), mode="lines+markers", name="actual", line=dict(color="blue"), showlegend=(i == 0)),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(x=pred1_timestamps, y=list(pred1_values.get(v, [])), mode="lines+markers", name="pred(t+1)", line=dict(color="#2ecc71"), showlegend=(i == 0)),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(x=horizon_timestamps, y=list(horizon_values.get(v, [])), mode="lines+markers", name="pred(t+1..t+6)", line=dict(color="red"), showlegend=(i == 0)),
                row=row,
                col=col,
            )

        fig.update_layout(height=300 * rows, margin=dict(l=30, r=30, t=60, b=30))
        return dcc.Graph(figure=fig)
