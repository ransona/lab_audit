#!/usr/bin/env python3
"""Persistent server CPU/GPU telemetry collector and viewer.

The collector is deliberately dependency-light: CPU use comes from /proc/stat
and NVIDIA GPU usage from nvidia-smi. Samples are retained in a local SQLite
database and can be collected by a systemd user service every 20 seconds.
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.dates import AutoDateFormatter, AutoDateLocator
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtWidgets


DEFAULT_DB = Path.home() / ".local" / "share" / "labaudit" / "server_monitor.sqlite"
SAMPLE_SECONDS = 20


class TelemetryDatabase:
    def __init__(self, path: Path = DEFAULT_DB):
        self.path = path

    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS samples (
                recorded_at TEXT PRIMARY KEY,
                cpu_percent REAL NOT NULL
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS gpu_samples (
                recorded_at TEXT NOT NULL,
                gpu_index INTEGER NOT NULL,
                gpu_name TEXT NOT NULL,
                compute_percent REAL NOT NULL,
                memory_used_mb REAL NOT NULL,
                memory_total_mb REAL NOT NULL,
                PRIMARY KEY (recorded_at, gpu_index)
            )"""
        )
        return connection

    def insert(self, recorded_at: datetime, cpu_percent: float, gpus: list[dict]):
        timestamp = recorded_at.isoformat(timespec="seconds")
        with self.connect() as connection:
            connection.execute("INSERT OR REPLACE INTO samples VALUES (?, ?)", (timestamp, cpu_percent))
            connection.executemany(
                """INSERT OR REPLACE INTO gpu_samples
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [(timestamp, gpu["index"], gpu["name"], gpu["compute"], gpu["used"], gpu["total"]) for gpu in gpus],
            )

    def query(self, start: datetime, end: datetime):
        with self.connect() as connection:
            cpu = connection.execute(
                "SELECT recorded_at, cpu_percent FROM samples WHERE recorded_at >= ? AND recorded_at <= ? ORDER BY recorded_at",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
            gpus = connection.execute(
                """SELECT recorded_at, gpu_index, gpu_name, compute_percent,
                          memory_used_mb, memory_total_mb
                   FROM gpu_samples WHERE recorded_at >= ? AND recorded_at <= ?
                   ORDER BY recorded_at, gpu_index""",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        return cpu, gpus


def _cpu_counters() -> tuple[int, int]:
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu "):
            values = [int(value) for value in line.split()[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            return sum(values), idle
    raise RuntimeError("Could not read aggregate CPU counters from /proc/stat")


def cpu_percent(previous: tuple[int, int] | None) -> tuple[float, tuple[int, int]]:
    current = _cpu_counters()
    if previous is None:
        return 0.0, current
    total = current[0] - previous[0]
    idle = current[1] - previous[1]
    return (100.0 * (1 - idle / total) if total else 0.0), current


def gpu_usage() -> list[dict]:
    command = [
        "nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    output = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            output.append({"index": int(fields[0]), "name": fields[1], "compute": float(fields[2]), "used": float(fields[3]), "total": float(fields[4])})
        except ValueError:
            continue
    return output


def collect_forever(database: TelemetryDatabase, interval: float = SAMPLE_SECONDS, once: bool = False):
    """Continuously write samples, accounting for collection work in cadence."""
    # Establish a short CPU-counter interval before the first persisted sample;
    # otherwise every collector restart would create a misleading 0% point.
    previous = _cpu_counters()
    time.sleep(0.2)
    while True:
        started = time.monotonic()
        value, previous = cpu_percent(previous)
        database.insert(datetime.now(), value, gpu_usage())
        if once:
            return
        time.sleep(max(0.0, interval - (time.monotonic() - started)))


class MonitorTab(QtWidgets.QWidget):
    PERIODS = {"Day": timedelta(days=1), "Week": timedelta(days=7), "2 weeks": timedelta(days=14), "Month": timedelta(days=31), "Year": timedelta(days=365)}

    def __init__(self, database: TelemetryDatabase):
        super().__init__()
        self.database = database
        self.active_period = "Day"
        self.end = datetime.now()
        self._build()
        self.refresh()
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(20_000)

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self.period_buttons = QtWidgets.QButtonGroup(self)
        for name in self.PERIODS:
            button = QtWidgets.QPushButton(name); button.setCheckable(True); button.setChecked(name == self.active_period)
            button.clicked.connect(lambda _checked=False, period=name: self._set_period(period))
            self.period_buttons.addButton(button); controls.addWidget(button)
        controls.addSpacing(20)
        previous = QtWidgets.QPushButton("◀ Previous"); previous.clicked.connect(lambda: self._shift(-1))
        following = QtWidgets.QPushButton("Next ▶"); following.clicked.connect(lambda: self._shift(1))
        controls.addWidget(previous); controls.addWidget(following); controls.addSpacing(20)
        self.start_edit = QtWidgets.QDateTimeEdit(); self.end_edit = QtWidgets.QDateTimeEdit()
        for edit in (self.start_edit, self.end_edit): edit.setCalendarPopup(True); edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        apply_range = QtWidgets.QPushButton("Apply range"); apply_range.clicked.connect(self._apply_range)
        controls.addWidget(QtWidgets.QLabel("From")); controls.addWidget(self.start_edit); controls.addWidget(QtWidgets.QLabel("to")); controls.addWidget(self.end_edit); controls.addWidget(apply_range); controls.addStretch()
        layout.addLayout(controls)
        self.status = QtWidgets.QLabel(); layout.addWidget(self.status)
        self.figure = Figure(constrained_layout=True); self.canvas = FigureCanvas(self.figure); layout.addWidget(self.canvas, 1)

    def _set_period(self, name):
        self.active_period = name
        for button in self.period_buttons.buttons(): button.setChecked(button.text() == name)
        self.end = datetime.now(); self.refresh()

    def _shift(self, direction):
        self.end += direction * (self.end - self._range()[0] if self.active_period == "Custom" else self.PERIODS[self.active_period])
        self.refresh()

    def _apply_range(self):
        self.active_period = "Custom"
        self.end = self.end_edit.dateTime().toPyDateTime()
        self._custom_start = self.start_edit.dateTime().toPyDateTime()
        self.refresh()

    def _range(self):
        if self.active_period == "Custom":
            return self._custom_start, self.end
        return self.end - self.PERIODS[self.active_period], self.end

    def refresh(self):
        start, end = self._range()
        self.start_edit.setDateTime(QtCore.QDateTime(start)); self.end_edit.setDateTime(QtCore.QDateTime(end))
        cpu, gpu = self.database.query(start, end)
        self.figure.clear()
        cpu_axis = self.figure.add_subplot(3, 1, 1)
        compute_axis = self.figure.add_subplot(3, 1, 2, sharex=cpu_axis)
        memory_axis = self.figure.add_subplot(3, 1, 3, sharex=cpu_axis)
        if cpu:
            times = [datetime.fromisoformat(row[0]) for row in cpu]
            cpu_axis.plot(times, [row[1] for row in cpu], color="#277da1", label="CPU")
        cpu_axis.set(ylabel="CPU (%)", ylim=(0, 100)); cpu_axis.grid(alpha=.25); cpu_axis.legend(loc="upper right")
        grouped: dict[int, list] = {}
        names = {}
        for row in gpu:
            grouped.setdefault(row[1], []).append(row); names[row[1]] = row[2]
        for index, rows in grouped.items():
            times = [datetime.fromisoformat(row[0]) for row in rows]
            label = f"GPU {index} ({names[index]})"
            compute_axis.plot(times, [row[3] for row in rows], label=label)
            memory_axis.plot(times, [100 * row[4] / row[5] if row[5] else 0 for row in rows], label=label)
        compute_axis.set(ylabel="GPU compute (%)", ylim=(0, 100)); compute_axis.grid(alpha=.25)
        memory_axis.set(ylabel="GPU memory (%)", xlabel="Time", ylim=(0, 100)); memory_axis.grid(alpha=.25)
        if grouped: compute_axis.legend(loc="upper right"); memory_axis.legend(loc="upper right")
        locator = AutoDateLocator(); memory_axis.xaxis.set_major_locator(locator); memory_axis.xaxis.set_major_formatter(AutoDateFormatter(locator))
        self.canvas.draw_idle()
        self.status.setText(f"{len(cpu):,} samples from {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}. Refreshes every 20 seconds.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="run the background collector")
    parser.add_argument("--once", action="store_true", help="write one collection sample and exit")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    database = TelemetryDatabase(args.database)
    if args.collect or args.once:
        collect_forever(database, once=args.once)
        return
    app = QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Lab server resource monitor")
    window.resize(1200, 800)
    window.setCentralWidget(MonitorTab(database))
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
