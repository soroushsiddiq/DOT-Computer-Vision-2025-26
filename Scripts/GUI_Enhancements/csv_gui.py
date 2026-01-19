import sys
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox, QMessageBox
)

import pyqtgraph as pg


POSES = ["Pose1", "Pose2", "True"]

# CSV column mapping
AXES = {
    "Tx": ("Pose1_Tx", "Pose2_Tx", "True_Tx"),
    "Ty": ("Pose1_Ty", "Pose2_Ty", "True_Ty"),
    "Tz": ("Pose1_Tz", "Pose2_Tz", "True_Tz"),
    "Rx": ("Pose1_Rx", "Pose2_Rx", "True_Rx"),
    "Ry": ("Pose1_Ry", "Pose2_Ry", "True_Ry"),
    "Rz": ("Pose1_Rz", "Pose2_Rz", "True_Rz"),
}

# Always-visible main graphs
MAIN_SIGNALS = ["Tx", "Ty", "Rz"]

# Dropdown options for extras (we rebuild plots dynamically)
EXTRA_OPTIONS = [
    ("None", []),
    ("Tz", ["Tz"]),
    ("Rx", ["Rx"]),
    ("Ry", ["Ry"]),
    ("Tz + Rx", ["Tz", "Rx"]),
    ("Tz + Ry", ["Tz", "Ry"]),
    ("Rx + Ry", ["Rx", "Ry"]),
    ("Tz + Rx + Ry", ["Tz", "Rx", "Ry"]),
]


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


class CsvPosePlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LAR Pose CSV Viewer")

        self.df = None
        self.i = 0
        self.playing = False

        # Playback speed control
        self.base_fps = 10
        self.playback_fps = self.base_fps

        # Plot only the last N seconds (scrolling window). Set None for full history.
        self.window_seconds = 8.0

        # "Looking Left/Right" threshold using chosen True_Tx (mm)
        self.look_threshold_mm = 5.0

        # Extras selection state
        self.extra_signals = []  # e.g., ["Tz","Rx"]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        # ---------- UI ----------
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        # Top controls row
        top = QHBoxLayout()
        self.btn_open = QPushButton("Open CSV")
        self.btn_open.clicked.connect(self.open_csv)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset)

        self.speed = QComboBox()
        self.speed.addItems(["0.5x", "1x", "2x", "4x"])
        self.speed.setCurrentText("1x")
        self.speed.currentTextChanged.connect(self.set_speed)

        self.extras = QComboBox()
        self.extras.addItems([name for name, _ in EXTRA_OPTIONS])
        self.extras.setCurrentText("None")
        self.extras.currentTextChanged.connect(self.set_extras)

        self.lbl_info = QLabel("No file loaded.")
        self.lbl_info.setMinimumWidth(520)

        self.lbl_key = QLabel("Blue=Pose1 | Orange=Pose2 | Green(thick)=True/Chosen")
        self.lbl_key.setStyleSheet("font-weight: 600;")

        top.addWidget(self.btn_open)
        top.addWidget(self.btn_play)
        top.addWidget(self.btn_reset)
        top.addSpacing(8)
        top.addWidget(QLabel("Speed:"))
        top.addWidget(self.speed)
        top.addSpacing(12)
        top.addWidget(QLabel("Extras:"))
        top.addWidget(self.extras)
        top.addSpacing(12)
        top.addWidget(self.lbl_info)
        top.addSpacing(18)
        top.addWidget(self.lbl_key)
        top.addStretch(1)

        main.addLayout(top)

        # Scrubber row
        scrub = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.slider_changed)

        self.lbl_frame = QLabel("Frame: - / -")
        self.lbl_time = QLabel("t = - s")
        self.lbl_lar_dir = QLabel("LAR: -")
        self.lbl_lar_dir.setStyleSheet("font-weight: 700;")

        scrub.addWidget(QLabel("Scrub:"))
        scrub.addWidget(self.slider, 1)
        scrub.addWidget(self.lbl_frame)
        scrub.addWidget(self.lbl_time)
        scrub.addSpacing(12)
        scrub.addWidget(self.lbl_lar_dir)
        main.addLayout(scrub)

        # Pose panels + chosen info
        pose_row = QHBoxLayout()
        self.pose_labels = {}
        for p in POSES:
            box = QVBoxLayout()
            title = QLabel(f"<b>{p}</b>")
            box.addWidget(title)

            lbl = QLabel("Tx: -\nTy: -\nTz: -\nRx: -\nRy: -\nRz: -")
            lbl.setStyleSheet("font-family: Consolas, monospace;")
            box.addWidget(lbl)

            self.pose_labels[p] = lbl
            pose_row.addLayout(box)

        self.lbl_choice = QLabel(
            "<b>Chosen:</b> -<br>"
            "<span style='font-size:11px;'>"
            "d1 = ||[Pose1_Tx,Pose1_Ty,Pose1_Rz] - [True_Tx,True_Ty,True_Rz]||<br>"
            "d2 = ||[Pose2_Tx,Pose2_Ty,Pose2_Rz] - [True_Tx,True_Ty,True_Rz]||"
            "</span>"
        )
        pose_row.addWidget(self.lbl_choice)
        main.addLayout(pose_row)

        # Plots
        self.plot_widget = pg.GraphicsLayoutWidget()
        main.addWidget(self.plot_widget, 1)

        self.plots = {}     # plots[sig] = PlotItem
        self.curves = {}    # curves[sig][pose] = PlotDataItem
        self.cursors = {}   # cursors[sig] = InfiniteLine

        self._setup_plots()  # initial with MAIN_SIGNALS only

    # ---------- Plot setup / rebuild ----------
    def _setup_plots(self):
        """Build plots based on MAIN_SIGNALS + current extras."""
        self.plot_widget.clear()
        self.plots.clear()
        self.curves.clear()
        self.cursors.clear()

        # Pen styles
        pose_styles = {
            "Pose1": pg.mkPen(color=(80, 170, 255), width=2),   # blue
            "Pose2": pg.mkPen(color=(255, 170, 80), width=2),   # orange
            "True":  pg.mkPen(color=(120, 255, 120), width=3),  # green thick
        }

        # Axis labels (best-effort)
        # Note: In your current pipeline, Rx/Ry/Rz are normal-vector components => unitless.
        y_labels = {
            "Tx": "Tx (mm)",
            "Ty": "Ty (mm)",
            "Tz": "Tz (mm)",
            "Rx": "Rx (unitless / normal comp)",
            "Ry": "Ry (unitless / normal comp)",
            "Rz": "Rz (unitless / normal comp)",
        }

        signals = MAIN_SIGNALS + list(self.extra_signals)

        for r, sig in enumerate(signals):
            p = self.plot_widget.addPlot(row=r, col=0)
            p.showGrid(x=True, y=True)
            p.setLabel("left", y_labels.get(sig, sig))
            p.setLabel("bottom", "Frame")
            self.plots[sig] = p

            self.curves[sig] = {}
            for pose_name in POSES:
                legend_name = {
                    "Pose1": "Pose1 (Blue)",
                    "Pose2": "Pose2 (Orange)",
                    "True":  "True/Chosen (Green)"
                }[pose_name]
                curve = p.plot([], [], pen=pose_styles[pose_name], name=legend_name)
                self.curves[sig][pose_name] = curve

            # Cursor line (current frame)
            cursor_pen = pg.mkPen(color=(180, 180, 180), width=2, style=Qt.DashLine)
            vline = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=cursor_pen)
            p.addItem(vline)
            self.cursors[sig] = vline

        # Legend on the first (top) plot only
        if signals:
            self.plots[signals[0]].addLegend()

        # If data is loaded, redraw to current frame
        if self.df is not None:
            self._update_frame_ui()

    def _clear_plots(self):
        signals = MAIN_SIGNALS + list(self.extra_signals)
        for sig in signals:
            for pose_name in POSES:
                if sig in self.curves and pose_name in self.curves[sig]:
                    self.curves[sig][pose_name].setData([], [])
            if sig in self.cursors:
                self.cursors[sig].setPos(0)

    # ---------- CSV loading ----------
    def open_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Jan12GNCTest.csv", "", "CSV Files (*.csv)")
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "CSV Error", f"Failed to read CSV:\n{e}")
            return

        required = ["Time [s]"]
        for sig in AXES:
            required.extend(list(AXES[sig]))

        missing = [c for c in required if c not in df.columns]
        if missing:
            QMessageBox.critical(self, "CSV Error", f"CSV missing columns:\n{missing}")
            return

        for c in required:
            df[c] = df[c].map(safe_float)

        self.df = df
        self.i = 0

        self.slider.setEnabled(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(df) - 1)
        self.slider.setValue(0)

        # Info label: frames + duration + CSV fps + playback fps
        N = len(df)
        t0 = df["Time [s]"].iloc[0]
        t1 = df["Time [s]"].iloc[-1]
        duration = float(t1 - t0) if np.isfinite(t1 - t0) else float("nan")
        fps_est = (N - 1) / duration if (np.isfinite(duration) and duration > 0) else float("nan")
        self.lbl_info.setText(
            f"Loaded: {path} | Frames: {N} | Duration: {duration:.2f}s | CSV FPS≈{fps_est:.2f} | Playback FPS={self.playback_fps}"
        )

        self._clear_plots()
        self._update_frame_ui()

    # ---------- Controls ----------
    def toggle_play(self):
        if self.df is None:
            return
        self.playing = not self.playing
        self.btn_play.setText("Pause" if self.playing else "Play")
        if self.playing:
            interval_ms = int(1000 / max(self.playback_fps, 1))
            self.timer.start(max(interval_ms, 1))
        else:
            self.timer.stop()

    def set_speed(self, text):
        mult = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}[text]
        self.playback_fps = max(1, int(self.base_fps * mult))

        # Restart timer if playing
        if self.playing:
            self.timer.stop()
            interval_ms = int(1000 / self.playback_fps)
            self.timer.start(max(interval_ms, 1))

        # Update info label if df loaded
        if self.df is not None:
            df = self.df
            N = len(df)
            t0 = df["Time [s]"].iloc[0]
            t1 = df["Time [s]"].iloc[-1]
            duration = float(t1 - t0) if np.isfinite(t1 - t0) else float("nan")
            fps_est = (N - 1) / duration if (np.isfinite(duration) and duration > 0) else float("nan")
            self.lbl_info.setText(
                f"Loaded | Frames: {N} | Duration: {duration:.2f}s | CSV FPS≈{fps_est:.2f} | Playback FPS={self.playback_fps}"
            )

    def set_extras(self, text):
        # Map dropdown selection -> signals list
        for name, sigs in EXTRA_OPTIONS:
            if name == text:
                self.extra_signals = sigs
                break

        # Rebuild plots so extras appear/disappear cleanly
        self._setup_plots()

    def reset(self):
        if self.df is None:
            return
        self.i = 0
        self._clear_plots()

        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        self._update_frame_ui()

    def slider_changed(self, v):
        if self.df is None:
            return
        self.i = int(v)
        self._update_frame_ui()

    # ---------- Playback tick ----------
    def _tick(self):
        if self.df is None:
            return

        self.i += 1
        if self.i >= len(self.df):
            self.i = len(self.df) - 1
            self.toggle_play()
            return

        self.slider.blockSignals(True)
        self.slider.setValue(self.i)
        self.slider.blockSignals(False)

        self._update_frame_ui()

    # ---------- Plot updates ----------
    def _update_plots_incremental(self):
        if self.df is None:
            return

        t_all = self.df["Time [s]"].to_numpy()
        t_now = t_all[self.i]

        # Windowing based on time (still nice), but x-axis is frame index
        if self.window_seconds is None:
            start = 0
        else:
            t_min = t_now - self.window_seconds
            start = int(np.searchsorted(t_all, t_min, side="left"))

        x = np.arange(start, self.i + 1)

        signals = MAIN_SIGNALS + list(self.extra_signals)
        for sig in signals:
            c1, c2, ct = AXES[sig]
            y1 = self.df[c1].to_numpy()[start:self.i + 1]
            y2 = self.df[c2].to_numpy()[start:self.i + 1]
            yt = self.df[ct].to_numpy()[start:self.i + 1]

            self.curves[sig]["Pose1"].setData(x, y1)
            self.curves[sig]["Pose2"].setData(x, y2)
            self.curves[sig]["True"].setData(x, yt)

            # Move cursor
            self.cursors[sig].setPos(self.i)

        if self.window_seconds is not None:
            for sig in signals:
                self.plots[sig].setXRange(start, self.i, padding=0)

    # ---------- Frame UI updates ----------
    def _update_frame_ui(self):
        if self.df is None:
            return

        row = self.df.iloc[self.i]
        n = len(self.df)
        t = row["Time [s]"]

        self.lbl_frame.setText(f"Frame: {self.i} / {n - 1}")
        self.lbl_time.setText(f"t = {t:.3f} s")

        # LAR direction line based on chosen/True Tx
        # Matches main.py convention: Tx >= 0 => RIGHT, Tx < 0 => LEFT
        # We add a small deadband around 0 to avoid rapid flicker.
        tx_true = float(row["True_Tx"])
        if np.isnan(tx_true):
            lar_dir = "LAR: Unknown"
        elif tx_true >= self.look_threshold_mm:
            lar_dir = "LAR: Looking RIGHT"
        elif tx_true <= -self.look_threshold_mm:
            lar_dir = "LAR: Looking LEFT"
        else:
            lar_dir = "LAR: Straight"
        self.lbl_lar_dir.setText(lar_dir)


        # Pose panel values (show all, even if not plotted)
        for pose in POSES:
            tx = row.get(f"{pose}_Tx", np.nan)
            ty = row.get(f"{pose}_Ty", np.nan)
            tz = row.get(f"{pose}_Tz", np.nan)
            rx = row.get(f"{pose}_Rx", np.nan)
            ry = row.get(f"{pose}_Ry", np.nan)
            rz = row.get(f"{pose}_Rz", np.nan)

            self.pose_labels[pose].setText(
                f"Tx: {tx: .3f}\nTy: {ty: .3f}\nTz: {tz: .3f}\nRx: {rx: .3f}\nRy: {ry: .3f}\nRz: {rz: .3f}"
            )

        # d1/d2: which candidate matches True more closely in (Tx, Ty, Rz)
        p1 = np.array([row["Pose1_Tx"], row["Pose1_Ty"], row["Pose1_Rz"]], dtype=float)
        p2 = np.array([row["Pose2_Tx"], row["Pose2_Ty"], row["Pose2_Rz"]], dtype=float)
        tr = np.array([row["True_Tx"], row["True_Ty"], row["True_Rz"]], dtype=float)

        d1 = np.linalg.norm(p1 - tr)
        d2 = np.linalg.norm(p2 - tr)
        chosen = "Pose1" if d1 <= d2 else "Pose2"

        self.lbl_choice.setText(
            f"<b>Chosen:</b> {chosen} (d1={d1:.4g}, d2={d2:.4g})<br>"
            "<span style='font-size:11px;'>"
            "d1 = ||[Pose1_Tx,Pose1_Ty,Pose1_Rz] - [True_Tx,True_Ty,True_Rz]||<br>"
            "d2 = ||[Pose2_Tx,Pose2_Ty,Pose2_Rz] - [True_Tx,True_Ty,True_Rz]||"
            "</span>"
        )

        # Update plots (frame-by-frame)
        self._update_plots_incremental()


def main():
    app = QApplication(sys.argv)
    w = CsvPosePlayer()
    w.resize(1220, 920)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
