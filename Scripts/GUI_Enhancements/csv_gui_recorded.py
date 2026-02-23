"""
csv_recorded_gui.py
-------------------
Recorded CSV pose visualization GUI for the DOT Capstone CV pipeline.

MANDATORY INPUT:
- CSV file (Jan12GNCTest.csv style)

OPTIONAL INPUT:
- MP4 video (raw footage). If loaded, it is shown in a separate OpenCV window and synced by frame index.

Key features:
- Frame-by-frame playback (Play/Pause) using a Qt timer
- Scrubber slider to jump to any frame
- Pose panels for Pose1, Pose2, True (shows Tx, Ty, Tz, Rx, Ry, Rz columns as-is)
- MAIN plots (3 only): Rz (depth mm), Rx (lateral mm), Yaw_abs = atan2(Tx, Tz)+180 degrees
- Vertical cursor line on each plot marking the current frame
- "LAR: Looking LEFT/RIGHT/Straight" derived from True_Rx sign (with deadband)
"""

import sys
import numpy as np
import pandas as pd
import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QMessageBox
)

import pyqtgraph as pg


POSES = ["Pose1", "Pose2", "True"]

# Raw CSV column mapping (as stored in your spreadsheet/CSV)
AXES = {
    "Tx": ("Pose1_Tx", "Pose2_Tx", "True_Tx"),  # normal component (unitless)
    "Ty": ("Pose1_Ty", "Pose2_Ty", "True_Ty"),  # normal component (unitless)
    "Tz": ("Pose1_Tz", "Pose2_Tz", "True_Tz"),  # normal component (unitless)
    "Rx": ("Pose1_Rx", "Pose2_Rx", "True_Rx"),  # translation distance (mm) — lateral
    "Ry": ("Pose1_Ry", "Pose2_Ry", "True_Ry"),  # translation distance (mm)
    "Rz": ("Pose1_Rz", "Pose2_Rz", "True_Rz"),  # translation distance (mm) — depth/range
}

# We only show these 3 graphs now:
# 1) Rz (depth, mm)
# 2) Rx (lateral, mm)
# 3) Yaw_abs_deg = atan2(Tx, Tz) + 180 degrees  (using normal-vector components)
MAIN_PLOTS = ["Rz", "Rx", "Yaw_abs_deg"]


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def yaw_abs_deg_from_normal(tx, tz):
    """
    Compute absolute yaw angle (deg) from normal components.
    yaw = atan2(tx, tz) in radians, converted to degrees, shifted by +180 for [0, 360).
    """
    if np.isnan(tx) or np.isnan(tz):
        return np.nan
    yaw = np.degrees(np.arctan2(tx, tz)) + 180.0
    # Keep it in [0, 360)
    yaw = yaw % 360.0
    return yaw


class CsvPosePlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LAR Pose CSV Viewer (Depth / Lateral / Yaw)")

        # CSV state
        self.df = None
        self.i = 0
        self.playing = False

        # Playback speed control
        self.base_fps = 10
        self.playback_fps = self.base_fps

        # Plot windowing: show last N seconds (time-based), x-axis is frame index
        self.window_seconds = 8.0

        # Left/right deadband on lateral translation (mm)
        self.look_threshold_mm = 5.0

        # OPTIONAL video state (separate OpenCV window)
        self.video_path = None
        self.cap = None
        self.video_frame_count = 0
        self.video_window_name = "LAR Raw Footage (Synced)"
        self.video_enabled = False
        self.last_video_frame_shown = -1

        # Timer drives playback
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        # ---------- UI ----------
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        # Top controls
        top = QHBoxLayout()

        self.btn_open_csv = QPushButton("Open CSV")
        self.btn_open_csv.clicked.connect(self.open_csv)

        self.btn_open_video = QPushButton("Open Video (Optional)")
        self.btn_open_video.clicked.connect(self.open_video)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset)

        self.lbl_info = QLabel("No CSV loaded.")
        self.lbl_info.setMinimumWidth(560)

        self.lbl_key = QLabel("Blue=Pose1 | Orange=Pose2 | Green(thick)=True/Chosen")
        self.lbl_key.setStyleSheet("font-weight: 600;")

        top.addWidget(self.btn_open_csv)
        top.addWidget(self.btn_open_video)
        top.addWidget(self.btn_play)
        top.addWidget(self.btn_reset)
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
        #self.lbl_lar_dir = QLabel("LAR: -")
        #self.lbl_lar_dir.setStyleSheet("font-weight: 700;")

        scrub.addWidget(QLabel("Scrub:"))
        scrub.addWidget(self.slider, 1)
        scrub.addWidget(self.lbl_frame)
        scrub.addWidget(self.lbl_time)
        scrub.addSpacing(12)
        #scrub.addWidget(self.lbl_lar_dir)

        main.addLayout(scrub)

        # Pose panels
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
            "d1 = ||[Pose1_Rx, Pose1_Rz, yaw1] - [True_Rx, True_Rz, yawT]||<br>"
            "d2 = ||[Pose2_Rx, Pose2_Rz, yaw2] - [True_Rx, True_Rz, yawT]||"
            "</span>"
        )
        pose_row.addWidget(self.lbl_choice)
        main.addLayout(pose_row)

        # Plots
        self.plot_widget = pg.GraphicsLayoutWidget()
        main.addWidget(self.plot_widget, 1)

        self.plots = {}
        self.curves = {}
        self.cursors = {}

        self._setup_plots()

    # ---------- Plot setup ----------
    def _setup_plots(self):
        self.plot_widget.clear()
        self.plots.clear()
        self.curves.clear()
        self.cursors.clear()

        pose_styles = {
            "Pose1": pg.mkPen(color=(80, 170, 255), width=2),   # blue
            "Pose2": pg.mkPen(color=(255, 170, 80), width=2),   # orange
            "True":  pg.mkPen(color=(120, 255, 120), width=3),  # green thick
        }

        y_labels = {
            "Rz": "Depth Translation (mm)",
            "Rx": "Lateral Translation (mm)",
            "Yaw_abs_deg": "Absolute Yaw (deg) = atan2(Tx, Tz)+180",
        }

        for r, sig in enumerate(MAIN_PLOTS):
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

            cursor_pen = pg.mkPen(color=(180, 180, 180), width=2, style=Qt.DashLine)
            vline = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=cursor_pen)
            p.addItem(vline)
            self.cursors[sig] = vline

        # Legend on the first plot only
        self.plots[MAIN_PLOTS[0]].addLegend()

        if self.df is not None:
            self._update_frame_ui()

    def _clear_plots(self):
        for sig in MAIN_PLOTS:
            for pose_name in POSES:
                self.curves[sig][pose_name].setData([], [])
            self.cursors[sig].setPos(0)

    # ---------- CSV loading ----------
    def open_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
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

        N = len(df)
        t0 = df["Time [s]"].iloc[0]
        t1 = df["Time [s]"].iloc[-1]
        duration = float(t1 - t0) if np.isfinite(t1 - t0) else float("nan")
        fps_est = (N - 1) / duration if (np.isfinite(duration) and duration > 0) else float("nan")

        self.lbl_info.setText(
            f"CSV: {path} | Frames: {N} | Duration: {duration:.2f}s | CSV FPS≈{fps_est:.2f}"
        )

        self._clear_plots()
        self._update_frame_ui()
        self._update_video_window(force=True)

    # ---------- Optional video ----------
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Video Error", "Failed to open video.")
            return

        self._close_video()

        self.video_path = path
        self.cap = cap
        self.video_enabled = True
        self.video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.last_video_frame_shown = -1

        cv2.namedWindow(self.video_window_name, cv2.WINDOW_NORMAL)
        self._update_video_window(force=True)

    def _update_video_window(self, force: bool = False):
        if not self.video_enabled or self.cap is None:
            return
        if not force and self.i == self.last_video_frame_shown:
            return

        frame_idx = int(self.i)
        if self.video_frame_count > 0:
            frame_idx = max(0, min(frame_idx, self.video_frame_count - 1))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        self.last_video_frame_shown = frame_idx
        cv2.imshow(self.video_window_name, frame)
        cv2.waitKey(1)

    def _close_video(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.video_enabled = False
        self.last_video_frame_shown = -1
        try:
            cv2.destroyWindow(self.video_window_name)
        except Exception:
            pass

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

    def reset(self):
        if self.df is None:
            return

        self.i = 0
        self._clear_plots()

        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        self._update_frame_ui()
        self._update_video_window(force=True)

    def slider_changed(self, v):
        if self.df is None:
            return
        self.i = int(v)
        self._update_frame_ui()
        self._update_video_window(force=True)

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
        self._update_video_window()

    # ---------- Plot update ----------
    def _update_plots_incremental(self):
        if self.df is None:
            return

        t_all = self.df["Time [s]"].to_numpy()
        t_now = t_all[self.i]

        if self.window_seconds is None:
            start = 0
        else:
            t_min = t_now - self.window_seconds
            start = int(np.searchsorted(t_all, t_min, side="left"))

        x = np.arange(start, self.i + 1)

        # Helper: fetch a Pose column slice
        def col_slice(col_name):
            return self.df[col_name].to_numpy()[start:self.i + 1]

        # 1) Rz plot (depth mm) — uses *_Rz columns directly
        for pose_name, (c1, c2, ct) in [("Pose1", AXES["Rz"]), ("Pose2", AXES["Rz"]), ("True", AXES["Rz"])]:
            pass  # (we’ll set below more cleanly)

        rz_cols = AXES["Rz"]
        rx_cols = AXES["Rx"]

        # For yaw, we compute from Tx and Tz normal components
        tx_cols = AXES["Tx"]
        tz_cols = AXES["Tz"]

        # Build arrays for each pose for each main plot
        for pose_idx, pose_name in enumerate(POSES):
            # Columns for this pose
            rz_col = rz_cols[pose_idx]
            rx_col = rx_cols[pose_idx]
            tx_col = tx_cols[pose_idx]
            tz_col = tz_cols[pose_idx]

            # Depth & lateral
            y_rz = col_slice(rz_col)
            y_rx = col_slice(rx_col)

            # Yaw from normal
            tx = col_slice(tx_col)
            tz = col_slice(tz_col)
            y_yaw = np.array([yaw_abs_deg_from_normal(a, b) for a, b in zip(tx, tz)], dtype=float)

            self.curves["Rz"][pose_name].setData(x, y_rz)
            self.curves["Rx"][pose_name].setData(x, y_rx)
            self.curves["Yaw_abs_deg"][pose_name].setData(x, y_yaw)

        # Cursor lines + x-range scroll
        for sig in MAIN_PLOTS:
            self.cursors[sig].setPos(self.i)
            if self.window_seconds is not None:
                self.plots[sig].setXRange(start, self.i, padding=0)

    # ---------- Frame UI update ----------
    def _update_frame_ui(self):
        if self.df is None:
            return

        row = self.df.iloc[self.i]
        n = len(self.df)
        t = row["Time [s]"]

        self.lbl_frame.setText(f"Frame: {self.i} / {n - 1}")
        self.lbl_time.setText(f"t = {t:.3f} s")

        # LEFT/RIGHT based on *lateral translation* (True_Rx)
        #rx_true = float(row["True_Rx"])
        #if np.isnan(rx_true):
        #    lar_dir = "LAR: Unknown"
        #elif rx_true >= self.look_threshold_mm:
        #    lar_dir = "LAR: Looking RIGHT"
        #elif rx_true <= -self.look_threshold_mm:
        #    lar_dir = "LAR: Looking LEFT"
        #else:
        #    lar_dir = "LAR: Straight"
        #self.lbl_lar_dir.setText(lar_dir)

        # Pose panels still show all raw columns (Tx..Rz) as stored
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

        # Update d1/d2 in the new 3DoF:
        #   lateral = Rx (mm)
        #   depth   = Rz (mm)
        #   yaw     = atan2(Tx, Tz)+180 (deg)
        def pose_triplet(prefix):
            rx = float(row[f"{prefix}_Rx"])
            rz = float(row[f"{prefix}_Rz"])
            tx = float(row[f"{prefix}_Tx"])
            tz = float(row[f"{prefix}_Tz"])
            yaw = yaw_abs_deg_from_normal(tx, tz)
            return np.array([rx, rz, yaw], dtype=float)

        p1 = pose_triplet("Pose1")
        p2 = pose_triplet("Pose2")
        tr = pose_triplet("True")

        # Handle NaNs safely: if yaw is NaN, distances become NaN — that’s OK to display
        d1 = np.linalg.norm(p1 - tr) if np.all(np.isfinite(p1)) and np.all(np.isfinite(tr)) else np.nan
        d2 = np.linalg.norm(p2 - tr) if np.all(np.isfinite(p2)) and np.all(np.isfinite(tr)) else np.nan

        if np.isnan(d1) or np.isnan(d2):
            chosen = "Pose1"  # fallback (True is already chosen in your CSV)
        else:
            chosen = "Pose1" if d1 <= d2 else "Pose2"

        self.lbl_choice.setText(
            f"<b>Chosen:</b> {chosen} (d1={d1:.4g}, d2={d2:.4g})<br>"
            "<span style='font-size:11px;'>"
            "d1 = ||[Pose1_Rx, Pose1_Rz, yaw1] - [True_Rx, True_Rz, yawT]||<br>"
            "d2 = ||[Pose2_Rx, Pose2_Rz, yaw2] - [True_Rx, True_Rz, yawT]||"
            "</span>"
        )

        self._update_plots_incremental()

    def closeEvent(self, event):
        self._close_video()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = CsvPosePlayer()
    w.resize(1220, 920)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()