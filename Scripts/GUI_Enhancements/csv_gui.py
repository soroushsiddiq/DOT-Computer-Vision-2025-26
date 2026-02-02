"""
csv_gui.py
----------
CSV-first pose visualization GUI for the DOT Capstone CV pipeline.

MANDATORY INPUT:
- CSV file (Jan12GNCTest.csv style)

OPTIONAL INPUT:
- MP4 video (raw footage). If loaded, it is shown in a separate OpenCV window and synced by frame index.

Key features:
- Frame-by-frame playback (Play/Pause) using a Qt timer
- Scrubber slider to jump to any frame
- Pose panels for Pose1, Pose2, True
- Real-time plots for Tx, Ty, Rz (main), and optional extra plots (Tz/Rx/Ry) via dropdown
- Vertical cursor line on each plot marking the current frame
- "LAR: Looking LEFT/RIGHT/Straight" derived from True_Tx sign (with deadband)
"""

import sys
import numpy as np
import pandas as pd
import cv2  # Only used if the user opens a video; safe to import either way.

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox, QMessageBox
)

import pyqtgraph as pg


# Pose sources / labels
POSES = ["Pose1", "Pose2", "True"]

# Mapping from signal name -> (Pose1 column, Pose2 column, True column)
AXES = {
    "Tx": ("Pose1_Tx", "Pose2_Tx", "True_Tx"),
    "Ty": ("Pose1_Ty", "Pose2_Ty", "True_Ty"),
    "Tz": ("Pose1_Tz", "Pose2_Tz", "True_Tz"),
    "Rx": ("Pose1_Rx", "Pose2_Rx", "True_Rx"),
    "Ry": ("Pose1_Ry", "Pose2_Ry", "True_Ry"),
    "Rz": ("Pose1_Rz", "Pose2_Rz", "True_Rz"),
}

# Always-visible “main” graphs (core 3-DoF)
MAIN_SIGNALS = ["Tx", "Ty", "Rz"]

# Dropdown options for extra plots (rebuild plot area when changed)
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
    """Convert to float; return NaN if conversion fails."""
    try:
        return float(x)
    except Exception:
        return np.nan


class CsvPosePlayer(QMainWindow):
    """
    Main GUI window.
    Holds:
    - CSV data + current index i
    - Playback timer
    - Plot objects + cursor lines
    - Optional video capture, shown in separate OpenCV window if loaded
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LAR Pose CSV Viewer")

        # -----------------------
        # CSV state (mandatory)
        # -----------------------
        self.df = None          # pandas DataFrame containing CSV rows
        self.i = 0              # current frame index (row index)
        self.playing = False    # playback state

        # Playback speed (GUI timer interval)
        self.base_fps = 10
        self.playback_fps = self.base_fps

        # Plot windowing:
        # - If None: show full history (0..i)
        # - If number: show only last N seconds by using Time[s] to find start index
        self.window_seconds = None

        # LAR direction threshold (deadband to avoid flicker near Tx ~ 0)
        self.look_threshold_mm = 5.0

        # Extra plot selection state
        self.extra_signals = []  # e.g., ["Tz","Rx"]

        # -----------------------
        # OPTIONAL video state
        # -----------------------
        self.video_path = None
        self.cap = None                  # cv2.VideoCapture or None
        self.video_frame_count = 0
        self.video_fps = 0.0
        self.video_window_name = "LAR Raw Footage (Synced)"
        self.video_enabled = False
        self.last_video_frame_shown = -1

        # -----------------------
        # Timer drives playback
        # -----------------------
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        # -----------------------
        # Build UI
        # -----------------------
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        # --- Top controls ---
        top = QHBoxLayout()

        self.btn_open_csv = QPushButton("Open CSV")
        self.btn_open_csv.clicked.connect(self.open_csv)

        self.btn_open_video = QPushButton("Open Video (Optional)")
        self.btn_open_video.clicked.connect(self.open_video)

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

        # File/status info label
        self.lbl_info = QLabel("No CSV loaded.")
        self.lbl_info.setMinimumWidth(560)

        # Color key for curves
        self.lbl_key = QLabel("Blue=Pose1 | Orange=Pose2 | Green(thick)=True/Chosen")
        self.lbl_key.setStyleSheet("font-weight: 600;")

        top.addWidget(self.btn_open_csv)
        top.addWidget(self.btn_open_video)
        top.addWidget(self.btn_play)
        top.addWidget(self.btn_reset)
        top.addSpacing(10)
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

        # --- Scrubber row ---
        scrub = QHBoxLayout()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)              # disabled until CSV loaded
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

        # --- Pose panels ---
        pose_row = QHBoxLayout()
        self.pose_labels = {}

        for p in POSES:
            box = QVBoxLayout()
            title = QLabel(f"<b>{p}</b>")
            box.addWidget(title)

            # Monospace label for readability
            lbl = QLabel("Tx: -\nTy: -\nTz: -\nRx: -\nRy: -\nRz: -")
            lbl.setStyleSheet("font-family: Consolas, monospace;")
            box.addWidget(lbl)

            self.pose_labels[p] = lbl
            pose_row.addLayout(box)

        # "Chosen" explanation + which candidate matches True
        self.lbl_choice = QLabel(
            "<b>Chosen:</b> -<br>"
            "<span style='font-size:11px;'>"
            "d1 = ||[Pose1_Tx,Pose1_Ty,Pose1_Rz] - [True_Tx,True_Ty,True_Rz]||<br>"
            "d2 = ||[Pose2_Tx,Pose2_Ty,Pose2_Rz] - [True_Tx,True_Ty,True_Rz]||"
            "</span>"
        )
        pose_row.addWidget(self.lbl_choice)

        main.addLayout(pose_row)

        # --- Plots area (pyqtgraph) ---
        self.plot_widget = pg.GraphicsLayoutWidget()
        main.addWidget(self.plot_widget, 1)

        # Plot objects
        self.plots = {}     # plots[sig] = PlotItem
        self.curves = {}    # curves[sig][pose] = PlotDataItem
        self.cursors = {}   # cursors[sig] = InfiniteLine

        # Initial plot build (Tx/Ty/Rz only)
        self._setup_plots()

    # ------------------------------------------------------------
    # Plot building / rebuilding (when extras dropdown changes)
    # ------------------------------------------------------------
    def _setup_plots(self):
        """Build plots for MAIN_SIGNALS + extra_signals."""
        self.plot_widget.clear()
        self.plots.clear()
        self.curves.clear()
        self.cursors.clear()

        # Line styles for each pose source
        pose_styles = {
            "Pose1": pg.mkPen(color=(80, 170, 255), width=2),   # blue
            "Pose2": pg.mkPen(color=(255, 170, 80), width=2),   # orange
            "True":  pg.mkPen(color=(120, 255, 120), width=3),  # green thick
        }

        # Y-axis labels
        # Note: In your pipeline, Rx/Ry/Rz appear to be normal-vector components (unitless),
        # not Euler angles; label accordingly.
        y_labels = {
            "Tx": "Tx (mm)",
            "Ty": "Ty (mm)",
            "Tz": "Tz (mm)",
            "Rx": "Rx (unitless / normal comp)",
            "Ry": "Ry (unitless / normal comp)",
            "Rz": "Rz (unitless / normal comp)",
        }

        # Signals displayed = main + extras
        signals = MAIN_SIGNALS + list(self.extra_signals)

        # Build one plot per signal vertically stacked
        for r, sig in enumerate(signals):
            p = self.plot_widget.addPlot(row=r, col=0)
            p.showGrid(x=True, y=True)
            p.setLabel("left", y_labels.get(sig, sig))
            p.setLabel("bottom", "Frame")  # x-axis is frame index
            self.plots[sig] = p

            # Create 3 curves: Pose1, Pose2, True
            self.curves[sig] = {}
            for pose_name in POSES:
                legend_name = {
                    "Pose1": "Pose1 (Blue)",
                    "Pose2": "Pose2 (Orange)",
                    "True":  "True/Chosen (Green)"
                }[pose_name]
                curve = p.plot([], [], pen=pose_styles[pose_name], name=legend_name)
                self.curves[sig][pose_name] = curve

            # Vertical cursor line marking current frame index
            cursor_pen = pg.mkPen(color=(180, 180, 180), width=2, style=Qt.DashLine)
            vline = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=cursor_pen)
            p.addItem(vline)
            self.cursors[sig] = vline

        # Only add legend to the top plot (saves space)
        if signals:
            self.plots[signals[0]].addLegend()

        # If we already have CSV loaded, refresh to current frame
        if self.df is not None:
            self._update_frame_ui()

    def _clear_plots(self):
        """Reset curves and cursor positions."""
        signals = MAIN_SIGNALS + list(self.extra_signals)
        for sig in signals:
            for pose_name in POSES:
                if sig in self.curves and pose_name in self.curves[sig]:
                    self.curves[sig][pose_name].setData([], [])
            if sig in self.cursors:
                self.cursors[sig].setPos(0)

    # ------------------------------------------------------------
    # CSV loading (mandatory input)
    # ------------------------------------------------------------
    def open_csv(self):
        """Open CSV and prepare slider + labels."""
        path, _ = QFileDialog.getOpenFileName(self, "Open Jan12GNCTest.csv", "", "CSV Files (*.csv)")
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "CSV Error", f"Failed to read CSV:\n{e}")
            return

        # Validate required columns
        required = ["Time [s]"]
        for sig in AXES:
            required.extend(list(AXES[sig]))

        missing = [c for c in required if c not in df.columns]
        if missing:
            QMessageBox.critical(self, "CSV Error", f"CSV missing columns:\n{missing}")
            return

        # Convert required columns to numeric
        for c in required:
            df[c] = df[c].map(safe_float)

        self.df = df
        self.i = 0

        # Enable scrubber
        self.slider.setEnabled(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(df) - 1)
        self.slider.setValue(0)

        # Update info label with frames + duration + estimated fps
        N = len(df)
        t0 = df["Time [s]"].iloc[0]
        t1 = df["Time [s]"].iloc[-1]
        duration = float(t1 - t0) if np.isfinite(t1 - t0) else float("nan")
        fps_est = (N - 1) / duration if (np.isfinite(duration) and duration > 0) else float("nan")

        self.lbl_info.setText(
            f"CSV: {path} | Frames: {N} | Duration: {duration:.2f}s | CSV FPS≈{fps_est:.2f} | Playback FPS={self.playback_fps}"
        )

        # Reset plots + draw first frame state
        self._clear_plots()
        self._update_frame_ui()

        # If video was loaded previously, refresh it to frame 0 (optional)
        self._update_video_window(force=True)

    # ------------------------------------------------------------
    # OPTIONAL video controls (separate OpenCV window)
    # ------------------------------------------------------------
    def open_video(self):
        """
        Open optional MP4 and display in a separate OpenCV window.
        This does NOT affect CSV plotting if video is not opened.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Video Error", "Failed to open video.")
            return

        # Close any previous video capture/window
        self._close_video()

        self.video_path = path
        self.cap = cap
        self.video_enabled = True
        self.video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.last_video_frame_shown = -1

        # Create separate window (independent of Qt layout)
        cv2.namedWindow(self.video_window_name, cv2.WINDOW_NORMAL)

        # Show current CSV frame if loaded; otherwise show frame 0
        self._update_video_window(force=True)

    def _update_video_window(self, force: bool = False):
        """
        If video is enabled, show the frame corresponding to self.i in a separate OpenCV window.
        Sync method = frame index (row i ≈ video frame i).
        """
        if not self.video_enabled or self.cap is None:
            return

        if not force and self.i == self.last_video_frame_shown:
            return

        frame_idx = int(self.i)

        # Clamp to available range if we know it
        if self.video_frame_count > 0:
            frame_idx = max(0, min(frame_idx, self.video_frame_count - 1))
        else:
            frame_idx = max(0, frame_idx)

        # Seek and read
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        self.last_video_frame_shown = frame_idx
        cv2.imshow(self.video_window_name, frame)
        cv2.waitKey(1)  # keeps the OpenCV window responsive

    def _close_video(self):
        """Release video resources and close the OpenCV window."""
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

    # ------------------------------------------------------------
    # Playback + UI controls
    # ------------------------------------------------------------
    def toggle_play(self):
        """Toggle play/pause (CSV playback)."""
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
        """Adjust playback speed (timer frequency)."""
        mult = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}[text]
        self.playback_fps = max(1, int(self.base_fps * mult))

        # If playing, restart timer with new interval
        if self.playing:
            self.timer.stop()
            interval_ms = int(1000 / self.playback_fps)
            self.timer.start(max(interval_ms, 1))

        # Update the info label's playback fps if CSV is loaded
        if self.df is not None:
            # Keep it simple: just rewrite the last part by recomputing stats quickly
            df = self.df
            N = len(df)
            t0 = df["Time [s]"].iloc[0]
            t1 = df["Time [s]"].iloc[-1]
            duration = float(t1 - t0) if np.isfinite(t1 - t0) else float("nan")
            fps_est = (N - 1) / duration if (np.isfinite(duration) and duration > 0) else float("nan")
            self.lbl_info.setText(
                f"CSV loaded | Frames: {N} | Duration: {duration:.2f}s | CSV FPS≈{fps_est:.2f} | Playback FPS={self.playback_fps}"
            )

    def set_extras(self, text):
        """Change extra plot signals and rebuild plot layout."""
        for name, sigs in EXTRA_OPTIONS:
            if name == text:
                self.extra_signals = sigs
                break
        self._setup_plots()

    def reset(self):
        """Reset to frame 0 and clear plots."""
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
        """When user scrubs, jump to that frame and update UI/plots/video."""
        if self.df is None:
            return

        self.i = int(v)
        self._update_frame_ui()
        self._update_video_window(force=True)

    # ------------------------------------------------------------
    # Timer tick: advance one frame
    # ------------------------------------------------------------
    def _tick(self):
        """Advance by one frame during playback."""
        if self.df is None:
            return

        self.i += 1
        if self.i >= len(self.df):
            self.i = len(self.df) - 1
            self.toggle_play()  # auto-stop at end
            return

        # Update slider without retriggering slider_changed
        self.slider.blockSignals(True)
        self.slider.setValue(self.i)
        self.slider.blockSignals(False)

        self._update_frame_ui()
        self._update_video_window()

    # ------------------------------------------------------------
    # Plot update helper
    # ------------------------------------------------------------
    def _update_plots_incremental(self):
        """
        Update plot curves up to the current frame index.
        Uses time-based windowing (window_seconds) but x-axis is frame number.
        """
        if self.df is None:
            return

        t_all = self.df["Time [s]"].to_numpy()
        t_now = t_all[self.i]

        # Determine the start index for the plotted window
        if self.window_seconds is None:
            start = 0
        else:
            t_min = t_now - self.window_seconds
            start = int(np.searchsorted(t_all, t_min, side="left"))

        # X-axis = frame indices
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

            # Move the vertical cursor to the current frame
            self.cursors[sig].setPos(self.i)

        # Keep plots scrolling if windowing is enabled
        if self.window_seconds is not None:
            for sig in signals:
                self.plots[sig].setXRange(start, self.i, padding=0)

    # ------------------------------------------------------------
    # Update all per-frame UI elements
    # ------------------------------------------------------------
    def _update_frame_ui(self):
        """Update labels, pose panels, chosen pose text, and plots for frame self.i."""
        if self.df is None:
            return

        row = self.df.iloc[self.i]
        n = len(self.df)
        t = row["Time [s]"]

        self.lbl_frame.setText(f"Frame: {self.i} / {n - 1}")
        self.lbl_time.setText(f"t = {t:.3f} s")

        # LAR direction based on True_Tx sign convention (matching main.py),
        # with deadband to avoid flicker around 0.
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

        # Pose panels (show all values even if not plotted)
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

        # d1/d2 tells which candidate pose matches "True" more closely in the 3-DoF you care about
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

        # Update plots incrementally
        self._update_plots_incremental()

    def closeEvent(self, event):
        """Cleanup on GUI close."""
        self._close_video()
        super().closeEvent(event)


def main():
    """Entry point."""
    app = QApplication(sys.argv)
    w = CsvPosePlayer()
    w.resize(1220, 920)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
