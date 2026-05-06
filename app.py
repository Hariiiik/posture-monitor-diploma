"""
Posture Monitor — PyQt5 GUI Application
========================================
Main window with two tabs:
  1. Live Dashboard  — real-time video, neck-angle chart, traffic-light, controls
  2. Analytics       — placeholder charts for future SQLCipher integration
"""

import sys
import math
import numpy as np
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGroupBox, QSpinBox, QDoubleSpinBox,
    QFormLayout, QSizePolicy, QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QFont, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from worker import PoseWorker

# ─── Audio warning helper ────────────────────────────────────────────────────

def send_warning_beep(duration_ms=500, freq=800):
    """
    Play an audible warning beep.
    Uses sounddevice + numpy to generate a sine-wave tone.
    Falls back to a simple print if sounddevice is unavailable.
    """
    try:
        import sounddevice as sd
        sr = 44100
        t = np.linspace(0, duration_ms / 1000.0, int(sr * duration_ms / 1000.0), endpoint=False)
        wave = (0.5 * np.sin(2 * math.pi * freq * t)).astype(np.float32)
        sd.play(wave, sr, blocking=False)
    except Exception:
        print("\a")  # terminal bell fallback


# ═══════════════════════════════════════════════════════════════════════════════
#  Live Dashboard Tab
# ═══════════════════════════════════════════════════════════════════════════════

class LiveDashboardTab(QWidget):
    """Tab 1 — real-time video feed, neck-angle chart, indicators, controls."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._angle_buf = deque(maxlen=300)
        self._time_buf = deque(maxlen=300)
        self._warning_active = False
        self._calibrating = False

        # ── Video widget ──────────────────────────────────────────────────
        self.video_label = QLabel("Натисніть «Start Monitoring» для початку")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background: #1a1a2e; color: #888; border: 2px solid #333; border-radius: 8px;"
        )

        # ── Side (picture-in-picture) video widget ────────────────────────
        self.side_video_label = QLabel("")
        self.side_video_label.setAlignment(Qt.AlignCenter)
        self.side_video_label.setMinimumSize(320, 240)
        self.side_video_label.setMaximumSize(360, 270)
        self.side_video_label.setStyleSheet(
            "background: #111827; color: #888; border: 2px solid #333; border-radius: 8px;"
        )

        # ── Matplotlib chart (in-place redraw) ────────────────────────────
        self.chart_fig = Figure(figsize=(6, 2.2), dpi=80)
        self.chart_fig.patch.set_facecolor('#1a1a2e')
        self.chart_ax = self.chart_fig.add_subplot(111)
        self.chart_ax.set_facecolor('#16213e')
        self.chart_ax.set_ylabel('2D CVA °', color='#aaa', fontsize=9)
        self.chart_ax.tick_params(colors='#888', labelsize=8)
        for spine in self.chart_ax.spines.values():
            spine.set_color('#333')
        self.chart_line, = self.chart_ax.plot([], [], color='#00d2ff', linewidth=1.5)
        self.chart_threshold_line = None  # drawn once we know threshold
        self.chart_canvas = FigureCanvas(self.chart_fig)
        self.chart_canvas.setMinimumHeight(150)

        # ── Traffic-light indicators ──────────────────────────────────────
        indicator_box = QGroupBox("Стан")
        indicator_box.setStyleSheet(
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 6px; "
            "margin-top: 10px; padding-top: 14px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-position: top center; padding: 0 8px; }"
        )
        ind_layout = QHBoxLayout()
        self.green_light = self._make_light('#2d6a2d', 30)
        self.red_light = self._make_light('#6a2d2d', 30)
        self.status_label = QLabel("—")
        self.status_label.setStyleSheet("color: #aaa; font-size: 13px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        ind_layout.addWidget(self.green_light)
        ind_layout.addWidget(self.red_light)
        ind_layout.addWidget(self.status_label)
        indicator_box.setLayout(ind_layout)

        # ── Time labels ───────────────────────────────────────────────────
        self.good_time_label = QLabel("Good: 0.0 s")
        self.good_time_label.setStyleSheet("color: #7fff7f; font-size: 12px;")
        self.bad_time_label = QLabel("Bad: 0.0 s")
        self.bad_time_label.setStyleSheet("color: #ff7f7f; font-size: 12px;")

        time_layout = QHBoxLayout()
        time_layout.addWidget(self.good_time_label)
        time_layout.addWidget(self.bad_time_label)

        # ── Controls ──────────────────────────────────────────────────────
        ctrl_box = QGroupBox("Управління")
        ctrl_box.setStyleSheet(
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 6px; "
            "margin-top: 10px; padding-top: 14px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-position: top center; padding: 0 8px; }"
        )
        btn_style = (
            "QPushButton { background: #16213e; color: #ddd; border: 1px solid #555; "
            "border-radius: 6px; padding: 8px 18px; font-size: 13px; }"
            "QPushButton:hover { background: #1a3a5c; }"
            "QPushButton:disabled { background: #111; color: #555; }"
        )
        self.btn_start = QPushButton("▶  Start Monitoring")
        self.btn_stop = QPushButton("■  Stop Monitoring")
        self.btn_calibrate = QPushButton("⊕  Calibrate")
        self.btn_stop.setEnabled(False)
        self.btn_calibrate.setEnabled(False)
        for b in (self.btn_start, self.btn_stop, self.btn_calibrate):
            b.setStyleSheet(btn_style)

        self.calibrate_label = QLabel("")
        self.calibrate_label.setStyleSheet("color: #ffcc00; font-size: 12px;")
        self.calibrate_label.setAlignment(Qt.AlignCenter)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_calibrate)
        ctrl_v = QVBoxLayout()
        ctrl_v.addLayout(ctrl_layout)
        ctrl_v.addWidget(self.calibrate_label)
        ctrl_box.setLayout(ctrl_v)

        # ── Settings panel (GUI thresholds) ───────────────────────────────
        settings_box = QGroupBox("Налаштування порогів")
        settings_box.setStyleSheet(
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 6px; "
            "margin-top: 10px; padding-top: 14px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-position: top center; padding: 0 8px; }"
            "QLabel { color: #bbb; font-size: 12px; }"
            "QSpinBox, QDoubleSpinBox { background: #16213e; color: #ddd; "
            "border: 1px solid #555; border-radius: 4px; padding: 3px; }"
        )
        form = QFormLayout()

        self.spin_neck = QSpinBox()
        self.spin_neck.setRange(5, 60)
        self.spin_neck.setValue(50)
        self.spin_neck.setSuffix(" °")
        form.addRow("Кут шиї:", self.spin_neck)

        self.spin_tilt = QDoubleSpinBox()
        self.spin_tilt.setRange(0.01, 0.30)
        self.spin_tilt.setValue(0.05)
        self.spin_tilt.setSingleStep(0.01)
        self.spin_tilt.setSuffix(" m")
        form.addRow("Нахил плечей:", self.spin_tilt)

        self.spin_lean = QDoubleSpinBox()
        self.spin_lean.setRange(0.01, 0.50)
        self.spin_lean.setValue(0.10)
        self.spin_lean.setSingleStep(0.01)
        self.spin_lean.setSuffix(" m")
        form.addRow("Нахил вперед:", self.spin_lean)

        self.spin_time = QSpinBox()
        self.spin_time.setRange(10, 600)
        self.spin_time.setValue(180)
        self.spin_time.setSuffix(" s")
        form.addRow("Час попередж.:", self.spin_time)

        settings_box.setLayout(form)

        # ── Right panel ───────────────────────────────────────────────────
        right_panel = QVBoxLayout()
        right_panel.addWidget(indicator_box)
        right_panel.addLayout(time_layout)
        right_panel.addWidget(self.chart_canvas)
        right_panel.addWidget(ctrl_box)
        right_panel.addWidget(settings_box)
        right_panel.addStretch()

        right_w = QWidget()
        right_w.setLayout(right_panel)
        right_w.setMaximumWidth(380)

        # ── Main layout ──────────────────────────────────────────────────
        main_layout = QHBoxLayout()
        left_videos = QHBoxLayout()
        left_videos.addWidget(self.video_label, stretch=3)
        left_videos.addWidget(self.side_video_label, stretch=1)
        left_w = QWidget()
        left_w.setLayout(left_videos)

        main_layout.addWidget(left_w, stretch=3)
        main_layout.addWidget(right_w, stretch=1)
        self.setLayout(main_layout)

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _make_light(color, size):
        frame = QFrame()
        frame.setFixedSize(size, size)
        frame.setStyleSheet(
            f"background: {color}; border-radius: {size // 2}px; border: 1px solid #555;"
        )
        return frame

    def set_green(self):
        self.green_light.setStyleSheet(
            "background: #33ff33; border-radius: 15px; border: 1px solid #555;"
        )
        self.red_light.setStyleSheet(
            "background: #6a2d2d; border-radius: 15px; border: 1px solid #555;"
        )

    def set_red(self):
        self.green_light.setStyleSheet(
            "background: #2d6a2d; border-radius: 15px; border: 1px solid #555;"
        )
        self.red_light.setStyleSheet(
            "background: #ff3333; border-radius: 15px; border: 1px solid #555;"
        )

    # ── slots ─────────────────────────────────────────────────────────

    def on_frame(self, qimage: 'QImage'):
        pix = QPixmap.fromImage(qimage)
        scaled = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def on_side_frame(self, qimage: 'QImage'):
        pix = QPixmap.fromImage(qimage)
        scaled = pix.scaled(self.side_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.side_video_label.setPixmap(scaled)

    def on_metrics(self, data: dict):
        cva = data.get('cva_2d', data.get('neck_angle', 0.0))
        good = data['is_good_posture']
        good_t = data['good_time']
        bad_t = data['bad_time']

        # Indicator
        if good:
            self.set_green()
            self.status_label.setText("Правильна постава ✔")
            self.status_label.setStyleSheet("color: #33ff33; font-size: 13px;")
        else:
            self.set_red()
            parts = []
            if data.get('is_leaning'):
                parts.append("Нахил вперед")
            if data.get('is_tilted'):
                parts.append("Перекіс плечей")
            if not parts:
                parts.append("Погана постава")
            self.status_label.setText(" | ".join(parts))
            self.status_label.setStyleSheet("color: #ff4444; font-size: 13px;")

        # Time labels
        self.good_time_label.setText(f"Good: {good_t:.1f} s")
        self.bad_time_label.setText(f"Bad: {bad_t:.1f} s")

        # Chart (in-place redraw)
        self._angle_buf.append(cva)
        n = len(self._angle_buf)
        fps = data.get('fps', 30)
        self._time_buf.append(n / fps)
        xs = list(self._time_buf)
        ys = list(self._angle_buf)
        self.chart_line.set_xdata(xs)
        self.chart_line.set_ydata(ys)
        self.chart_ax.set_xlim(xs[0], xs[-1] + 0.01)
        y_min = min(ys) - 5
        y_max = max(ys) + 5
        self.chart_ax.set_ylim(y_min, y_max)

        # Threshold line
        thresh = self.spin_neck.value()
        if self.chart_threshold_line is None:
            self.chart_threshold_line = self.chart_ax.axhline(
                y=thresh, color='#ff5555', linestyle='--', linewidth=1, alpha=0.7
            )
        else:
            self.chart_threshold_line.set_ydata([thresh, thresh])

        self.chart_canvas.draw_idle()

        # Warning if bad posture > time_threshold
        time_threshold = self.spin_time.value()
        if bad_t > time_threshold and not self._warning_active:
            self._warning_active = True
            send_warning_beep()
            self.window().setStyleSheet(
                self.window().styleSheet() + " QMainWindow { border: 3px solid #ff3333; }"
            )
        elif good:
            self._warning_active = False
            mw = self.window()
            if isinstance(mw, MainWindow):
                mw.setStyleSheet(mw._base_style)


# ═══════════════════════════════════════════════════════════════════════════════
#  Analytics Tab
# ═══════════════════════════════════════════════════════════════════════════════

class AnalyticsTab(QWidget):
    """Tab 2 — placeholder charts for future historical analytics."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        title = QLabel("Аналітика сеансів")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ddd; font-size: 18px; font-weight: bold; padding: 12px;")
        layout.addWidget(title)

        info = QLabel(
            "Тут будуть відображатися історичні дані після підключення бази SQLCipher."
        )
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: #888; font-size: 13px; padding-bottom: 10px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # ── Pie chart placeholder ─────────────────────────────────────────
        # TODO: підключити SQLCipher для зчитування даних сеансів.
        # Цей Canvas буде показувати кругову діаграму:
        #   - відсоток часу з правильною поставою vs поганою.
        # Приклад запиту до БД:
        #   SELECT SUM(good_time), SUM(bad_time) FROM sessions
        #   WHERE date BETWEEN ? AND ?
        pie_group = QGroupBox("Розподіл часу сеансу")
        pie_group.setStyleSheet(
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 6px; "
            "margin-top: 10px; padding-top: 14px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-position: top center; padding: 0 8px; }"
        )
        self.pie_fig = Figure(figsize=(4, 3), dpi=80)
        self.pie_fig.patch.set_facecolor('#1a1a2e')
        self.pie_ax = self.pie_fig.add_subplot(111)
        self.pie_ax.set_facecolor('#16213e')
        self.pie_ax.text(0.5, 0.5, 'Дані відсутні',
                         ha='center', va='center', color='#666', fontsize=14,
                         transform=self.pie_ax.transAxes)
        self.pie_canvas = FigureCanvas(self.pie_fig)
        pie_layout = QVBoxLayout()
        pie_layout.addWidget(self.pie_canvas)
        pie_group.setLayout(pie_layout)
        layout.addWidget(pie_group)

        # ── Bar chart placeholder ─────────────────────────────────────────
        # TODO: підключити SQLCipher для зчитування історії порушень.
        # Цей Canvas буде показувати гістограму:
        #   - кількість порушень постави по днях/годинах.
        # Приклад запиту до БД:
        #   SELECT DATE(timestamp), COUNT(*) FROM violations
        #   GROUP BY DATE(timestamp) ORDER BY DATE(timestamp) DESC LIMIT 30
        bar_group = QGroupBox("Гістограма порушень")
        bar_group.setStyleSheet(
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 6px; "
            "margin-top: 10px; padding-top: 14px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-position: top center; padding: 0 8px; }"
        )
        self.bar_fig = Figure(figsize=(4, 3), dpi=80)
        self.bar_fig.patch.set_facecolor('#1a1a2e')
        self.bar_ax = self.bar_fig.add_subplot(111)
        self.bar_ax.set_facecolor('#16213e')
        self.bar_ax.text(0.5, 0.5, 'Дані відсутні',
                         ha='center', va='center', color='#666', fontsize=14,
                         transform=self.bar_ax.transAxes)
        self.bar_canvas = FigureCanvas(self.bar_fig)
        bar_layout = QVBoxLayout()
        bar_layout.addWidget(self.bar_canvas)
        bar_group.setLayout(bar_layout)
        layout.addWidget(bar_group)

        layout.addStretch()
        self.setLayout(layout)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Window
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """Top-level window with two tabs: Live Dashboard and Analytics."""

    _base_style = """
        QMainWindow { background: #0f0f23; }
        QTabWidget::pane { border: 1px solid #333; background: #0f0f23; }
        QTabBar::tab {
            background: #16213e; color: #aaa; padding: 10px 28px;
            border: 1px solid #333; border-bottom: none; border-radius: 6px 6px 0 0;
            font-size: 13px; min-width: 140px;
        }
        QTabBar::tab:selected { background: #1a1a2e; color: #fff; font-weight: bold; }
        QTabBar::tab:hover { background: #1a3a5c; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Posture Monitor")
        self.resize(1200, 750)
        self.setStyleSheet(self._base_style)

        # ── Tabs ──────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.live_tab = LiveDashboardTab()
        self.analytics_tab = AnalyticsTab()
        self.tabs.addTab(self.live_tab, "Live Dashboard")
        self.tabs.addTab(self.analytics_tab, "Analytics")
        self.setCentralWidget(self.tabs)

        # ── Worker (not started yet) ──────────────────────────────────────
        self.worker = None

        # ── Connect buttons ──────────────────────────────────────────────
        self.live_tab.btn_start.clicked.connect(self._start_monitoring)
        self.live_tab.btn_stop.clicked.connect(self._stop_monitoring)
        # Calibration removed from worker; keep the button disabled.
        self.live_tab.btn_calibrate.setEnabled(False)

        # Push threshold changes to worker
        self.live_tab.spin_neck.valueChanged.connect(self._push_thresholds)
        self.live_tab.spin_tilt.valueChanged.connect(self._push_thresholds)
        self.live_tab.spin_lean.valueChanged.connect(self._push_thresholds)
        self.live_tab.spin_time.valueChanged.connect(self._push_thresholds)

    # ── monitoring lifecycle ─────────────────────────────────────────

    def _start_monitoring(self):
        if self.worker and self.worker.isRunning():
            return

        self.worker = PoseWorker(camera_front_id=0, camera_side_id=1)
        self._push_thresholds()

        self.worker.front_frame_ready.connect(self.live_tab.on_frame)
        self.worker.side_frame_ready.connect(self.live_tab.on_side_frame)
        self.worker.metrics_ready.connect(self.live_tab.on_metrics)
        self.worker.finished.connect(self._on_worker_done)

        self.worker.start()

        self.live_tab.btn_start.setEnabled(False)
        self.live_tab.btn_stop.setEnabled(True)
        self.live_tab.btn_calibrate.setEnabled(True)
        self.live_tab.video_label.setText("")
        self.live_tab.side_video_label.setText("")

    def _stop_monitoring(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(5000)
            self.worker = None

        self.live_tab.btn_start.setEnabled(True)
        self.live_tab.btn_stop.setEnabled(False)
        self.live_tab.btn_calibrate.setEnabled(False)
        self.live_tab.video_label.setText("Моніторинг зупинено")

    def _on_worker_done(self):
        self.live_tab.btn_start.setEnabled(True)
        self.live_tab.btn_stop.setEnabled(False)
        self.live_tab.btn_calibrate.setEnabled(False)

    # ── push GUI thresholds to worker ────────────────────────────────

    def _push_thresholds(self):
        if not self.worker:
            return
        self.worker.set_thresholds(
            medical_cva_threshold=float(self.live_tab.spin_neck.value()),
            shoulder_tilt_threshold=self.live_tab.spin_tilt.value(),
            forward_lean_threshold=self.live_tab.spin_lean.value(),
            time_threshold=self.live_tab.spin_time.value(),
        )

    # ── cleanup ──────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._stop_monitoring()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Dark palette for Fusion
    from PyQt5.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(15, 15, 35))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(22, 33, 62))
    palette.setColor(QPalette.AlternateBase, QColor(26, 26, 46))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(22, 33, 62))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.Highlight, QColor(0, 210, 255))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
