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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QSpinBox, QDoubleSpinBox,
    QFormLayout, QGridLayout,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

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

# ── Pastel color constants ────────────────────────────────────────────────────
PASTEL_GREEN_BG = "#c8f7c5"       # card background when OK
PASTEL_GREEN_BORDER = "#7bc67e"   # card border when OK
PASTEL_GREEN_TEXT = "#2d6a2d"     # card text when OK

PASTEL_RED_BG = "#f7c5c5"         # card background when bad
PASTEL_RED_BORDER = "#c66e6e"     # card border when bad
PASTEL_RED_TEXT = "#8b2222"       # card text when bad

PASTEL_YELLOW_BG = "#fdf3c5"      # card background for warning
PASTEL_YELLOW_BORDER = "#d4b84a"  # card border for warning
PASTEL_YELLOW_TEXT = "#7a6a1a"    # card text for warning

DORMANT_BG = "#1e2a3a"            # card background when idle
DORMANT_BORDER = "#333"           # card border when idle
DORMANT_TEXT = "#888"             # card text when idle


def _card_style(bg, border, text_color):
    return (
        f"background: {bg}; border: 2px solid {border}; border-radius: 10px; "
        f"padding: 10px; color: {text_color}; font-size: 13px; font-weight: bold;"
    )


class LiveDashboardTab(QWidget):
    """Tab 1 — real-time video feed, status cards, controls."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._warning_active = False
        self._calibrating = False

        # ── Video widget ──────────────────────────────────────────────────
        self.video_label = QLabel("Натисніть «Start Monitoring» для початку")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background: #1a1a2e; color: #888; border: 2px solid #333; border-radius: 8px;"
        )

        # ── Side (picture-in-picture) video widget ────────────────────────
        self.side_video_label = QLabel("")
        self.side_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.side_video_label.setMinimumSize(320, 240)
        self.side_video_label.setMaximumSize(360, 270)
        self.side_video_label.setStyleSheet(
            "background: #111827; color: #888; border: 2px solid #333; border-radius: 8px;"
        )

        # ── Status indicator cards ─────────────────────────────────────────
        cards_box = QGroupBox("Стан постави")
        cards_box.setStyleSheet(
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 6px; "
            "margin-top: 10px; padding-top: 18px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-position: top center; padding: 0 8px; }"
        )
        cards_grid = QGridLayout()
        cards_grid.setSpacing(8)

        dormant = _card_style(DORMANT_BG, DORMANT_BORDER, DORMANT_TEXT)

        self.card_posture = QLabel("Постава\n—")
        self.card_posture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.card_posture.setMinimumHeight(62)
        self.card_posture.setStyleSheet(dormant)

        self.card_trunk = QLabel("Нахил тулуба\n—")
        self.card_trunk.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.card_trunk.setMinimumHeight(62)
        self.card_trunk.setStyleSheet(dormant)

        self.card_shoulders = QLabel("Плечі\n—")
        self.card_shoulders.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.card_shoulders.setMinimumHeight(62)
        self.card_shoulders.setStyleSheet(dormant)

        self.card_neck = QLabel("Шия\n—")
        self.card_neck.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.card_neck.setMinimumHeight(62)
        self.card_neck.setStyleSheet(dormant)

        cards_grid.addWidget(self.card_posture, 0, 0)
        cards_grid.addWidget(self.card_trunk, 0, 1)
        cards_grid.addWidget(self.card_shoulders, 1, 0)
        cards_grid.addWidget(self.card_neck, 1, 1)
        cards_box.setLayout(cards_grid)

        # ── Time labels ───────────────────────────────────────────────────
        self.good_time_label = QLabel("✔ 0.0 s")
        self.good_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.good_time_label.setStyleSheet(f"color: {PASTEL_GREEN_TEXT}; background: {PASTEL_GREEN_BG}; border-radius: 6px; padding: 4px; font-size: 12px; font-weight: bold;")
        self.bad_time_label = QLabel("✖ 0.0 s")
        self.bad_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bad_time_label.setStyleSheet(f"color: {PASTEL_RED_TEXT}; background: {PASTEL_RED_BG}; border-radius: 6px; padding: 4px; font-size: 12px; font-weight: bold;")

        time_layout = QHBoxLayout()
        time_layout.setSpacing(8)
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
        self.calibrate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

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
        right_panel.addWidget(cards_box)
        right_panel.addLayout(time_layout)
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

    def _set_card_ok(self, card, label, value_text):
        card.setText(f"{label}\n{value_text}")
        card.setStyleSheet(_card_style(PASTEL_GREEN_BG, PASTEL_GREEN_BORDER, PASTEL_GREEN_TEXT))

    def _set_card_bad(self, card, label, value_text):
        card.setText(f"{label}\n{value_text}")
        card.setStyleSheet(_card_style(PASTEL_RED_BG, PASTEL_RED_BORDER, PASTEL_RED_TEXT))

    def _set_card_warn(self, card, label, value_text):
        card.setText(f"{label}\n{value_text}")
        card.setStyleSheet(_card_style(PASTEL_YELLOW_BG, PASTEL_YELLOW_BORDER, PASTEL_YELLOW_TEXT))

    # ── slots ─────────────────────────────────────────────────────────

    def on_frame(self, qimage: QImage):
        pix = QPixmap.fromImage(qimage)
        scaled = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)  # type: ignore[arg-type]
        self.video_label.setPixmap(scaled)

    def on_side_frame(self, qimage: QImage):
        pix = QPixmap.fromImage(qimage)
        scaled = pix.scaled(self.side_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)  # type: ignore[arg-type]
        self.side_video_label.setPixmap(scaled)

    def on_metrics(self, data: dict):
        cva = data.get('cva_2d') or data.get('neck_angle') or 0.0
        good = data['is_good_posture']
        good_t = data['good_time']
        bad_t = data['bad_time']

        is_leaning = data.get('is_leaning', False)
        is_tilted = data.get('is_tilted', False)
        is_trunk_tilted = data.get('is_trunk_tilted', False)
        is_hunched = data.get('is_hunched', False)
        is_good_neck = cva >= self.spin_neck.value()
        shoulder_tilt = data.get('shoulder_tilt', 0)
        shoulder_depth = data.get('shoulder_depth', 0)
        trunk_tilt_deg = data.get('trunk_tilt_deg')
        hunch_ratio = data.get('hunch_ratio')

        # ── Card: Постава (overall) ──────────────────────────────────
        if good:
            self._set_card_ok(self.card_posture, "Постава", "Правильна ✔")
        else:
            self._set_card_bad(self.card_posture, "Постава", "Погана ✖")

        # ── Card: Нахил тулуба ───────────────────────────────────────
        if is_leaning:
            self._set_card_bad(self.card_trunk, "Нахил тулуба", f"Нахил вперед ({shoulder_depth:+.2f} m)")
        elif is_trunk_tilted:
            self._set_card_warn(self.card_trunk, "Нахил тулуба", f"Бічний ({trunk_tilt_deg:.1f}°)" if trunk_tilt_deg else "Бічний нахил")
        else:
            self._set_card_ok(self.card_trunk, "Нахил тулуба", "Норма ✔")

        # ── Card: Плечі ──────────────────────────────────────────────
        if is_tilted:
            self._set_card_bad(self.card_shoulders, "Плечі", f"Перекіс ({shoulder_tilt*100:.1f} см)")
        elif is_hunched:
            self._set_card_warn(self.card_shoulders, "Плечі", f"Підняті ({hunch_ratio:.2f})" if hunch_ratio else "Підняті")
        else:
            self._set_card_ok(self.card_shoulders, "Плечі", "Норма ✔")

        # ── Card: Шия ────────────────────────────────────────────────
        if not is_good_neck:
            self._set_card_bad(self.card_neck, "Шия", f"Висунута ({cva:.1f}°)")
        else:
            self._set_card_ok(self.card_neck, "Шия", f"Норма ({cva:.1f}°)")

        # Time labels
        self.good_time_label.setText(f"✔ {good_t:.1f} s")
        self.bad_time_label.setText(f"✖ {bad_t:.1f} s")

        # Warning if bad posture > time_threshold
        time_threshold = self.spin_time.value()
        if bad_t > time_threshold and not self._warning_active:
            self._warning_active = True
            send_warning_beep()
            mw = self.window()
            if mw is not None:
                mw.setStyleSheet(
                    mw.styleSheet() + " QMainWindow { border: 3px solid #ff3333; }"
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
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #ddd; font-size: 18px; font-weight: bold; padding: 12px;")
        layout.addWidget(title)

        info = QLabel(
            "Тут будуть відображатися історичні дані після підключення бази SQLCipher."
        )
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet("color: #888; font-size: 13px; padding-bottom: 10px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # ── Pie chart placeholder ─────────────────────────────────────────
        # TODO: підключити SQLCipher для зчитування даних сеансів.
        pie_group = QGroupBox("Розподіл часу сеансу")
        pie_group.setStyleSheet(
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 6px; "
            "margin-top: 10px; padding-top: 14px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-position: top center; padding: 0 8px; }"
        )
        pie_placeholder = QLabel("Дані відсутні")
        pie_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pie_placeholder.setMinimumHeight(150)
        pie_placeholder.setStyleSheet("color: #666; font-size: 14px; background: #16213e; border-radius: 6px;")
        pie_layout = QVBoxLayout()
        pie_layout.addWidget(pie_placeholder)
        pie_group.setLayout(pie_layout)
        layout.addWidget(pie_group)

        # ── Bar chart placeholder ─────────────────────────────────────────
        # TODO: підключити SQLCipher для зчитування історії порушень.
        bar_group = QGroupBox("Гістограма порушень")
        bar_group.setStyleSheet(
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 6px; "
            "margin-top: 10px; padding-top: 14px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-position: top center; padding: 0 8px; }"
        )
        bar_placeholder = QLabel("Дані відсутні")
        bar_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar_placeholder.setMinimumHeight(150)
        bar_placeholder.setStyleSheet("color: #666; font-size: 14px; background: #16213e; border-radius: 6px;")
        bar_layout = QVBoxLayout()
        bar_layout.addWidget(bar_placeholder)
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
        self.live_tab.btn_start.clicked.connect(self._start_monitoring)  # type: ignore[attr-defined]
        self.live_tab.btn_stop.clicked.connect(self._stop_monitoring)  # type: ignore[attr-defined]
        # Calibration removed from worker; keep the button disabled.
        self.live_tab.btn_calibrate.setEnabled(False)

        # Push threshold changes to worker
        self.live_tab.spin_neck.valueChanged.connect(self._push_thresholds)  # type: ignore[attr-defined]
        self.live_tab.spin_tilt.valueChanged.connect(self._push_thresholds)  # type: ignore[attr-defined]
        self.live_tab.spin_lean.valueChanged.connect(self._push_thresholds)  # type: ignore[attr-defined]
        self.live_tab.spin_time.valueChanged.connect(self._push_thresholds)  # type: ignore[attr-defined]

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

    def closeEvent(self, a0):
        self._stop_monitoring()
        a0.accept()


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
