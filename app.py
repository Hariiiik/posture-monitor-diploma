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

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QSpinBox, QDoubleSpinBox,
    QFormLayout, QGridLayout, QLineEdit, QMessageBox,
    QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

from worker import PoseWorker
from database import DatabaseManager

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
        self.btn_stop.setEnabled(False)
        for b in (self.btn_start, self.btn_stop):
            b.setStyleSheet(btn_style)

        self.input_cam_front = QLineEdit("0")
        self.input_cam_front.setPlaceholderText("Фронтальна (ID або URL)")
        self.input_cam_front.setStyleSheet("background: #16213e; color: #ddd; border: 1px solid #555; border-radius: 4px; padding: 4px;")

        self.input_cam_side = QLineEdit("1")
        self.input_cam_side.setPlaceholderText("Бокова (ID або URL)")
        self.input_cam_side.setStyleSheet("background: #16213e; color: #ddd; border: 1px solid #555; border-radius: 4px; padding: 4px;")

        cam_layout = QFormLayout()
        cam_layout.addRow("Фронтальна:", self.input_cam_front)
        cam_layout.addRow("Бокова:", self.input_cam_side)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_v = QVBoxLayout()
        ctrl_v.addLayout(cam_layout)
        ctrl_v.addLayout(ctrl_layout)
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
        self.spin_tilt.setValue(0.03)
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

        is_leaning = data.get('is_leaning_smooth', data.get('is_leaning', False))
        is_tilted = data.get('is_tilted_smooth', data.get('is_tilted', False))
        is_trunk_tilted = data.get('is_trunk_tilted_smooth', data.get('is_trunk_tilted', False))
        is_hunched = data.get('is_hunched_smooth', data.get('is_hunched', False))
        
        is_bad_neck_smooth = data.get('is_bad_neck_smooth')
        if is_bad_neck_smooth is not None:
            is_good_neck = not is_bad_neck_smooth
        else:
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
    """Tab 2 — Matplotlib charts for session history & posture events."""

    # Ukrainian labels for event types
    _EVENT_LABELS = {
        'bad_neck':    'Висунута шия',
        'bad_lean':    'Нахил',
        'bad_tilt':    'Перекіс плечей',
        'bad_hunch':   'Сутулість',
        'bad_trunk':   'Бічний нахил',
        'bad_posture': 'Погана постава',
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)

        # ── Title ───────────────────────────────────────────────────────
        title = QLabel("Аналітика сеансів")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "color: #ddd; font-size: 18px; font-weight: bold; padding: 8px;"
        )
        layout.addWidget(title)

        # ── Matplotlib figure (dark-themed, two subplots) ───────────────
        self._fig = Figure(figsize=(10, 4), dpi=100, facecolor='#0f0f23')
        self._ax_pie = self._fig.add_subplot(1, 2, 1)
        self._ax_bar = self._fig.add_subplot(1, 2, 2)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setStyleSheet("background: #0f0f23; border-radius: 8px;")
        self._canvas.setMinimumHeight(340)
        layout.addWidget(self._canvas, stretch=1)

        # ── Refresh button ──────────────────────────────────────────────
        btn_refresh = QPushButton("⊕  Оновити дані")
        btn_refresh.setStyleSheet(
            "QPushButton { background: #16213e; color: #ddd; border: 1px solid #555; "
            "border-radius: 6px; padding: 8px 22px; font-size: 13px; }"
            "QPushButton:hover { background: #1a3a5c; }"
        )
        btn_refresh.setMaximumWidth(200)
        btn_refresh.clicked.connect(self.refresh_data)  # type: ignore[attr-defined]

        # ── Session Selector ────────────────────────────────────────────
        self.session_combo = QComboBox()
        self.session_combo.setStyleSheet(
            "QComboBox { background: #16213e; color: #ddd; border: 1px solid #555; "
            "border-radius: 4px; padding: 6px; font-size: 13px; min-width: 220px; }"
            "QComboBox QAbstractItemView { background: #16213e; color: #ddd; selection-background-color: #1a3a5c; }"
        )
        self.session_combo.currentIndexChanged.connect(self._on_session_changed)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        
        lbl = QLabel("Оберіть сесію:")
        lbl.setStyleSheet("color: #ddd; font-weight: bold; font-size: 13px;")
        btn_row.addWidget(lbl)
        btn_row.addWidget(self.session_combo)
        btn_row.addSpacing(10)
        btn_row.addWidget(btn_refresh)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch()
        self.setLayout(layout)

        # Draw empty state
        self._draw_empty()
        
        # Load sessions once
        self.load_sessions()
        self._load_analytics_for_current()

    # ── data loading ─────────────────────────────────────────────────────

    def _on_session_changed(self, index):
        self._load_analytics_for_current()

    def load_sessions(self):
        """Load available sessions into combo box."""
        current_data = self.session_combo.currentData() if self.session_combo.count() > 0 else None
        
        try:
            db = DatabaseManager()
            sessions = db.get_all_sessions()
            db.close()
            
            self.session_combo.blockSignals(True)
            self.session_combo.clear()
            self.session_combo.addItem("Всі сесії", None)
            
            idx_to_select = 0
            for i, s in enumerate(sessions):
                sid = s['session_id']
                start = s['start_time']
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(start)
                    start_str = dt.strftime("%d.%m.%Y %H:%M:%S")
                except Exception:
                    start_str = str(start)
                    
                self.session_combo.addItem(f"Сесія {sid}: {start_str}", sid)
                if sid == current_data:
                    idx_to_select = i + 1
                    
            self.session_combo.setCurrentIndex(idx_to_select)
            self.session_combo.blockSignals(False)
        except Exception as exc:
            print(f"[AnalyticsTab] Failed to load sessions: {exc}")

    def refresh_data(self):
        """Query the database and redraw both charts."""
        self.load_sessions()
        self._load_analytics_for_current()

    def _load_analytics_for_current(self):
        """Load and display analytics for the currently selected session."""
        session_id = self.session_combo.currentData() if self.session_combo.count() > 0 else None
        try:
            db = DatabaseManager()
            summary = db.get_event_summary(session_id)
            time_ratio = db.get_posture_time_ratio(session_id)
            db.close()
        except Exception as exc:
            print(f"[AnalyticsTab] DB query failed: {exc}")
            self._draw_empty()
            return

        has_time = (time_ratio['good_time'] + time_ratio['bad_time']) > 0
        has_events = bool(summary)

        if not has_time and not has_events:
            self._draw_empty()
            return

        self._draw_charts(time_ratio, summary)

    # ── drawing ──────────────────────────────────────────────────────────

    def _style_ax(self, ax, title: str):
        ax.set_facecolor('#16213e')
        ax.set_title(title, color='#ddd', fontsize=13, fontweight='bold', pad=10)
        ax.tick_params(colors='#aaa', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#333')

    def _draw_empty(self):
        for ax in (self._ax_pie, self._ax_bar):
            ax.clear()
            ax.set_facecolor('#16213e')
            ax.text(
                0.5, 0.5, 'Дані відсутні',
                ha='center', va='center', color='#666',
                fontsize=14, transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        self._fig.tight_layout(pad=2.0)
        self._canvas.draw()

    def _draw_charts(self, time_ratio: dict, summary: dict):
        # ── PIE: good vs bad time ────────────────────────────────────
        self._ax_pie.clear()
        self._style_ax(self._ax_pie, 'Розподіл часу')

        good = time_ratio['good_time']
        bad = time_ratio['bad_time']
        total = good + bad

        if total > 0:
            sizes = [good, bad]
            labels = [f'Правильна\n{good/60:.1f} хв', f'Погана\n{bad/60:.1f} хв']
            colors = ['#7bc67e', '#c66e6e']
            explode = (0.03, 0.03)
            wedges, texts, autotexts = self._ax_pie.pie(  # type: ignore[misc]
                sizes, labels=labels, colors=colors, explode=explode,
                autopct='%1.1f%%', startangle=90,
                textprops={'color': '#ddd', 'fontsize': 10},
                pctdistance=0.55,
            )
            for at in autotexts:
                at.set_color('#fff')
                at.set_fontsize(11)
                at.set_fontweight('bold')
        else:
            self._ax_pie.text(
                0.5, 0.5, 'Дані відсутні',
                ha='center', va='center', color='#666',
                fontsize=14, transform=self._ax_pie.transAxes,
            )

        # ── BAR: event counts by type ────────────────────────────────
        self._ax_bar.clear()
        self._style_ax(self._ax_bar, 'Кількість порушень')

        if summary:
            # Ensure consistent ordering
            db_types = ['bad_neck', 'bad_lean', 'bad_tilt', 'bad_hunch', 'bad_trunk']
            types = [t for t in db_types if t in summary]
            counts = [summary[t] for t in types]

            # Compute bad_posture as total of all violations (not stored in DB)
            total_violations = sum(summary.values())
            types.append('bad_posture')
            counts.append(total_violations)

            labels = [self._EVENT_LABELS.get(t, t) for t in types]
            all_types = ['bad_neck', 'bad_lean', 'bad_tilt', 'bad_hunch', 'bad_trunk', 'bad_posture']
            bar_colors = ['#e07a5f', '#f2cc8f', '#81b29a', '#3d405b', '#b07d62', '#c66e6e']
            used_colors = [bar_colors[all_types.index(t)] for t in types]

            bars = self._ax_bar.bar(
                range(len(labels)), counts, color=used_colors,
                edgecolor='#333', linewidth=0.8, width=0.6,
            )
            self._ax_bar.set_xticks([])

            # Value labels on top of each bar
            for bar, count in zip(bars, counts):
                self._ax_bar.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(count), ha='center', va='bottom',
                    color='#ddd', fontsize=11, fontweight='bold',
                )
            self._ax_bar.set_ylabel('Кількість', color='#aaa', fontsize=11)
            
            # Add legend
            self._ax_bar.legend(bars, labels, facecolor='#16213e', edgecolor='#555', labelcolor='#ddd', fontsize=10, loc='best')

            import matplotlib.ticker as mticker
            self._ax_bar.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        else:
            self._ax_bar.text(
                0.5, 0.5, 'Порушень не зафіксовано',
                ha='center', va='center', color='#666',
                fontsize=14, transform=self._ax_bar.transAxes,
            )

        self._fig.tight_layout(pad=2.0)
        self._canvas.draw()


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

        # ── Tab switch → auto-refresh analytics ───────────────────────
        self.tabs.currentChanged.connect(self._on_tab_changed)  # type: ignore[attr-defined]

        # ── Worker (not started yet) ──────────────────────────────────────
        self.worker = None

        # ── Connect buttons ──────────────────────────────────────────────
        self.live_tab.btn_start.clicked.connect(self._start_monitoring)  # type: ignore[attr-defined]
        self.live_tab.btn_stop.clicked.connect(self._stop_monitoring)  # type: ignore[attr-defined]

        # Push threshold changes to worker
        self.live_tab.spin_neck.valueChanged.connect(self._push_thresholds)  # type: ignore[attr-defined]
        self.live_tab.spin_tilt.valueChanged.connect(self._push_thresholds)  # type: ignore[attr-defined]
        self.live_tab.spin_lean.valueChanged.connect(self._push_thresholds)  # type: ignore[attr-defined]
        self.live_tab.spin_time.valueChanged.connect(self._push_thresholds)  # type: ignore[attr-defined]

    # ── monitoring lifecycle ─────────────────────────────────────────

    def _start_monitoring(self):
        if self.worker and self.worker.isRunning():
            return

        front_val = self.live_tab.input_cam_front.text().strip()
        side_val = self.live_tab.input_cam_side.text().strip()
        
        cam_front = int(front_val) if front_val.isdigit() else front_val
        cam_side = int(side_val) if side_val.isdigit() else side_val

        self.worker = PoseWorker(
            camera_front_id=cam_front, 
            camera_side_id=cam_side, 
            front_video_path=None, 
            side_video_path=None
        )
        self._push_thresholds()

        self.worker.front_frame_ready.connect(self.live_tab.on_frame)
        self.worker.side_frame_ready.connect(self.live_tab.on_side_frame)
        self.worker.metrics_ready.connect(self.live_tab.on_metrics)
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.session_saved.connect(self.analytics_tab.refresh_data)

        self.worker.start()

        self.live_tab.btn_start.setEnabled(False)
        self.live_tab.btn_stop.setEnabled(True)
        self.live_tab.input_cam_front.setEnabled(False)
        self.live_tab.input_cam_side.setEnabled(False)
        self.live_tab.video_label.setText("")
        self.live_tab.side_video_label.setText("")

    def _on_worker_error(self, message: str):
        QMessageBox.critical(self, "Помилка підключення", message)
        self._stop_monitoring()

    def _stop_monitoring(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(5000)
            self.worker = None

        self.live_tab.btn_start.setEnabled(True)
        self.live_tab.btn_stop.setEnabled(False)
        self.live_tab.input_cam_front.setEnabled(True)
        self.live_tab.input_cam_side.setEnabled(True)
        self.live_tab.video_label.setText("Моніторинг зупинено")

    def _on_worker_done(self):
        self.live_tab.btn_start.setEnabled(True)
        self.live_tab.btn_stop.setEnabled(False)
        self.live_tab.input_cam_front.setEnabled(True)
        self.live_tab.input_cam_side.setEnabled(True)

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

    def _on_tab_changed(self, index: int):
        """Auto-refresh analytics when the user switches to that tab."""
        if index == 1:  # Analytics tab
            self.analytics_tab.refresh_data()


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
