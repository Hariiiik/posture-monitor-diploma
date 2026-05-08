"""
PoseWorker — QThread that captures frames from OpenCV, runs MediaPipe
pose detection, applies OneEuroFilter smoothing, and emits processed
frames + metrics to the GUI thread via pyqtSignal.
"""

import cv2
import math as m
import os

# Приховуємо спам від FFmpeg (повідомлення 'Stream ends prematurely')
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import time
import datetime
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QImage

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from one_euro_filter import OneEuroFilter

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")

# Landmark indices (MediaPipe Pose)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_HIP = 23
NOSE = 0


def _find_distance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _find_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dy == 0:
        return 90.0
    return m.degrees(m.atan2(abs(dx), abs(dy)))


def _draw_unicode_batch(img, items):
    try:
        from PIL import Image as PilImage, ImageDraw, ImageFont
        rgb_img = PilImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(rgb_img)
        for pos, text, font_scale, bgr in items:
            font_px = max(16, int(font_scale * 30))
            try:
                fnt = ImageFont.truetype("arial.ttf", font_px)
            except OSError:
                fnt = ImageFont.load_default()
            b, g, r_ch = bgr
            draw.text(pos, text, font=fnt, fill=(r_ch, g, b))
        img[:] = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
    except ImportError:
        for pos, text, font_scale, bgr in items:
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, bgr, 2)


class PoseWorker(QThread):
    """
    Worker thread: camera capture → MediaPipe → skeleton overlay → signals.

    Signals
    -------
    frame_ready(QImage)
        BGR frame with skeleton drawn, converted to QImage (RGB888).
    metrics_ready(dict)
        Dictionary with keys: neck_angle, shoulder_tilt, shoulder_depth,
        is_good_posture, good_time, bad_time, is_leaning, is_tilted, fps.
    """

    front_frame_ready = pyqtSignal(QImage)
    side_frame_ready = pyqtSignal(QImage)
    metrics_ready = pyqtSignal(dict)
    session_saved = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        video_path=None,
        parent=None,
        camera_front_id: int | str = 0,
        camera_side_id: int | str = 1,
        front_video_path: int | str | None = 0,
        side_video_path: int | str | None = "http://192.168.1.4:8080/video",
    ):
        super().__init__(parent)
        # Back-compat: video_path is treated as front input if provided.
        self.video_path = video_path
        self.camera_front_id = camera_front_id
        self.camera_side_id = camera_side_id
        self.front_video_path = front_video_path
        self.side_video_path = side_video_path
        self._running = False

        # Thresholds (can be updated from GUI via set_thresholds)
        self.offset_threshold = 0.3
        self.medical_cva_threshold = 50.0
        self.trunk_tilt_threshold_deg = 12.0
        self.hunch_ratio_threshold = 0.55
        self.bad_pose_timeout_s = 10.0
        self.shoulder_tilt_threshold = 0.05
        self.forward_lean_threshold = 0.10
        self.time_threshold = 180
        self.filter_min_cutoff = 0.004
        self.filter_beta = 0.7
        self._mutex = QMutex()

        # ── Event accumulation state ─────────────────────────────────
        self._session_start_time: float | None = None
        self._pending_events: list[dict] = []
        self._cva_accumulator: list[float] = []

        # Per-condition independent 10-second timers
        # Stores (monotonic_time, epoch_time) tuples for correct ISO conversion
        self._bad_neck_since: tuple[float, float] | None = None
        self._bad_lean_since: tuple[float, float] | None = None
        self._bad_tilt_since: tuple[float, float] | None = None
        self._bad_hunch_since: tuple[float, float] | None = None
        self._bad_trunk_since: tuple[float, float] | None = None

        # Grace-period: monotonic time when condition last went False
        # (absorbs sensor noise flickers for up to _GRACE_S seconds)
        self._bad_neck_grace: float | None = None
        self._bad_lean_grace: float | None = None
        self._bad_tilt_grace: float | None = None
        self._bad_hunch_grace: float | None = None
        self._bad_trunk_grace: float | None = None

        # Max deviation accumulators (reset per event)
        self._max_neck_dev: float = 0.0
        self._max_lean_dev: float = 0.0
        self._max_tilt_dev: float = 0.0
        self._max_hunch_dev: float = 0.0
        self._max_trunk_dev: float = 0.0

        self._EVENT_HOLD_S = 10.0  # min seconds of continuous bad to qualify as event
        self._GRACE_S = 2.0       # seconds to tolerate flicker before resetting timer

    # ── public API (called from GUI thread) ──────────────────────────

    def set_thresholds(self, **kw):
        self._mutex.lock()
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._mutex.unlock()

    def stop(self):
        self._running = False

    # ── thread entry point ───────────────────────────────────────────

    def run(self):
        self._running = True
        self._session_start_time = time.time()
        self._pending_events = []
        self._cva_accumulator = []
        self._bad_neck_since = None
        self._bad_lean_since = None
        self._bad_tilt_since = None
        self._bad_hunch_since = None
        self._bad_trunk_since = None
        self._bad_neck_grace = None
        self._bad_lean_grace = None
        self._bad_tilt_grace = None
        self._bad_hunch_grace = None
        self._bad_trunk_grace = None
        self._max_neck_dev = 0.0
        self._max_lean_dev = 0.0
        self._max_tilt_dev = 0.0
        self._max_hunch_dev = 0.0
        self._max_trunk_dev = 0.0

        # Colors
        green = (127, 255, 0)
        red = (50, 50, 255)
        yellow = (0, 255, 255)
        white = (255, 255, 255)
        pink = (255, 0, 255)
        light_green = (127, 233, 100)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # MediaPipe init
        base_options = mp_python.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
        )
        landmarker_front = mp_vision.PoseLandmarker.create_from_options(options)
        landmarker_side = mp_vision.PoseLandmarker.create_from_options(options)

        # VideoCapture init (front + side)
        front_src = (
            self.front_video_path
            if self.front_video_path is not None
            else (self.video_path if self.video_path is not None else self.camera_front_id)
        )
        side_src = self.side_video_path if self.side_video_path is not None else self.camera_side_id

        def create_capture(src):
            # Якщо це системна камера (int) на Windows, використовуємо DSHOW (менше лагів)
            if isinstance(src, int) and os.name == 'nt':
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(src)
            # Вимикаємо буферизацію старих кадрів, щоб завжди отримувати найсвіжіший
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        cap_front = create_capture(front_src)
        cap_side = create_capture(side_src)

        if not cap_front.isOpened() or not cap_side.isOpened():
            self.error_occurred.emit("Не вдалося підключитися до обох камер.\nЗастосунок не може бути використаний.")
            cap_front.release()
            cap_side.release()
            self._running = False
            return

        fps_front = cap_front.get(cv2.CAP_PROP_FPS) or 30.0
        fps_side = cap_side.get(cv2.CAP_PROP_FPS) or 30.0

        t0 = time.monotonic()
        good_frames = 0
        bad_frames = 0

        cva_filter = None
        trunk_filter = None
        hunch_filter = None
        tilt_filter = None
        depth_filter = None
        bad_since_s: float | None = None
        bad_grace_s: float | None = None

        while self._running:
            ok_front, image_front = cap_front.read()
            ok_side, image_side = cap_side.read()
            if not ok_front or not ok_side:
                if good_frames == 0 and bad_frames == 0:
                    self.error_occurred.emit("Не вдалося отримати відео з камер.\nПеревірте підключення.")
                break

            live_fps_front = cap_front.get(cv2.CAP_PROP_FPS)
            if live_fps_front and live_fps_front > 0:
                fps_front = live_fps_front
            live_fps_side = cap_side.get(cv2.CAP_PROP_FPS)
            if live_fps_side and live_fps_side > 0:
                fps_side = live_fps_side

            hf, wf = image_front.shape[:2]
            hs, ws = image_side.shape[:2]

            front_rgb = cv2.cvtColor(image_front, cv2.COLOR_BGR2RGB)
            side_rgb = cv2.cvtColor(image_side, cv2.COLOR_BGR2RGB)
            mp_front = mp.Image(image_format=mp.ImageFormat.SRGB, data=front_rgb)
            mp_side = mp.Image(image_format=mp.ImageFormat.SRGB, data=side_rgb)

            timestamp_ms = int((time.monotonic() - t0) * 1000.0)

            res_front = landmarker_front.detect_for_video(mp_front, timestamp_ms)
            res_side = landmarker_side.detect_for_video(mp_side, timestamp_ms)

            if (
                (not res_front.pose_world_landmarks or not res_front.pose_landmarks)
                or (not res_side.pose_landmarks)
            ):
                self._emit_front_frame(image_front, hf, wf)
                self._emit_side_frame(image_side, hs, ws)
                continue

            lm_world_f = res_front.pose_world_landmarks[0]
            lm_norm_f = res_front.pose_landmarks[0]
            lm_norm_s = res_side.pose_landmarks[0]

            def px(idx):
                return int(lm_norm_f[idx].x * wf), int(lm_norm_f[idx].y * hf)

            def px_side(idx):
                return int(lm_norm_s[idx].x * ws), int(lm_norm_s[idx].y * hs)

            LS = LEFT_SHOULDER
            RS = RIGHT_SHOULDER
            LE = LEFT_EAR
            RE = RIGHT_EAR
            NS = NOSE

            shoulder_depth = (lm_world_f[LS].z + lm_world_f[RS].z) / 2.0
            shoulder_tilt = abs(lm_world_f[LS].y - lm_world_f[RS].y)

            raw_cva_2d = self._calculate_2d_cva(lm_norm_s, ws, hs)
            if raw_cva_2d is None:
                self._emit_front_frame(image_front, hf, wf)
                self._emit_side_frame(image_side, hs, ws)
                continue

            # Trunk tilt (frontal): angle between vertical axis and vector from mid-shoulders -> nose
            trunk_tilt_deg = self._calculate_trunk_tilt_deg(lm_norm_f, wf, hf)
            # Hunch ratio (frontal): normalized vertical distance ear->shoulder by shoulder width
            hunch_ratio = self._calculate_hunch_ratio(lm_norm_f, wf, hf)

            # One Euro Filter
            t = timestamp_ms / 1000.0
            if cva_filter is None:
                cva_filter = OneEuroFilter(
                    t,
                    raw_cva_2d,
                    min_cutoff=self.filter_min_cutoff,
                    beta=self.filter_beta,
                )
                if trunk_tilt_deg is not None:
                    trunk_filter = OneEuroFilter(
                        t,
                        trunk_tilt_deg,
                        min_cutoff=self.filter_min_cutoff,
                        beta=self.filter_beta,
                    )
                if hunch_ratio is not None:
                    hunch_filter = OneEuroFilter(
                        t,
                        hunch_ratio,
                        min_cutoff=self.filter_min_cutoff,
                        beta=self.filter_beta,
                    )
                tilt_filter = OneEuroFilter(t, shoulder_tilt,
                                            min_cutoff=self.filter_min_cutoff, beta=self.filter_beta)
                depth_filter = OneEuroFilter(t, shoulder_depth,
                                             min_cutoff=self.filter_min_cutoff, beta=self.filter_beta)
            else:
                raw_cva_2d = float(cva_filter(t, raw_cva_2d))
                if trunk_tilt_deg is not None and trunk_filter is not None:
                    trunk_tilt_deg = float(trunk_filter(t, trunk_tilt_deg))
                if hunch_ratio is not None and hunch_filter is not None:
                    hunch_ratio = float(hunch_filter(t, hunch_ratio))
                if tilt_filter is not None:
                    shoulder_tilt = float(tilt_filter(t, shoulder_tilt))
                if depth_filter is not None:
                    shoulder_depth = float(depth_filter(t, shoulder_depth))

            # Read thresholds
            self._mutex.lock()
            cva_limit = self.medical_cva_threshold
            trunk_limit = self.trunk_tilt_threshold_deg
            hunch_limit = self.hunch_ratio_threshold
            bad_timeout = self.bad_pose_timeout_s
            fl = self.forward_lean_threshold
            st = self.shoulder_tilt_threshold
            tt = self.time_threshold
            self._mutex.unlock()

            is_good_neck = raw_cva_2d >= cva_limit
            is_leaning = shoulder_depth < -fl
            is_tilted = shoulder_tilt > st
            is_trunk_tilted = (trunk_tilt_deg is not None) and (trunk_tilt_deg > trunk_limit)
            is_hunched = (hunch_ratio is not None) and (hunch_ratio < hunch_limit)

            # Instant (raw) assessment
            is_bad_instant = (not is_good_neck) or is_leaning or is_tilted or is_trunk_tilted or is_hunched

            now_s = time.monotonic()
            if is_bad_instant:
                bad_grace_s = None
                if bad_since_s is None:
                    bad_since_s = now_s
            else:
                if bad_since_s is not None:
                    if bad_grace_s is None:
                        bad_grace_s = now_s
                    elif (now_s - bad_grace_s) >= self._GRACE_S:
                        bad_since_s = None
                        bad_grace_s = None

            if bad_since_s is not None:
                if bad_grace_s is not None:
                    bad_elapsed_s = bad_grace_s - bad_since_s
                else:
                    bad_elapsed_s = now_s - bad_since_s
            else:
                bad_elapsed_s = 0.0

            is_bad_confirmed = (bad_since_s is not None) and (bad_elapsed_s >= bad_timeout)
            is_good_posture = not is_bad_confirmed

            if is_good_posture:
                good_frames += 1
                bad_frames = 0
            else:
                bad_frames += 1
                good_frames = 0

            fps = min(fps_front, fps_side) or 30.0
            good_time = (1 / fps) * good_frames
            bad_time = (1 / fps) * bad_frames

            # ── Draw skeleton ──────────────────────────────────────
            skel_color = green if is_good_posture else red
            l_s = px(LS); r_s = px(RS)
            l_e = px(LE); r_e = px(RE)

            cv2.circle(image_front, l_s, 7, white, 2)
            cv2.circle(image_front, r_s, 7, pink, -1)
            cv2.line(image_front, l_s, r_s, skel_color, 2)

            hunch_color = red if is_hunched else green
            cv2.circle(image_front, l_e, 5, yellow, -1)
            cv2.circle(image_front, r_e, 5, yellow, -1)
            cv2.line(image_front, l_e, l_s, hunch_color, 2)
            cv2.line(image_front, r_e, r_s, hunch_color, 2)

            # Side overlay (skeleton points only)
            side_pts = self._get_side_cva_points(lm_norm_s, ws, hs)
            if side_pts is not None:
                (ear_x, ear_y), (c7_x, c7_y) = side_pts
                ear_px = (int(ear_x), int(ear_y))
                c7_px = (int(c7_x), int(c7_y))
                cv2.circle(image_side, c7_px, 6, (0, 255, 255), -1)   # C7 proxy (shoulder)
                cv2.circle(image_side, ear_px, 6, (255, 255, 255), 2)  # Ear
                cv2.line(image_side, c7_px, ear_px, skel_color, 2)
                cv2.line(image_side, c7_px, (c7_px[0] + 120, c7_px[1]), (180, 180, 180), 2)

            # ── Accumulate CVA for session average ──────────────────
            self._cva_accumulator.append(raw_cva_2d)

            # ── Per-condition event detection (10 s hold) ─────────────
            epoch_now = time.time()
            self._check_event_condition(
                is_bad=not is_good_neck,
                event_type='bad_neck',
                deviation=abs(cva_limit - raw_cva_2d),
                mono_now=now_s,
                epoch_now=epoch_now,
            )
            self._check_event_condition(
                is_bad=is_leaning,
                event_type='bad_lean',
                deviation=abs(shoulder_depth),
                mono_now=now_s,
                epoch_now=epoch_now,
            )
            self._check_event_condition(
                is_bad=is_tilted,
                event_type='bad_tilt',
                deviation=shoulder_tilt,
                mono_now=now_s,
                epoch_now=epoch_now,
            )
            self._check_event_condition(
                is_bad=is_hunched,
                event_type='bad_hunch',
                deviation=abs(hunch_limit - hunch_ratio) if hunch_ratio is not None else 0.0,
                mono_now=now_s,
                epoch_now=epoch_now,
            )
            self._check_event_condition(
                is_bad=is_trunk_tilted,
                event_type='bad_trunk',
                deviation=abs(trunk_limit - trunk_tilt_deg) if trunk_tilt_deg is not None else 0.0,
                mono_now=now_s,
                epoch_now=epoch_now,
            )

            smooth_bad_neck = self._bad_neck_since is not None
            smooth_bad_lean = self._bad_lean_since is not None
            smooth_bad_tilt = self._bad_tilt_since is not None
            smooth_bad_hunch = self._bad_hunch_since is not None
            smooth_bad_trunk = self._bad_trunk_since is not None

            self._emit_front_frame(image_front, hf, wf)
            self._emit_side_frame(image_side, hs, ws)
            self.metrics_ready.emit({
                'neck_angle': raw_cva_2d,  # kept for back-compat with GUI
                'cva_2d': raw_cva_2d,
                'raw_cva_2d': raw_cva_2d,
                'cva_threshold': cva_limit,
                'trunk_tilt_deg': trunk_tilt_deg,
                'trunk_tilt_threshold': trunk_limit,
                'is_trunk_tilted': is_trunk_tilted,
                'hunch_ratio': hunch_ratio,
                'hunch_ratio_threshold': hunch_limit,
                'is_hunched': is_hunched,
                'is_bad_instant': is_bad_instant,
                'is_bad_confirmed': is_bad_confirmed,
                'bad_elapsed_s': bad_elapsed_s,
                'bad_timeout_s': bad_timeout,
                'shoulder_tilt': shoulder_tilt,
                'shoulder_depth': shoulder_depth,
                'is_good_posture': is_good_posture,
                'is_leaning': is_leaning,
                'is_tilted': is_tilted,
                'is_bad_neck_smooth': smooth_bad_neck,
                'is_leaning_smooth': smooth_bad_lean,
                'is_tilted_smooth': smooth_bad_tilt,
                'is_trunk_tilted_smooth': smooth_bad_trunk,
                'is_hunched_smooth': smooth_bad_hunch,
                'good_time': good_time,
                'bad_time': bad_time,
                'fps': fps,
            })

        landmarker_front.close()
        landmarker_side.close()
        cap_front.release()
        cap_side.release()

        # ── Flush accumulated data to database ───────────────────────
        self._flush_session_to_db()

    # ── Event accumulation helpers ────────────────────────────────────

    _TIMER_ATTR_MAP = {
        'bad_neck':  ('_bad_neck_since',  '_max_neck_dev',  '_bad_neck_grace'),
        'bad_lean':  ('_bad_lean_since',  '_max_lean_dev',  '_bad_lean_grace'),
        'bad_tilt':  ('_bad_tilt_since',  '_max_tilt_dev',  '_bad_tilt_grace'),
        'bad_hunch': ('_bad_hunch_since', '_max_hunch_dev', '_bad_hunch_grace'),
        'bad_trunk': ('_bad_trunk_since', '_max_trunk_dev', '_bad_trunk_grace'),
    }

    def _check_event_condition(
        self, *, is_bad: bool, event_type: str, deviation: float,
        mono_now: float, epoch_now: float,
    ):
        since_attr, max_attr, grace_attr = self._TIMER_ATTR_MAP[event_type]
        since = getattr(self, since_attr)  # None or (mono, epoch) tuple
        grace = getattr(self, grace_attr)  # None or monotonic time

        if is_bad:
            # Condition is bad — clear any grace timer, accumulate
            setattr(self, grace_attr, None)

            cur_max = getattr(self, max_attr)
            setattr(self, max_attr, max(cur_max, deviation))

            if since is None:
                # Condition just turned bad — start the timer
                setattr(self, since_attr, (mono_now, epoch_now))
        else:
            # Condition is OK this frame
            if since is not None:
                if grace is None:
                    # First False frame — start grace period
                    setattr(self, grace_attr, mono_now)
                elif (mono_now - grace) >= self._GRACE_S:
                    # Grace period expired — truly cleared, finalize event
                    mono_start, epoch_start = since
                    # Duration up to when grace started (last True frame)
                    elapsed = grace - mono_start
                    if elapsed >= self._EVENT_HOLD_S:
                        self._pending_events.append({
                            'start_time': datetime.datetime.fromtimestamp(epoch_start).isoformat(),
                            'end_time':   datetime.datetime.fromtimestamp(
                                epoch_now - (mono_now - grace)
                            ).isoformat(),
                            'duration':   elapsed,
                            'event_type': event_type,
                            'deviation_value': getattr(self, max_attr),
                        })
                    # Reset
                    setattr(self, since_attr, None)
                    setattr(self, max_attr, 0.0)
                    setattr(self, grace_attr, None)
                # else: still within grace period — keep waiting

    def _flush_session_to_db(self):
        """Save the completed session and all accumulated events to the DB."""
        if self._session_start_time is None:
            return

        session_end = time.time()
        duration = session_end - self._session_start_time

        avg_cva = (
            sum(self._cva_accumulator) / len(self._cva_accumulator)
            if self._cva_accumulator
            else 0.0
        )

        # Finalize any still-open events
        mono_now = time.monotonic()
        for event_type, (since_attr, max_attr, grace_attr) in self._TIMER_ATTR_MAP.items():
            since = getattr(self, since_attr)
            if since is not None:
                mono_start, epoch_start = since
                elapsed = mono_now - mono_start
                if elapsed >= self._EVENT_HOLD_S:
                    self._pending_events.append({
                        'start_time': datetime.datetime.fromtimestamp(epoch_start).isoformat(),
                        'end_time':   datetime.datetime.fromtimestamp(session_end).isoformat(),
                        'duration':   elapsed,
                        'event_type': event_type,
                        'deviation_value': getattr(self, max_attr),
                    })
                setattr(self, since_attr, None)
                setattr(self, max_attr, 0.0)
                setattr(self, grace_attr, None)

        start_iso = datetime.datetime.fromtimestamp(self._session_start_time).isoformat()
        end_iso = datetime.datetime.fromtimestamp(session_end).isoformat()

        try:
            from database import DatabaseManager
            db = DatabaseManager()
            db.save_session(
                start_time=start_iso,
                end_time=end_iso,
                duration=duration,
                avg_cva=avg_cva,
                events=self._pending_events,
            )
            db.close()
            self.session_saved.emit()
        except Exception as exc:
            print(f"[PoseWorker] Failed to save session to DB: {exc}")

        # Cleanup
        self._pending_events = []
        self._cva_accumulator = []
        self._session_start_time = None

    def _emit_front_frame(self, bgr_image, h, w):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.front_frame_ready.emit(qimg.copy())

    def _emit_side_frame(self, bgr_image, h, w):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.side_frame_ready.emit(qimg.copy())

    @staticmethod
    def _calculate_2d_cva(landmarks, width: int, height: int) -> float | None:
        """
        2D Craniovertebral Angle (CVA) from side view using pixel coords.

        Uses a horizontal reference through the shoulder (C7 proxy) and the
        vector from shoulder -> ear (tragus proxy). Angle is in degrees.
        """
        try:
            l_ear = landmarks[LEFT_EAR]
            r_ear = landmarks[RIGHT_EAR]
            l_sh = landmarks[LEFT_SHOULDER]
            r_sh = landmarks[RIGHT_SHOULDER]
        except Exception:
            return None

        pts = PoseWorker._get_side_cva_points(landmarks, width, height)
        if pts is None:
            return None
        (ear_x, ear_y), (sh_x, sh_y) = pts

        dx = ear_x - sh_x
        dy = sh_y - ear_y  # positive if ear is above shoulder
        if dx == 0 and dy == 0:
            return None

        # CVA is the acute angle between the horizontal through C7 (shoulder proxy)
        # and the line from C7 -> ear (tragus proxy). Ensure 0..90° regardless of
        # facing direction (dx sign) and image y-axis direction.
        angle = m.degrees(m.atan2(abs(dy), abs(dx)))
        return angle

    @staticmethod
    def _get_side_cva_points(landmarks, width: int, height: int):
        """
        Return ((ear_x, ear_y), (c7_x, c7_y)) in pixel coords for side CVA calc.
        Ear: LEFT_EAR or RIGHT_EAR
        C7 proxy: LEFT_SHOULDER or RIGHT_SHOULDER
        Picks the more visible side.
        """
        try:
            l_ear = landmarks[LEFT_EAR]
            r_ear = landmarks[RIGHT_EAR]
            l_sh = landmarks[LEFT_SHOULDER]
            r_sh = landmarks[RIGHT_SHOULDER]
        except Exception:
            return None

        ear = l_ear if getattr(l_ear, "visibility", 0.0) >= getattr(r_ear, "visibility", 0.0) else r_ear
        sh = l_sh if getattr(l_sh, "visibility", 0.0) >= getattr(r_sh, "visibility", 0.0) else r_sh

        ear_x = float(ear.x) * width
        ear_y = float(ear.y) * height
        sh_x = float(sh.x) * width
        sh_y = float(sh.y) * height

        if not (m.isfinite(ear_x) and m.isfinite(ear_y) and m.isfinite(sh_x) and m.isfinite(sh_y)):
            return None

        return (ear_x, ear_y), (sh_x, sh_y)

    @staticmethod
    def _calculate_trunk_tilt_deg(landmarks, width: int, height: int) -> float | None:
        """
        Trunk lateral tilt (frontal view): deviation of the spine line from
        the absolute vertical axis of the camera.

        Uses mid-point between shoulders as the lower point and the nose as the
        upper point. Returns degrees (0..90).
        """
        try:
            ls = landmarks[LEFT_SHOULDER]
            rs = landmarks[RIGHT_SHOULDER]
            nose = landmarks[NOSE]
        except Exception:
            return None

        ls_x, ls_y = float(ls.x) * width, float(ls.y) * height
        rs_x, rs_y = float(rs.x) * width, float(rs.y) * height
        nose_x, nose_y = float(nose.x) * width, float(nose.y) * height

        if not all(m.isfinite(v) for v in (ls_x, ls_y, rs_x, rs_y, nose_x, nose_y)):
            return None

        mid_x = (ls_x + rs_x) / 2.0
        mid_y = (ls_y + rs_y) / 2.0

        dx = nose_x - mid_x
        dy = mid_y - nose_y  # positive if nose above shoulders
        if dx == 0 and dy == 0:
            return None

        # Angle between vertical axis and the vector mid->nose
        # 0° when aligned with vertical, increases with lateral lean.
        angle = m.degrees(m.atan2(abs(dx), abs(dy)))
        return angle

    @staticmethod
    def _calculate_hunch_ratio(landmarks, width: int, height: int) -> float | None:
        """
        Hunch (shrug/neck tuck) proxy from frontal view.

        Compute normalized vertical distance ear->shoulder:
            ratio = (shoulder_y - ear_y) / shoulder_width_px
        Smaller ratio => ear is closer to shoulder in Y => more "hunched".
        """
        try:
            le = landmarks[LEFT_EAR]
            re = landmarks[RIGHT_EAR]
            ls = landmarks[LEFT_SHOULDER]
            rs = landmarks[RIGHT_SHOULDER]
        except Exception:
            return None

        ls_x, ls_y = float(ls.x) * width, float(ls.y) * height
        rs_x, rs_y = float(rs.x) * width, float(rs.y) * height
        le_y = float(le.y) * height
        re_y = float(re.y) * height

        if not all(m.isfinite(v) for v in (ls_x, ls_y, rs_x, rs_y, le_y, re_y)):
            return None

        shoulder_w = abs(ls_x - rs_x)
        if shoulder_w < 1.0:
            return None

        # Use the more visible ear/shoulder side
        use_left = getattr(le, "visibility", 0.0) >= getattr(re, "visibility", 0.0)
        ear_y = le_y if use_left else re_y
        sh_y = ls_y if use_left else rs_y

        dy = sh_y - ear_y  # positive when ear is above shoulder
        return dy / shoulder_w
