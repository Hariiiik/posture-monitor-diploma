

import cv2
import math as m
import argparse
import os
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmark

# Path to the downloaded model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")

def findDistance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        x1, y1: Coordinates of the first point.
        x2, y2: Coordinates of the second point.

    Returns:
        Distance between the two points.
    """
    dist = m.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def findAngle(x1, y1, x2, y2):
    """
    Calculate the inclination angle (degrees) of the vector from (x1, y1)
    to (x2, y2) relative to the vertical axis.
    Works correctly with world-space coordinates (metres).

    Args:
        x1, y1: World-space origin point (metres).
        x2, y2: World-space target point (metres).

    Returns:
        Angle in degrees from vertical (0° = perfectly upright).
    """
    dx = x2 - x1
    dy = y2 - y1
    if dy == 0:
        return 90.0
    return m.degrees(m.atan2(abs(dx), abs(dy)))

def sendWarning(x):
    """
    Placeholder function for sending a warning.
    """
    pass


def _draw_unicode_batch(img, items):
    """
    Render a batch of Unicode (Cyrillic) text onto an OpenCV BGR image.
    Uses a single PIL round-trip for all items.

    items: list of (pos, text, font_scale, bgr_color)
    Falls back to cv2.putText (ASCII-only) if Pillow is unavailable.
    """
    try:
        from PIL import Image as PilImage, ImageDraw, ImageFont
        rgb_img = PilImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(rgb_img)
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
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, bgr, 2)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Posture Monitor with MediaPipe')
    parser.add_argument('--video', type=str, default=None, help='Path to the input video file. If not provided, the webcam will be used.')
    parser.add_argument('--offset-threshold', type=float, default=0.3, help='Shoulder alignment threshold in world metres (default: 0.3 m). Shoulder-to-shoulder distance below this value indicates a side-view pose.')
    parser.add_argument('--neck-angle-threshold', type=int, default=25, help='Threshold value for neck inclination angle.')
    parser.add_argument('--torso-angle-threshold', type=int, default=10, help='Threshold value for torso inclination angle.')
    parser.add_argument('--time-threshold', type=int, default=180, help='Time threshold for triggering a posture alert.')
    parser.add_argument('--shoulder-tilt-threshold', type=float, default=0.05,
                        help='Max allowed Y-difference between shoulders in world metres (default: 0.05 m = 5 cm).')
    parser.add_argument('--forward-lean-threshold', type=float, default=0.10,
                        help='Shoulder Z depth below which a forward lean is detected (default: 0.10 m).')
    return parser.parse_args()

def main(video_path=None, offset_threshold=0.3, neck_angle_threshold=25, torso_angle_threshold=10,
         time_threshold=180, shoulder_tilt_threshold=0.05, forward_lean_threshold=0.10):
    # Initialize frame counters.
    good_frames = 0
    bad_frames = 0

    # Font type.
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Colors.
    blue = (255, 127, 0)
    red = (50, 50, 255)
    green = (127, 255, 0)
    dark_blue = (127, 20, 0)
    light_green = (127, 233, 100)
    yellow = (0, 255, 255)
    pink = (255, 0, 255)
    white = (255, 255, 255)

    # Initialize MediaPipe PoseLandmarker (Tasks API).
    base_options = mp_python.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    # For file input, replace file name with <path>.
    cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(0)

    # Meta.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0  # fallback for webcams that report 0

    frame_index = 0

    while True:
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break

        # Get fps.
        live_fps = cap.get(cv2.CAP_PROP_FPS)
        if live_fps > 0:
            fps = live_fps

        # Get height and width of the frame.
        h, w = image.shape[:2]

        # Convert BGR to RGB for MediaPipe.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Calculate timestamp in milliseconds.
        timestamp_ms = int(frame_index * (1000.0 / fps))
        frame_index += 1

        # Run pose detection.
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Skip frame if world landmarks are unavailable.
        if (not result.pose_world_landmarks or len(result.pose_world_landmarks) == 0
                or not result.pose_landmarks or len(result.pose_landmarks) == 0):
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 3-D world landmarks (metres, origin at hip midpoint) — used for analysis.
        lm_world = result.pose_world_landmarks[0]
        # Normalised 2-D landmarks — used only for drawing on the frame.
        lm_norm  = result.pose_landmarks[0]

        def px(landmark_idx):
            """Return pixel (x, y) position from normalised 2-D landmarks."""
            return int(lm_norm[landmark_idx].x * w), int(lm_norm[landmark_idx].y * h)

        # Landmark indices from PoseLandmark enum.
        LEFT_SHOULDER  = PoseLandmark.LEFT_SHOULDER.value
        RIGHT_SHOULDER = PoseLandmark.RIGHT_SHOULDER.value
        LEFT_EAR       = PoseLandmark.LEFT_EAR.value
        RIGHT_EAR      = PoseLandmark.RIGHT_EAR.value
        LEFT_HIP       = PoseLandmark.LEFT_HIP.value

        # ── World-space coordinates (metres) ──────────────────────────────
        l_shldr_wx = lm_world[LEFT_SHOULDER].x
        l_shldr_wy = lm_world[LEFT_SHOULDER].y
        l_shldr_wz = lm_world[LEFT_SHOULDER].z
        r_shldr_wx = lm_world[RIGHT_SHOULDER].x
        r_shldr_wy = lm_world[RIGHT_SHOULDER].y
        r_shldr_wz = lm_world[RIGHT_SHOULDER].z
        # Use midpoint of both ears to reduce bias from head rotation / camera angle
        l_ear_wx   = lm_world[LEFT_EAR].x
        l_ear_wy   = lm_world[LEFT_EAR].y
        r_ear_wx   = lm_world[RIGHT_EAR].x
        r_ear_wy   = lm_world[RIGHT_EAR].y
        mid_ear_wx = (l_ear_wx + r_ear_wx) / 2.0
        mid_ear_wy = (l_ear_wy + r_ear_wy) / 2.0
        # Midpoint between shoulders — base of neck reference
        mid_shldr_wx = (l_shldr_wx + r_shldr_wx) / 2.0
        mid_shldr_wy = (l_shldr_wy + r_shldr_wy) / 2.0
        l_hip_wx   = lm_world[LEFT_HIP].x
        l_hip_wy   = lm_world[LEFT_HIP].y

        # ── Pixel coordinates (drawing only) ─────────────────────────────
        l_shldr_x, l_shldr_y = px(LEFT_SHOULDER)
        r_shldr_x, r_shldr_y = px(RIGHT_SHOULDER)
        l_ear_x,   l_ear_y   = px(LEFT_EAR)
        r_ear_x,   r_ear_y   = px(RIGHT_EAR)
        mid_ear_px = ((l_ear_x + r_ear_x) // 2, (l_ear_y + r_ear_y) // 2)
        mid_shldr_px = ((l_shldr_x + r_shldr_x) // 2, (l_shldr_y + r_shldr_y) // 2)
        l_hip_x,   l_hip_y   = px(LEFT_HIP)

        # Shoulder-to-shoulder distance in world space (metres).
        # Large value = frontal view (both shoulders visible) — this is NORMAL.
        # Small value = side view (one shoulder hidden behind the other).
        offset = findDistance(l_shldr_wx, l_shldr_wy, r_shldr_wx, r_shldr_wy)

        # Average shoulder depth: negative z → leaning toward the camera.
        shoulder_depth = (l_shldr_wz + r_shldr_wz) / 2.0

        # Show a warning only if offset is too small (side/profile view).
        if offset >= offset_threshold:
            cv2.putText(image, f'{offset:.2f} m  Frontal view OK', (w - 300, 30), font, 0.6, green, 2)
        else:
            cv2.putText(image, f'{offset:.2f} m  Side view detected', (w - 330, 30), font, 0.6, red, 2)

        # ── Posture analysis (upper body only, frontal view) ─────────────────

        # Neck inclination: shoulder midpoint → ear midpoint, angle from vertical.
        # Using midpoints of both shoulders and both ears eliminates left/right bias.
        neck_inclination = findAngle(mid_shldr_wx, mid_shldr_wy, mid_ear_wx, mid_ear_wy)

        # Shoulder Y-asymmetry: |Δy| between left and right shoulder (metres).
        # Non-zero value indicates one shoulder is raised/lowered — back tilt.
        shoulder_tilt = abs(l_shldr_wy - r_shldr_wy)

        # ── Status flags ──────────────────────────────────────────────────────
        is_good_neck    = neck_inclination < neck_angle_threshold
        is_leaning      = shoulder_depth   < -forward_lean_threshold   # z<0 = toward camera
        is_tilted       = shoulder_tilt    >  shoulder_tilt_threshold

        is_good_posture = is_good_neck and not is_leaning and not is_tilted

        if is_good_posture:
            good_frames += 1
            bad_frames   = 0
        else:
            bad_frames  += 1
            good_frames  = 0

        # ── Draw upper-body skeleton (no hip / leg landmarks) ─────────────────
        skel_color = green if is_good_posture else red
        cv2.circle(image, (l_shldr_x,    l_shldr_y),    7, white, 2)
        cv2.circle(image, (r_shldr_x,    r_shldr_y),    7, pink,  -1)
        cv2.circle(image, mid_ear_px,                   7, white, 2)
        cv2.circle(image, mid_shldr_px,                 5, yellow, -1)  # midpoint ref dot
        # Shoulder bar — highlights tilt visually
        cv2.line(image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), skel_color, 2)
        # Neck line: shoulder midpoint → ear midpoint
        cv2.line(image, mid_shldr_px, mid_ear_px, skel_color, 2)
        # Vertical reference from shoulder midpoint
        cv2.line(image, mid_shldr_px, (mid_shldr_px[0], mid_shldr_px[1] - 100), skel_color, 2)
        # Angle annotation next to midpoint
        cv2.putText(image, f'{int(neck_inclination)}',
                    (mid_shldr_px[0] + 10, mid_shldr_px[1]), font, 0.9,
                    light_green if is_good_neck else red, 2)

        # ── HUD: numeric metrics (top-left) ───────────────────────────────────
        cv2.putText(image, f'Neck:  {int(neck_inclination)} deg', (10, 30),  font, 0.6,
                    light_green if is_good_neck else red, 2)
        cv2.putText(image, f'Tilt:  {shoulder_tilt * 100:.1f} cm',  (10, 60),  font, 0.6,
                    light_green if not is_tilted else red, 2)
        cv2.putText(image, f'Depth: {shoulder_depth:+.2f} m', (10, 90),  font, 0.6,
                    light_green if not is_leaning else yellow, 2)

        # ── HUD: Ukrainian status labels (rendered via PIL for Cyrillic) ───────
        status_y     = 130
        unicode_items = []
        if is_good_posture:
            unicode_items.append(((10, status_y), 'Правильна постава', 0.85, light_green))
        else:
            if is_leaning:
                unicode_items.append(((10, status_y), 'Нахил вперед!', 0.85, yellow))
                status_y += 40
            if is_tilted:
                unicode_items.append(((10, status_y), 'Перекіс плечей!', 0.85, red))
        if unicode_items:
            _draw_unicode_batch(image, unicode_items)

        # Calculate the time of remaining in a particular posture.
        good_time = (1 / fps) * good_frames
        bad_time = (1 / fps) * bad_frames

        # Pose time.
        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(float(good_time), 1)) + 's'
            cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(float(bad_time), 1)) + 's'
            cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

        # If you stay in bad posture for more than the threshold, send an alert.
        if bad_time > time_threshold:
            sendWarning(bad_time)

        # Show the frame.
        cv2.imshow('MediaPipe Pose', image)

        # Exit the loop if 'q' is pressed OR the window's ✕ button is clicked.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('MediaPipe Pose', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the camera and close the windows.
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_arguments()

    print("Arguments:")
    print(f"Video:                   {args.video}")
    print(f"Offset Threshold:        {args.offset_threshold} m")
    print(f"Neck Angle Threshold:    {args.neck_angle_threshold} deg")
    print(f"Torso Angle Threshold:   {args.torso_angle_threshold} deg")
    print(f"Shoulder Tilt Threshold: {args.shoulder_tilt_threshold} m")
    print(f"Forward Lean Threshold:  {args.forward_lean_threshold} m")
    print(f"Time Threshold:          {args.time_threshold} s")

    main(
        video_path=args.video,
        offset_threshold=args.offset_threshold,
        neck_angle_threshold=args.neck_angle_threshold,
        torso_angle_threshold=args.torso_angle_threshold,
        time_threshold=args.time_threshold,
        shoulder_tilt_threshold=args.shoulder_tilt_threshold,
        forward_lean_threshold=args.forward_lean_threshold,
    )
