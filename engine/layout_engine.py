"""
Layout engine for OpenShorts Cloud Render Farm:
Provides deterministic layout calculations and FFmpeg filter graph generation
for SPLIT, TRACK, and GENERAL reframing modes.
"""

import os
from typing import Any, Dict, Optional, Tuple

VALID_LAYOUT_MODES = {"SPLIT", "TRACK", "GENERAL", "SCREENCAST"}


def validate_job_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates job_manifest dictionary fail-closed.
    Raises ValueError with descriptive reason if invalid.
    """
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")

    required_sections = ["project_id", "source", "timing", "layout", "output_spec"]
    for sec in required_sections:
        if sec not in manifest or manifest[sec] is None:
            raise ValueError(f"Missing required manifest section: {sec}")
        if sec != "project_id" and not isinstance(manifest[sec], dict):
            raise ValueError(f"Manifest section '{sec}' must be a JSON object")

    # Validate project_id
    if not isinstance(manifest["project_id"], str) or not manifest["project_id"].strip():
        raise ValueError("project_id must be a non-empty string")

    # Validate source
    source = manifest["source"]
    src_type = source.get("type")
    if src_type not in ("url", "local"):
        raise ValueError("source.type must be 'url' or 'local'")
    if src_type == "url":
        url = source.get("url", "")
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError(f"source.url must be a valid HTTP/HTTPS URL, got: '{url}'")
    if src_type == "local" and not source.get("local_asset_path"):
        raise ValueError("source.local_asset_path cannot be empty when source.type is 'local'")

    # Validate timing
    timing = manifest["timing"]
    start_sec = timing.get("start_seconds")
    end_sec = timing.get("end_seconds")
    dur_sec = timing.get("duration_seconds")

    if start_sec is None or end_sec is None or dur_sec is None:
        raise ValueError("timing must define start_seconds, end_seconds, and duration_seconds")

    try:
        start_sec = float(start_sec)
        end_sec = float(end_sec)
        dur_sec = float(dur_sec)
    except (TypeError, ValueError):
        raise ValueError("timing parameters must be valid numeric seconds")

    if start_sec < 0 or end_sec <= 0 or dur_sec <= 0:
        raise ValueError("timing seconds must be positive values")

    if start_sec >= end_sec:
        raise ValueError(
            f"start_seconds ({start_sec}) must be strictly less than end_seconds ({end_sec})"
        )

    if abs((end_sec - start_sec) - dur_sec) > 0.5:
        raise ValueError(
            f"duration_seconds ({dur_sec}) does not match end_seconds ({end_sec}) - start_seconds ({start_sec})"
        )

    # Validate layout
    layout = manifest["layout"]
    mode = layout.get("mode")
    if mode not in VALID_LAYOUT_MODES:
        raise ValueError(f"Unsupported layout mode: '{mode}'. Allowed: {sorted(list(VALID_LAYOUT_MODES))}")

    # Validate output_spec
    out_spec = manifest["output_spec"]
    res = out_spec.get("resolution")
    if not isinstance(res, (list, tuple)) or len(res) != 2:
        raise ValueError("output_spec.resolution must be a 2-element array [width, height]")
    try:
        w_val = int(res[0])
        h_val = int(res[1])
        if w_val <= 0 or h_val <= 0:
            raise ValueError("output_spec.resolution dimensions must be positive integers")
    except (TypeError, ValueError):
        raise ValueError("output_spec.resolution dimensions must be integers")

    fps = out_spec.get("fps", 30)
    try:
        fps_val = int(fps)
        if fps_val <= 0 or fps_val > 120:
            raise ValueError(f"output_spec.fps must be between 1 and 120, got: {fps}")
    except (TypeError, ValueError):
        raise ValueError(f"output_spec.fps must be a valid integer, got: {fps}")

    return manifest


def split_geometry(
    orig_w: int,
    orig_h: int,
    out_w: int,
    out_h: int,
    centre: Tuple[float, float, float, float],
    tightness: float = 0.8,
) -> Tuple[int, int, int, int, int]:
    """
    Crop box (crop_w, crop_h, crop_x, crop_y, half_h) framing centre for one half of SPLIT stack.
    Each half occupies out_w x (out_h // 2).
    """
    half_h = out_h // 2
    half_h -= half_h % 2
    aspect = out_w / float(half_h)

    clamped_tightness = max(0.3, min(tightness, 1.0))
    crop_h = int(round(orig_h * clamped_tightness))
    crop_w = int(round(crop_h * aspect))

    if crop_w > orig_w:
        crop_w = orig_w
        crop_h = int(round(crop_w / aspect))

    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    cx, cy = centre[0], centre[1]
    x = int(round(cx - crop_w / 2.0))
    x = max(0, min(x, orig_w - crop_w))
    x -= x % 2

    # Headroom adjustment: bias down slightly so face doesn't touch the top
    y = int(round(cy - crop_h * 0.42))
    y = max(0, min(y, orig_h - crop_h))
    y -= y % 2

    return crop_w, crop_h, x, y, half_h


def build_split_filter(
    orig_w: int,
    orig_h: int,
    out_w: int = 1080,
    out_h: int = 1920,
    left_centre: Optional[Tuple[float, float, float, float]] = None,
    right_centre: Optional[Tuple[float, float, float, float]] = None,
    tightness: float = 0.8,
) -> str:
    """
    Generates FFmpeg filter complex string for SPLIT mode (2 stacked speakers).
    Left speaker is placed on top half; right speaker on bottom half.
    """
    if left_centre is None:
        left_centre = (orig_w * 0.25, orig_h * 0.5, orig_w * 0.15, orig_h * 0.2)
    if right_centre is None:
        right_centre = (orig_w * 0.75, orig_h * 0.5, orig_w * 0.15, orig_h * 0.2)

    top_w, top_h, top_x, top_y, half_h = split_geometry(
        orig_w, orig_h, out_w, out_h, left_centre, tightness=tightness
    )
    bot_w, bot_h, bot_x, bot_y, _ = split_geometry(
        orig_w, orig_h, out_w, out_h, right_centre, tightness=tightness
    )

    filter_str = (
        f"[0:v]split=2[ta][ba];"
        f"[ta]crop=w={top_w}:h={top_h}:x={top_x}:y={top_y},"
        f"scale={out_w}:{half_h}[top];"
        f"[ba]crop=w={bot_w}:h={bot_h}:x={bot_x}:y={bot_y},"
        f"scale={out_w}:{half_h}[bot];"
        f"[top][bot]vstack=inputs=2,"
        f"pad={out_w}:{out_h}:0:0,setsar=1[v]"
    )
    return filter_str


def build_general_filter(
    orig_w: int,
    orig_h: int,
    out_w: int = 1080,
    out_h: int = 1920,
) -> str:
    """
    Generates FFmpeg filter complex string for GENERAL mode:
    Background blurred and scaled to cover 9:16 frame.
    Foreground centered and scaled maintaining original aspect ratio.
    """
    filter_str = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h},"
        f"boxblur=luma_radius=20:luma_power=2[bg_blur];"
        f"[fg]scale={out_w}:-2:force_original_aspect_ratio=decrease[fg_scaled];"
        f"[bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"
    )
    return filter_str


def build_track_crop_geometry(
    orig_w: int,
    orig_h: int,
    face_box: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[int, int, int, int]:
    """
    Calculates 9:16 vertical crop (crop_w, crop_h, crop_x, crop_y) around face_box.
    face_box is (x1, y1, x2, y2). If None, defaults to center of frame.
    """
    crop_h = orig_h
    crop_w = int(round(crop_h * (9.0 / 16.0)))
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    if face_box:
        x1, y1, x2, y2 = face_box
        cx = (x1 + x2) / 2.0
    else:
        cx = orig_w / 2.0

    x = int(round(cx - crop_w / 2.0))
    x = max(0, min(x, orig_w - crop_w))
    x -= x % 2

    return crop_w, crop_h, x, 0


def build_track_filter(
    orig_w: int,
    orig_h: int,
    out_w: int = 1080,
    out_h: int = 1920,
    face_box: Optional[Tuple[float, float, float, float]] = None,
) -> str:
    """
    Generates FFmpeg filter complex for single face tracking crop.
    """
    crop_w, crop_h, crop_x, crop_y = build_track_crop_geometry(orig_w, orig_h, face_box)
    filter_str = (
        f"[0:v]crop=w={crop_w}:h={crop_h}:x={crop_x}:y={crop_y},"
        f"scale={out_w}:{out_h},setsar=1[v]"
    )
    return filter_str


def detect_person_box_yolo(
    video_path: str,
    start_sec: float,
    duration_sec: float,
    orig_w: int,
    orig_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Samples frames from video_path during the target interval and uses YOLOv8 (yolov8n.pt)
    to detect the primary person/speaker bounding box (x1, y1, x2, y2).
    Returns None if YOLO is unavailable, video cannot be opened, or no person is detected.
    """
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        return None

    try:
        model_path = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        sample_times = [
            start_sec + duration_sec * 0.25,
            start_sec + duration_sec * 0.50,
            start_sec + duration_sec * 0.75,
        ]

        detected_boxes = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        for t in sample_times:
            frame_num = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            results = model(frame, classes=[0], verbose=False)
            if not results:
                continue

            max_area = 0
            best_box = None
            for r in results:
                for b in r.boxes:
                    coords = [float(x) for x in b.xyxy[0]]
                    w = coords[2] - coords[0]
                    h = coords[3] - coords[1]
                    area = w * h
                    if area > max_area:
                        max_area = area
                        best_box = (coords[0], coords[1], coords[2], coords[3])

            if best_box:
                detected_boxes.append(best_box)

        cap.release()

        if not detected_boxes:
            return None

        avg_x1 = sum(b[0] for b in detected_boxes) / len(detected_boxes)
        avg_y1 = sum(b[1] for b in detected_boxes) / len(detected_boxes)
        avg_x2 = sum(b[2] for b in detected_boxes) / len(detected_boxes)
        avg_y2 = sum(b[3] for b in detected_boxes) / len(detected_boxes)

        return (avg_x1, avg_y1, avg_x2, avg_y2)
    except Exception as e:
        print(f"⚠️ [YOLO] Detection exception: {e}, falling back to center layout")
        return None


def detect_split_speakers_yolo(
    video_path: str,
    start_sec: float,
    duration_sec: float,
    orig_w: int,
    orig_h: int,
) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[Tuple[float, float, float, float]]]:
    """
    Samples video to detect two distinct speakers for SPLIT layout.
    Returns (left_centre, right_centre) or (None, None) if detection falls back.
    """
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        return None, None

    try:
        model_path = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None

        mid_time = start_sec + duration_sec * 0.5
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid_time * fps))
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None, None

        results = model(frame, classes=[0], verbose=False)
        if not results:
            return None, None

        persons = []
        for r in results:
            for b in r.boxes:
                coords = [float(x) for x in b.xyxy[0]]
                cx = (coords[0] + coords[2]) / 2.0
                cy = (coords[1] + coords[3]) / 2.0
                bw = coords[2] - coords[0]
                bh = coords[3] - coords[1]
                if bw > orig_w * 0.05:  # ignore small noise
                    persons.append((cx, cy, bw, bh))

        if len(persons) >= 2:
            persons.sort(key=lambda p: p[0])
            left = persons[0]
            right = persons[-1]
            if (right[0] - left[0]) > orig_w * 0.15:
                return left, right

        return None, None
    except Exception as e:
        print(f"⚠️ [YOLO] Split detection exception: {e}")
        return None, None
