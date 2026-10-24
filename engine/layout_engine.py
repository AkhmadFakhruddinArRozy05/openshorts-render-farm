"""
Layout engine for OpenShorts Cloud Render Farm:
Provides deterministic layout calculations and FFmpeg filter graph generation
for SPLIT, TRACK, and GENERAL reframing modes.
"""

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

    # Validate project_id
    if not isinstance(manifest["project_id"], str) or not manifest["project_id"].strip():
        raise ValueError("project_id must be a non-empty string")

    # Validate source
    source = manifest["source"]
    src_type = source.get("type")
    if src_type not in ("url", "local"):
        raise ValueError("source.type must be 'url' or 'local'")
    if src_type == "url" and not source.get("url"):
        raise ValueError("source.url cannot be empty when source.type is 'url'")
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
    if res[0] <= 0 or res[1] <= 0:
        raise ValueError("output_spec.resolution dimensions must be positive integers")

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
