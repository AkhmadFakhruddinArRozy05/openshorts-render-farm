"""
Cloud Render Worker for OpenShorts Render Farm:
Executes the rendering pipeline in GitHub Actions cloud runner (Ubuntu 16GB RAM):
1. Ingests source video (URL or downloaded release asset).
2. Performs YOLOv8 face detection / layout framing.
3. Generates FFmpeg filter complex (SPLIT, TRACK, GENERAL).
4. Burns subtitles and hook text overlays.
5. Encodes 1080x1920 9:16 vertical MP4 and verifies integrity with ffprobe.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure layout_engine is importable
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from layout_engine import (
    validate_job_manifest,
    build_split_filter,
    build_general_filter,
    build_track_filter,
    detect_person_box_yolo,
    detect_split_speakers_yolo,
)


def escape_ffmpeg_text(text: str) -> str:
    """Escapes single quotes and special characters for FFmpeg drawtext."""
    if not text:
        return ""
    # Replace backslash first, then colon, single quote, percent
    return str(text).replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'").replace("%", r"\%")


def escape_ffmpeg_path(path: str) -> str:
    """Escapes path for FFmpeg subtitles filter on Unix."""
    return path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def build_ffmpeg_render_command(
    source_path: str,
    output_path: str,
    manifest: Dict[str, Any],
    filter_complex: Optional[str] = None,
    subtitle_path: Optional[str] = None,
) -> List[str]:
    """
    Assembles FFmpeg CLI command list for 9:16 vertical render.
    """
    timing = manifest["timing"]
    start_sec = float(timing["start_seconds"])
    duration_sec = float(timing["duration_seconds"])

    out_spec = manifest.get("output_spec", {})
    res = out_spec.get("resolution", [1080, 1920])
    out_w, out_h = res[0], res[1]
    fps = out_spec.get("fps", 30)
    codec = out_spec.get("codec", "libx264")

    # If no base filter graph was supplied, generate default based on layout mode
    if not filter_complex:
        layout_mode = manifest.get("layout", {}).get("mode", "GENERAL")
        if layout_mode == "SPLIT":
            filter_complex = build_split_filter(1920, 1080, out_w, out_h)
        elif layout_mode == "TRACK":
            filter_complex = build_track_filter(1920, 1080, out_w, out_h)
        else:
            filter_complex = build_general_filter(1920, 1080, out_w, out_h)

    # Chain subtitle filter if enabled
    current_label = "[v]"
    subs_cfg = manifest.get("subtitles", {})
    if subs_cfg.get("enabled", False) and subtitle_path and Path(subtitle_path).exists():
        escaped_sub = escape_ffmpeg_path(str(Path(subtitle_path).resolve()))
        filter_complex += f";{current_label}subtitles='{escaped_sub}'[v_sub]"
        current_label = "[v_sub]"

    # Chain hook overlay if enabled
    hook_cfg = manifest.get("hook_overlay", {})
    if hook_cfg.get("enabled", False) and hook_cfg.get("text"):
        hook_text = escape_ffmpeg_text(hook_cfg["text"])
        hook_dur = float(hook_cfg.get("duration_seconds", 4.0))
        # Draw hook card at top third with safe margins
        drawtext_filter = (
            f";{current_label}drawtext=text='{hook_text}':"
            f"fontcolor=white:fontsize=48:box=1:boxcolor=black@0.6:boxborderw=10:"
            f"x=(w-text_w)/2:y=240:enable='between(t,0,{hook_dur})'[v_hook]"
        )
        filter_complex += drawtext_filter
        current_label = "[v_hook]"

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", source_path,
        "-avoid_negative_ts", "make_zero",
        "-filter_complex", filter_complex,
        "-map", current_label,
        "-map", "0:a?",
        "-c:v", codec,
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]
    return cmd


def probe_video_dimensions(video_path: str) -> Tuple[int, int]:
    """Probes video width and height using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            video_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = res.stdout.strip().split("x")
        return int(parts[0]), int(parts[1])
    except Exception:
        # Fallback to standard 1080p 16:9
        return 1920, 1080


def verify_rendered_mp4(
    output_path: str,
    expected_resolution: Optional[Tuple[int, int]] = (1080, 1920),
) -> bool:
    """Verifies that output file exists, is non-empty, matches expected resolution, and passes ffprobe decode test."""
    p = Path(output_path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        if expected_resolution:
            cmd_probe = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x",
                str(p),
            ]
            probe_res = subprocess.run(cmd_probe, capture_output=True, text=True, timeout=15)
            if probe_res.returncode == 0:
                dim = probe_res.stdout.strip()
                exp_str = f"{expected_resolution[0]}x{expected_resolution[1]}"
                if exp_str not in dim:
                    print(f"⚠️ [WORKER] Output resolution {dim} does not match expected {exp_str}")
                    return False

        cmd = ["ffmpeg", "-v", "error", "-i", str(p), "-f", "null", "-"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return res.returncode == 0
    except Exception:
        return False


def run_worker(manifest_path: str, output_path: str) -> int:
    """Main execution loop for cloud runner."""
    print(f"🚀 [WORKER] Starting OpenShorts render worker for: {manifest_path}")
    m_file = Path(manifest_path)
    if not m_file.exists():
        print(f"❌ [WORKER] Manifest file not found: {manifest_path}")
        return 1

    try:
        manifest = json.loads(m_file.read_text(encoding="utf-8"))
        validate_job_manifest(manifest)
    except Exception as e:
        print(f"❌ [WORKER] Manifest validation failed: {e}")
        return 1

    source_cfg = manifest["source"]
    source_type = source_cfg.get("type")
    input_file = "source_video.mp4"

    # Step 1: Ingest source
    if source_type == "url":
        url = source_cfg["url"]
        print(f"📥 [WORKER] Downloading video stream from URL: {url}")
        ytdlp_cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", input_file,
            url,
        ]
        res = subprocess.run(ytdlp_cmd)
        if res.returncode != 0 or not Path(input_file).exists():
            print("❌ [WORKER] yt-dlp download failed!")
            return 1
    else:
        local_path = source_cfg.get("local_asset_path")
        if not local_path:
            print("❌ [WORKER] Missing source.local_asset_path")
            return 1

        candidate = Path(local_path)
        if candidate.is_file():
            input_file = str(candidate)
        elif Path(candidate.name).is_file():
            input_file = candidate.name
        else:
            print(f"❌ [WORKER] Local asset does not exist: {local_path} (also checked {candidate.name})")
            return 1

    # Step 2: Dimensions & Layout filter
    in_w, in_h = probe_video_dimensions(input_file)
    print(f"📐 [WORKER] Source resolution: {in_w}x{in_h}")

    layout_mode = manifest.get("layout", {}).get("mode", "GENERAL")
    res = manifest.get("output_spec", {}).get("resolution", [1080, 1920])
    out_w, out_h = res[0], res[1]

    timing = manifest["timing"]
    start_sec = float(timing["start_seconds"])
    dur_sec = float(timing["duration_seconds"])

    if layout_mode == "SPLIT":
        print("👥 [WORKER] Checking YOLO speaker coordinates for SPLIT layout...")
        left_c, right_c = detect_split_speakers_yolo(input_file, start_sec, dur_sec, in_w, in_h)
        fgraph = build_split_filter(in_w, in_h, out_w, out_h, left_centre=left_c, right_centre=right_c)
    elif layout_mode == "TRACK":
        print("🎯 [WORKER] Running YOLOv8 face/person tracking for TRACK layout...")
        face_box = detect_person_box_yolo(input_file, start_sec, dur_sec, in_w, in_h)
        fgraph = build_track_filter(in_w, in_h, out_w, out_h, face_box=face_box)
    else:
        print("🖼️ [WORKER] Building GENERAL blurred background filtergraph...")
        fgraph = build_general_filter(in_w, in_h, out_w, out_h)

    # Step 3: Subtitles
    sub_path = None
    subs_cfg = manifest.get("subtitles", {})
    if subs_cfg.get("enabled", False):
        if subs_cfg.get("file_path"):
            fp = Path(subs_cfg["file_path"])
            if fp.is_file():
                sub_path = str(fp)
            elif Path(fp.name).is_file():
                sub_path = fp.name
        if not sub_path:
            if Path("subtitles.srt").is_file():
                sub_path = "subtitles.srt"
            elif Path("subtitles.ass").is_file():
                sub_path = "subtitles.ass"
            else:
                for f in Path(".").glob("*.srt"):
                    sub_path = str(f)
                    break
                if not sub_path:
                    for f in Path(".").glob("*.ass"):
                        sub_path = str(f)
                        break

    # Step 4: Render FFmpeg
    cmd = build_ffmpeg_render_command(
        source_path=input_file,
        output_path=output_path,
        manifest=manifest,
        filter_complex=fgraph,
        subtitle_path=sub_path,
    )

    print(f"🎬 [WORKER] Executing FFmpeg command: {' '.join(cmd)}")
    render_res = subprocess.run(cmd)
    if render_res.returncode != 0:
        print(f"❌ [WORKER] FFmpeg render failed with exit code {render_res.returncode}")
        return 1

    # Step 5: Verification
    print("🔍 [WORKER] Verifying output MP4 integrity...")
    if not verify_rendered_mp4(output_path, expected_resolution=(out_w, out_h)):
        print("❌ [WORKER] Rendered artifact verification failed (corrupt or resolution mismatch)!")
        return 1

    print(f"✅ [WORKER] Render completed successfully! Saved to: {output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="OpenShorts Cloud Render Worker")
    parser.add_argument("--manifest", required=True, help="Path to job_manifest.json")
    parser.add_argument("--output", default="output.mp4", help="Path to output rendered MP4")
    args = parser.parse_args()

    sys.exit(run_worker(args.manifest, args.output))


if __name__ == "__main__":
    main()
