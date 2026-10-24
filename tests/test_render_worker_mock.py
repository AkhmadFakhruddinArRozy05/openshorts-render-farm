"""
Unit tests for OpenShorts Cloud Render Farm engine components:
- manifest validation
- layout calculations (SPLIT, TRACK, GENERAL)
- ffmpeg command assembly
"""

import pytest
import sys
from pathlib import Path

# Add openshorts-render-farm to sys.path
PROJECT_DIR = Path(__file__).resolve().parents[1]
RENDER_FARM_DIR = PROJECT_DIR / "Tools/OpenShorts-Render-Farm"
if str(RENDER_FARM_DIR) not in sys.path:
    sys.path.append(str(RENDER_FARM_DIR))

from engine.layout_engine import (
    validate_job_manifest,
    split_geometry,
    build_split_filter,
    build_general_filter,
    build_track_crop_geometry,
    build_track_filter,
)
from engine.render_worker import build_ffmpeg_render_command


@pytest.fixture
def sample_valid_manifest():
    return {
        "project_id": "PODCAST-EPS01-CLIP01",
        "source": {
            "type": "url",
            "url": "https://www.youtube.com/watch?v=mock_video_id",
            "local_asset_path": None,
        },
        "timing": {
            "start_seconds": 120.0,
            "end_seconds": 175.0,
            "duration_seconds": 55.0,
        },
        "layout": {
            "mode": "SPLIT",
            "split_seam_y": 0.5,
            "crop_smoothing": True,
        },
        "subtitles": {
            "enabled": True,
            "style": "karaoke",
            "font_size": 24,
            "color": "#FFFFFF",
            "highlight_color": "#FFDF00",
        },
        "hook_overlay": {
            "enabled": True,
            "text": "Rahasia Sukses Terbesar!",
            "duration_seconds": 4.0,
        },
        "output_spec": {
            "resolution": [1080, 1920],
            "fps": 30,
            "codec": "libx264",
        },
    }


def test_validate_job_manifest_valid(sample_valid_manifest):
    cleaned = validate_job_manifest(sample_valid_manifest)
    assert cleaned["project_id"] == "PODCAST-EPS01-CLIP01"
    assert cleaned["layout"]["mode"] == "SPLIT"
    assert cleaned["timing"]["duration_seconds"] == 55.0


def test_validate_job_manifest_invalid_timing(sample_valid_manifest):
    # start >= end
    bad = dict(sample_valid_manifest)
    bad["timing"] = {"start_seconds": 100.0, "end_seconds": 50.0, "duration_seconds": 50.0}
    with pytest.raises(ValueError, match="must be strictly less than end_seconds"):
        validate_job_manifest(bad)

    # negative duration
    bad_neg = dict(sample_valid_manifest)
    bad_neg["timing"] = {"start_seconds": 10.0, "end_seconds": 50.0, "duration_seconds": -40.0}
    with pytest.raises(ValueError, match="timing seconds must be positive values"):
        validate_job_manifest(bad_neg)



def test_validate_job_manifest_missing_required_fields(sample_valid_manifest):
    bad = dict(sample_valid_manifest)
    del bad["source"]
    with pytest.raises(ValueError, match="Missing required manifest section: source"):
        validate_job_manifest(bad)


def test_validate_job_manifest_invalid_layout_mode(sample_valid_manifest):
    bad = dict(sample_valid_manifest)
    bad["layout"] = {"mode": "INVALID_MODE"}
    with pytest.raises(ValueError, match="Unsupported layout mode"):
        validate_job_manifest(bad)


def test_split_geometry_within_boundaries():
    in_w, in_h = 1920, 1080
    out_w, out_h = 1080, 1920
    # Left speaker center at (480, 540)
    w, h, x, y, half_h = split_geometry(in_w, in_h, out_w, out_h, (480.0, 540.0, 150.0, 150.0))
    
    assert w <= in_w
    assert h <= in_h
    assert x >= 0
    assert y >= 0
    assert x + w <= in_w
    assert y + h <= in_h
    assert half_h == 960
    assert w % 2 == 0
    assert h % 2 == 0


def test_build_split_filter_graph():
    in_w, in_h = 1920, 1080
    out_w, out_h = 1080, 1920
    left_c = (480.0, 540.0, 120.0, 120.0)
    right_c = (1440.0, 540.0, 120.0, 120.0)

    fgraph = build_split_filter(in_w, in_h, out_w, out_h, left_c, right_c)
    assert "[0:v]split=2" in fgraph
    assert "vstack=inputs=2" in fgraph
    assert f"pad={out_w}:{out_h}" in fgraph
    assert "scale=1080:960" in fgraph


def test_build_general_filter_graph():
    in_w, in_h = 1920, 1080
    out_w, out_h = 1080, 1920

    fgraph = build_general_filter(in_w, in_h, out_w, out_h)
    assert "[0:v]split=2" in fgraph
    assert "boxblur=" in fgraph
    assert "overlay=(W-w)/2:(H-h)/2" in fgraph
    assert "setsar=1" in fgraph


def test_build_track_crop_geometry():
    in_w, in_h = 1920, 1080
    face_box = (800, 300, 950, 500) # x1, y1, x2, y2 -> center_x = 875, center_y = 400
    w, h, x, y = build_track_crop_geometry(in_w, in_h, face_box)

    assert h == in_h
    # 9:16 aspect of 1080 height is 607.5 -> rounded to even 606 or 608
    assert abs(w - int(round(in_h * 9 / 16))) <= 2
    assert x >= 0
    assert x + w <= in_w
    assert y == 0


def test_build_track_filter_graph():
    in_w, in_h = 1920, 1080
    out_w, out_h = 1080, 1920
    face_box = (800, 300, 950, 500)

    fgraph = build_track_filter(in_w, in_h, out_w, out_h, face_box)
    assert "crop=" in fgraph
    assert f"scale={out_w}:{out_h}" in fgraph
    assert "setsar=1" in fgraph


def test_build_ffmpeg_render_command(sample_valid_manifest, tmp_path):
    source_file = tmp_path / "input.mp4"
    source_file.write_bytes(b"dummy input")
    output_file = tmp_path / "output.mp4"
    srt_file = tmp_path / "subtitles.srt"
    srt_file.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello World\n", encoding="utf-8")

    cmd = build_ffmpeg_render_command(
        source_path=str(source_file),
        output_path=str(output_file),
        manifest=sample_valid_manifest,
        filter_complex="[0:v]scale=1080:1920[v]",
        subtitle_path=str(srt_file),
    )

    cmd_str = " ".join(cmd)
    assert "-avoid_negative_ts make_zero" in cmd_str
    assert "-ss 120.0" in cmd_str
    assert "-t 55.0" in cmd_str
    assert "-c:v libx264" in cmd_str
    assert "-c:a aac" in cmd_str
    assert "subtitles=" in cmd_str
    assert "drawtext=" in cmd_str
    assert str(output_file) in cmd
