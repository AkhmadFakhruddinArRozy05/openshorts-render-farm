# OpenShorts Cloud Render Farm

Cloud rendering worker and orchestrator for OpenShorts podcast and talking-head 9:16 vertical video reframing on GitHub Actions runners (Ubuntu, 4 vCPU, 16 GB RAM).

## Overview

OpenShorts reframes wide-angle podcast/talking-head videos into 9:16 vertical short-form clips using:
1. **SPLIT Layout**: Stacks two participants vertically (1080x960 each) when both are in frame.
2. **TRACK Layout**: Follows an active speaker using YOLOv8 face tracking with horizontal smoothing.
3. **GENERAL Layout**: Centers the video frame with a blurred background fill.

Running YOLOv8 inference and heavy FFmpeg encoding locally on resource-constrained hardware (e.g. Apple Silicon Mac M1 with 8 GB RAM) causes high memory pressure and thermal throttling. This render farm offloads the compute-heavy phases to GitHub Actions cloud runners, returning the verified 1080x1920 MP4 back to the local Clipper workspace.

## Architecture & Workflow

```
[Mac M1 Control Plane]                [GitHub Actions 16 GB RAM]
Moment Selection & Manifest     --->   Workflow Trigger (render.yml)
      │                                       │
      ▼                                       ▼
Job Manifest & Asset Upload     --->   Download / Ingest Video
                                              │
                                              ▼
                                       YOLOv8 Reframing & Split Layout
                                              │
                                              ▼
                                       FFmpeg 9:16 Render & Subtitle Burn
                                              │
                                              ▼
MP4 Download & Tri-Level QA    <---   Upload Artifact (1080x1920)
```

## Job Manifest Specification (`job_manifest.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "project_id": "PODCAST-EPS01-CLIP01",
  "source": {
    "type": "url",
    "url": "https://www.youtube.com/watch?v=example",
    "local_asset_path": null
  },
  "timing": {
    "start_seconds": 120.0,
    "end_seconds": 175.0,
    "duration_seconds": 55.0
  },
  "layout": {
    "mode": "SPLIT",
    "split_seam_y": 0.5,
    "crop_smoothing": true
  },
  "subtitles": {
    "enabled": true,
    "style": "karaoke",
    "font_size": 24,
    "color": "#FFFFFF",
    "highlight_color": "#FFDF00"
  },
  "hook_overlay": {
    "enabled": true,
    "text": "Rahasia Sukses Bisnis!",
    "duration_seconds": 4.0
  },
  "output_spec": {
    "resolution": [1080, 1920],
    "fps": 30,
    "codec": "libx264"
  }
}
```

## Local CLI Usage

```bash
# Render using local orchestrator
python3 scripts/render_cloud.py \
  --manifest /path/to/job_manifest.json \
  --output /path/to/final/output.mp4

# Dry run (validates schema & rate limits without triggering run)
python3 scripts/render_cloud.py \
  --manifest /path/to/job_manifest.json \
  --dry-run
```

## Security & Fail-Closed Guardrails

- Subtitles and text overlays are sanitized against command injection and shell escapement.
- Artifact sizes are checked for 0-byte failures and upper thresholds.
- Bitstream integrity is verified with `ffprobe` null-muxer decode test before marking success.
- Final publishing to social platforms remains strictly locked to human review (Human Gate 4).
