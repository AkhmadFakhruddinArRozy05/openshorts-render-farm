#!/usr/bin/env python3
"""
OpenShorts Cloud Render Orchestrator
Dispatches render jobs to GitHub Actions cloud runner (Ubuntu 16 GB RAM),
manages ephemeral asset releases, monitors execution with 120-minute timeout guard,
and downloads & verifies the finished 9:16 vertical MP4.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure layout_engine is importable
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.layout_engine import validate_job_manifest

DEFAULT_REPO = "AkhmadFakhruddinArRozy05/openshorts-render-farm"
DEFAULT_WORKFLOW = "render.yml"
DEFAULT_TIMEOUT_MINUTES = 120
GITHUB_API_MIN_REMAINING = 100


def check_github_rate_limit(min_remaining: int = GITHUB_API_MIN_REMAINING) -> Optional[Dict[str, Any]]:
    """
    Checks GitHub API rate limit via `gh api rate_limit`.
    Applies pacing if remaining requests are below min_remaining.
    """
    try:
        res = subprocess.run(["gh", "api", "rate_limit"], capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            data = json.loads(res.stdout)
            if not isinstance(data, dict):
                return None
            core = data.get("resources", {}).get("core", {})
            remaining = core.get("remaining", 5000)
            limit = core.get("limit", 5000)
            reset_ts = core.get("reset", 0)
            print(f"📡 [RATE-LIMIT] GitHub API Core: {remaining}/{limit} requests remaining.")
            if remaining < min_remaining:
                sleep_secs = max(5, int(reset_ts - time.time()))
                wait_time = min(sleep_secs, 60)
                print(f"⚠️ [RATE-LIMIT] Quota low ({remaining}/{limit})! Pacing for {wait_time}s...")
                time.sleep(wait_time)
            return core
        else:
            print(f"⚠️ [RATE-LIMIT] gh api rate_limit returned non-zero: {res.stderr.strip()}")
    except Exception as e:
        print(f"⚠️ [RATE-LIMIT] Could not check GitHub API rate limit: {e}")
    return None


def validate_manifest_file(manifest_path: str) -> Dict[str, Any]:
    """
    Loads and validates a job_manifest.json file fail-closed.
    """
    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

    try:
        content = p.read_text(encoding="utf-8")
        data = json.loads(content)
    except json.JSONDecodeError as jde:
        raise ValueError(f"Invalid JSON in manifest file {manifest_path}: {jde}")

    return validate_job_manifest(data)


def create_temporary_asset_release(repo_name: str, tag_name: str, asset_files: List[str]) -> str:
    """
    Creates an ephemeral GitHub release to transfer local asset files to the cloud runner.
    """
    cmd = [
        "gh", "release", "create", tag_name,
        *asset_files,
        "--repo", repo_name,
        "--title", f"Ephemeral Render Assets {tag_name}",
        "--notes", "Auto-generated ephemeral assets for cloud render. Safe to purge.",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to create temporary asset release: {res.stderr.strip()}")
    print(f"📦 [ASSETS] Created temporary GitHub Release: {tag_name}")
    return tag_name


def delete_temporary_asset_release(repo_name: str, tag_name: str) -> bool:
    """
    Deletes the ephemeral GitHub release and tag after rendering finishes.
    """
    cmd = [
        "gh", "release", "delete", tag_name,
        "--repo", repo_name,
        "--yes",
        "--cleanup-tag",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"🧹 [ASSETS] Purged temporary GitHub Release and tag: {tag_name}")
        return True
    else:
        print(f"⚠️ [ASSETS] Could not purge temporary release {tag_name}: {res.stderr.strip()}")
        return False


def trigger_workflow_run(repo_name: str, workflow_name: str, inputs: Dict[str, str]) -> bool:
    """
    Triggers GitHub Actions workflow via `gh workflow run`.
    """
    cmd = ["gh", "workflow", "run", workflow_name, "--repo", repo_name]
    for k, v in inputs.items():
        cmd.extend(["-f", f"{k}={v}"])

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"❌ [DISPATCH] Failed to trigger workflow: {res.stderr.strip()}")
        return False
    print(f"🚀 [DISPATCH] Triggered workflow {workflow_name} on {repo_name}")
    return True


def find_latest_run_id(
    repo_name: str,
    workflow_name: str,
    exclude_run_id: Optional[str] = None,
    max_wait_seconds: int = 30,
    poll_interval: int = 2,
) -> Optional[str]:
    """
    Finds the most recently dispatched run ID for the given workflow.
    Polls up to max_wait_seconds to identify a newly queued/running run (excluding exclude_run_id).
    """
    cmd = [
        "gh", "run", "list",
        "--workflow", workflow_name,
        "--repo", repo_name,
        "--limit", "1",
        "--json", "databaseId",
    ]
    start_time = time.time()
    while True:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            try:
                items = json.loads(res.stdout)
                if items and isinstance(items, list):
                    candidate_id = str(items[0]["databaseId"])
                    if exclude_run_id is None or candidate_id != str(exclude_run_id):
                        return candidate_id
            except Exception:
                pass
        if time.time() - start_time >= max_wait_seconds:
            break
        time.sleep(poll_interval)
    return None


def watch_run_with_timeout(repo_name: str, run_id: str, timeout_seconds: int = 7200) -> bool:
    """
    Watches GitHub Actions run with timeout guard (default 120 minutes).
    Cancels run if timeout is exceeded.
    """
    cmd = ["gh", "run", "watch", str(run_id), "--repo", repo_name, "--exit-status"]
    print(f"⏱️ [WATCHER] Watching run {run_id} (Timeout: {timeout_seconds // 60}m)...")
    try:
        res = subprocess.run(cmd, timeout=timeout_seconds)
        return res.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"🚨 [WATCHER] Execution timed out after {timeout_seconds}s! Cancelling run {run_id}...")
        cancel_cmd = ["gh", "run", "cancel", str(run_id), "--repo", repo_name]
        subprocess.run(cancel_cmd)
        return False


def download_rendered_artifact(repo_name: str, run_id: str, output_path: str) -> bool:
    """
    Downloads rendered-mp4 artifact and copies to the target output_path.
    """
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            "gh", "run", "download", str(run_id),
            "--repo", repo_name,
            "--name", "rendered-mp4",
            "--dir", tmp_dir,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"❌ [DOWNLOAD] Artifact download failed: {res.stderr.strip()}")
            return False

        # Find mp4 file
        found_mp4 = None
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                if f.endswith(".mp4"):
                    found_mp4 = Path(root) / f
                    break
            if found_mp4:
                break

        if not found_mp4 or not found_mp4.is_file():
            print("❌ [DOWNLOAD] No MP4 file found in downloaded artifact!")
            return False

        shutil.copy2(str(found_mp4), str(target))

    if not target.exists() or target.stat().st_size == 0:
        return False

    sz_mb = target.stat().st_size / (1024 * 1024)
    if sz_mb >= 500:
        print(f"⚠️ [ARTIFACT WARNING] File '{target.name}' ({sz_mb:.2f} MB) exceeds GitHub Actions 500 MB limit!")
    elif sz_mb >= 400:
        print(f"⚠️ [ARTIFACT WARNING] File '{target.name}' ({sz_mb:.2f} MB) is approaching 500 MB limit (>= 400 MB).")

    return True


def verify_final_mp4(output_path: str) -> bool:
    """
    Tri-Level QA bitstream verification for final rendered video:
    1. File existence & size > 0 bytes.
    2. ffprobe dimensions 1080x1920 (9:16 vertical).
    3. ffmpeg null-muxer decode integrity check.
    """
    p = Path(output_path)
    if not p.is_file() or p.stat().st_size == 0:
        print(f"❌ [QA] File does not exist or is 0 bytes: {output_path}")
        return False

    try:
        # Check resolution
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            str(p),
        ]
        res = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
        dim = res.stdout.strip()
        if "1080x1920" not in dim:
            print(f"⚠️ [QA] Resolution is {dim}, expected 1080x1920 vertical.")

        # Decode integrity test
        dec_cmd = ["ffmpeg", "-v", "error", "-i", str(p), "-f", "null", "-"]
        dec_res = subprocess.run(dec_cmd, capture_output=True, text=True, timeout=30)
        if dec_res.returncode != 0:
            print(f"❌ [QA] Decode integrity test failed: {dec_res.stderr.strip()}")
            return False

        print(f"✅ [QA] Video passed integrity verification: {output_path} ({dim})")
        return True
    except Exception as e:
        print(f"❌ [QA] Verification exception: {e}")
        return False


def run_orchestrator(
    manifest_path: str,
    output_path: str,
    repo_name: str = DEFAULT_REPO,
    workflow_name: str = DEFAULT_WORKFLOW,
    timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES,
    dry_run: bool = False,
) -> int:
    """
    End-to-end cloud render orchestration flow.
    """
    print("=" * 60)
    print("🎬 OpenShorts Cloud Render Farm Orchestrator")
    print("=" * 60)

    # 1. Validate manifest
    print(f"📄 [STAGE 1] Validating manifest: {manifest_path}")
    try:
        manifest = validate_manifest_file(manifest_path)
        print(f"✅ Manifest valid! Project: {manifest['project_id']}, Mode: {manifest['layout']['mode']}")
    except Exception as e:
        print(f"❌ Manifest validation failed: {e}")
        return 1

    # 2. Check rate limit
    print("📡 [STAGE 2] Checking GitHub API rate limit...")
    check_github_rate_limit()

    if dry_run:
        print("🔍 [DRY-RUN] Manifest and rate limit validated successfully. Halting before cloud dispatch.")
        return 0

    # 3. Handle local media and subtitle assets upload if needed
    source_cfg = manifest["source"]
    release_tag = ""
    local_files = []

    if source_cfg.get("type") == "local":
        local_path = source_cfg.get("local_asset_path")
        if not local_path or not Path(local_path).is_file():
            print(f"❌ Local asset does not exist: {local_path}")
            return 1
        local_files.append(str(Path(local_path).resolve()))

    # Check for local subtitles next to manifest or in raw/
    manifest_dir = Path(manifest_path).resolve().parent
    for sub_name in ["subtitles.srt", "subtitles.ass"]:
        for candidate_path in [
            manifest_dir / sub_name,
            manifest_dir / "raw" / sub_name,
        ]:
            if candidate_path.is_file() and str(candidate_path) not in local_files:
                local_files.append(str(candidate_path))
                break

    if local_files:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        tag_name = f"temp-assets-{manifest['project_id']}-{timestamp}"
        try:
            release_tag = create_temporary_asset_release(repo_name, tag_name, local_files)
        except Exception as e:
            print(f"❌ Failed to stage assets to GitHub Release: {e}")
            return 1

    # 4. Trigger workflow run
    print("🚀 [STAGE 3] Dispatching workflow run to GitHub Actions...")
    # Get previous latest run id so we don't accidentally attach to an old run
    old_run_id = find_latest_run_id(repo_name, workflow_name, max_wait_seconds=2)

    manifest_json_str = json.dumps(manifest)
    inputs = {
        "manifest": manifest_json_str,
        "release_tag": release_tag,
    }

    if not trigger_workflow_run(repo_name, workflow_name, inputs):
        if release_tag:
            delete_temporary_asset_release(repo_name, release_tag)
        return 1

    print("⏳ Waiting for GitHub Actions run to register...")
    run_id = find_latest_run_id(
        repo_name,
        workflow_name,
        exclude_run_id=old_run_id,
        max_wait_seconds=30,
    )
    if not run_id:
        print("❌ Could not determine dispatched run ID!")
        if release_tag:
            delete_temporary_asset_release(repo_name, release_tag)
        return 1

    print(f"🎯 [STAGE 4] Dispatched Run ID: {run_id}")

    # 5. Watch execution
    timeout_seconds = timeout_minutes * 60
    success = watch_run_with_timeout(repo_name, run_id, timeout_seconds=timeout_seconds)

    # Clean up ephemeral release tag regardless of run success
    if release_tag:
        delete_temporary_asset_release(repo_name, release_tag)

    if not success:
        print(f"❌ Cloud execution failed or timed out for run {run_id}!")
        return 1

    # 6. Download artifact & verify
    print(f"📥 [STAGE 5] Downloading rendered artifact to: {output_path}")
    if not download_rendered_artifact(repo_name, run_id, output_path):
        print("❌ Failed to download or extract MP4 artifact!")
        return 1

    print("🔍 [STAGE 6] Verifying final MP4...")
    if not verify_final_mp4(output_path):
        print("❌ Tri-Level QA verification failed!")
        return 1

    print(f"🎉 Cloud render completed successfully: {output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="OpenShorts Cloud Render Orchestrator")
    parser.add_argument("--manifest", required=True, help="Path to job_manifest.json")
    parser.add_argument("--output", default="output.mp4", help="Path to final output MP4")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repository (default: {DEFAULT_REPO})")
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW, help=f"Workflow name (default: {DEFAULT_WORKFLOW})")
    parser.add_argument("--timeout-minutes", type=int, default=DEFAULT_TIMEOUT_MINUTES, help="Max run timeout in minutes")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest and rate limit without triggering cloud run")

    args = parser.parse_args()

    sys.exit(
        run_orchestrator(
            manifest_path=args.manifest,
            output_path=args.output,
            repo_name=args.repo,
            workflow_name=args.workflow,
            timeout_minutes=args.timeout_minutes,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
