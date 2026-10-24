"""
Unit tests for OpenShorts Cloud Render Orchestrator (render_cloud.py)
"""

import json
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.render_openshorts_cloud import (
    check_github_rate_limit,
    validate_manifest_file,
    create_temporary_asset_release,
    delete_temporary_asset_release,
    trigger_workflow_run,
    download_rendered_artifact,
    verify_final_mp4,
)


@pytest.fixture
def valid_manifest_dict():
    return {
        "project_id": "TEST-PROJ-01",
        "source": {
            "type": "url",
            "url": "https://www.youtube.com/watch?v=12345",
            "local_asset_path": None,
        },
        "timing": {
            "start_seconds": 10.0,
            "end_seconds": 40.0,
            "duration_seconds": 30.0,
        },
        "layout": {
            "mode": "TRACK",
        },
        "output_spec": {
            "resolution": [1080, 1920],
            "fps": 30,
            "codec": "libx264",
        },
    }


def test_validate_manifest_file_success(tmp_path, valid_manifest_dict):
    m_path = tmp_path / "job_manifest.json"
    m_path.write_text(json.dumps(valid_manifest_dict), encoding="utf-8")

    manifest = validate_manifest_file(str(m_path))
    assert manifest["project_id"] == "TEST-PROJ-01"
    assert manifest["layout"]["mode"] == "TRACK"


def test_validate_manifest_file_not_found():
    with pytest.raises(FileNotFoundError):
        validate_manifest_file("/path/to/nonexistent/manifest.json")


def test_validate_manifest_file_invalid_json(tmp_path):
    m_path = tmp_path / "bad_manifest.json"
    m_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        validate_manifest_file(str(m_path))


def test_check_github_rate_limit_sufficient():
    mock_out = json.dumps({
        "resources": {
            "core": {
                "limit": 5000,
                "remaining": 4900,
                "reset": 1700000000,
            }
        }
    })
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_out, stderr="")
        core = check_github_rate_limit(min_remaining=100)
        assert core is not None
        assert core["remaining"] == 4900


def test_check_github_rate_limit_low_quota_pacing():
    mock_out = json.dumps({
        "resources": {
            "core": {
                "limit": 5000,
                "remaining": 50,
                "reset": 1700000000,
            }
        }
    })
    with patch("subprocess.run") as mock_run, patch("time.sleep") as mock_sleep:
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_out, stderr="")
        core = check_github_rate_limit(min_remaining=100)
        assert core["remaining"] == 50
        mock_sleep.assert_called_once()


def test_create_temporary_asset_release(tmp_path):
    asset_file = tmp_path / "raw_video.mp4"
    asset_file.write_bytes(b"dummy video data")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="release created", stderr="")
        tag = create_temporary_asset_release(
            repo_name="AkhmadFakhruddinArRozy05/openshorts-render-farm",
            tag_name="temp-assets-test-1234",
            asset_files=[str(asset_file)],
        )
        assert tag == "temp-assets-test-1234"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "release" in cmd
        assert "create" in cmd
        assert "temp-assets-test-1234" in cmd


def test_delete_temporary_asset_release():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="deleted", stderr="")
        res = delete_temporary_asset_release(
            repo_name="AkhmadFakhruddinArRozy05/openshorts-render-farm",
            tag_name="temp-assets-test-1234",
        )
        assert res is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "delete" in cmd
        assert "--cleanup-tag" in cmd


def test_trigger_workflow_run():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        success = trigger_workflow_run(
            repo_name="AkhmadFakhruddinArRozy05/openshorts-render-farm",
            workflow_name="render.yml",
            inputs={"manifest": '{"project_id":"TEST"}', "release_tag": ""},
        )
        assert success is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "workflow" in cmd
        assert "run" in cmd
        assert "render.yml" in cmd


def test_download_rendered_artifact(tmp_path):
    out_file = tmp_path / "final" / "output.mp4"
    run_id = "12345678"

    def mock_download_side_effect(cmd, **kwargs):
        # Emulate gh run download writing output.mp4 inside download dir
        dest_dir = Path(cmd[cmd.index("--dir") + 1])
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "output.mp4").write_bytes(b"dummy rendered mp4")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=mock_download_side_effect):
        ok = download_rendered_artifact(
            repo_name="AkhmadFakhruddinArRozy05/openshorts-render-farm",
            run_id=run_id,
            output_path=str(out_file),
        )
        assert ok is True
        assert out_file.exists()
        assert out_file.read_bytes() == b"dummy rendered mp4"


def test_verify_final_mp4(tmp_path):
    mp4 = tmp_path / "output.mp4"
    mp4.write_bytes(b"dummy")

    with patch("subprocess.run") as mock_run:
        # ffprobe returns 1080x1920, and ffmpeg null-muxer decode succeeds
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="1080x1920\n", stderr=""),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]
        assert verify_final_mp4(str(mp4)) is True

    # Test failure on 0 bytes
    empty_mp4 = tmp_path / "empty.mp4"
    empty_mp4.write_bytes(b"")
    assert verify_final_mp4(str(empty_mp4)) is False
