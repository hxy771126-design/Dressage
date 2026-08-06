import csv
import os
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MONITOR = REPO_ROOT / "examples" / "scripts" / "gpu_utilization_monitor.sh"


def _monitor_env(tmp_path: Path, fake_bin: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if fake_bin is not None:
        env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env.update(
        {
            "GPU_UTIL_STATE_DIR": str(tmp_path / "state"),
            "GPU_UTIL_LOG_DIR": str(tmp_path / "log"),
            "GPU_UTIL_SAMPLE_INTERVAL": "0.05",
            "GPU_UTIL_GPU_IDS": "",
        }
    )
    return env


def _run_monitor(
    action: str,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(MONITOR), action],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def _write_state(tmp_path: Path, csv_contents: str) -> dict[str, str]:
    state_dir = tmp_path / "state"
    log_dir = tmp_path / "log"
    state_dir.mkdir()
    log_dir.mkdir()

    csv_file = log_dir / "samples.csv"
    summary_file = log_dir / "samples.summary.txt"
    csv_file.write_text(csv_contents, encoding="utf-8")

    (state_dir / "pid").write_text("999999999\n", encoding="utf-8")
    (state_dir / "csv_file").write_text(f"{csv_file}\n", encoding="utf-8")
    (state_dir / "summary_file").write_text(
        f"{summary_file}\n", encoding="utf-8"
    )
    (state_dir / "start_epoch").write_text(
        f"{int(time.time()) - 1}\n", encoding="utf-8"
    )
    (state_dir / "start_time").write_text(
        "2026-08-02T14:12:01+00:00\n", encoding="utf-8"
    )
    (state_dir / "interval").write_text("1\n", encoding="utf-8")
    return _monitor_env(tmp_path)


def test_monitor_reports_real_gpu_averages_and_four_column_csv(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' '0, GPU-zero, 25' '1, GPU-one, 75'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    env = _monitor_env(tmp_path, fake_bin)

    started = _run_monitor("start", env)
    assert started.returncode == 0, started.stderr

    try:
        time.sleep(0.2)
        stopped = _run_monitor("stop", env)
    finally:
        if (tmp_path / "state" / "pid").exists():
            _run_monitor("stop", env)

    assert stopped.returncode == 0, stopped.stderr
    assert "status: ok" in stopped.stdout
    assert "average_gpu_utilization_percent: 50.00" in stopped.stdout
    assert "GPU 0 (GPU-zero): avg=25.00%" in stopped.stdout
    assert "GPU 1 (GPU-one): avg=75.00%" in stopped.stdout

    csv_files = list((tmp_path / "log").glob("gpu_utilization_*.csv"))
    assert len(csv_files) == 1
    with csv_files[0].open(newline="", encoding="utf-8") as csv_stream:
        rows = list(csv.reader(csv_stream))

    assert len(rows) > 1
    assert all(len(row) == 4 for row in rows)
    assert {row[1] for row in rows[1:]} == {"0", "1"}


def test_monitor_rejects_legacy_comma_timestamp_rows(tmp_path: Path) -> None:
    env = _write_state(
        tmp_path,
        "timestamp,gpu_index,gpu_uuid,utilization_percent\n"
        "2026-08-02T14:12:01,946737542+00:00,0,GPU-zero,75\n",
    )

    stopped = _run_monitor("stop", env)

    assert stopped.returncode != 0
    assert "status: error" in stopped.stdout
    assert "valid_readings: 0" in stopped.stdout
    assert "invalid_readings: 1" in stopped.stdout
    assert "average_gpu_utilization_percent: N/A" in stopped.stdout
    assert "GPU 946737542+00:00" not in stopped.stdout
    assert "avg=0.00%" not in stopped.stdout


def test_monitor_fails_when_there_are_no_samples(tmp_path: Path) -> None:
    env = _write_state(
        tmp_path,
        "timestamp,gpu_index,gpu_uuid,utilization_percent\n",
    )

    stopped = _run_monitor("stop", env)

    assert stopped.returncode != 0
    assert "status: error" in stopped.stdout
    assert "valid_readings: 0" in stopped.stdout
    assert "invalid_readings: 0" in stopped.stdout
    assert "average_gpu_utilization_percent: N/A" in stopped.stdout
