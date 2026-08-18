#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SOURCE_RECIPE="${SCRIPT_DIR}/run_blackbox_qwen3.5_4b_sync_local_l3_hicache.sh"
LONG_TAIL_TOOL="${REPO_ROOT}/examples/data/prepare_dapo_long_tail.py"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${REPO_ROOT}/log/benchmarks/engine_rebalancing}"
PROMPT_SOURCE="${LONG_TAIL_PROMPT_DATA:-${REPO_ROOT}/examples/data/dressage_dapo_prompts_step_balanced_256.jsonl}"
PROMPT_EFFECTIVE="${BENCHMARK_ROOT}/prompts.deterministic.jsonl"

BENCHMARK_SEED="${BENCHMARK_SEED:-20260806}"
BENCHMARK_DRY_RUN="${BENCHMARK_DRY_RUN:-0}"
ENGINE_REBALANCING_LOAD_BATCH_COALESCING_WINDOW_MS="${ENGINE_REBALANCING_LOAD_BATCH_COALESCING_WINDOW_MS:-60}"
ENGINE_REBALANCING_MIN_LOAD_IMPROVEMENT_RATIO="${ENGINE_REBALANCING_MIN_LOAD_IMPROVEMENT_RATIO:-0.10}"

# These values intentionally are not tunable in the A/B benchmark. Keeping the
# workload fixed is part of the validity check.
ROLLOUT_TEMPERATURE=0
ROLLOUT_BATCH_SIZE=256
N_SAMPLES_PER_PROMPT=1
GLOBAL_BATCH_SIZE=256
ROLLOUT_MAX_RESPONSE_LEN=12288
DRESSAGE_BLACKBOX_SLOTS_PER_NODE=24
DRESSAGE_BLACKBOX_ACQUIRE_TIMEOUT_SEC=3600
DRESSAGE_BLACKBOX_MAX_STEPS=20
MOONCAKE_GLOBAL_SEGMENT_SIZE=24gb

RUN_NAMES=(
  "seed${BENCHMARK_SEED}-off-r1"
  "seed${BENCHMARK_SEED}-on-r1"
)
RUN_MODES=(off on)

print_plan() {
  echo "Engine Rebalancing A/B benchmark"
  echo "  source recipe: ${SOURCE_RECIPE}"
  echo "  output root:   ${BENCHMARK_ROOT}"
  echo "  prompt source: ${PROMPT_SOURCE}"
  echo "  prompt data:   ${PROMPT_EFFECTIVE}"
  echo "  seed:          ${BENCHMARK_SEED}"
  echo "  temperature:   ${ROLLOUT_TEMPERATURE}"
  echo "  rollout batch: ${ROLLOUT_BATCH_SIZE}"
  echo "  samples/prompt:${N_SAMPLES_PER_PROMPT}"
  echo "  global batch:  ${GLOBAL_BATCH_SIZE}"
  echo "  response max:  ${ROLLOUT_MAX_RESPONSE_LEN}"
  echo "  sandbox slots: ${DRESSAGE_BLACKBOX_SLOTS_PER_NODE}"
  echo "  slot timeout:  ${DRESSAGE_BLACKBOX_ACQUIRE_TIMEOUT_SEC}"
  echo "  Blackbox max steps: ${DRESSAGE_BLACKBOX_MAX_STEPS}"
  echo "  Mooncake size: ${MOONCAKE_GLOBAL_SEGMENT_SIZE}"
  echo "  load batch window: ${ENGINE_REBALANCING_LOAD_BATCH_COALESCING_WINDOW_MS} ms"
  echo "  min load improvement ratio: ${ENGINE_REBALANCING_MIN_LOAD_IMPROVEMENT_RATIO}"
  echo "  fixed flags:   --seed ${BENCHMARK_SEED} --rollout-seed ${BENCHMARK_SEED}"
  echo "                 --sglang-enable-deterministic-inference"
  echo "                 --sglang-enable-cache-report"
  echo "  OFF Proxy:     no --enable-engine-rebalancing"
  echo "  ON Proxy:      --enable-engine-rebalancing; calibration must be READY"
  echo "  matrix:"
  local index mode
  for index in "${!RUN_NAMES[@]}"; do
    mode="${RUN_MODES[${index}]}"
    printf '    %d. %-36s mode=%-3s %s\n' \
      "$((index + 1))" "${RUN_NAMES[${index}]}" "${mode}" "measured"
  done
}

if [[ "${BENCHMARK_DRY_RUN}" != "0" && "${BENCHMARK_DRY_RUN}" != "1" ]]; then
  echo "BENCHMARK_DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${ENGINE_REBALANCING_LOAD_BATCH_COALESCING_WINDOW_MS}" =~ ^[0-9]+$ ]]; then
  echo "ENGINE_REBALANCING_LOAD_BATCH_COALESCING_WINDOW_MS must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "${ENGINE_REBALANCING_MIN_LOAD_IMPROVEMENT_RATIO}" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
  echo "ENGINE_REBALANCING_MIN_LOAD_IMPROVEMENT_RATIO must be a number between 0 and 1" >&2
  exit 2
fi

if [[ ! -f "${SOURCE_RECIPE}" ]]; then
  echo "Cannot find source recipe: ${SOURCE_RECIPE}" >&2
  exit 1
fi
if [[ ! -f "${LONG_TAIL_TOOL}" ]]; then
  echo "Cannot find long-tail preparation tool: ${LONG_TAIL_TOOL}" >&2
  exit 1
fi

print_plan
if [[ "${BENCHMARK_DRY_RUN}" == "1" ]]; then
  echo "Dry-run only; no services or output directories were created."
  exit 0
fi

for command_name in python3 git curl tee; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 1
  fi
done

# Refuse to mix a new matrix with old artifacts. To retain multiple matrices,
# point BENCHMARK_ROOT at a new directory for each invocation.
for run_name in "${RUN_NAMES[@]}"; do
  if [[ -e "${BENCHMARK_ROOT}/${run_name}" ]]; then
    echo "Benchmark output already exists: ${BENCHMARK_ROOT}/${run_name}" >&2
    echo "Remove it explicitly or use a different BENCHMARK_ROOT." >&2
    exit 1
  fi
done

mkdir -p "${BENCHMARK_ROOT}"

prepare_long_tail_prompts() {
  local source_path="$1"
  local output_path="$2"
  local seed="$3"

  python3 "${LONG_TAIL_TOOL}" sample \
    --input "${source_path}" \
    --output "${output_path}" \
    --seed "${seed}"
}

prepare_long_tail_prompts "${PROMPT_SOURCE}" "${PROMPT_EFFECTIVE}" "${BENCHMARK_SEED}"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dressage-rebalancing-benchmark.XXXXXX")"
trap 'rm -rf -- "${TEMP_DIR}"' EXIT

record_environment() {
  local output_path="$1"
  local run_name="$2"
  local mode="$3"

  python3 - \
    "${REPO_ROOT}" \
    "${SOURCE_RECIPE}" \
    "${BASH_SOURCE[0]}" \
    "${PROMPT_SOURCE}" \
    "${PROMPT_EFFECTIVE}" \
    "${output_path}" \
    "${run_name}" \
    "${mode}" \
    "${BENCHMARK_SEED}" \
    "${ROLLOUT_TEMPERATURE}" \
    "${ROLLOUT_BATCH_SIZE}" \
    "${N_SAMPLES_PER_PROMPT}" \
    "${GLOBAL_BATCH_SIZE}" \
    "${ROLLOUT_MAX_RESPONSE_LEN}" \
    "${DRESSAGE_BLACKBOX_SLOTS_PER_NODE}" \
    "${DRESSAGE_BLACKBOX_ACQUIRE_TIMEOUT_SEC}" \
    "${DRESSAGE_BLACKBOX_MAX_STEPS}" \
    "${MOONCAKE_GLOBAL_SEGMENT_SIZE}" \
    "${ENGINE_REBALANCING_LOAD_BATCH_COALESCING_WINDOW_MS}" \
    "${ENGINE_REBALANCING_MIN_LOAD_IMPROVEMENT_RATIO}" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import socket
import subprocess
import sys
from collections import Counter

(
    repo,
    source_recipe,
    benchmark_script,
    prompt_source,
    prompt_effective,
    output,
    run_name,
    mode,
    seed,
    rollout_temperature,
    rollout_batch_size,
    n_samples_per_prompt,
    global_batch_size,
    rollout_max_response_len,
    sandbox_slots_per_node,
    sandbox_acquire_timeout_sec,
    blackbox_max_steps,
    mooncake_global_segment_size,
    load_batch_coalescing_window_ms,
    min_load_improvement_ratio,
) = sys.argv[1:]
repo_path = pathlib.Path(repo)


def command(args: list[str], *, cwd: pathlib.Path = repo_path) -> str:
    proc = subprocess.run(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.stdout.decode("utf-8", errors="replace").strip()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: str) -> str:
    return digest(pathlib.Path(path).read_bytes())


def workload_distribution(path: str) -> str:
    counts: Counter[str] = Counter()
    try:
        lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        for line in lines:
            row = json.loads(line)
            workload_class = row.get("metadata", {}).get("workload_class")
            if isinstance(workload_class, str):
                counts[workload_class] += 1
    except (OSError, json.JSONDecodeError, AttributeError):
        return "{}"
    return json.dumps(dict(sorted(counts.items())), sort_keys=True, separators=(",", ":"))


git_head = command(["git", "rev-parse", "HEAD"])
slime_head = command(["git", "-C", "slime", "rev-parse", "HEAD"])
slime_gitlink = command(["git", "ls-tree", "HEAD", "slime"])
tracked_diff = subprocess.run(
    ["git", "diff", "--binary", "HEAD"],
    cwd=repo_path,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
).stdout
slime_diff = subprocess.run(
    ["git", "diff", "--binary", "HEAD"],
    cwd=repo_path / "slime",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
).stdout

gpu_inventory = command(
    [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version",
        "--format=csv,noheader,nounits",
    ]
)
gpu_lines = [line.strip() for line in gpu_inventory.splitlines() if line.strip()]

fingerprint_payload = {
    "git_head": git_head,
    "slime_head": slime_head,
    "slime_gitlink": slime_gitlink,
    "tracked_diff_sha256": digest(tracked_diff),
    "slime_diff_sha256": digest(slime_diff),
    "source_recipe_sha256": file_digest(source_recipe),
    "benchmark_script_sha256": file_digest(benchmark_script),
    "prompt_source_sha256": file_digest(prompt_source),
    "prompt_effective_sha256": file_digest(prompt_effective),
}
code_fingerprint = digest(
    json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
)

values = {
    "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "run_name": run_name,
    "mode": mode,
    "hostname": socket.gethostname(),
    "gpu_count": len(gpu_lines),
    "gpu_inventory_json": json.dumps(gpu_lines, ensure_ascii=False),
    "gpu_inventory_sha256": digest(gpu_inventory.encode()),
    "git_head": git_head,
    "slime_head": slime_head,
    "slime_gitlink": slime_gitlink,
    "worktree_diff_sha256": digest(tracked_diff),
    "slime_worktree_diff_sha256": digest(slime_diff),
    "source_recipe_sha256": fingerprint_payload["source_recipe_sha256"],
    "benchmark_script_sha256": fingerprint_payload["benchmark_script_sha256"],
    "prompt_source": prompt_source,
    "prompt_source_sha256": fingerprint_payload["prompt_source_sha256"],
    "prompt_source_workload_distribution_json": workload_distribution(prompt_source),
    "prompt_effective": prompt_effective,
    "prompt_effective_sha256": fingerprint_payload["prompt_effective_sha256"],
    "prompt_effective_workload_distribution_json": workload_distribution(prompt_effective),
    "code_fingerprint": code_fingerprint,
    "benchmark_seed": seed,
    "rollout_temperature": rollout_temperature,
    "rollout_batch_size": rollout_batch_size,
    "n_samples_per_prompt": n_samples_per_prompt,
    "global_batch_size": global_batch_size,
    "rollout_max_response_len": rollout_max_response_len,
    "sandbox_slots_per_node": sandbox_slots_per_node,
    "sandbox_acquire_timeout_sec": sandbox_acquire_timeout_sec,
    "blackbox_max_steps": blackbox_max_steps,
    "mooncake_global_segment_size": mooncake_global_segment_size,
    "load_batch_coalescing_window_ms": load_batch_coalescing_window_ms,
    "min_load_improvement_ratio": min_load_improvement_ratio,
    "engine_load_snapshot_interval_seconds": 5,
}

path = pathlib.Path(output)
path.write_text(
    "".join(f"{key}={value}\n" for key, value in values.items()),
    encoding="utf-8",
)
PY
}

build_temporary_recipe() {
  local output_path="$1"
  local mode="$2"

  python3 - "${SOURCE_RECIPE}" "${output_path}" "${mode}" <<'PY'
from __future__ import annotations

import pathlib
import sys

source, output, mode = sys.argv[1:]
if mode not in {"off", "on"}:
    raise SystemExit(f"unsupported benchmark mode: {mode}")

text = pathlib.Path(source).read_text(encoding="utf-8")

# A generated recipe lives outside examples/scripts. Keep all relative sources
# anchored to the original recipe directory supplied by the benchmark wrapper.
script_dir_line = (
    'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" '
    '&>/dev/null && pwd)"'
)
replacement = (
    'SCRIPT_DIR="${DRESSAGE_BENCHMARK_SOURCE_SCRIPT_DIR:-'
    '$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)}"'
)
if script_dir_line not in text:
    raise SystemExit("source recipe SCRIPT_DIR marker changed")
text = text.replace(script_dir_line, replacement, 1)

proxy_flag = "  --enable-engine-rebalancing\n"
coalescing_flag = (
    "  --engine-rebalancing-load-batch-coalescing-window-ms "
    '"${ENGINE_REBALANCING_LOAD_BATCH_COALESCING_WINDOW_MS}"\n'
)
improvement_flag = (
    "  --engine-rebalancing-min-load-improvement-ratio "
    '"${ENGINE_REBALANCING_MIN_LOAD_IMPROVEMENT_RATIO}"\n'
)
if mode == "off":
    if proxy_flag not in text:
        raise SystemExit("source recipe no longer contains the rebalancing flag")
    text = text.replace(proxy_flag, "", 1)
else:
    if text.count(proxy_flag) != 1:
        raise SystemExit("expected exactly one rebalancing flag in source recipe")
    text = text.replace(
        proxy_flag,
        proxy_flag + coalescing_flag + improvement_flag,
        1,
    )

if '--rollout-seed "${BENCHMARK_SEED}"' not in text:
    marker = "  --rollout-shuffle\n"
    if marker not in text:
        raise SystemExit("source recipe rollout marker changed")
    text = text.replace(
        marker,
        marker + '  --rollout-seed "${BENCHMARK_SEED}"\n',
        1,
    )

if "--sglang-enable-deterministic-inference" not in text:
    marker = "  --sglang-mem-fraction-static 0.7\n"
    if marker not in text:
        raise SystemExit("source recipe SGLang marker changed")
    text = text.replace(
        marker,
        marker
        + "  --sglang-enable-deterministic-inference\n"
        + "  --sglang-enable-cache-report\n",
        1,
    )
elif "--sglang-enable-cache-report" not in text:
    marker = "  --sglang-enable-deterministic-inference\n"
    text = text.replace(marker, marker + "  --sglang-enable-cache-report\n", 1)

if '--seed "${BENCHMARK_SEED}"' not in text:
    marker = "  --debug-rollout-only \\\n"
    if marker not in text:
        raise SystemExit("source recipe training command marker changed")
    text = text.replace(
        marker,
        marker + '  --seed "${BENCHMARK_SEED}" \\\n',
        1,
    )

calibration_start = 'CALIBRATION_STATE=""\n'
calibration_end = 'if [[ "${DRESSAGE_SANDBOX_PROVIDER}" == "local_bwrap"'
start = text.find(calibration_start)
end = text.find(calibration_end, start)
if start < 0 or end < 0:
    raise SystemExit("source recipe calibration block markers changed")

if mode == "on":
    calibration = r'''CALIBRATION_STARTED_EPOCH="$(date +%s)"
CALIBRATION_STATE=""
for i in $(seq 1 900); do
  CALIBRATION_STATE="$(
    curl -sf "${DRESSAGE_PROXY_URL}/v1/engines/calibration" |
      python3 -c 'import json,sys; print(json.load(sys.stdin).get("state", ""))'
  )" || true
  if [[ "${CALIBRATION_STATE}" == "READY" ]]; then
    echo "Engine rebalancing calibration reached READY"
    break
  fi
  if [[ "${CALIBRATION_STATE}" == "DEGRADED" ]]; then
    echo "Engine rebalancing calibration reached DEGRADED; benchmark requires READY" >&2
    exit 1
  fi
  if [[ "${i}" -eq 900 ]]; then
    echo "Engine rebalancing calibration did not reach READY" >&2
    exit 1
  fi
  sleep 1
done
CALIBRATION_ENDED_EPOCH="$(date +%s)"
echo "BENCHMARK_CALIBRATION_SECONDS=$((CALIBRATION_ENDED_EPOCH - CALIBRATION_STARTED_EPOCH))"

'''
else:
    calibration = '''CALIBRATION_STATE="OFF"
echo "Engine rebalancing disabled; skipping calibration wait"
echo "BENCHMARK_CALIBRATION_SECONDS=0"

'''
text = text[:start] + calibration + text[end:]

capture_function = r'''_capture_benchmark_json() {
  local endpoint="$1"
  local output_path="$2"
  local temporary_path="${output_path}.tmp"

  if curl -sf "${DRESSAGE_PROXY_URL}${endpoint}" >"${temporary_path}"; then
    mv -f -- "${temporary_path}" "${output_path}"
  else
    rm -f -- "${temporary_path}"
    printf '{"error":"endpoint unavailable","endpoint":"%s"}\n' \
      "${endpoint}" >"${output_path}"
  fi
}

_capture_benchmark_snapshots() {
  if [[ -z "${DRESSAGE_BENCHMARK_OUTPUT_DIR:-}" ]]; then
    return
  fi
  mkdir -p "${DRESSAGE_BENCHMARK_OUTPUT_DIR}"
  _capture_benchmark_json "/v1/engines/load" \
    "${DRESSAGE_BENCHMARK_OUTPUT_DIR}/engine_load.json"
  _capture_benchmark_json "/v1/engines/calibration" \
    "${DRESSAGE_BENCHMARK_OUTPUT_DIR}/calibration.json"
}

'''
sampler_function = ""
sampler_cleanup = ""
sampler_start = ""
if mode == "on":
    sampler_function = r'''_capture_benchmark_engine_load_history() {
  local phase="$1"
  local output_path="${DRESSAGE_BENCHMARK_OUTPUT_DIR}/engine_load_snapshots.jsonl"
  local temporary_path="${output_path}.tmp"

  if curl -sf --max-time 2 "${DRESSAGE_PROXY_URL}/v1/engines/load" \
      >"${temporary_path}"; then
    python3 - "${phase}" "${temporary_path}" >>"${output_path}" <<'PY_SNAPSHOT'
import json
import pathlib
import sys
import time

phase, path = sys.argv[1:]
payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
print(
    json.dumps(
        {"captured_at": time.time(), "phase": phase, "payload": payload},
        ensure_ascii=False,
        separators=(",", ":"),
    )
)
PY_SNAPSHOT
  fi
  rm -f -- "${temporary_path}"
}

_start_benchmark_engine_load_sampler() {
  if [[ -z "${DRESSAGE_BENCHMARK_OUTPUT_DIR:-}" ]]; then
    return
  fi
  mkdir -p "${DRESSAGE_BENCHMARK_OUTPUT_DIR}"
  : >"${DRESSAGE_BENCHMARK_OUTPUT_DIR}/engine_load_snapshots.jsonl"
  BENCHMARK_ENGINE_LOAD_SAMPLER_STOP_PATH="${DRESSAGE_BENCHMARK_OUTPUT_DIR}/.engine_load_sampler.stop"
  rm -f -- "${BENCHMARK_ENGINE_LOAD_SAMPLER_STOP_PATH}"
  _capture_benchmark_engine_load_history baseline
  (
    while [[ ! -e "${BENCHMARK_ENGINE_LOAD_SAMPLER_STOP_PATH}" ]]; do
      for _ in $(seq 1 50); do
        if [[ -e "${BENCHMARK_ENGINE_LOAD_SAMPLER_STOP_PATH}" ]]; then
          exit 0
        fi
        sleep 0.1
      done
      if [[ -e "${BENCHMARK_ENGINE_LOAD_SAMPLER_STOP_PATH}" ]]; then
        exit 0
      fi
      _capture_benchmark_engine_load_history sample
    done
  ) &
  BENCHMARK_ENGINE_LOAD_SAMPLER_PID=$!
}

_stop_benchmark_engine_load_sampler() {
  local pid="${BENCHMARK_ENGINE_LOAD_SAMPLER_PID:-}"
  if [[ -n "${pid}" ]]; then
    touch "${BENCHMARK_ENGINE_LOAD_SAMPLER_STOP_PATH}"
    wait "${pid}" 2>/dev/null || true
    BENCHMARK_ENGINE_LOAD_SAMPLER_PID=""
    rm -f -- "${BENCHMARK_ENGINE_LOAD_SAMPLER_STOP_PATH}"
  fi
  _capture_benchmark_engine_load_history final
}

'''
    sampler_cleanup = "  _stop_benchmark_engine_load_sampler\n"
    sampler_start = "_start_benchmark_engine_load_sampler\n\n"
cleanup_marker = "cleanup() {\n  status=$?\n  set +e\n"
if cleanup_marker not in text:
    raise SystemExit("source recipe cleanup marker changed")
text = text.replace(
    cleanup_marker,
    capture_function
    + sampler_function
    + "cleanup() {\n  status=$?\n  set +e\n\n"
    + sampler_cleanup
    + "  _capture_benchmark_snapshots\n",
    1,
)

if mode == "on":
    rollout_marker = "ray job submit \\\n"
    if rollout_marker not in text:
        raise SystemExit("source recipe rollout launch marker changed")
    text = text.replace(rollout_marker, sampler_start + rollout_marker, 1)

pathlib.Path(output).write_text(text, encoding="utf-8")
PY

  chmod +x "${output_path}"
}

collect_run() {
  local run_dir="$1"
  local run_name="$2"
  local mode="$3"
  local process_status="$4"
  local process_started_epoch="$5"
  local process_ended_epoch="$6"

  python3 - \
    "${run_dir}" \
    "${run_name}" \
    "${mode}" \
    "${process_status}" \
    "${process_started_epoch}" \
    "${process_ended_epoch}" \
    "${N_SAMPLES_PER_PROMPT}" <<'PY'
from __future__ import annotations

import ast
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import re
import sys
from collections import defaultdict

run_dir = pathlib.Path(sys.argv[1])
run_name = sys.argv[2]
mode = sys.argv[3]
process_status = int(sys.argv[4])
process_started_epoch = int(sys.argv[5])
process_ended_epoch = int(sys.argv[6])
expected_sampling_seeds = int(sys.argv[7])

run_log_path = run_dir / "run.log"
run_log = run_log_path.read_text(encoding="utf-8", errors="replace") if run_log_path.exists() else ""
ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
clean_log = ansi_re.sub("", run_log)


def load_json(path: pathlib.Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"value": value}
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": str(exc)}


def non_negative_number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def distribution(values) -> dict[str, float | int | None]:
    ordered = sorted(
        parsed
        for value in values
        if (parsed := non_negative_number(value)) is not None
    )

    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (
            position - lower
        )

    return {
        "sample_count": len(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1] if ordered else None,
    }


def parse_environment(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


perf: dict = {}
perf_line = ""
rollout_end: dt.datetime | None = None
for line in clean_log.splitlines():
    if "perf 0:" not in line:
        continue
    candidate = line.split("perf 0:", 1)[1].strip()
    try:
        parsed = ast.literal_eval(candidate)
    except (SyntaxError, ValueError):
        continue
    if isinstance(parsed, dict):
        perf = parsed
        perf_line = line
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
        if match:
            rollout_end = dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")

calibration_match = re.findall(r"BENCHMARK_CALIBRATION_SECONDS=(\d+)", clean_log)
calibration_seconds = int(calibration_match[-1]) if calibration_match else None

trajectory_root = run_dir / "runtime" / "traj_payload" / run_name
sample_paths = sorted(trajectory_root.glob("**/samples/*.json"))
records: list[dict] = []
effective_tokens = 0
effective_tokens_available = True
artifact_retry_count = 0
aborted_sample_count = 0
hash_lines: list[str] = []
for sample_path in sample_paths:
    try:
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    metadata = sample.get("metadata")
    sampling_seed = None
    if isinstance(metadata, dict):
        try:
            sampling_seed = int(metadata["rollout_sampling_seed"])
        except (KeyError, TypeError, ValueError):
            pass
    record = {
        "instance_id": str(sample.get("instance_id")),
        "sampling_seed": sampling_seed,
        "segment_index": sample.get("segment_index"),
        "tokens": sample.get("tokens"),
        "status": sample.get("status"),
        "reward": sample.get("reward"),
    }
    records.append(record)
    status_text = str(sample.get("status") or "").lower()
    if "abort" in status_text:
        aborted_sample_count += 1
    if isinstance(metadata, dict):
        try:
            artifact_retry_count += int(metadata.get("dressage_retry_count", 0) or 0)
        except (TypeError, ValueError):
            artifact_retry_count += 1
    loss_mask = sample.get("loss_mask")
    if not isinstance(loss_mask, list):
        effective_tokens_available = False
    else:
        try:
            effective_tokens += sum(int(value) for value in loss_mask)
        except (TypeError, ValueError):
            effective_tokens_available = False

records.sort(
    key=lambda item: (
        item["instance_id"],
        item["sampling_seed"] if item["sampling_seed"] is not None else -1,
        item["segment_index"] if isinstance(item["segment_index"], int) else -1,
    )
)
canonical_chunks: list[bytes] = []
for record in records:
    canonical = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    canonical_chunks.append(canonical)
    item_hash = hashlib.sha256(canonical).hexdigest()
    hash_lines.append(
        f"{item_hash} instance_id={record['instance_id']} "
        f"sampling_seed={record['sampling_seed']} "
        f"segment_index={record['segment_index']}"
    )
aggregate_hash = hashlib.sha256(b"\n".join(canonical_chunks)).hexdigest() if records else None
if aggregate_hash:
    hash_lines.append(f"aggregate_sha256 {aggregate_hash}")
(run_dir / "trajectory_hashes.txt").write_text(
    "\n".join(hash_lines) + ("\n" if hash_lines else ""),
    encoding="utf-8",
)

environment = parse_environment(run_dir / "environment.txt")
gpu_count = int(environment.get("gpu_count", "0") or 0)
rollout_seconds = perf.get("perf/rollout_time")
effective_tps = perf.get("perf/effective_tokens_per_gpu_per_sec")
derived_effective_tokens = None
if isinstance(rollout_seconds, (int, float)) and isinstance(effective_tps, (int, float)) and gpu_count:
    derived_effective_tokens = round(float(rollout_seconds) * float(effective_tps) * gpu_count)
effective_token_total = effective_tokens if records and effective_tokens_available else derived_effective_tokens

trajectory_count = len({record["instance_id"] for record in records})
if not trajectory_count and isinstance(perf.get("rollout/num_trajectories"), (int, float)):
    trajectory_count = int(perf["rollout/num_trajectories"])
trajectory_per_minute = None
if trajectory_count and isinstance(rollout_seconds, (int, float)) and rollout_seconds > 0:
    trajectory_per_minute = trajectory_count * 60.0 / float(rollout_seconds)

# Slice the raw GPU samples to exactly the rollout window. The logger and
# nvidia-smi monitor run on the same host; use the CSV timezone for the naive
# logger timestamp.
gpu_values: dict[str, list[float]] = defaultdict(list)
gpu_window_start = None
gpu_window_end = None
csv_paths = sorted((run_dir / "gpu").glob("gpu_utilization_*.csv"))
if rollout_end is not None and isinstance(rollout_seconds, (int, float)) and csv_paths:
    parsed_rows: list[tuple[dt.datetime, str, float]] = []
    for csv_path in csv_paths:
        with csv_path.open(encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    timestamp = dt.datetime.fromisoformat(row["timestamp"])
                    parsed_rows.append((timestamp, row["gpu_uuid"], float(row["utilization_percent"])))
                except (KeyError, TypeError, ValueError):
                    continue
    if parsed_rows:
        timezone = parsed_rows[0][0].tzinfo
        rollout_end = rollout_end.replace(tzinfo=timezone)
        rollout_start = rollout_end - dt.timedelta(seconds=float(rollout_seconds))
        gpu_window_start = rollout_start.isoformat()
        gpu_window_end = rollout_end.isoformat()
        for timestamp, gpu_uuid, utilization in parsed_rows:
            if rollout_start <= timestamp <= rollout_end:
                gpu_values[gpu_uuid].append(utilization)

per_gpu_average = {
    gpu_uuid: sum(values) / len(values)
    for gpu_uuid, values in sorted(gpu_values.items())
    if values
}
gpu_average = None
gpu_spread = None
if per_gpu_average:
    gpu_average = sum(per_gpu_average.values()) / len(per_gpu_average)
    gpu_spread = max(per_gpu_average.values()) - min(per_gpu_average.values())

engine_load = load_json(run_dir / "engine_load.json")
calibration = load_json(run_dir / "calibration.json")
calibration_state = calibration.get("state")
decisions = engine_load.get("recent_decisions", [])
observations = engine_load.get("recent_context_observations", [])
decisions = decisions if isinstance(decisions, list) else []
observations = observations if isinstance(observations, list) else []
moved_decisions = [item for item in decisions if isinstance(item, dict) and item.get("moved") is True]
moved_sessions = {
    (str(item.get("session_id")), str(item.get("target_worker_url")))
    for item in moved_decisions
}
mooncake_observations = [
    item
    for item in observations
    if isinstance(item, dict)
    and item.get("cache_source") == "mooncake"
    and isinstance(item.get("actual_cached_tokens"), (int, float))
    and item["actual_cached_tokens"] > 0
]
matched_mooncake_observations = [
    item
    for item in mooncake_observations
    if (str(item.get("session_id")), str(item.get("engine_url"))) in moved_sessions
]

request_steps: dict[tuple[str, str], dict] = {}
for session_path in sorted(trajectory_root.glob("**/session.json")):
    session_payload = load_json(session_path)
    trajectory_id = str(
        session_payload.get("trajectory_id")
        or session_payload.get("session_id")
        or session_path.parent
    )
    segments = session_payload.get("data")
    if not isinstance(segments, list):
        continue
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        extra = segment.get("extra_info")
        if not isinstance(extra, dict):
            continue
        if extra.get("segment_view") == "timeline":
            step_id = str(extra.get("step_id") or f"segment-{segment_index}")
            request_steps[(trajectory_id, step_id)] = extra
            continue
        if extra.get("segment_view") != "lineage":
            continue
        for metric_index, metric in enumerate(extra.get("request_metrics") or []):
            if not isinstance(metric, dict):
                continue
            step_id = str(
                metric.get("step_id")
                or f"segment-{segment_index}-request-{metric_index}"
            )
            request_steps[(trajectory_id, step_id)] = metric

request_e2e_values = [
    extra.get("request_e2e_latency_seconds") for extra in request_steps.values()
]
request_queue_values = [
    extra.get("request_queue_seconds") for extra in request_steps.values()
]
moved_e2e_values = [
    extra.get("request_e2e_latency_seconds")
    for extra in request_steps.values()
    if extra.get("rebalancing_moved") is True
]
sticky_e2e_values = [
    extra.get("request_e2e_latency_seconds")
    for extra in request_steps.values()
    if extra.get("rebalancing_moved") is not True
]

snapshot_records: list[dict] = []
snapshot_path = run_dir / "engine_load_snapshots.jsonl"
if snapshot_path.exists():
    for line in snapshot_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            snapshot_records.append(record)

baseline_batch_id = None
for record in snapshot_records:
    if record.get("phase") != "baseline":
        continue
    payload = record.get("payload")
    traces = payload.get("recent_load_batches") if isinstance(payload, dict) else None
    if not isinstance(traces, list):
        continue
    baseline_ids = [
        trace.get("batch", {}).get("id")
        for trace in traces
        if isinstance(trace, dict) and isinstance(trace.get("batch"), dict)
    ]
    baseline_ids = [value for value in baseline_ids if isinstance(value, int)]
    baseline_batch_id = max(baseline_ids, default=0)
    break

batch_traces: dict[int, dict] = {}
for record in snapshot_records:
    if record.get("phase") == "baseline":
        continue
    payload = record.get("payload")
    traces = payload.get("recent_load_batches") if isinstance(payload, dict) else None
    if not isinstance(traces, list):
        continue
    for trace in traces:
        batch = trace.get("batch") if isinstance(trace, dict) else None
        batch_id = batch.get("id") if isinstance(batch, dict) else None
        if not isinstance(batch_id, int):
            continue
        if baseline_batch_id is None or batch_id > baseline_batch_id:
            batch_traces[batch_id] = trace

ordered_batch_ids = sorted(batch_traces)
ordered_traces = [batch_traces[batch_id] for batch_id in ordered_batch_ids]
registered_counts = [trace.get("batch", {}).get("registered_count") for trace in ordered_traces]
batch_total_values = [trace.get("batch", {}).get("total_seconds") for trace in ordered_traces]
batch_collect_values = [trace.get("batch", {}).get("collect_seconds") for trace in ordered_traces]
batch_wait_values = [
    trace.get("batch", {}).get("wait_for_previous_seconds") for trace in ordered_traces
]
batch_fetch_values = [trace.get("batch", {}).get("fetch_seconds") for trace in ordered_traces]
solve_values = [
    trace.get("batch", {}).get("solve_seconds")
    for trace in ordered_traces
    if isinstance(trace.get("sticky"), dict)
    or trace.get("fallback_reason") == "sticky_solver_failure"
]
sticky_values = [
    trace["sticky"].get("elapsed_seconds")
    for trace in ordered_traces
    if isinstance(trace.get("sticky"), dict)
]
optimized_values = [
    trace["optimized"].get("elapsed_seconds")
    for trace in ordered_traces
    if isinstance(trace.get("optimized"), dict)
]

fetch_status_counts = {status: 0 for status in ("ok", "timeout", "error", "invalid")}
engine_fetch_values: list[float] = []
for trace in ordered_traces:
    for engine in trace.get("engines") or []:
        if not isinstance(engine, dict):
            continue
        status = engine.get("fetch_status")
        if status in fetch_status_counts:
            fetch_status_counts[status] += 1
        duration = non_negative_number(engine.get("fetch_duration_seconds"))
        if duration is not None:
            engine_fetch_values.append(duration)
fetch_attempts = sum(fetch_status_counts.values())
fetch_status_rates = {
    status: (count / fetch_attempts if fetch_attempts else None)
    for status, count in fetch_status_counts.items()
}

fallback_counts = {
    reason: 0
    for reason in (
        "target_load_infeasible",
        "target_solver_deadline",
        "target_solver_failure",
        "frozen_state_changed",
    )
}
for trace in ordered_traces:
    reason = trace.get("fallback_reason")
    if reason in fallback_counts:
        fallback_counts[reason] += 1
fallback_rates = {
    reason: (count / len(ordered_traces) if ordered_traces else None)
    for reason, count in fallback_counts.items()
}

incomplete_reasons: list[str] = []
request_e2e_count = distribution(request_e2e_values)["sample_count"]
request_queue_count = distribution(request_queue_values)["sample_count"]
if not request_steps:
    incomplete_reasons.append("no trajectory request metrics")
if request_e2e_count != len(request_steps):
    incomplete_reasons.append(
        f"request E2E coverage is {request_e2e_count}/{len(request_steps)}"
    )
if request_queue_count != len(request_steps):
    incomplete_reasons.append(
        f"request queue coverage is {request_queue_count}/{len(request_steps)}"
    )
missing_batch_ids: list[int] = []
if mode == "on":
    if not any(record.get("phase") == "final" for record in snapshot_records):
        incomplete_reasons.append("missing final load snapshot")
    if baseline_batch_id is None:
        incomplete_reasons.append("missing baseline load snapshot")
    elif ordered_batch_ids:
        missing_batch_ids = sorted(
            set(range(baseline_batch_id + 1, ordered_batch_ids[-1] + 1))
            - set(ordered_batch_ids)
        )
        if missing_batch_ids:
            incomplete_reasons.append(
                "missing load batch IDs: "
                + ",".join(str(batch_id) for batch_id in missing_batch_ids)
            )
    else:
        incomplete_reasons.append("no load batches captured")

valid_registered_counts = [
    value
    for value in registered_counts
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0
]
tail_metrics = {
    "request": {
        "e2e_latency_seconds": distribution(request_e2e_values),
        "queue_seconds": distribution(request_queue_values),
        "moved_e2e_latency_seconds": distribution(moved_e2e_values),
        "sticky_e2e_latency_seconds": distribution(sticky_e2e_values),
    },
    "batch": {
        "total_seconds": distribution(batch_total_values),
        "collect_seconds": distribution(batch_collect_values),
        "wait_for_previous_seconds": distribution(batch_wait_values),
        "registered_count": distribution(registered_counts),
        "singleton_ratio": (
            sum(value == 1 for value in valid_registered_counts)
            / len(valid_registered_counts)
            if valid_registered_counts
            else None
        ),
    },
    "load_fetch": {
        "batch_fetch_seconds": distribution(batch_fetch_values),
        "engine_fetch_duration_seconds": distribution(engine_fetch_values),
        "status_counts": fetch_status_counts,
        "status_rates": fetch_status_rates,
    },
    "milp": {
        "solve_seconds": distribution(solve_values),
        "sticky_elapsed_seconds": distribution(sticky_values),
        "optimized_elapsed_seconds": distribution(optimized_values),
        "fallback_counts": fallback_counts,
        "fallback_rates": fallback_rates,
    },
    "coverage": {
        "trajectory_step_count": len(request_steps),
        "request_e2e_count": request_e2e_count,
        "request_queue_count": request_queue_count,
        "baseline_batch_id": baseline_batch_id,
        "batch_count": len(ordered_batch_ids),
        "first_batch_id": ordered_batch_ids[0] if ordered_batch_ids else None,
        "last_batch_id": ordered_batch_ids[-1] if ordered_batch_ids else None,
        "batch_ids": ordered_batch_ids,
        "missing_batch_ids": missing_batch_ids,
        "complete": not incomplete_reasons,
        "incomplete_reasons": incomplete_reasons,
    },
}

patterns = {
    "rollout_retry": re.compile(
        r"resubmitting rollout group for retry|returned group[^\n]*to rollout buffer for retry|"
        r"exhausted[^\n]*rollout[^\n]*retries",
        re.IGNORECASE,
    ),
    "rollout_abort": re.compile(
        r"rollout group[^\n]*abort(?:ed|ing)?|aborted rollout group",
        re.IGNORECASE,
    ),
    "engine_failure": re.compile(
        r"(?:sglang|engine)[^\n]*(?:process exited|crashed|unhealthy|died)|"
        r"engine[^\n]*failed health check",
        re.IGNORECASE,
    ),
    "batch_put_failed": re.compile(r"BatchPut failed", re.IGNORECASE),
    "insufficient_space": re.compile(r"insufficient space", re.IGNORECASE),
}
error_matches: dict[str, list[str]] = {key: [] for key in patterns}
log_paths = {run_log_path}
log_paths.update((run_dir / "runtime").glob("**/*.log"))
for path in sorted(log_paths):
    if not path.is_file():
        continue
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        continue
    with handle:
        for line in handle:
            clean_line = ansi_re.sub("", line.rstrip("\n"))
            for key, pattern in patterns.items():
                if len(error_matches[key]) < 20 and pattern.search(clean_line):
                    error_matches[key].append(
                        f"{path.relative_to(run_dir)}: {clean_line[:500]}"
                    )

acceptance_errors: list[str] = []
if process_status != 0:
    acceptance_errors.append(f"benchmark process exited with status {process_status}")
if not perf:
    acceptance_errors.append("missing perf 0 metrics")
if not records:
    acceptance_errors.append("missing trajectory sample artifacts")
if gpu_count != 8:
    acceptance_errors.append(f"expected 8 GPUs, found {gpu_count}")
if mode == "on" and calibration_state != "READY":
    acceptance_errors.append(f"calibration state is {calibration_state!r}, expected 'READY'")
if artifact_retry_count:
    acceptance_errors.append(f"trajectory artifacts report {artifact_retry_count} rollout retries")
if aborted_sample_count:
    acceptance_errors.append(f"found {aborted_sample_count} aborted trajectory samples")
sampling_seeds_by_instance: dict[str, set[int]] = defaultdict(set)
missing_sampling_seed_instances: set[str] = set()
for record in records:
    if record["sampling_seed"] is None:
        missing_sampling_seed_instances.add(record["instance_id"])
    else:
        sampling_seeds_by_instance[record["instance_id"]].add(record["sampling_seed"])
for instance_id in sorted({record["instance_id"] for record in records}):
    if instance_id in missing_sampling_seed_instances:
        acceptance_errors.append(
            f"instance {instance_id} is missing a rollout sampling seed"
        )
    elif len(sampling_seeds_by_instance[instance_id]) != expected_sampling_seeds:
        acceptance_errors.append(
            f"instance {instance_id} has {len(sampling_seeds_by_instance[instance_id])} "
            f"distinct rollout sampling seeds, expected {expected_sampling_seeds}"
        )
for key, matches in error_matches.items():
    if matches:
        acceptance_errors.append(f"detected {key}: {len(matches)} matching log lines")

startup_seconds = None
if rollout_end is not None and isinstance(rollout_seconds, (int, float)):
    rollout_start_epoch = rollout_end.timestamp() - float(rollout_seconds)
    startup_seconds = max(0.0, rollout_start_epoch - process_started_epoch)

metrics = {
    "run_name": run_name,
    "mode": mode,
    "process_exit_status": process_status,
    "process_wall_seconds": process_ended_epoch - process_started_epoch,
    "startup_seconds": startup_seconds,
    "calibration_seconds": calibration_seconds,
    "rollout_time_seconds": rollout_seconds,
    "effective_tokens_per_gpu_per_sec": effective_tps,
    "effective_token_total": effective_token_total,
    "derived_effective_token_total": derived_effective_tokens,
    "trajectory_count": trajectory_count,
    "sample_count": len(records),
    "artifact_retry_count": artifact_retry_count,
    "aborted_sample_count": aborted_sample_count,
    "trajectory_per_minute": trajectory_per_minute,
    "trajectory_hash": aggregate_hash,
    "gpu_rollout_window_start": gpu_window_start,
    "gpu_rollout_window_end": gpu_window_end,
    "gpu_rollout_average_utilization_percent": gpu_average,
    "gpu_rollout_per_gpu_average_percent": per_gpu_average,
    "gpu_rollout_spread_percent": gpu_spread,
    "calibration_state": calibration_state,
    "recent_moved_decisions": len(moved_decisions),
    "recent_mooncake_cached_observations": len(mooncake_observations),
    "matched_mooncake_migrations": len(matched_mooncake_observations),
    "kv_migration_evidence": bool(matched_mooncake_observations),
    "tail_metrics": tail_metrics,
    "hostname": environment.get("hostname"),
    "gpu_inventory_sha256": environment.get("gpu_inventory_sha256"),
    "code_fingerprint": environment.get("code_fingerprint"),
    "perf": perf,
    "perf_log_line": perf_line,
    "error_matches": error_matches,
    "acceptance_errors": acceptance_errors,
    "valid_run": not acceptance_errors,
}
(run_dir / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

for required_name in ("engine_load.json", "calibration.json"):
    required_path = run_dir / required_name
    if not required_path.exists():
        required_path.write_text(
            json.dumps({"error": "not captured"}, indent=2) + "\n",
            encoding="utf-8",
        )
PY
}

run_one() {
  local run_name="$1"
  local mode="$2"
  local run_dir="${BENCHMARK_ROOT}/${run_name}"
  local temporary_recipe="${TEMP_DIR}/${run_name}.sh"
  local process_started_epoch process_ended_epoch process_status

  echo
  echo "===== ${run_name} (${mode}) ====="
  mkdir -p "${run_dir}/gpu" "${run_dir}/runtime"
  record_environment "${run_dir}/environment.txt" "${run_name}" "${mode}"
  build_temporary_recipe "${temporary_recipe}" "${mode}"

  process_started_epoch="$(date +%s)"
  set +e
  (
    unset DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR
    unset DRESSAGE_TRAJECTORY_ERROR_LOG_DIR
    unset GPU_UTIL_STATE_DIR
    unset GPU_UTIL_LOG_DIR
    unset LOG_DIR
    unset RUN_NAME

    export REPO_ROOT
    export DRESSAGE_BENCHMARK_SOURCE_SCRIPT_DIR="${SCRIPT_DIR}"
    export DRESSAGE_BENCHMARK_OUTPUT_DIR="${run_dir}"
    export RUN_NAME="${run_name}"
    export PROMPT_DATA="${PROMPT_EFFECTIVE}"
    export LOG_DIR="${run_dir}/runtime"
    export GPU_UTIL_STATE_DIR="${TEMP_DIR}/gpu-state-${run_name}"
    export GPU_UTIL_LOG_DIR="${run_dir}/gpu"
    export DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR="${run_dir}/runtime/traj_payload/${run_name}"
    export DRESSAGE_TRAJECTORY_ERROR_LOG_DIR="${run_dir}/runtime/traj_err/${run_name}"
    export ENABLE_ENGINE_REBALANCING="$([[ "${mode}" == "on" ]] && echo 1 || echo 0)"
    export BENCHMARK_SEED
    export ROLLOUT_TEMPERATURE
    export ROLLOUT_BATCH_SIZE
    export N_SAMPLES_PER_PROMPT
    export GLOBAL_BATCH_SIZE
    export ROLLOUT_MAX_RESPONSE_LEN
    export DRESSAGE_BLACKBOX_SLOTS_PER_NODE
    export DRESSAGE_BLACKBOX_ACQUIRE_TIMEOUT_SEC
    export DRESSAGE_BLACKBOX_MAX_STEPS
    export MOONCAKE_GLOBAL_SEGMENT_SIZE
    export ENGINE_REBALANCING_LOAD_BATCH_COALESCING_WINDOW_MS
    export ENGINE_REBALANCING_MIN_LOAD_IMPROVEMENT_RATIO

    bash "${temporary_recipe}"
  ) 2>&1 | tee "${run_dir}/run.log"
  process_status="${PIPESTATUS[0]}"
  set -e
  process_ended_epoch="$(date +%s)"

  collect_run \
    "${run_dir}" \
    "${run_name}" \
    "${mode}" \
    "${process_status}" \
    "${process_started_epoch}" \
    "${process_ended_epoch}"

  if [[ "${process_status}" -ne 0 ]]; then
    echo "Run ${run_name} failed with status ${process_status}; continuing the matrix." >&2
  fi
}

write_summary() {
  python3 - "${BENCHMARK_ROOT}" "${BENCHMARK_SEED}" <<'PY'
from __future__ import annotations

import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
seed = sys.argv[2]


def load(run_name: str) -> dict:
    path = root / run_name / "metrics.json"
    if not path.exists():
        return {
            "run_name": run_name,
            "valid_run": False,
            "acceptance_errors": ["metrics.json is missing"],
        }
    return json.loads(path.read_text(encoding="utf-8"))


pairs = [
    (1, f"seed{seed}-off-r1", f"seed{seed}-on-r1"),
]


def tail_value(run: dict, *path):
    value = run.get("tail_metrics")
    if path != ("coverage", "complete"):
        coverage = value.get("coverage") if isinstance(value, dict) else None
        if not isinstance(coverage, dict) or coverage.get("complete") is not True:
            return None
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


rows: list[dict] = []
tail_metrics_by_run: dict[str, dict | None] = {}
for pair_number, off_name, on_name in pairs:
    off = load(off_name)
    on = load(on_name)
    tail_metrics_by_run[off_name] = off.get("tail_metrics")
    tail_metrics_by_run[on_name] = on.get("tail_metrics")
    reasons: list[str] = []
    if not off.get("valid_run"):
        reasons.append("OFF invalid: " + "; ".join(off.get("acceptance_errors", [])))
    if not on.get("valid_run"):
        reasons.append("ON invalid: " + "; ".join(on.get("acceptance_errors", [])))

    environment_equal = all(
        off.get(key) == on.get(key) and off.get(key) is not None
        for key in ("hostname", "gpu_inventory_sha256", "code_fingerprint")
    )
    token_total_equal = (
        off.get("effective_token_total") is not None
        and off.get("effective_token_total") == on.get("effective_token_total")
    )
    trajectory_hash_equal = (
        off.get("trajectory_hash") is not None
        and off.get("trajectory_hash") == on.get("trajectory_hash")
    )
    calibration_ready = on.get("calibration_state") == "READY"
    if not environment_equal:
        reasons.append("hostname/GPU/code fingerprint differs")
    if not token_total_equal:
        reasons.append("effective token totals differ")
    if not trajectory_hash_equal:
        reasons.append("trajectory token/status/reward hashes differ")
    if not calibration_ready:
        reasons.append("ON calibration is not READY")

    speedup = None
    throughput_gain = None
    off_time = off.get("rollout_time_seconds")
    on_time = on.get("rollout_time_seconds")
    off_tps = off.get("effective_tokens_per_gpu_per_sec")
    on_tps = on.get("effective_tokens_per_gpu_per_sec")
    if isinstance(off_time, (int, float)) and isinstance(on_time, (int, float)) and on_time > 0:
        speedup = off_time / on_time
    if isinstance(off_tps, (int, float)) and isinstance(on_tps, (int, float)) and off_tps > 0:
        throughput_gain = on_tps / off_tps - 1.0

    rows.append(
        {
            "pair": pair_number,
            "valid": not reasons,
            "off_run": off_name,
            "on_run": on_name,
            "off_rollout_time_seconds": off_time,
            "on_rollout_time_seconds": on_time,
            "rollout_speedup": speedup,
            "off_effective_tps": off_tps,
            "on_effective_tps": on_tps,
            "throughput_gain": throughput_gain,
            "off_trajectory_per_minute": off.get("trajectory_per_minute"),
            "on_trajectory_per_minute": on.get("trajectory_per_minute"),
            "off_gpu_utilization": off.get("gpu_rollout_average_utilization_percent"),
            "on_gpu_utilization": on.get("gpu_rollout_average_utilization_percent"),
            "off_gpu_spread": off.get("gpu_rollout_spread_percent"),
            "on_gpu_spread": on.get("gpu_rollout_spread_percent"),
            "off_request_e2e_p95": tail_value(
                off, "request", "e2e_latency_seconds", "p95"
            ),
            "off_request_e2e_count": tail_value(
                off, "request", "e2e_latency_seconds", "sample_count"
            ),
            "off_request_e2e_p99": tail_value(
                off, "request", "e2e_latency_seconds", "p99"
            ),
            "on_request_e2e_p95": tail_value(
                on, "request", "e2e_latency_seconds", "p95"
            ),
            "on_request_e2e_count": tail_value(
                on, "request", "e2e_latency_seconds", "sample_count"
            ),
            "on_request_e2e_p99": tail_value(
                on, "request", "e2e_latency_seconds", "p99"
            ),
            "off_request_queue_p95": tail_value(
                off, "request", "queue_seconds", "p95"
            ),
            "off_request_queue_count": tail_value(
                off, "request", "queue_seconds", "sample_count"
            ),
            "off_request_queue_p99": tail_value(
                off, "request", "queue_seconds", "p99"
            ),
            "on_request_queue_p95": tail_value(
                on, "request", "queue_seconds", "p95"
            ),
            "on_request_queue_count": tail_value(
                on, "request", "queue_seconds", "sample_count"
            ),
            "on_request_queue_p99": tail_value(
                on, "request", "queue_seconds", "p99"
            ),
            "on_moved_e2e_p95": tail_value(
                on, "request", "moved_e2e_latency_seconds", "p95"
            ),
            "on_moved_e2e_count": tail_value(
                on, "request", "moved_e2e_latency_seconds", "sample_count"
            ),
            "on_moved_e2e_p99": tail_value(
                on, "request", "moved_e2e_latency_seconds", "p99"
            ),
            "on_sticky_e2e_p95": tail_value(
                on, "request", "sticky_e2e_latency_seconds", "p95"
            ),
            "on_sticky_e2e_count": tail_value(
                on, "request", "sticky_e2e_latency_seconds", "sample_count"
            ),
            "on_sticky_e2e_p99": tail_value(
                on, "request", "sticky_e2e_latency_seconds", "p99"
            ),
            "on_batch_total_p95": tail_value(
                on, "batch", "total_seconds", "p95"
            ),
            "on_batch_count": tail_value(
                on, "batch", "total_seconds", "sample_count"
            ),
            "on_batch_total_p99": tail_value(
                on, "batch", "total_seconds", "p99"
            ),
            "on_batch_fetch_p95": tail_value(
                on, "load_fetch", "batch_fetch_seconds", "p95"
            ),
            "on_batch_fetch_count": tail_value(
                on, "load_fetch", "batch_fetch_seconds", "sample_count"
            ),
            "on_batch_fetch_p99": tail_value(
                on, "load_fetch", "batch_fetch_seconds", "p99"
            ),
            "on_solve_p95": tail_value(
                on, "milp", "solve_seconds", "p95"
            ),
            "on_solve_count": tail_value(
                on, "milp", "solve_seconds", "sample_count"
            ),
            "on_solve_p99": tail_value(
                on, "milp", "solve_seconds", "p99"
            ),
            "on_batch_size_p50": tail_value(
                on, "batch", "registered_count", "p50"
            ),
            "on_batch_size_p95": tail_value(
                on, "batch", "registered_count", "p95"
            ),
            "on_batch_singleton_ratio": tail_value(
                on, "batch", "singleton_ratio"
            ),
            "off_tail_metrics_complete": tail_value(off, "coverage", "complete"),
            "on_tail_metrics_complete": tail_value(on, "coverage", "complete"),
            "environment_equal": environment_equal,
            "effective_tokens_equal": token_total_equal,
            "trajectory_hash_equal": trajectory_hash_equal,
            "on_calibration_ready": calibration_ready,
            "kv_migration_evidence": bool(on.get("kv_migration_evidence")),
            "invalid_reasons": " | ".join(reasons),
        }
    )

valid_rows = [row for row in rows if row["valid"]]
speedups = [row["rollout_speedup"] for row in valid_rows if row["rollout_speedup"] is not None]
gains = [row["throughput_gain"] for row in valid_rows if row["throughput_gain"] is not None]
rollout_speedup = speedups[0] if speedups else None
throughput_gain = gains[0] if gains else None
kv_evidence = any(row["kv_migration_evidence"] for row in rows)
all_pairs_valid = len(valid_rows) == len(rows)

fieldnames = list(rows[0].keys())
with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def fmt(value, *, percent: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def fmt_count(value) -> str:
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else "N/A"


status = "PASS"
exit_code = 0
if not all_pairs_valid:
    status = "INVALID"
    exit_code = 1
elif not kv_evidence:
    status = "INCONCLUSIVE_NO_KV_MIGRATION"
    exit_code = 2

lines = [
    "# Engine Rebalancing A/B Benchmark",
    "",
    f"- Status: `{status}`",
    f"- Seed: `{seed}`",
    f"- Valid measured pairs: `{len(valid_rows)}/1`",
    f"- Rollout speedup (OFF / ON): `{fmt(rollout_speedup)}`",
    f"- Effective throughput gain: `{fmt(throughput_gain, percent=True)}`",
    f"- Observed moved=true + Mooncake cached tokens: `{'yes' if kv_evidence else 'no'}`",
    "",
    "| Pair | Valid | OFF rollout (s) | ON rollout (s) | Speedup | Throughput gain | KV evidence |",
    "|---:|:---:|---:|---:|---:|---:|:---:|",
]
for row in rows:
    lines.append(
        f"| {row['pair']} | {'yes' if row['valid'] else 'no'} | "
        f"{fmt(row['off_rollout_time_seconds'])} | "
        f"{fmt(row['on_rollout_time_seconds'])} | "
        f"{fmt(row['rollout_speedup'])} | "
        f"{fmt(row['throughput_gain'], percent=True)} | "
        f"{'yes' if row['kv_migration_evidence'] else 'no'} |"
    )

lines.extend(
    [
        "",
        "## Tail latency",
        "",
        "| Pair | Metric | OFF N | OFF P95 | OFF P99 | ON N | ON P95 | ON P99 |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
)
for row in rows:
    lines.extend(
        [
            f"| {row['pair']} | Request E2E | "
            f"{fmt_count(row['off_request_e2e_count'])} | "
            f"{fmt(row['off_request_e2e_p95'])} | "
            f"{fmt(row['off_request_e2e_p99'])} | "
            f"{fmt_count(row['on_request_e2e_count'])} | "
            f"{fmt(row['on_request_e2e_p95'])} | "
            f"{fmt(row['on_request_e2e_p99'])} |",
            f"| {row['pair']} | Request queue | "
            f"{fmt_count(row['off_request_queue_count'])} | "
            f"{fmt(row['off_request_queue_p95'])} | "
            f"{fmt(row['off_request_queue_p99'])} | "
            f"{fmt_count(row['on_request_queue_count'])} | "
            f"{fmt(row['on_request_queue_p95'])} | "
            f"{fmt(row['on_request_queue_p99'])} |",
            f"| {row['pair']} | ON moved E2E | N/A | N/A | N/A | "
            f"{fmt_count(row['on_moved_e2e_count'])} | "
            f"{fmt(row['on_moved_e2e_p95'])} | "
            f"{fmt(row['on_moved_e2e_p99'])} |",
            f"| {row['pair']} | ON sticky E2E | N/A | N/A | N/A | "
            f"{fmt_count(row['on_sticky_e2e_count'])} | "
            f"{fmt(row['on_sticky_e2e_p95'])} | "
            f"{fmt(row['on_sticky_e2e_p99'])} |",
            f"| {row['pair']} | Batch total | N/A | N/A | N/A | "
            f"{fmt_count(row['on_batch_count'])} | "
            f"{fmt(row['on_batch_total_p95'])} | "
            f"{fmt(row['on_batch_total_p99'])} |",
            f"| {row['pair']} | /v1/loads fetch | N/A | N/A | N/A | "
            f"{fmt_count(row['on_batch_fetch_count'])} | "
            f"{fmt(row['on_batch_fetch_p95'])} | "
            f"{fmt(row['on_batch_fetch_p99'])} |",
            f"| {row['pair']} | MILP solve | N/A | N/A | N/A | "
            f"{fmt_count(row['on_solve_count'])} | "
            f"{fmt(row['on_solve_p95'])} | "
            f"{fmt(row['on_solve_p99'])} |",
            "",
            f"- Pair {row['pair']} ON batch size P50/P95: "
            f"`{fmt(row['on_batch_size_p50'])}` / "
            f"`{fmt(row['on_batch_size_p95'])}`; singleton ratio: "
            f"`{fmt(row['on_batch_singleton_ratio'], percent=True)}`.",
            f"- Pair {row['pair']} tail metrics complete: "
            f"OFF=`{'yes' if row['off_tail_metrics_complete'] else 'no'}`, "
            f"ON=`{'yes' if row['on_tail_metrics_complete'] else 'no'}`.",
        ]
    )

invalid = [row for row in rows if not row["valid"]]
if invalid or not kv_evidence:
    lines.extend(["", "## Acceptance notes", ""])
    for row in invalid:
        lines.append(f"- Pair {row['pair']}: {row['invalid_reasons']}")
    if not kv_evidence:
        lines.append(
            "- No correlated moved decision with `cache_source=mooncake` and "
            "`actual_cached_tokens>0` was observed; do not claim a KV-migration result."
        )

(root / "summary.json").write_text(
    json.dumps(
        {
            "status": status,
            "seed": seed,
            "valid_pair_count": len(valid_rows),
            "rollout_speedup": rollout_speedup,
            "throughput_gain": throughput_gain,
            "pairs": rows,
            "tail_metrics_by_run": tail_metrics_by_run,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"summary: {root / 'summary.md'}")
print(f"status: {status}")
raise SystemExit(exit_code)
PY
}

for index in "${!RUN_NAMES[@]}"; do
  run_one "${RUN_NAMES[${index}]}" "${RUN_MODES[${index}]}"
done

set +e
write_summary
summary_status=$?
set -e

echo "Benchmark artifacts: ${BENCHMARK_ROOT}"
exit "${summary_status}"
