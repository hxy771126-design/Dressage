#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 -f '[m]ooncake_master' 2>/dev/null || true

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=1

# unset proxy to avoid distributed startup issues
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
BASE_FOLDER="${BASE_FOLDER:-/root}"

# GPU utilization monitor.
GPU_UTIL_MONITOR="/root/Dressage/examples/scripts/gpu_utilization_monitor.sh"
GPU_UTIL_MONITOR_STARTED=0

export GPU_UTIL_STATE_DIR="${GPU_UTIL_STATE_DIR:-${TMPDIR:-/tmp}/dressage-gpu-util-$$}"
export GPU_UTIL_LOG_DIR="${GPU_UTIL_LOG_DIR:-${REPO_ROOT}/log/gpu_utilization}"
export GPU_UTIL_SAMPLE_INTERVAL="${GPU_UTIL_SAMPLE_INTERVAL:-1}"
export GPU_UTIL_GPU_IDS="${GPU_UTIL_GPU_IDS:-}"

_stop_gpu_util_monitor_on_exit() {
  if [[ "${GPU_UTIL_MONITOR_STARTED:-0}" == "1" ]]; then
    bash "${GPU_UTIL_MONITOR}" stop || true
    GPU_UTIL_MONITOR_STARTED=0
  fi
}

# 在完整的 cleanup() 注册前发生错误时，也能停止 GPU 监控。
trap _stop_gpu_util_monitor_on_exit EXIT

if [[ ! -f "${GPU_UTIL_MONITOR}" ]]; then
  echo "Cannot find GPU utilization monitor: ${GPU_UTIL_MONITOR}" >&2
  exit 1
fi

bash "${GPU_UTIL_MONITOR}" start
GPU_UTIL_MONITOR_STARTED=1

if [[ ! -f "${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh" ]]; then
  echo "Cannot find slime model config: ${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh" >&2
  echo "Set REPO_ROOT or SLIME_ROOT to match the current checkout layout." >&2
  exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"

if [[ -z "${BASE_FOLDER:-}" ]]; then
  echo "BASE_FOLDER is not set. Please set it to the base directory of your checkpoints." >&2
  exit 1
fi

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "MASTER_ADDR is not set. Please set it to the master node address." >&2
  exit 1
fi

# Single-node 8-GPU synchronous colocate setup.
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
RAY_NUM_GPUS_PER_NODE=${RAY_NUM_GPUS_PER_NODE:-8}
CP_SIZE=${CP_SIZE:-4}
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
HOSTFILE=${HOSTFILE:-}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

source "${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh"
source "${SCRIPT_DIR}/default/dressage_env_defaults.sh"

dressage_apply_common_defaults \
  "qwen3.5-4B-sync-local-l3-hicache" \
  "blackbox" \
  "local_bwrap"
dressage_apply_local_bwrap_defaults 16

# Self-contained single-node Mooncake L3 configuration. Each SGLang Engine
# contributes one segment, so the 4GB default yields 32GB across 8 Engines.
MOONCAKE_MASTER_HOST="127.0.0.1"
MOONCAKE_MASTER_PORT=50051
MOONCAKE_METADATA_PORT=8080
MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-4gb}"
MOONCAKE_MASTER_ADDRESS="${MOONCAKE_MASTER_HOST}:${MOONCAKE_MASTER_PORT}"
MOONCAKE_METADATA_SERVER="http://${MOONCAKE_MASTER_HOST}:${MOONCAKE_METADATA_PORT}/metadata"
MOONCAKE_LOG_DIR="${LOG_DIR}/mooncake/${RUN_NAME}"
MOONCAKE_MASTER_LOG_FILE="${MOONCAKE_LOG_DIR}/master.log"
MOONCAKE_MASTER_PID_FILE="${MOONCAKE_LOG_DIR}/master.pid"
export DRESSAGE_ENGINE_REBALANCING_DEPLOYMENT_CONFIG="${LOG_DIR}/proxy/${RUN_NAME}.rebalancing-deployment.json"
MOONCAKE_BACKEND_EXTRA_CONFIG="$(
  python3 - \
    "${MOONCAKE_MASTER_ADDRESS}" \
    "${MOONCAKE_METADATA_SERVER}" \
    "${MOONCAKE_GLOBAL_SEGMENT_SIZE}" <<'PY'
import json
import sys

master_address, metadata_server, global_segment_size = sys.argv[1:]
print(
    json.dumps(
        {
            "master_server_address": master_address,
            "local_hostname": "127.0.0.1",
            "metadata_server": metadata_server,
            "global_segment_size": global_segment_size,
            "protocol": "tcp",
            "device_name": "",
        },
        separators=(",", ":"),
    )
)
PY
)"

if [[ "${DRESSAGE_BLACKBOX_RUNNER_MODE}" == "bwrap" ||
      "${DRESSAGE_BLACKBOX_RUNNER_MODE}" == "bubblewrap" ]]; then
  command -v "${DRESSAGE_BLACKBOX_BWRAP_BIN}" >/dev/null || {
    echo "missing bubblewrap binary: ${DRESSAGE_BLACKBOX_BWRAP_BIN}" >&2
    exit 1
  }
fi

WORKER_COUNT=0
if [[ -n "${HOSTFILE}" && -f "${HOSTFILE}" ]]; then
  WORKER_COUNT=$(
    awk -v master="${MASTER_ADDR}" \
      '$1 != master { count += 1 } END { print count + 0 }' \
      "${HOSTFILE}"
  )
fi

dressage_compute_local_bwrap_resources "${WORKER_COUNT}"
dressage_validate_proxy_defaults
dressage_clear_trajectory_logs

if [[ "${TOKEN_BUILD_MODE}" != "snapshot" &&
      "${TOKEN_BUILD_MODE}" != "tito" ]]; then
  echo "TOKEN_BUILD_MODE must be snapshot or tito, got: ${TOKEN_BUILD_MODE}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${SLIME_ROOT}:${PYTHONPATH:-}"

dressage_export_common_env
dressage_export_local_bwrap_env
dressage_compute_context_window 16384 "${CP_SIZE}"

mkdir -p "$(dirname "${DRESSAGE_ENGINE_REBALANCING_DEPLOYMENT_CONFIG}")"
python3 - \
  "${DRESSAGE_ENGINE_REBALANCING_DEPLOYMENT_CONFIG}" \
  "${MASTER_ADDR}" \
  "${HOSTFILE}" \
  "${RAY_NUM_GPUS_PER_NODE}" \
  "${SOCKET_IFNAME}" \
  "${BASE_FOLDER}/Qwen3.5-4B" \
  "${MOONCAKE_METADATA_SERVER}" <<'PY'
import json
import pathlib
import sys

output, master_addr, hostfile, gpu_count, nic, model_config_path, metadata_server = sys.argv[1:]
addresses = [master_addr]
if hostfile and pathlib.Path(hostfile).is_file():
    addresses.extend(
        line.split()[0]
        for line in pathlib.Path(hostfile).read_text(encoding="utf-8").splitlines()
        if line.split()
    )
nodes = [
    {
        "node_id": address,
        "gpu_count": int(gpu_count),
        "gpu_ids": list(range(int(gpu_count))),
        "nic": nic,
    }
    for address in dict.fromkeys(addresses)
]
payload = {
    "schema_version": 1,
    "ray_address": "auto",
    "model_config_path": model_config_path,
    "nodes": nodes,
    "hicache": {
        "enabled": True,
        "storage_backend": "mooncake",
        "write_policy": "write_through",
        "gpudirect": False,
    },
    "mooncake": {
        "protocol": "tcp",
        "device_name": "",
        "metadata_server": metadata_server,
        "gpudirect": False,
    },
    "model_deployment": {},
}
path = pathlib.Path(output)
path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

COMM_ARGS=(
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
)

PROXY_ARGS=(
  --tokenizer-path "${BASE_FOLDER}/Qwen3.5-4B"
  --host "${PROXY_HOST}"
  --port "${PROXY_PORT}"
  --token-build-mode "${TOKEN_BUILD_MODE}"
  --token-build-model "${TOKEN_BUILD_MODEL}"
  "${COMM_ARGS[@]}"
  --context-window "${CONTEXT_WINDOW}"
  --enable-engine-rebalancing
)

CKPT_ARGS=(
  --hf-checkpoint "${BASE_FOLDER}/Qwen3.5-4B"
  --ref-load "${BASE_FOLDER}/Qwen3.5-4B_torch_dist/"
  --load "${BASE_FOLDER}/Qwen3.5-4B_slime/"
  --save "${BASE_FOLDER}/Qwen3.5-4B_slime/"
  --save-interval 20
)

ROLLOUT_ARGS=(
  --rollout-function-path \
    dressage.rollout.sync_rollout.generate_rollout_sync
  --custom-generate-function-path \
    dressage.rollout.generate.blackbox_dispatch.generate
  --custom-rm-path \
    dressage.reward.custom_rm.custom_rm
  --data-source-path \
    dressage.rollout.data_source.DressageDataSource
  --custom-reward-post-process-path \
    dressage.training.reward_post_process.reward_post_process
  --custom-convert-samples-to-train-data-path \
    dressage.rollout.convert_samples.convert_samples_to_train_data
  --custom-rollout-log-function-path \
    dressage.rollout.log_rollout.log_rollout_data

  --prompt-data \
    "${PROMPT_DATA:-${REPO_ROOT}/examples/data/dressage_dapo_prompts_dynamic.jsonl}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --rollout-shuffle
  --num-rollout 1
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-128}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-2}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
  --balance-data
  --rollout-global-dataset
)

EVAL_ARGS=(
  # Sync blackbox rollout does not support evaluation yet.
  # --eval-interval 20
)

PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size 1024
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --kl-coef 0.00
  --entropy-coef 0.00
  --eps-clip 0.2
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

WANDB_ARGS=(
  # --use-wandb
  # --wandb-project slime-dev
  # --wandb-group qwen3.5-4B-dressage-sync-8gpu
  # --wandb-key "${WANDB_KEY}"
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.7
  --sglang-enable-hierarchical-cache
  --sglang-hicache-ratio 2.0
  --sglang-hicache-write-policy write_through
  --sglang-hicache-mem-layout page_first_direct
  --sglang-hicache-storage-backend mooncake
  --sglang-hicache-storage-backend-extra-config \
    "${MOONCAKE_BACKEND_EXTRA_CONFIG}"
  --sglang-reasoning-parser qwen3
  --sglang-tool-call-parser qwen3_coder
  --sglang-log-level warning
  --sglang-router-port "${SGLANG_ROUTER_PORT}"
  --router-policy consistent_hashing
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

if [[ -f "${PROXY_PID_FILE}" ]]; then
  OLD_PROXY_PID="$(cat "${PROXY_PID_FILE}")"
  if ! kill -0 "${OLD_PROXY_PID}" 2>/dev/null; then
    rm -f "${PROXY_PID_FILE}"
  fi
fi

if [[ ! -f "${PROXY_PID_FILE}" ]]; then
  cd "${REPO_ROOT}"
  python3 -m dressage.proxy.server \
    "${PROXY_ARGS[@]}" \
    >"${PROXY_LOG_FILE}" 2>&1 &
  echo $! >"${PROXY_PID_FILE}"
  echo "Started Dressage proxy: pid=$(cat "${PROXY_PID_FILE}") log=${PROXY_LOG_FILE}"
fi

_stop_local_bwrap_pool_on_exit() {
  if [[ "${DRESSAGE_LOCAL_BWRAP_CLEANUP_ON_EXIT:-1}" == "1" &&
        "${DRESSAGE_SANDBOX_PROVIDER}" == "local_bwrap" ]]; then
    python -m dressage.sandbox.scripts.stop_local_bwrap || true
  fi
}

_stop_ray_cluster_on_exit() {
  if [[ "${DRESSAGE_RAY_STOP_ON_EXIT:-1}" != "1" ]]; then
    return
  fi

  if [[ -n "${HOSTFILE}" && -f "${HOSTFILE}" ]]; then
    for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
      if [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]]; then
        continue
      fi
      ssh root@"${WORKER_IP}" "ray stop --force || true" &
    done
    wait || true
  fi

  ray stop --force || true
}

_stop_proxy_on_exit() {
  local proxy_pid wait_count=0

  if [[ -f "${PROXY_PID_FILE}" ]]; then
    proxy_pid="$(cat "${PROXY_PID_FILE}")"
    if [[ "${proxy_pid}" =~ ^[0-9]+$ ]] && kill -0 "${proxy_pid}" 2>/dev/null; then
      kill "${proxy_pid}" 2>/dev/null || true
      while kill -0 "${proxy_pid}" 2>/dev/null && [[ "${wait_count}" -lt 100 ]]; do
        sleep 0.1
        wait_count=$((wait_count + 1))
      done
      if kill -0 "${proxy_pid}" 2>/dev/null; then
        echo "Dressage proxy did not stop gracefully; sending SIGKILL" >&2
        kill -9 "${proxy_pid}" 2>/dev/null || true
      fi
      wait "${proxy_pid}" 2>/dev/null || true
    fi
    rm -f "${PROXY_PID_FILE}"
  fi
}

_is_port_open() {
  local host="$1"
  local port="$2"
  (exec 3<>"/dev/tcp/${host}/${port}") >/dev/null 2>&1
}

_is_mooncake_master_process() {
  local pid="$1"
  ps -p "${pid}" -o args= 2>/dev/null | grep -q '[m]ooncake_master'
}

_stop_mooncake_master_on_exit() {
  local pid wait_count=0

  if [[ ! -f "${MOONCAKE_MASTER_PID_FILE}" ]]; then
    return
  fi

  pid="$(cat "${MOONCAKE_MASTER_PID_FILE}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    if ! _is_mooncake_master_process "${pid}"; then
      echo "Refusing to stop non-Mooncake process from stale PID file: ${pid}" >&2
      return 1
    fi
    kill "${pid}" 2>/dev/null || true
    while kill -0 "${pid}" 2>/dev/null && [[ "${wait_count}" -lt 50 ]]; do
      sleep 0.1
      wait_count=$((wait_count + 1))
    done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  fi
  rm -f "${MOONCAKE_MASTER_PID_FILE}"
}

_start_mooncake_master() {
  local pid

  command -v mooncake_master >/dev/null 2>&1 || {
    echo "Cannot find mooncake_master in PATH" >&2
    return 1
  }

  mkdir -p "${MOONCAKE_LOG_DIR}"
  if [[ -f "${MOONCAKE_MASTER_PID_FILE}" ]]; then
    pid="$(cat "${MOONCAKE_MASTER_PID_FILE}")"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null &&
       ! _is_mooncake_master_process "${pid}"; then
      echo "Stale Mooncake PID file points to another process: ${pid}" >&2
      return 1
    fi
    _stop_mooncake_master_on_exit
  fi

  if _is_port_open "${MOONCAKE_MASTER_HOST}" "${MOONCAKE_MASTER_PORT}" ||
     _is_port_open "${MOONCAKE_MASTER_HOST}" "${MOONCAKE_METADATA_PORT}"; then
    echo "Mooncake port ${MOONCAKE_MASTER_PORT} or ${MOONCAKE_METADATA_PORT} is already in use" >&2
    return 1
  fi

  mooncake_master \
    --enable_http_metadata_server=true \
    --http_metadata_server_port="${MOONCAKE_METADATA_PORT}" \
    --eviction_high_watermark_ratio=0.95 \
    >"${MOONCAKE_MASTER_LOG_FILE}" 2>&1 &
  pid=$!
  echo "${pid}" >"${MOONCAKE_MASTER_PID_FILE}"

  for _ in $(seq 1 60); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "Mooncake master exited during startup; see ${MOONCAKE_MASTER_LOG_FILE}" >&2
      return 1
    fi
    if _is_port_open "${MOONCAKE_MASTER_HOST}" "${MOONCAKE_MASTER_PORT}" &&
       _is_port_open "${MOONCAKE_MASTER_HOST}" "${MOONCAKE_METADATA_PORT}"; then
      echo "Started Mooncake master: pid=${pid} log=${MOONCAKE_MASTER_LOG_FILE}"
      return 0
    fi
    sleep 1
  done

  echo "Mooncake master failed readiness checks; see ${MOONCAKE_MASTER_LOG_FILE}" >&2
  return 1
}

cleanup() {
  status=$?
  set +e

  # 先停止 GPU 采样，避免把 Ray、代理等收尾时间计入平均值。
  _stop_gpu_util_monitor_on_exit

  # Proxy lifespan writes final.json before Ray and Mooncake are stopped.
  _stop_proxy_on_exit
  _stop_local_bwrap_pool_on_exit
  _stop_ray_cluster_on_exit
  _stop_mooncake_master_on_exit || true
  pkill -9 -f '[m]ooncake_master' 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT

_stop_local_bwrap_pool_on_exit
_stop_ray_cluster_on_exit
_start_mooncake_master

for i in $(seq 1 60); do
  if curl -sf "${DRESSAGE_PROXY_URL}/health" >/dev/null 2>&1; then
    echo "Dressage proxy is healthy"
    break
  fi

  if [[ "${i}" -eq 60 ]]; then
    echo "Dressage proxy failed health check; see ${PROXY_LOG_FILE}" >&2
    exit 1
  fi

  sleep 1
done

export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${PROXY_PUBLIC_HOST},${SGLANG_ROUTER_HOST}"

cd "${SLIME_ROOT}"

ray start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${RAY_NUM_GPUS_PER_NODE}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  "${DRESSAGE_BLACKBOX_HEAD_RESOURCE_ARGS[@]}"

if [[ -n "${HOSTFILE}" ]]; then
  for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    if [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]]; then
      continue
    fi

    echo "Starting Ray worker on ${WORKER_IP}"
    ssh root@"${WORKER_IP}" \
      "ray stop --force || true ; \
       ray start \
         --address=${MASTER_ADDR}:6379 \
         --num-gpus ${RAY_NUM_GPUS_PER_NODE} \
         --node-ip-address ${WORKER_IP} \
         --disable-usage-stats \
         ${DRESSAGE_BLACKBOX_WORKER_RESOURCE_ARGS}" &
  done
  wait
fi

CALIBRATION_STATE=""
for i in $(seq 1 900); do
  CALIBRATION_STATE="$(
    curl -sf "${DRESSAGE_PROXY_URL}/v1/engines/calibration" |
      python3 -c 'import json,sys; print(json.load(sys.stdin).get("state", ""))'
  )" || true
  if [[ "${CALIBRATION_STATE}" == "READY" ||
        "${CALIBRATION_STATE}" == "DEGRADED" ]]; then
    echo "Engine rebalancing calibration reached ${CALIBRATION_STATE}"
    break
  fi
  if [[ "${i}" -eq 900 ]]; then
    echo "Engine rebalancing calibration did not reach a terminal state" >&2
    exit 1
  fi
  sleep 1
done

if [[ "${DRESSAGE_SANDBOX_PROVIDER}" == "local_bwrap" &&
      "${DRESSAGE_LOCAL_BWRAP_AUTO_START}" == "1" ]]; then
  python -m dressage.sandbox.scripts.start_local_bwrap
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR},${PROXY_PUBLIC_HOST},${SGLANG_ROUTER_HOST}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "/root/Megatron-LM/:${REPO_ROOT}:${SLIME_ROOT}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "DRESSAGE_PROXY_URL": "${DRESSAGE_PROXY_URL}",
    "DRESSAGE_PADDOCK_MODE": "${DRESSAGE_PADDOCK_MODE}",
    "DRESSAGE_SANDBOX_PROVIDER": "${DRESSAGE_SANDBOX_PROVIDER}",
    "DRESSAGE_BLACKBOX_MAX_STEPS": "${DRESSAGE_BLACKBOX_MAX_STEPS}",
    "DRESSAGE_BLACKBOX_COMPACT_THRESHOLD": "${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD}",
    "DRESSAGE_LOCAL_BWRAP_POOL_MODE": "${DRESSAGE_LOCAL_BWRAP_POOL_MODE}",
    "DRESSAGE_LOCAL_BWRAP_RAY_NAMESPACE": "${DRESSAGE_LOCAL_BWRAP_RAY_NAMESPACE}",
    "DRESSAGE_LOCAL_BWRAP_MANAGER_NAME": "${DRESSAGE_LOCAL_BWRAP_MANAGER_NAME}",
    "DRESSAGE_LOCAL_BWRAP_TOTAL_SERVERS": "${DRESSAGE_LOCAL_BWRAP_TOTAL_SERVERS}",
    "DRESSAGE_LOCAL_BWRAP_BASE_PORT": "${DRESSAGE_LOCAL_BWRAP_BASE_PORT}",
    "DRESSAGE_BLACKBOX_ACQUIRE_TIMEOUT_SEC": "${DRESSAGE_BLACKBOX_ACQUIRE_TIMEOUT_SEC}",
    "DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR": "${DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR}",
    "DRESSAGE_TRAJECTORY_ERROR_LOG_DIR": "${DRESSAGE_TRAJECTORY_ERROR_LOG_DIR}",
    "DRESSAGE_REWARD_MODULES": "${DRESSAGE_REWARD_MODULES:-}"
  }
}
EOF_JSON
)

ray job submit \
  --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- \
  python3 train.py \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --colocate \
  --debug-rollout-only \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${COMM_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}"
