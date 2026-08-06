#!/usr/bin/env bash

set -u

SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"

STATE_DIR="${GPU_UTIL_STATE_DIR:-${TMPDIR:-/tmp}/dressage-gpu-util-${UID:-$(id -u)}}"
LOG_DIR="${GPU_UTIL_LOG_DIR:-${PWD}/log/gpu_utilization}"
INTERVAL="${GPU_UTIL_SAMPLE_INTERVAL:-1}"
GPU_IDS="${GPU_UTIL_GPU_IDS:-}"

query_gpu() {
    local gpu_ids="$1"
    local args=(
        --query-gpu=index,uuid,utilization.gpu
        --format=csv,noheader,nounits
    )

    if [[ -n "${gpu_ids}" ]]; then
        args+=(--id="${gpu_ids}")
    fi

    LC_ALL=C nvidia-smi "${args[@]}"
}

collect() {
    local csv_file="$1"
    local interval="$2"
    local gpu_ids="$3"
    local timestamp gpu_output sleep_pid=""

    stop_collecting() {
        if [[ -n "${sleep_pid}" ]]; then
            kill "${sleep_pid}" 2>/dev/null || true
        fi
        exit 0
    }

    trap stop_collecting TERM INT HUP

    while true; do
        # Use a literal decimal point instead of the locale-specific separator
        # emitted by `date --iso-8601=ns`. A comma here would corrupt the CSV.
        timestamp="$(LC_ALL=C date '+%Y-%m-%dT%H:%M:%S.%N%z')"

        if gpu_output="$(query_gpu "${gpu_ids}" 2>&1)"; then
            printf '%s\n' "${gpu_output}" |
                LC_ALL=C awk -F',' -v timestamp="${timestamp}" '
                {
                    gsub(/^[ \t]+|[ \t]+$/, "", $1)
                    gsub(/^[ \t]+|[ \t]+$/, "", $2)
                    gsub(/^[ \t]+|[ \t]+$/, "", $3)

                    if ($1 ~ /^[0-9]+$/ &&
                        $2 != "" &&
                        $3 ~ /^[0-9]+([.][0-9]+)?$/ &&
                        $3 + 0 >= 0 && $3 + 0 <= 100) {
                        printf "%s,%s,%s,%s\n",
                            timestamp, $1, $2, $3
                    } else {
                        print "Warning: ignored malformed nvidia-smi row: " $0 \
                            > "/dev/stderr"
                    }
                }
                ' >>"${csv_file}"
        else
            printf 'Warning: nvidia-smi query failed: %s\n' \
                "${gpu_output}" >&2
        fi

        sleep "${interval}" &
        sleep_pid=$!
        wait "${sleep_pid}" 2>/dev/null || true
        sleep_pid=""
    done
}

start_monitor() {
    local run_id pid csv_file summary_file collector_log initial_result

    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "错误：找不到 nvidia-smi" >&2
        return 1
    }

    if ! [[ "${INTERVAL}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        ! awk -v value="${INTERVAL}" \
            'BEGIN { exit !(value > 0) }'; then
        echo "错误：GPU_UTIL_SAMPLE_INTERVAL 必须大于 0" >&2
        return 1
    fi

    if ! initial_result="$(query_gpu "${GPU_IDS}" 2>&1)" ||
        [[ -z "${initial_result//[[:space:]]/}" ]]; then
        echo "错误：无法查询 NVIDIA GPU：${initial_result}" >&2
        return 1
    fi

    mkdir -p "${STATE_DIR}" "${LOG_DIR}"

    if [[ -f "${STATE_DIR}/pid" ]]; then
        pid="$(<"${STATE_DIR}/pid")"
        if [[ "${pid}" =~ ^[0-9]+$ ]] &&
            kill -0 "${pid}" 2>/dev/null; then
            echo "错误：GPU 监控已经启动，PID=${pid}" >&2
            return 1
        fi
    fi

    run_id="$(date '+%Y%m%d_%H%M%S')_$$"
    csv_file="${LOG_DIR}/gpu_utilization_${run_id}.csv"
    summary_file="${LOG_DIR}/gpu_utilization_${run_id}.summary.txt"
    collector_log="${LOG_DIR}/gpu_utilization_${run_id}.collector.log"

    printf 'timestamp,gpu_index,gpu_uuid,utilization_percent\n' \
        >"${csv_file}"

    printf '%s\n' "${csv_file}" >"${STATE_DIR}/csv_file"
    printf '%s\n' "${summary_file}" >"${STATE_DIR}/summary_file"
    printf '%s\n' "$(date +%s)" >"${STATE_DIR}/start_epoch"
    printf '%s\n' \
        "$(date --iso-8601=seconds 2>/dev/null ||
            date '+%Y-%m-%dT%H:%M:%S%z')" \
        >"${STATE_DIR}/start_time"
    printf '%s\n' "${INTERVAL}" >"${STATE_DIR}/interval"

    nohup bash "${SELF}" __collect \
        "${csv_file}" "${INTERVAL}" "${GPU_IDS}" \
        </dev/null >>"${collector_log}" 2>&1 &

    pid=$!
    printf '%s\n' "${pid}" >"${STATE_DIR}/pid"

    sleep 0.1
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "错误：GPU 监控启动失败，请检查 ${collector_log}" >&2
        return 1
    fi

    echo "GPU 监控已启动：PID=${pid}"
    echo "采样间隔：${INTERVAL} 秒"
    echo "原始数据：${csv_file}"
}

stop_monitor() {
    local pid csv_file summary_file
    local start_epoch start_time interval
    local end_epoch end_time duration summary_status wait_count=0

    if [[ ! -f "${STATE_DIR}/pid" ]]; then
        echo "没有正在运行的 GPU 监控"
        return 0
    fi

    pid="$(<"${STATE_DIR}/pid")"
    csv_file="$(<"${STATE_DIR}/csv_file")"
    summary_file="$(<"${STATE_DIR}/summary_file")"
    start_epoch="$(<"${STATE_DIR}/start_epoch")"
    start_time="$(<"${STATE_DIR}/start_time")"
    interval="$(<"${STATE_DIR}/interval")"

    if [[ "${pid}" =~ ^[0-9]+$ ]] &&
        kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}" 2>/dev/null || true

        while kill -0 "${pid}" 2>/dev/null &&
            [[ "${wait_count}" -lt 50 ]]; do
            sleep 0.1
            wait_count=$((wait_count + 1))
        done
    fi

    end_epoch="$(date +%s)"
    end_time="$(date --iso-8601=seconds 2>/dev/null ||
        date '+%Y-%m-%dT%H:%M:%S%z')"
    duration=$((end_epoch - start_epoch))

    LC_ALL=C awk -F',' \
        -v start_time="${start_time}" \
        -v end_time="${end_time}" \
        -v duration="${duration}" \
        -v interval="${interval}" \
        -v csv_file="${csv_file}" '
        NR == 1 {
            next
        }

        {
            if (NF != 4) {
                invalid_readings++
                next
            }

            timestamp = trim($1)
            gpu = trim($2)
            uuid = trim($3)
            utilization_text = trim($4)

            if (timestamp == "" ||
                gpu !~ /^[0-9]+$/ ||
                uuid == "" ||
                utilization_text !~ /^[0-9]+([.][0-9]+)?$/) {
                invalid_readings++
                next
            }

            utilization = utilization_text + 0
            if (utilization < 0 || utilization > 100) {
                invalid_readings++
                next
            }

            if ((gpu in seen) && gpu_uuid[gpu] != uuid) {
                invalid_readings++
                next
            }

            if (!(timestamp in seen_timestamp)) {
                seen_timestamp[timestamp] = 1
                sample_points++
            }

            if (!(gpu in seen)) {
                seen[gpu] = 1
                gpu_order[++gpu_count] = gpu
                gpu_uuid[gpu] = uuid
                gpu_min[gpu] = utilization
                gpu_max[gpu] = utilization
            }

            valid_readings++
            total += utilization

            gpu_samples[gpu]++
            gpu_total[gpu] += utilization

            if (utilization < gpu_min[gpu]) {
                gpu_min[gpu] = utilization
            }
            if (utilization > gpu_max[gpu]) {
                gpu_max[gpu] = utilization
            }
        }

        function trim(value) {
            gsub(/^[ \t]+|[ \t]+$/, "", value)
            return value
        }

        END {
            print "GPU utilization summary"
            if (valid_readings == 0) {
                print "status: error"
            } else if (invalid_readings > 0) {
                print "status: partial"
            } else {
                print "status: ok"
            }
            print "start_time: " start_time
            print "end_time: " end_time
            print "duration_seconds: " duration
            print "sample_interval_seconds: " interval
            print "sample_points: " (sample_points + 0)
            print "gpu_readings: " (valid_readings + 0)
            print "valid_readings: " (valid_readings + 0)
            print "invalid_readings: " (invalid_readings + 0)

            if (valid_readings > 0) {
                printf "average_gpu_utilization_percent: %.2f\n",
                    total / valid_readings
            } else {
                print "average_gpu_utilization_percent: N/A"
            }

            print "per_gpu:"

            for (i = 1; i <= gpu_count; i++) {
                gpu = gpu_order[i]

                printf \
                    "  GPU %s (%s): avg=%.2f%% min=%.2f%% max=%.2f%% samples=%d\n",
                    gpu,
                    gpu_uuid[gpu],
                    gpu_total[gpu] / gpu_samples[gpu],
                    gpu_min[gpu],
                    gpu_max[gpu],
                    gpu_samples[gpu]
            }

            print "raw_samples: " csv_file

            if (valid_readings == 0) {
                exit 1
            }
        }
    ' "${csv_file}" >"${summary_file}"
    summary_status=$?

    cat "${summary_file}"

    rm -f \
        "${STATE_DIR}/pid" \
        "${STATE_DIR}/csv_file" \
        "${STATE_DIR}/summary_file" \
        "${STATE_DIR}/start_epoch" \
        "${STATE_DIR}/start_time" \
        "${STATE_DIR}/interval"

    rmdir "${STATE_DIR}" 2>/dev/null || true

    echo "统计结果：${summary_file}"
    return "${summary_status}"
}

monitor_status() {
    local pid

    if [[ ! -f "${STATE_DIR}/pid" ]]; then
        echo "GPU 监控未运行"
        return 1
    fi

    pid="$(<"${STATE_DIR}/pid")"

    if [[ "${pid}" =~ ^[0-9]+$ ]] &&
        kill -0 "${pid}" 2>/dev/null; then
        echo "GPU 监控正在运行：PID=${pid}"
        echo "原始数据：$(<"${STATE_DIR}/csv_file")"
        return 0
    fi

    echo "GPU 监控进程已经停止，但存在旧状态文件"
    return 1
}

case "${1:-}" in
    start)
        start_monitor
        ;;
    stop)
        stop_monitor
        ;;
    status)
        monitor_status
        ;;
    __collect)
        collect "$2" "$3" "$4"
        ;;
    *)
        echo "用法：$0 {start|stop|status}" >&2
        exit 2
        ;;
esac
