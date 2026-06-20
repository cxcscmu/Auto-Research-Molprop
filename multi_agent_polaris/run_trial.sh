#!/usr/bin/env bash
# run_trial.sh — Drug Discovery local trial entry point.
# Called by harness as: bash run_trial.sh <workdir>
# Env vars injected by harness: WORKDIR, HARNESS_PYTHON, AGENT_PYTHON.

set -euo pipefail

WORKDIR="${1:-${WORKDIR:-.}}"
# In local mode the harness spawns this script with cwd already set to the
# workdir, then passes the workdir as a *relative* path from the supervisor's
# own cwd.  That relative path does not resolve correctly from inside the
# workdir, so a plain `cd "$WORKDIR"` would fail under `set -e` and crash the
# trial.  Tolerate it: attempt cd; if it fails (local mode) we are already in
# the workdir, so fall back to $PWD.  In remote/job mode the cd succeeds as before.
cd "${WORKDIR}" 2>/dev/null || true
WORKDIR="${PWD}"

# HARNESS_PYTHON: full venv with PyTDC — runs run_trial_drug.py itself.
PYTHON="${HARNESS_PYTHON:-python3}"
if [ "$PYTHON" = "skip" ]; then PYTHON="python3"; fi

# AGENT_PYTHON: stripped venv WITHOUT PyTDC — runs experiment.py subprocesses.
# Passed through env so run_trial_drug.py can forward it to _run_subprocess.
export AGENT_PYTHON="${AGENT_PYTHON:-}"

OUT_DIR="full_eval_results/$(basename "$WORKDIR")"
mkdir -p "$OUT_DIR"

export HARNESS_WORKDIR="$WORKDIR"
export HARNESS_TDC_DATA_DIR="${HARNESS_TDC_DATA_DIR:-$HOME/drug_dev/tdc_data}"
export HARNESS_BASELINE_SCORES="${HARNESS_BASELINE_SCORES:-}"
# Honour whatever pod_env_for_trial injects (default 3600s); fall back to 3600
# if not set.  The outer local_config.timeout_s=4200 gives a 600s buffer for
# Python to finish writing the JSONL before the external SIGTERM fires.
export HARNESS_WALL_LIMIT_S="${HARNESS_WALL_LIMIT_S:-3600}"
export PYTHONUNBUFFERED=1

# 限制每个 trial 子进程的线程数，防止多路并发时 XGBoost(n_jobs=-1) 各自吃满 64 核
# 造成 oversubscribe（load 爆炸 → 撞 wall / fit 超时）。64 核 / ~8 路 ≈ 8。
# OMP 压住 XGBoost(OpenMP)，OPENBLAS/MKL 压住 numpy/sklearn。可被外部 env 覆盖。
# 辅助层（对 XGBoost/numpy 的 OpenMP 有效，但压不住 LightGBM/sklearn 的 n_jobs）
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# 主力：OS 层 CPU 亲和性硬限（库无关）。按 specialist 把整个 trial 绑到固定 8 核段，
# taskset 的亲和性被所有子进程（experiment.py 的 XGBoost / LightGBM / torch / 任何库）
# 继承——物理上锁死并行度 ≤ 8 核，agent 用任何模型都绕不过。
# 64 核 / 8 specialist = 每个一段；meta 跨 feature+model 两组时共享其段（可接受）。
SPEC=$(basename "$WORKDIR" | sed 's/^workdir_//')
case "$SPEC" in
  fphs)  CORES=0-7  ;; fsub)  CORES=8-15 ;; lit)   CORES=16-23 ;; data) CORES=24-31 ;;
  daugm) CORES=32-39;; modl)  CORES=40-47;; calib) CORES=48-55 ;; meta) CORES=56-63 ;;
  *)     CORES=0-7  ;;
esac
echo "[run_trial.sh] specialist=$SPEC → taskset -c $CORES (8 核硬限)" >&2

# taskset 不可用时（极少）退回直接执行，避免 trial 起不来
if command -v taskset >/dev/null 2>&1; then
  exec taskset -c "$CORES" "$PYTHON" run_trial_drug.py --out "$OUT_DIR/run_seed0.jsonl"
else
  echo "[run_trial.sh] WARN: taskset 不可用，未做 CPU 硬限" >&2
  exec "$PYTHON" run_trial_drug.py --out "$OUT_DIR/run_seed0.jsonl"
fi
