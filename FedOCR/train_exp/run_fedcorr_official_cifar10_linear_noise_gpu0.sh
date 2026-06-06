#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/maojianhua/model_select_on_physical_perspective"
PYTHON_BIN="/home/maojianhua/anaconda3/envs/MS/bin/python3.12"

# 通用训练超参 (可被环境变量覆盖)
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
SEED="${SEED:-2025}"
NUM_CLIENTS="${NUM_CLIENTS:-50}"
TARGET_CLIENT_NUM="${TARGET_CLIENT_NUM:-20}"
NUM_GLOBAL_ROUNDS="${NUM_GLOBAL_ROUNDS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LOCAL_LR="${LOCAL_LR:-1e-4}"
ALPHA="${ALPHA:-0.5}"
IMBALANCE_RATIO="${IMBALANCE_RATIO:-5}"
MODEL="${MODEL:-ResNet18}"
DATA_NAME="${DATA_NAME:-cifar10}"
NOISE_MODE="${NOISE_MODE:-linear_noise}"

# FedCorr 专用超参 (与已有基线对齐: 200 评估 + 每轮 ~20 客户端)
FEDCORR_ITERATION1="${FEDCORR_ITERATION1:-5}"
FEDCORR_ROUNDS1="${FEDCORR_ROUNDS1:-100}"
FEDCORR_ROUNDS2="${FEDCORR_ROUNDS2:-95}"
FEDCORR_FRAC1="${FEDCORR_FRAC1:-0.1}"
FEDCORR_FRAC2="${FEDCORR_FRAC2:-0.4}"
FEDCORR_BETA="${FEDCORR_BETA:-5.0}"
FEDCORR_CORRECTION="${FEDCORR_CORRECTION:-1}"
FEDCORR_FINE_TUNING="${FEDCORR_FINE_TUNING:-1}"
FEDCORR_RELABEL_RATIO="${FEDCORR_RELABEL_RATIO:-0.5}"
FEDCORR_CONFIDENCE_THRES="${FEDCORR_CONFIDENCE_THRES:-0.5}"
FEDCORR_CLEAN_SET_THRES="${FEDCORR_CLEAN_SET_THRES:-0.1}"
FEDCORR_LID_K="${FEDCORR_LID_K:-20}"

# 命令行覆写：--gpu_device / --run_tag
RUN_TAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu_device) GPU_DEVICE="$2"; shift 2 ;;
    --run_tag)    RUN_TAG="$2"; shift 2 ;;
    --seed)       SEED="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${RUN_TAG}" ]]; then
  RUN_TAG="cifar10_linear_noise_$(echo "${GPU_DEVICE}" | tr -d ':')_$(date +%Y%m%d_%H%M%S)"
fi

LOG_DIR="${ROOT_DIR}/result/log/linear_noise_fedcorr_official/${DATA_NAME}/fedcorr"
LOG_FILE="${DATA_NAME}_fedcorr_official.log"
TRAIN_LOG="${LOG_DIR}/${LOG_FILE}"
RUN_DIR="${ROOT_DIR}/result/log/fedcorr_official_runner/${RUN_TAG}"
RUNNER_LOG="${RUN_DIR}/runner_fedcorr.log"
mkdir -p "${LOG_DIR}" "${RUN_DIR}"

{
  echo "[INFO] mode=fedcorr"
  echo "[INFO] gpu_device=${GPU_DEVICE}"
  echo "[INFO] run_tag=${RUN_TAG}"
  echo "[INFO] data_name=${DATA_NAME} noise_mode=${NOISE_MODE}"
  echo "[INFO] iteration1=${FEDCORR_ITERATION1} rounds1=${FEDCORR_ROUNDS1} rounds2=${FEDCORR_ROUNDS2}"
  echo "[INFO] frac1=${FEDCORR_FRAC1} frac2=${FEDCORR_FRAC2} beta=${FEDCORR_BETA}"
  echo "[INFO] correction=${FEDCORR_CORRECTION} fine_tuning=${FEDCORR_FINE_TUNING}"
  echo "[INFO] relabel_ratio=${FEDCORR_RELABEL_RATIO} conf_thres=${FEDCORR_CONFIDENCE_THRES}"
  echo "[INFO] clean_set_thres=${FEDCORR_CLEAN_SET_THRES} lid_k=${FEDCORR_LID_K}"
  echo "[INFO] train_log=${TRAIN_LOG}"
  echo "[INFO] runner_log=${RUNNER_LOG}"
  echo "[INFO] start_time=$(date '+%F %T')"
} >> "${RUNNER_LOG}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/main.py" \
  --seed "${SEED}" \
  --data_name "${DATA_NAME}" \
  --num_clients "${NUM_CLIENTS}" \
  --target_client_num "${TARGET_CLIENT_NUM}" \
  --num_global_rounds "${NUM_GLOBAL_ROUNDS}" \
  --batch_size "${BATCH_SIZE}" \
  --local_lr "${LOCAL_LR}" \
  --alpha "${ALPHA}" \
  --imbalance_ratio "${IMBALANCE_RATIO}" \
  --model "${MODEL}" \
  --device "${GPU_DEVICE}" \
  --mode fedcorr \
  --noise_mode "${NOISE_MODE}" \
  --fedcorr_iteration1 "${FEDCORR_ITERATION1}" \
  --fedcorr_rounds1 "${FEDCORR_ROUNDS1}" \
  --fedcorr_rounds2 "${FEDCORR_ROUNDS2}" \
  --fedcorr_frac1 "${FEDCORR_FRAC1}" \
  --fedcorr_frac2 "${FEDCORR_FRAC2}" \
  --fedcorr_beta "${FEDCORR_BETA}" \
  --fedcorr_correction "${FEDCORR_CORRECTION}" \
  --fedcorr_fine_tuning "${FEDCORR_FINE_TUNING}" \
  --fedcorr_relabel_ratio "${FEDCORR_RELABEL_RATIO}" \
  --fedcorr_confidence_thres "${FEDCORR_CONFIDENCE_THRES}" \
  --fedcorr_clean_set_thres "${FEDCORR_CLEAN_SET_THRES}" \
  --fedcorr_lid_k "${FEDCORR_LID_K}" \
  --logdir "${LOG_DIR}" \
  --logfile "${LOG_FILE}" \
  >> "${RUNNER_LOG}" 2>&1
