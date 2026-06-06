#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/maojianhua/model_select_on_physical_perspective"

GPU_DEVICE="${GPU_DEVICE:-cuda:2}"
SEED="${SEED:-2025}"
NUM_CLIENTS="${NUM_CLIENTS:-50}"
TARGET_CLIENT_NUM="${TARGET_CLIENT_NUM:-20}"
NUM_GLOBAL_ROUNDS="${NUM_GLOBAL_ROUNDS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LOCAL_LR="${LOCAL_LR:-1e-4}"
ALPHA="${ALPHA:-0.5}"
IMBALANCE_RATIO="${IMBALANCE_RATIO:-5}"
MODEL="${MODEL:-ResNet18}"

OURS_USE_DISTRIBUTION="${OURS_USE_DISTRIBUTION:-1}"
OURS_USE_QUALITY="${OURS_USE_QUALITY:-1}"
OURS_ADAPTIVE_BALANCE="${OURS_ADAPTIVE_BALANCE:-1}"
OURS_SELECTION_COUNT_MODE="${OURS_SELECTION_COUNT_MODE:-theory_optimal}"
OURS_THEORY_OPT_MIN_CLIENTS="${OURS_THEORY_OPT_MIN_CLIENTS:-5}"

OURS_FIXED_SCALE_SCORING="${OURS_FIXED_SCALE_SCORING:-1}"
OURS_KL_TEMPERATURE="${OURS_KL_TEMPERATURE:--1.0}"
OURS_KL_TEMPERATURE_QUANTILE="${OURS_KL_TEMPERATURE_QUANTILE:-0.5}"
OURS_ETA_MIN="${OURS_ETA_MIN:-0.20}"
OURS_ETA_MAX="${OURS_ETA_MAX:-0.50}"
OURS_SCORE_EPS="${OURS_SCORE_EPS:-1e-8}"

RUN_TAG="cifar10_fixed_scale_gpu2_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/result/log/fixed_scale_runner/${RUN_TAG}"
TRAIN_ROOT="${ROOT_DIR}/result/log/fixed_scale/${RUN_TAG}"
MANIFEST="${RUN_DIR}/manifest.csv"

mkdir -p "${RUN_DIR}" "${TRAIN_ROOT}"
echo "noise_mode,dataset,model,batch_size,target_client_num,screen,runner_log,train_log" > "${MANIFEST}"

submit_job() {
  local noise_mode="$1"
  local noise_short="$2"

  local dataset="cifar10"
  local logdir="${TRAIN_ROOT}/${noise_mode}/${dataset}/ours"
  local logfile="${dataset}_full_ours.log"
  local train_log="${logdir}/${logfile}"
  local runner_log="${RUN_DIR}/runner_${noise_mode}_${dataset}.log"
  local screen_name="fscale_g2_${noise_short}_cf10_${RUN_TAG#cifar10_fixed_scale_gpu2_}"

  mkdir -p "${logdir}"

  local cmd
  cmd="cd ${ROOT_DIR} && bash train_exp/train.sh \
    --seed ${SEED} \
    --data_name ${dataset} \
    --num_clients ${NUM_CLIENTS} \
    --target_client_num ${TARGET_CLIENT_NUM} \
    --num_global_rounds ${NUM_GLOBAL_ROUNDS} \
    --batch_size ${BATCH_SIZE} \
    --local_lr ${LOCAL_LR} \
    --alpha ${ALPHA} \
    --imbalance_ratio ${IMBALANCE_RATIO} \
    --model ${MODEL} \
    --device ${GPU_DEVICE} \
    --mode ours \
    --ablation full \
    --ours_use_distribution ${OURS_USE_DISTRIBUTION} \
    --ours_use_quality ${OURS_USE_QUALITY} \
    --ours_adaptive_balance ${OURS_ADAPTIVE_BALANCE} \
    --ours_selection_count_mode ${OURS_SELECTION_COUNT_MODE} \
    --ours_theory_opt_min_clients ${OURS_THEORY_OPT_MIN_CLIENTS} \
    --ours_fixed_scale_scoring ${OURS_FIXED_SCALE_SCORING} \
    --ours_kl_temperature ${OURS_KL_TEMPERATURE} \
    --ours_kl_temperature_quantile ${OURS_KL_TEMPERATURE_QUANTILE} \
    --ours_eta_min ${OURS_ETA_MIN} \
    --ours_eta_max ${OURS_ETA_MAX} \
    --ours_score_eps ${OURS_SCORE_EPS} \
    --noise_mode ${noise_mode} \
    --logdir ${logdir} \
    --logfile ${logfile} \
    >> ${runner_log} 2>&1"

  screen -dmS "${screen_name}" bash -lc "${cmd}"
  echo "${noise_mode},${dataset},${MODEL},${BATCH_SIZE},${TARGET_CLIENT_NUM},${screen_name},${runner_log},${train_log}" >> "${MANIFEST}"
  echo "[INFO] started noise=${noise_mode} dataset=${dataset} model=${MODEL} screen=${screen_name}"
}

echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "[INFO] RUN_DIR=${RUN_DIR}"
echo "[INFO] TRAIN_ROOT=${TRAIN_ROOT}"
echo "[INFO] GPU_DEVICE=${GPU_DEVICE}"
echo "[INFO] count_mode=${OURS_SELECTION_COUNT_MODE}"
echo "[INFO] fixed_scale=${OURS_FIXED_SCALE_SCORING} tau=${OURS_KL_TEMPERATURE} q=${OURS_KL_TEMPERATURE_QUANTILE}"
echo "[INFO] eta=[${OURS_ETA_MIN},${OURS_ETA_MAX}] eps=${OURS_SCORE_EPS}"

submit_job "linear_noise" "lin"
submit_job "gaussian_noise" "gau"

echo "[INFO] All fixed-scale cifar10 jobs submitted on ${GPU_DEVICE}."
echo "[INFO] Manifest: ${MANIFEST}"
echo "[INFO] Check screens with: screen -ls | grep fscale_g2"
