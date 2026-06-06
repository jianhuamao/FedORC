#!/bin/bash

set -e  # 遇到错误立即退出
# 允许使用 `sh train_exp/train.sh` 启动：若当前不是 bash，则自动切到 bash 执行。
if [ -z "${BASH_VERSION:-}" ]; then
    if command -v bash >/dev/null 2>&1; then
        exec bash "$0" "$@"
    fi
    echo "[ERROR] bash 未安装，无法运行 train.sh" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 默认参数
DATA_NAME='cifar10' # cifar10, cifar100
MODE='ours' # “ours, PoC, EMD, RS, GS, fedcor, fedcs”
DEVICE='cuda:2'
NOISE_MODE='linear_noise' #"linear_noise/gaussian_noise"
NOISE_GAUSSIAN_MEAN=0.3
NOISE_GAUSSIAN_STD=0.15
NOISE_GAUSSIAN_MIN=0.0
NOISE_GAUSSIAN_MAX=1.0
LOGDIR="result"
MODEL='ResNet18'
LOGFILE="theory_optimal.log"
LOGDIR_SET=0
LOGFILE_SET=0
ABLATION_PROFILE=""
ABLATION_NAME="none"
OURS_USE_DISTRIBUTION=1
OURS_USE_QUALITY=1
OURS_ADAPTIVE_BALANCE=1
OURS_ADAPTIVE_STRENGTH=1.0
OURS_ADAPTIVE_MEAN_WEIGHT=0.7
OURS_ADAPTIVE_HIGH_QUANTILE=0.75
OURS_ADAPTIVE_A_MIN=0.1
OURS_ADAPTIVE_A_MAX=4.0
OURS_ADAPTIVE_B_MIN=0.1
OURS_ADAPTIVE_B_MAX=4.0
OURS_SELECTION_COUNT_MODE='theory_optimal' #"theory_optimal, fixed"
OURS_THEORY_OPT_MIN_CLIENTS=5
OURS_FIXED_SCALE_SCORING=1
OURS_KL_TEMPERATURE=-1.0
OURS_KL_TEMPERATURE_QUANTILE=0.5
OURS_ETA_MIN=0.10
OURS_ETA_MAX=0.30
OURS_ETA_CLASS_SCALING=1
OURS_ETA_CLASS_REF=10.0
OURS_ETA_CLASS_GAMMA=0.18
OURS_QUALITY_CLASS_SCALING=1
OURS_QUALITY_CLASS_REF=100.0
OURS_QUALITY_CLASS_GAMMA=0.35
OURS_SCORE_EPS=1e-8
IMBALANCE_RATIO=5
SEED=2025
NUM_CLIENTS=50
TARGET_CLIENT_NUM=20
NUM_GLOBAL_ROUNDS=200
BATCH_SIZE=64
LOCAL_LR=1e-4
ALPHA=0.5 #“迪利克雷Non-IID参数”
CLIENT_MODEL_ON_CPU=1
USE_AMP=1
CUDA_EMPTY_CACHE=1
FEDCS_CLIENT_SELECTION='fixed'
FEDCS_PRUNING_RATE_F=0.5
FEDCS_PRUNING_RATE_L=0.1
FEDCS_BETA=0.5
FEDCS_SELECT_INTERVAL=2000
FEDCS_MIN_KEEP_SAMPLES=1

# 解析脚本参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift 2 ;;
        --data_name) DATA_NAME="$2"; shift 2 ;;
        --num_clients) NUM_CLIENTS="$2"; shift 2 ;;
        --num_global_rounds) NUM_GLOBAL_ROUNDS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --local_lr) LOCAL_LR="$2"; shift 2 ;;
        --alpha) ALPHA="$2"; shift 2 ;;
        --imbalance_ratio) IMBALANCE_RATIO="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --client_model_on_cpu) CLIENT_MODEL_ON_CPU="$2"; shift 2 ;;
        --use_amp) USE_AMP="$2"; shift 2 ;;
        --cuda_empty_cache) CUDA_EMPTY_CACHE="$2"; shift 2 ;;
        --logdir) LOGDIR="$2"; LOGDIR_SET=1; shift 2 ;;
        # --- 添加缺失的参数 ---
        --target_client_num) TARGET_CLIENT_NUM="$2"; shift 2 ;;
        --noise_mode) NOISE_MODE="$2"; shift 2 ;;
        --noise_gaussian_mean) NOISE_GAUSSIAN_MEAN="$2"; shift 2 ;;
        --noise_gaussian_std) NOISE_GAUSSIAN_STD="$2"; shift 2 ;;
        --noise_gaussian_min) NOISE_GAUSSIAN_MIN="$2"; shift 2 ;;
        --noise_gaussian_max) NOISE_GAUSSIAN_MAX="$2"; shift 2 ;;
        --logfile) LOGFILE="$2"; LOGFILE_SET=1; shift 2 ;;
        --ablation) ABLATION_PROFILE="$2"; shift 2 ;;
        --ours_use_distribution) OURS_USE_DISTRIBUTION="$2"; shift 2 ;;
        --ours_use_quality) OURS_USE_QUALITY="$2"; shift 2 ;;
        --ours_adaptive_balance) OURS_ADAPTIVE_BALANCE="$2"; shift 2 ;;
        --ours_adaptive_strength) OURS_ADAPTIVE_STRENGTH="$2"; shift 2 ;;
        --ours_adaptive_mean_weight) OURS_ADAPTIVE_MEAN_WEIGHT="$2"; shift 2 ;;
        --ours_adaptive_high_quantile) OURS_ADAPTIVE_HIGH_QUANTILE="$2"; shift 2 ;;
        --ours_adaptive_a_min) OURS_ADAPTIVE_A_MIN="$2"; shift 2 ;;
        --ours_adaptive_a_max) OURS_ADAPTIVE_A_MAX="$2"; shift 2 ;;
        --ours_adaptive_b_min) OURS_ADAPTIVE_B_MIN="$2"; shift 2 ;;
        --ours_adaptive_b_max) OURS_ADAPTIVE_B_MAX="$2"; shift 2 ;;
        --ours_selection_count_mode) OURS_SELECTION_COUNT_MODE="$2"; shift 2 ;;
        --ours_theory_opt_min_clients) OURS_THEORY_OPT_MIN_CLIENTS="$2"; shift 2 ;;
        --ours_fixed_scale_scoring) OURS_FIXED_SCALE_SCORING="$2"; shift 2 ;;
        --ours_kl_temperature) OURS_KL_TEMPERATURE="$2"; shift 2 ;;
        --ours_kl_temperature_quantile) OURS_KL_TEMPERATURE_QUANTILE="$2"; shift 2 ;;
        --ours_eta_min) OURS_ETA_MIN="$2"; shift 2 ;;
        --ours_eta_max) OURS_ETA_MAX="$2"; shift 2 ;;
        --ours_eta_class_scaling) OURS_ETA_CLASS_SCALING="$2"; shift 2 ;;
        --ours_eta_class_ref) OURS_ETA_CLASS_REF="$2"; shift 2 ;;
        --ours_eta_class_gamma) OURS_ETA_CLASS_GAMMA="$2"; shift 2 ;;
        --ours_quality_class_scaling) OURS_QUALITY_CLASS_SCALING="$2"; shift 2 ;;
        --ours_quality_class_ref) OURS_QUALITY_CLASS_REF="$2"; shift 2 ;;
        --ours_quality_class_gamma) OURS_QUALITY_CLASS_GAMMA="$2"; shift 2 ;;
        --ours_score_eps) OURS_SCORE_EPS="$2"; shift 2 ;;
        --fedcs_client_selection) FEDCS_CLIENT_SELECTION="$2"; shift 2 ;;
        --fedcs_pruning_rate_f) FEDCS_PRUNING_RATE_F="$2"; shift 2 ;;
        --fedcs_pruning_rate_l) FEDCS_PRUNING_RATE_L="$2"; shift 2 ;;
        --fedcs_beta) FEDCS_BETA="$2"; shift 2 ;;
        --fedcs_select_interval) FEDCS_SELECT_INTERVAL="$2"; shift 2 ;;
        --fedcs_min_keep_samples) FEDCS_MIN_KEEP_SAMPLES="$2"; shift 2 ;;
        # -------------------
        --help)
            echo "用法：$0 [选项]"
            echo "选项:"
            echo "  --seed              随机种子 (默认：2025)"
            echo "  --data_name         数据集名称 (默认：cifar10)"
            echo "  --num_clients       客户端数量 (默认：50)"
            echo "  --target_client_num 目标客户端数 (默认：20)"
            echo "  --num_global_rounds 全局训练轮数 (默认：200)"
            echo "  --batch_size        批大小 (默认：64)"
            echo "  --local_lr          本地学习率 (默认：1e-4)"
            echo "  --alpha             数据分布 Dirichlet 参数 (默认：0.5)"
            echo "  --imbalance_ratio   类别不平衡比例 (默认：5)"
            echo "  --model             模型类型 (默认：ResNet34)"
            echo "  --device            计算设备 (默认：cuda:4)"
            echo "  --mode              选择算法 (默认：PoC)"
            echo "  --client_model_on_cpu 客户端模型默认驻留CPU (默认：1)"
            echo "  --use_amp           启用混合精度训练 (默认：1)"
            echo "  --cuda_empty_cache  每个客户端后清理CUDA缓存 (默认：1)"
            echo "  --noise_mode        噪声模式: linear_noise/gaussian_noise (默认：linear_noise)"
            echo "  --noise_gaussian_mean gaussian_noise 客户端噪声率均值 (默认：0.3)"
            echo "  --noise_gaussian_std  gaussian_noise 客户端噪声率标准差 (默认：0.15)"
            echo "  --noise_gaussian_min  gaussian_noise 噪声率截断下界 (默认：0.0)"
            echo "  --noise_gaussian_max  gaussian_noise 噪声率截断上界 (默认：1.0)"
            echo "  --logdir            日志目录 (默认：./result/log)"
            echo "  --logfile           日志文件名 (默认：mode.log)"
            echo "  --ablation          ours 消融配置: full/distribution_only/quality_only"
            echo "  --ours_use_distribution ours 是否启用分布分数 (默认：1)"
            echo "  --ours_use_quality  ours 是否启用质量分数 (默认：1)"
            echo "  --ours_adaptive_balance ours 是否启用噪声自适应权重 (默认：0)"
            echo "  --ours_adaptive_strength ours 自适应强度 lambda (默认：1.0)"
            echo "  --ours_adaptive_mean_weight ours 噪声强度均值权重 (默认：0.7)"
            echo "  --ours_adaptive_high_quantile ours 噪声强度高分位点 (默认：0.75)"
            echo "  --ours_adaptive_a_min ours a_eff 下界 (默认：0.1)"
            echo "  --ours_adaptive_a_max ours a_eff 上界 (默认：4.0)"
            echo "  --ours_adaptive_b_min ours b_eff 下界 (默认：0.1)"
            echo "  --ours_adaptive_b_max ours b_eff 上界 (默认：4.0)"
            echo "  --ours_selection_count_mode ours 客户端数量策略: fixed/theory_optimal (默认：fixed)"
            echo "  --ours_theory_opt_min_clients theory_optimal 模式下最小客户端数 (默认：1)"
            echo "  --ours_fixed_scale_scoring ours 固定尺度打分开关: 1/0 (默认：1)"
            echo "  --ours_kl_temperature ours KL 温度tau（>0固定，<0自适应）"
            echo "  --ours_kl_temperature_quantile ours KL温度自适应分位点q (默认：0.5)"
            echo "  --ours_eta_min ours 收益基准下界 eta_min (默认：0.20)"
            echo "  --ours_eta_max ours 收益基准上界 eta_max (默认：0.50)"
            echo "  --ours_eta_class_scaling ours 是否按类别数缩放 eta: 1/0 (默认：1)"
            echo "  --ours_eta_class_ref ours eta 类别数缩放参考类别数 K_ref (默认：10)"
            echo "  --ours_eta_class_gamma ours eta 类别数缩放指数 gamma (默认：0.18)"
            echo "  --ours_quality_class_scaling ours 是否按类别数缩放 b_eff: 1/0 (默认：1)"
            echo "  --ours_quality_class_ref ours b_eff 类别数缩放参考类别数 K_ref (默认：100)"
            echo "  --ours_quality_class_gamma ours b_eff 类别数缩放指数 gamma (默认：0.35)"
            echo "  --ours_score_eps ours 数值稳定epsilon (默认：1e-8)"
            echo "  --fedcs_client_selection FedCS 客户端集合策略: fixed/random (默认：fixed)"
            echo "  --fedcs_pruning_rate_f FedCS 第一阶段剪枝比例 (默认：0.5)"
            echo "  --fedcs_pruning_rate_l FedCS 第二阶段剪枝比例 (默认：0.1)"
            echo "  --fedcs_beta FedCS 大样本类阈值 beta (默认：0.5)"
            echo "  --fedcs_select_interval FedCS 重算客户端子集间隔轮数 (默认：2000)"
            echo "  --fedcs_min_keep_samples FedCS 每客户端最少保留样本数 (默认：1)"
            echo "  --help              显示帮助信息"
            exit 0
            ;;
        *)
            log_error "未知参数：$1"
            exit 1
            ;;
    esac
done

# ours 消融快捷配置
if [[ -n "${ABLATION_PROFILE}" ]]; then
    case "${ABLATION_PROFILE}" in
        full|both)
            MODE='ours'
            ABLATION_NAME='full'
            OURS_USE_DISTRIBUTION=1
            OURS_USE_QUALITY=1
            ;;
        distribution_only|wo_quality)
            MODE='ours_distribution_only'
            ABLATION_NAME='distribution_only'
            OURS_USE_DISTRIBUTION=1
            OURS_USE_QUALITY=0
            ;;
        quality_only|wo_distribution)
            MODE='ours_quality_only'
            ABLATION_NAME='quality_only'
            OURS_USE_DISTRIBUTION=0
            OURS_USE_QUALITY=1
            ;;
        *)
            log_error "未知 ablation 配置：${ABLATION_PROFILE}，可选 full/distribution_only/quality_only"
            exit 1
            ;;
    esac
fi

# 若未显式使用 --ablation，但 mode/开关已是消融配置，也推断出消融名用于日志记录
if [[ "${ABLATION_NAME}" == "none" ]]; then
    if [[ "${MODE}" == "ours_distribution_only" || "${MODE}" == "ours_wo_quality" ]]; then
        ABLATION_NAME='distribution_only'
    elif [[ "${MODE}" == "ours_quality_only" || "${MODE}" == "ours_wo_distribution" ]]; then
        ABLATION_NAME='quality_only'
    elif [[ "${MODE}" == "ours" && "${OURS_USE_DISTRIBUTION}" -eq 1 && "${OURS_USE_QUALITY}" -eq 1 ]]; then
        ABLATION_NAME='full'
    elif [[ "${MODE}" == "ours" && "${OURS_USE_DISTRIBUTION}" -eq 1 && "${OURS_USE_QUALITY}" -eq 0 ]]; then
        ABLATION_NAME='distribution_only'
    elif [[ "${MODE}" == "ours" && "${OURS_USE_DISTRIBUTION}" -eq 0 && "${OURS_USE_QUALITY}" -eq 1 ]]; then
        ABLATION_NAME='quality_only'
    fi
fi

# 根据 mode 统一计算实验名
EXPERIMENT_NAME="${MODE}"

if [ "$LOGFILE_SET" -eq 0 ]; then
    if [[ "${ABLATION_NAME}" != "none" ]]; then
        LOGFILE="${DATA_NAME}_${ABLATION_NAME}_${EXPERIMENT_NAME}.log"
    else
        LOGFILE="${EXPERIMENT_NAME}.log"
    fi
fi
if [ "$LOGDIR_SET" -eq 0 ]; then
    LOG_ROOT="${PROJECT_ROOT}/result/log"
    if [[ "${OURS_SELECTION_COUNT_MODE}" == "theory_optimal" ]]; then
        # YJQ4.21: theory_optimal 实验日志单独隔离到 log4theory_optimal，避免和既有 result/log 混放。
        LOG_ROOT="${PROJECT_ROOT}/result/log4theory_optimal"
    fi
    LOGDIR="${LOG_ROOT}/${NOISE_MODE}/${DATA_NAME}/${MODE}"
elif [[ "${LOGDIR}" != /* ]]; then
    # 用户传相对路径时，统一按项目根目录解析，避免从不同 cwd 启动导致日志写到意外位置。
    LOGDIR="${PROJECT_ROOT}/${LOGDIR#./}"
fi

# 保持调用方显式指定的 batch_size；如 ResNet34 显存不足，请在提交命令中手动调小。
if [[ "${MODEL}" == "ResNet34" && "${BATCH_SIZE}" -gt 32 ]]; then
    log_warning "检测到模型为 ResNet34 且 batch_size=${BATCH_SIZE}，将按原值运行"
fi

# 创建日志目录
mkdir -p "${LOGDIR}"
LOG_PATH="${LOGDIR}/${LOGFILE}"
touch "${LOG_PATH}"

log_info "========== 实验配置 =========="
log_info "实验名称：${EXPERIMENT_NAME}"
log_info "随机种子：${SEED}"
log_info "数据集：${DATA_NAME}"
log_info "客户端数：${NUM_CLIENTS}"
log_info "目标客户端数：${TARGET_CLIENT_NUM}"
log_info "全局轮数：${NUM_GLOBAL_ROUNDS}"
log_info "批大小：${BATCH_SIZE}"
log_info "学习率：${LOCAL_LR}"
log_info "Alpha：${ALPHA}"
log_info "不平衡比：${IMBALANCE_RATIO}"
log_info "模型：${MODEL}"
log_info "设备：${DEVICE}"
log_info "客户端模型驻留CPU：${CLIENT_MODEL_ON_CPU}"
log_info "混合精度训练：${USE_AMP}"
log_info "训练后清理CUDA缓存：${CUDA_EMPTY_CACHE}"
log_info "选择算法：${MODE}"
log_info "消融配置：${ABLATION_PROFILE:-none}"
log_info "消融标识：${ABLATION_NAME}"
log_info "ours 分布分数开关：${OURS_USE_DISTRIBUTION}"
log_info "ours 质量分数开关：${OURS_USE_QUALITY}"
log_info "ours 自适应权重开关：${OURS_ADAPTIVE_BALANCE}"
log_info "ours 自适应强度(lambda)：${OURS_ADAPTIVE_STRENGTH}"
log_info "ours 自适应均值权重：${OURS_ADAPTIVE_MEAN_WEIGHT}"
log_info "ours 自适应高分位点：${OURS_ADAPTIVE_HIGH_QUANTILE}"
log_info "ours 自适应 a 范围：[${OURS_ADAPTIVE_A_MIN}, ${OURS_ADAPTIVE_A_MAX}]"
log_info "ours 自适应 b 范围：[${OURS_ADAPTIVE_B_MIN}, ${OURS_ADAPTIVE_B_MAX}]"
log_info "ours 客户端数量策略：${OURS_SELECTION_COUNT_MODE}"
log_info "ours theory-optimal 最小客户端数：${OURS_THEORY_OPT_MIN_CLIENTS}"
log_info "ours 固定尺度打分：${OURS_FIXED_SCALE_SCORING}"
log_info "ours KL温度 tau：${OURS_KL_TEMPERATURE} (q=${OURS_KL_TEMPERATURE_QUANTILE})"
log_info "ours 收益基准区间：[${OURS_ETA_MIN}, ${OURS_ETA_MAX}]"
log_info "ours 收益基准类别缩放：${OURS_ETA_CLASS_SCALING} (K_ref=${OURS_ETA_CLASS_REF}, gamma=${OURS_ETA_CLASS_GAMMA})"
log_info "ours 质量指数类别缩放：${OURS_QUALITY_CLASS_SCALING} (K_ref=${OURS_QUALITY_CLASS_REF}, gamma=${OURS_QUALITY_CLASS_GAMMA})"
log_info "ours 数值稳定 epsilon：${OURS_SCORE_EPS}"
if [[ "${MODE,,}" == "fedcs" ]]; then
    log_info "FedCS 客户端集合策略：${FEDCS_CLIENT_SELECTION}"
    log_info "FedCS 剪枝比例：first=${FEDCS_PRUNING_RATE_F}, second=${FEDCS_PRUNING_RATE_L}"
    log_info "FedCS beta：${FEDCS_BETA}"
    log_info "FedCS 子集重算间隔：${FEDCS_SELECT_INTERVAL}"
    log_info "FedCS 最少保留样本数：${FEDCS_MIN_KEEP_SAMPLES}"
fi
log_info "噪声模式：${NOISE_MODE}"
if [[ "${NOISE_MODE}" == "gaussian_noise" || "${NOISE_MODE}" == "gaussian" || "${NOISE_MODE}" == "normal_noise" || "${NOISE_MODE}" == "normal" ]]; then
    log_info "高斯噪声参数：mean=${NOISE_GAUSSIAN_MEAN}, std=${NOISE_GAUSSIAN_STD}, clip=[${NOISE_GAUSSIAN_MIN}, ${NOISE_GAUSSIAN_MAX}]"
fi
log_info "日志目录：${LOGDIR}"
log_info "日志文件：${LOGFILE}"
log_info "日志完整路径：${LOG_PATH}"
log_info "Python：${PYTHON_BIN}"
log_info "=============================="

# 运行训练
log_info "开始训练..."

"${PYTHON_BIN}" "${PROJECT_ROOT}/main.py" \
    --seed ${SEED} \
    --data_name ${DATA_NAME} \
    --num_clients ${NUM_CLIENTS} \
    --target_client_num ${TARGET_CLIENT_NUM} \
    --num_global_rounds ${NUM_GLOBAL_ROUNDS} \
    --batch_size ${BATCH_SIZE} \
    --local_lr ${LOCAL_LR} \
    --alpha ${ALPHA} \
    --imbalance_ratio ${IMBALANCE_RATIO} \
    --model ${MODEL} \
    --device ${DEVICE} \
    --client_model_on_cpu ${CLIENT_MODEL_ON_CPU} \
    --use_amp ${USE_AMP} \
    --cuda_empty_cache ${CUDA_EMPTY_CACHE} \
    --mode ${MODE} \
    --fedcs_client_selection ${FEDCS_CLIENT_SELECTION} \
    --fedcs_pruning_rate_f ${FEDCS_PRUNING_RATE_F} \
    --fedcs_pruning_rate_l ${FEDCS_PRUNING_RATE_L} \
    --fedcs_beta ${FEDCS_BETA} \
    --fedcs_select_interval ${FEDCS_SELECT_INTERVAL} \
    --fedcs_min_keep_samples ${FEDCS_MIN_KEEP_SAMPLES} \
    --ours_use_distribution ${OURS_USE_DISTRIBUTION} \
    --ours_use_quality ${OURS_USE_QUALITY} \
    --ours_adaptive_balance ${OURS_ADAPTIVE_BALANCE} \
    --ours_adaptive_strength ${OURS_ADAPTIVE_STRENGTH} \
    --ours_adaptive_mean_weight ${OURS_ADAPTIVE_MEAN_WEIGHT} \
    --ours_adaptive_high_quantile ${OURS_ADAPTIVE_HIGH_QUANTILE} \
    --ours_adaptive_a_min ${OURS_ADAPTIVE_A_MIN} \
    --ours_adaptive_a_max ${OURS_ADAPTIVE_A_MAX} \
    --ours_adaptive_b_min ${OURS_ADAPTIVE_B_MIN} \
    --ours_adaptive_b_max ${OURS_ADAPTIVE_B_MAX} \
    --ours_selection_count_mode ${OURS_SELECTION_COUNT_MODE} \
    --ours_theory_opt_min_clients ${OURS_THEORY_OPT_MIN_CLIENTS} \
    --ours_fixed_scale_scoring ${OURS_FIXED_SCALE_SCORING} \
    --ours_kl_temperature ${OURS_KL_TEMPERATURE} \
    --ours_kl_temperature_quantile ${OURS_KL_TEMPERATURE_QUANTILE} \
    --ours_eta_min ${OURS_ETA_MIN} \
    --ours_eta_max ${OURS_ETA_MAX} \
    --ours_eta_class_scaling ${OURS_ETA_CLASS_SCALING} \
    --ours_eta_class_ref ${OURS_ETA_CLASS_REF} \
    --ours_eta_class_gamma ${OURS_ETA_CLASS_GAMMA} \
    --ours_quality_class_scaling ${OURS_QUALITY_CLASS_SCALING} \
    --ours_quality_class_ref ${OURS_QUALITY_CLASS_REF} \
    --ours_quality_class_gamma ${OURS_QUALITY_CLASS_GAMMA} \
    --ours_score_eps ${OURS_SCORE_EPS} \
    --noise_mode ${NOISE_MODE} \
    --noise_gaussian_mean ${NOISE_GAUSSIAN_MEAN} \
    --noise_gaussian_std ${NOISE_GAUSSIAN_STD} \
    --noise_gaussian_min ${NOISE_GAUSSIAN_MIN} \
    --noise_gaussian_max ${NOISE_GAUSSIAN_MAX} \
    --logdir ${LOGDIR} \
    --logfile ${LOGFILE}

if [ $? -eq 0 ]; then
    log_success "训练完成！日志文件：${LOGDIR}/${LOGFILE}"
else
    log_error "训练失败！"
    exit 1
fi
