
import torch
import numpy as np
from lib.settings import parse_args, set_seed, get_model_class, get_client_args, Config
from lib.dataset.pre_data import prepare_data
from lib.dataset.dataset import get_noise_datasets, get_loaders
from train import train_script
from lib.utils import init_log, get_dataset_arg, get_noise_rate
import logging


def resolve_ablation_name(args):
    if args.mode in {'ours_distribution_only', 'ours_wo_quality'}:
        return 'distribution_only'
    if args.mode in {'ours_quality_only', 'ours_wo_distribution'}:
        return 'quality_only'
    if args.mode != 'ours':
        return 'none'

    if args.ours_use_distribution and args.ours_use_quality:
        return 'full'
    if args.ours_use_distribution and (not args.ours_use_quality):
        return 'distribution_only'
    if (not args.ours_use_distribution) and args.ours_use_quality:
        return 'quality_only'
    return 'invalid'


if __name__ == '__main__':
    args = Config()
    params = parse_args()
    args.update_from_args(params)

    # Parse client_subset and compute dynamic target_client_num
    if getattr(args, 'client_subset', None):
        subset = [int(x.strip()) for x in args.client_subset.split(',') if x.strip()]
        args.client_subset = subset
        pool_size = len(subset)
        if pool_size > 30:
            dynamic_target = 15
        elif pool_size < 15:
            dynamic_target = max(1, pool_size // 2)
        else:
            dynamic_target = pool_size
        args.target_client_num = dynamic_target

    init_log(args)

    if getattr(args, 'client_subset', None):
        logging.info(
            "Client subset specified: %d clients, target_client_num=%d",
            len(args.client_subset), args.target_client_num,
        )
    ablation_name = resolve_ablation_name(args)
    logging.info(
        f"Experiment Meta | dataset={args.data_name} mode={args.mode} "
        f"ablation={ablation_name} noise_mode={args.noise_mode}"
    )
    if str(args.noise_mode).strip().lower().replace("-", "_") in {"gaussian_noise", "gaussian", "normal_noise", "normal"}:
        logging.info(
            f"Gaussian Noise | mean={args.noise_gaussian_mean} "
            f"std={args.noise_gaussian_std} "
            f"clip=[{args.noise_gaussian_min},{args.noise_gaussian_max}]"
        )
    logging.info(
        f"Ours Flags | use_distribution={int(args.ours_use_distribution)} "
        f"use_quality={int(args.ours_use_quality)} "
        f"distribution_strategy={args.ours_distribution_strategy} "
        f"distribution_source={args.ours_distribution_source} "
        f"a={args.ours_distribution_power} b={args.ours_quality_power} "
        f"adaptive={int(args.ours_adaptive_balance)} "
        f"adaptive_lambda={args.ours_adaptive_strength} "
        f"adaptive_mean_w={args.ours_adaptive_mean_weight} "
        f"adaptive_q={args.ours_adaptive_high_quantile} "
        f"a_range=[{args.ours_adaptive_a_min},{args.ours_adaptive_a_max}] "
        f"b_range=[{args.ours_adaptive_b_min},{args.ours_adaptive_b_max}] "
        f"count_mode={args.ours_selection_count_mode} "
        f"theory_min_clients={args.ours_theory_opt_min_clients} "
        f"fixed_scale={int(args.ours_fixed_scale_scoring)} "
        f"tau={args.ours_kl_temperature} "
        f"tau_q={args.ours_kl_temperature_quantile} "
        f"eta=[{args.ours_eta_min},{args.ours_eta_max}] "
        f"eta_class_scaling={int(args.ours_eta_class_scaling)} "
        f"eta_class_ref={args.ours_eta_class_ref} "
        f"eta_class_gamma={args.ours_eta_class_gamma} "
        f"quality_class_scaling={int(args.ours_quality_class_scaling)} "
        f"quality_class_ref={args.ours_quality_class_ref} "
        f"quality_class_gamma={args.ours_quality_class_gamma} "
        f"score_eps={args.ours_score_eps}"
    )
    logging.info(
        f"Local Denoise | method={args.local_denoise_method} "
        f"forget_rate={args.denoise_forget_rate} "
        f"num_gradual={args.denoise_num_gradual} "
        f"warmup={args.denoise_warmup_epochs} "
        f"lambda_u={args.denoise_lambda_u} "
        f"temperature={args.denoise_temperature} "
        f"mixup_alpha={args.denoise_mixup_alpha} "
        f"p_threshold={args.denoise_p_threshold} "
        f"rampup_len={args.denoise_rampup_length} "
        f"jocor_lambda={args.jocor_co_lambda}"
    )
    logging.info(
        f"FedFixer | lambda={args.fedfixer_lambda} "
        f"beta={args.fedfixer_beta} "
        f"forget_rate={args.fedfixer_forget_rate} "
        f"num_gradual={args.fedfixer_num_gradual} "
        f"warmup_epochs={args.fedfixer_warmup_epochs}"
    )
    logging.info(
        f"FedDiv | warmup_rounds={args.feddiv_warmup_rounds} "
        f"confidence_threshold={args.feddiv_confidence_threshold} "
        f"consistency_weight={args.feddiv_consistency_weight} "
        f"gmm_max_iter={args.feddiv_gmm_max_iter}"
    )
    logging.info(
        f"Label Correction | enabled={int(args.enable_label_correction)} "
        f"method={args.label_correction_method} "
        f"start_min_rounds={args.label_correction_start_min_rounds} "
        f"patience={args.label_correction_patience} "
        f"min_delta={args.label_correction_min_delta} "
        f"top_fraction={args.label_correction_top_fraction} top_k={args.label_correction_top_k} "
        f"proxy_count={args.label_correction_proxy_count} "
        f"conf_th={args.label_correction_confidence_threshold} "
        f"consensus_th={args.label_correction_consensus_threshold} "
        f"noisy_q={args.label_correction_noisy_quantile} "
        f"max_relabel_ratio={args.label_correction_max_relabel_ratio} "
        f"accept_conf={args.label_correction_accept_conf} "
        f"accept_entropy={args.label_correction_accept_entropy} "
        f"min_conf_gain={args.label_correction_accept_min_conf_gain} "
        f"min_ent_drop={args.label_correction_accept_min_entropy_drop} "
        f"join_patience={args.label_correction_accept_join_patience} "
        f"evict_patience={args.label_correction_accept_evict_patience} "
        f"dyn_q={args.label_correction_accept_dynamic_quantile} "
        f"dyn_margin={args.label_correction_accept_dynamic_margin} "
        f"new_boost={args.label_correction_new_client_boost} "
        f"boost_decay={args.label_correction_boost_decay}"
    )
    logging.info(
        f"Federated Optimizer | aggregation={args.aggregation} "
        f"fedprox_mu={args.fedprox_mu} "
        f"fltrust_root_lr={args.fltrust_root_lr} "
        f"fltrust_root_steps={args.fltrust_root_steps} "
        f"fltrust_root_batch_size={args.fltrust_root_batch_size} "
        f"fltrust_global_lr={args.fltrust_global_lr} "
        f"fltrust_eps={args.fltrust_eps} "
        f"fltg_global_lr={args.fltg_global_lr} "
        f"fltg_eps={args.fltg_eps} "
        f"fedgreed_batch_size={args.fedgreed_batch_size} "
        f"fedgreed_max_clients={args.fedgreed_max_clients} "
        f"lasa_sparsity_ratio={args.lasa_sparsity_ratio} "
        f"lasa_lambda_m={args.lasa_lambda_m} "
        f"lasa_lambda_d={args.lasa_lambda_d} "
        f"lasa_eps={args.lasa_eps}"
    )
    if str(args.mode).strip().lower() == "fedcor":
        logging.info(
            f"FedCor | warmup={args.fedcor_warmup} interval={args.fedcor_gpr_interval} "
            f"gamma={args.fedcor_gpr_gamma} group_size={args.fedcor_group_size} "
            f"gpr_epochs={args.fedcor_gpr_train_epochs} dim={args.fedcor_dimension} "
            f"kernel={args.fedcor_kernel} eps={args.fedcor_epsilon_greedy} "
            f"discount={args.fedcor_discount} update_mean={int(args.fedcor_update_mean)} "
            f"verbose={int(args.fedcor_verbose)} train_method={args.fedcor_train_method}"
        )
    if str(args.mode).strip().lower() == "fedcs":
        logging.info(
            f"FedCS | client_selection={args.fedcs_client_selection} "
            f"pruning_rate_f={args.fedcs_pruning_rate_f} "
            f"pruning_rate_l={args.fedcs_pruning_rate_l} "
            f"beta={args.fedcs_beta} "
            f"select_interval={args.fedcs_select_interval} "
            f"min_keep_samples={args.fedcs_min_keep_samples}"
        )
    if str(args.mode).strip().lower() == "fedcorr":
        logging.info(
            f"FedCorrOfficial | iteration1={args.fedcorr_iteration1} "
            f"rounds1={args.fedcorr_rounds1} rounds2={args.fedcorr_rounds2} "
            f"frac1={args.fedcorr_frac1} frac2={args.fedcorr_frac2} "
            f"beta={args.fedcorr_beta} correction={int(args.fedcorr_correction)} "
            f"fine_tuning={int(args.fedcorr_fine_tuning)} "
            f"relabel_ratio={args.fedcorr_relabel_ratio} "
            f"confidence_thres={args.fedcorr_confidence_thres} "
            f"clean_set_thres={args.fedcorr_clean_set_thres} "
            f"lid_k={args.fedcorr_lid_k}"
        )
    if str(args.mode).strip().lower() == "fedned":
        logging.info(
            f"FedNed | warmup_rounds={args.fedned_warmup_rounds} "
            f"lambda={args.fedned_lambda} "
            f"temperature={args.fedned_temperature} "
            f"pseudo_threshold={args.fedned_pseudo_threshold} "
            f"distill_batch_size={args.fedned_distill_batch_size} "
            f"distill_steps={args.fedned_distill_steps} "
            f"distill_lr={args.fedned_distill_lr} "
            f"distill_weight_decay={args.fedned_distill_weight_decay} "
            f"uncertainty_mc_passes={args.fedned_uncertainty_mc_passes} "
            f"uncertainty_batches={args.fedned_uncertainty_batches} "
            f"use_pseudo_for_noisy={int(args.fedned_use_pseudo_for_noisy)}"
        )
    set_seed(args.seed)
    DEVICE = torch.device(f"{args.device}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {DEVICE}")
    args.device = DEVICE
    model = get_model_class(args.model)
    args.model = model
    # 根据数据集更新类别数
    in_channels, num_classes = get_dataset_arg(args.data_name)
    args.in_channels = in_channels
    args.num_classes = num_classes
    datasets, testdatasets, test_dataset1, test_dataset2 = prepare_data(args)
    noise_rates = get_noise_rate(args)
    datasets, noise_group = get_noise_datasets(args, datasets, noise_rates)
    args.noise_group = noise_group
    train_dataloaders, test_dataloaders = get_loaders(args, datasets, testdatasets)
    args.client_args = get_client_args(args, datasets, train_dataloaders, test_dataloaders)
    if str(args.mode).strip().lower() == "fedcorr":
        from lib.model.fedcorr_official import train_fedcorr_official
        train_fedcorr_official(args)
    elif str(args.mode).strip().lower() == "fedned":
        train_script(args)
    else:
        train_script(args)
