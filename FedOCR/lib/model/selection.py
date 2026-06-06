import torch
from lib.model.EMD import compute_label_distribution, emd_client_selection,emd_client_selection_minimize_subset_emd
from lib.model.fedcor_selector import FedCorClientSelector
from lib.model.server import Server
from lib.utils import get_client_instance, get_current_exp_name, fedocr_client_selection, random_client_selection, PoC_client_selection, single_training_loop
from lib.model.fedgap import FedOCRSelector
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import wasserstein_distance


def _cost_tracker(args):
    return getattr(args, "cost_tracker", None)


def _client_total_samples(args):
    counts = getattr(args.data_config, "num_samples_per_client", None)
    if counts is not None and len(counts) > 0:
        return int(sum(int(x) for x in counts))
    return int(sum(len(dataset) for dataset in args.data_config.datasets))


def _record_label_scan_cost(args, phase, scan_samples, aux_client_count=None):
    tracker = _cost_tracker(args)
    if tracker is None:
        return
    aux_clients = args.num_clients if aux_client_count is None else int(aux_client_count)
    tracker.record_aux_phase(
        phase=phase,
        sample_scans=int(scan_samples),
        aux_bytes=tracker.class_histogram_bytes(aux_clients),
    )


def _resolve_fedocr_ablation_flags(args, ablation_profile=None):
    use_distribution_score = bool(args.ours_use_distribution)
    use_quality_score = bool(args.ours_use_quality)
    if ablation_profile == 'distribution_only':
        use_distribution_score = True
        use_quality_score = False
    elif ablation_profile == 'quality_only':
        use_distribution_score = False
        use_quality_score = True

    if not (use_distribution_score or use_quality_score):
        raise ValueError(
            "Invalid ours ablation config: both distribution and quality scores are disabled."
        )
    return use_distribution_score, use_quality_score


def get_EMD_selection(args):
    data_config = args.data_config
    global_label_distribution = compute_label_distribution(
        torch.utils.data.DataLoader(data_config.train_dataset2, batch_size=args.batch_size, shuffle=True),
        num_classes=args.num_classes
    )
    print(f"全局标签分布：{global_label_distribution}")
    print(f"\n使用 EMD 选择 {args.target_client_num} 个客户端 ")
    selected_samples = emd_client_selection(
        client_dataloaders=data_config.train_dataloaders,
        target_client_num=args.target_client_num,
        global_distribution=global_label_distribution,
        num_classes=args.num_classes,
        selection_strategy= 'min_emd'
    )
    print(f"选中的客户端索引：{selected_samples}")
    
    from lib.model.EMD import compute_emd_matrix
    emd_distances = compute_emd_matrix(data_config.train_dataloaders, global_label_distribution, args.num_classes)
    print(f"\n选中客户端的 EMD 距离:")
    for idx in selected_samples:
        print(f"  Client {idx}: EMD = {emd_distances[idx]:.4f}")
    _record_label_scan_cost(
        args,
        phase="emd_static_label_distribution",
        scan_samples=len(data_config.train_dataset2) + 2 * _client_total_samples(args),
    )
    return selected_samples
    
def get_GS_selection(args):
    data_config = args.data_config   
    global_label_distribution = compute_label_distribution(
        torch.utils.data.DataLoader(data_config.train_dataset2, batch_size=args.batch_size, shuffle=True),
        num_classes=args.num_classes
    )
    selected_samples = emd_client_selection_minimize_subset_emd(
    client_dataloaders=data_config.train_dataloaders,
    target_client_num=args.target_client_num,
    global_distribution=global_label_distribution,
    num_classes=args.num_classes
)
    _record_label_scan_cost(
        args,
        phase="gs_static_label_distribution",
        scan_samples=len(data_config.train_dataset2) + _client_total_samples(args),
    )
    return selected_samples


def _compute_client_distributions_and_counts(args):
    client_distributions = []
    client_sample_counts = []
    for dataloader in args.data_config.train_dataloaders:
        labels = []
        for _, label in dataloader:
            labels.extend(label.cpu().numpy())
        dist = np.bincount(labels, minlength=args.num_classes).astype(np.float64)
        total = dist.sum()
        if total > 0:
            dist = dist / total
        client_distributions.append(dist)
        client_sample_counts.append(len(labels))
    return np.asarray(client_distributions), np.asarray(client_sample_counts, dtype=np.float64)


def _subset_aggregated_distribution(subset, client_distributions, client_sample_counts, num_classes):
    if len(subset) == 0:
        return np.zeros(num_classes, dtype=np.float64)
    total_samples = float(np.sum(client_sample_counts[subset]))
    if total_samples <= 0:
        return np.mean(client_distributions[subset], axis=0)
    agg = np.zeros(num_classes, dtype=np.float64)
    for idx in subset:
        w = float(client_sample_counts[idx]) / total_samples
        agg += w * client_distributions[idx]
    return agg


def _subset_emd_to_proxy(subset, client_distributions, client_sample_counts, proxy_distribution, num_classes):
    agg = _subset_aggregated_distribution(
        subset=subset,
        client_distributions=client_distributions,
        client_sample_counts=client_sample_counts,
        num_classes=num_classes,
    )
    return float(
        wasserstein_distance(
            u_values=np.arange(num_classes),
            v_values=np.arange(num_classes),
            u_weights=agg,
            v_weights=proxy_distribution,
        )
    )


def build_gs_subset_pool(
    args,
    num_subsets=8,
    candidate_trials=64,
    beam_width=6,
):
    """
    Build a diverse pool of GS subsets (for per=1) such that every subset's
    aggregated label distribution is close to proxy distribution.
    """
    proxy_distribution = compute_label_distribution(
        DataLoader(args.data_config.test_dataset1, batch_size=args.batch_size, shuffle=True),
        num_classes=args.num_classes,
    ).astype(np.float64)
    client_distributions, client_sample_counts = _compute_client_distributions_and_counts(args)
    _record_label_scan_cost(
        args,
        phase="gs_per_subset_pool",
        scan_samples=len(args.data_config.test_dataset1) + _client_total_samples(args),
    )
    num_clients = args.num_clients
    target_k = args.target_client_num
    if target_k >= num_clients:
        return [list(range(num_clients))]

    # Keep best unique subsets across randomized-greedy trials.
    best = []
    seen = set()
    for _ in range(max(int(candidate_trials), int(num_subsets))):
        selected = []
        remaining = list(range(num_clients))
        for _step in range(target_k):
            candidate_scores = []
            for cid in remaining:
                score = _subset_emd_to_proxy(
                    subset=selected + [cid],
                    client_distributions=client_distributions,
                    client_sample_counts=client_sample_counts,
                    proxy_distribution=proxy_distribution,
                    num_classes=args.num_classes,
                )
                candidate_scores.append((score, cid))
            candidate_scores.sort(key=lambda x: x[0])
            top = candidate_scores[: max(1, min(len(candidate_scores), int(beam_width)))]
            pick_idx = np.random.randint(len(top))
            chosen_cid = int(top[pick_idx][1])
            selected.append(chosen_cid)
            remaining.remove(chosen_cid)

        selected_sorted = tuple(sorted(selected))
        if selected_sorted in seen:
            continue
        seen.add(selected_sorted)
        final_score = _subset_emd_to_proxy(
            subset=list(selected_sorted),
            client_distributions=client_distributions,
            client_sample_counts=client_sample_counts,
            proxy_distribution=proxy_distribution,
            num_classes=args.num_classes,
        )
        best.append((final_score, list(selected_sorted)))

    if not best:
        # Conservative fallback
        one = get_GS_selection(args)
        return [sorted(one)]

    best.sort(key=lambda x: x[0])
    pool = [subset for _score, subset in best[: max(1, int(num_subsets))]]
    print(f"[GS per=1] built {len(pool)} subsets, best EMD={best[0][0]:.6f}")
    return pool


def get_gs_selection_from_pool(subset_pool, round_idx):
    if not subset_pool:
        return []
    return list(subset_pool[int(round_idx) % len(subset_pool)])

def get_PoC_selection(args):
    global_model = args.model(in_channels=args.in_channels, num_classes=args.num_classes).to(args.device)
    server = Server(global_model, args.num_clients, args.device)
    clients = [get_client_instance(client_idx=idx, global_state_dict=server.global_model.state_dict(), round_num=0, **args.client_args) for idx in range(args.num_clients)]
    selected_samples = PoC_client_selection(
        clients,
        TARGET_CLIENT_NUM=args.target_client_num,
        cost_tracker=_cost_tracker(args),
    )
    return selected_samples

def get_PoC_each_selection(clients, TARGET_CLIENT_NUM):
    loss_list = []
    id_list = []
    client_models_state_dicts = []
    infos = []
    for client in clients:
        updated_weights, _, info = client.train() 
        infos.append(info)
        client_models_state_dicts.append(updated_weights)
        client_idx = client.client_id
        id_list.append(client_idx)
        loss_train = info['loss'] if info else 0.0
        loss_list.append(loss_train)
        print(f'PoC: client{client_idx}, loss: {loss_train}')
    poc_selected_client_indices  = np.argsort(loss_list)[-TARGET_CLIENT_NUM:]    
    return [id_list[i] for i in poc_selected_client_indices], client_models_state_dicts, infos

def get_random_selection(args):
    selected_samples = random_client_selection(args.num_clients, args.target_client_num)
    return selected_samples


def get_fedocr_selection(args, ablation_profile=None):
    data_config = args.data_config
    use_distribution_score, use_quality_score = _resolve_fedocr_ablation_flags(
        args, ablation_profile=ablation_profile
    )
    print(
        f"[FedOCR] Selection config -> use_distribution_score={use_distribution_score}, "
        f"use_quality_score={use_quality_score}, "
        f"distribution_strategy={args.ours_distribution_strategy}, "
        f"distribution_source={args.ours_distribution_source}, "
        f"a={args.ours_distribution_power}, b={args.ours_quality_power}, "
        f"adaptive={int(args.ours_adaptive_balance)}, "
        f"adaptive_lambda={args.ours_adaptive_strength}, "
        f"count_mode={args.ours_selection_count_mode}, "
        f"theory_min_clients={args.ours_theory_opt_min_clients}"
    )
    global_model = args.model(in_channels=args.in_channels, num_classes=args.num_classes).to(args.device)
    server = Server(global_model, args.num_clients, args.device)
    proxy_dataloader = DataLoader(data_config.test_dataset1, args.batch_size, shuffle=True)
    selector = FedOCRSelector(
            num_classes=args.num_classes,
            server_model=server.global_model,
            proxy_data_loader=proxy_dataloader,
            device=args.device,
            use_distribution_score=use_distribution_score,
            use_quality_score=use_quality_score,
            distribution_score_strategy=args.ours_distribution_strategy,
            distribution_source=args.ours_distribution_source,
            default_client_alpha=args.alpha,
            distribution_power=args.ours_distribution_power,
            quality_power=args.ours_quality_power,
            adaptive_balance_enabled=args.ours_adaptive_balance,
            adaptive_strength=args.ours_adaptive_strength,
            adaptive_mean_weight=args.ours_adaptive_mean_weight,
            adaptive_high_quantile=args.ours_adaptive_high_quantile,
            adaptive_a_min=args.ours_adaptive_a_min,
            adaptive_a_max=args.ours_adaptive_a_max,
            adaptive_b_min=args.ours_adaptive_b_min,
            adaptive_b_max=args.ours_adaptive_b_max,
            selection_count_mode=args.ours_selection_count_mode,
            theory_opt_min_clients=args.ours_theory_opt_min_clients,
            fixed_scale_scoring=args.ours_fixed_scale_scoring,
            kl_temperature=args.ours_kl_temperature,
            kl_temperature_quantile=args.ours_kl_temperature_quantile,
            eta_min=args.ours_eta_min,
            eta_max=args.ours_eta_max,
            eta_class_scaling_enabled=args.ours_eta_class_scaling,
            eta_class_ref=args.ours_eta_class_ref,
            eta_class_gamma=args.ours_eta_class_gamma,
            quality_class_scaling_enabled=args.ours_quality_class_scaling,
            quality_class_ref=args.ours_quality_class_ref,
            quality_class_gamma=args.ours_quality_class_gamma,
            score_eps=args.ours_score_eps,
            min_samples_threshold=(
                50 if str(getattr(args, 'data_name', '')).lower() == 'bloodmnist' else 160
            ),
    )
    selector.cost_tracker = _cost_tracker(args)

    # 仅在筛选审计阶段关闭本地抗噪，避免把训练阶段 denoise 副作用带入客户端选择
    selection_client_args = dict(args.client_args)
    selection_client_args['local_denoise_method'] = 'none'
    clients = [
        get_client_instance(
            client_idx=idx,
            global_state_dict=server.global_model.state_dict(),
            round_num=0,
            **selection_client_args,
        )
        for idx in range(args.num_clients)
    ]
    selected_samples = fedocr_client_selection(
        client_candidates=clients,
        selector=selector,
        noise_group=args.noise_group,
        TARGET_CLIENT_NUM=args.target_client_num,
        device=args.device,
        num_classes=args.num_classes,
        client_alphas=data_config.client_alphas,
        cost_tracker=_cost_tracker(args),
    )
    return selected_samples


# Backward-compatible aliases for older names.
_resolve_ours_ablation_flags = _resolve_fedocr_ablation_flags
get_ours_selection = get_fedocr_selection

def get_fixed_selection(args):
    return list(range(args.target_client_num))


def get_fedcor_selection(args):
    """
    Static FedCor selection:
    run FedCor selector once before training and keep selected clients fixed.
    """
    selector = FedCorClientSelector(
        num_users=args.num_clients,
        target_client_num=args.target_client_num,
        device=args.device,
        warmup_rounds=args.fedcor_warmup,
        gpr_interval=args.fedcor_gpr_interval,
        gpr_gamma=args.fedcor_gpr_gamma,
        group_size=args.fedcor_group_size,
        gpr_train_epochs=args.fedcor_gpr_train_epochs,
        gpr_dimension=args.fedcor_dimension,
        gpr_kernel=args.fedcor_kernel,
        epsilon_greedy=args.fedcor_epsilon_greedy,
        discount=args.fedcor_discount,
        update_mean=args.fedcor_update_mean,
        verbose=args.fedcor_verbose,
        train_method=args.fedcor_train_method,
    )
    # Force one-shot FedCor policy instead of warmup-random behavior.
    selected_samples = selector.select_clients(round_idx=max(int(args.fedcor_warmup), 0))
    return selected_samples


def build_fedcor_selector(args):
    return FedCorClientSelector(
        num_users=args.num_clients,
        target_client_num=args.target_client_num,
        device=args.device,
        warmup_rounds=args.fedcor_warmup,
        gpr_interval=args.fedcor_gpr_interval,
        gpr_gamma=args.fedcor_gpr_gamma,
        group_size=args.fedcor_group_size,
        gpr_train_epochs=args.fedcor_gpr_train_epochs,
        gpr_dimension=args.fedcor_dimension,
        gpr_kernel=args.fedcor_kernel,
        epsilon_greedy=args.fedcor_epsilon_greedy,
        discount=args.fedcor_discount,
        update_mean=args.fedcor_update_mean,
        verbose=args.fedcor_verbose,
        train_method=args.fedcor_train_method,
    )


def get_fedcs_selection(args):
    selection_mode = str(getattr(args, "fedcs_client_selection", "fixed")).strip().lower()
    if selection_mode == "random":
        return get_random_selection(args)
    if selection_mode == "fixed":
        return list(range(args.num_clients))
    return get_fixed_selection(args)


def get_selected_samples(args):
    mode_lower = str(args.mode).strip().lower()
    if args.mode == 'PoC':
        selected_samples = get_PoC_selection(args)
    elif args.mode == 'EMD':
        selected_samples = get_EMD_selection(args)
    elif args.mode == 'ours':
        selected_samples = get_fedocr_selection(args)
    elif args.mode in {'ours_distribution_only', 'ours_wo_quality'}:
        selected_samples = get_fedocr_selection(args, ablation_profile='distribution_only')
    elif args.mode in {'ours_quality_only', 'ours_wo_distribution'}:
        selected_samples = get_fedocr_selection(args, ablation_profile='quality_only')
    elif args.mode in {'RS', 'random'}:
        selected_samples = get_random_selection(args)
    elif args.mode == 'GS':
        selected_samples = get_GS_selection(args)
    elif mode_lower == 'fedcor':
        selected_samples = get_fedcor_selection(args)
    elif mode_lower == 'fedcs':
        selected_samples = get_fedcs_selection(args)
    elif mode_lower == 'fednoro':
        selected_samples = list(range(args.num_clients))
    elif mode_lower == 'fedned':
        selected_samples = list(range(args.num_clients))
    elif mode_lower == 'fedfixer':
        selected_samples = list(range(args.num_clients))
    elif mode_lower == 'feddiv':
        selected_samples = list(range(args.num_clients))
    elif mode_lower == 'fltrust':
        selected_samples = list(range(args.num_clients))
    elif mode_lower == 'fltg':
        selected_samples = list(range(args.num_clients))
    elif mode_lower == 'fedgreed':
        selected_samples = list(range(args.num_clients))
    elif mode_lower == 'lasa':
        selected_samples = list(range(args.num_clients))
    elif mode_lower == 'fedcorr':
        selected_samples = []
    elif args.mode == 'all':
        selected_samples = list(range(args.num_clients))
    elif args.mode == 'fixed':
        selected_samples = get_fixed_selection(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    return selected_samples
