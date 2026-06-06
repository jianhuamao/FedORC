from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from lib.model.client import Client
from lib.model.model import CNN, MLP, ResNet18, ResNet34, ResNet50, VGG


_ARG_SPECS = [
    ("seed", int, 2025),
    ("client_subset", str, None),
    ("data_name", str, "cifar10"),
    ("imbalance_ratio", int, 5),
    ("split_ratio", float, 0.1),
    ("alpha", float, 0.5),
    ("client_alpha_min", float, None),
    ("client_alpha_max", float, None),
    ("IS_NOISE", int, 0),
    ("noise_mode", str, "linear_noise"),
    ("noise_type", str, "symmetric"),
    ("noise_gaussian_mean", float, 0.3),
    ("noise_gaussian_std", float, 0.15),
    ("noise_gaussian_min", float, 0.0),
    ("noise_gaussian_max", float, 1.0),
    ("num_clients", int, 50),
    ("target_client_num", int, 20),
    ("num_local_epochs", int, 10),
    ("num_global_rounds", int, 200),
    ("batch_size", int, 64),
    ("client_fraction", float, 1.0),
    ("local_lr", float, 1e-4),
    ("client_epochs", int, 10),
    ("in_channels", int, 3),
    ("num_classes", int, 10),
    ("model", str, "ResNet18"),
    ("device", str, "cuda:1"),
    ("client_model_on_cpu", int, 1),
    ("use_amp", int, 1),
    ("cuda_empty_cache", int, 1),
    ("logdir", str, "./result/log/linear_noise/all/0.5/client_50"),
    ("logfile", str, "client_50_wo_noise.log"),
    ("project", str, "MS"),
    ("experiment_name", str, None),
    ("mode", str, "ours"),
    ("aggregation", str, "fedavg"),
    ("fedprox_mu", float, 0.0),
    ("fltrust_root_lr", float, 1e-3),
    ("fltrust_root_steps", int, 1),
    ("fltrust_root_batch_size", int, 0),
    ("fltrust_global_lr", float, 1.0),
    ("fltrust_eps", float, 1e-12),
    ("fltg_global_lr", float, 1.0),
    ("fltg_eps", float, 1e-12),
    ("fedgreed_batch_size", int, 0),
    ("fedgreed_max_clients", int, 0),
    ("lasa_sparsity_ratio", float, 0.2),
    ("lasa_lambda_m", float, 2.0),
    ("lasa_lambda_d", float, 2.0),
    ("lasa_eps", float, 1e-12),
    ("force_full_participation_50", int, 0),
    ("fedcor_warmup", int, 20),
    ("fedcor_gpr_interval", int, 10),
    ("fedcor_gpr_gamma", float, 0.95),
    ("fedcor_group_size", int, 100),
    ("fedcor_gpr_train_epochs", int, 100),
    ("fedcor_dimension", int, 10),
    ("fedcor_kernel", str, "Poly"),
    ("fedcor_epsilon_greedy", float, 0.0),
    ("fedcor_discount", float, 0.95),
    ("fedcor_update_mean", int, 1),
    ("fedcor_verbose", int, 0),
    ("fedcor_train_method", str, "MML"),
    ("fedcs_client_selection", str, "fixed"),
    ("fedcs_pruning_rate_f", float, 0.5),
    ("fedcs_pruning_rate_l", float, 0.1),
    ("fedcs_beta", float, 0.5),
    ("fedcs_select_interval", int, 2000),
    ("fedcs_min_keep_samples", int, 1),
    ("fednoro_warmup_rounds", int, 5),
    ("fednoro_detect_interval", int, 10),
    ("fednoro_gmm_trials", int, 9),
    ("fednoro_min_clean_clients", int, 1),
    ("fednoro_kd_weight", float, 0.5),
    ("fednoro_kd_temperature", float, 0.8),
    ("fedned_warmup_rounds", int, 10),
    ("fedned_lambda", float, 0.12),
    ("fedned_temperature", float, 2.0),
    ("fedned_pseudo_threshold", float, 0.95),
    ("fedned_distill_batch_size", int, 128),
    ("fedned_distill_steps", int, 100),
    ("fedned_distill_lr", float, 0.01),
    ("fedned_distill_weight_decay", float, 0.002),
    ("fedned_uncertainty_mc_passes", int, 10),
    ("fedned_uncertainty_batches", int, 1),
    ("fedned_use_pseudo_for_noisy", int, 1),
    ("fedcorr_iteration1", int, 5),
    ("fedcorr_rounds1", int, 100),
    ("fedcorr_rounds2", int, 95),
    ("fedcorr_frac1", float, 0.1),
    ("fedcorr_frac2", float, 0.4),
    ("fedcorr_beta", float, 5.0),
    ("fedcorr_correction", int, 1),
    ("fedcorr_fine_tuning", int, 1),
    ("fedcorr_relabel_ratio", float, 0.5),
    ("fedcorr_confidence_thres", float, 0.5),
    ("fedcorr_clean_set_thres", float, 0.1),
    ("fedcorr_lid_k", int, 20),
    ("fedcorr_stage2_all_clients", int, 0),
    ("ours_use_distribution", int, 1),
    ("ours_use_quality", int, 1),
    ("ours_distribution_strategy", str, "legacy_kl"),
    ("ours_distribution_source", str, "proxy_pred"),
    ("ours_distribution_power", float, 1.0),
    ("ours_quality_power", float, 1.0),
    ("ours_adaptive_balance", int, 1),
    ("ours_adaptive_strength", float, 1.0),
    ("ours_adaptive_mean_weight", float, 0.7),
    ("ours_adaptive_high_quantile", float, 0.75),
    ("ours_adaptive_a_min", float, 0.1),
    ("ours_adaptive_a_max", float, 4.0),
    ("ours_adaptive_b_min", float, 0.1),
    ("ours_adaptive_b_max", float, 4.0),
    ("ours_selection_count_mode", str, "theory_optimal"),
    ("ours_theory_opt_min_clients", int, 5),
    ("ours_fixed_scale_scoring", int, 1),
    ("ours_kl_temperature", float, -1.0),
    ("ours_kl_temperature_quantile", float, 0.5),
    ("ours_eta_min", float, 0.1),
    ("ours_eta_max", float, 0.3),
    ("ours_eta_class_scaling", int, 1),
    ("ours_eta_class_ref", float, 10.0),
    ("ours_eta_class_gamma", float, 0.18),
    ("ours_quality_class_scaling", int, 1),
    ("ours_quality_class_ref", float, 100.0),
    ("ours_quality_class_gamma", float, 0.35),
    ("ours_score_eps", float, 1e-8),
    ("ours_subsample_rate", float, 1.0),
    ("local_denoise_method", str, "none"),
    ("denoise_forget_rate", float, 0.2),
    ("denoise_num_gradual", int, 10),
    ("denoise_warmup_epochs", int, 1),
    ("denoise_lambda_u", float, 1.0),
    ("denoise_temperature", float, 0.5),
    ("denoise_mixup_alpha", float, 4.0),
    ("denoise_p_threshold", float, 0.5),
    ("denoise_rampup_length", int, 16),
    ("jocor_co_lambda", float, 0.1),
    ("fedfixer_lambda", float, 1.0),
    ("fedfixer_beta", float, 0.1),
    ("fedfixer_forget_rate", float, 0.2),
    ("fedfixer_num_gradual", int, 10),
    ("fedfixer_warmup_epochs", int, 1),
    ("feddiv_warmup_rounds", int, 10),
    ("feddiv_confidence_threshold", float, 0.5),
    ("feddiv_consistency_weight", float, 0.1),
    ("feddiv_gmm_max_iter", int, 10),
    ("enable_label_correction", int, 0),
    ("label_correction_method", str, "fedcorr_proxy"),
    ("label_correction_start_min_rounds", int, 40),
    ("label_correction_patience", int, 8),
    ("label_correction_min_delta", float, 0.05),
    ("label_correction_top_fraction", float, 0.3),
    ("label_correction_top_k", int, 0),
    ("label_correction_proxy_count", int, 3),
    ("label_correction_confidence_threshold", float, 0.7),
    ("label_correction_consensus_threshold", float, 0.67),
    ("label_correction_noisy_quantile", float, 0.7),
    ("label_correction_max_relabel_ratio", float, 0.4),
    ("label_correction_accept_conf", float, 0.45),
    ("label_correction_accept_entropy", float, 0.85),
    ("label_correction_accept_min_conf_gain", float, 0.01),
    ("label_correction_accept_min_entropy_drop", float, 0.01),
    ("label_correction_accept_join_patience", int, 2),
    ("label_correction_accept_evict_patience", int, 2),
    ("label_correction_accept_dynamic_quantile", float, 0.6),
    ("label_correction_accept_dynamic_margin", float, 0.03),
    ("label_correction_new_client_boost", float, 2.0),
    ("label_correction_boost_decay", float, 0.9),
]


_BOOL_FIELDS = {
    "IS_NOISE",
    "client_model_on_cpu",
    "use_amp",
    "cuda_empty_cache",
    "force_full_participation_50",
    "fedcor_update_mean",
    "fedcor_verbose",
    "fedcorr_correction",
    "fedcorr_fine_tuning",
    "fedcorr_stage2_all_clients",
    "fedned_use_pseudo_for_noisy",
    "ours_use_distribution",
    "ours_use_quality",
    "ours_adaptive_balance",
    "ours_fixed_scale_scoring",
    "ours_eta_class_scaling",
    "ours_quality_class_scaling",
    "enable_label_correction",
}


_DEFAULTS = {name: default for name, _, default in _ARG_SPECS}


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            object.__setattr__(cls._instance, "_initialized", False)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        object.__setattr__(self, "_callbacks", {})
        self._load_defaults()
        object.__setattr__(self, "_initialized", True)

    def _load_defaults(self) -> None:
        for name, value in _DEFAULTS.items():
            setattr(self, name, copy.deepcopy(value))

        self.noise_group = [-1]
        self.data_config = None
        self.client_config = None
        self.client_args = None
        self.cost_tracker = None
        self.train_subset_indices_map = None
        self.fednoro_noisy_clients = None
        self.feddiv_global_gmm = None
        self.client_subset = None
        self.fedned_use_pseudo_labels = False

    def __setattr__(self, name: str, value: Any):
        old_value = self.__dict__.get(name, None)
        object.__setattr__(self, name, value)
        if name.startswith("_") or not self.__dict__.get("_initialized", False):
            return
        callbacks = self.__dict__.get("_callbacks", {})
        if name in callbacks and old_value != value:
            for callback in list(callbacks[name]):
                callback(name, old_value, value)

    def register_callback(self, attr_name: str, callback: Callable):
        self._callbacks.setdefault(attr_name, []).append(callback)

    def unregister_callback(self, attr_name: str, callback: Callable):
        if attr_name in self._callbacks and callback in self._callbacks[attr_name]:
            self._callbacks[attr_name].remove(callback)

    def update_from_args(self, args: argparse.Namespace):
        for key, value in vars(args).items():
            if key in _BOOL_FIELDS and value is not None:
                value = bool(int(value))
            if hasattr(self, key):
                setattr(self, key, value)

    def get_model_class(self):
        return get_model_class(self.model)

    def get_device(self):
        return torch.device(self.device if torch.cuda.is_available() else "cpu")

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Config({attrs})"


@dataclass
class DataConfig:
    data_name: str
    datasets: List[Any]
    testdatasets: List[Any]
    test_dataset1: Any
    test_dataset2: Any
    train_dataset2: Any
    train_dataset1: Optional[Any] = None
    train_dataloaders: List[Any] = field(default_factory=list)
    test_dataloaders: List[Any] = field(default_factory=list)
    num_samples_per_client: List[int] = field(default_factory=list)
    client_alphas: List[float] = field(default_factory=list)
    in_channels: int = 3
    num_classes: int = 10


def _add_argument(parser: argparse.ArgumentParser, name: str, arg_type: type, default: Any):
    parser.add_argument(f"--{name}", type=arg_type, default=default)


def parse_args():
    parser = argparse.ArgumentParser(description="Federated learning experiment")
    for name, arg_type, default in _ARG_SPECS:
        _add_argument(parser, name, arg_type, default)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_class(model_name):
    model_dict = {
        "MLP": MLP,
        "VGG": VGG,
        "CNN": CNN,
        "ResNet18": ResNet18,
        "ResNet34": ResNet34,
        "ResNet50": ResNet50,
    }
    if model_name not in model_dict:
        raise ValueError(f"Unsupported model: {model_name}")
    return model_dict[model_name]


def get_client_args(args, datasets, train_dataloaders, test_dataloaders):
    num_epochs = int(getattr(args, "client_epochs", getattr(args, "num_local_epochs", 10)))
    return {
        "Client": Client,
        "model": args.model,
        "device": args.device,
        "train_dataloaders": train_dataloaders,
        "test_dataloaders": test_dataloaders,
        "datasets": datasets,
        "num_epochs": num_epochs,
        "lr": args.local_lr,
        "in_channels": args.in_channels,
        "num_classes": args.num_classes,
        "client_model_on_cpu": args.client_model_on_cpu,
        "use_amp": args.use_amp,
        "cuda_empty_cache": args.cuda_empty_cache,
        "local_denoise_method": args.local_denoise_method,
        "denoise_forget_rate": args.denoise_forget_rate,
        "denoise_num_gradual": args.denoise_num_gradual,
        "denoise_warmup_epochs": args.denoise_warmup_epochs,
        "denoise_lambda_u": args.denoise_lambda_u,
        "denoise_temperature": args.denoise_temperature,
        "denoise_mixup_alpha": args.denoise_mixup_alpha,
        "denoise_p_threshold": args.denoise_p_threshold,
        "denoise_rampup_length": args.denoise_rampup_length,
        "jocor_co_lambda": args.jocor_co_lambda,
        "fedfixer_lambda": args.fedfixer_lambda,
        "fedfixer_beta": args.fedfixer_beta,
        "fedfixer_forget_rate": args.fedfixer_forget_rate,
        "fedfixer_num_gradual": args.fedfixer_num_gradual,
        "fedfixer_warmup_epochs": args.fedfixer_warmup_epochs,
        "feddiv_warmup_rounds": args.feddiv_warmup_rounds,
        "feddiv_confidence_threshold": args.feddiv_confidence_threshold,
        "feddiv_consistency_weight": args.feddiv_consistency_weight,
        "feddiv_gmm_max_iter": args.feddiv_gmm_max_iter,
        "aggregation": args.aggregation,
        "fedprox_mu": args.fedprox_mu,
        "train_subset_indices_map": None,
        "fednoro_noisy_clients": None,
        "fednoro_kd_weight": args.fednoro_kd_weight,
        "fednoro_kd_temperature": args.fednoro_kd_temperature,
        "fedned_use_pseudo_labels": False,
        "fedned_pseudo_threshold": args.fedned_pseudo_threshold,
        "fedned_temperature": args.fedned_temperature,
    }


def init_swanlab(args):
    try:
        import swanlab
    except ImportError:
        return None

    model_name = getattr(args.model, "__name__", str(args.model))
    experiment_name = args.experiment_name or f"{model_name}_{args.data_name}_{args.mode}"
    return swanlab.init(
        project=args.project,
        experiment_name=experiment_name,
        config={
            "num_clients": args.num_clients,
            "num_local_epochs": args.num_local_epochs,
            "client_epochs": args.client_epochs,
            "batch_size": args.batch_size,
            "local_lr": args.local_lr,
        },
    )
