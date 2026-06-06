import copy
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.cost import loader_sample_count

class FedOCRSelector:
    SUPPORTED_DISTRIBUTION_STRATEGIES = {"legacy_kl", "dirichlet_D"}
    SUPPORTED_DISTRIBUTION_SOURCES = {"local_label", "proxy_pred"}
    SUPPORTED_SELECTION_COUNT_MODES = {"fixed", "theory_optimal"}

    def __init__(
        self,
        server_model,
        proxy_data_loader,
        device='cuda',
        num_classes=10,
        use_distribution_score=True,
        use_quality_score=True,
        distribution_score_strategy='legacy_kl',
        distribution_source='local_label',
        default_client_alpha=0.5,
        distribution_power=1.0,
        quality_power=1.0,
        adaptive_balance_enabled=False,
        adaptive_strength=1.0,
        adaptive_mean_weight=0.7,
        adaptive_high_quantile=0.75,
        adaptive_a_min=0.1,
        adaptive_a_max=4.0,
        adaptive_b_min=0.1,
        adaptive_b_max=4.0,
        selection_count_mode='fixed',
        theory_opt_min_clients=1,
        fixed_scale_scoring=True,
        kl_temperature=-1.0,
        kl_temperature_quantile=0.5,
        eta_min=0.20,
        eta_max=0.50,
        eta_class_scaling_enabled=True,
        eta_class_ref=10.0,
        eta_class_gamma=0.18,
        quality_class_scaling_enabled=True,
        quality_class_ref=100.0,
        quality_class_gamma=0.35,
        score_eps=1e-8,
        min_samples_threshold=160,
    ):
        self.server_model = server_model
        self.proxy_loader = proxy_data_loader
        self.device = device
        self.num_classes = num_classes
        self.use_distribution_score = use_distribution_score
        self.use_quality_score = use_quality_score
        self.distribution_score_strategy = distribution_score_strategy
        self.distribution_source = distribution_source
        self.default_client_alpha = float(default_client_alpha)
        self.distribution_power = float(distribution_power)
        self.quality_power = float(quality_power)
        self.adaptive_balance_enabled = bool(adaptive_balance_enabled)
        self.adaptive_strength = max(float(adaptive_strength), 0.0)
        self.adaptive_mean_weight = float(np.clip(adaptive_mean_weight, 0.0, 1.0))
        self.adaptive_high_quantile = float(np.clip(adaptive_high_quantile, 0.0, 1.0))
        self.adaptive_a_min = max(float(adaptive_a_min), 0.0)
        self.adaptive_a_max = max(float(adaptive_a_max), 0.0)
        self.adaptive_b_min = max(float(adaptive_b_min), 0.0)
        self.adaptive_b_max = max(float(adaptive_b_max), 0.0)
        self.selection_count_mode = str(selection_count_mode).strip().lower()
        self.theory_opt_min_clients = max(int(theory_opt_min_clients), 1)
        self.fixed_scale_scoring = bool(fixed_scale_scoring)
        self.kl_temperature = float(kl_temperature)
        self.kl_temperature_quantile = float(np.clip(kl_temperature_quantile, 0.0, 1.0))
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.eta_class_scaling_enabled = bool(eta_class_scaling_enabled)
        self.eta_class_ref = max(float(eta_class_ref), 1.0)
        self.eta_class_gamma = max(float(eta_class_gamma), 0.0)
        self.quality_class_scaling_enabled = bool(quality_class_scaling_enabled)
        self.quality_class_ref = max(float(quality_class_ref), 1.0)
        self.quality_class_gamma = max(float(quality_class_gamma), 0.0)
        self.score_eps = max(float(score_eps), 1e-12)
        if self.adaptive_a_min > self.adaptive_a_max:
            self.adaptive_a_min, self.adaptive_a_max = self.adaptive_a_max, self.adaptive_a_min
        if self.adaptive_b_min > self.adaptive_b_max:
            self.adaptive_b_min, self.adaptive_b_max = self.adaptive_b_max, self.adaptive_b_min
        if self.eta_min > self.eta_max:
            self.eta_min, self.eta_max = self.eta_max, self.eta_min
        if not (self.use_distribution_score or self.use_quality_score):
            raise ValueError("At least one of distribution score or quality score must be enabled.")
        if self.distribution_score_strategy not in self.SUPPORTED_DISTRIBUTION_STRATEGIES:
            raise ValueError(
                f"Unsupported distribution_score_strategy={self.distribution_score_strategy}. "
                f"Supported: {sorted(self.SUPPORTED_DISTRIBUTION_STRATEGIES)}"
            )
        if self.distribution_source not in self.SUPPORTED_DISTRIBUTION_SOURCES:
            raise ValueError(
                f"Unsupported distribution_source={self.distribution_source}. "
                f"Supported: {sorted(self.SUPPORTED_DISTRIBUTION_SOURCES)}"
            )
        if self.selection_count_mode not in self.SUPPORTED_SELECTION_COUNT_MODES:
            raise ValueError(
                f"Unsupported selection_count_mode={self.selection_count_mode}. "
                f"Supported: {sorted(self.SUPPORTED_SELECTION_COUNT_MODES)}"
            )
        if self.distribution_power < 0:
            raise ValueError("distribution_power must be >= 0.")
        if self.quality_power < 0:
            raise ValueError("quality_power must be >= 0.")
        if self.kl_temperature == 0:
            raise ValueError("kl_temperature cannot be 0. Use >0 for fixed tau or <0 for auto-calibration.")
        if self.eta_min < 0 or self.eta_max > 1:
            raise ValueError("eta_min/eta_max should be in [0, 1].")

                
        # Clients below this sample count are excluded from FedOCR selection.
        self.min_samples_threshold = max(int(min_samples_threshold), 0)
        self._cached_proxy_distribution = None
        self.cost_tracker = None
        self._cached_proxy_sample_count = None

    def _proxy_sample_count(self):
        if self._cached_proxy_sample_count is None:
            self._cached_proxy_sample_count = loader_sample_count(self.proxy_loader)
        return int(self._cached_proxy_sample_count)

    def _record_proxy_forward(self, phase, multiplier=1.0):
        if self.cost_tracker is None:
            return
        self.cost_tracker.add_selection_compute(
            compute_fep=float(multiplier) * float(self._proxy_sample_count())
        )

    def _evaluate_proxy_efficient(self, client_model_state_dict, max_batches=5):
        """ 修改版：支持输出各类别准确率的高效 Proxy 评估 """
        model = copy.deepcopy(self.server_model)
        model.load_state_dict(client_model_state_dict)
        model.to(self.device)
        model.eval()
        total_loss = 0.0
        count = 0
        criterion = nn.CrossEntropyLoss()
        num_classes = self.num_classes if hasattr(self, 'num_classes') else 10 
        class_correct = torch.zeros(num_classes).to(self.device)
        class_total = torch.zeros(num_classes).to(self.device)
        with torch.no_grad():
            for i, (data, target) in enumerate(self.proxy_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                count += data.size(0)
                
                # 计算各类别准确率
                pred = output.argmax(dim=1)
                correct_tensor = pred.eq(target)
                
                for label in range(num_classes):
                    # 找到当前 label 在 target 中的索引
                    label_mask = (target == label)
                    class_correct[label] += (correct_tensor & label_mask).sum().item()
                    class_total[label] += label_mask.sum().item()
        if self.cost_tracker is not None:
            # max_batches may truncate the loader; use the actually evaluated count.
            self.cost_tracker.add_selection_compute(compute_fep=float(count))
        overall_acc = 100 * class_correct.sum().item() / max(count, 1)
        avg_loss = total_loss / max(count, 1)
        per_class_acc = {}
        for i in range(num_classes):
            if class_total[i] > 0:
                per_class_acc[f"class_{i}"] = 100 * class_correct[i].item() / class_total[i].item()
            else:
                per_class_acc[f"class_{i}"] = 0.0  

        return overall_acc, avg_loss, per_class_acc

    def _calc_efficiency_score(self, n_k):
        """
        --- 修改点 3: 移除 Log,使用 Sqrt 或 Linear ---
        Log 函数会让 1000 样本和 100 样本的差距变小，导致大客户优势不明显。
        Sqrt 是一个很好的折中：比 Linear 平滑，但比 Log 激进。
        """
        return math.sqrt(n_k) 

    def _calc_kl_gain(self, p_current, p_candidate, n_current, n_candidate, p_target):
        """ (保持不变) 计算分布增益 """
        total_n = n_current + n_candidate
        if total_n == 0: return 0 
        p_new = (n_current * p_current + n_candidate * p_candidate) / total_n
        eps = 1e-9 
        def kl_divergence(p, q): return np.sum(p * np.log((p + eps) / (q + eps)))
        current_div = kl_divergence(p_target, p_current)
        new_div = kl_divergence(p_target, p_new)
        return current_div - new_div

    def _normalize(self, values):
        """ (保持不变) Z-Score 归一化 """
        arr = np.array(values)
        if len(arr) == 0: return arr
        std = np.std(arr)
        if std < 1e-9: return arr - np.mean(arr)
        return (arr - np.mean(arr)) / std
    
    def calculate_quality_contribution(self, scores, beta=2.0):
        scores = np.array(scores, dtype=float)
        min_score = np.min(scores)
        max_score = np.max(scores)

        # Reverse min-max to preserve "smaller score => larger contribution".
        if max_score == min_score:
            return np.ones_like(scores) / len(scores)

        return (max_score - scores) / (max_score - min_score)
    
    def calculate_distribution_contribution(self, scores, beta=3.0):
        # Keep the same normalization logic as quality contribution.
        return self.calculate_quality_contribution(scores, beta=beta)

    def _compute_adaptive_noise_profile(self, entropies):
        entropy_arr = np.asarray(entropies, dtype=float)
        if entropy_arr.size == 0:
            return np.array([], dtype=float), 0.0
        max_entropy = float(math.log(max(self.num_classes, 2)))
        if max_entropy <= 1e-12:
            h_i = np.zeros_like(entropy_arr, dtype=float)
        else:
            h_i = np.clip(entropy_arr / max_entropy, 0.0, 1.0)
        mean_h = float(np.mean(h_i))
        quantile_h = float(np.quantile(h_i, self.adaptive_high_quantile))
        s = self.adaptive_mean_weight * mean_h + (1.0 - self.adaptive_mean_weight) * quantile_h
        s = float(np.clip(s, 0.0, 1.0))
        return h_i, s

    def _resolve_effective_powers(self, s):
        if not self.adaptive_balance_enabled:
            return float(self.distribution_power), float(self.quality_power)
        g = float(np.clip(2.0 * (0.5 - float(s)), -1.0, 1.0))
        a_eff = self.distribution_power * (1.0 + self.adaptive_strength * g)
        b_eff = self.quality_power * (1.0 - self.adaptive_strength * g)
        a_eff = float(np.clip(a_eff, self.adaptive_a_min, self.adaptive_a_max))
        b_eff = float(np.clip(b_eff, self.adaptive_b_min, self.adaptive_b_max))
        return a_eff, b_eff

    def _get_proxy_target_distribution(self):
        """Proxy-label distribution p_proxy used by KL representativeness."""
        if self._cached_proxy_distribution is not None:
            return self._cached_proxy_distribution

        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        scanned = 0
        for _images, labels in self.proxy_loader:
            if isinstance(labels, torch.Tensor):
                labels_cpu = labels.detach().cpu().long().reshape(-1)
            else:
                labels_cpu = torch.tensor(labels, dtype=torch.long).reshape(-1)
            if labels_cpu.numel() == 0:
                continue
            scanned += int(labels_cpu.numel())
            counts += torch.bincount(labels_cpu, minlength=self.num_classes).to(dtype=torch.float32)

        if self.cost_tracker is not None:
            self.cost_tracker.add_selection_compute(sample_scans=scanned)

        total = float(counts.sum().item())
        if total <= self.score_eps:
            target = torch.full((self.num_classes,), 1.0 / max(self.num_classes, 1), dtype=torch.float32)
        else:
            target = counts / (total + self.score_eps)
            target = torch.clamp(target, min=self.score_eps)
            target = target / torch.clamp(target.sum(), min=self.score_eps)

        self._cached_proxy_distribution = target
        return target

    def _compute_quality_scores(self, entropies):
        """Map entropy to reliability score in [0, 1]."""
        if len(entropies) == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        h_i, _s = self._compute_adaptive_noise_profile(entropies)
        if self.fixed_scale_scoring:
            score_m = np.clip(1.0 - h_i, self.score_eps, 1.0).astype(np.float32)
        else:
            score_m = self.calculate_quality_contribution(entropies).astype(np.float32)
            score_m = np.clip(score_m, self.score_eps, 1.0)
        return score_m, h_i

    def _distance_to_distribution_score(self, distance, tau):
        """Convert distance (smaller-better) to Cs in [0,1]."""
        if not np.isfinite(distance):
            return float(self.score_eps)
        score = math.exp(-max(float(tau), self.score_eps) * max(float(distance), 0.0))
        return float(np.clip(score, self.score_eps, 1.0))

    def _compute_singleton_distance(self, cand, target_distribution):
        if self.distribution_score_strategy == "dirichlet_D":
            return self._calc_dirichlet_d_score_for_set([cand])
        return self._calc_kl_score_after_add(
            current_P=torch.zeros(self.num_classes, dtype=torch.float32),
            candidate_p_k=cand['p_k'],
            target_distribution=target_distribution,
        )

    def _resolve_distribution_temperature(self, valid_candidates, target_distribution):
        """
        Resolve tau in Cs=exp(-tau*d). If kl_temperature>0 use fixed value;
        otherwise calibrate tau from candidate-pool singleton distances.
        """
        if self.kl_temperature > 0:
            return float(self.kl_temperature)

        distances = []
        for cand in valid_candidates:
            d = self._compute_singleton_distance(cand, target_distribution)
            if np.isfinite(d):
                distances.append(max(float(d), 0.0))

        if len(distances) == 0:
            return 1.0

        anchor = float(np.quantile(np.asarray(distances, dtype=np.float64), self.kl_temperature_quantile))
        if anchor <= self.score_eps:
            return 1.0
        return float(1.0 / (anchor + self.score_eps))

    def _resolve_eta_components(self, s):
        eta_base = float(np.clip(
            self.eta_min + (self.eta_max - self.eta_min) * float(s),
            self.score_eps,
            1.0,
        ))
        if not self.eta_class_scaling_enabled or self.eta_class_gamma <= 0:
            return eta_base, 1.0, eta_base

        class_ratio = max(float(self.num_classes) / self.eta_class_ref, 1.0)
        class_scale = float(class_ratio ** (-self.eta_class_gamma))
        eta_scaled = float(np.clip(eta_base * class_scale, self.score_eps, 1.0))
        return eta_base, class_scale, eta_scaled

    def _resolve_quality_power_components(self, b_eff_base):
        b_eff_base = max(float(b_eff_base), 0.0)
        if not self.quality_class_scaling_enabled or self.quality_class_gamma <= 0:
            return b_eff_base, 1.0, b_eff_base

        class_ratio = max(float(self.num_classes) / self.quality_class_ref, 1.0)
        class_scale = float(class_ratio ** (-self.quality_class_gamma))
        b_eff_scaled = float(max(b_eff_base * class_scale, 0.0))
        return b_eff_base, class_scale, b_eff_scaled

    def _choose_best_prefix_length(self, prefix_objective_history, min_clients, max_clients):
        """
        Pick the best greedy prefix after the full greedy order is built.

        The final prefix is constrained to [min_clients, max_clients]. In
        theory_optimal mode the caller passes the complete greedy sequence as
        the search horizon, so target_count_M is not used as an upper bound.
        """
        total_prefixes = len(prefix_objective_history)
        if total_prefixes <= 0:
            return 0, -float("inf")

        if max_clients is None or int(max_clients) <= 0:
            upper = total_prefixes
        else:
            upper = min(max(int(max_clients), 1), total_prefixes)
        lower = min(max(int(min_clients), 1), upper)

        best_count = lower
        best_objective = -float("inf")
        for count in range(lower, upper + 1):
            objective = float(prefix_objective_history[count - 1])
            if (objective > best_objective + 1e-12) or (
                abs(objective - best_objective) <= 1e-12 and count > best_count
            ):
                best_objective = objective
                best_count = count
        return best_count, best_objective

    def get_entropy(self,client_model_state_dict):
        model = copy.deepcopy(self.server_model)
        model.load_state_dict(client_model_state_dict)
        model.to(self.device)
        model.eval()
        total_entropy = 0.0
        sample_count = 0
        correct = 0
        with torch.no_grad():
            for images, labels in self.proxy_loader:
                images = images.to(self.device)
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                log_probs = torch.log(probs + 1e-8)
                entropy = -torch.sum(probs * log_probs, dim=1)
                total_entropy += entropy.sum().item()
                sample_count += images.size(0)

        avg_entropy = total_entropy / sample_count
        if self.cost_tracker is not None:
            self.cost_tracker.add_selection_compute(compute_fep=float(sample_count))
        return avg_entropy

    def get_proxy_pred_distribution(self, client_model_state_dict):
        """
        使用客户端上传模型在代理数据集上的预测类别统计作为分布向量。
        返回 shape=[num_classes] 的计数向量（float32, CPU）。
        """
        model = copy.deepcopy(self.server_model)
        model.load_state_dict(client_model_state_dict)
        model.to(self.device)
        model.eval()

        pred_counts = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for images, _ in self.proxy_loader:
                images = images.to(self.device)
                logits = model(images)
                pred_labels = torch.argmax(logits, dim=1)
                batch_counts = torch.bincount(
                    pred_labels.detach().cpu(),
                    minlength=self.num_classes,
                ).to(dtype=torch.float32, device=self.device)
                pred_counts += batch_counts

        if self.cost_tracker is not None:
            self.cost_tracker.add_selection_compute(compute_fep=float(pred_counts.sum().item()))

        pred_counts_cpu = pred_counts.detach().cpu()
        if pred_counts_cpu.sum().item() <= 0:
            pred_counts_cpu = torch.ones(self.num_classes, dtype=torch.float32)

        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            model.to("cpu")
            torch.cuda.empty_cache()
        del model
        return pred_counts_cpu

    def _calc_kl_score_after_add(self, current_P, candidate_p_k, target_distribution, eps=1e-8):
        """
        KL(target || new_selected_distribution), smaller is better.
        """
        new_counts = current_P + candidate_p_k
        total = float(new_counts.sum().item())
        if total <= eps:
            return float("inf")
        new_distribution = new_counts / (total + eps)
        kl_value = torch.sum(
            target_distribution * torch.log((target_distribution + eps) / (new_distribution + eps))
        )
        return float(kl_value.item())

    def _resolve_client_alpha(self, cand):
        alpha_i = cand.get('alpha_i', None)
        if alpha_i is None:
            return max(self.default_client_alpha, 1e-8)
        try:
            alpha_i = float(alpha_i)
        except (TypeError, ValueError):
            return max(self.default_client_alpha, 1e-8)
        if not np.isfinite(alpha_i) or alpha_i <= 0:
            return max(self.default_client_alpha, 1e-8)
        return alpha_i

    def _calc_dirichlet_d_score_for_set(self, selected_candidates, eps=1e-12):
        """
        D(S) = 1/2 * sqrt((K-1) * sum_{i in S} (w_i^2 / (K*alpha_i + 1)) )
        where w_i = n_i / sum_{j in S} n_j.
        Smaller is better.
        """
        if len(selected_candidates) == 0:
            return float("inf")

        total_samples = float(sum(max(float(cand['n_k']), 0.0) for cand in selected_candidates))
        if total_samples <= eps:
            return float("inf")

        K = float(max(self.num_classes, 1))
        weighted_term = 0.0
        for cand in selected_candidates:
            n_i = max(float(cand['n_k']), 0.0)
            w_i = n_i / (total_samples + eps)
            alpha_i = self._resolve_client_alpha(cand)
            denom = max(K * alpha_i + 1.0, eps)
            weighted_term += (w_i * w_i) / denom

        inner = max((K - 1.0) * weighted_term, 0.0)
        return 0.5 * math.sqrt(inner)

    def select_clients(self, candidates_info, target_count_M):
        """执行选择逻辑。"""

        requested_target = max(int(target_count_M), 0)
        valid_candidates = []
        for cand in candidates_info:
            if cand['n_k'] < self.min_samples_threshold:
                continue
            valid_candidates.append(cand)

        if requested_target <= 0:
            logging.warning(
                "target_count_M=%s is non-positive, return empty selection.",
                target_count_M,
            )
            return []

        if self.selection_count_mode == "fixed":
            if len(valid_candidates) < requested_target:
                print("Warning: Too few valid clients after thresholding, using all candidates.")
                valid_candidates = candidates_info
            max_select_count = min(requested_target, len(valid_candidates))
        else:
            if len(valid_candidates) == 0:
                print("Warning: No valid clients after thresholding, using all candidates.")
                valid_candidates = candidates_info
            max_select_count = len(valid_candidates)

        if len(valid_candidates) == 0 or max_select_count <= 0:
            logging.warning("No candidates available for selection.")
            return []

        # Distribution target:
        # fixed-scale -> proxy label distribution p_proxy
        # legacy -> keep old uniform target for backward comparability
        legacy_total_class_vector = None
        if self.distribution_score_strategy == "legacy_kl":
            if self.fixed_scale_scoring:
                target_distribution = self._get_proxy_target_distribution()
            else:
                legacy_total_class_vector = torch.zeros(self.num_classes, dtype=torch.float32)
                for cand in valid_candidates:
                    legacy_total_class_vector += cand['p_k']
                legacy_total_class_vector = torch.clamp(legacy_total_class_vector, min=self.score_eps)
                target_distribution = torch.full_like(
                    legacy_total_class_vector, 1.0 / max(self.num_classes, 1)
                )
        else:
            target_distribution = torch.full(
                (self.num_classes,), 1.0 / max(self.num_classes, 1), dtype=torch.float32
            )

        need_entropy_profile = (
            self.use_quality_score
            or self.adaptive_balance_enabled
            or self.selection_count_mode == "theory_optimal"
        )
        entropies = []
        if need_entropy_profile:
            for cand in valid_candidates:
                entropies.append(self.get_entropy(cand['params']))

        h_i = np.array([], dtype=float)
        s = 0.5
        if len(entropies) > 0:
            h_i, s = self._compute_adaptive_noise_profile(entropies)

        if self.use_quality_score:
            if self.fixed_scale_scoring:
                base_score_m = np.clip(1.0 - h_i, self.score_eps, 1.0).astype(np.float32)
            else:
                base_score_m = self.calculate_quality_contribution(entropies).astype(np.float32)
                base_score_m = np.clip(base_score_m, self.score_eps, 1.0)
        else:
            base_score_m = np.ones(len(valid_candidates), dtype=np.float32)

        for i, cand in enumerate(valid_candidates):
            cand['norm_score_m'] = float(base_score_m[i])

        a_eff, b_eff_base = self._resolve_effective_powers(s)
        b_eff_base, quality_class_scale, b_eff = (
            self._resolve_quality_power_components(b_eff_base)
            if self.fixed_scale_scoring
            else (float(b_eff_base), 1.0, float(b_eff_base))
        )
        eta_base, eta_class_scale, eta_s = (
            self._resolve_eta_components(s) if self.fixed_scale_scoring else (1.0, 1.0, 1.0)
        )
        tau = 1.0
        if self.use_distribution_score and self.fixed_scale_scoring:
            tau = self._resolve_distribution_temperature(valid_candidates, target_distribution)

        if self.adaptive_balance_enabled:
            hi_map = {
                int(valid_candidates[i]['id']): round(float(h_i[i]), 6)
                for i in range(min(len(valid_candidates), len(h_i)))
            }
            logging.info(
                "[AdaptiveBalance] enabled=1 s=%.6f a_eff=%.6f b_eff_base=%.6f b_eff=%.6f h_i=%s",
                s,
                a_eff,
                b_eff_base,
                b_eff,
                hi_map,
            )
            print(
                f"[AdaptiveBalance] s={s:.6f}, a_eff={a_eff:.6f}, "
                f"b_eff_base={b_eff_base:.6f}, b_eff={b_eff:.6f}"
            )
            print(f"[AdaptiveBalance] h_i={hi_map}")
        else:
            logging.info(
                "[AdaptiveBalance] enabled=0 a_eff=%.6f b_eff_base=%.6f b_eff=%.6f",
                a_eff,
                b_eff_base,
                b_eff,
            )
        logging.info(
            "[FixedScale] enabled=%d tau=%.6f eta_base=%.6f "
            "eta_class_scale=%.6f eta_s=%.6f eta_class_scaling=%d "
            "eta_class_ref=%.2f eta_class_gamma=%.4f "
            "b_eff_base=%.6f quality_class_scale=%.6f b_eff=%.6f "
            "quality_class_scaling=%d quality_class_ref=%.2f quality_class_gamma=%.4f "
            "score_eps=%.2e",
            int(self.fixed_scale_scoring),
            tau,
            eta_base,
            eta_class_scale,
            eta_s,
            int(self.eta_class_scaling_enabled),
            self.eta_class_ref,
            self.eta_class_gamma,
            b_eff_base,
            quality_class_scale,
            b_eff,
            int(self.quality_class_scaling_enabled),
            self.quality_class_ref,
            self.quality_class_gamma,
            self.score_eps,
        )

        selected_set = []
        selected_candidate_indices = []
        remaining_indices = list(range(len(valid_candidates)))
        prefix_objective_history = []
        selected_stage_records = []
        running_log_utility = 0.0

        current_P = torch.zeros(self.num_classes, dtype=torch.float32)
        for step in range(max_select_count):
            if not remaining_indices:
                break

            step_indices = []
            raw_distances = []
            for idx in remaining_indices:
                cand = valid_candidates[idx]
                step_indices.append(idx)

                if self.distribution_score_strategy == "dirichlet_D":
                    candidate_set = [valid_candidates[i] for i in selected_candidate_indices] + [cand]
                    dist_raw = self._calc_dirichlet_d_score_for_set(candidate_set)
                else:
                    if self.fixed_scale_scoring:
                        dist_raw = self._calc_kl_score_after_add(
                            current_P=current_P,
                            candidate_p_k=cand['p_k'],
                            target_distribution=target_distribution,
                        )
                    else:
                        # Legacy first-step heuristic.
                        if step == 0:
                            current_d = (current_P + cand['p_k']) / legacy_total_class_vector
                            dist_raw = -float(current_d.pow(2).sum().item())
                        else:
                            dist_raw = self._calc_kl_score_after_add(
                                current_P=current_P,
                                candidate_p_k=cand['p_k'],
                                target_distribution=target_distribution,
                            )
                raw_distances.append(dist_raw)

            if self.use_distribution_score:
                if self.fixed_scale_scoring:
                    norm_cs_scores = np.asarray(
                        [self._distance_to_distribution_score(d, tau) for d in raw_distances],
                        dtype=np.float32,
                    )
                else:
                    norm_cs_scores = self.calculate_distribution_contribution(raw_distances)
            else:
                norm_cs_scores = np.ones(len(step_indices), dtype=np.float32)

            best_utility = -float("inf")
            best_idx_in_remaining = -1
            best_log_u = -float("inf")
            best_raw_dist = float("nan")
            best_distribution_score = 1.0
            best_quality_score = 1.0
            for i, idx in enumerate(step_indices):
                cand = valid_candidates[idx]
                raw_dist = float(raw_distances[i]) if i < len(raw_distances) else float("nan")
                distribution_factor = float(norm_cs_scores[i]) if self.use_distribution_score else 1.0
                quality_factor = float(cand['norm_score_m']) if self.use_quality_score else 1.0
                utility = (distribution_factor ** a_eff) * (quality_factor ** b_eff)
                if self.fixed_scale_scoring:
                    log_u = math.log(max(utility, self.score_eps)) - math.log(max(eta_s, self.score_eps))
                else:
                    log_u = math.log(max(utility, self.score_eps))
                print(
                    f"client{cand['id']}, raw_dist: {raw_dist:.6f}, Cs: {distribution_factor:.6f}, "
                    f"Cm: {float(cand['norm_score_m']):.6f}, A: {utility:.6f}, logU: {log_u:.6f}"
                )
                logging.info(
                    "[FedOCR][CandidateUtility] step=%d client=%d raw_dist=%.6f "
                    "distribution_score=%.6f quality_score=%.6f A=%.6f logU=%.6f",
                    step + 1,
                    int(cand["id"]),
                    raw_dist,
                    distribution_factor,
                    quality_factor,
                    utility,
                    log_u,
                )
                if utility > best_utility:
                    best_utility = utility
                    best_idx_in_remaining = i
                    best_log_u = log_u
                    best_raw_dist = raw_dist
                    best_distribution_score = distribution_factor
                    best_quality_score = quality_factor

            if best_idx_in_remaining == -1:
                break

            real_idx = step_indices[best_idx_in_remaining]
            best_cand = valid_candidates[real_idx]
            selected_set.append(best_cand['id'])
            selected_candidate_indices.append(real_idx)
            remaining_indices.remove(real_idx)
            current_P += best_cand['p_k']

            if self.selection_count_mode == "theory_optimal":
                running_log_utility += float(best_log_u)
                prefix_objective_history.append(running_log_utility)
                selected_stage_records.append(
                    {
                        "step": len(selected_set),
                        "client": int(best_cand["id"]),
                        "raw_dist": float(best_raw_dist),
                        "distribution_score": float(best_distribution_score),
                        "quality_score": float(best_quality_score),
                        "A": float(best_utility),
                        "logU": float(best_log_u),
                        "cum_logU": float(running_log_utility),
                    }
                )

            print(
                f" ============ Select {best_cand['id']}: "
                f"N={best_cand['n_k']}, alpha={self._resolve_client_alpha(best_cand):.4f}, "
                f"strategy={self.distribution_score_strategy}, "
                f"a_eff={a_eff:.2f}, b_eff={b_eff:.2f}, "
                f"A={best_utility:.4f}, logU={best_log_u:.4f}, eta={eta_s:.4f}, "
                f"Cs={best_distribution_score:.4f}, Cm={best_quality_score:.4f}=============="
            )

        if self.selection_count_mode == "theory_optimal" and selected_set:
            greedy_count = len(selected_set)
            prefix_limit = greedy_count
            min_clients = min(max(int(self.theory_opt_min_clients), 1), prefix_limit)
            best_prefix_count, best_objective = self._choose_best_prefix_length(
                prefix_objective_history=prefix_objective_history,
                min_clients=min_clients,
                max_clients=prefix_limit,
            )
            selected_set = selected_set[:best_prefix_count]
            print(
                "[TheoryOptimal] "
                f"best_prefix_count={len(selected_set)}, objective_log={best_objective:.6f}, "
                f"min_clients={min_clients}, prefix_limit={prefix_limit}, greedy_count={greedy_count}"
            )
            logging.info(
                "[TheoryOptimal] best_prefix_count=%d objective_log=%.6f "
                "min_clients=%d prefix_limit=%d greedy_count=%d",
                len(selected_set),
                best_objective,
                min_clients,
                prefix_limit,
                greedy_count,
            )
            for record in selected_stage_records:
                in_final_prefix = int(record["step"]) <= len(selected_set)
                logging.info(
                    "[TheoryOptimal][StageUtility] step=%d client=%d A=%.6f "
                    "distribution_score=%.6f quality_score=%.6f raw_dist=%.6f "
                    "logU=%.6f cum_logU=%.6f in_final_prefix=%d",
                    int(record["step"]),
                    int(record["client"]),
                    float(record["A"]),
                    float(record["distribution_score"]),
                    float(record["quality_score"]),
                    float(record["raw_dist"]),
                    float(record["logU"]),
                    float(record["cum_logU"]),
                    int(in_final_prefix),
                )

        logging.info(
            "Selected Clients: %s | selected_count=%d | count_mode=%s",
            selected_set,
            len(selected_set),
            self.selection_count_mode,
        )
        return selected_set


class FedOCRSelectorPerClassEntropy(FedOCRSelector):
    """Use per-class predictive entropy vectors in client selection."""

    def normalize_entropy_across_clients_per_class(self, entropy_matrix, beta=2.0):
        """
        entropy_matrix: shape [num_clients, num_classes]
        Normalize on each class dimension across different clients.
        """
        shifted = entropy_matrix - np.min(entropy_matrix, axis=0, keepdims=True)
        unnormalized = np.exp(-beta * shifted)
        denom = np.sum(unnormalized, axis=0, keepdims=True)
        denom = np.clip(denom, a_min=1e-12, a_max=None)
        return unnormalized / denom

    def get_per_class_entropy(self, client_model_state_dict):
        model = copy.deepcopy(self.server_model)
        model.load_state_dict(client_model_state_dict)
        model.to(self.device)
        model.eval()

        entropy_sum = torch.zeros(self.num_classes, device=self.device)
        class_count = torch.zeros(self.num_classes, device=self.device)
        max_entropy = math.log(max(self.num_classes, 2))

        with torch.no_grad():
            for images, labels in self.proxy_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                log_probs = torch.log(probs + 1e-8)
                sample_entropy = -torch.sum(probs * log_probs, dim=1)

                entropy_sum.scatter_add_(0, labels, sample_entropy)
                class_count.scatter_add_(
                    0,
                    labels,
                    torch.ones_like(sample_entropy, dtype=entropy_sum.dtype),
                )

        if self.cost_tracker is not None:
            self.cost_tracker.add_selection_compute(compute_fep=float(class_count.sum().item()))

        class_entropy = torch.full_like(entropy_sum, fill_value=max_entropy)
        observed_mask = class_count > 0
        class_entropy[observed_mask] = entropy_sum[observed_mask] / class_count[observed_mask]
        return class_entropy.detach().cpu().numpy()

    def select_clients(self, candidates_info, target_count_M):
        valid_candidates = []
        for cand in candidates_info:
            if cand['n_k'] < self.min_samples_threshold:
                continue
            valid_candidates.append(cand)

        if len(valid_candidates) < target_count_M:
            print("Warning: Too few valid clients after thresholding, using all candidates.")
            valid_candidates = candidates_info

        if len(valid_candidates) == 0:
            return []

        per_client_entropy_vecs = []
        for cand in valid_candidates:
            per_class_entropy = self.get_per_class_entropy(cand['params'])
            per_client_entropy_vecs.append(per_class_entropy)

        entropy_matrix = np.stack(per_client_entropy_vecs, axis=0)
        quality_matrix = self.normalize_entropy_across_clients_per_class(entropy_matrix)

        corrected_vectors = []
        for i, cand in enumerate(valid_candidates):
            quality_vec = quality_matrix[i]

            quality_vec_t = torch.tensor(quality_vec, dtype=torch.float32)
            class_vec_t = cand['p_k'].to(dtype=torch.float32)

            # Element-wise product (Hadamard product) gives the corrected class vector.
            corrected_class_vec = quality_vec_t * class_vec_t
            cand['quality_score_vec'] = quality_vec_t
            cand['corrected_class_vec'] = corrected_class_vec
            corrected_vectors.append(corrected_class_vec)

        if len(corrected_vectors) == 0:
            return []

        total_corrected_class_vector = torch.stack(corrected_vectors, dim=0).sum(dim=0)
        total_corrected_class_vector = torch.clamp(total_corrected_class_vector, min=1e-8)

        selected_set = []
        remaining_indices = list(range(len(valid_candidates)))
        current_corrected_P = torch.zeros(self.num_classes, dtype=torch.float32)

        for _ in range(target_count_M):
            if len(remaining_indices) == 0:
                break

            step_gains_s = []
            step_indices = []
            for idx in remaining_indices:
                cand = valid_candidates[idx]
                current_d = (current_corrected_P + cand['corrected_class_vec']) / total_corrected_class_vector
                step_gains_s.append(current_d)
                step_indices.append(idx)

            best_utility = -float('inf')
            best_idx_in_remaining = -1

            for i, idx in enumerate(step_indices):
                if len(remaining_indices) == len(valid_candidates):
                    cs = step_gains_s[i].pow(2).sum()
                else:
                    cs = 1 - step_gains_s[i].var()

                utility = 10 * float(cs)
                if utility > best_utility:
                    best_utility = utility
                    best_idx_in_remaining = i

            if best_idx_in_remaining == -1:
                break

            real_idx = step_indices[best_idx_in_remaining]
            best_cand = valid_candidates[real_idx]
            selected_set.append(best_cand['id'])
            remaining_indices.remove(real_idx)
            current_corrected_P += best_cand['corrected_class_vec']

            print(
                f" ============ Select {best_cand['id']}: "
                f"N={best_cand['n_k']}, Util={best_utility:.2f}=============="
            )

        logging.info(f"Selected Clients: {selected_set}")
        return selected_set


# Backward-compatible aliases for older imports and scripts.
FedGAPSelector = FedOCRSelector
FedGAPSelectorPerClassEntropy = FedOCRSelectorPerClassEntropy
    
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from PIL import Image

# --- 辅助类：用于包装数据、自定义标签和Transform ---
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        """
        Args:
            data: 图像数据 (可以是 numpy, tensor, 或 list)
            targets: 标签
            transform: torchvision transforms
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # --- 修正部分开始 ---
        
        # 1. 如果是 PyTorch Tensor，先转成 Numpy
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        
        # 2. 如果是 Numpy 数组，转成 PIL Image
        # 这是因为 torchvision 的大多数增强操作（如 AutoAugment）和 ToTensor 都预期输入是 PIL 格式
        if isinstance(img, np.ndarray):
            # 处理 MNIST/FashionMNIST 等灰度图 (H, W) -> 需要指定 mode='L' 
            # 或者 CIFAR (H, W, C) -> mode='RGB'
            # Image.fromarray 通常能自动推断，但如果是单通道有时候需要注意
            try:
                img = Image.fromarray(img)
            except Exception:
                # 如果自动推断失败（例如某些单通道数据），尝试强制指定模式
                if len(img.shape) == 2:  # (H, W) -> 灰度
                    img = Image.fromarray(img, mode='L')
                else:
                    img = Image.fromarray(img, mode='RGB')

        # --- 修正部分结束 ---

        # 3. 应用 Transform (此时 img 必然是 PIL Image)
        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

def extract_data_and_targets(dataset, indices):
    """
    辅助函数：从 Subset 或 Dataset 中根据 indices 提取数据和标签
    注意：这里假设 dataset 是类似 CIFAR/MNIST 的内存数据集 (有 .data 和 .targets 属性)
    """
    # 找到最底层的 dataset
    original_dataset = dataset
    if isinstance(dataset, Subset):
        original_dataset = dataset.dataset
        # 如果是 Subset 嵌套 Subset，这里可能需要递归，简单起见假设只有一层或直接操作
    
    # 提取特定索引的数据
    # 注意：根据数据集类型，这里可能需要调整 (CIFAR 使用 .data, .targets)
    if hasattr(original_dataset, 'data'):
        x_data = original_dataset.data[indices]
    else:
        # Fallback: 如果没有 .data 属性，只能一个一个取 (慢)
        # x_data = [original_dataset[i][0] for i in indices]
        raise AttributeError("Dataset need .data attribute for efficient splitting")

    if hasattr(original_dataset, 'targets'):
        y_targets = np.array(original_dataset.targets)[indices]
    else:
        # Fallback
        # y_targets = [original_dataset[i][1] for i in indices]
        raise AttributeError("Dataset need .targets attribute")
        
    return x_data, y_targets

# --- 定义不同的 Transform ---
def get_transforms(mode='normal'):
    if mode == 'hard':
        # 强增强：自动增强 + 随机擦除
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), # 强增强
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5), # 随机擦除
        ])
    elif mode == 'test':
        # 测试/验证：仅标准化
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        # 普通增强
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

from lib.dataset.dataset import partition_dataset_by_dirichlet

import torch
import numpy as np
from torch.utils.data import Dataset, Subset

class AddGaussianNoise(object):
    """
    自定义 Transform:添加高斯噪声
    noise_level (std) 越大，噪声越强。
    """
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # 生成标准正态分布噪声并缩放
        noise = torch.randn(tensor.size()) * self.std + self.mean
        # 叠加并截断到 [0, 1] 防止越界
        return torch.clamp(tensor + noise, 0., 1.)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
    

# Morphology-guided class-dependent label noise mapping for BloodMNIST.
# Based on blood cell morphological similarity in 28x28 microscopic images.
#  0 Basophil              <-> 1 Eosinophil         (both granulocytes)
#  2 Erythroblast           -> 4 Lymphocyte          (round nucleated cells)
#  3 Immature granulocytes <-> 6 Neutrophil          (developmental lineage)
#  4 Lymphocyte            <-> 5 Monocyte            (mononuclear leukocytes)
#  7 Platelet               -> random non-self        (morphologically distinct)
BLOODMNIST_NOISE_MAP = {
    0: 1,   # Basophil -> Eosinophil
    1: 0,   # Eosinophil -> Basophil
    2: 4,   # Erythroblast -> Lymphocyte
    3: 6,   # Immature granulocytes -> Neutrophil
    4: 5,   # Lymphocyte -> Monocyte
    5: 4,   # Monocyte -> Lymphocyte
    6: 3,   # Neutrophil -> Immature granulocytes
    7: -1,  # Platelet -> random non-self (handled in noise generation)
}

BLOODMNIST_NON_PLATELET_CLASSES = [0, 1, 2, 3, 4, 5, 6]


class NoisyDatasetWrapper(Dataset):
    def __init__(self, original_subset, noise_rate=0.2, num_classes=10,
                 noise_type='symmetric', dataset_name=None):
        """
        Args:
            original_subset: 原始数据集或其子集
            noise_rate: 噪声比例 (0.0 <= noise_rate <= 1.0)
            num_classes: 数据集的总类别数
            noise_type: 'symmetric' | 'morphology_guided'
            dataset_name: 数据集名称，用于选择特定数据集的噪声映射
        """
        self.original_subset = original_subset
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.noise_type = noise_type
        self.dataset_name = dataset_name

        self.targets = np.array(self._extract_targets(original_subset))

        if noise_type == 'morphology_guided' and num_classes == 8:
            self.noisy_targets = self._generate_morphology_guided_noise()
        else:
            self.noisy_targets = self._generate_symmetric_noise()

        # 统计实际变动的标签比例
        actual_noise = np.sum(self.targets != self.noisy_targets) / len(self.targets)
        noise_label = "Morphology-guided" if (noise_type == 'morphology_guided' and num_classes == 8) else "Symmetric"
        print(f"-> {noise_label} noise applied: {noise_rate*100:.1f}% planned, "
              f"{actual_noise*100:.1f}% labels actually changed.")

    def _generate_symmetric_noise(self):
        """ 生成对称噪声的核心逻辑 """
        noisy_targets = self.targets.copy()
        n_samples = len(self.targets)

        n_noisy = int(n_samples * self.noise_rate)
        if n_noisy == 0:
            return noisy_targets
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        for idx in noisy_indices:
            old_label = self.targets[idx]
            offset = np.random.randint(1, self.num_classes)
            new_label = (old_label + offset) % self.num_classes
            noisy_targets[idx] = new_label
        return noisy_targets

    def _generate_morphology_guided_noise(self):
        """Morphology-guided class-dependent label noise for BloodMNIST.

        Uses the BLOODMNIST_NOISE_MAP to flip labels to morphologically
        similar classes. Platelet (class 7) flips to a random non-platelet
        class since platelets are morphologically distinct from all others.
        """
        assert self.num_classes == 8, \
            f"Morphology-guided noise only supports 8-class BloodMNIST, got {self.num_classes}"

        noisy_targets = self.targets.copy()
        n_samples = len(self.targets)

        n_noisy = int(n_samples * self.noise_rate)
        if n_noisy == 0:
            return noisy_targets
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

        for idx in noisy_indices:
            old_label = int(self.targets[idx])
            target_class = BLOODMNIST_NOISE_MAP[old_label]

            if target_class == -1:
                # Platelet (class 7): flip to random non-self class
                candidates = [c for c in range(self.num_classes) if c != old_label]
                new_label = np.random.choice(candidates)
            elif target_class == old_label:
                # Identity mapping — keep original (shouldn't happen in our map)
                new_label = old_label
            else:
                new_label = target_class

            noisy_targets[idx] = new_label

        return noisy_targets

    def _extract_targets(self, dataset):
        """辅助函数：提取标签，兼容嵌套 Subset 和 ImageFolder"""
        from torch.utils.data import Subset
        import torch
        
        # 递归处理嵌套 Subset
        if isinstance(dataset, Subset):
            parent_targets = self._extract_targets(dataset.dataset)
            indices = dataset.indices
            
            # 将 parent_targets 转换为 numpy 数组
            if isinstance(parent_targets, torch.Tensor):
                parent_targets = parent_targets.cpu().numpy()
            elif not isinstance(parent_targets, (list, np.ndarray)):
                parent_targets = list(parent_targets)
            
            # 将 indices 转换为整数列表
            if isinstance(indices, np.ndarray):
                indices = indices.tolist()
            elif isinstance(indices, torch.Tensor):
                indices = indices.cpu().tolist()
            elif len(indices) > 0 and isinstance(indices[0], torch.Tensor):
                indices = [i.cpu().item() if isinstance(i, torch.Tensor) else i for i in indices]
            elif len(indices) > 0 and isinstance(indices[0], (tuple, list)):
                indices = [i[0] if isinstance(i, (tuple, list)) else i for i in indices]
            
            return [parent_targets[i] for i in indices]
        
        # 处理有 targets 属性的数据集
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            if isinstance(targets, torch.Tensor):
                return targets.cpu().numpy().tolist()
            return list(targets)
        elif hasattr(dataset, 'labels'):
            targets = dataset.labels
            if isinstance(targets, torch.Tensor):
                return targets.cpu().numpy().tolist()
            return list(targets)
        elif hasattr(dataset, 'imgs'):
            return [img[1] for img in dataset.imgs]
        elif hasattr(dataset, 'samples'):
            return [sample[1] for sample in dataset.samples]
        
        # Fallback: 遍历获取标签
        targets = []
        for i in range(len(dataset)):
            try:
                item = dataset[i]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    targets.append(item[1])
                else:
                    targets.append(0)
            except Exception as e:
                print(f"Warning: Failed to extract target at index {i}: {e}")
                targets.append(0)
        
        return targets
    def __getitem__(self, index):
        img, _ = self.original_subset[index]
        target = int(self.noisy_targets[index])
        return img, target

    def __len__(self):
        return len(self.original_subset)
# class NoisyDatasetWrapper(Dataset):
#     def __init__(self, original_subset):
#         """
#         Args:
#             original_subset: 原本分配给该客户端的 Dataset 或 Subset
#         """
#         self.original_subset = original_subset
        
#         # 1. 提取原始标签
#         # 注意：我们需要高效地获取标签。
#         # 如果是 Subset，我们需要通过 indices 去找原始 dataset 的 targets
#         self.targets = self._extract_targets(original_subset)
        
#         # 2. 打乱标签 (制造噪声)
#         # 这里使用 copy 防止修改原始数据，虽然从逻辑上提取出来已经是新的 list/array 了
#         # self.noisy_targets = np.array(self.targets).copy()
#         # np.random.shuffle(self.noisy_targets)
#         num_classes = 10 # 假设有10类
#         self.noisy_targets = np.random.randint(0, num_classes, size=len(self.targets))
#         print(f"-> Label noise applied: {len(self.noisy_targets)} labels shuffled.")

#     def _get_noisy_targets(self):
#         return self.noisy_targets
#     def _extract_targets(self, dataset):
#         """ 辅助函数：尝试从 Subset 或 Dataset 中提取所有标签 """
#         # 情况 A: 如果是 Subset，递归找到最底层的 targets
#         if isinstance(dataset, Subset):
#             # 获取 subset 对应的索引
#             indices = dataset.indices
#             parent = dataset.dataset
            
#             # 如果 parent 也是 subset，递归处理（虽然通常只有一层）
#             # 这里简化处理：假设 parent 就是原始数据集 (CIFAR/MNIST)
#             if hasattr(parent, 'targets'):
#                 # CIFAR/MNIST style
#                 all_targets = np.array(parent.targets)
#                 return all_targets[indices]
#             elif hasattr(parent, 'labels'):
#                 all_targets = np.array(parent.labels)
#                 return all_targets[indices]
#             else:
#                 # 最慢的方法：遍历 dataset (Fallback)
#                 return [dataset[i][1] for i in range(len(dataset))]
        
#         # 情况 B: 普通 Dataset
#         elif hasattr(dataset, 'targets'):
#             return np.array(dataset.targets)
#         else:
#             return [dataset[i][1] for i in range(len(dataset))]

#     def __getitem__(self, index):
#         # 1. 从原始 subset 获取图片 (忽略原始标签)
#         img, _ = self.original_subset[index]
        
#         # 2. 获取我们自己生成的噪声标签
#         noisy_target = self.noisy_targets[index]
        
#         # 确保返回类型一致 (通常转为 tensor 或 int)
#         if isinstance(noisy_target, np.ndarray) or isinstance(noisy_target, np.generic):
#             noisy_target = noisy_target.item() # 转为 Python int
            
#         return img, noisy_target

#     def __len__(self):
#         return len(self.original_subset)
    


def get_targets(dataset):
    """
    递归提取 Dataset 中的真实标签，自动处理 Subset 和 Wrapper。
    """
    if hasattr(dataset, 'noisy_targets'):
        return np.array(dataset.noisy_targets)
        
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    
    # case 3: 如果是 Subset，这是最关键的逻辑
    # 我们需要先拿到父级的所有标签，然后根据当前 Subset 的 indices 进行切片
    if isinstance(dataset, torch.utils.data.Subset):
        parent_targets = get_targets(dataset.dataset)
        # 利用 numpy 的数组索引功能进行映射
        return parent_targets[dataset.indices]
        
    # case 4: 如果只是普通的 Wrapper (如没有 targets 属性的 Dataset)，继续向下递归
    if hasattr(dataset, 'dataset'):
        return get_targets(dataset.dataset)
        
    raise ValueError(f"无法从 {type(dataset)} 中提取标签，请检查数据集结构。")
