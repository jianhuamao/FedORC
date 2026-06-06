import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
def get_client_dataloaders(client_train_datasets, client_test_datasets, batch_size=32):
    client_train_loaders = [DataLoader(d, batch_size=batch_size, shuffle=True) for d in client_train_datasets]
    client_test_loaders = [DataLoader(d, batch_size=batch_size, shuffle=False) for d in client_test_datasets]
    return client_train_loaders, client_test_loaders

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Sequence, Union

# 自定义一个 Dataset 以便更好地处理 subsets
class CustomSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)



def _apply_partition_logic(
    dataset: Dataset,
    num_clients: int,
    class_distribution: np.ndarray,
    seed: int = None
) -> Dict[int, List[int]]:
    """
    [辅助函数] 将预先计算好的狄利克雷分布应用到给定的数据集上。

    参数:
    - dataset (Dataset): 待划分的数据集。
    - num_clients (int): 客户端数量。
    - class_distribution (np.ndarray): 预先计算好的分布矩阵，shape (num_classes, num_clients)。
    - seed (int): 随机种子，用于控制 shuffle 的随机性。

    返回:
    - Dict[int, List[int]]: 客户端 ID 到数据索引列表的字典。
    """
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
    
    if isinstance(dataset, Subset):
        try:
            labels = torch.tensor(dataset.dataset.targets)[dataset.indices]
        except AttributeError:
            labels = torch.tensor(dataset.dataset.labels)[dataset.indices]
    else:
        try:
            labels = torch.tensor(dataset.targets)
        except AttributeError:
            labels = torch.tensor(dataset.labels)

    num_classes = len(torch.unique(labels))
    class_indices = [torch.where(labels == i)[0].tolist() for i in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        total_samples_in_class = len(class_indices[class_id])
        if total_samples_in_class == 0:
            continue

        np.random.shuffle(class_indices[class_id])  # 受 seed 控制
        
        proportions = class_distribution[class_id]
        
        samples_per_client = (proportions * total_samples_in_class).astype(int)
        
        remainder = total_samples_in_class - samples_per_client.sum()
        if remainder > 0:
            add_to_client = np.argmax(proportions)
            samples_per_client[add_to_client] += remainder
        
        current_sum = samples_per_client.sum()
        while current_sum > total_samples_in_class:
            samples_per_client[np.argmax(samples_per_client)] -= 1
            current_sum -= 1

        start_ptr = 0
        for client_id in range(num_clients):
            num_samples = samples_per_client[client_id]
            end_ptr = start_ptr + num_samples
            
            assigned_indices = class_indices[class_id][start_ptr:end_ptr]
            client_indices[client_id].extend(assigned_indices)
            
            start_ptr = end_ptr

    return {i: client_indices[i] for i in range(num_clients)}


def partition_dataset_by_dirichlet(
    train_dataset,
    test_dataset,
    num_clients=10,
    alpha: Union[float, Sequence[float]] = 0.5,
    seed=42,
):
    """
    通过狄利克雷分布将训练集和测试集划分为 Non-IID 子集。
    保证每个客户端的训练集和测试集遵循相同的类别分布逻辑。

    参数:
    - train_dataset (Dataset): 完整的训练数据集。
    - test_dataset (Dataset): 完整的测试数据集。
    - num_clients (int): 客户端数量。
    - alpha (float | Sequence[float]):
      狄利克雷分布参数。支持两种形式：
      1) float：所有客户端共享同一个 alpha（原始行为）
      2) 长度为 num_clients 的序列：每个客户端使用不同 alpha
    - seed (int): 随机种子，用于控制 Dirichlet 分布和 shuffle 的随机性。

    返回:
    - Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
      一个元组，包含两个字典：(train_partition_dict, test_partition_dict)。
    """
    # 设置随机种子
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    if isinstance(train_dataset, Subset):
        try:
            train_labels = torch.tensor(train_dataset.dataset.targets)[train_dataset.indices]
        except AttributeError:
            train_labels = torch.tensor(train_dataset.dataset.labels)[train_dataset.indices]
    else:
        try:
            train_labels = torch.tensor(train_dataset.targets)
        except AttributeError:
            train_labels = torch.tensor(train_dataset.labels)
    
    num_classes = len(torch.unique(train_labels))
    
    # 兼容 alpha 为 float（统一）或 list/ndarray（每客户端不同）
    alpha_arr = np.asarray(alpha, dtype=float)
    if alpha_arr.ndim == 0:
        alpha_vector = np.full(num_clients, float(alpha_arr))
        alpha_desc = f"scalar alpha={float(alpha_arr):.4f}"
    else:
        if len(alpha_arr) != num_clients:
            raise ValueError(
                f"len(alpha)={len(alpha_arr)} must equal num_clients={num_clients}"
            )
        alpha_vector = alpha_arr
        alpha_desc = (
            f"per-client alpha, min={alpha_vector.min():.4f}, "
            f"max={alpha_vector.max():.4f}, mean={alpha_vector.mean():.4f}"
        )
    if np.any(alpha_vector <= 0):
        raise ValueError("All alpha values must be > 0 for Dirichlet.")

    # 【核心】生成一次分布，这个分布将同时用于训练集和测试集
    print(f"Generating a single distribution blueprint with {alpha_desc}, seed={seed}...")
    class_distribution = rng.dirichlet(alpha_vector, num_classes)

    # 2. 将此蓝图分别应用于训练集和测试集
    print("Applying blueprint to train dataset...")
    train_partition_dict = _apply_partition_logic(
        train_dataset, num_clients, class_distribution, seed=seed
    )
    
    print("Applying blueprint to test dataset...")
    test_seed = (seed + 1) if seed is not None else None
    test_partition_dict = _apply_partition_logic(
        test_dataset, num_clients, class_distribution, seed=test_seed
    )
    
    # 3. 验证并返回结果
    train_total_assigned = sum(len(indices) for indices in train_partition_dict.values())
    print(f"Train Dataset: {len(train_dataset)} samples -> Assigned: {train_total_assigned} samples.")

    test_total_assigned = sum(len(indices) for indices in test_partition_dict.values())
    print(f"Test Dataset:  {len(test_dataset)} samples -> Assigned: {test_total_assigned} samples.")
    
    return train_partition_dict, test_partition_dict



def visualize_partition(partition_dict: Dict[int, List[int]], dataset: Dataset, num_clients: int):
    """
    可视化数据划分结果。
    """
    
    if isinstance(dataset, Subset):
        # 如果是 Subset 对象，需要通过原始数据集和索引来获取标签
        # 注意: 原始数据集可能用 .targets 或 .labels，两个都试试
        try:
            # torchvision 数据集通常用 .targets
            labels = torch.tensor(dataset.dataset.targets)[dataset.indices]
        except AttributeError:
            # 其他数据集可能用 .labels
            labels = torch.tensor(dataset.dataset.labels)[dataset.indices]
    else:
        # 如果是普通数据集，使用原来的方法
        try:
            train_labelslabels = torch.tensor(dataset.targets)
        except AttributeError:
            labels = torch.tensor(dataset.labels)
        
    num_classes = len(torch.unique(labels))

    # 创建一个矩阵来存储每个客户端的标签分布
    distribution_matrix = np.zeros((num_clients, num_classes))

    for client_id, indices in partition_dict.items():
        client_labels = labels[indices]
        for class_id in range(num_classes):
            distribution_matrix[client_id, class_id] = (client_labels == class_id).sum().item()
            
    plt.figure(figsize=(12, 8))
    sns.heatmap(distribution_matrix, annot=True, fmt=".0f", cmap="viridis")
    plt.xlabel("Class ID")
    plt.ylabel("Client ID")
    plt.title(f"Data Distribution Across Clients")
    plt.show()
    import swanlab
    if swanlab:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(distribution_matrix, annot=True, fmt=".0f", cmap="viridis", ax=ax)
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Client ID")
        ax.set_title(f"Data Distribution (alpha={0.1})")
        caption = 'Data_Distribution_Heatmap'
        swanlab.log({"Data_Distribution_Heatmap": swanlab.Image(fig, caption=caption)})
        plt.close(fig)

def get_client_dataset(
    train_dataset,
    test_dataset=None,
    alpha=0.1,
    NUM_CLIENTS=5,
    BATCH_SIZE=32,
    seed=None,
    client_alpha_range: Tuple[float, float] = None,
    client_alphas: Sequence[float] = None,
) -> dict:
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        import random
        random.seed(seed)

    rng = np.random.RandomState(seed) if seed is not None else np.random

    if client_alphas is not None and client_alpha_range is not None:
        raise ValueError("Please provide either client_alphas or client_alpha_range, not both.")

    # alpha 优先级：client_alphas > client_alpha_range > alpha(统一值)
    alpha_used = alpha
    if client_alphas is not None:
        alpha_used = np.asarray(client_alphas, dtype=float)
        if len(alpha_used) != NUM_CLIENTS:
            raise ValueError(
                f"len(client_alphas)={len(alpha_used)} must equal NUM_CLIENTS={NUM_CLIENTS}"
            )
    elif client_alpha_range is not None:
        low, high = client_alpha_range
        if not (low > 0 and high > 0 and high >= low):
            raise ValueError(
                "client_alpha_range must satisfy: low>0, high>0, and high>=low"
            )
        alpha_used = rng.uniform(low, high, size=NUM_CLIENTS)

    if np.isscalar(alpha_used):
        alpha_list = [float(alpha_used)] * NUM_CLIENTS
    else:
        alpha_list = np.asarray(alpha_used, dtype=float).tolist()
    
    train_client_partition, test_client_partition = partition_dataset_by_dirichlet(
        train_dataset, test_dataset, NUM_CLIENTS, alpha_used, seed=seed
    )
    
    client_train_datasets = []
    client_test_datasets = []
    num = 0

    for client_id in range(NUM_CLIENTS):
        client_indices = train_client_partition[client_id]
        num += len(client_indices)
        client_dataset = Subset(train_dataset, client_indices)
        num_samples_in_client = len(client_dataset)
        local_indices = list(range(num_samples_in_client))
        
        # 使用固定种子打乱（每个客户端使用不同但可复现的种子）
        if seed is not None:
            rng = np.random.RandomState(seed + client_id)
            rng.shuffle(local_indices)
        else:
            np.random.shuffle(local_indices)
            
        client_train_datasets.append(client_dataset)
        print(f"Client {client_id}: {len(client_dataset)} training samples.")

    print(f'train num {num}')
    
    for client_id in range(NUM_CLIENTS):
        client_indices = test_client_partition[client_id]
        client_test_dataset = Subset(test_dataset, client_indices)
        client_test_datasets.append(client_test_dataset)
        print(f"Client {client_id}: {len(client_test_dataset)} test samples.")
        
    return {
        'client_train_datsets': client_train_datasets,
        'client_test_datsets': client_test_datasets,
        'client_alphas': alpha_list,

    }
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import copy
def create_imbalanced_dataset(original_dataset, num_classes=10, total_size=50000, imbalance_ratio=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # --- 适配点 1: 获取 targets ---
    if hasattr(original_dataset, 'labels') and isinstance(original_dataset.labels, np.ndarray):
        # MedMNIST-style: .labels is a numpy array
        targets = np.array(original_dataset.labels).flatten()
    elif hasattr(original_dataset, 'targets'):
        targets = np.array(original_dataset.targets)
    else:
        # 兼容某些特殊封装
        targets = np.array([s[1] for s in original_dataset.samples])
    
    total_indices = []
    
    # 计算每个类别的目标样本数 (Step-Imbalance 逻辑)
    if imbalance_ratio == 1:
        img_num_per_cls = [int(total_size / num_classes)] * num_classes
    else:
        mu = imbalance_ratio ** (1.0 / (num_classes - 1.0))
        weights = [1.0 / (mu ** c) for c in range(num_classes)]
        weights = np.array(weights)
        weights_norm = weights / weights.sum()
        img_num_per_cls = (weights_norm * total_size).astype(int)
        diff = total_size - img_num_per_cls.sum()
        img_num_per_cls[0] += diff
    
    print(f"目标各类别样本分布 (前 5 类)：{img_num_per_cls[:5]} ...")
    
    actual_img_num_per_cls = []
    for cls_idx in range(num_classes):
        all_indices = np.where(targets == cls_idx)[0]
        available_num = len(all_indices)
        target_num = img_num_per_cls[cls_idx]
        
        if target_num <= available_num:
            selected_idx = np.random.choice(all_indices, target_num, replace=False)
        else:
            print(f"⚠️  类别 {cls_idx}: 目标 {target_num}，实际仅有 {available_num}")
            selected_idx = all_indices
        
        actual_img_num_per_cls.append(len(selected_idx))
        total_indices.extend(selected_idx)
    
    np.random.shuffle(total_indices)
    
    # --- 适配点 2: 创建新对象并更新索引 ---
    new_dataset = copy.deepcopy(original_dataset)

    # 情况 A: MedMNIST 格式 (BloodMNIST etc.) — .imgs/.labels 是 numpy 数组
    if hasattr(new_dataset, 'imgs') and hasattr(new_dataset, 'labels') \
       and isinstance(new_dataset.imgs, np.ndarray):
        new_dataset.imgs = new_dataset.imgs[total_indices]
        new_dataset.labels = new_dataset.labels[total_indices]
        # 更新 info 中的样本数，避免 __len__ 中的 assert 失败
        if hasattr(new_dataset, 'info') and 'n_samples' in new_dataset.info:
            new_dataset.info['n_samples'][new_dataset.split] = len(total_indices)
        # 兼容 downstream 对 .targets 的访问
        new_dataset.targets = new_dataset.labels.flatten().tolist()

    # 情况 B: 适配 ImageFolder (Tiny-ImageNet 常规格式)
    elif hasattr(new_dataset, 'samples') and len(new_dataset.samples) > 0:
        # 更新 samples (包含路径和标签)
        new_dataset.samples = [original_dataset.samples[i] for i in total_indices]
        # 更新 imgs (很多时候 imgs 是 samples 的别名，但显式更新更保险)
        new_dataset.imgs = new_dataset.samples
        # 更新 targets 列表
        new_dataset.targets = [s[1] for s in new_dataset.samples]
        
    # 情况 B: 适配 CIFAR/MNIST 格式 (保留原有逻辑)
    elif hasattr(new_dataset, 'data'):
        if torch.is_tensor(new_dataset.data):
            new_dataset.data = new_dataset.data[total_indices]
        else:
            new_dataset.data = np.array(new_dataset.data)[total_indices]
            
        if torch.is_tensor(new_dataset.targets):
            new_dataset.targets = new_dataset.targets[total_indices]
        else:
            new_dataset.targets = np.array(new_dataset.targets)[total_indices].tolist()

    print(f"✅ 适配完成。实际总样本数：{len(total_indices)}")
    return new_dataset

def get_noise_datasets(args, datasets, noise_rates):
    if args.noise_mode == 'None':
        return datasets, []
    noise_group = []
    from lib.model.fedocr import NoisyDatasetWrapper
    noise_type = getattr(args, 'noise_type', 'symmetric')
    dataset_name = getattr(args, 'data_name', None)
    for idx, dataset in enumerate(datasets):
        datasets[idx] = NoisyDatasetWrapper(
            dataset,
            noise_rate=noise_rates[idx],
            num_classes=args.num_classes,
            noise_type=noise_type,
            dataset_name=dataset_name,
        )
        noise_group.append(idx)
    return datasets, noise_group

def get_loaders(args, datasets, testdatasets):
    train_dataloaders = []
    test_dataloaders = []
    for client_id, (client_train_dataset, client_test_dataset) in enumerate(zip(datasets, testdatasets)):
        train_size = len(client_train_dataset)
        test_size = len(client_test_dataset)

        # RandomSampler requires at least one sample; fallback to non-shuffled
        # loader when a client split is empty.
        train_shuffle = train_size > 0
        if train_size == 0:
            print(f"Warning: client {client_id} has an empty train split; using shuffle=False.")

        if test_size == 0:
            print(f"Warning: client {client_id} has an empty test split.")

        train_dataloader = DataLoader(
            client_train_dataset,
            batch_size=args.batch_size,
            shuffle=train_shuffle,
        )
        train_dataloaders.append(train_dataloader)

        # Keep evaluation deterministic and robust for empty test splits.
        test_dataloader = DataLoader(
            client_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_dataloaders.append(test_dataloader)
    args.data_config.train_dataloaders = train_dataloaders
    args.data_config.test_dataloaders = test_dataloaders
    return train_dataloaders, test_dataloaders
