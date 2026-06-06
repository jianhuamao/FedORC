# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import os
import pdb
import torchvision.transforms
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
__LAYER_LIST__ = ['layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5']

def img_preprocess(x, y=None, use_gpu=True):
    x = torch.tensor(x) / 255.0
    if use_gpu:
        x = x.cuda()
    if y is not None:
        y = torch.LongTensor(y)
        if use_gpu:
            y = y.cuda()
        return x, y

    else:
        return x

def img_preprocess_cifar(x, y=None, use_gpu=True):
    mean_list = [125.3, 123.0, 113.9]
    std_list = [63.0, 62.1, 66.7]

    new_x_list = []
    for i, m in enumerate(mean_list):
        x_ = (x[:,i] - m) / (std_list[i])
        new_x_list.append(x_)
    
    x = np.array(new_x_list).transpose(1,0,2,3)
    
    # flatten
    x = x.reshape(len(x), 3*32*32)
    x = torch.Tensor(x)

    if use_gpu:
        x = x.cuda()

    if y is not None:
        y = torch.LongTensor(y)
        if use_gpu:
            y = y.cuda()

        return x, y

    else:
        return x

def train(model,
    sub_idx,
    train_dataset,
    val_dataset,
    num_epoch,
    batch_size,
    lr, 
    weight_decay,
    early_stop_ckpt_path,
    early_stop_tolerance=3,
    verbose=True,
    ):
    """Given selected subset, train the model until converge.
    """
    # early stop
    best_va_acc = 0
    num_all_train = 0
    early_stop_counter = 0
    x_tr = []
    y_tr = []
    x_va = []
    y_va = []
    for (image, label) in train_dataset:
        x_tr.append(image)
        y_tr.append(label)
    for (image, label) in val_dataset:
        x_va.append(image)
        y_va.append(label)
    x_tr = torch.stack(x_tr).to(device='cuda')
    y_tr = torch.LongTensor(y_tr).to(device='cuda')
    x_va = torch.stack(x_va).to(device='cuda')
    y_va = torch.LongTensor(y_va).to(device='cuda')
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    
    # init training
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    num_all_tr_batch = int(np.ceil(len(sub_idx) / batch_size))

    # num class
    
    num_class = torch.unique(y_va).shape[0]
    for epoch in tqdm(range(num_epoch)):
        total_loss = 0
        model.train()
        np.random.shuffle(sub_idx)

        for idx in range(num_all_tr_batch):
            batch_idx = sub_idx[idx*batch_size:(idx+1)*batch_size]
            x_batch = x_tr[batch_idx]
            y_batch = y_tr[batch_idx]

            pred = model(x_batch)
            if num_class > 2:
                loss = F.cross_entropy(pred, y_batch,
                    reduction="none")
            else:
                loss = F.binary_cross_entropy(pred[:,0], y_batch.float(), 
                    reduction="none")

            sum_loss = torch.sum(loss)
            avg_loss = torch.mean(loss)

            num_all_train += len(x_batch)
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()

            total_loss = total_loss + sum_loss.detach()

        if x_va is not None:
            # evaluate on va set
            model.eval()
            pred_va = predict(model, x_va)
            acc_va = eval_metric(pred_va, y_va, num_class)
            if verbose:
                print("epoch: {}, acc: {}".format(epoch, acc_va.item()))
            
            if epoch == 0:
                best_va_acc = acc_va

            if acc_va > best_va_acc:
                best_va_acc = acc_va
                early_stop_counter = 0
                # save model
                save_model(early_stop_ckpt_path, model)

            else:
                early_stop_counter += 1

            if early_stop_counter >= early_stop_tolerance:
                if verbose:
                    print("early stop on epoch {}, val acc {}".format(epoch, best_va_acc))
                # load model from the best checkpoint
                load_model(early_stop_ckpt_path, model)
                break

    return best_va_acc

def train_prior(model,
    x_tr, y_tr,
    num_epoch=10,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-5,
    early_stop_ckpt_path="./checkpoints/mlp_prior.pth",
    verbose=False,
    ):
    all_tr_idx = np.arange(len(x_tr))
    train(model, all_tr_idx, x_tr, y_tr, x_tr, y_tr, 
        num_epoch=num_epoch,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        early_stop_ckpt_path=early_stop_ckpt_path,
        verbose=verbose,
        )
    w0_dict = dict()
    for param in model.named_parameters():
        w0_dict[param[0]] = param[1].clone().detach() # detach but still on gpu
    model.w0_dict = w0_dict
    model._initialize_weights()
    print("done get prior weights")

def train_track_info(model,
    sub_idx,
    x_tr, y_tr, 
    x_va, y_va, 
    num_epoch,
    batch_size,
    lr,
    weight_decay,
    track_info_per_iter=-1,
    verbose=True,
    ):
    """Given selected subset, train the model until converge.
    Args:
        model: the trained model class
        sub_idx: picked sample indices in training data
        x_tr, y_tr, x_va, y_va: tr/va data set and labels
        track_info_per_iter: evaluate information per %S iterations (SGD updates),
            if set to -1, track info at the end of every epoch
    """

    info_dict = defaultdict(list)
    loss_acc_dict = defaultdict(list)

    # init training with the SGLD optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    # num class
    num_class = torch.unique(y_va).shape[0]
    num_all_tr_batch = int(np.ceil(len(sub_idx) / batch_size))
    num_all_train = 0
    iteration = 0
    for epoch in range(num_epoch):
        total_loss = 0
        model.train()
        np.random.shuffle(sub_idx)

        for idx in range(num_all_tr_batch):
            iteration += 1
            batch_idx = sub_idx[idx*batch_size:(idx+1)*batch_size]
            x_batch = x_tr[batch_idx]
            y_batch = y_tr[batch_idx]

            pred = model(x_batch)

            if num_class > 2:
                loss = F.cross_entropy(pred, y_batch,
                    reduction="none")
            else:
                loss = F.binary_cross_entropy(pred[:,0], y_batch.float(), 
                    reduction="none")

            avg_loss = torch.mean(loss)

            optimizer.zero_grad()

            avg_loss.backward()

            optimizer.step()

            num_all_train += len(x_batch)

            total_loss = total_loss + avg_loss.item()

            if iteration % track_info_per_iter == 0 and track_info_per_iter > 0:
                # estimate information stored in weights
                info = model.compute_information_bp_fast(x_tr, y_tr, no_bp=True)
                for k in info.keys():
                    info_dict[k].append(info[k])
                if verbose:
                    print("iteration/epoch: {}/{}, info: {}".format(iteration, epoch, info))
        if verbose:
            print("epoch: {}, tr loss: {}, lr: {:.6f}".format(epoch, total_loss/num_all_tr_batch, lr))

        # start to evaluate
        if epoch % 1 == 0:
            model.eval()
            pred_tr = predict(model, x_tr)
            acc_tr = eval_metric(pred_tr, y_tr, num_class)

            loss_acc_dict["tr_loss"].append((total_loss/num_all_tr_batch))
            loss_acc_dict["tr_acc"].append(acc_tr.item())

            if x_va is not None:
                # evaluate on va set
                model.eval()
                pred_va = predict(model, x_va)
                acc_va = eval_metric(pred_va, y_va, num_class)
                if verbose:
                    print("epoch: {}, va acc: {}".format(epoch, acc_va.item()))
                loss_acc_dict["va_acc"].append(acc_va.item())
        
        # track info every epoch        
        if track_info_per_iter == -1:
            info = model.compute_information_bp_fast(x_tr, y_tr, no_bp=True)
            for k in info.keys():
                info_dict[k].append(info[k])
            if verbose:
                print("epoch: {}, info: {}".format(epoch, info))
        
            l2_norm = 0
            for pa in model.named_parameters():
                l2_norm += pa[1].data.norm(2)
            loss_acc_dict["l2_norm"].append(l2_norm.cpu().item())



    return info_dict, loss_acc_dict

def save_model(ckpt_path, model):
    torch.save(model.state_dict(), ckpt_path)
    return

def load_model(ckpt_path, model):
    try:
        model.load_state_dict(torch.load(ckpt_path))
    except:
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    return

def predict(model, x, batch_size=100):
    model.eval()
    num_all_batch = np.ceil(len(x)/batch_size).astype(int)
    pred = []
    for i in range(num_all_batch):
        with torch.no_grad():
            pred_ = model(x[i*batch_size:(i+1)*batch_size])
            pred.append(pred_)

    pred_all = torch.cat(pred) # ?, num_class
    return pred_all

def eval_metric(pred, y, num_class):
    if num_class > 2:
        pred_argmax = torch.max(pred, 1)[1]
        acc = torch.sum((pred_argmax == y).float()) / len(y)
    else:
        acc = eval_metric_binary(pred, y)
    return acc

def eval_metric_binary(pred, y):
    pred_label = np.ones(len(pred))
    y_label = y.detach().cpu().numpy()
    pred_prob = pred.flatten().cpu().detach().numpy()
    pred_label[pred_prob < 0.5] = 0.0
    acc = torch.Tensor(y_label == pred_label).float().mean()
    return acc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def feature_map_size(dataname):
    ft_map_size = {
        'cifar10':4,
        'cifar100':4,
        'stl10':12,
        'svhn':4,
        }
    return ft_map_size[dataname]


'''specifically used for plot jupyter notebook.
'''
def plot_info_acc(info_dict, loss_acc_list, act, fig_dir='./figure'):
    df_info = pd.DataFrame(info_dict)
    with plt.style.context(['science','nature',]):
        fig, axs = plt.subplots(2, 1, figsize=(6,8))
        for i,col in enumerate(df_info.columns):
            axs[0].plot(df_info[col], label=__LAYER_LIST__[i], lw=2)
        axs[0].set_xlabel('epoch', size=24)
        axs[0].set_ylabel('IIW',size=24)
        axs[0].tick_params(labelsize=20)
        axs[0].set_title('IIW of {} MLP'.format(act), size=20)
        axs[0].legend(fontsize=24)

        # plot loss acc
        ax1 = axs[1]
        ax2 = ax1.twinx()
        lns1 = ax1.plot(loss_acc_list['tr_loss'], label='train loss', color='r', lw=2)
        lns2 = ax2.plot(loss_acc_list['va_acc'], label='test acc', lw=2)
        ax1.set_xlabel('epoch', size=24)
        ax1.set_ylabel('loss', size=24)
        ax2.set_ylabel('acc', size=24)
        ax1.tick_params(labelsize=20)
        ax2.tick_params(labelsize=20)
        ax1.set_ylim(0.3,2.5)
        ax2.set_ylim(0.5,0.8)
        ax1.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
        ax2.set_yticks([0.5,0.6,0.7,0.8])
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, fontsize=24)
        plt.tight_layout()


    plt.savefig(os.path.join(fig_dir,"{}_acc_loss.png".format(act)),bbox_inches = 'tight')
    plt.show()


def plot_info(info_dict, fig_dir='./figure', use_legend=True):
    '''specifically used for plot jupyter notebook.
    '''
    df_info = pd.DataFrame(info_dict)
    with plt.style.context(['science','nature',]):
        fig, axs = plt.subplots(figsize=(6,4))
        for i,col in enumerate(df_info.columns):
            axs.plot(df_info[col], label=__LAYER_LIST__[i], lw=2)
        axs.set_xlabel('iteration', size=28)
        axs.set_ylabel('IIW',size=28)
        axs.tick_params(labelsize=24)
        axs.yaxis.get_major_formatter().set_powerlimits((0,1))
        axs.set_title('IIW of {}-layer MLP'.format(int(len(df_info.columns))), size=28)
        if use_legend:
            axs.legend(fontsize=26)
        plt.tight_layout()
    plt.savefig(os.path.join(fig_dir,"mlp_{}_info.pdf".format(int(len(df_info.columns)))),bbox_inches = 'tight')
    plt.show()

def get_client_instance(client_idx, Client, model, global_state_dict, device, 
                        train_dataloaders, test_dataloaders, datasets, 
                        num_epochs, lr, in_channels, num_classes, round_num,
                        client_model_on_cpu=True, use_amp=True, cuda_empty_cache=True,
                        local_denoise_method='none',
                        denoise_forget_rate=0.2,
                        denoise_num_gradual=10,
                        denoise_warmup_epochs=1,
                        denoise_lambda_u=1.0,
                        denoise_temperature=0.5,
                        denoise_mixup_alpha=4.0,
                        denoise_p_threshold=0.5,
                        denoise_rampup_length=16,
                        jocor_co_lambda=0.1,
                        aggregation='fedavg',
                        fedprox_mu=0.0,
                        train_subset_indices_map=None,
                        fednoro_noisy_clients=None,
                        fednoro_kd_weight=0.5,
                        fednoro_kd_temperature=0.8,
                        fedned_use_pseudo_labels=False,
                        fedned_pseudo_threshold=0.95,
                        fedned_temperature=2.0,
                        fedfixer_lambda=1.0,
                        fedfixer_beta=0.1,
                        fedfixer_forget_rate=0.2,
                        fedfixer_num_gradual=10,
                        fedfixer_warmup_epochs=1,
                        feddiv_warmup_rounds=10,
                        feddiv_confidence_threshold=0.5,
                        feddiv_consistency_weight=0.1,
                        feddiv_gmm_max_iter=10,
                        feddiv_global_gmm=None):
    """实例化一个客户端对象"""
    local_model = model(in_channels, num_classes)
    local_model.load_state_dict(global_state_dict)

    train_dataset = datasets[client_idx]
    train_loader = train_dataloaders[client_idx]
    if train_subset_indices_map is not None and client_idx in train_subset_indices_map:
        subset_indices = list(train_subset_indices_map[client_idx])
        train_dataset = Subset(train_dataset, subset_indices)
        train_loader = DataLoader(
            train_dataset,
            batch_size=getattr(train_dataloaders[client_idx], "batch_size", 64),
            shuffle=len(train_dataset) > 0,
        )

    if not client_model_on_cpu:
        local_model = local_model.to(device)
    fednoro_noisy_set = set(fednoro_noisy_clients or [])
    return Client(
        client_id=client_idx,
        model=local_model,
        train_loader=train_loader,
        test_loader=test_dataloaders[client_idx],
        train_dataset=train_dataset,
        val_dataset=None,
        device=device,
        epochs=num_epochs,
        lr=lr,
        global_model_state_dict=global_state_dict,
        round=round_num,
        info=None,
        client_model_on_cpu=client_model_on_cpu,
        use_amp=use_amp,
        cuda_empty_cache=cuda_empty_cache,
        local_denoise_method=local_denoise_method,
        denoise_forget_rate=denoise_forget_rate,
        denoise_num_gradual=denoise_num_gradual,
        denoise_warmup_epochs=denoise_warmup_epochs,
        denoise_lambda_u=denoise_lambda_u,
        denoise_temperature=denoise_temperature,
        denoise_mixup_alpha=denoise_mixup_alpha,
        denoise_p_threshold=denoise_p_threshold,
        denoise_rampup_length=denoise_rampup_length,
        jocor_co_lambda=jocor_co_lambda,
        aggregation=aggregation,
        fedprox_mu=fedprox_mu,
        fednoro_is_noisy=client_idx in fednoro_noisy_set,
        fednoro_kd_weight=fednoro_kd_weight,
        fednoro_kd_temperature=fednoro_kd_temperature,
        fedned_use_pseudo_labels=fedned_use_pseudo_labels,
        fedned_pseudo_threshold=fedned_pseudo_threshold,
        fedned_temperature=fedned_temperature,
        fedfixer_lambda=fedfixer_lambda,
        fedfixer_beta=fedfixer_beta,
        fedfixer_forget_rate=fedfixer_forget_rate,
        fedfixer_num_gradual=fedfixer_num_gradual,
        fedfixer_warmup_epochs=fedfixer_warmup_epochs,
        feddiv_warmup_rounds=feddiv_warmup_rounds,
        feddiv_confidence_threshold=feddiv_confidence_threshold,
        feddiv_consistency_weight=feddiv_consistency_weight,
        feddiv_gmm_max_iter=feddiv_gmm_max_iter,
        feddiv_global_gmm=feddiv_global_gmm,
    )
def single_training_loop(clients):
    client_models_state_dicts = []
    infos = []
    for client in clients:
        updated_weights, _, info = client.train() 
        # updated_weights = move_params_to_cpu(updated_weights)
        client_models_state_dicts.append(updated_weights)
        infos.append(info)
    return client_models_state_dicts, infos

import copy
from lib.model.fedgap import get_targets
import logging
def move_params_to_cpu(weights_after_train):
    weight_on_cpu = {k: v.cpu() for k, v in weights_after_train.items()}
    return weight_on_cpu

def fedocr_client_selection(
    client_candidates,
    selector,
    noise_group,
    TARGET_CLIENT_NUM,
    device,
    num_classes,
    client_alphas=None,
    cost_tracker=None,
):
    print(">>> Performing FedOCR Auditing & Selection...")
    distribution_source = getattr(selector, 'distribution_source', 'local_label')
    if distribution_source not in {'local_label', 'proxy_pred'}:
        raise ValueError(
            f"Unsupported distribution_source={distribution_source}, "
            "expected one of: local_label, proxy_pred"
        )
    print(f">>> FedOCR distribution source: {distribution_source}")
    if cost_tracker is not None and hasattr(selector, "cost_tracker"):
        selector.cost_tracker = cost_tracker

    NUM_CLIENTS = len(client_candidates)
    candidates_info = []
    selection_infos = []
    train_acc_fep = 0.0
    for client in client_candidates:
        client_idx = client.client_id
        if client_idx in noise_group:
            targets = np.array(client.train_dataset.noisy_targets)
        else:
            targets = get_targets(client.train_dataset)
        n_k = len(targets)

        updated_weights, _, info = client.train() 
        selection_infos.append(info)
        updated_weights_cpu = move_params_to_cpu(updated_weights)

        if distribution_source == 'proxy_pred':
            # 新接口：使用上传后的客户端模型在代理数据上的预测类别分布。
            p_k = selector.get_proxy_pred_distribution(updated_weights_cpu).to(dtype=torch.float32)
        else:
            # 兼容旧接口：使用客户端本地标签统计分布。
            p_k = np.bincount(targets, minlength=num_classes)
            p_k = torch.from_numpy(p_k).to(dtype=torch.float32)

        acc, loss = client.get_train_acc()
        train_acc_fep += float(n_k)
        train_loss = info['loss'] if info else 0.0 
        alpha_i = None
        if client_alphas is not None and client_idx < len(client_alphas):
            alpha_i = float(client_alphas[client_idx])
        candidates_info.append({
            'id': client_idx,
            'params': updated_weights_cpu,  # 训练后的参数
            'n_k': n_k,                 # 样本量
            'p_k': p_k,                 # 分布向量
            'distribution_source': distribution_source,
            'alpha_i': alpha_i,         # 客户端 Dirichlet alpha
            'train_loss': train_loss,    # 本地 Loss
            'train_acc': acc
        })
    gap_selected_client_indices = selector.select_clients(candidates_info, TARGET_CLIENT_NUM)
    if cost_tracker is not None:
        aux_bytes = cost_tracker.scalar_bytes(NUM_CLIENTS * 2)
        if distribution_source == 'local_label':
            aux_bytes += cost_tracker.class_histogram_bytes(NUM_CLIENTS)
        cost_tracker.record_selection_phase(
            phase=f"fedocr_audit_{distribution_source}",
            trained_count=NUM_CLIENTS,
            uploaded_model_count=NUM_CLIENTS,
            train_infos=selection_infos,
            extra_compute_fep=train_acc_fep,
            aux_bytes=aux_bytes,
            selected_count=len(gap_selected_client_indices),
        )
    loss_list = []
    print(
        ">>> FedOCR Selection Done! "
        f"Selected Count: {len(gap_selected_client_indices)}, "
        f"Selected Clients: {gap_selected_client_indices}"
    )
    logging.info(
        "FedOCR Selection Done | selected_count=%d selected_clients=%s",
        len(gap_selected_client_indices),
        gap_selected_client_indices,
    )
    return gap_selected_client_indices


# Backward-compatible alias.
gap_client_selection = fedocr_client_selection

def PoC_client_selection(client_candidates, TARGET_CLIENT_NUM, cost_tracker=None):
    loss_list = []
    id_list = []
    infos = []
    for client in client_candidates:
        updated_weights, _, info = client.train() 
        infos.append(info)
        client_idx = client.client_id
        id_list.append(client_idx)
        loss_train = info['loss'] if info else 0.0
        loss_list.append(loss_train)
        logging.info(f'PoC: client{client_idx}, loss: {loss_train}, num_samples: {client.data_num}')
        print(f'PoC: client{client_idx}, loss: {loss_train}, num_samples: {client.data_num}')
    poc_selected_client_indices  = np.argsort(loss_list)[-TARGET_CLIENT_NUM:]    
    selected = [id_list[i] for i in poc_selected_client_indices]
    if cost_tracker is not None:
        cost_tracker.record_selection_phase(
            phase="poc_static_candidate_training",
            trained_count=len(client_candidates),
            uploaded_model_count=0,
            train_infos=infos,
            aux_bytes=cost_tracker.scalar_bytes(len(client_candidates)),
            selected_count=len(selected),
        )
    return selected

import random
def random_client_selection(TOTAL_CLIENT_NUM, TARGET_CLIENT_NUM):
    samples = random.sample(range(TOTAL_CLIENT_NUM),TARGET_CLIENT_NUM)
    return samples 



def get_current_exp_name(mode, TARGET_CLIENT_NUM, TOTAL_CLIENT_NUM):
    if TARGET_CLIENT_NUM == TOTAL_CLIENT_NUM:
        current_exp_name = "all seltected"
    else:
        current_exp_name = f"{mode} {TARGET_CLIENT_NUM} selected" 
    return current_exp_name

def get_dataset_arg(data_name):
    if data_name == 'mnist':
        return 1, 10
    elif data_name == 'cifar10':
        return 3, 10
    elif data_name == 'cifar100':
        return 3, 100
    elif data_name == 'tinyimagenet':
        return 3, 200
    elif data_name == 'bloodmnist':
        return 3, 8
    else:
        raise EOFError('data set have not recorded')

import torch
import torch.nn.functional as F

def get_feedback(client, dataloader, device):
    model = client.model.to(device)
    model = model.eval()
    total_entropy = 0.0
    sample_count = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            log_probs = torch.log(probs + 1e-8)
            entropy = -torch.sum(probs * log_probs, dim=1)
            total_entropy += entropy.sum().item()
            sample_count += images.size(0)

    avg_entropy = total_entropy / max(sample_count, 1)
    if getattr(client, 'client_model_on_cpu', False):
        model.to("cpu")
    if getattr(client, 'cuda_empty_cache', False) and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return avg_entropy

def calculate_quality_contribution(scores, beta=2.0):
    scores = np.array(scores)
    shifted_scores = scores - np.min(scores)
    unnormalized_weights = np.exp(-beta * shifted_scores)
    weights = unnormalized_weights / np.sum(unnormalized_weights)
    return weights

def init_log(args):
    import os
    logdir = args.logdir
    logfile = args.logfile
    log_path = os.path.join(logdir, logfile)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    # 检查文件是否存在，不存在则创建
    if not os.path.exists(log_path):
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write('')  # 创建空文件
        print(f"✓ 已创建日志文件：{log_path}")
    else:
        print(f"✓ 日志文件已存在：{log_path}")

    import logging 
    basic_config_kwargs = {
        "level": logging.INFO,
        "filename": os.path.join(logdir, logfile),  # 取消注释可写入文件
        "filemode": "a",  # 'w' 覆盖模式, 'a' 追加模式
    }
    try:
        logging.basicConfig(force=True, **basic_config_kwargs)
    except TypeError:
        # 兼容不支持 force 参数的老版本 Python
        logging.basicConfig(**basic_config_kwargs)

def get_noise_rate(args):
    noise_mode = str(args.noise_mode).strip().lower().replace("-", " ").replace("_", " ")
    if noise_mode == 'none':
        return [0.0] * args.num_clients
    elif noise_mode in {'linear denoising', 'linear noise'}:
        return np.linspace(0, 1, args.num_clients).tolist()
    elif noise_mode in {'gaussian noise', 'gaussian', 'normal noise', 'normal'}:
        mean = float(args.noise_gaussian_mean)
        std = float(args.noise_gaussian_std)
        low = float(args.noise_gaussian_min)
        high = float(args.noise_gaussian_max)
        if std < 0:
            raise ValueError("--noise_gaussian_std must be >= 0.")
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError("--noise_gaussian_min/max must satisfy 0 <= min <= max <= 1.")

        rng = np.random.default_rng(int(args.seed))
        noise_rates = rng.normal(loc=mean, scale=std, size=int(args.num_clients))
        noise_rates = np.clip(noise_rates, low, high)
        logging.info(
            "Gaussian noise rates | mean=%.4f std=%.4f clip=[%.4f, %.4f] "
            "actual_mean=%.4f actual_std=%.4f min=%.4f max=%.4f",
            mean,
            std,
            low,
            high,
            float(np.mean(noise_rates)),
            float(np.std(noise_rates)),
            float(np.min(noise_rates)),
            float(np.max(noise_rates)),
        )
        return noise_rates.tolist()
    elif noise_mode == '3phase':
        low_inter = int(args.num_clients * 0.6)
        mid_inter = int(args.num_clients * 0.3)
        high_inter = args.num_clients - low_inter - mid_inter
        return np.concatenate([
            np.linspace(0, 0.1, low_inter),      
            np.linspace(0.11, 0.3, mid_inter),   
            np.linspace(0.31, 1.0, high_inter)    
        ]).tolist()
    elif noise_mode == '2phase':
        return np.concatenate([
            np.linspace(0, 0.1, int(args.num_clients * 0.8)),
            np.linspace(0.11, 0.15, args.num_clients - int(args.num_clients * 0.8))
        ]).tolist()
    else:
        raise ValueError(
            "noise_mode not supported. Use: None | linear_noise | gaussian_noise | 2phase | 3phase"
        )
    
        
