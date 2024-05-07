"""
    This script provides a demo to draw rc curves using the collected data from <collect_feature_logits.py>
"""
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)

# === Env imports ===
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import seaborn as sns
sns.set_theme()
COLORS = list(mcolors.TABLEAU_COLORS)
CF_METHOD_STR_LIST = []
DATASET_NAME_LIST = []
PLOT_SYMBOL_DICT = {}


def calculate_score_residual(
        logits, labels, features, weights, bias,
        clean_set_features, clean_set_logits, training_based_scores=None
    ):
    scores_dict = {}
    residuals_dict = {}

    weight_norm = np.linalg.norm(weights, axis=1, ord=2)
    # bias_aug = bias[:, np.newaxis]
    # weight_aug = np.concatenate([weights, bias_aug], axis=1)
    # weight_norm = np.linalg.norm(weight_aug, axis=1, ord=2)

    # === Prepare sample set to compute H_res scores ===
    # vim_dim = 512
    vim_dim = 64
    in_d_sampled_features = clean_set_features
    in_d_sampled_logits = clean_set_logits
    w, b = weights, bias
    u = -np.matmul(pinv(w), b)

    # === Scores used in previous version ===
    logits_tensor = torch.from_numpy(logits).to(dtype=torch.float)
    # max logit
    max_logit_scores = np.amax(logits, axis=1)
    max_logit_pred = np.argmax(logits, axis=1)
    max_logit_residuals = calculate_residual(max_logit_pred, labels)
    method_name = "max_logit"
    scores_dict[method_name] = max_logit_scores
    residuals_dict[method_name] = max_logit_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [1, None, r"$RL_{max}$", "dashed"]

    
    sr = torch.softmax(logits_tensor, dim=1)
    max_sr_scores = torch.amax(sr, dim=1).numpy()
    # max_sr 
    max_sr_pred = max_logit_pred
    max_sr_residuals = calculate_residual(max_sr_pred, labels)
    method_name = "max_sr"
    scores_dict[method_name] = max_sr_scores
    residuals_dict[method_name] = max_sr_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [0, "o", r"$SR_{max}$", "solid"]

    # doctor
    gx_hat_sqrt = torch.linalg.norm(sr, dim=1, ord=2)
    gx_hat = gx_hat_sqrt ** 2
    doctor_scores = - (torch.ones_like(gx_hat) - gx_hat) / gx_hat
    doctor_scores = doctor_scores.numpy()
    doctor_residuals = max_logit_residuals
    method_name = "doctor"
    scores_dict[method_name] = doctor_scores
    residuals_dict[method_name] = doctor_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [1, None, r"$SR_{doctor}$", "solid"]

    # entropy (neg)
    entropy_scores = sstats.entropy(sr.cpu().numpy(), axis=1) * (-1)
    entropy_residuals = max_logit_residuals
    method_name = "entropy"
    scores_dict[method_name] = entropy_scores
    residuals_dict[method_name] = entropy_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [2, None, r"$SR_{ent}$", "solid"]

    # === Additional Methods from OODs ===
    # # neg L1
    # l1_norm = np.linalg.norm(features, axis=1, ord=1) * (-1)
    # l1_norm_residuals = max_logit_residuals
    # method_name = "l1_norm"
    # scores_dict[method_name] = l1_norm
    # residuals_dict[method_name] = l1_norm_residuals
    # if method_name not in CF_METHOD_STR_LIST:
    #     CF_METHOD_STR_LIST.append(method_name)
    #     PLOT_SYMBOL_DICT[method_name] = [6, None, r"$||z||_1$"]

    # vim residual
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(in_d_sampled_features - u)
    # ec.fit(in_d_sampled_features)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    print("Eig Vals: ", eig_vals)
    NS = np.ascontiguousarray(
        (eigen_vectors.T[np.argsort(eig_vals * -1)[vim_dim:]]).T)
    vim_res_scores = (-1) * np.linalg.norm(np.matmul(features - u, NS), axis=-1)
    # vim_res_scores = (-1) * np.linalg.norm(np.matmul(features, NS), axis=-1)
    vim_res_residuals = max_logit_residuals
    method_name = "vim_residual"
    scores_dict[method_name] = vim_res_scores
    residuals_dict[method_name] = vim_res_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [5, None, r"$Vim_{res}$", "dashed"]
    
    # vim scores
    print('computing alpha...')
    # vlogit_id_train = np.linalg.norm(np.matmul(in_d_sampled_features, NS), axis=-1)
    vlogit_id_train = np.linalg.norm(np.matmul(in_d_sampled_features-u, NS), axis=-1)
    alpha = in_d_sampled_logits.max(axis=-1).mean() / vlogit_id_train.mean()
    print(f'{alpha=:.4f}')
    # vlogit_id_val = np.linalg.norm(np.matmul(features, NS), axis=-1) * alpha
    vlogit_id_val = np.linalg.norm(np.matmul(features-u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logits, axis=-1)
    vim_scores = -vlogit_id_val + energy_id_val
    vim_residuals = max_logit_residuals
    method_name = "vim"
    scores_dict[method_name] = vim_scores
    residuals_dict[method_name] = vim_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [6, None, r"$ViM$", "dashed"]
        # PLOT_SYMBOL_DICT[method_name] = [2, None, r"$S_2$"]
    
    # === SIRC ===
    s2_mean, s2_std = np.mean(vim_res_scores), np.std(vim_res_scores)
    a, b = get_sirc_params(s2_mean, s2_std)
    sirc_scores = sirc(
        max_sr_scores, vim_res_scores, a, b, s1_max=1
    )
    sirc_residuals = max_logit_residuals
    method_name = "sirc"
    scores_dict[method_name] = sirc_scores
    residuals_dict[method_name] = sirc_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [7, None, r"$SIRC$", "solid"]

    # ==== KNN ====
    K = 2  # According to the paper, K corresponds to the num of in-d samples 
    knn_feats = F.normalize(
        torch.from_numpy(clean_set_features).to(dtype=torch.float), dim=-1, p=2
    ).cpu().numpy()
    # create index with vector dimensionality
    index = faiss.IndexFlatL2(knn_feats.shape[-1])
    index.add(knn_feats)

    # Normalize Test Feature
    normed_features = F.normalize(
        torch.from_numpy(features).to(dtype=torch.float), dim=-1, p=2
    ).cpu().numpy()
    # KNN search
    D, _ = index.search(normed_features, K)

    # uncertainty
    knn_uncertainty = torch.tensor(D[:, -1])
    knn_scores = -1 * knn_uncertainty.numpy() # Change to confidence score
    knn_residuals = max_logit_residuals
    # log result
    method_name = "knn"
    scores_dict[method_name] = knn_scores
    residuals_dict[method_name] = knn_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [8, None, r"$KNN$", "dashed"]

    # ==== Energy ====
    energy_score = torch.logsumexp(
        torch.from_numpy(logits).to(dtype=torch.float), 
    dim=-1).numpy()
    energy_residuals = max_logit_residuals
    method_name = "energy"
    scores_dict[method_name] = energy_score
    residuals_dict[method_name] = energy_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [9, None, r"$Energy$", "dashed"]

    # === ReAct (Energy) ===
    c = np.quantile(clean_set_features, 0.9)
    react_features = np.clip(features, a_min=None, a_max=c)
    react_logits = react_features @ weights.T + bias
    react_energy_score = torch.logsumexp(
        torch.from_numpy(react_logits).to(dtype=torch.float), 
    dim=-1).numpy()
    # react_pred = np.argmax(react_logits, axis=1)
    # react_residuals = calculate_residual(react_pred, labels)
    react_residuals = max_logit_residuals
    method_name = "ReAct"
    scores_dict[method_name] = react_energy_score
    residuals_dict[method_name] = react_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [0, None, r"$ReAct$", "dashed"]
    

    # === OURS ====
    # raw margin
    values, indices = torch.topk(logits_tensor, 2, axis=1)
    raw_margin_scores = (values[:, 0] - values[:, 1]).cpu().numpy()
    raw_margin_pred = max_logit_pred
    raw_margin_residuals = calculate_residual(raw_margin_pred, labels)
    method_name = "raw_margin"
    scores_dict[method_name] = raw_margin_scores
    residuals_dict[method_name] = raw_margin_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [4, "p", r"$RL_{conf-M}$", "solid"]
        

    # geo_margin
    geo_distance = logits / weight_norm[np.newaxis, :]
    geo_values, geo_indices = torch.topk(torch.from_numpy(geo_distance).to(dtype=torch.float), 2, axis=1)
    geo_margin_scores = (geo_values[:, 0] - geo_values[:, 1]).cpu().numpy()
    geo_margin_pred = geo_indices[:, 0].cpu().numpy()
    geo_margin_residuals = calculate_residual(geo_margin_pred, labels)
    method_name = "geo_margin"
    scores_dict[method_name] = geo_margin_scores
    residuals_dict[method_name] = geo_margin_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [3, "p", r"$RL_{geo-M}$", "solid"]
        # PLOT_SYMBOL_DICT[method_name] = [3, "p", r"$S_1$"]

    return scores_dict, residuals_dict


def read_data(dir, load_classifier_weight=False):
    raw_logits = np.load(os.path.join(dir, "pred_logits.npy"))
    labels = np.load(os.path.join(dir, "labels.npy"))
    features = np.load(os.path.join(dir, "features.npy"))
    if load_classifier_weight:
        last_layer_weights = np.load(os.path.join(dir, "last_layer_weights.npy"))
        last_layer_bias = np.load(os.path.join(dir, "last_layer_bias.npy"))
    else:
        last_layer_weights = None
        last_layer_bias = None
    return raw_logits, labels, features, last_layer_weights, last_layer_bias


def main(args):
    # === Load collected data ===
    load_data_root = args.data_dir
    logits, labels, features, last_layer_weights, last_layer_bias = read_data(load_data_root, True)
    print("Shape Check: ", logits.shape, labels.shape, features.shape)

    # === Calculate Scores and Residuals for RC ===
    in_scores_dict, in_residuals_dict = calculate_score_residual(
        in_logits, in_labels, in_features, last_layer_weights, last_layer_bias,
        clean_set_features=cali_features, clean_set_logits=cali_logits, training_based_scores=in_train_based_scores
    )

if __name__ == "__main__":
    print("Plotting RC curve using collected data.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", dest="data_dir", type=str,
        default=os.path.join(".", "collected_data_0"),
        help="Folder where the collected logits data are located."
    )
    args = parser.parse_args()
    main(args)
    print("Completed.")