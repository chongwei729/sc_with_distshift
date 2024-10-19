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
import scipy.stats as sstats
from numpy.linalg import pinv
from sklearn.covariance import EmpiricalCovariance
from scipy.special import logsumexp
import torch.nn.functional as F
import faiss
import pandas as pd
# === For plotting ===
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors
import seaborn as sns
sns.set_theme()
COLORS = list(mcolors.TABLEAU_COLORS)
CF_METHOD_STR_LIST = []
PLOT_SYMBOL_DICT = {}


def format_float_list(float_list):
    res = []
    for number in float_list:
        res.append("%.03f" % number)
    return res


def calc_aurc_coverage(coverage_array, sc_risk_array, alphas=[0.1, 0.25, 0.5, 0.75, 1.0]):
    total_len = len(coverage_array)
    res_list = []
    for alpha in alphas:
        end_idx = int(alpha * total_len)
        coverage_slice = coverage_array[-end_idx:]
        risk_slice = sc_risk_array[-end_idx:]

        AUC = np.sum(risk_slice) / len(coverage_slice) * 100
        res_list.append(AUC)
    return res_list


def calculate_residual(pred, label):
    """
        residual --- 0 if pred == label
                --- 1 if pred != label
    """
    pred_tensor = torch.from_numpy(pred)
    label_tensor = torch.from_numpy(label)
    predict_correct_bool = pred_tensor == label_tensor
    residual_tensor = torch.where(predict_correct_bool, 0, 1)
    return residual_tensor.cpu().numpy()


def get_sirc_params(unc_mean, unc_std):
    """
        Adapted from: https://github.com/Guoxoug/SIRC
    """
    # remember that the values are negative
    a = unc_mean - 3 * unc_std
    # investigate effect of varying b
    b = 1/ (unc_std + 1e-12)
    return a, b


def sirc(s1, s2, a, b, s1_max=1):
    """
        Adapted from: https://github.com/Guoxoug/SIRC
        
        Combine 2 confidence metrics with SIRC.
    
    """
    s1_tensor = torch.from_numpy(s1)
    s2_tensor = torch.from_numpy(s2)
    # use logarithm for stability
    soft = (s1_max - s1_tensor).log()
    additional = torch.logaddexp(
        torch.zeros(len(s2_tensor)),
        -b * (s2_tensor - a) 
    )
    score = - soft - additional # return as confidence
    return score.cpu().numpy()


def calculate_score_residual(
    logits, labels, features, weights, bias,
    clean_set_features, clean_set_logits,
):
    scores_dict = {}
    residuals_dict = {}

    # weight_norm = np.linalg.norm(weights, axis=1, ord=2)
    bias_aug = bias[:, np.newaxis]
    weight_aug = np.concatenate([weights, bias_aug], axis=1)
    weight_norm = np.linalg.norm(weight_aug, axis=1, ord=2)

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
    # vim residual
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(in_d_sampled_features - u)
    # ec.fit(in_d_sampled_features)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    # print("Eig Vals: ", eig_vals)
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
    react_residuals = max_logit_residuals
    method_name = "ReAct"
    scores_dict[method_name] = react_energy_score
    residuals_dict[method_name] = react_residuals
    if method_name not in CF_METHOD_STR_LIST:
        CF_METHOD_STR_LIST.append(method_name)
        PLOT_SYMBOL_DICT[method_name] = [0, None, r"$ReAct$", "dashed"]
    

    # === OURS ====
    # RL-conf-M
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
        
    # RL-geo-M
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
    return scores_dict, residuals_dict


def read_data(dir, load_classifier_weight=False):
    """
        Read in the pre-collected data.
    """
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


def RC_curve(residuals, confidence):
    """
        Calculate each point on the RC curve based on residuals and SC scores.
    """
    curve = []
    m = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    cov = len(temp1)
    acc = sum(temp1)
    curve.append((cov/ m, acc / len(temp1)))
    for i in range(0, len(idx_sorted)-1):
        cov = cov-1
        acc = acc-residuals[idx_sorted[i]]
        curve.append((cov / m, acc /(m-i)))
    curve = np.asarray(curve)
    coverage, risk = curve[:, 0], curve[:, 1]
    return coverage, risk


def select_RC_curve_points(coverage_array, risk_array, n_plot_points=40,  min_n_samples=-10):
    """
        Evenly sample some RC points to display.
    """
    plot_interval = len(coverage_array) // n_plot_points
    coverage_plot, risk_plot = coverage_array[0::plot_interval].tolist(), risk_array[0::plot_interval].tolist()
    coverage_plot.append(coverage_array[min_n_samples])
    risk_plot.append(risk_array[min_n_samples])
    return coverage_plot, risk_plot


def plot_rc_curve_demo(total_scores_dict, total_residuals_dict, fig_name, method_name_list=None):
    """
        Plot the RC curve for each score, respectively.
    """
    coverage_dict, risk_dict = {}, {}

    if method_name_list is None:
        method_name_list = CF_METHOD_STR_LIST

    for method_name in method_name_list:
        coverage, sc_risk = RC_curve(
            total_residuals_dict[method_name], total_scores_dict[method_name]
        )
        coverage_dict[method_name] = coverage
        risk_dict[method_name] = sc_risk

    # === Plot RC Curve ===
    plot_n_points = 30
    min_num_samples = -100
    save_path = fig_name
    line_width = 4
    markersize = 8
    alpha = 0.5

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    font_size = 19
    tick_size = 20

    y_min = 0
    y_max = 0
    for method_name in method_name_list:
        coverage_plot, sc_risk_plot = coverage_dict[method_name], risk_dict[method_name]
        x_plot, y_plot = select_RC_curve_points(coverage_plot, sc_risk_plot, plot_n_points, min_num_samples)
        y_max, y_min = max(y_plot[0], y_max), min(np.amin(y_plot), y_min)
        # y_max, y_min = max(np.amax(y_plot), y_max), min(np.amin(y_plot), y_min)
        plot_settings = PLOT_SYMBOL_DICT[method_name]
        ax.plot(
            x_plot, y_plot,
            label=plot_settings[2], lw=line_width, alpha=alpha,
            color=COLORS[plot_settings[0]], marker=plot_settings[1], ls=plot_settings[3], markersize=markersize
        )

    ax.legend(
        loc='lower left', bbox_to_anchor=(-0.25, 1, 1.25, 0.2), mode="expand", 
        borderaxespad=0,
        ncol=3, fancybox=True, shadow=False, fontsize=font_size, framealpha=0.3
    )
    ax.tick_params(axis='x', which='major', colors='black', labelsize=tick_size)
    ax.tick_params(axis='y', which='major', colors='black', labelsize=tick_size)
    ax.set_ylabel(r"Selection Risk", fontsize=font_size)
    ax.set_xlabel(r"Coverage", fontsize=font_size)
    ax.set_ylim([y_min-0.05*y_max, 1.10*y_max])
    # ax.set_xticks([0, 0.5, 1])
    ax.set_xticks([0, 0.5, 1])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim([-0.02, 1.05])
    # ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.set_yticks([y_max/2, y_max])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return coverage_dict, risk_dict





def main(args):
    # === Load collected data ===
    load_data_root = args.data_dir
    # Below loads pre-collected ImageNet (val) data
    in_logits, in_labels, in_features, last_layer_weights, last_layer_bias = read_data(load_data_root, True)
    print("Shape Check: ", in_logits.shape, in_labels.shape, in_features.shape)

    # === Get calibration set indices ===
    # Below are only used for OOD detection scores
    # According to the OOD literature, these should be collected using training (not validation) data
    # However, we assume no-access to training data in our work, so we sample a small subset in the validation set
    # (which is in fact the test set for SC scores), so the OOD scores have a little advantage in this sense.
    in_set_length = in_features.shape[0]
    cali_indices = np.random.choice(in_set_length, args.cali_size, replace=False)
    cali_features = in_features[cali_indices, :]
    cali_logits = in_logits[cali_indices, :]

    # === Calculate Scores and Residuals for RC ===
    in_scores_dict, in_residuals_dict = calculate_score_residual(
        in_logits, in_labels, in_features, last_layer_weights, last_layer_bias,
        clean_set_features=cali_features, clean_set_logits=cali_logits
    )

    # === Generate RC curve ====
    method_name_list = CF_METHOD_STR_LIST
    print(method_name_list)
    save_root = os.path.join(".", "Demo-Vis")
    os.makedirs(save_root, exist_ok=True)
    fig_name = "RC-test.png"
    save_path = os.path.join(save_root, fig_name)
    # Plot RC curve to 'save_path', and return the plot values for AURC calculation
    coverage_dict, residual_dict = plot_rc_curve_demo(
        in_scores_dict, in_residuals_dict, save_path,
        method_name_list=method_name_list
    )
    # Calculate 
    alphas = [0.1, 0.5, 1]
    aurc_res_dict = {"alpha": alphas}
    for method_name in method_name_list:
        coverage_array, residual_array = coverage_dict[method_name], residual_dict[method_name]
        aurc_partial_list = calc_aurc_coverage(coverage_array, residual_array, alphas)
        aurc_res_dict[method_name] = format_float_list(aurc_partial_list)
    save_aurc_dir = os.path.join(save_root, "Demo-AURC.csv")
    df = pd.DataFrame.from_dict(aurc_res_dict).T
    df.to_csv(save_aurc_dir, index=True, header=False)



if __name__ == "__main__":
    print("Plotting RC curve using collected data.\n")
    msg = """
    Due to the large amount of data involed in the original paper, here we only provide an RC curve sample using:

    1) ImageNet (val) set 
    2) collected by EVA model

    only. That means this demo does NOT have any distribution shifted data.

    However, we did include all non-training based SC and OOD scores to profile the RC curves to make it a little more convenient if the readers

    want to try their own experiments with their own data collected. """
    print(msg)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", dest="data_dir", type=str,
        default=os.path.join(".", "collected_data"),
        help="Folder where the collected logits data are located."
    )
    parser.add_argument(
        "--cali_size", dest="cali_size", type=int,
        default=5,
        help="Calibration data size used to determin the OOD score hyper params."
    )
    args = parser.parse_args()
    main(args)
    print("Completed.")