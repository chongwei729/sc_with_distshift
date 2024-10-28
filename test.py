

import random
import argparse
import os
from plot_rc_curve import calculate_score_residual, read_data
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
def compare_two_points_confidence(scores_dict, data, point1, point2, method="max_logit"):
    confidence1 = scores_dict[method][point1]
    confidence2 = scores_dict[method][point2]
    confidence_residual = abs(confidence1 - confidence2)
  
    feature1 = data['features'][point1]
    feature2 = data['features'][point2]
    similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    similarity = (similarity + 1) / 2
    return confidence_residual, similarity
def plot_similarity_vs_residual(similarities, residuals, output_path, method="max_logit"):
    plt.figure(figsize=(8, 6))
    plt.scatter(similarities, residuals, c='blue', alpha=0.5)
    plt.title(f"Similarity vs Confidence Residual {method}")
    plt.xlabel("Similarity")
    plt.ylabel("Confidence Residual")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
def main(args):
    load_data_root = args.data_dir
    in_logits, in_labels, in_features, last_layer_weights, last_layer_bias = read_data(load_data_root, True)
    in_set_length = in_features.shape[0]
    cali_indices = np.random.choice(in_set_length, args.cali_size, replace=False)
    cali_features = in_features[cali_indices, :]
    cali_logits = in_logits[cali_indices, :]
    in_scores_dict, in_residuals_dict = calculate_score_residual(
        in_logits, in_labels, in_features, last_layer_weights, last_layer_bias,
        clean_set_features=cali_features, clean_set_logits=cali_logits
    )
    data = {
        'logits': in_logits,
        'features': in_features,
        'labels': in_labels
    }

    for key in in_scores_dict:
        similarities = []
        confidence_residuals = []
        for _ in range(100000):
            i = random.randint(0, in_set_length - 1)
            j = random.randint(0, in_set_length - 1)
        
            confidence_residual, similarity = compare_two_points_confidence(in_scores_dict, data, i, j, key)
            similarities.append(similarity)
            confidence_residuals.append(confidence_residual)
        output_path = os.path.join(".", f"similarity_vs_confidence_residual_{key}.png")
        plot_similarity_vs_residual(similarities, confidence_residuals, output_path, key)
    print(f"Plot saved at {output_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", dest="data_dir", type=str,
        default=os.path.join(".", "collected_data"),
        help="Folder where the collected logits data are located."
    )
    parser.add_argument(
        "--cali_size", dest="cali_size", type=int,
        default=5,
        help="Calibration data size used to determine the OOD score hyperparams."
    )
    args = parser.parse_args()
    main(args)
    print("Completed.")