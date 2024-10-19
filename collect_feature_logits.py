"""
    This script presents a demo to collect necessary components:
        logits, features, linear weights and biases from pretrained models
    to enable SC evaluation. 

    This demo script uses:
        1) ImageNet-2012 val set
        2) timm (pretrained models)
    for data collection.

    One can modify the related part and adjust to other datasets/models.
"""

import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)

# === Env Imports ===
import argparse, torch, torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import timm
import numpy as np

# Some not so important params
VAL_TRANSFORM_TIMM = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
BATCH_SIZE = 128
NUM_WORKER = 4


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float


    # Directory to save the collected data
    save_root = os.path.join(".", "collected_data")
    os.makedirs(save_root, exist_ok=True)
    
    # ==== Change here if you want to experiment with a different dataset ===
    # Create ImageNet DataLoader
    dataset_val = torchvision.datasets.ImageNet(
        args.dataset_dir,
        split="val",
        transform=VAL_TRANSFORM_TIMM
    )

    loader_val = DataLoader(
        dataset_val, batch_size=BATCH_SIZE, 
        shuffle=False, pin_memory=False,
        num_workers=NUM_WORKER
    )
    # =======================================================================

    # ==== Change here if you want to experiment with a different pretrained mode ===
    # Load pretrained model
    model_name = "eva_giant_patch14_224.clip_ft_in1k"
    classifier_model = timm.create_model(
        model_name, pretrained=True
    )
    # Get the last linear layer weights and bias (for RL-geo-M computation)
    last_layer = classifier_model.head
    last_layer_weight, last_layer_bias = last_layer.weight.detach().cpu().numpy(), last_layer.bias.detach().cpu().numpy()
    # ===============================================================================
    
    classifier_model = classifier_model.to(device, dtype=dtype)
    classifier_model.eval()
    logits_log = []
    labels_log = []
    features_log = []

    # === Collect data ===
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader_val):
            print("Collecting batch {}".format(batch_idx+1))

            inputs = inputs.to(device, dtype=dtype)
            pred_logits = classifier_model(inputs)
            features = classifier_model.forward_head(classifier_model.forward_features(inputs), pre_logits=True)
            
            # Log result 
            logits_np = pred_logits.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            logits_log.append(logits_np)
            labels_log.append(labels_np)
            features_np = features.detach().cpu().numpy()
            features_log.append(features_np)

    # === Save collected data ===
    save_weight_name = os.path.join(save_root, "last_layer_weights.npy")
    save_bias_name = os.path.join(save_root, "last_layer_bias.npy")
    save_features_name = os.path.join(save_root, "features.npy")
    save_logits_name = os.path.join(save_root, "pred_logits.npy")
    save_labels_name = os.path.join(save_root, "labels.npy")
    np.save(save_weight_name, last_layer_weight)
    np.save(save_bias_name, last_layer_bias)
    np.save(save_features_name, np.concatenate(features_log, axis=0))
    np.save(save_logits_name, np.concatenate(logits_log, axis=0))
    np.save(save_labels_name, np.concatenate(labels_log, axis=0))


if __name__ == "__main__":

    print("Collecting logits (for future SC analysis).")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", dest="dataset_dir", type=str,
        default="/users/9/chen8596/sc_with_distshift",
        help="Path to the imagenet (val) set."
    )
    args = parser.parse_args()
    main(args)
    print("Completed.")