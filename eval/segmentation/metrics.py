"""
Description: Code adapted from aicc-ognet-global/eval/metrics.py 
"""

import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.nn.functional import sigmoid, cross_entropy
from monai.metrics.utils import get_surface_distance, get_mask_edges
from monai.metrics import compute_meandice

import pdb

def compute_dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    if (im1.sum() + im2.sum()) == 0:
        return 1
    else:
        return 2. * intersection.sum() / (im1.sum() + im2.sum())

# https://www.kaggle.com/code/yiheng/50-times-faster-way-get-hausdorff-with-monai
def compute_hausdorff(pred, gt, max_dist):
    if np.all(pred == gt):
        return 0.0
    (edges_pred, edges_gt) = get_mask_edges(pred, gt)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 0.0
    dist = surface_distance.max()
    if dist > max_dist:
        return 1.0
    return dist / max_dist

def get_metrics(preds, labels):
    if isinstance(labels, torch.Tensor):
        labels_ar = labels.cpu().numpy()
    
    if isinstance(preds, torch.Tensor):
        preds_ar = preds.cpu().numpy()

    N, C, H, W = labels.shape
    
    #dice_coeff_mat = compute_meandice(preds, labels)
    #dice_coeff = torch.mean(torch.mean(dice_coeff_mat, dim = 1)).item()

    max_dist = np.sqrt(H**2 + W**2)
    hausdorff = 0.0
    dice_coeff = 0.0

    for i in range(N):
        hs = 0.0
        dc = 0.0
        for c in range (C):
            hs += compute_hausdorff(preds_ar[i, c], labels_ar[i, c], max_dist)
            dc += compute_dice(preds_ar[i, c], labels_ar[i, c])
        hausdorff += hs/C
        dice_coeff += dc/C

    hausdorff = hausdorff / N
    dice_coeff = dice_coeff / N
    
    return {
        'dice': dice_coeff,
        'hausdorff': hausdorff,
        'combined': 0.4 * dice_coeff + 0.6 * hausdorff
    }

if __name__ == '__main__':
    preds = torch.randint(low = 0, high = 2, size = (10, 3, 256, 256))
    labels = torch.randint(low = 0, high = 2, size = (10, 3, 256, 256))

    print(get_metrics(preds, labels))
