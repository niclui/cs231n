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
        labels = labels.cpu().numpy()
    
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    N1 = len(labels)

    #dice_coeff_mat = compute_meandice(preds, labels)
    #dice_coeff = torch.mean(torch.mean(dice_coeff_mat, dim = 1)).item()

    #max_dist = np.sqrt(H**2 + W**2)
    #hausdorff = 0.0
    dice_coeff = 0.0
    num_obs = 0.0
    small_bowel_dc = 0.0
    large_bowel_dc = 0.0
    stomach_dc = 0.0

    for i1 in range(N1):
        N2, C, H, W = labels[i1].shape
        for i2 in range(N2):
            #hs = 0.0
            dc = 0.0

            for c in range (C):
                #hs += compute_hausdorff(preds[i1][i2, c], labels[i1][i2, c], max_dist)
                to_add = compute_dice(preds[i1][i2, c].cpu(), labels[i1][i2, c].cpu())
                dc += to_add
                small_bowel_dc += to_add*(c == 0)
                large_bowel_dc += to_add*(c == 1)
                stomach_dc += to_add*(c == 2)
                    
            #hausdorff += hs/C
            dice_coeff += dc/C
            num_obs += 1

    #hausdorff = hausdorff / N
    dice_coeff /= num_obs
    small_bowel_dc /= num_obs
    large_bowel_dc /= num_obs
    stomach_dc /= num_obs
    
    return {
        'dice': dice_coeff,
        'small_bowel': small_bowel_dc,
        'large_bowel': large_bowel_dc,
        'stomach': stomach_dc
        #'hausdorff': hausdorff,
        #'combined': 0.4 * dice_coeff + 0.6 * hausdorff
    }

if __name__ == '__main__':

    #Checking dice function
    preds = [torch.randint(low = 0, high = 2, size = (32, 3, 256, 256)) for i in range(20)]
    labels = [torch.randint(low = 0, high = 2, size = (32, 3, 256, 256)) for i in range(20)]

    print(get_metrics(preds, labels))
