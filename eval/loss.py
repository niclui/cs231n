import torch
import argparse
import segmentation_models_pytorch as smp
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, truth):
        DiceLoss = smp.losses.DiceLoss(mode = 'multilabel', from_logits = True)
        BCELoss = smp.losses.SoftBCEWithLogitsLoss()
        return 0.4 * DiceLoss(pred, truth) + 0.6 * BCELoss(pred,truth)  

def get_loss_fn(loss_args):
    loss_args_ = loss_args
    if isinstance(loss_args, argparse.Namespace):
        loss_args_ = vars(loss_args)
    loss_fn = loss_args_.get("loss_fn")

    if loss_fn == "BCE":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_fn == "CE":
        return torch.nn.CrossEntropyLoss()
    elif loss_fn == 'DBE':
        return smp.losses.DiceLoss(mode = 'binary', from_logits = True)
    elif loss_fn == 'DLE':
        return smp.losses.DiceLoss(mode = 'multilabel', from_logits = True)
    elif loss_fn == 'Combined':
        return CombinedLoss()
    else:
        raise ValueError(f"loss_fn {loss_args.loss_fn} not supported.")
