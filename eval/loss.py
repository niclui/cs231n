import torch
import argparse


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
        return 0.4 * smp.losses.DiceLoss(mode = 'multilabel', from_logits = True) + 0.6 * smp.losses.SoftBCEWithLogitsLoss()
    else:
        raise ValueError(f"loss_fn {loss_args.loss_fn} not supported.")
