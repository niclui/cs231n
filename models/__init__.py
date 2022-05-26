import argparse

from util import Args
from .classification import *
from .segmentation import *
from .detection import * #Import all files from the folder in the future so you can just do folder.method

import torch

def get_model(model_args):
    model_args_ = model_args

    if model_args_.get("model") == 'CLUNet':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = CLUNet(in_channels=3, n_classes=3, n_channels=48)
        model.to(device)
        return model

    elif model_args_.get("model") == 'Multitask':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Multitask(in_channels=3, n_classes=3, n_channels=48)
        model.to(device)
        return model

    else:
        if isinstance(model_args, argparse.Namespace):
            model_args_ = Args(vars(model_args))
        return globals().copy()[model_args_.get("model")](model_args_) 

    #OVERALL: get_model retains a pretrained or trained model architecture

    #globals is a dictionary, gets all the global variables (all the imports)
    #keys are the names of what is imported and values are the names of the functions
    #model_args_ are everything that went into the train, model is the name of the model
    #model_args_ are the params in train
    #"model" would call the specific model class, must match actual name in models/segmentation.py


