import pytorch_lightning as pl
import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from ignite.metrics import Accuracy
import segmentation_models_pytorch as smp

from models import get_model
from eval import get_loss_fn, SegmentationEvaluator
from util import constants as C
from .logger import TFLogger
from data import SegmentationDemoDataset, SegmentationDataset
from torch.nn.functional import softmax

import pdb

class SegmentationTask(pl.LightningModule, TFLogger):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super().__init__() #Initialize parent classes (pl.LightningModule params)
        self.save_hyperparameters(params) #Save hyperparameters to experiment directory, pytorch lightning function
        self.model = get_model(params) #Instantiates model from model folder
        self.loss = get_loss_fn(params)
        self.dataset_folder = params['dataset_folder']
        self.evaluator = SegmentationEvaluator()
        self.augmentation = params['augmentation']
        self.n_workers = params['num_workers']

    def training_step(self, batch, batch_nb): #Batch of data from train dataloader passed here
        images, masks = map(list, zip(*batch))
        images = torch.stack(images)
        masks = torch.stack(masks)

        logits_masks = self.model.forward(images)
        loss = self.loss(logits_masks, masks)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_nb): #Called once for every batch

        images, masks = map(list, zip(*batch))
        images = torch.stack(images)
        masks = torch.stack(masks)

        logits_masks = self.model.forward(images)
        loss = self.loss(logits_masks, masks)

        self.evaluator.process(batch, logits_masks)
        return loss

    def validation_epoch_end(self, outputs): #outputs are loss tensors from validation step
        avg_loss = torch.stack(outputs).mean()
        self.log("val_loss", avg_loss)
        metrics = self.evaluator.evaluate()
        self.evaluator.reset()
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_nb):
<<<<<<< HEAD

=======
>>>>>>> 85eb3ab337959ee9c2a498c8cd4052b4e6f11821
        images, masks = map(list, zip(*batch))
        images = torch.stack(images)
        masks = torch.stack(masks)

        logits_masks = self.model.forward(images)
        loss = self.loss(logits_masks, masks)

        self.evaluator.process(batch, logits_masks)

    def test_epoch_end(self, outputs):
        metrics = self.evaluator.evaluate()
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx):
        images, masks = map(list, zip(*batch))
        images = torch.stack(images)
        masks = torch.stack(masks)

        logits_masks = self.model.forward(images)

        self.evaluator.process(batch, logits_masks)

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    def train_dataloader(self): #Called during init
        dataset = SegmentationDataset(os.path.join(self.dataset_folder, 'train_dataset.csv'),
                                                split="train",
                                                augmentation=self.augmentation,
                                                image_size=256,
                                                pretrained=True)
        
        return DataLoader(dataset, shuffle=True, #For entire batch
                          batch_size=2, num_workers=self.n_workers,
                          collate_fn=lambda x: x)

    def val_dataloader(self): #Called during init
        dataset = SegmentationDataset(os.path.join(self.dataset_folder, 'val_dataset.csv'),
                                                split="val",
                                                augmentation='none',
                                                image_size=256,
                                                pretrained=True)

        return DataLoader(dataset, shuffle=False, num_workers = self.n_workers,
                batch_size=2, collate_fn=lambda x: x)

    def test_dataloader(self): #Called during init
        dataset = SegmentationDataset(os.path.join(self.dataset_folder, 'test_dataset.csv'),
                                                split="test",
                                                augmentation='none',
                                                image_size=256,
                                                pretrained=True)
        
        return DataLoader(dataset, shuffle=False,
                batch_size=1, num_workers=self.n_workers, collate_fn=lambda x: x)

    def predict_dataloader(self): #Called during init
        dataset = SegmentationDataset(os.path.join(self.dataset_folder, 'test_dataset.csv'),
                                                split="test",
                                                augmentation='none',
                                                image_size=256,
                                                pretrained=True)
        return DataLoader(dataset, shuffle=False,
                batch_size=1, num_workers=self.n_workers, collate_fn=lambda x: x)

    #Process
    #1. Call Trainer.fit
    #2. Run validation step twice
    #3. Run validation epoch end as a dummy run
    #4. Call training step for all training batches, till end of epoch
    #5. Call validation step for all validation batches till validation batches exhausted
    #6. Compute  validation metric at validation epoch end, log it, save checkpoint if validation metrics improve
    #7. Return to training step 4 and continue

    #Test steps are only called when you call main.py
