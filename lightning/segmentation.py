import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from ignite.metrics import Accuracy
import segmentation_models_pytorch as smp

from models import get_model
from eval import get_loss_fn, SegmentationEvaluator
from util import constants as C
from .logger import TFLogger
from data import SegmentationDemoDataset
from torch.nn.functional import softmax

import pdb

class SegmentationTask(pl.LightningModule, TFLogger):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super().__init__() #Initialize parent classes (pl.LightningModule params)
        self.save_hyperparameters(params) #Save hyperparameters to experiment directory, pytorch lightning function
        self.model = get_model(params) #Instantiates model from model folder
        self.loss = get_loss_fn(params)
        self.evaluator = SegmentationEvaluator()

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

        pdb.set_trace()

    def test_step(self, batch, batch_nb):
        preds = self.model.infer(batch)
        self.evaluator.process(batch, preds)

    def test_epoch_end(self, outputs):
        metrics = self.evaluator.evaluate()
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    def train_dataloader(self): #Called during init
        dataset = SegmentationDemoDataset() #For specific example
        return DataLoader(dataset, shuffle=True, #For entire batch
                          batch_size=2, num_workers=8,
                          collate_fn=lambda x: x)

    def val_dataloader(self): #Called during init
        dataset = SegmentationDemoDataset()

        return DataLoader(dataset, shuffle=False, #num_workers = 8,
                batch_size=2, collate_fn=lambda x: x)

    def test_dataloader(self): #Called during init
        dataset = SegmentationDemoDataset()
        return DataLoader(dataset, shuffle=False,
                batch_size=1, num_workers=8, collate_fn=lambda x: x)

    #Process
    #1. Call Trainer.fit
    #2. Run validation step twice
    #3. Run validation epoch end as a dummy run
    #4. Call training step for all training batches, till end of epoch
    #5. Call validation step for all validation batches till validation batches exhausted
    #6. Compute  validation metric at validation epoch end, log it, save checkpoint if validation metrics improve
    #7. Return to training step 4 and continue

    #Test steps are only called when you call main.py