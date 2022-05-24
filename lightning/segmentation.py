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

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

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
        self.lr = params['lr']
        self.batch_size = params['batch_size']

        self.data_transforms = {
            "train": A.Compose([
                A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.OneOf([
                    A.RandomContrast(),
                    A.RandomGamma(),
                    A.RandomBrightness(),
                    ], p=0.2),
                ], p=1.0),
            "valid": A.Compose([
                A.Resize(*(C.IMAGE_SIZE, C.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST),
                ], p=1.0)
        }

#DEBUGGING STEPS - DISPLAY IMAGES########################################

    def show_img(self, img, mask=None):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        plt.imshow(img, cmap='bone')
        
        if mask is not None:
            plt.imshow(mask, alpha=0.5)
            handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
            labels = ["Large Bowel", "Small Bowel", "Stomach"]
            plt.legend(handles,labels)
        plt.axis('off')

    def plot_batch(self, imgs, msks, batch_idx, size=3):
        plt.figure(figsize=(5*5, 5))
        for idx in range(size):
            plt.subplot(1, 5, idx+1)
            img = imgs[idx,].permute((1, 2, 0)).cpu().numpy()*255.0
            img = img.astype('uint8')
            msk = msks[idx,].permute((1, 2, 0)).cpu().numpy()*255.0
            self.show_img(img, msk)
        plt.tight_layout()
        #plt.show()

        plt.savefig( './images/masks_'+ str(batch_idx) + '.png', bbox_inches='tight')

########################################################################

    def training_step(self, batch, batch_idx): #Batch of data from train dataloader passed here

        images, masks = map(list, zip(*batch))
        images = torch.stack(images)
        masks = torch.stack(masks)

        #Display images
        #self.plot_batch(images, masks, batch_idx, 5)

        logits_masks = self.model(images)
        loss = self.loss(logits_masks, masks)
        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx): #Called once for every batch

        images, masks = map(list, zip(*batch))
        images = torch.stack(images)
        masks = torch.stack(masks)

        logits_masks = self.model(images)
        loss = self.loss(logits_masks, masks)

        self.evaluator.process(batch, logits_masks)
        return loss

    def validation_epoch_end(self, outputs): #outputs are loss tensors from validation step
        avg_loss = torch.stack(outputs).mean()
        self.log("val_loss", avg_loss)
        metrics = self.evaluator.evaluate()
        self.evaluator.reset()
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
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
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50, eta_min=1e-6)
        return [optimizer]

    def train_dataloader(self): #Called during init
        dataset = SegmentationDataset(os.path.join(self.dataset_folder, 'train_dataset.csv'),
                                                split="train",
                                                augmentation=self.augmentation,
                                                image_size=224,
                                                pretrained=True)

        return DataLoader(dataset, shuffle=True, #For entire batch
                          batch_size=self.batch_size, num_workers=self.n_workers,
                          collate_fn=lambda x: x)

    def val_dataloader(self): #Called during init
        dataset = SegmentationDataset(os.path.join(self.dataset_folder, 'val_dataset.csv'),
                                                split="val",
                                                augmentation=self.augmentation,
                                                image_size=224,
                                                pretrained=True)

        return DataLoader(dataset, shuffle=False, num_workers = self.n_workers,
                batch_size=2, collate_fn=lambda x: x)

    def test_dataloader(self): #Called during init
        dataset = SegmentationDataset(os.path.join(self.dataset_folder, 'test_dataset.csv'),
                                                split="test",
                                                augmentation=self.augmentation,
                                                image_size=224,
                                                pretrained=True)
        
        return DataLoader(dataset, shuffle=False,
                batch_size=1, num_workers=self.n_workers, collate_fn=lambda x: x)

    def predict_dataloader(self): #Called during init
        dataset = SegmentationDataset(os.path.join(self.dataset_folder, 'test_dataset.csv'),
                                                split="test",
                                                augmentation='none',
                                                image_size=224,
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
