import segmentation_models_pytorch as smp
from torch import nn
import torch
import torch.nn.functional as F

import pdb

#Updated U-Net

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
          )     
        
    def forward(self,x):
        return self.double_conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)

    
class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x       

class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)

class CLearn(nn.Module):
    def __init__(self, in_channels, n_channels, feature_dim):
        super().__init__()
        self.compress= Down(in_channels, 4 * n_channels)
        self.g = nn.Sequential(nn.Linear(1728, 512, bias=False),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True), 
                                nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x1 = self.compress(x)
        x2 = torch.flatten(x1, start_dim = 1)
        x3 = self.g(x2)
        return x3

class Regression(nn.Module):
    def __init__(self, in_channels, n_channels):
        super().__init__()
        self.compress= Down(in_channels, 4 * n_channels)
        self.g = nn.Sequential(nn.Linear(1728, 512, bias=False),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True), 
                                nn.Linear(512, 128, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 1, bias=True))

    def forward(self, x):
        x1 = self.compress(x)
        x2 = torch.flatten(x1, start_dim = 1)
        x3 = self.g(x2)
        return x3

class Multitask(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 16 * n_channels)
        self.enc5 = Down(16 * n_channels, 16 * n_channels)

        self.regress = Regression(16 * n_channels, n_channels)
        
        self.dec1 = Up(32 * n_channels, 8 * n_channels)
        self.dec2 = Up(16 * n_channels, 4 * n_channels)
        self.dec3 = Up(8 * n_channels, 2 * n_channels)
        self.dec4 = Up(4 * n_channels, n_channels)
        self.dec5 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x, train = False):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        x6 = self.enc5(x5)

        regress_out = self.regress(x6)

        mask = self.dec1(x6, x5)
        mask = self.dec2(mask, x4)
        mask = self.dec3(mask, x3)
        mask = self.dec4(mask, x2)
        mask = self.dec5(mask, x1)
        mask = self.out(mask)
        
        return mask, regress_out

class CLUNet(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 16 * n_channels)
        self.enc5 = Down(16 * n_channels, 16 * n_channels)

        self.clearn = CLearn(16 * n_channels, n_channels, feature_dim = 128)
        
        self.dec1 = Up(32 * n_channels, 8 * n_channels)
        self.dec2 = Up(16 * n_channels, 4 * n_channels)
        self.dec3 = Up(8 * n_channels, 2 * n_channels)
        self.dec4 = Up(4 * n_channels, n_channels)
        self.dec5 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x, train = False):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        x6 = self.enc5(x5)

        if train == True:
            clearn_out = self.clearn(x6)

        mask = self.dec1(x6, x5)
        mask = self.dec2(mask, x4)
        mask = self.dec3(mask, x3)
        mask = self.dec4(mask, x2)
        mask = self.dec5(mask, x1)
        mask = self.out(mask)
        
        if train == True:
            return mask, clearn_out
        else:
            return mask

class SegmentationModel(nn.Module):
    """Segmentation model interface."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError('Subclass of PretrainedModel ' +
                                  'must implement forward method.')

class SMPModel(SegmentationModel):
    """
    PyTorch Segmentation models from
    https://github.com/qubvel/segmentation_models.pytorch
    """
    SMP_ARCHITECTURE_MAP = {
        "UNet": smp.Unet,
        "UNetPlusPlus": smp.UnetPlusPlus,
        "Linknet": smp.Linknet,
        "DeepLabV3": smp.DeepLabV3,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "FPN": smp.FPN,
        "PSPNet": smp.PSPNet,
    }

    def __init__(
            self,
            model_args=None):
        num_classes = model_args.get("num_classes", None)
        num_channels= model_args.get("num_channels", 3)
        
        encoder_name = model_args.get("encoder", None) 
        encoder_weights = model_args.get("pretrained", None)
        super().__init__()
        architecture = self.__class__.__name__
        if architecture not in self.SMP_ARCHITECTURE_MAP.keys():
            raise ValueError(
                f"Unknown architecture of SMPModel. Please choose from {list(SMP_ARCHITECTURE_MAP.keys())}.")
        _model_fn = self.SMP_ARCHITECTURE_MAP[architecture]
        self.model = _model_fn(encoder_name=encoder_name,
                               encoder_weights=encoder_weights,
                               in_channels=num_channels,
                               classes=num_classes) 
                               #decoder_channels = (1024, 512, 256, 128, 64))
        

    def forward(self, x): #override forward function 
        x = self.model(x)
        return x

class UNet(SMPModel):
    def __init__(self,  hyper_params):
        model_args = {'num_classes': 3,
                      'num_channels': 3,
                      'encoder': 'resnet50',
                      'pretrained': 'imagenet'}

        super().__init__(model_args = model_args)
        
class UNetPlusPlus(SMPModel):
    pass

class Linknet(SMPModel):
    pass

class DeepLabV3(SMPModel):
    def __init__(self,  hyper_params):
        model_args = {'num_classes': 3,
                        'num_channels': 3,
                        'encoder': 'resnet50',
                        'pretrained': 'imagenet'}

        super().__init__(model_args = model_args)


class DeepLabV3Plus(SMPModel):
    pass


class FPN(SMPModel):
    pass


class PSPNet(SMPModel):
    pass