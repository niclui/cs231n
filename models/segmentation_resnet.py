import segmentation_models_pytorch as smp
from torch import nn
import torch
import torch.nn.functional as F
import torchvision

import pdb

#Updated U-Net
#Reference: https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py

class Conv(nn.Module):
    """(Conv2D -> BN -> ReLU)"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
          )     
        
    def forward(self,x):
        return self.conv(x)

class Bridge(nn.Module):
    """
    This is the middle layer of the UNet.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            Conv(in_channels, out_channels),
            Conv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)

class Up(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = Conv(in_channels, out_channels)
        self.conv_block_2 = Conv(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1)

    def forward(self, x):
        return self.conv(x)

class CLearn(nn.Module):

    def __init__(self, n_channels, feature_dim):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(32 * n_channels, 32 * n_channels, bias=False),
                                nn.BatchNorm1d(32 * n_channels),
                                nn.ReLU(inplace=True), 
                                nn.Linear(32 * n_channels, feature_dim, bias=True))


    def forward(self, x):
        x = self.g(x)
        return x

class CLUNet(nn.Module):

    DEPTH = 5

    def __init__(self, in_channels, n_classes, n_channels, pretraining= False):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        resnet = torchvision.models.resnet.resnet50(pretrained=True)

        #Freeze layers
        if pretraining:
            ct = 0
            for child in resnet.children():
                ct += 1
                if ct < 9:
                    for param in child.parameters():
                        param.requires_grad = False

        down_blocks = []
        up_blocks = []

        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)

        #Make it smaller
        del down_blocks[-1]
        self.down_blocks = nn.ModuleList(down_blocks)

        #pdb.set_trace()

        #PUT YOUR MULTI-TASK CODE HERE/REG CODE HERE
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.clearn = CLearn(32, feature_dim = 32)

        self.bridge = Bridge(32 * n_channels, 32 * n_channels)
        
        up_blocks.append(Up(32 * n_channels, 16 * n_channels))
        up_blocks.append(Up(16 * n_channels, 8 * n_channels))
        #up_blocks.append(Up(8 * n_channels, 4 * n_channels))

        up_blocks.append(Up(4 * n_channels + 2 * n_channels, 4 * n_channels, 8 * n_channels, 4 * n_channels))
        up_blocks.append(Up(2 * n_channels + in_channels, 2 * n_channels, 4 * n_channels, 2 * n_channels))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = Out(2 * n_channels, n_classes)

    def forward(self, x, aux = None, pretraining = False):

        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (CLUNet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        if aux == "simclr":
            aux_out = self.clearn(self.pooling(x).squeeze())
        elif aux == "reg":
            #ADD YOUR REG CODE HERE
            pass

        if pretraining == False:

            x = self.bridge(x)
            for i, block in enumerate(self.up_blocks, 1):
                key = f"layer_{CLUNet.DEPTH - 1 - i}"
                x = block(x, pre_pools[key])
            output_feature_map = x
            mask = self.out(x)
            del pre_pools
        
        if (aux is not None) & (pretraining == False):
            return mask, aux_out
        elif (aux is not None) & (pretraining == True):
            return aux_out
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