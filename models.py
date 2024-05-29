import torch 
from torch import nn, Tensor
from torchvision import models
from typing import Any, Callable, List, Optional, Type, Union
from einops.layers.torch import Rearrange



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock1x1(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = norm_layer(planes)


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        out += identity
        out = self.relu(out)

        return out


class CLREncoder(nn.Module):
    
    def __init__(self, arch="resnet50", out_dim=1024, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        base_model=arch
        out_dim=out_dim

        if arch=="resnet50":
            self.encoder = models.resnet50(num_classes=out_dim)
        elif arch=="resnet34":
            self.encoder = models.resnet34(num_classes=out_dim)
        else:
            self.encoder = models.resnet18(num_classes=out_dim)
        dim_mlp = self.encoder.fc.in_features

        # add mlp projection head
        self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.fc)

    def forward(self, x):
        return self.encoder(x)
    
    def __pretrained_weights__(self, checkpoint):
        state = torch.load(checkpoint)
        self.encoder.load_state_dict(state["state_dict"])


class CLRDecoder(nn.Module):
    def __init__(self, n_blocks: List[int] = [6, 2, 2, 2, 4],  *args, **kwargs) -> None:
        """Decoder for Contrastive Learning Representation 

        Args:
            n_blocks (List[int]): Number of residual blocks for each layer
        """

        super().__init__(*args, **kwargs)


        self.decoder = nn.Sequential(
            Rearrange("b (c h w) -> b c h w", c=1, h=32,w=32),
            conv1x1(1,16*64),
            nn.ReLU(inplace=True),
            self._make_layer(16*64, 16*64, n_blocks[0]),
            nn.PixelShuffle(2),
            self._make_layer(16*16, 16*16, n_blocks[1]),
            nn.PixelShuffle(2),
            self._make_layer(16*4, 16*4, n_blocks[2]),
            nn.PixelShuffle(2),
            self._make_layer(16, 16, n_blocks[3]),
            conv1x1(16,3),
            nn.ReLU(inplace=True),
            self._make_layer(3, 3, n_blocks[4]) )


    def _make_layer(self, in_planes, out_planes, num_blocks):
        layers = []
        if in_planes!=out_planes:
            layers.append(BasicBlock(in_planes,out_planes))
        for i in range(num_blocks):
            layers.append(BasicBlock(out_planes, out_planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)

    

class CLRAutoencoder(nn.Module):
    def __init__(self, checkpoint=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if checkpoint is not None:
            state = torch.load(checkpoint)
            arch = state["arch"]
            self.encoder = CLREncoder(arch=arch, out_dim=1024)
            self.encoder.load_state_dict(state["state_dict"])
            #self.encoder.__pretrained_weights__(checkpoint)
        else:
            self.encoder = CLREncoder(*args, **kwargs)
        
        self.decoder = CLRDecoder(*args, **kwargs)


    def forward(self, x):
        return self.decoder(self.encoder(x))
    
