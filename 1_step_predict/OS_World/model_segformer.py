import torch
import torch.nn as nn
from torchvision import models as ML
import math
import copy
import numpy as np
import torch.nn.functional as F
# from KFBNet import KFB_VGG16
from torch.autograd import Variable
import torchvision.models as models
# from MSI_Model import MSINet
# from hrps_model import HpNet
# import hrnet
import pretrainedmodels
# from block import fusions
import argparse
from torchvision.models import resnet50, resnext50_32x4d, densenet121
import pretrainedmodels
from pretrainedmodels.models import *
# from models.segformer import SegFormer
import torch
import torch.nn as nn
import os
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
import time
from torch import nn, Tensor
from torch.nn import functional as F
from tabulate import tabulate

import torch
from torch import nn, Tensor
from torch.nn import functional as F
# from resnet import ResNet

import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F
from backbones import MiT, ResNet, PVTv2
from backbones.layers import trunc_normal_
from heads import SegFormerHead, FaPNHead
# from heads import SegFormerHead, FaPNHead, SFHead, UPerHead, FPNHead, FaPNCBAMHead
# from heads import SFHead
# from Deformable_ConvNet import DeformConv2D
# from baseline_models import FCN, deeplabv3
# from Unet import UNet
# class ConvModule(nn.Sequential):
#     def __init__(self, c1, c2, k, s=1, p=0):
#         super().__init__(
#             nn.Conv2d(c1, c2, k, s, p, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.ReLU(),
#         )


class PPM(nn.Module):
    """Pyramid Pooling Module in PSPNet
    """
    def __init__(self, c1, c2=128, scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(c1, c2, 1)
            )
        for scale in scales])

        self.bottleneck = ConvModule(c1 + c2 * len(scales), c2, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for stage in self.stages:
            outs.append(F.interpolate(stage(x), size=x.shape[-2:], mode='bilinear', align_corners=True))

        outs = [x] + outs[::-1]
        out = self.bottleneck(torch.cat(outs, dim=1))
        return out

class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out

segformer_settings = {
    'B0': 256,        # head_dim
    'B1': 256,
    'B2': 768,
    'B3': 768,
    'B4': 768,
    'B5': 768
}


class SegFormer(nn.Module):
    def __init__(self, variant: str = 'B0', num_classes: int = 19) -> None:
        super().__init__()
        self.backbone = MiT(variant)
        self.decode_head = SegFormerHead(self.backbone.embed_dims, segformer_settings[variant], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        # for i in y:
        #     print(i.shape)
        y = self.decode_head(y)   # 4x reduction in image size
        # y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y

class SegFormerB5(nn.Module):
    def __init__(self, variant: str = 'B5', num_classes: int = 19) -> None:
        super().__init__()
        self.backbone = MiT(variant)
        self.decode_head = SegFormerHead(self.backbone.embed_dims, segformer_settings[variant], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        # for i in y:
        #     print(i.shape)
        y = self.decode_head(y)   # 4x reduction in image size
        # y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y

class Segformer_baseline(nn.Module):
    def __init__(self, n_class):
        super(Segformer_baseline, self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B2', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('.\\segformer.b2.ade.pth', map_location='cpu'))
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))
        self.semantic_img_model.decode_head = SegFormerHead([64, 128, 320, 512], 768, self.n_class)
        # self.conv_block_fc = nn.Sequential(
        #     # FCViewer(),
        #     nn.Conv2d(150, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
        #     # nn.ReLU(inplace=True)
        #     # # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        # )
    def forward(self, h_rs):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        out = self.semantic_img_model(h_rs)

        # out = self.conv_block_fc(features)
        # print(out.shape)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out

class Segformer_FaPN(nn.Module):
    def __init__(self, n_class):
        super(Segformer_FaPN, self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B2', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('.\\segformer.b2.ade.pth', map_location='cpu'))
        self.semantic_img_model.decode_head = FaPNHead([64, 128, 320, 512], 128, self.n_class)
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        # self.conv_block_fc = nn.Sequential(
        #     # FCViewer(),
        #     nn.Conv2d(150, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
        #     # nn.ReLU(inplace=True)
        #     # # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        # )
    def forward(self, h_rs):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        out = self.semantic_img_model(h_rs)

        # out = self.conv_block_fc(features)
        # print(out.shape)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out

if __name__ == "__main__":
    # model = SegFormerB5()
    model = torch.load('.\\mit_b5_20220624-658746d9.pth', map_location='cpu')
    print('model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print(model)
