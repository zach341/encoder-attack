from .simclr_model import SimCLR
from .clip_model import CLIP
from .imagenet_model import ImageNetResNet
from .effcientnet_model import Effcientencoder
from .vit import vit_tiny,vit_small,vit_base
import torch

import os
from pathlib import Path
import torch.nn as nn
from .resnet import resnet18 as resnet18_1
from .resnet import resnet34 as resnet34_1
from .resnet import resnet50 as resnet50_1
from torchvision.models import mobilenet_v2, densenet121,vgg19_bn, shufflenet_v2_x1_0
import torch.nn.functional as F
import numpy as np
import copy

class modified_resnet34(nn.Module):
    def __init__(self, feature_dim):
        super(modified_resnet34, self).__init__()
        self.f = resnet34_1(num_classes=20)

        projection_model = nn.Sequential(nn.Linear(512, feature_dim, bias=False), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True), nn.Linear(feature_dim, feature_dim, bias=True))

        self.g = projection_model
    def forward(self, x):

        feature = self.f(x)
        out = self.g(feature)
        return F.normalize(out, dim=-1)
    
class modified_resnet50(nn.Module):
    def __init__(self,feature_dim):
        super(modified_resnet50, self).__init__()
        self.f = resnet50_1(num_classes=10)

        projection_model = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

        self.g = projection_model
    def forward(self, x):

        feature = self.f(x)
        out = self.g(feature)
        return F.normalize(out, dim=-1)

def load_victim(encoder_path, arch):
    if arch == 'resnet18':
        model = resnet18_1(num_classes=10)
    elif arch == 'resnet34':
        model = resnet34_1(num_classes=20)
    elif arch == 'modified_resnet50':
        model = modified_resnet50(feature_dim=512)
    elif arch == 'vit_base':
        model = vit_base()
    elif arch == 'vit_tiny':
        model = vit_tiny()
    elif arch == 'vit_small':
        model = vit_small()
    elif arch == "vgg19_bn":
        model = vgg19_bn()
    elif arch == "CLIP":
        model = CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
    elif arch == "eff":
        model = Effcientencoder()
    elif arch == "mobilenetv2":
        model = mobilenet_v2()
    elif arch.startswith("modified"):
        model = modified_resnet34(feature_dim=1024)
        # model = modified_resnet50()
    elif arch.startswith("densenet"):
        model = densenet121()
    elif arch.startswith("shufflenet"):
        model = shufflenet_v2_x1_0()
    elif arch == "robust_resnet18":
        model = resnet18_1(num_classes=10)
    else:
        pass
    if True:
        if arch.startswith("modified"):
            model.f.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)
        elif arch == "resnet18" or arch == "resnet34":
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                        bias=False)
        elif arch.startswith("mobile"):
            
            pass
        elif arch.startswith("vit"):
            pass
        elif arch.startswith("vgg"):
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif arch.startswith("densenet"):
            model.features[0] = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
            bias=False)
        elif arch.startswith("shufflenet"):
            model.fc = nn.Identity()
        elif arch.startswith("robust"):
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                        bias=False)
    if encoder_path:
        checkpoint = torch.load(encoder_path)
        state_dict = checkpoint['state_dict']
        new_ckpt = dict()
        for k, value in state_dict.items():
            if k.startswith('backbone'):
                new_ckpt[k.replace('backbone.', '')] = value
            elif k.startswith('classifier'):
                new_ckpt[k.replace('classifier', 'fc')] = value
            else:
                new_ckpt[k] = value
        if arch.startswith("modified"):
            model.f.load_state_dict(new_ckpt, strict=False)
        else:
            model.load_state_dict(new_ckpt, strict=False)
    if arch.startswith("modified"):
        model.f.fc = nn.Identity()
        model.f.maxpool = nn.Identity()
    elif arch.startswith("vit"):
        pass
    elif arch.startswith("vgg"):
        model.classifier = nn.Identity()
    elif arch.startswith("mobile"):
        model.classifier = nn.Identity()
    elif arch.startswith("densenet"):
        model.classifier = nn.Identity()
    elif arch.startswith("shufflenet"):
        pass
    else:
        # pass
        model.fc = nn.Identity()
        model.maxpool = nn.Identity()

    return model