from .simclr_model import SimCLR
from .clip_model import CLIP
from .imagenet_model import ImageNetResNet
from .vit_model import Vitencoder
from .effcientnet_model import Effcientencoder
import torch

import os
from pathlib import Path
import torch.nn as nn
from .resnet import resnet18 as resnet18_1
from .resnet import resnet34 as resnet34_1

def get_encoder_architecture(args, arch):
    if args.pretraining_dataset == 'cifar10':
        if arch == 'Vit':
            print('subsititute encoder architecture: Vit')
            return Vitencoder()
        elif arch == 'Eff':
            print('subsititute encoder architecture: Eff')
            return Effcientencoder()
        else:
            return SimCLR(arch=arch)
    elif args.pretraining_dataset == 'stl10':
        return SimCLR(arch=arch)
    else:
        raise ValueError('Unknown pretraining dataset: {}'.format(args.pretraining_dataset))


def get_encoder_architecture_usage(args, arch):
    if args.encoder_usage_info == 'cifar10':
        if arch == 'Vit':
            print('subsititute encoder architecture: Vit')
            return Vitencoder()
        elif arch == 'Eff':
            print('subsititute encoder architecture: Eff')
            return Effcientencoder()
        else:
            return SimCLR(arch=arch)
    elif args.encoder_usage_info == 'stl10':
        if arch == 'Vit':
            print('subsititute encoder architecture: Vit')
            return Vitencoder()
        elif arch == 'Eff':
            print('subsititute encoder architecture: Eff')
            return Effcientencoder()
        return SimCLR(arch=arch)
    elif args.encoder_usage_info == 'imagenet':
        return ImageNetResNet()
    elif args.encoder_usage_info == 'CLIP':
        return CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
    elif args.encoder_usage_info == 'DINO':
        pass
    elif args.encoder_usage_info == 'MAE':
        pass
    else:
        raise ValueError('Unknown pretraining dataset: {}'.format(args.pretraining_dataset))
    

def get_encoder_architecture_ensemble(args, arch):

    model_1 = SimCLR(arch=arch)
    model_2 = Vitencoder()
    model_3 = Effcientencoder()

    model_pool = [model_1, model_2, model_3]

    return model_pool


def load_victim(encoder_path, arch):

    # if args.pre_dataset == 'cifar10':
    num_classes = 10
    # victim_path = os.path.join('victims', 'cifar10', str(args.victim))
    # encoder_path = [Path(victim_path) / ckpt for ckpt in os.listdir(Path(victim_path)) if ckpt.endswith(".ckpt")][0]
    if arch == 'resnet18':
        model = resnet18_1(num_classes=10)
    elif arch == 'resnet34':
        model = resnet34_1(num_classes=10)
    checkpoint = torch.load(encoder_path)
    state_dict = checkpoint['state_dict']
    # print('yyy: ', state_dict)
    new_ckpt = dict()
    for k, value in state_dict.items():
        if k.startswith('backbone'):
            new_ckpt[k.replace('backbone.', '')] = value
        elif k.startswith('classifier'):
            new_ckpt[k.replace('classifier', 'fc')] = value
        else:
            new_ckpt[k] = value
    if True:
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                        bias=False)
    # print(model)
    # print('yyy: ', new_ckpt.keys())
    model.load_state_dict(new_ckpt, strict=False)
    model.fc = nn.Identity()
    model.maxpool = nn.Identity()

    return model