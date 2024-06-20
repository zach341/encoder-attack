import os
import argparse
import random

import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import load_victim
from datasets import get_dataset_evaluation
from evaluation import NeuralNet, test_robust
import os
import torch.utils.data.sampler as sp


def train(encoders, dataloader, optimizer_encoder, args, epoch):

    # align
    clone_encoder, target_encoder = encoders

    target_encoder.eval()

    clone_encoder.train()
    for module in clone_encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    total_loss = 0
    count = 0
    criterion = nn.MSELoss()
    # criterion = simlilary_loss()

    for i, (imgs,_) in enumerate(dataloader):
        img_clean = imgs.cuda()
        count += img_clean.shape[0]

        with torch.no_grad():
            target_feature = target_encoder(img_clean)
            target_feature = F.normalize(target_feature, dim=-1)

        clone_feature = clone_encoder(img_clean)
        clone_feature = F.normalize(clone_feature, dim=-1)

        loss_1 = criterion(clone_feature, target_feature)

        loss = loss_1

        total_loss += loss
        
        optimizer_encoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()

    print('Train Epoch: [{}/{}],loss:{:.6f}, count:{}'.format(epoch, args.epochs, total_loss, count))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune the encoder to get the backdoored encoder')
    parser.add_argument('--batch_size', default=128,type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate in SGD')
    parser.add_argument('--epochs', default=200, type=int, help='training epoch')

    parser.add_argument('--pretrained_encoder', default='', type=str, help='path to the clean encoder used to finetune the backdoored encoder')
    parser.add_argument('--encoder_usage_info', default='cifar10', type=str, help='used to locate target encoder usage info')
    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the substitute encoder')

    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--downstream_dataset', default='', type=str, help='downstream dataset')
    parser.add_argument('--encoder', default='', type=str, choices=['dino','simclr','BYOL','mocov3'],help='contrast training method')
    ## /data/ZC/Dataset/imagenet_2_class_prompt_2500
    ## /data/ZC/Dataset/train2500
    ## /data/ZC/Dataset/imagenet_2_pic2pic_2500
    parser.add_argument('--shadow_dataset_src', default='/data/ZC/Dataset/only_class_prompt_imagenet_2500', type=str, help='training data folder, the dataset use for substitute dataset')
    parser.add_argument('--sub_encoder', default='/data/ZC/encoder_attack/output/ssl_custom_encoder/simclr-only_prompt_imagenet_2500-32-ep=999.ckpt', type=str, help='ssl subsititute encoder')

    args = parser.parse_args()

    # Set the seed and determine the GPU
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    
    print(args)


    train_transform = transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.RandAugment(2, 14),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

    ])

    # Create the Pytorch Datasets, and  create the data loader for the training set 

    print('load dataset from:', args.shadow_dataset_src)
    shadow_data = torchvision.datasets.ImageFolder(args.src, train_transform)
    train_loader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    print('load encoder:', args.encoder)
    if args.encoder_usage_info == 'cifar10':
        if args.encoder == 'simclr':
            args.pretrained_encoder = '/data/ZC/encoder_attack/output/cifar10/clean_encoder/simclr-cifar10-b30xch14-ep=999.ckpt'
        elif args.encoder == 'dino':
            args.pretrained_encoder = '/data/ZC/encoder_attack/output/cifar10/clean_encoder/dino-cifar10-13wu9ixc-ep=999.ckpt'
        elif args.encoder == 'mocov3':
            args.pretrained_encoder = '/data/ZC/encoder_attack/output/cifar10/clean_encoder/mocov3-cifar10-3gpr99oc-ep=999.ckpt'
        elif args.encoder == 'BYOL':
            args.pretrained_encoder = '/data/ZC/encoder_attack/output/cifar10/clean_encoder/byol-cifar10-32brzx9a-ep=999.ckpt'
    elif args.encoder_usage_info == 'stl10':
        if args.encoder == 'simclr':
            args.pretrained_encoder = '/data/ZC/encoder_attack/output/stl10/clean_encoder/simclr-stl10-h97ei2jj-ep=968.ckpt'
        elif args.encoder == 'dino':
            args.pretrained_encoder = '/data/ZC/encoder_attack/output/stl10/clean_encoder/dino-stl10-pe7pjftu-ep=905.ckpt'
        elif args.encoder == 'mocov3':
            args.pretrained_encoder = '/data/ZC/encoder_attack/output/stl10/clean_encoder/mocov3-stl10-uhd52tqj-ep=907.ckpt'
        elif args.encoder == 'BYOL':
            args.pretrained_encoder = '/data/ZC/encoder_attack/output/stl10/clean_encoder/byol-stl10-q2qvttd0-ep=896.ckpt'
    print('load victim encoder: ', args.pretrained_encoder)
    clean_model = load_victim(args.pretrained_encoder, 'resnet18').cuda()

    print('load from self-supervision checkpoint: ', args.sub_encoder)
    substitute_model = load_victim(args.sub_encoder, 'resnet34').cuda()

    optimizer_encoder = torch.optim.Adam(substitute_model.parameters(), lr=args.lr)

    args.data_dir = f'./data/{args.downstream_dataset}/'
    _, test_asr_data = get_dataset_evaluation(args) # downstream_test_data

    if args.encoder_usage_info == 'cifar10':
        if args.downstream_dataset == 'stl10':
            dataset_length = 8000
            num_of_classes = 10
            target_classifier_path = '/data/ZC/encoder_attack/output/cifar10/downstream_classifier/{}/stl10_downstream_classifier_nonorm_{}.pth'.format(args.encoder,args.encoder)
        elif args.downstream_dataset == 'gtsrb':
            dataset_length = 12630
            num_of_classes = 43
            target_classifier_path = '/data/ZC/encoder_attack/output/cifar10/downstream_classifier/{}/gtsrb_downstream_classifier_nonorm_{}.pth'.format(args.encoder,args.encoder)
        elif args.downstream_dataset == 'svhn':
            dataset_length = 26032
            num_of_classes = 10
            target_classifier_path = '/data/ZC/encoder_attack/output/cifar10/downstream_classifier/{}/svhn_downstream_classifier_nonorm_{}.pth'.format(args.encoder,args.encoder)
    elif args.encoder_usage_info == 'stl10':
        if args.downstream_dataset == 'cifar10':
            dataset_length = 10000
            num_of_classes = 10
            target_classifier_path = '/data/ZC/encoder_attack/output/stl10/downstream_classifier/{}/cifar10_downstream_classifier_nonorm_{}.pth'.format(args.encoder,args.encoder)
        elif args.downstream_dataset == 'gtsrb':
            dataset_length = 12630
            num_of_classes = 43
            target_classifier_path = '/data/ZC/encoder_attack/output/stl10/downstream_classifier/{}/gtsrb_downstream_classifier_nonorm_{}.pth'.format(args.encoder,args.encoder)
        elif args.downstream_dataset == 'svhn':
            dataset_length = 26032
            num_of_classes = 10
            target_classifier_path = '/data/ZC/encoder_attack/output/stl10/downstream_classifier/{}/svhn_downstream_classifier_nonorm_{}.pth'.format(args.encoder,args.encoder)
    
    list = [i for i in range(0, dataset_length)]
    data_list = random.sample(list, 1024)
    test_loader_asr = DataLoader(test_asr_data, batch_size=args.batch_size, shuffle=False, sampler= sp.SubsetRandomSampler(data_list), num_workers=2, pin_memory=True)
    
    net_target = NeuralNet(512, [512, 256], num_of_classes).cuda()
    net_state = torch.load(target_classifier_path)
    net_target.load_state_dict(net_state['state_dict'])

    best_asr = -1
    
    for epoch in range(1, args.epochs + 1):
        print("=================================================")

        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            train([substitute_model, clean_model], train_loader,optimizer_encoder, args,epoch=epoch)
            test_asr = test_robust(clean_model, net_target, substitute_model, test_loader_asr, None, 'PGD')
        else:
            raise NotImplementedError()
        
        if test_asr > best_asr:
            best_asr = test_asr
            print('{{"metric": "Eval - {}", "value": {} }}'.format('best_asr', best_asr))
            torch.save({'epoch': epoch, 'state_dict': substitute_model.state_dict(), 'asr': test_asr}, args.results_dir + '/' + args.encoder + '/' +  args.downstream_dataset + '_substitute_encoder_sub_imagenetonlyprompt_norm_2500_1000.pth')