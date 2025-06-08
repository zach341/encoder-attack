import os
import argparse
import random

import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.sampler as sp

from datasets import get_dataset_evaluation
from models import load_victim
from evaluation import create_torch_dataloader, NeuralNet, net_train, net_test, predict_feature
from torchvision import datasets, transforms

# class Full_model(nn.Module):
#     def __init__(self, model, input_size, num_classes):
#         super(Full_model, self).__init__()
#         linear= nn.Linear(input_size, num_classes)
#         self.f = nn.Sequential(model, linear)
#     def forward(self, x):
#         out = self.f(x)
#         return out

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate the encoders')
    parser.add_argument('--downstream_dataset', default='', type=str, help='downstream dataset')
    parser.add_argument('--encoder', default='', type=str, help='path to the image encoder')
    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the downstream classfier')
    parser.add_argument('--encoder_usage_info', default='', type=str, help='encoder usage info')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--seed', default=300, type=int, help='seed')
    parser.add_argument('--nn_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--arch', default='resnet34', type=str, help='encoder arch')
    parser.add_argument('--method', default='', type=str, help='downsteam, full_finetune')

    args = parser.parse_args()

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args.data_dir = f'./data/{args.downstream_dataset}/'
    train_data, test_data_clean= get_dataset_evaluation(args)
        
    if args.downstream_dataset == 'stl10':
        dataset_length = 8000
    elif args.downstream_dataset == 'gtsrb':
        dataset_length = 12600
    elif args.downstream_dataset == 'svhn':
        dataset_length = 26032
    elif args.downstream_dataset == 'cifar10':
        dataset_length = 10000

    list = [i for i in range(0, dataset_length)]
    data_list = random.sample(list, 1024)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader_clean = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                      pin_memory=True)
    
    if args.encoder != '':
        print('Loaded from: {}'.format(args.encoder))
        checkpoint = torch.load(args.encoder)
        if args.encoder_usage_info in ['CLIP']:
            pass
        else:
            model = load_victim(args.encoder, args.arch).cuda()
            model.train()
    
    # Linear probing
    if args.method == 'downstream':
        if args.encoder_usage_info in ['CLIP']:
            feature_bank_training, label_bank_training = predict_feature(model.visual, train_loader)
            feature_bank_testing, label_bank_testing = predict_feature(model.visual, test_loader_clean)
        else:
            feature_bank_training, label_bank_training = predict_feature(model, train_loader)
            feature_bank_testing, label_bank_testing = predict_feature(model, test_loader_clean)

        nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size)
        nn_test_loader = create_torch_dataloader(feature_bank_testing, label_bank_testing, args.batch_size)

        input_size = feature_bank_training.shape[1] # 
    
    # full_finetune
    elif args.method == 'full_finetune':
        input_size = 512

    num_of_classes = len(train_data.classes)
    if args.method == 'downstream':
        net = NeuralNet(input_size, [512, 256], num_of_classes).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    elif args.method == 'full_finetune':
        model.fc = nn.Linear(input_size, num_of_classes).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, args.nn_epochs + 1):
        if args.method == 'downstream':
            net_train(net, nn_train_loader, optimizer, epoch, criterion) # 
            net_test(net, nn_test_loader, epoch, criterion, 'Accuracy')
        elif args.method == 'full_finetune':
            net_train(model, train_loader, optimizer, epoch, criterion)
            net_test(model, test_loader_clean, epoch, criterion, 'Accuracy')
    
    if args.method == 'downstream':
        torch.save({'epoch': epoch, 'state_dict': net.state_dict()}, args.results_dir + '/{}_downstream_classifier_nonorm_{}_robust_1.pth'.format(args.downstream_dataset, args.arch))
    elif args.method == 'full_finetune':
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, args.results_dir + '/{}_finetuning_nonorm_{}_byol.pth'.format(args.downstream_dataset, args.arch))