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
from evaluation import NeuralNet, test_robust, test_robust_finetune
import os
import torch.utils.data.sampler as sp
from replay import init_replay_memory
from advertorch.attacks import GradientSignAttack
from tqdm import tqdm

class StolenEncoderTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            #transforms.Resize(32),
            transforms.ToTensor()])
        self.aug_transform = transforms.Compose([
            # transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])

    def __call__(self, x):
        raw_img = self.transform(x)
        views = [self.aug_transform(x) for _ in range(9)]
        return [raw_img, views]
    
class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.data_transforms = transforms.Compose([
                                            #   transforms.Resize((self.img_size, self.img_size)),
                                            #   # transforms.CenterCrop(224),
                                            #   transforms.RandomHorizontalFlip(),
                                            #   transforms.RandomApply([color_jitter], p=0.8),
                                            #   transforms.RandomGrayscale(p=0.2),
                                            #   transforms.ToTensor(),
                                              # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                                            transforms.Resize((32,32)),
                                            transforms.RandAugment(2, 14),
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                    ])
        self.data_transforms = self.transform
    def __call__(self, x):
        return [self.transform(x), self.data_transforms(x)]
    
def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class Full_model(nn.Module):
    def __init__(self, model, input_size, num_classes):
        super(Full_model, self).__init__()
        linear= nn.Linear(input_size, num_classes)
        self.f = nn.Sequential(model, linear)
    def forward(self, x):
        out = self.f(x)
        return out
    
def train_dast(teacher, student, optimizer_encoder, optimizer_generator,generator, args):
    """
    生成器与替代encoder进行对抗训练，生成器最小化合成样本余弦相似度，替代encoder最大化余弦相似度
    """
    student.train()
    teacher.eval()
    for module in student.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    criterion = nn.MSELoss()
    loss_G_all = 0 
    loss_D_all = 0
    for ii in range(500):
        ## update student network
        z = torch.randn(args.batch_size, 256).cuda()
        data = generator(z)

        with torch.no_grad():
            target_feature = teacher(data)
            target_feature = F.normalize(target_feature, dim=-1)

        clone_feature = student(data.detach())
        clone_feature = F.normalize(clone_feature, dim=-1)

        loss_D = criterion(clone_feature, target_feature)
        optimizer_encoder.zero_grad()
        loss_D.backward()
        loss_D_all += loss_D
        optimizer_encoder.step()
        ## update generator network
        clone_feature = student(data)
        clone_feature = F.normalize(clone_feature, dim=-1)
        loss_G = -criterion(clone_feature, target_feature)
        optimizer_generator.zero_grad()
        loss_G.backward()
        loss_G_all += loss_G
        optimizer_generator.step()

    print('Train Epoch: [{}]/[{}], loss_G:{:.6f}, loss_encoder:{:.6f}'.format(epoch, args.epochs, loss_G_all, loss_D_all))

def train_stolen_encoder(clean_encoder, clone_encoder, optimizer_encoder, dataloader,args):

    clean_model.eval()
    clone_encoder.train()
    for module in clone_encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    total_loss = 0
    total_num = 0
    for img_raw, views in tqdm(dataloader):
        img_raw = img_raw.cuda(non_blocking=True)
        for i in range(len(views)):
            views[i] = views[i].cuda(non_blocking=True)
            views[i] = F.normalize(clone_encoder(views[i]), dim=-1) # 每个增强的特征
        
        clone_feature = clone_encoder(img_raw) 
        clone_feature = F.normalize(clone_feature, dim=-1)
        with torch.no_grad():
            victim_feature = clean_encoder(img_raw)
            victim_feature = F.normalize(victim_feature, dim=-1)

        loss1 = loss2 = 0
        for i in range(len(clone_feature)):
            loss1 += torch.dist(clone_feature[i], victim_feature[i], 2) # 求l2距离
            loss2 += sum([torch.dist(f[i], victim_feature[i], 2) for f in views])# 9个增强特征l2距离之和

        loss = loss1 + ((9*loss2)/9)

        optimizer_encoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()

        total_num += img_raw.size(0)
        total_loss += loss.item() * img_raw.size(0)
        print('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}'.format(epoch, args.epochs, total_loss / total_num))

    return total_loss / total_num

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

    adversary = GradientSignAttack(
            clone_encoder,
            loss_fn=nn.MSELoss(),
            eps=8.0/255, targeted=False)

    total_loss = 0
    count = 0
    criterion = nn.MSELoss()
    # criterion = simlilary_loss()

    for i, (imgs,_) in enumerate(dataloader):
        img_clean = imgs.cuda()
        count += img_clean.shape[0]
        clone_feature = clone_encoder(img_clean)
        clone_feature = F.normalize(clone_feature, dim=-1)

        img_adv = adversary.perturb(img_clean, clone_feature)

        with torch.no_grad():
            target_feature_adv = target_encoder.eval()(img_adv, 'normal')
            target_feature_adv = F.normalize(target_feature_adv, dim=-1)
            replay_memory.update(img_adv.cpu(), target_feature_adv.cpu())
        clone_feature_adv = clone_encoder(img_adv)
        clone_feature_adv = F.normalize(clone_feature_adv, dim=-1)
        
        loss_1 = criterion(clone_feature_adv, target_feature_adv)
        loss = loss_1
        total_loss += loss
        optimizer_encoder.zero_grad()
        loss.backward()

        optimizer_encoder.step()
    
    print('Train Epoch: [{}/{}],loss:{:.6f}, count:{}'.format(epoch, args.epochs, total_loss, count))

def train_replay(clone_encoder,optimizer_encoder, args, rep_iter):

    clone_encoder.train()
    for module in clone_encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    total_replay_loss = 0
    count = 0
    criterion = nn.MSELoss()

    data, t_embedding = replay_memory.sample()
    data.cuda()
    t_embedding.cuda()
    count += data.shape[0]
    clone_feature = clone_encoder(data)
    clone_feature = F.normalize(clone_feature, dim=-1)

    loss_1 = criterion(clone_feature, t_embedding)
    loss = loss_1
    total_replay_loss += loss
    
    optimizer_encoder.zero_grad()
    loss.backward()
    optimizer_encoder.step()
    
    print('Train iter: [{}/{}],replay_loss:{:.6f}, count:{}'.format(rep_iter, args.rep_iters, total_replay_loss, count))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune the encoder to get the backdoored encoder')
    parser.add_argument('--batch_size', default=256,type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate in SGD')
    parser.add_argument('--epochs', default=200, type=int, help='training epoch')

    parser.add_argument('--pretrained_encoder', default='', type=str, help='path to the clean encoder used to finetune the backdoored encoder')
    parser.add_argument('--encoder_usage_info', default='cifar10', type=str, help='used to locate target encoder usage info')
    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the substitute encoder')

    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--downstream_dataset', default='', type=str, help='downstream dataset')
    parser.add_argument('--encoder', default='', type=str, choices=['dino','simclr','BYOL','mocov3'],help='contrast training method')

    parser.add_argument('--shadow_dataset_src', default='/data/ZC/Dataset/train2500', type=str, help='training data folder, the dataset use for substitute dataset')
    ## /data/ZC/Dataset/imagenet_2_class_prompt_2500
    ## /data/ZC/Dataset/train2500
    ## /data/ZC/Dataset/imagenet_2_pic2pic_2500
    parser.add_argument('--sub_encoder', default='/data/ZC/encoder-attack/output/ssl_custom_encoder/simclr-imagenet_real--2500-999-solo_pic_norm.ckpt', type=str, help='ssl subsititute encoder')

    parser.add_argument('--replay_size', default=125000, type=int, help='replay buffer size') # cifar10 1000000 tiny 500000
    parser.add_argument('--rep_iters', default=1000000, type=int, help='Number of consecutive times to sample from replay')
    parser.add_argument('--replay', default='Classic', type=str)

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
    shadow_data = torchvision.datasets.ImageFolder(args.shadow_dataset_src, train_transform)
    train_loader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    print('load encoder:', args.encoder)
    if args.encoder_usage_info == 'cifar10':
        if args.encoder == 'simclr':
            args.pretrained_encoder = '/data/ZC/encoder-attack/output/cifar10/clean_encoder/simclr-cifar10-b30xch14-ep=999.ckpt'
        elif args.encoder == 'dino':
            args.pretrained_encoder = '/data/ZC/encoder-attack/output/cifar10/clean_encoder/dino-cifar10-13wu9ixc-ep=999.ckpt'
        elif args.encoder == 'mocov3':
            args.pretrained_encoder = '/data/ZC/encoder-attack/output/cifar10/clean_encoder/mocov3-cifar10-3gpr99oc-ep=999.ckpt'
        elif args.encoder == 'BYOL':
            args.pretrained_encoder = '/data/ZC/encoder-attack/output/cifar10/clean_encoder/byol-cifar10-32brzx9a-ep=999.ckpt'
    elif args.encoder_usage_info == 'stl10':
        if args.encoder == 'simclr':
            args.pretrained_encoder = '/data/ZC/encoder-attack/output/stl10/clean_encoder/simclr-stl10-h97ei2jj-ep=968.ckpt'
        elif args.encoder == 'dino':
            args.pretrained_encoder = '/data/ZC/encoder-attack/output/stl10/clean_encoder/dino-stl10-pe7pjftu-ep=905.ckpt'
        elif args.encoder == 'mocov3':
            args.pretrained_encoder = '/data/ZC/encoder-attack/output/stl10/clean_encoder/mocov3-stl10-uhd52tqj-ep=907.ckpt'
        elif args.encoder == 'BYOL':
            args.pretrained_encoder = '/data/ZC/encoder-attack/output/stl10/clean_encoder/byol-stl10-q2qvttd0-ep=896.ckpt'
    
    # args.pretrained_encoder = '/data/ZC/encoder-attack/output/cifar10/clean_encoder/ACL_DS.pt'

    print('load victim encoder: ', args.pretrained_encoder)
    clean_model = load_victim(args.pretrained_encoder, 'resnet18').cuda()

    print('load from self-supervision checkpoint: ', args.sub_encoder)
    substitute_model = load_victim(None, 'resnet34').cuda()
    
    optimizer_encoder = torch.optim.Adam(substitute_model.parameters(), lr=args.lr)
    args.data_dir = f'./data/{args.downstream_dataset}/'
    memory_data, test_asr_data = get_dataset_evaluation(args) # downstream_test_data

    if args.encoder_usage_info == 'cifar10':
        if args.downstream_dataset == 'stl10':
            dataset_length = 8000
            num_of_classes = 10
            target_classifier_path = '/data/ZC/encoder-attack/output/cifar10/downstream_classifier/{}/stl10_downstream_classifier_nonorm_{}.pth'.format(args.encoder,args.encoder)
        elif args.downstream_dataset == 'gtsrb':
            dataset_length = 12630
            num_of_classes = 43
            target_classifier_path = '/data/ZC/encoder-attack/output/cifar10/downstream_classifier/{}/gtsrb_downstream_classifier_nonorm_{}.pth'.format(args.encoder,args.encoder)
        elif args.downstream_dataset == 'svhn':
            dataset_length = 26032
            num_of_classes = 10
            target_classifier_path = '/data/ZC/encoder-attack/output/cifar10/downstream_classifier/{}/svhn_downstream_classifier_nonorm_{}.pth'.format(args.encoder,args.encoder)
    elif args.encoder_usage_info == 'stl10':
        if args.downstream_dataset == 'cifar10':
            dataset_length = 10000
            num_of_classes = 10
            target_classifier_path = '/data/ZC/encoder-attack/output/stl10/downstream_classifier/{}/cifar10_downstream_classifier_nonorm_{}.pth'.format(args.encoder,args.encoder)
        elif args.downstream_dataset == 'gtsrb':
            dataset_length = 12630
            num_of_classes = 43
            target_classifier_path = '/data/ZC/encoder-attack/output/stl10/downstream_classifier/{}/gtsrb_downstream_classifier_nonorm_{}.pth'.format(args.encoder,args.encoder)
        elif args.downstream_dataset == 'svhn':
            dataset_length = 26032
            num_of_classes = 10
            target_classifier_path = '/data/ZC/encoder-attack/output/stl10/downstream_classifier/{}/svhn_downstream_classifier_nonorm_{}.pth'.format(args.encoder,args.encoder)
    
    list = [i for i in range(0, dataset_length)]
    data_list = random.sample(list, 1024)
    test_loader_asr = DataLoader(test_asr_data, batch_size=args.batch_size, shuffle=False, sampler= sp.SubsetRandomSampler(data_list), num_workers=2, pin_memory=True)
    
    net_target = NeuralNet(512, [512, 256], num_of_classes).cuda()
    net_state = torch.load(target_classifier_path)
    net_target.load_state_dict(net_state['state_dict'])
    
    ########################### full_finetuning #######################
    # net = load_victim(None, 'resnet18').cuda()
    # net.fc = nn.Linear(512, num_of_classes)
    # net.cuda()
    # net_state = torch.load(target_classifier_path)['state_dict']
    # net.load_state_dict(net_state)

    best_asr = -1

    replay_memory = init_replay_memory(args)
    
    ########################## train dast ###########################
    # generator = Generator_2(nz=256, ngf=64, img_size=32,nc=3).cuda()
    # optimizer_gan = torch.optim.Adam(generator.parameters(), lr=1e-4)
    # reset_model(generator)

    for epoch in range(1, args.epochs + 1):
        print("=================================================")
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            # train_dast(clean_model, substitute_model, optimizer_encoder, optimizer_gan, generator, args)
            # train_stolen_encoder(clean_model, substitute_model, optimizer_encoder, train_loader, args)
            train([substitute_model, clean_model], train_loader,optimizer_encoder, args , epoch=epoch)
            test_asr = test_robust(clean_model, net_target, substitute_model, test_loader_asr, 'PGD')
            # test_asr = test_robust_finetune(clean_model, net, substitute_model, test_loader_asr, 'PGD')

        else:
            raise NotImplementedError()
        
        if test_asr > best_asr:
            best_asr = test_asr
            print('{{"metric": "Eval - {}", "value": {} }}'.format('best_asr', best_asr))
            # torch.save({'epoch': epoch, 'state_dict': substitute_model.state_dict(), 'asr': test_asr}, args.results_dir + '/' + args.encoder + '/' +  args.downstream_dataset + '_substitute_encoder_testtest.pth')
    
    for rep_iter in range(args.rep_iters):

        train_replay(substitute_model,optimizer_encoder, args,rep_iter=rep_iter)
        if rep_iter % 20 == 0:
            test_asr = test_robust(clean_model, net_target, substitute_model, test_loader_asr, 'PGD')
            # test_asr = test_robust_finetune(clean_model, net, substitute_model, test_loader_asr,'PGD')
        
            if test_asr > best_asr:
                best_asr = test_asr
                print('{{"metric": "Eval - {}", "value": {} }}'.format('best_asr', best_asr))
    #             # torch.save({'epoch': epoch, 'state_dict': substitute_model.state_dict(), 'asr': test_asr}, args.results_dir + '/' + args.encoder + '/' +  args.downstream_dataset + '_substitute_encoder_sub_test_replay_2500_1000.pth')