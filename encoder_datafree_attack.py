import os
import argparse
import random

import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia import augmentation
import matplotlib.pyplot as plt
from auto_augment import AutoAugment

from models import get_encoder_architecture_usage, Generator_2, Generator_4, Discriminator, GeneratorResnet, Synthesizer
from datasets import get_shadow_dataset, get_dataset_evaluation
from evaluation import test, NeuralNet, test_robust, simlilary_loss
import os
from utils import MultiTransform, reset_model, TwoCropTransform, weights_init, filter_indices, generate_high,get_gaussian_kernel

from advertorch.attacks import PGDAttack
from Loss import ContrastiveLoss, InfoNCE, L_H, simlilary_loss
from models.adv_gan import Generator
import torch.utils.data.sampler as sp
from torch.autograd import Variable

def train_dast(student, teacher, optimizer_encoder, optimizer_generator,generator, args):
    ## 生成器与替代encoder进行对抗训练，生成器最小化合成样本余弦相似度，替代encoder最大化余弦相似度
    student.train()
    teacher.eval()
    for module in student.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    for ii in range(500):

        ## update student network
        student.zero_grad()
        z = torch.randn(args.batch_size, args.nz).cuda()
        data = generator(z)

        with torch.no_grad():
            target_feature = teacher(data)
            target_feature = F.normalize(target_feature, dim=-1)

        clone_feature = student(data.detach())
        clone_feature = F.normalize(clone_feature, dim=-1)

        loss_D = - torch.sum(clone_feature * target_feature, dim=-1).mean()
        loss_D.backward()
        optimizer_encoder.step()

        ## update generator network
        generator.zero_grad()
        clone_feature = student(data)
        clone_feature = F.normalize(clone_feature, dim=-1)
        loss_G = torch.sum(clone_feature * target_feature, dim=-1).mean()
        loss_G.backward()
        optimizer_generator.step()

    print('Train Epoch: [{}]/[{}], loss_G:{:.6f}, loss_encoder:{:.6f}'.format(epoch, args.epochs, loss_G, loss_D))

def train_kd_TEDF(synthesizer, models, optimizer):
    ## 取出提前合成的图像，编码器进行对齐训练
    target_encoder, clone_encoder = models
    clone_encoder.train()
    for module in clone_encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    target_encoder.eval()
    data = synthesizer.get_data()

    # aug = transforms.Compose([
    #                             transforms.RandomHorizontalFlip(),
    #                             transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    #                             transforms.RandomGrayscale(p=0.2),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    #                         ])

    for idx, (images) in enumerate(data):
        img = images.cuda()
    
        feature_shadow = clone_encoder(img)
        feature_shadow = F.normalize(feature_shadow, dim=-1)
        with torch.no_grad():
            feature_teacher = target_encoder(img)
            feature_teacher = F.normalize(feature_teacher, dim=-1)
        
        loss = - torch.sum(feature_shadow * feature_teacher, dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Train Epoch: [{}/{}], lr:{:.6f}, loss:{:.6f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], loss))
    
def train_DFMS_1(encoders, netG_D, optimizers, epoch, dataloader, args):
    ## 原DFMS代码
    clone_encoder, target_encoder = encoders
    optimizer_G, optimizer_encoder, optimizer_D = optimizers
    netG, netD = netG_D 

    criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0
    # trainloader_proxy = torch.utils.data.DataLoader(proxy_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # iter_proxy = iter(trainloader_proxy)
    loss_G_sum = 0
    loss_D_sum = 0
    loss_encoder_sum =0
    for i, (data,_) in enumerate(dataloader, 0):
        real_cpu = data[0].cuda()
        noise_data = torch.randn(args.batch_size, 100, 1, 1).cuda()
        netG.train()
        netG.zero_grad()

        gen_images = netG(noise_data)

        if args.auto_augment == True:
            pil_transform = transforms.Compose([
            transforms.ToPILImage(),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))]) 
            imgs = (gen_images * 0.5 + 0.5)
            for im in range(len(imgs)):
                imgs[im] = pil_transform(imgs[im])
        else:
            imgs = gen_images

        imgs = imgs.detach()

        target_encoder.eval()
        clone_encoder.train()

        for module in clone_encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()
        
        optimizer_encoder.zero_grad()

        with torch.no_grad():
            target_feature_shadow = target_encoder(imgs)
            target_feature_shadow = F.normalize(target_feature_shadow, dim=-1)
            
        clone_feature_shadow = clone_encoder(imgs)
        clone_feature_shadow = F.normalize(clone_feature_shadow, dim=-1)

        loss_encoder = - torch.sum(clone_feature_shadow * target_feature_shadow, dim=-1).mean()
        loss_encoder_sum += loss_encoder
        loss_encoder.backward()
        optimizer_encoder.step()

        if args.use_proxy_data == True and i % args.use_proxy_frequency == 0:
            optimizer_encoder.zero_grad()
            # inputs1, targets1 = iter_proxy.next()
            # inputs1, targets1 = inputs1.cuda(), targets1.cuda()
            img_real = real_cpu.detach()
            with torch.no_grad():
                target_feature_shadow = target_encoder(img_real)
                target_feature_shadow = F.normalize(target_feature_shadow, dim=-1)
            clone_feature_shadow = clone_encoder(img_real)
            clone_feature_shadow = F.normalize(clone_feature_shadow, dim=-1)
            loss_encoder = - torch.sum(clone_feature_shadow * target_feature_shadow, dim=-1).mean()
            loss_encoder_sum += loss_encoder
            loss_encoder.backward()
            optimizer_encoder.step()
        
        ## update G network maximize log(D(G(z)))
        netD.train()
        clone_encoder.eval()

        fake = gen_images
        clone_feature_shadow = clone_encoder(fake)
        clone_feature_shadow = F.normalize(clone_feature_shadow, dim=-1)
        with torch.no_grad():
            target_feature_shadow = target_encoder(fake)
            target_feature_shadow = F.normalize(target_feature_shadow, dim=-1)
        loss_div_G = - torch.sum(clone_feature_shadow * target_feature_shadow, dim=-1).mean()
        kl_loss_G = nn.KLDivLoss(reduction='sum')(clone_feature_shadow, target_feature_shadow)
        label = torch.full((fake.shape[0],), real_label, device='cuda')
        label = label.type(torch.cuda.FloatTensor)
        label.fill_(real_label)

        output = netD(fake)
        errG_adv = criterion(output, label)
        errG = errG_adv - loss_div_G
        # errG = errG_adv - kl_loss_G
        loss_G_sum += errG
        errG.backward()
        optimizer_G.step()

        ## Update D network
        netD.zero_grad()

        label = torch.full((real_cpu.size(0),), real_label, device='cuda')
        label = label.type(torch.cuda.FloatTensor)
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()

        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        loss_D_sum = loss_D_sum + errD_real + errD_fake
        optimizer_D.step()

    print('Train Epoch: [{}/{}], loss_G:{:.6f}, loss_D:{:.6f}, loss_encoder:{:.6f}'.format(epoch, args.epochs, loss_G_sum, loss_D_sum, loss_encoder_sum))

def train_perturb(encoders, dataloader, optimizer_encoder, args, epoch, method='pgd'):

    # 在正常图像进行对齐，自身对抗样本进行对齐，目标对抗样本进行对齐，对抗样本周围进行对齐
    clone_encoder, target_encoder = encoders

    aug = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.ToTensor(),
            # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    if method == 'pgd':
        adversary = PGDAttack(
            clone_encoder,
            loss_fn=simlilary_loss(),
            eps=8.0/255,
            nb_iter=20,
            eps_iter=2.0/255,
            clip_min=0.0,
            clip_max=1.0,
            targeted=False
        )

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
    for i, (imgs,_) in enumerate(dataloader):
        img_clean = imgs[0].cuda()
        img_aug = imgs[1].cuda()

        with torch.no_grad():
            target_feature = target_encoder(img_clean)
            target_feature = F.normalize(target_feature, dim=-1)

        clone_feature = clone_encoder(img_clean)
        clone_feature = F.normalize(clone_feature, dim=-1)

        adv_input_ori = adversary.perturb(img_clean, target_feature)
        adv_input_ori_aug = aug(adv_input_ori)
        with torch.no_grad():
            target_feature_adv = target_encoder(adv_input_ori)
            target_feature_adv = F.normalize(target_feature_adv, dim=-1)

            target_feature_aug = target_encoder(img_aug)
            target_feature_aug = F.normalize(target_feature_aug, dim=-1)

            target_feature_adv_aug = target_encoder(adv_input_ori_aug)
            target_feature_adv_aug = F.normalize(target_feature_adv_aug, dim=-1)
        
        clone_feature_adv = clone_encoder(adv_input_ori)
        clone_feature_adv = F.normalize(clone_feature_adv, dim=-1)

        clone_feature_aug = clone_encoder(img_aug)
        clone_feature_aug = F.normalize(clone_feature_aug, dim=-1)
        
        clone_feature_adv_aug = clone_encoder(adv_input_ori_aug)
        clone_feature_adv_aug = F.normalize(clone_feature_adv_aug, dim=-1)

        loss_1 = - torch.sum(clone_feature * target_feature, dim=-1).mean()
        loss_2 = - torch.sum(clone_feature_adv * target_feature_adv, dim=-1).mean()
        loss_3 = - torch.sum(clone_feature_aug * target_feature, dim=-1).mean()
        loss_4 = - torch.sum(clone_feature_adv_aug * target_feature_adv_aug, dim=-1).mean()

        loss = loss_1 + loss_2 + loss_3 + loss_4
        total_loss += loss
        optimizer_encoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()

    print('Train Epoch: [{}/{}],loss:{:.6f}'.format(epoch, args.epochs, loss))

def train_with_gan_1(train_loader, generator, encoders, optimizers, args):
    ### 使用生成器生成对抗样本，两个编码器关于对抗样本对齐
    
    clone_encoder, target_encoder = encoders
    G_optimizer,encoder_optimizers = optimizers
    target_encoder.eval()
    clone_encoder.train()
    generator.train()

    for module in clone_encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()
    
    criterion_l2 = nn.MSELoss()
    # criterion_constrastive = nn.CosineSimilarity(dim=0, eps=1e-6)
    # criterion_constrastive = simlilary_loss()
    criterion_constrastive = InfoNCE()
    # criterion_constrastive = nn.MSELoss()
    
    eps = 10/255
    G_loss_all = 0.0
    D_loss_all = 0.0
    
    for i, (imgs, _) in enumerate(train_loader):

        imgs = imgs[0].cuda()
        f_x = generator(imgs)
        f_x = torch.min(torch.max(f_x, imgs - eps), f_x + eps)
        f_x = torch.clamp(f_x, 0.0, 1.0)
        
        x = imgs

        reconstruction_loss = criterion_l2(f_x, x)
        ## 减小高频特征
        clean_hfc = generate_high(x, r=8)
        per_hfc = generate_high(f_x, r=8)
        HFC_loss = -criterion_l2(clean_hfc, per_hfc)

        ## 增大对抗损失
        with torch.no_grad():
            target_feature = target_encoder(f_x)
            target_feature = F.normalize(target_feature, dim=-1)

        ## 对抗损失 
        clone_feature = clone_encoder(f_x)
        clone_feature = F.normalize(clone_feature, dim=-1)
        adv_loss = - criterion_constrastive(clone_feature, target_feature).mean()

        G_loss = 5 * adv_loss + HFC_loss + reconstruction_loss
        G_loss_all += G_loss

        generator.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        ## encoder training
        clone_feature = clone_encoder(f_x.detach())
        clone_feature = F.normalize(clone_feature, dim=-1)

        # distribution matching
        D_loss = - torch.sum(clone_feature * target_feature, dim=-1).mean()
        D_loss_all += D_loss
        
        clone_encoder.zero_grad()
        D_loss.backward()
        encoder_optimizers.step()

    print('Train Epoch: [{}/{}],loss_g:{:.6f}, loss_d:{:.6f}'.format(epoch, args.epochs,G_loss_all, D_loss_all))

# def train(backdoored_encoder, clean_encoder, train_optimizer, args):
def train_stolen(backdoored_encoder, clean_encoder, train_optimizer, args, data_loader):
    ## 对2500张imagenet的原始版本和增强版本的图像，进行两个编码器余弦相似度的拉近
    
    backdoored_encoder.train()
    for module in backdoored_encoder.modules():
    # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    clean_encoder.eval()
    total_loss= 0

    for i, (imgs,_) in enumerate(data_loader):
        img_clean = imgs[0].cuda() # 原始图像
        img_aug = imgs[1].cuda() # 执行增强操作后的图像

        with torch.no_grad():
            clean_feature_raw = clean_encoder(img_clean)
            clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)

        feature_raw = backdoored_encoder(img_clean)
        feature_raw = F.normalize(feature_raw, dim=-1)
        
        feature_aug = backdoored_encoder(img_aug)
        feature_aug = F.normalize(feature_aug, dim=-1)

        loss_1 = - torch.sum(feature_raw * clean_feature_raw, dim=-1).mean() # 原始图像loss
        loss_2 = - torch.sum(feature_aug * clean_feature_raw, dim=-1).mean() # 增强图像loss

        loss = loss_1 + 20 * loss_2

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_loss += loss.item()
        print('Train Epoch: [{}/{}], lr:{:.6f}, loss:{:.6f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], loss))

    return 0 , 0 

def train_DFMS_2(netG, netD, optimizer, clone_encoder, target_encoder, optimizer_encoder, dataloader,args):
    ## 使用生成器，鉴别器进行训练，生成器增大MSE距离，且与鉴别器进行对抗训练，encoders拉近余弦相似度

    clone_encoder.train()
    for module in clone_encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    criterion = nn.BCELoss()

    real_label = 1
    fake_label = 0
    optimizerG, optimizerD = optimizer
    data_iter = iter(dataloader) ## 代理数据集

    for _ in range(100):
        # train with real
        try:
            img_real = next(data_iter)[0]
        except StopIteration:
            data_iter = iter(dataloader)
            img_real = next(data_iter)[0]
        img_real = img_real.cuda()
        netD.zero_grad()
        batch_size = img_real.size(0)
        if batch_size ==0:
            continue
        label = torch.full((batch_size,), real_label, device='cuda')
        label = label.type(torch.cuda.FloatTensor)
        output = netD(img_real)

        errD_real = criterion(output, label)
        errD_real.backward()
        
        # train with fake 
        noise = torch.randn(args.batch_size, 100, 1, 1, device='cuda')
        fake = netG(noise)
        
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)    
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        ## update G network
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake)
        errG_adv = criterion(output, label)

        clone_feature_shadow = clone_encoder(fake)
        clone_feature_shadow = F.normalize(clone_feature_shadow, dim=-1)

        with torch.no_grad():
            target_feature_shadow = target_encoder(fake)
            target_feature_shadow = F.normalize(target_feature_shadow, dim=-1)

        G_mse = nn.MSELoss()(clone_feature_shadow, target_feature_shadow)
        errG = errG_adv - G_mse
        errG.backward()
        optimizerG.step()

        ## update substitute encoder
        
        optimizer_encoder.zero_grad()
        fake = fake.clone().detach()
        clone_feature_shadow = clone_encoder(fake)
        clone_feature_shadow = F.normalize(clone_feature_shadow, dim=-1)

        with torch.no_grad():
            target_feature_shadow = target_encoder(fake)
            target_feature_shadow = F.normalize(target_feature_shadow, dim=-1)  

        loss_encoder_1 = - torch.sum(clone_feature_shadow * target_feature_shadow, dim=-1).mean()
        loss_encoder = loss_encoder_1
        
        loss_encoder.backward()
        optimizer_encoder.step()

    print('Train Epoch: [{}/{}], loss_encoder:{:.6f}, loss_G:{:.6f}, loss_D:{:6f}'.format(epoch, args.epochs, loss_encoder, errG, errD))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune the encoder to get the backdoored encoder')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in SGD')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the shadow dataset to inject the backdoor')

    parser.add_argument('--shadow_dataset', default='imagenet', type=str,  help='the dataset use for substitute dataset')
    parser.add_argument('--pretrained_encoder', default='', type=str, help='path to the clean encoder used to finetune the backdoored encoder')
    parser.add_argument('--encoder_usage_info', default='cifar10', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the substitute encoder')

    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--downstream_dataset', default='', type=str, help='downstream dataset')
    
    ## use for attack (new add)
    parser.add_argument('--src', default='/home/user1/ZC/Dataset/pack', type=str, help='imagenet folder')
    parser.add_argument('--nz', default=256, type=int, help='noise embeeding')
    parser.add_argument('--g_steps', default=5, type=int, help='generator iteration')
    parser.add_argument('--lr_g', default=1e-3, type=float, help='generator learning rate')
    parser.add_argument('--save_dir', default='run/cifar10', type=str)

    parser.add_argument('--proxy_ds_name', default='10_class', type=str, help='proxy dataset name')
    parser.add_argument('--auto_augment', default='False', type=str, help='auto_augment')
    parser.add_argument('--use_proxy_data', default='True', type=str, help='use proxy data')
    parser.add_argument('--use_proxy_frequency', default=10, type=int, help='use proxy data frequency')

    
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

    # Specify the pre-training data directory
    args.data_dir = f'./data/{args.downstream_dataset.split("_")[0]}/'
    args.knn_k = 200
    args.knn_t = 0.5
    # args.reference_label = 0
    print(args)

    scale_size = 32

    train_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    

    # Create the Pytorch Datasets, and create the data loader for the training set
    # memory_data, test_data_clean, and test_data_backdoor are used to monitor the finetuning process. They are not reqruied by our BadEncoder
    ### test dataset, downstream test dataset
    memory_data, test_data_clean = get_shadow_dataset(args)
    _, test_asr_data = get_dataset_evaluation(args) # downstream_test_data


    ## Imagenet dataset
    shadow_data = torchvision.datasets.ImageFolder(args.src, TwoCropTransform(train_transform, scale_size))
    
    train_loader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

 
    clean_model = get_encoder_architecture_usage(args, 'resnet18').cuda()
    substitute_model = get_encoder_architecture_usage(args, 'resnet34').cuda() # subsitutute encoder architecture

    # Create the extra data loaders for testing purpose and define the optimizer
    print("Optimizer: SGD")
    if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
        # note that the following three dataloaders are used to monitor the finetune of the pre-trained encoder, they are not required by our BadEncoder. They can be ignored if you do not need to monitor the finetune of the pre-trained encoder
        memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader_clean = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader_asr = DataLoader(test_asr_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        optimizer_encoder = torch.optim.SGD(substitute_model.f.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

    # Initialize the BadEncoder and load the pretrained encoder
    if args.pretrained_encoder != '':
        print(f'load the clean model from {args.pretrained_encoder}')
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            checkpoint = torch.load(args.pretrained_encoder)
            clean_model.load_state_dict(checkpoint['state_dict'])
        else:
            raise NotImplementedError()

    test_acc_1 = test(clean_model.f, memory_loader, test_loader_clean,0, args)
    test_acc_2 = test(substitute_model.f, memory_loader, test_loader_clean,0, args)
    print('target test acc: {}'.format(test_acc_1))
    print('initial test acc: {}'.format(test_acc_2))

    input_size = 512
    img_size = 32
    img_size2 = (3, 32, 32)
    nc = 3
    nz = args.nz
    num_class = 10
    # training loop

    ############################################## generator definition ############################################

    ## train_with_gan_1 generator
    generator = GeneratorResnet().cuda()
    # generator = Generator_2(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda()
    # generator = Generator(100, [1024, 512, 256], 3, args.batch_size).cuda()
    # generator = Generator_4(0).cuda()
    # generator.apply(weights_init)

    # state_dict_g = torch.load('/data2/ZC/Bad_encoder_copy_1/saved_model/netG_epoch_199.pth')
    # generator.load_state_dict(state_dict_g)

    ## train_with_gan_1 optimizer 
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5,0.999))
    # optimizer_generator = torch.optim.Adam(generator.parameters(), lr = 1e-3)
    
    # z = torch.randn(args.batch_size, 100).view(-1, 100, 1,1)
    # z = Variable(z.cuda())

    ############################################ synthesizer for TEDF ############################################
    # synthesizer = Synthesizer(generator,
    #                           nz=nz,
    #                           num_classes=num_class,
    #                           img_size=img_size2,
    #                           iterations=args.g_steps,
    #                           lr_g=args.lr_g,
    #                           sample_batch_size=args.batch_size,
    #                           save_dir=args.save_dir,
    #                           dataset=args.encoder_usage_info)

    ############################################ Gaussian smoothing #############################################
    
    # kernel_size = 3
    # pad = 2
    # sigma = 1
    # kernel = get_gaussian_kernel(kernel_size=kernel_size, pad=pad,sigma=sigma).cuda()

    ########################################### proxy dataset for DFMS ##########################################
    # transforms_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
    # ])

    # if args.proxy_ds_name == '10_class':
    #     classes_set = {'plate', 'rose', 'castle', 'keyboard', 'house', 'forest', 'road', 'television', 'bottle', 'wardrobe'}
    #     trainset = torchvision.datasets.CIFAR100(root='../Dataset', train=True, download=False, transform=transforms_train)
        
    #     print(trainset.class_to_idx)
    #     id_to_class_mapping = {}
    #     for cl, idx in trainset.class_to_idx.items():
    #         id_to_class_mapping[idx] = cl
    #     print(id_to_class_mapping)

    #     classes_indices = []
    #     for k in classes_set:
    #         classes_indices.append(trainset.class_to_idx[k])

    #     index_list = filter_indices(trainset, classes_indices)
    #     # index_list = filter_indices(trainset)
    #     dataset = torch.utils.data.Subset(trainset, index_list)
    #     print("代理数据集大小: ", len(dataset))
    
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    
    ################################################### Discriminator for DFMS ##############################
    # netD = Discriminator(0).cuda()
    # netD.apply(weights_init)
    # optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # state_dict_d = torch.load('/data2/ZC/Bad_encoder_copy_1/saved_model/netD_epoch_199.pth')
    # netD.load_state_dict(state_dict_d)
    
    ##########################################################################################################

    args.data_dir = f'./data/{args.downstream_dataset}/'
    test_data_clean = get_dataset_evaluation(args)
    if args.downstream_dataset == 'stl10':
        dataset_length = 8000
        num_of_classes = 10
        target_classifier_path = 'output/cifar10/downstream_classifier/stl10_downstream_classifier_clean.pth'
    elif args.downstream_dataset == 'gtsrb':
        dataset_length = 12630
        num_of_classes = 43
        target_classifier_path = 'output/cifar10/downstream_classifier/gtsrb_downstream_classifier_clean.pth'
    elif args.downstream_dataset == 'svhn':
        dataset_length = 26032
        num_of_classes = 10
        target_classifier_path = 'output/cifar10/downstream_classifier/svhn_downstream_classifier_clean.pth'
    
    list = [i for i in range(0, dataset_length)]
    data_list = random.sample(list, 1024)
    test_loader_asr = DataLoader(test_asr_data, batch_size=args.batch_size, shuffle=False, sampler= sp.SubsetRandomSampler(data_list), num_workers=2, pin_memory=True)
    
    net_target = NeuralNet(input_size, [512, 256], num_of_classes).cuda()
    net_state = torch.load(target_classifier_path)
    net_target.load_state_dict(net_state['state_dict'])

    best_acc = -1
    best_asr = -1
    
    for epoch in range(1, args.epochs + 1):
        print("=================================================") # 三阶段

        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            
            # synthesizer.gen_data_TEDF(model.f, clean_model.f)
            
            # train_kd_TEDF(synthesizer, [clean_model.f, model.f], optimizer_encoder)
            
            # train_dast(model.f, clean_model.f, optimizer_encoder, optimizer_generator,generator, args)
            
            # train_DFMS([model.f, clean_model.f], [generator, netD], [optimizer_generator, optimizer_encoder, optimizerD], epoch,train_loader,args)
            
            # train_DFMS_2(generator, netD, [optimizerG, optimizerD], model.f, clean_model.f, optimizer_encoder, dataloader, args)
            
            # train_perturb([model.f, clean_model.f], train_loader, optimizer_encoder, args, epoch, method='pgd')
            
            train_with_gan_1(train_loader, generator, [substitute_model.f, clean_model.f], [optimizer_generator, optimizer_encoder],args)
            
            # train_perturb([substitute_model.f, clean_model.f], train_loader,optimizer_encoder, args,epoch=epoch )
            
            test_acc = test(substitute_model.f, memory_loader, test_loader_clean ,epoch, args)
            test_asr = test_robust(clean_model.f, net_target, substitute_model.f, test_loader_asr, generator, 'generator')
        else:
            raise NotImplementedError()
        if epoch % args.epochs == 0:
            torch.save({'epoch': epoch, 'state_dict': substitute_model.state_dict(), 'optimizer' : optimizer_encoder.state_dict(),'asr': test_asr}, args.results_dir + '/' + args.downstream_dataset + '_substitute_encoder.pth')
