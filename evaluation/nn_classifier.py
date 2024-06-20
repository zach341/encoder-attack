import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from Loss import InfoNCE
from advertorch.attacks import PGDAttack, LinfBasicIterativeAttack, GradientSignAttack,CarliniWagnerL2Attack
import matplotlib.pyplot as plt
import os
from Loss import simlilary_loss


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out
    
class linear(nn.Module):
    def __init__(self, n_features, n_classes) -> None:
        super(linear, self).__init__()
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)

def create_torch_dataloader(feature_bank, label_bank, batch_size, shuffle=False, num_workers=2, pin_memory=True):
    # transform to torch tensor
    tensor_x, tensor_y = torch.Tensor(feature_bank), torch.Tensor(label_bank)

    dataloader = DataLoader(
        TensorDataset(tensor_x, tensor_y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader


def net_train(net, train_loader, optimizer, epoch, criterion):
    """Training"""
    net.train()
    overall_loss = 0.0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, label.long())

        loss.backward()
        optimizer.step()
        overall_loss += loss.item()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, overall_loss*train_loader.batch_size/len(train_loader.dataset)))


def net_test(net, test_loader, epoch, criterion, keyword='Accuracy'):
    ## 测试下游分类器的图像准确率
    """Testing"""
    net.eval()
    test_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = net(data)
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('{{"metric": "Eval - {}", "value": {}, "epoch": {}}}'.format(
        keyword, 100. * correct / len(test_loader.dataset), epoch))

    return test_acc

def predict_feature(net, data_loader):
    ## 提取图像特征
    net.eval()
    feature_bank, target_bank = [], []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target_bank.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        target_bank = torch.cat(target_bank, dim=0).contiguous()

    return feature_bank.cpu().detach().numpy(), target_bank.detach().numpy()

def predict_feature_adv(clone_encoder, test_loader, target_encoder, target=False, method='pgd'):
    ## 提取添加扰动后的图像特征
    cfgs = dict(test_step_size=2.0 / 255, test_epsilon = 8.0/255)

    clone_encoder.eval()
    if method == 'pgd':
        adversary = PGDAttack(
            clone_encoder,
            # loss_fn= InfoNCE(), # nn.CosineSimilarity(dim=0, eps=1e-6),
            # loss_fn=simlilary_loss(),
            eps=cfgs['test_epsilon'],
            nb_iter=20, eps_iter=cfgs['test_step_size'], clip_min=0.0, clip_max=1.0, targeted=False    
        )
    target_encoder.eval()
    total_L2_distance = 0
    feature_bank, target_bank = [], []
    for data in tqdm(test_loader, desc='Feature extracting'):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = target_encoder(inputs)
        outputs = F.normalize(outputs, dim = -1)
        
        if target:
            pass
        else:
            adv_input_ori = adversary.perturb(inputs, outputs)
            
            L2_distance = torch.norm(adv_input_ori - inputs).item()
            total_L2_distance += L2_distance
            with torch.no_grad():
                outputs = target_encoder(adv_input_ori)
                outputs = F.normalize(outputs, dim = -1)
                feature_bank.append(outputs)
                target_bank.append(labels)
    feature_bank = torch.cat(feature_bank, dim=0).contiguous()
    target_bank = torch.cat(target_bank, dim=0).contiguous()

    return feature_bank.cpu().detach().numpy(), target_bank.cpu().detach().numpy()

def test_robust(target_encoder, target_classifier, clone_encoder, test_loader, generator, method ='PGD'):
    ## 测试扰动后的图像准确率
    target_classifier.eval()
    target_encoder.eval()
    clone_encoder.eval()
    
    cfgs = dict(test_step_size = 2.0/255, test_epsilon = 10.0/255)
    sum = 0.0
    L2_distance_sum = 0.0
    correct = 0.0
    correct_ghost = 0.0
    if method == 'PGD':
        adversary = PGDAttack(
            clone_encoder,
            loss_fn=simlilary_loss(),
            eps=cfgs['test_epsilon'],
            nb_iter=20, eps_iter=cfgs['test_step_size'], clip_min=0, clip_max=1, targeted=False
        )
    elif method == 'FGSM':
        adversary = GradientSignAttack(
            clone_encoder,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=cfgs['test_epsilon'],
            targeted=False)
    elif method == 'BIM':
        adversary = LinfBasicIterativeAttack(
            clone_encoder,
            loss_fn=simlilary_loss(),
            eps=cfgs['test_epsilon'],
            nb_iter=120, eps_iter=cfgs['test_step_size'], clip_min=0.0, clip_max=1.0,
            targeted=False)
    elif method == 'CW':
        adversary = CarliniWagnerL2Attack(
            clone_encoder,
            num_classes=10,
            learning_rate=0.45,
            binary_search_steps=10,
            max_iterations=20,
            targeted=False)
        
    for data in test_loader:
        inputs, labels = data
        sum += inputs.size(0)
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.no_grad():
            outputs = target_encoder(inputs)
            outputs = F.normalize(outputs, dim = 1)

        outputs_t = target_classifier(outputs)
        _, t_label = torch.max(outputs_t.data, 1)
        t_label = t_label.detach().cpu().numpy()
        t_label = torch.from_numpy(t_label).cuda().long()
        idx = torch.where(t_label == labels)[0]
        correct += idx.shape[0]
        adv_input_ori = adversary.perturb(inputs[idx], outputs[idx])
        
        ## show adversarial example

        # plt.subplot(121)
        # plt.imshow(inputs[0,...].permute(1,2,0).detach().cpu())
        # plt.axis('off')
        # plt.subplot(122)
        # plt.imshow(adv_input_ori[0,...].permute(1,2,0).detach().cpu())
        # plt.axis('off')
        # save_path = 'output/cifar10/pic'
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # plt.savefig(os.path.join(save_path, 'ttt.png'), bbox_inches='tight',dpi=3000)

        L2_distance = (adv_input_ori - inputs[idx]).squeeze()
        L2_distance = (torch.linalg.norm(L2_distance.flatten(start_dim=1), dim=1)).data
        L2_distance_sum +=L2_distance.sum()
        
        with torch.no_grad():
            outputs = target_encoder(adv_input_ori)
            outputs = F.normalize(outputs, dim = 1)
        output_adv = target_classifier(outputs)
        _, label_adv = torch.max(output_adv.data, 1)
        label_adv = label_adv.detach().cpu().numpy()
        predicted = torch.from_numpy(label_adv).cuda().long()
        correct_ghost += (predicted != labels[idx]).sum()

    print('{{"metric": "Eval - {}", "value": {}, "epoch": {}, "L2_distance":{}, "val_acc":{} }}'.format(
        'asr', 100. * correct_ghost / correct, 0, L2_distance_sum/ correct, 100. * correct / sum))
    
    return 100. * correct_ghost / correct

