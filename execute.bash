############ train pretrain encoder (target encoder) ##############
python scripts/run_pretraining_encoder.py

############ encoder attack ##############
python scripts/run_encoder_attack.py ### contains different methods

## 不执行训练 直接对白盒模型进行攻击对应 Table.1 
# test_asr = test_robust(clean_model.f, net_target, clean_model.f, test_loader_asr, None, 'pgd')

## train_stolen对应 Table.2
# train_stolen(substitute_model.f, clean_model.f, optimizer_encoder, args, train_loader)

## train_dast, train_kd_TEDF, train_perturb, train_DFMS... 对应 Table.3 (需要设置对应生成器与配置)

############## train downstream classifier ##############
python scripts/run_cifar10_training_downstream_classifier.py





##################################### 文件夹说明 #####################################

## data: 存放训练数据与测试数据
## dataset: 数据集处理相关代码
## evaluation: 存放测试准确率，攻击成功率相关代码
## log: 
    # clean_encoder: 存放预训练encoder的log
    # downstream_encoder: 存放训练下游分类器的log
    # substitute_encoder: 存放进行encoder攻击时的log

## models： 存放用到的一些模型架构
## outputs：存放对应目标信息下训练的预训练编码器，下游分类器，替代编码器权重，以及存放对抗样本可视化图像

## saved_model_only_for_DFMS: 存放了经过cifar100随机10个类训练过后的生成器与鉴别器
## scripts: 存放执行py文件
