############ train pretrain encoder (target encoder) and Downstream classifier ##############
## sole_leann
python scripts/run_training_downstream_classifier.py

############ encoder attack ##############
python scripts/run_encoder_attack.py ### contains different methods

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

## scripts: 存放执行py文件
