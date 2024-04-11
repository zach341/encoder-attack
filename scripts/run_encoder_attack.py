import os

def run_finetune(encoder_usage_info, shadow_dataset, downstream_dataset,clean_encoder='model_1000.pth'):

    save_path = f'./output/{encoder_usage_info}/substitute_encoder'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cmd = f'nohup python3 -u encoder_datafree_attack.py \
    --results_dir {save_path}/ \
    --shadow_dataset {shadow_dataset} \
    --pretrained_encoder ./output/{encoder_usage_info}/clean_encoder/{clean_encoder} \
    --encoder_usage_info {encoder_usage_info} \
    --downstream_dataset {downstream_dataset} \
    > ./log/substitute_encoder/{encoder_usage_info}_{downstream_dataset}.log &'
    os.system(cmd)

run_finetune('cifar10', 'imagenet', 'stl10')
run_finetune('cifar10', 'imagenet', 'gtsrb')
run_finetune('cifar10', 'imagenet', 'svhn')
