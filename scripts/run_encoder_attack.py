import os

def run_attack(encoder_usage_info, shadow_dataset, downstream_dataset,encoder,clean_encoder='model_1000_nonorm.pth'):

    save_path = f'./output/{encoder_usage_info}/substitute_encoder'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cmd = f'nohup python3 -u encoder_datafree_attack.py \
    --results_dir {save_path} \
    --shadow_dataset {shadow_dataset} \
    --encoder_usage_info {encoder_usage_info} \
    --downstream_dataset {downstream_dataset} \
    --encoder {encoder} \
    > ./log/substitute_encoder/{encoder_usage_info}_{downstream_dataset}_{encoder}_imagenetonlyprompt_norm_2500_1000.log &'
    os.system(cmd)

run_attack('cifar10', 'imagenet', 'stl10','mocov3')
run_attack('cifar10', 'imagenet', 'gtsrb','mocov3')
run_attack('cifar10', 'imagenet', 'svhn','mocov3')

run_attack('cifar10', 'imagenet', 'stl10','simclr')
run_attack('cifar10', 'imagenet', 'gtsrb','simclr')
run_attack('cifar10', 'imagenet', 'svhn','simclr')

run_attack('cifar10', 'imagenet', 'stl10','dino')
run_attack('cifar10', 'imagenet', 'gtsrb','dino')
run_attack('cifar10', 'imagenet', 'svhn','dino')

run_attack('cifar10', 'imagenet', 'stl10','BYOL')
run_attack('cifar10', 'imagenet', 'gtsrb','BYOL')
run_attack('cifar10', 'imagenet', 'svhn','BYOL')

#run_attack('stl10', 'imagenet', 'cifar10', 'BYOL')
#run_attack('stl10', 'imagenet', 'gtsrb', 'BYOL')
#run_attack('stl10', 'imagenet', 'svhn', 'BYOL')
