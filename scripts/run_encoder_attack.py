import os

def run_attack(encoder_usage_info, downstream_dataset,encoder,clean_encoder='model_1000_nonorm.pth'):

    save_path = f'./output/{encoder_usage_info}/substitute_encoder'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cmd = f'nohup python3 -u encoder_datafree_attack.py \
    --results_dir {save_path} \
    --encoder_usage_info {encoder_usage_info} \
    --downstream_dataset {downstream_dataset} \
    --encoder {encoder} \
    > ./log/substitute_encoder/{encoder_usage_info}_{downstream_dataset}_{encoder}_stolen_encoder_tttttttt.log &'
    os.system(cmd)

run_attack('cifar10','stl10','simclr')
run_attack('cifar10','gtsrb','simclr')
run_attack('cifar10','svhn','simclr')

   