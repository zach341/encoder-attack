import os

def run_eval(encoder_usage_info, downstream_dataset, encoder, arch='resnet18'):
    save_path = f'./output/{encoder_usage_info}/downstream_classifier'
    
    cmd = f"nohup python3 -u training_downstream_classifier.py \
            --downstream_dataset {downstream_dataset} \
            --arch {arch}\
            --results_dir {save_path} \
            --encoder {encoder} \
            --encoder_usage_info {encoder_usage_info} \
            >./log/downstream_classifier/evaluation_{encoder_usage_info}_{downstream_dataset}_sub.txt &"

    os.system(cmd)

run_eval('stl10', 'cifar10', '/data/ZC/encoder_attack/output/stl10/substitute_encoder/BYOL/cifar10_substitute_encoder_sub.pth', 'resnet34')
run_eval('stl10', 'gtsrb', '/data/ZC/encoder_attack/output/stl10/substitute_encoder/BYOL/gtsrb_substitute_encoder_sub.pth', 'resnet34')
run_eval('stl10', 'svhn', '/data/ZC/encoder_attack/output/stl10/substitute_encoder/BYOL/svhn_substitute_encoder_sub.pth',  'resnet34')

