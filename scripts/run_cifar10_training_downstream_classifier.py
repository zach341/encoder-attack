import os

if not os.path.exists('./log/cifar10'):
    os.makedirs('./log/cifar10')

def run_eval(encoder_usage_info, downstream_dataset, encoder, arch='resnet18'):
    save_path = f'./output/{encoder_usage_info}/downstream_classifier'
    
    cmd = f"nohup python3 -u training_downstream_classifier.py \
            --downstream_dataset {downstream_dataset} \
            --arch {arch}\
            --results_dir {save_path} \
            --encoder {encoder} \
            --encoder_usage_info {encoder_usage_info} \
            >./log/downstream_classifier/evaluation_{encoder_usage_info}_{downstream_dataset}.txt &"

    os.system(cmd)

run_eval('cifar10', 'stl10', 'output/cifar10/clean_encoder/model_1000.pth', 'resnet18')
run_eval('cifar10', 'gtsrb', 'output/cifar10/clean_encoder/model_1000.pth', 'resnet18')
run_eval('cifar10', 'svhn', 'output/cifar10/clean_encoder/model_1000.pth',  'resnet18')

