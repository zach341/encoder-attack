import os

def run_eval(encoder_usage_info, downstream_dataset, encoder, method, arch='densenet121'):
    save_path = f'./output/{encoder_usage_info}/downstream_classifier'
    
    cmd = f"nohup python3 -u training_downstream_classifier.py \
            --downstream_dataset {downstream_dataset} \
            --arch {arch}\
            --results_dir {save_path} \
            --encoder {encoder} \
            --encoder_usage_info {encoder_usage_info} \
            --method {method} \
            >./log/downstream_classifier/evaluation_{encoder_usage_info}_{downstream_dataset}_{arch}_{method}_robust_1.txt &"
 
    os.system(cmd)

run_eval('cifar10', 'gtsrb', '/data/ZC/encoder-attack/output/cifar10/clean_encoder/ACL_DS.pt', 'downstream','robust_resnet18')
run_eval('cifar10', 'stl10', '/data/ZC/encoder-attack/output/cifar10/clean_encoder/ACL_DS.pt', 'downstream','robust_resnet18')
run_eval('cifar10', 'svhn', '/data/ZC/encoder-attack/output/cifar10/clean_encoder/ACL_DS.pt',  'downstream','robust_resnet18')

