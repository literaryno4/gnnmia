import os.path as osp
import argparse
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, GCNConv  # noqa

from train import train_model
from models import GAT, GCN, GraphSage, GraphSageTopK, get_opts
from GnnAttack import GnnAttack
from utils import process_mask, format_model_name
print("prerequirement satisfied!")



def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer', 'PubMed'], help='dataset used to train target mode and shadow_model')
    parser.add_argument('--target_model', type=str, default='GCN', choices=['GCN', 'GraphSage', 'GAT'], help='target model of the attack')
    parser.add_argument('--shadow_model', type=str, default='GCN', choices=['GCN', 'GraphSage', 'GAT'], help='shadow model to imitate target model')
    parser.add_argument('--data_path', type=str, default='./data', help='data path of dataset')
    parser.add_argument('--log_dir', type=str, default='./gammia_mm_diff_target_log.txt', help='log file path')
    parser.add_argument('--epoches', type=int, default=200, help='epoches to train target model and shadow model')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='devic')
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
    parser.add_argument('--wandb', action='store_true', help='Track experiment')

    return parser.parse_args()

def attack_nodes_list(models, data):
    attack_in_list = []
    attack_out_list = []
    for i in range(dataset.num_classes):
        atk_in = []
        atk_out = []
        for model in models:
            model.eval()
            logits = model(data.x, data.edge_index, data.edge_attr).detach().cpu().numpy()
            for l in range(len(logits)):
                if data.y[l] == i and data.train_mask[l]:
                    atk_in.append(logits[l])
                if data.y[l] == i and data.test_mask[l]:
                    atk_out.append(logits[l])
        attack_in_list.append(atk_in)
        attack_out_list.append(atk_out)
    
    return attack_in_list, attack_out_list

def attack(model, model_name, attack_in_train_list, attack_out_train_list, attack_in_list, attack_out_list, dataset, verbose=2):
    best_test_acc_list = []
    data = dataset[0].to(device)
    for i in range(dataset.num_classes):
        optimizer = get_opts(model, model_name)
        gam = GnnAttack(data, attack_in_train_list[i], attack_out_train_list[i], attack_in_list[i], attack_out_list[i])
        best_test_acc_list.append(gam.attack(model, optimizer, device=device, epoches=1000, verbose=verbose))
    for i in range(len(best_test_acc_list)):
        print("class {}: {}".format(i, best_test_acc_list[i]))
        
    return best_test_acc_list

def write_log(best_test_acc_list, dataset_name, model_name, model_name_shadow):
    with open(log_dir, 'a') as f:
        f.write('{:.2f}/{:.2f}    '.format(np.mean(best_test_acc_list) * 100, np.max(best_test_acc_list) * 100))


log_title = '\n\n-------------------------------Cora-------------------  ----------------------CiteSeer--------------  --------------------PubMed-----------------\ntar.\\shad.      GCN         GraphSage         GAT            GCN         GraphSage        GAT            GCN          GraphSage        GAT  \n------------------------------------------------------  --------------------------------------------  -------------------------------------------'

def train_target_model(dataset, model_name):
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    model = model_dict[model_name](num_features, num_classes).to(device)
    data = dataset[0]
    data = data.to(device)
    optimizer = get_opts(model, model_name)
    best_model = train_model(model, optimizer, data)
    return best_model

def train_shadow_models(dataset, model_name):
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    data = dataset[0]
    data = process_mask(data).to(device)
    shadow_models = []
    for i in range(dataset.num_classes):
#             for i in range(50):
        model_shadow = model_dict[model_name](num_features, num_classes).to(device)
        optimizer = get_opts(model_shadow, model_name)
        shadow_models.append(train_model(model_shadow, optimizer, data))
    return shadow_models

if __name__ == '__main__':
    model_dict = {'GCN': GCN, 'GAT': GAT, 'GraphSage': GraphSage}
    args = argparser()
    device = args.device
    device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print("devece: {}".format(device))
    dataset_name = args.dataset
    target_model_name = args.target_model
    shadow_model_name = args.shadow_model
    path = args.data_path
    log_dir = args.log_dir

    print("dataset: {}, target_model: {}, shadow_model: {}".format(dataset_name, target_model_name, shadow_model_name))

    with open(log_dir, 'a') as f:
        f.write(log_title)

    dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())

    # train target model
    best_model = train_target_model(dataset, target_model_name)
    # take logits of target model as test set of attack model
    attack_in_list, attack_out_list = attack_nodes_list([best_model], dataset[0])

    # train target model
    shadow_models = train_shadow_models(dataset, shadow_model_name)
    # take logits of target model as test set of attack model
    attack_in_train_list, attack_out_train_list = attack_nodes_list(shadow_models, dataset[0])

    # construct graph dataset and do attack
    num_features = len(attack_in_train_list[0][0])
    num_classes = 2

    attack_model = GraphSage(num_features, num_classes).to(device)
    best_test_acc_list = attack(attack_model, 'GraphSage', attack_in_train_list, attack_out_train_list, attack_in_list, attack_out_list, dataset, verbose=2)
    write_log(best_test_acc_list, dataset_name, target_model_name, shadow_model_name)

    #attack_model = GraphSageTopK(num_features, num_classes).to(device)
    #best_test_acc_list = attack(attack_model, 'GraphSage', attack_in_train_list, attack_out_train_list, attack_in_list, attack_out_list, dataset, verbose=2)
    #write_log(best_test_acc_list, dataset_name, target_model_name, shadow_model_name)

    #for model_name in model_names:
    #    with open(log_dir, 'a') as f:
    #        f.write('\n{}'.format(format_model_name(model_name)))
    #    for dataset_name in dataset_names:
    #        best_model = train_target_model(dataset_name, model_name)
    #        attack_in_list, attack_out_list = attack_nodes_list([best_model], data)
    #        for model_name_shadow in model_names:
    #            print("dataset: {}, target_model: {}, shadow_model: {}".format(dataset_name, model_name, model_name_shadow))
    #            shadow_models = train_shadow_models(dataset_name, model_name_shadow)
    #            attack_in_train_list, attack_out_train_list = attack_nodes_list(shadow_models, data)
    #            best_test_acc_list = attack(attack_in_train_list, attack_out_train_list, attack_in_list, attack_out_list, dataset, verbose=2)
    #            write_log(best_test_acc_list, dataset_name, model_name, model_name_shadow)