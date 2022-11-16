import os.path as osp
import argparse
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, GCNConv  # noqa

from train import train_model
from models import GAT, GCN, GraphSage, get_opts
from GnnAttack import GnnAttack
from utils import process_mask, format_model_name
print("prerequirement satisfied!")

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
print("devece: {}".format(device))


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', default=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--model', type=str, nargs='+', default=['GCN', 'GraphSage', 'GAT'])
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./gammia_mm_diff_target_log.txt')
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    return parser.parse_args()

args = argparser()
dataset_names = args.dataset
model_names = args.model
path = args.data_path
log_dir = args.log_dir
print(dataset_names)
print(model_names)

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

def attack(attack_in_train_list, attack_out_train_list, attack_in_list, attack_out_list, dataset, verbose=2):
    num_features = len(attack_in_train_list[0][0])
    num_classes = 2
    best_test_acc_list = []
    data = dataset[0].to(device)
    for i in range(dataset.num_classes):
        model = GraphSage(num_features, num_classes).to(device)
        optimizer = get_opts(model, 'GraphSage')
        gam = GnnAttack(data, attack_in_train_list[i], attack_out_train_list[i], attack_in_list[i], attack_out_list[i])
        best_test_acc_list.append(gam.attack(model, optimizer, device=device, epoches=1000, verbose=verbose))
    for i in range(len(best_test_acc_list)):
        print("class {}: {}".format(i, best_test_acc_list[i]))
        
    return best_test_acc_list

def write_log(best_test_acc_list, dataset_name, model_name, model_name_shadow):
    with open(log_dir, 'a') as f:
        f.write('{:.2f}/{:.2f}    '.format(np.mean(best_test_acc_list) * 100, np.max(best_test_acc_list) * 100))


log_title = '\n\n-------------------------------Cora-------------------  ----------------------CiteSeer--------------  --------------------PubMed-----------------\ntar.\\shad.      GCN         GraphSage         GAT            GCN         GraphSage        GAT            GCN          GraphSage        GAT  \n------------------------------------------------------  --------------------------------------------  -------------------------------------------'
with open(log_dir, 'a') as f:
    f.write(log_title)

for model_name in model_names:
    with open(log_dir, 'a') as f:
        f.write('\n{}'.format(format_model_name(model_name)))
    for dataset_name in dataset_names:
        dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]
        num_features = dataset.num_features
        num_classes = dataset.num_classes
    
        exec("model = {}(num_features, num_classes).to(device)".format(model_name))
        data = data.to(device)
        optimizer = get_opts(model, model_name)
        best_model = train_model(model, optimizer, data)
        attack_in_list, attack_out_list = attack_nodes_list([best_model], data)
        for model_name_shadow in model_names:
            dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
            data = dataset[0]
            data = process_mask(data).to(device)
            print("dataset: {}, target_model: {}, shadow_model: {}".format(dataset_name, model_name, model_name_shadow))
            shadow_models = []
            for i in range(dataset.num_classes):
#             for i in range(50):
                exec("model_shadow = {}(num_features, num_classes).to(device)".format(model_name_shadow))
                optimizer = get_opts(model_shadow, model_name_shadow)
                shadow_models.append(train_model(model_shadow, optimizer, data))
            attack_in_train_list, attack_out_train_list = attack_nodes_list(shadow_models, data)
#             attack_in_list, attack_out_list = attack_nodes_list([best_model], data)
            best_test_acc_list = attack(attack_in_train_list, attack_out_train_list, attack_in_list, attack_out_list, dataset, verbose=2)
            write_log(best_test_acc_list, dataset_name, model_name, model_name_shadow)