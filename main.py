import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, GCNConv  # noqa

from train import train_model
from models import GAT, GCN, GraphSage, GraphSageTopK, get_opts
from construct_attack_dataset import construct_attack_dataset
from utils import process_mask, format_model_name
from argparser import argparser
print("prerequirement satisfied!")


def attack_nodes_list(models, data):
    attack_in_list = []
    attack_out_list = []
    for i in range(dataset.num_classes):
        atk_in = []
        atk_out = []
        for model in models:
            model.eval()
            logits = model(data.x, data.edge_index,
                           data.edge_attr).detach().cpu().numpy()
            for l in range(len(logits)):
                if data.y[l] == i and data.train_mask[l]:
                    atk_in.append(logits[l])
                if data.y[l] == i and data.test_mask[l]:
                    atk_out.append(logits[l])
        attack_in_list.append(atk_in)
        attack_out_list.append(atk_out)

    return attack_in_list, attack_out_list


def write_log(best_test_acc_list, dataset_name, model_name, model_name_shadow):
    with open(log_dir, 'a') as f:
        f.write('{:.2f}/{:.2f}    '.format(np.mean(best_test_acc_list)
                * 100, np.max(best_test_acc_list) * 100))


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
        model_shadow = model_dict[model_name](
            num_features, num_classes).to(device)
        optimizer = get_opts(model_shadow, model_name)
        shadow_models.append(train_model(model_shadow, optimizer, data))
    return shadow_models


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model(data.x, data.edge_index, data.edge_attr)[
        data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def test(model, data):
    model.eval()
    logits, accs = model(data.x, data.edge_index, data.edge_attr), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def train_attack_model(dataset,
                       model,
                       optimizer,
                       device='cuda',
                       epoches=1000,
                       verbose=1):
    """GNN as Attack Model Membership Inference Attack

    Args:
        model: Attack model, common gnn model
        optimizer: finetune it according to the gnn model
        device: device
        epoches: epoches
    """
    gnn_mia_dataset = dataset

    best_val_acc = test_acc = 0
    for epoch in range(1, epoches + 1):
        loss = train(model, gnn_mia_dataset, optimizer)
        train_acc, val_acc, tmp_test_acc = test(
            model, gnn_mia_dataset)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if test_acc < tmp_test_acc:
                test_acc = tmp_test_acc
        if verbose == 1:
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Loss: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc, loss))

    if verbose == 2:
        print("*"*80)
        print("test_acc: {}".format(test_acc))
        print("*"*80)

    return test_acc


def attack(model, model_name, attack_in_train_list, attack_out_train_list, attack_in_list, attack_out_list, dataset, verbose=2):
    best_test_acc_list = []
    data = dataset[0].to(device)
    for i in range(dataset.num_classes):
        optimizer = get_opts(model, model_name)
        dataset = construct_attack_dataset(
            data, attack_in_train_list[i], attack_out_train_list[i], attack_in_list[i], attack_out_list[i])
        best_test_acc_list.append(train_attack_model(
            dataset, model, optimizer, device=device, epoches=1000, verbose=verbose))
    for i in range(len(best_test_acc_list)):
        print("class {}: {}".format(i, best_test_acc_list[i]))

    return best_test_acc_list


if __name__ == '__main__':
    model_dict = {'GCN': GCN, 'GAT': GAT, 'GraphSage': GraphSage}
    args = argparser()
    device = args.device
    device = torch.device('cuda' if device ==
                          'cuda' and torch.cuda.is_available() else 'cpu')
    print("devece: {}".format(device))
    dataset_name = args.dataset
    target_model_name = args.target_model
    shadow_model_name = args.shadow_model
    path = args.data_path
    log_dir = args.log_dir

    print("dataset: {}, target_model: {}, shadow_model: {}".format(
        dataset_name, target_model_name, shadow_model_name))

    with open(log_dir, 'a') as f:
        f.write(log_title)

    dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())

    # train target model
    best_model = train_target_model(dataset, target_model_name)
    # take logits of target model as test set of attack model
    attack_in_list, attack_out_list = attack_nodes_list(
        [best_model], dataset[0])

    # train target model
    shadow_models = train_shadow_models(dataset, shadow_model_name)
    # take logits of target model as test set of attack model
    attack_in_train_list, attack_out_train_list = attack_nodes_list(
        shadow_models, dataset[0])

    # construct graph dataset and do attack
    num_features = len(attack_in_train_list[0][0])
    num_classes = 2

    attack_model = GraphSage(num_features, num_classes).to(device)
    best_test_acc_list = attack(attack_model, 'GraphSage', attack_in_train_list,
                                attack_out_train_list, attack_in_list, attack_out_list, dataset, verbose=2)
    write_log(best_test_acc_list, dataset_name,
              target_model_name, shadow_model_name)

    #attack_model = GraphSageTopK(num_features, num_classes).to(device)
    #best_test_acc_list = attack(attack_model, 'GraphSage', attack_in_train_list, attack_out_train_list, attack_in_list, attack_out_list, dataset, verbose=2)
    #write_log(best_test_acc_list, dataset_name, target_model_name, shadow_model_name)

    # for model_name in model_names:
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
