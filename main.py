import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from train import train_model
from models import get_opts, model_dict
from construct_attack_dataset import construct_attack_dataset
from utils import process_mask, print_attack_result
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


def train_target_model(dataset, model_name):
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    model = model_dict[model_name](num_features, num_classes).to(device)
    data = dataset[0]
    data = data.to(device)
    optimizer = get_opts(model, model_name)
    best_model, _ = train_model(model, optimizer, data)
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
        best_model, _ = train_model(model_shadow, optimizer, data)
        shadow_models.append(best_model)
    return shadow_models


def construct_dataset_and_train_attck_model(model, model_name, attack_in_train_list, attack_out_train_list, attack_in_list, attack_out_list, refdataset):
    best_test_acc_list = []
    data = refdataset[0].to(device)
    for i in range(refdataset.num_classes):
        optimizer = get_opts(model, model_name)
        dataset = construct_attack_dataset(
            data, attack_in_train_list[i], attack_out_train_list[i], attack_in_list[i], attack_out_list[i])
        _, test_acc = train_model(model, optimizer, dataset, epoches=1000)
        best_test_acc_list.append(test_acc)

    return best_test_acc_list


if __name__ == '__main__':
    args = argparser()
    device = args.device
    device = torch.device('cuda' if device ==
                          'cuda' and torch.cuda.is_available() else 'cpu')
    print("devece: {}".format(device))
    dataset_name = args.dataset
    target_model_name = args.target_model
    shadow_model_name = args.shadow_model
    attack_model_name = args.attack_model
    path = args.data_path
    log_dir = args.log_dir

    print("dataset: {}, target_model: {}, shadow_model: {}".format(
        dataset_name, target_model_name, shadow_model_name))

    dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())

    # train target model
    target_model = train_target_model(dataset, target_model_name)
    # take logits of target model as test set of attack model
    attack_in_list, attack_out_list = attack_nodes_list(
        [target_model], dataset[0])

    # train shadow model
    shadow_models = train_shadow_models(dataset, shadow_model_name)
    # take logits of shadow model as test set of attack model
    attack_in_train_list, attack_out_train_list = attack_nodes_list(
        shadow_models, dataset[0])

    # construct graph dataset and do attack
    num_features = len(attack_in_train_list[0][0])
    num_classes = 2

    attack_model = model_dict[attack_model_name](
        num_features, num_classes).to(device)
    best_test_acc_list = construct_dataset_and_train_attck_model(
        attack_model, attack_model_name, attack_in_train_list, attack_out_train_list, attack_in_list, attack_out_list, dataset)

    print_attack_result(best_test_acc_list)

    #attack_model = GraphSageTopK(num_features, num_classes).to(device)
    #best_test_acc_list = attack(attack_model, 'GraphSage', attack_in_train_list, attack_out_train_list, attack_in_list, attack_out_list, dataset, verbose=2)
    #write_log(best_test_acc_list, dataset_name, target_model_name, shadow_model_name)
