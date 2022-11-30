import logging
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
rng = np.random.default_rng()


def make_train_test_set(refdataset, in_train, out_train, in_test, out_test, usesoftmax=False):
    """make nodes, labels, masks of train set and test set

    Args:
        usesoftmax (bool, optional): Defaults to False.

    Returns:
        x: All nodes of train set and test set
        y: membership labels
        *_mask: mask of final dataset
    """
    data = refdataset
    attack_in_train, attack_out_train, attack_in, attack_out = in_train, out_train, in_test, out_test

    # randomly choose nodes to form train, val and test set.
    # number of nodes depends on the reference dataset mask.
    length = int(data.train_mask.sum() / 2)

    x_attack_in_train = rng.choice(attack_in_train, length)
    x_attack_out_train = rng.choice(attack_out_train, length)

    lenofleft = len(data.x) - data.train_mask.sum()

    length = int(lenofleft // 2)
    length_test = int(data.test_mask.sum() // 2)
    length_val = int(data.val_mask.sum() // 2)
    length_holdout = int(length - length_test - length_val)

    member_test = attack_in
    nonmember_test = attack_out

    x_attack_in_test = rng.choice(member_test, length_test)
    x_attack_in_val = rng.choice(member_test, length_val)
    x_attack_in_holdout = rng.choice(member_test, length_holdout)

    x_attack_out_test = rng.choice(nonmember_test, length_test)
    x_attack_out_val = rng.choice(nonmember_test, length_val)
    if len(data.x) % 2 == 1:
        x_attack_out_holdout = rng.choice(nonmember_test, length_holdout+1)
    else:
        x_attack_out_holdout = rng.choice(nonmember_test, length_holdout)

    x = np.concatenate(
        (x_attack_in_train, x_attack_in_val, x_attack_in_holdout, x_attack_in_test,
            x_attack_out_train, x_attack_out_val, x_attack_out_holdout, x_attack_out_test))
    y = np.array(
        [1] * (len(x_attack_in_train) + len(x_attack_in_val) + len(x_attack_in_holdout) + len(x_attack_in_test)) +
        [0] * (len(x_attack_out_train) + len(x_attack_out_val) +
               len(x_attack_out_holdout) + len(x_attack_out_test))
    )

    # remake mask for nodes we have choice
    train_mask = []
    val_mask = []
    test_mask = []

    train_ind = 0
    val_ind = len(x_attack_in_train)
    holdout_ind = val_ind + len(x_attack_in_val)
    test_ind = holdout_ind + len(x_attack_in_holdout)
    out_ind = (len(x) // 2)

    for i in range(len(x)):
        if i < val_ind or out_ind <= i < out_ind + val_ind:
            train_mask.append(True)
        else:
            train_mask.append(False)
        if val_ind <= i < holdout_ind or out_ind + val_ind <= i < out_ind + holdout_ind:
            val_mask.append(True)
        else:
            val_mask.append(False)
        if test_ind <= i < out_ind or test_ind + out_ind <= i < len(x):
            test_mask.append(True)
        else:
            test_mask.append(False)
    x = torch.tensor(x)
    y = torch.tensor(y)

    if usesoftmax:
        x = F.softmax(torch.tensor(x), dim=-1)

    train_mask = torch.tensor(train_mask)
    val_mask = torch.tensor(val_mask)
    test_mask = torch.tensor(test_mask)

    logging.info("member: {}, all: {}".format(sum(y), len(y)))
    logging.info("member train: {}, train: {}".format(
        sum(y[train_mask]), len(y[train_mask])))

    return x, y, train_mask, val_mask, test_mask


def process_edges(x, refdataset, in_train, out_train):
    """To generate edge_index according to the train set info
    Args:
        x: All nodes

    Returns:
        edge_index: final edge index of our graph dataset
    """
    data = refdataset
    attack_in_train, attack_out_train = in_train, out_train
    length = int(data.train_mask.sum() / 2)

    # TODO: can be optimized
    x_attack_in_train = rng.choice(attack_in_train, length)
    x_attack_out_train = rng.choice(attack_out_train, length)

    threshold = (np.mean(np.max(x_attack_out_train, axis=1)) +
                 np.mean(np.max(x_attack_in_train, axis=1))) / 2

    # keep nodes adj relation for nodes of train set because it is known from shadow model;
    # construct nodes adj relation for nodes of test set using threshold which we choose mean
    # of L_infty norm of member and non-member train set nodes. If both of two nodes connected in reference dataset
    # are above or blow the threshold, it means they are probably both member(above) or non-member(blow), so we connect them.
    edge_index = [[], []]

    for i in range(len(data.edge_index[0])):
        if (x[data.edge_index[0][i]] in x_attack_in_train and x[data.edge_index[1][i]] in x_attack_in_train or
            x[data.edge_index[0][i]] in x_attack_out_train and x[data.edge_index[1][i]] in x_attack_out_train or
            max(x[data.edge_index[0][i]]) < threshold and max(x[data.edge_index[1][i]]) < threshold) or \
                (max(x[data.edge_index[0][i]]) > threshold and max(x[data.edge_index[1][i]]) > threshold):
            edge_index[0].append(data.edge_index[0][i])
            edge_index[1].append(data.edge_index[1][i])

    edge_index = torch.tensor(edge_index)
    logging.info("number of edges: {}".format(len(edge_index[0])))

    return edge_index


def construct_attack_dataset(refdataset, in_train, out_train, in_test, out_test):
    """Combine all info into a graph dataset

    Returns:
        gnn_mia_dataset: final graph dataset to be attacked
    """
    logging.info("Constructing graph dataset...")
    logging.info("preparing nodes...")
    x, y, train_mask, val_mask, test_mask = make_train_test_set(
        refdataset, in_train, out_train, in_test, out_test)
    logging.info("Nodes: {}".format(x.shape))
    logging.info("preparing edges...")
    edge_index = process_edges(x, refdataset, in_train, out_train)

    # make graph dataset to attack using pytorch geometric api
    gnn_mia_dataset = Data(x=x, y=y, test_mask=test_mask,
                           train_mask=train_mask, val_mask=val_mask, edge_index=edge_index)
    logging.info("graph dataset constructed!")

    return gnn_mia_dataset
