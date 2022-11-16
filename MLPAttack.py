""" 
Author: Chao Shu
Date: 07/22/21
"""
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

rng = np.random.default_rng()

class MLPAttack:
    """MLP as Attack model for MIA. This is used for comparision.
    """

    def __init__(self,
                 referdataset,
                 attack_in_train,
                 attack_out_train,
                 attack_in,
                 attack_out
                 ):
        """
        Args:
            referdataset: a reference dataset from shadow model training
            attack_in_train: logits from shadow model training
            attack_out_train: logits from shadow model testing
            attack_in: logits from target model training
            attack_out: logis from target model testing
        """
        self.data = referdataset
        self.attack_in_train, self.attack_out_train, self.attack_in, self.attack_out = \
            attack_in_train, attack_out_train, attack_in, attack_out

    def make_dataset(self, usesoftmax=False):
        import random
        data = self.data
        attack_in_train, attack_out_train, attack_in, attack_out = \
            self.attack_in_train, self.attack_out_train, self.attack_in, self.attack_out

        x_attack_in = attack_in
        x_attack_out = attack_out
        x_attack_out = np.array(random.sample(
            x_attack_out.tolist(), x_attack_in.shape[0]))
        if usesoftmax:
            x_attack_in = F.softmax(torch.tensor(
                x_attack_in), dim=-1).detach().numpy()
            x_attack_out = F.softmax(torch.tensor(
                x_attack_out), dim=-1).detach().numpy()
        x_attack = torch.cat(
            (torch.tensor(x_attack_in), torch.tensor(x_attack_out))).detach().numpy()
        y_attack = [1] * len(x_attack_in) + [0] * len(x_attack_out)

        x_attack_in_train = attack_in_train
        x_attack_out_train = attack_out_train
        x_attack_out_train = np.array(random.sample(
            x_attack_out_train.tolist(), x_attack_in_train.shape[0]))
        x_attack_train = torch.cat((torch.tensor(x_attack_in_train), torch.tensor(
            x_attack_out_train))).detach().numpy()
        if usesoftmax:
            x_attack_in_train = F.softmax(torch.tensor(
                x_attack_in_train), dim=-1).detach().numpy()
            x_attack_out_train = F.softmax(torch.tensor(
                x_attack_out_train), dim=-1).detach().numpy()
        y_attack_train = y_attack[:]

        return x_attack, y_attack, x_attack_train, y_attack_train

    def attack(self, clf=MLPClassifier(random_state=1, solver='adam', max_iter=1000)):
        x_attack, y_attack, x_attack_train, y_attack_train = self.make_dataset()
        history = clf.fit(x_attack_train, y_attack_train)

        print("Test set score: {}".format(clf.score(x_attack, y_attack)))

        test_outputs = clf.predict(x_attack)
        print(metrics.classification_report(
            y_attack, test_outputs, labels=range(2)))
        print(metrics.roc_auc_score(y_attack, test_outputs))
