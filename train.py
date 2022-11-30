import logging
import torch
import torch.nn.functional as F


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data.x, data.edge_index, data.edge_attr)[
               data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

    return model


@torch.no_grad()
def test(model, data):
    model.eval()
    logits, accs = model(data.x, data.edge_index, data.edge_attr), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def train_model(model, optimizer, data, epoches=200):
    best_model = None
    best_val_acc = test_acc = 0
    for epoch in range(1, epoches + 1):
        model = train(model, optimizer, data)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            best_model = model
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    logging.info(log.format(epoch, train_acc, best_val_acc, test_acc))
    return best_model, test_acc
