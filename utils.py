import torch

def format_model_name(model_name):
    return model_name + '          '

def get_opts(model, model_name):
    if model_name == 'GCN':
        return torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=0.01)  # Only perform weight-decay on first convolution.
#         return torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    elif model_name == 'GraphSage':
        return torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    else:
        return torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    

def process_mask(data):
    train_mask = []
    val_mask = []
    test_mask = []
    train_num = sum(data.train_mask)
    val_num = sum(data.val_mask)
    test_num = sum(data.test_mask)
    for i in range(len(data.x)):
        if i < test_num:
            test_mask.append(True)
        else:
            test_mask.append(False)
        if test_num <= i < test_num + val_num:
            val_mask.append(True)
        else:
            val_mask.append(False)
        if (test_num + val_num) <= i < (test_num + val_num + train_num):
            train_mask.append(True)
        else:
            train_mask.append(False)
    data.train_mask = torch.tensor(train_mask)
    data.val_mask = torch.tensor(val_mask)
    data.test_mask = torch.tensor(test_mask)
    
    return data