import torch
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GINConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
class GraphSage(torch.nn.Module):
    def __init__(self,num_features, num_classes):
        super(GraphSage, self).__init__()
        self.sage1 = SAGEConv(num_features, 16)  # 定义两层GraphSAGE层
        self.sage2 = SAGEConv(16, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)

        x = F.log_softmax(x, dim=1)
        return x
    
class GraphSageTopK(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.conv1 = SAGEConv(num_features, 16)
        self.pool1 = TopKPooling(16, ratio=1)
        self.conv2 = SAGEConv(16, 16)
        self.pool2 = TopKPooling(16, ratio=1)
        self.conv3 = SAGEConv(16, 16)
        self.pool3 = TopKPooling(16, ratio=1)

        self.lin1 = torch.nn.Linear(16, 16)
        self.lin2 = torch.nn.Linear(16, 8)
        self.lin3 = torch.nn.Linear(8, num_classes)

    def forward(self, x, edge_index, edge_weight):
        batch = None

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        #x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        #x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        #x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        #x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=1)

        return x


#     optimizer = torch.optim.Adam([
#         dict(params=model.conv1.parameters(), weight_decay=5e-4),
#         dict(params=model.conv2.parameters(), weight_decay=0)
#     ], lr=0.01)  # Only perform weight-decay on first convolution.
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16, cached=True,
                             normalize=True)
        self.conv2 = GCNConv(16, num_classes, cached=True,
                             normalize=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()

        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)
    

# model = Net(dataset.num_features, 32, dataset.num_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
class GIN(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels):
        super(GIN, self).__init__()

        self.conv1 = GINConv(
            Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv5 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
#         x = global_add_pool(x)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

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