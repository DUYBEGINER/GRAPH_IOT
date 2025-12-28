# src/gnn_ids/models/sage.py
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FlowGraphSAGE(nn.Module):
    """GraphSAGE cho bài toán phân loại node (flow)."""
    def __init__(self, in_dim, hidden_dim, num_classes=2, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.lin(x)
