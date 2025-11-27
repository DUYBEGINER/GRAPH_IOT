"""
GNN Model Architecture for IoT Network Anomaly Detection
Các kiến trúc GNN để phát hiện anomaly trong network traffic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.nn import BatchNorm, LayerNorm


# ============================================================================
# GCN-based Model
# ============================================================================

class GCN(nn.Module):
    """Graph Convolutional Network for anomaly detection"""

    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input layer
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))

        # Output layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))

        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # Graph convolution layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Classification
        x = self.classifier(x)

        return x


# ============================================================================
# GAT-based Model
# ============================================================================

class GAT(nn.Module):
    """Graph Attention Network for anomaly detection"""

    def __init__(self, in_channels, hidden_channels, num_classes,
                 num_layers=3, heads=4, dropout=0.5):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.batch_norms.append(BatchNorm(hidden_channels * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                     heads=heads, dropout=dropout))
            self.batch_norms.append(BatchNorm(hidden_channels * heads))

        # Output layer (concat=False for final layer)
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                 heads=1, concat=False, dropout=dropout))
        self.batch_norms.append(BatchNorm(hidden_channels))

        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.classifier(x)

        return x


# ============================================================================
# GraphSAGE-based Model
# ============================================================================

class GraphSAGE(nn.Module):
    """GraphSAGE for anomaly detection"""

    def __init__(self, in_channels, hidden_channels, num_classes,
                 num_layers=3, dropout=0.5, aggregator='mean'):
        super(GraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        self.batch_norms.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
            self.batch_norms.append(BatchNorm(hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
        self.batch_norms.append(BatchNorm(hidden_channels))

        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.classifier(x)

        return x


# ============================================================================
# Hybrid GNN Model
# ============================================================================

class HybridGNN(nn.Module):
    """
    Hybrid GNN combining GCN and GAT
    Sử dụng cả GCN và GAT để tận dụng ưu điểm của cả hai
    """

    def __init__(self, in_channels, hidden_channels, num_classes,
                 num_layers=3, heads=4, dropout=0.5):
        super(HybridGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GCN branch
        self.gcn_convs = nn.ModuleList()
        self.gcn_bns = nn.ModuleList()

        # GAT branch
        self.gat_convs = nn.ModuleList()
        self.gat_bns = nn.ModuleList()

        # First layer
        self.gcn_convs.append(GCNConv(in_channels, hidden_channels))
        self.gcn_bns.append(BatchNorm(hidden_channels))

        self.gat_convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        self.gat_bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.gcn_bns.append(BatchNorm(hidden_channels))

            self.gat_convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
            self.gat_bns.append(BatchNorm(hidden_channels))

        # Fusion layer
        self.fusion = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fusion_bn = BatchNorm(hidden_channels)

        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # GCN branch
        x_gcn = x
        for i in range(self.num_layers - 1):
            x_gcn = self.gcn_convs[i](x_gcn, edge_index)
            x_gcn = self.gcn_bns[i](x_gcn)
            x_gcn = F.relu(x_gcn)
            x_gcn = F.dropout(x_gcn, p=self.dropout, training=self.training)

        # GAT branch
        x_gat = x
        for i in range(self.num_layers - 1):
            x_gat = self.gat_convs[i](x_gat, edge_index)
            x_gat = self.gat_bns[i](x_gat)
            x_gat = F.elu(x_gat)
            x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)

        # Concatenate and fuse
        x = torch.cat([x_gcn, x_gat], dim=1)
        x = self.fusion(x)
        x = self.fusion_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Classification
        x = self.classifier(x)

        return x


# ============================================================================
# Model Factory
# ============================================================================

def create_model(model_name, in_channels, hidden_channels, num_classes, **kwargs):
    """
    Factory function để tạo GNN model

    Args:
        model_name: 'GCN', 'GAT', 'GraphSAGE', hoặc 'Hybrid'
        in_channels: Số features đầu vào
        hidden_channels: Số hidden units
        num_classes: Số classes
        **kwargs: Các tham số khác

    Returns:
        GNN model
    """

    models = {
        'GCN': GCN,
        'GAT': GAT,
        'GraphSAGE': GraphSAGE,
        'Hybrid': HybridGNN
    }

    if model_name not in models:
        raise ValueError(f"Model {model_name} không được hỗ trợ. Chọn từ: {list(models.keys())}")

    model = models[model_name](in_channels, hidden_channels, num_classes, **kwargs)

    return model


# ============================================================================
# Model Summary
# ============================================================================

def count_parameters(model):
    """Đếm số parameters trong model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, model_name):
    """In thông tin model"""
    print("\n" + "=" * 80)
    print(f"MODEL SUMMARY: {model_name}")
    print("=" * 80)
    print(f"Total parameters: {count_parameters(model):,}")
    print("\nArchitecture:")
    print(model)
    print("=" * 80)


if __name__ == "__main__":
    # Test models
    in_channels = 79
    hidden_channels = 128
    num_classes = 2

    print("Testing GNN models...")

    for model_name in ['GCN', 'GAT', 'GraphSAGE', 'Hybrid']:
        model = create_model(model_name, in_channels, hidden_channels, num_classes)
        model_summary(model, model_name)

