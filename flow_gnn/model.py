"""GraphSAGE model for flow classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FlowGraphSAGE(nn.Module):
    """GraphSAGE model for flow classification (node classification)."""
    
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 2, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Classifier - output 1 logit for binary classification
        self.classifier = nn.Linear(hidden_dim, 1)
        
        print(f"   FlowGraphSAGE: {in_dim}→{hidden_dim}x{num_layers}→1 (binary)")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits (no sigmoid)."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.classifier(x)
        return x.squeeze(-1)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings before classification."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
