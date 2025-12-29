"""GraphSAGE model for flow classification."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

logger = logging.getLogger(__name__)


class FlowGraphSAGE(nn.Module):
    """
    GraphSAGE for network flow classification.
    Node = flow, Edge = KNN similarity, Task = node classification
    """
    
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int = 128, 
        num_classes: int = 2, 
        num_layers: int = 2,
        dropout: float = 0.3
    ):
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
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        logger.info(f"FlowGraphSAGE: {in_dim}→{hidden_dim}x{num_layers}→{num_classes}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.classifier(x)
        return x
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings before classification."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
