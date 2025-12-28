"""GraphSAGE model for flow classification."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

logger = logging.getLogger(__name__)


class FlowGraphSAGE(nn.Module):
    """
    GraphSAGE model for network flow classification.
    
    Args:
        in_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes (default: 2 for binary)
        num_layers: Number of GraphSAGE layers (default: 2)
        dropout: Dropout rate (default: 0.3)
    """
    
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int, 
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
        
        # Batch normalization for stability
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        logger.info(f"Initialized FlowGraphSAGE: "
                   f"{in_dim}→{hidden_dim} ({num_layers} layers) →{num_classes}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Logits [num_nodes, num_classes]
        """
        # GraphSAGE layers with ReLU, BatchNorm, and Dropout
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        
        return x
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get node embeddings before classification layer.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
