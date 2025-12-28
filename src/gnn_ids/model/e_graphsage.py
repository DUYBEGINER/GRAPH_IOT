"""E-GraphSAGE model for edge classification.

Based on E-GraphSAGE paper approach:
- Message passing uses edge features to update node embeddings
- Edge embedding = concat(z_u, z_v) where z_u, z_v are node embeddings
- Edge classifier predicts on concatenated embeddings
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

logger = logging.getLogger(__name__)


class EdgeFeatureSAGEConv(nn.Module):
    """
    SAGEConv layer that incorporates edge features during aggregation.
    
    For E-GraphSAGE: aggregate edge features of incident edges.
    """
    
    def __init__(self, in_dim: int, out_dim: int, aggr: str = "mean"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggr = aggr
        
        # Transform for source node
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        # Transform for aggregated edge features
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with edge features.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: [2, num_edges]
            edge_attr: Edge features [num_edges, in_dim]
            
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        src, dst = edge_index
        num_nodes = x.size(0)
        
        # Self transformation
        out_self = self.lin_self(x)
        
        # Aggregate edge features for each node
        # For each destination node, aggregate features of incoming edges
        aggregated = torch.zeros(num_nodes, self.in_dim, device=x.device)
        
        # Use scatter to aggregate edge features
        if self.aggr == "mean":
            ones = torch.ones(edge_index.size(1), device=x.device)
            degree = torch.zeros(num_nodes, device=x.device)
            degree.scatter_add_(0, dst, ones)
            degree = degree.clamp(min=1)  # Avoid division by zero
            
            aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_attr), edge_attr)
            aggregated = aggregated / degree.unsqueeze(1)
        elif self.aggr == "sum":
            aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_attr), edge_attr)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggr}")
        
        # Transform aggregated features
        out_neigh = self.lin_neigh(aggregated)
        
        # Combine
        out = out_self + out_neigh + self.bias
        
        return out


class EGraphSAGE(nn.Module):
    """
    E-GraphSAGE for edge classification.
    
    Architecture:
    - K layers of EdgeFeatureSAGEConv (default K=2)
    - Node embeddings updated via edge feature aggregation
    - Edge representation: concat(z_u, z_v)
    - Edge classifier: Linear(2*hidden_dim -> num_classes)
    
    Args:
        in_dim: Input feature dimension (must match edge feature dim)
        hidden_dim: Hidden dimension (default: 128)
        num_classes: Number of output classes (2 for binary)
        num_layers: Number of GraphSAGE layers (default: 2)
        dropout: Dropout rate (default: 0.2)
        aggr: Aggregation function (mean, sum)
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        aggr: str = "mean"
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Build GraphSAGE layers with edge feature support
        self.convs = nn.ModuleList()
        self.convs.append(EdgeFeatureSAGEConv(in_dim, hidden_dim, aggr=aggr))
        
        for _ in range(num_layers - 1):
            self.convs.append(EdgeFeatureSAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        
        # Batch normalization for stability
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Edge classifier: takes concat(z_u, z_v)
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        logger.info(f"Initialized E-GraphSAGE: "
                   f"{in_dim}→{hidden_dim} ({num_layers} layers) → edge_clf({2*hidden_dim}→{num_classes})")
        logger.info(f"Parameters: hidden={hidden_dim}, dropout={dropout}, aggr={aggr}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_label_index: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for edge classification.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, in_dim]
            edge_label_index: Edges to classify [2, num_target_edges]
                            If None, uses edge_index
        
        Returns:
            Edge logits [num_target_edges, num_classes]
        """
        # Update node embeddings via edge-feature-based message passing
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_attr)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Get node embeddings for edges to classify
        if edge_label_index is None:
            edge_label_index = edge_index
        
        src_emb = h[edge_label_index[0]]  # [num_target_edges, hidden_dim]
        dst_emb = h[edge_label_index[1]]  # [num_target_edges, hidden_dim]
        
        # Concatenate source and destination embeddings
        edge_emb = torch.cat([src_emb, dst_emb], dim=1)  # [num_target_edges, 2*hidden_dim]
        
        # Classify edges
        edge_logits = self.edge_classifier(edge_emb)  # [num_target_edges, num_classes]
        
        return edge_logits
    
    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Get node embeddings after message passing.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            
        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_attr)
            h = self.bns[i](h)
            h = F.relu(h)
        
        return h
