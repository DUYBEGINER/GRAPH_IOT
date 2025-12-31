"""E-GraphSAGE model for edge classification.

Node = endpoint (IP/IP:Port), Edge = flow, Task = edge classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeFeatureSAGEConv(nn.Module):
    """SAGEConv layer that incorporates edge features during aggregation."""
    
    def __init__(self, in_dim: int, out_dim: int, aggr: str = "mean"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggr = aggr
        
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward with edge features."""
        src, dst = edge_index
        num_nodes = x.size(0)
        
        # Self transformation
        out_self = self.lin_self(x)
        
        # Aggregate edge features
        aggregated = torch.zeros(num_nodes, self.in_dim, device=x.device)
        
        if self.aggr == "mean":
            ones = torch.ones(edge_index.size(1), device=x.device)
            degree = torch.zeros(num_nodes, device=x.device)
            degree.scatter_add_(0, dst, ones)
            degree = degree.clamp(min=1)
            
            aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_attr), edge_attr)
            aggregated = aggregated / degree.unsqueeze(1)
        elif self.aggr == "sum":
            aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_attr), edge_attr)
        
        out_neigh = self.lin_neigh(aggregated)
        out = out_self + out_neigh + self.bias
        
        return out


class EGraphSAGE(nn.Module):
    """E-GraphSAGE for edge classification.
    
    Architecture:
    - K layers of EdgeFeatureSAGEConv
    - Edge representation: concat(z_u, z_v)
    - Edge classifier: Linear(2*hidden_dim -> 1)
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
        
        # Build layers
        self.convs = nn.ModuleList()
        self.convs.append(EdgeFeatureSAGEConv(in_dim, hidden_dim, aggr=aggr))
        
        for _ in range(num_layers - 1):
            self.convs.append(EdgeFeatureSAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Edge classifier - output 1 logit for binary classification
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        print(f"   E-GraphSAGE: {in_dim}→{hidden_dim}x{num_layers}→1 (edge binary)")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_label_index: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass for edge classification.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: All edges [2, num_edges]
            edge_attr: Edge features [num_edges, in_dim]
            edge_label_index: Edges to classify [2, num_target_edges]
            
        Returns:
            Edge logits [num_target_edges]
        """
        # Update node embeddings
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_attr)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Get edges to classify
        if edge_label_index is None:
            edge_label_index = edge_index
        
        src_emb = h[edge_label_index[0]]
        dst_emb = h[edge_label_index[1]]
        
        # Concatenate embeddings
        edge_emb = torch.cat([src_emb, dst_emb], dim=1)
        
        # Classify edges
        edge_logits = self.edge_classifier(edge_emb)
        
        return edge_logits.squeeze(-1)
