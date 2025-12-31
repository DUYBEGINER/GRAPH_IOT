"""E-GraphSAGE model for edge classification.

Node = endpoint (IP/IP:Port), Edge = flow, Task = edge classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeFeatureSAGEConv(nn.Module):
    """SAGEConv layer that incorporates edge features during aggregation.   """
    
    def __init__(self, in_dim: int, out_dim: int, in_edge_dim: int, aggr: str = "mean"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_edge_dim = in_edge_dim
        self.aggr = aggr
        
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neigh = nn.Linear(out_dim, out_dim, bias=False)  # Now takes projected edge dim
        self.lin_edge = nn.Linear(in_edge_dim, out_dim, bias=False)  # Project edge_attr to out_dim
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward with edge features.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edges [2, num_edges]
            edge_attr: Original edge features [num_edges, in_edge_dim] (always raw)
            
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        src, dst = edge_index
        num_nodes = x.size(0)
        
        # Self transformation
        out_self = self.lin_self(x)
        
        # Project edge features: edge_attr (in_edge_dim) -> projected (out_dim)
        edge_projected = self.lin_edge(edge_attr)
        
        # Aggregate projected edge features
        aggregated = torch.zeros(num_nodes, self.out_dim, device=x.device)
        
        if self.aggr == "mean":
            ones = torch.ones(edge_index.size(1), device=x.device)
            degree = torch.zeros(num_nodes, device=x.device)
            degree.scatter_add_(0, dst, ones)
            degree = degree.clamp(min=1)
            
            aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_projected), edge_projected)
            aggregated = aggregated / degree.unsqueeze(1)
        elif self.aggr == "sum":
            aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_projected), edge_projected)
        
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
        self.in_edge_dim = in_dim  # Store original edge dimension
        
        # Build layers - all layers receive original edge_attr (in_dim)
        self.convs = nn.ModuleList()
        # First layer: node in_dim -> hidden_dim, edge in_dim -> hidden_dim
        self.convs.append(EdgeFeatureSAGEConv(in_dim, hidden_dim, in_edge_dim=in_dim, aggr=aggr))
        
        # Subsequent layers: node hidden_dim -> hidden_dim, edge still in_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(EdgeFeatureSAGEConv(hidden_dim, hidden_dim, in_edge_dim=in_dim, aggr=aggr))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Edge classifier - output num_classes logits for softmax + cross entropy (paper)
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        print(f"   E-GraphSAGE (Hướng 1): edge_dim={in_dim}, {in_dim}→{hidden_dim}x{num_layers}→{num_classes}")
        print(f"   Each layer projects edge_attr: {in_dim}→{hidden_dim}")
        print(f"   Loss: Softmax + CrossEntropy (paper)")
    
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
            edge_attr: Edge features [num_edges, in_edge_dim] - always original!
            edge_label_index: Edges to classify [2, num_target_edges]
            
        Returns:
            Edge logits [num_target_edges]
        """
        # Update node embeddings - pass original edge_attr to all layers
        h = x
        for i, conv in enumerate(self.convs):
            # Each layer projects edge_attr internally via lin_edge
            h = conv(h, edge_index, edge_attr)  # edge_attr is always original
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
        
        # Classify edges - output [num_edges, num_classes] logits
        edge_logits = self.edge_classifier(edge_emb)
        
        return edge_logits  # [num_edges, num_classes]
