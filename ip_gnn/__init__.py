"""Endpoint-based GNN package - Edge classification using flow graph."""

from .model import EGraphSAGE
from .graph import create_endpoint_graph
from .train import train_endpoint_gnn
from .utils import compute_metrics, get_device

__all__ = ['EGraphSAGE', 'create_endpoint_graph', 'train_endpoint_gnn', 'compute_metrics', 'get_device']
