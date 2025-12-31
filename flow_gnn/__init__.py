"""Flow-based GNN package - Node classification using KNN graph."""

from .model import FlowGraphSAGE
from .graph import build_knn_graph
from .train import train_flow_gnn
from .utils import compute_metrics, get_device, set_seed
from . import config

__all__ = [
    'FlowGraphSAGE', 
    'build_knn_graph', 
    'train_flow_gnn', 
    'compute_metrics', 
    'get_device',
    'set_seed',
    'config'
]
