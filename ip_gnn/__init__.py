"""IP-based GNN package - Edge classification using endpoint graph."""

from .model import EGraphSAGE
from .graph import create_endpoint_graph
from .train import train_ip_gnn
from .utils import compute_metrics, get_device, set_seed
from . import config

__all__ = [
    'EGraphSAGE', 
    'create_endpoint_graph', 
    'train_ip_gnn', 
    'compute_metrics', 
    'get_device',
    'set_seed',
    'config'
]
