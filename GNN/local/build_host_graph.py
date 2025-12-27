"""
Host-Connection Graph Construction for Network Traffic Data
Xây dựng đồ thị Host-Flow từ dữ liệu network traffic với IP addresses
Trong đồ thị này:
- Nodes: IP addresses (hosts)
- Edges: Network flows giữa các hosts
- Node features: Aggregated statistics từ các flows
- Edge features: Flow-level features
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import pickle
import os
from tqdm import tqdm
from collections import defaultdict
import ipaddress

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = r"/CICIDS2018-CSV"
GRAPH_DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\graph_data_host"
PROCESSED_DATA_DIR = r"/processed_data"

# Chọn file có IP information
CSV_FILE = "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"

# Graph construction parameters
MAX_FLOWS = 200000  # Giới hạn số flows để xử lý (tránh memory overflow)
MIN_FLOW_COUNT = 2  # Chỉ giữ hosts có ít nhất N flows
USE_PORTS = True  # Có sử dụng port information không

# Features để aggregate cho mỗi host
AGGREGATE_FEATURES = [
    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Flow Byts/s', 'Flow Pkts/s',
    'Pkt Len Mean', 'Pkt Len Std',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt'
]

# Features cho edges (flow-level)
EDGE_FEATURES = [
    'Flow Duration', 'Protocol', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Flow Byts/s', 'Flow Pkts/s',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt'
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ip_to_int(ip_str):
    """Chuyển IP address string sang integer để so sánh"""
    try:
        # Xử lý IPv4
        return int(ipaddress.IPv4Address(ip_str))
    except:
        # Nếu không parse được, return hash
        return hash(ip_str) % (2**32)


def is_private_ip(ip_str):
    """Kiểm tra IP có phải private không"""
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private
    except:
        return False


# ============================================================================
# HOST-CONNECTION GRAPH BUILDER
# ============================================================================

class HostConnectionGraphBuilder:
    """Class để xây dựng Host-Connection Graph từ network traffic"""

    def __init__(self, use_ports=True, min_flow_count=2):
        self.use_ports = use_ports
        self.min_flow_count = min_flow_count
        self.host_to_id = {}  # Mapping IP -> node_id
        self.id_to_host = {}  # Mapping node_id -> IP
        self.host_features = defaultdict(list)  # Features cho mỗi host
        self.edges = []  # List of (src_id, dst_id, edge_features)
        self.host_labels = {}  # Label cho mỗi host (majority vote)

    def load_and_prepare_data(self, csv_path, max_flows=None):
        """
        Load CSV data và chuẩn bị cho graph construction

        Args:
            csv_path: Path to CSV file
            max_flows: Maximum number of flows to load

        Returns:
            DataFrame with flow data
        """
        print(f"\nLoading data from: {csv_path}")

        # Load data
        if max_flows:
            df = pd.read_csv(csv_path, encoding='latin-1', low_memory=False, nrows=max_flows)
        else:
            df = pd.read_csv(csv_path, encoding='latin-1', low_memory=False)

        print(f"✓ Loaded {len(df):,} flows")

        # Filter out header rows (nếu có)
        if 'Label' in df.columns:
            df = df[df['Label'] != 'Label']

        # Reset index
        df = df.reset_index(drop=True)

        # Check for required columns
        required_cols = ['Src IP', 'Dst IP', 'Label']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(f"✓ Data shape: {df.shape}")
        print(f"✓ Unique labels: {df['Label'].nunique()}")
        print(f"  Label distribution:\n{df['Label'].value_counts()}")

        return df

    def build_host_mapping(self, df):
        """
        Tạo mapping giữa IP addresses và node IDs

        Args:
            df: DataFrame with 'Src IP' and 'Dst IP' columns
        """
        print("\nBuilding host mapping...")

        # Collect all unique IPs
        all_ips = set(df['Src IP'].unique()) | set(df['Dst IP'].unique())

        # Count flows per IP
        ip_flow_count = defaultdict(int)
        for ip in df['Src IP']:
            ip_flow_count[ip] += 1
        for ip in df['Dst IP']:
            ip_flow_count[ip] += 1

        # Filter IPs với số flows >= threshold
        valid_ips = sorted([ip for ip in all_ips if ip_flow_count[ip] >= self.min_flow_count])

        # Create mapping
        for idx, ip in enumerate(valid_ips):
            self.host_to_id[ip] = idx
            self.id_to_host[idx] = ip

        print(f"✓ Total unique IPs: {len(all_ips):,}")
        print(f"✓ IPs with >= {self.min_flow_count} flows: {len(valid_ips):,}")
        print(f"✓ Created {len(self.host_to_id):,} host nodes")

        return valid_ips

    def aggregate_host_features(self, df, feature_cols):
        """
        Aggregate features cho mỗi host từ các flows

        Args:
            df: DataFrame
            feature_cols: List of features to aggregate
        """
        print("\nAggregating host features...")

        # Convert numeric columns
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Aggregate cho mỗi host (cả source và destination)
        host_stats = defaultdict(lambda: defaultdict(list))

        for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing flows"):
            src_ip = row['Src IP']
            dst_ip = row['Dst IP']

            # Nếu IP không trong mapping (filtered out), skip
            if src_ip not in self.host_to_id or dst_ip not in self.host_to_id:
                continue

            # Aggregate features for source host
            for col in feature_cols:
                if col in row and not pd.isna(row[col]):
                    host_stats[src_ip][col].append(float(row[col]))

            # Aggregate features for destination host
            for col in feature_cols:
                if col in row and not pd.isna(row[col]):
                    host_stats[dst_ip][col].append(float(row[col]))

        # Compute statistics (mean, std, max, min, sum)
        print("  Computing statistics...")
        for host_id in tqdm(range(len(self.host_to_id)), desc="  Computing stats"):
            ip = self.id_to_host[host_id]
            features = []

            for col in feature_cols:
                if col in host_stats[ip] and len(host_stats[ip][col]) > 0:
                    values = np.array(host_stats[ip][col])
                    # Mean, Std, Max, Min, Sum
                    features.extend([
                        np.mean(values),
                        np.std(values),
                        np.max(values),
                        np.min(values),
                        np.sum(values)
                    ])
                else:
                    # No data, use zeros
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            self.host_features[host_id] = features

        n_features = len(self.host_features[0]) if 0 in self.host_features else 0
        print(f"✓ Host features shape: ({len(self.host_to_id)}, {n_features})")

    def build_edges_and_labels(self, df, edge_feature_cols):
        """
        Build edges từ flows và assign labels cho hosts

        Args:
            df: DataFrame
            edge_feature_cols: Features cho edges
        """
        print("\nBuilding edges and labels...")

        # Convert numeric columns
        for col in edge_feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Track labels for each host
        host_label_counts = defaultdict(lambda: defaultdict(int))

        # Build edges
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing flows"):
            src_ip = row['Src IP']
            dst_ip = row['Dst IP']
            label = row['Label']

            # Skip if IPs not in mapping
            if src_ip not in self.host_to_id or dst_ip not in self.host_to_id:
                continue

            src_id = self.host_to_id[src_ip]
            dst_id = self.host_to_id[dst_ip]

            # Extract edge features
            edge_feats = []
            for col in edge_feature_cols:
                if col in row and not pd.isna(row[col]):
                    edge_feats.append(float(row[col]))
                else:
                    edge_feats.append(0.0)

            # Add edge (directed)
            self.edges.append((src_id, dst_id, edge_feats))

            # Track labels (majority vote)
            host_label_counts[src_id][label] += 1
            host_label_counts[dst_id][label] += 1

        # Assign labels to hosts (majority vote)
        for host_id in range(len(self.host_to_id)):
            if host_id in host_label_counts and len(host_label_counts[host_id]) > 0:
                # Get most common label
                most_common_label = max(host_label_counts[host_id].items(),
                                       key=lambda x: x[1])[0]
                self.host_labels[host_id] = most_common_label
            else:
                # Default to 'Benign' if no flows
                self.host_labels[host_id] = 'Benign'

        print(f"✓ Created {len(self.edges):,} directed edges")
        print(f"✓ Assigned labels to {len(self.host_labels)} hosts")

        # Print label distribution
        label_dist = defaultdict(int)
        for label in self.host_labels.values():
            label_dist[label] += 1
        print(f"  Host label distribution:")
        for label, count in sorted(label_dist.items(), key=lambda x: -x[1]):
            print(f"    {label}: {count}")

    def create_graph_data(self):
        """
        Tạo PyTorch Geometric Data object

        Returns:
            Data object với node features, edge index, edge features, và labels
        """
        print("\nCreating PyTorch Geometric graph...")

        # Node features
        n_nodes = len(self.host_to_id)
        n_node_features = len(self.host_features[0]) if 0 in self.host_features else 0

        node_features = np.zeros((n_nodes, n_node_features))
        for node_id in range(n_nodes):
            if node_id in self.host_features:
                node_features[node_id] = self.host_features[node_id]

        x = torch.tensor(node_features, dtype=torch.float)

        # Edge index and edge features
        edge_index = []
        edge_attr = []

        for src_id, dst_id, edge_feats in self.edges:
            edge_index.append([src_id, dst_id])
            edge_attr.append(edge_feats)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Labels (encode to integers)
        unique_labels = sorted(set(self.host_labels.values()))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

        y = torch.zeros(n_nodes, dtype=torch.long)
        for node_id, label in self.host_labels.items():
            y[node_id] = label_to_id[label]

        # Binary labels (Benign=0, Attack=1)
        y_binary = torch.zeros(n_nodes, dtype=torch.long)
        for node_id, label in self.host_labels.items():
            y_binary[node_id] = 0 if label == 'Benign' else 1

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            y_binary=y_binary
        )

        print(f"✓ Graph created:")
        print(f"  Nodes: {data.num_nodes:,}")
        print(f"  Edges: {data.num_edges:,}")
        print(f"  Node features: {data.num_node_features}")
        print(f"  Edge features: {edge_attr.shape[1] if edge_attr.shape[0] > 0 else 0}")
        print(f"  Classes: {len(unique_labels)} ({', '.join(unique_labels)})")
        print(f"  Binary labels: 0=Benign, 1=Attack")

        # Store metadata
        metadata = {
            'n_nodes': data.num_nodes,
            'n_edges': data.num_edges,
            'n_node_features': data.num_node_features,
            'n_edge_features': edge_attr.shape[1] if edge_attr.shape[0] > 0 else 0,
            'n_classes': len(unique_labels),
            'labels': unique_labels,
            'label_to_id': label_to_id,
            'host_to_id': self.host_to_id,
            'id_to_host': self.id_to_host,
            'avg_degree': data.num_edges / data.num_nodes,
            'graph_type': 'host_connection',
            'use_ports': self.use_ports,
            'min_flow_count': self.min_flow_count
        }

        return data, metadata


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def build_host_graph_dataset():
    """Xây dựng Host-Connection Graph từ network traffic data"""

    print("=" * 80)
    print("BUILDING HOST-CONNECTION GRAPH FOR GNN")
    print("=" * 80)

    # Create output directory
    os.makedirs(GRAPH_DATA_DIR, exist_ok=True)

    # Initialize builder
    builder = HostConnectionGraphBuilder(
        use_ports=USE_PORTS,
        min_flow_count=MIN_FLOW_COUNT
    )

    # Load data
    csv_path = os.path.join(DATA_DIR, CSV_FILE)
    df = builder.load_and_prepare_data(csv_path, max_flows=MAX_FLOWS)

    # Build host mapping
    valid_ips = builder.build_host_mapping(df)

    # Aggregate features for each host
    builder.aggregate_host_features(df, AGGREGATE_FEATURES)

    # Build edges and assign labels
    builder.build_edges_and_labels(df, EDGE_FEATURES)

    # Create graph data
    graph_data, metadata = builder.create_graph_data()

    # Save graph
    print("\nSaving graph data...")
    output_file = os.path.join(GRAPH_DATA_DIR, "host_graph.pt")
    torch.save(graph_data, output_file, _use_new_zipfile_serialization=True)
    print(f"✓ Graph saved to: {output_file}")

    # Save metadata
    metadata_file = os.path.join(GRAPH_DATA_DIR, "host_graph_metadata.pkl")
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Metadata saved to: {metadata_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("HOST-CONNECTION GRAPH CONSTRUCTION COMPLETED!")
    print("=" * 80)
    print(f"Graph Type: Host-Connection Graph (IP-Flow Graph)")
    print(f"Nodes (Hosts): {metadata['n_nodes']:,}")
    print(f"Edges (Flows): {metadata['n_edges']:,}")
    print(f"Node Features: {metadata['n_node_features']} (aggregated from flows)")
    print(f"Edge Features: {metadata['n_edge_features']} (flow-level)")
    print(f"Classes: {metadata['n_classes']}")
    print(f"Average Degree: {metadata['avg_degree']:.2f}")
    print(f"Output: {GRAPH_DATA_DIR}")
    print("=" * 80)

    return graph_data, metadata


# ============================================================================
# VISUALIZATION & ANALYSIS
# ============================================================================

def analyze_graph(graph_data, metadata):
    """Phân tích graph structure"""

    print("\n" + "=" * 80)
    print("GRAPH ANALYSIS")
    print("=" * 80)

    # Basic stats
    print(f"\nBasic Statistics:")
    print(f"  Nodes: {graph_data.num_nodes:,}")
    print(f"  Edges: {graph_data.num_edges:,}")
    print(f"  Average degree: {metadata['avg_degree']:.2f}")

    # Degree distribution
    edge_index = graph_data.edge_index
    in_degrees = torch.bincount(edge_index[1], minlength=graph_data.num_nodes)
    out_degrees = torch.bincount(edge_index[0], minlength=graph_data.num_nodes)

    print(f"\nDegree Distribution:")
    print(f"  In-degree  - Mean: {in_degrees.float().mean():.2f}, "
          f"Max: {in_degrees.max()}, Min: {in_degrees.min()}")
    print(f"  Out-degree - Mean: {out_degrees.float().mean():.2f}, "
          f"Max: {out_degrees.max()}, Min: {out_degrees.min()}")

    # Label distribution
    print(f"\nLabel Distribution:")
    for label, label_id in sorted(metadata['label_to_id'].items(), key=lambda x: x[1]):
        count = (graph_data.y == label_id).sum().item()
        pct = 100 * count / graph_data.num_nodes
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    # Binary label distribution
    benign_count = (graph_data.y_binary == 0).sum().item()
    attack_count = (graph_data.y_binary == 1).sum().item()
    print(f"\nBinary Classification:")
    print(f"  Benign: {benign_count:,} ({100*benign_count/graph_data.num_nodes:.1f}%)")
    print(f"  Attack: {attack_count:,} ({100*attack_count/graph_data.num_nodes:.1f}%)")

    # Feature statistics
    print(f"\nNode Feature Statistics:")
    print(f"  Shape: {graph_data.x.shape}")
    print(f"  Mean: {graph_data.x.mean():.4f}")
    print(f"  Std: {graph_data.x.std():.4f}")
    print(f"  Min: {graph_data.x.min():.4f}")
    print(f"  Max: {graph_data.x.max():.4f}")

    if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
        print(f"\nEdge Feature Statistics:")
        print(f"  Shape: {graph_data.edge_attr.shape}")
        print(f"  Mean: {graph_data.edge_attr.mean():.4f}")
        print(f"  Std: {graph_data.edge_attr.std():.4f}")

    print("=" * 80)


if __name__ == "__main__":
    # Build graph
    graph_data, metadata = build_host_graph_dataset()

    # Analyze graph
    analyze_graph(graph_data, metadata)

