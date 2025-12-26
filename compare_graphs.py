"""
Compare Similarity-Based Graph vs Host-Connection Graph
Script để so sánh hai cách tiếp cận xây dựng graph
"""

import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# ============================================================================
# PATHS
# ============================================================================
GRAPH_DATA_DIR = r"D:\PROJECT\Machine Learning\IOT\graph_data"
RESULTS_DIR = r"D:\PROJECT\Machine Learning\IOT\results"

# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def load_graphs():
    """Load both types of graphs"""

    graphs = {}

    # Similarity-based graph (original)
    try:
        if os.path.exists(os.path.join(GRAPH_DATA_DIR, "graph_binary.pt")):
            graphs['similarity'] = {
                'data': torch.load(os.path.join(GRAPH_DATA_DIR, "graph_binary.pt"), weights_only=False),
                'metadata': None
            }
            if os.path.exists(os.path.join(GRAPH_DATA_DIR, "graph_metadata.pkl")):
                with open(os.path.join(GRAPH_DATA_DIR, "graph_metadata.pkl"), 'rb') as f:
                    graphs['similarity']['metadata'] = pickle.load(f)
            print("✓ Loaded similarity-based graph")
    except Exception as e:
        print(f"⚠ Could not load similarity graph: {e}")

    # Host-connection graph
    try:
        if os.path.exists(os.path.join(GRAPH_DATA_DIR, "host_graph.pt")):
            graphs['host'] = {
                'data': torch.load(os.path.join(GRAPH_DATA_DIR, "host_graph.pt"), weights_only=False),
                'metadata': None
            }
            if os.path.exists(os.path.join(GRAPH_DATA_DIR, "host_graph_metadata.pkl")):
                with open(os.path.join(GRAPH_DATA_DIR, "host_graph_metadata.pkl"), 'rb') as f:
                    graphs['host']['metadata'] = pickle.load(f)
            print("✓ Loaded host-connection graph")
    except Exception as e:
        print(f"⚠ Could not load host graph: {e}")

    return graphs


def compute_graph_statistics(graph_data):
    """Compute detailed graph statistics"""

    stats = {}

    # Basic stats
    stats['num_nodes'] = graph_data.num_nodes
    stats['num_edges'] = graph_data.num_edges
    stats['num_features'] = graph_data.num_node_features
    stats['avg_degree'] = graph_data.num_edges / graph_data.num_nodes

    # Degree distribution
    edge_index = graph_data.edge_index
    degrees = torch.bincount(edge_index[0], minlength=graph_data.num_nodes)

    stats['degree_mean'] = degrees.float().mean().item()
    stats['degree_std'] = degrees.float().std().item()
    stats['degree_min'] = degrees.min().item()
    stats['degree_max'] = degrees.max().item()
    stats['degree_median'] = degrees.float().median().item()

    # Label distribution
    if hasattr(graph_data, 'y'):
        unique, counts = torch.unique(graph_data.y, return_counts=True)
        stats['num_classes'] = len(unique)
        stats['class_distribution'] = {int(u): int(c) for u, c in zip(unique, counts)}

        # Class imbalance ratio
        if len(counts) == 2:
            stats['imbalance_ratio'] = max(counts).item() / min(counts).item()

    # Feature statistics
    if hasattr(graph_data, 'x'):
        stats['feature_mean'] = graph_data.x.mean().item()
        stats['feature_std'] = graph_data.x.std().item()
        stats['feature_min'] = graph_data.x.min().item()
        stats['feature_max'] = graph_data.x.max().item()

    # Edge attributes (if available)
    if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
        stats['num_edge_features'] = graph_data.edge_attr.shape[1]
        stats['edge_feature_mean'] = graph_data.edge_attr.mean().item()
        stats['edge_feature_std'] = graph_data.edge_attr.std().item()
    else:
        stats['num_edge_features'] = 0

    return stats


def create_comparison_table(graphs):
    """Create comparison table"""

    print("\n" + "=" * 100)
    print("GRAPH COMPARISON TABLE")
    print("=" * 100)

    comparison_data = []

    for graph_type, graph_info in graphs.items():
        stats = compute_graph_statistics(graph_info['data'])
        metadata = graph_info['metadata']

        row = {
            'Graph Type': 'Similarity-Based (k-NN)' if graph_type == 'similarity' else 'Host-Connection (IP-Flow)',
            'Nodes': f"{stats['num_nodes']:,}",
            'Edges': f"{stats['num_edges']:,}",
            'Node Features': stats['num_features'],
            'Edge Features': stats.get('num_edge_features', 0),
            'Avg Degree': f"{stats['avg_degree']:.2f}",
            'Max Degree': stats['degree_max'],
            'Classes': stats.get('num_classes', 'N/A'),
            'Graph Density': f"{stats['num_edges'] / (stats['num_nodes'] * (stats['num_nodes'] - 1)):.6f}" if stats['num_nodes'] > 1 else 'N/A'
        }

        comparison_data.append(row)

    # Print table
    if comparison_data:
        headers = comparison_data[0].keys()
        rows = [[row[h] for h in headers] for row in comparison_data]
        print(tabulate(rows, headers=headers, tablefmt='grid'))

    print("=" * 100)


def plot_degree_distributions(graphs, output_path):
    """Plot degree distributions for both graphs"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for idx, (graph_type, graph_info) in enumerate(graphs.items()):
        data = graph_info['data']
        edge_index = data.edge_index

        # Compute degrees
        degrees = torch.bincount(edge_index[0], minlength=data.num_nodes).numpy()

        # Plot
        ax = axes[idx]
        ax.hist(degrees, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_yscale('log')

        title = 'Similarity-Based Graph' if graph_type == 'similarity' else 'Host-Connection Graph'
        ax.set_title(f'{title}\n(Mean: {degrees.mean():.2f}, Max: {degrees.max()})')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Degree distribution plot saved to: {output_path}")


def plot_feature_distributions(graphs, output_path):
    """Plot feature value distributions"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for idx, (graph_type, graph_info) in enumerate(graphs.items()):
        data = graph_info['data']

        if hasattr(data, 'x'):
            # Sample features to plot
            features_flat = data.x.flatten().numpy()

            # Remove outliers for better visualization
            q1, q99 = np.percentile(features_flat, [1, 99])
            features_filtered = features_flat[(features_flat >= q1) & (features_flat <= q99)]

            # Plot
            ax = axes[idx]
            ax.hist(features_filtered, bins=100, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Frequency')
            ax.set_yscale('log')

            title = 'Similarity-Based Graph' if graph_type == 'similarity' else 'Host-Connection Graph'
            ax.set_title(f'{title}\n(Mean: {features_flat.mean():.2f}, Std: {features_flat.std():.2f})')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Feature distribution plot saved to: {output_path}")


def print_advantages_comparison():
    """Print advantages of each approach"""

    print("\n" + "=" * 100)
    print("ADVANTAGES COMPARISON")
    print("=" * 100)

    print("\n📊 Similarity-Based Graph (k-NN)")
    print("-" * 100)
    advantages_similarity = [
        "✓ Works with any dataset (no IP information needed)",
        "✓ Captures feature-space similarity",
        "✓ Good for detecting novel attacks (similar patterns)",
        "✓ Standard ML approach",
        "✓ Easier to tune (just k parameter)"
    ]
    for adv in advantages_similarity:
        print(f"  {adv}")

    disadvantages_similarity = [
        "✗ No network topology information",
        "✗ Computationally expensive (O(n²) or O(nk))",
        "✗ Edges are abstract (not real connections)",
        "✗ Hard to interpret results",
        "✗ Doesn't scale well to large datasets"
    ]
    print("\n  Limitations:")
    for dis in disadvantages_similarity:
        print(f"  {dis}")

    print("\n🌐 Host-Connection Graph (IP-Flow)")
    print("-" * 100)
    advantages_host = [
        "✓ Models real network topology",
        "✓ Preserves network structure and communication patterns",
        "✓ Highly interpretable (nodes = IPs, edges = flows)",
        "✓ Scales better (# nodes << # flows)",
        "✓ Can use graph metrics (centrality, communities)",
        "✓ Detects network-level attacks (C&C, DDoS, lateral movement)",
        "✓ Actionable insights (identify malicious IPs)"
    ]
    for adv in advantages_host:
        print(f"  {adv}")

    disadvantages_host = [
        "✗ Requires IP information in dataset",
        "✗ Need to aggregate flow features",
        "✗ Label assignment can be noisy (majority voting)",
        "✗ May miss individual malicious flows",
        "✗ Graph structure depends on observation period"
    ]
    print("\n  Limitations:")
    for dis in disadvantages_host:
        print(f"  {dis}")

    print("=" * 100)


def print_use_case_recommendations():
    """Print recommendations for when to use each approach"""

    print("\n" + "=" * 100)
    print("USE CASE RECOMMENDATIONS")
    print("=" * 100)

    print("\n🎯 When to use Similarity-Based Graph:")
    print("-" * 100)
    use_cases_similarity = [
        "• Dataset doesn't have IP address information",
        "• Focus on individual flow classification",
        "• Need to detect anomalies based on feature patterns",
        "• Small to medium datasets (<100K flows)",
        "• Research/experimental setups"
    ]
    for uc in use_cases_similarity:
        print(f"  {uc}")

    print("\n🎯 When to use Host-Connection Graph:")
    print("-" * 100)
    use_cases_host = [
        "• Dataset has IP address information",
        "• Need to identify malicious hosts/IPs",
        "• Want to leverage network topology",
        "• Large-scale networks (millions of flows)",
        "• Production IDS deployment",
        "• Need interpretable results for SOC analysts",
        "• Detect network-level attacks (botnets, DDoS, C&C)",
        "• Want to track attacker behavior across time"
    ]
    for uc in use_cases_host:
        print(f"  {uc}")

    print("=" * 100)


def compare_complexity():
    """Compare computational complexity"""

    print("\n" + "=" * 100)
    print("COMPUTATIONAL COMPLEXITY COMPARISON")
    print("=" * 100)

    complexity_table = [
        {
            'Operation': 'Graph Construction',
            'Similarity-Based': 'O(n² * d) or O(n * k * d)',
            'Host-Connection': 'O(m)',
            'Note': 'n=flows, m=flows, k=neighbors, d=features'
        },
        {
            'Operation': 'Memory (Adjacency)',
            'Similarity-Based': 'O(n * k)',
            'Host-Connection': 'O(m)',
            'Note': 'k << n typically, m can be large'
        },
        {
            'Operation': 'GNN Training (per epoch)',
            'Similarity-Based': 'O(|E| * d * h)',
            'Host-Connection': 'O(|E| * d * h)',
            'Note': '|E|=edges, d=features, h=hidden'
        },
        {
            'Operation': 'Inference',
            'Similarity-Based': 'O(n) for n flows',
            'Host-Connection': 'O(h) for h hosts',
            'Note': 'h << n in most networks'
        }
    ]

    headers = ['Operation', 'Similarity-Based', 'Host-Connection', 'Note']
    rows = [[row[h] for h in headers] for row in complexity_table]
    print(tabulate(rows, headers=headers, tablefmt='grid'))

    print("\n💡 Key Insight:")
    print("  Host-Connection Graph is more efficient for large datasets because:")
    print("  • # nodes (hosts) << # flows in typical networks")
    print("  • Graph construction is linear in # flows")
    print("  • GNN operates on much smaller graph")

    print("=" * 100)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main comparison function"""

    print("=" * 100)
    print("GRAPH APPROACH COMPARISON")
    print("Similarity-Based (k-NN) vs Host-Connection (IP-Flow)")
    print("=" * 100)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load graphs
    print("\nLoading graphs...")
    graphs = load_graphs()

    if not graphs:
        print("⚠ No graphs found. Please build graphs first:")
        print("  - python build_graph.py  (for similarity-based)")
        print("  - python build_host_graph.py  (for host-connection)")
        return

    # Create comparison table
    create_comparison_table(graphs)

    # Plot comparisons
    if len(graphs) >= 2:
        print("\nGenerating comparison plots...")

        # Degree distributions
        degree_plot_path = os.path.join(RESULTS_DIR, "graph_comparison_degrees.png")
        plot_degree_distributions(graphs, degree_plot_path)

        # Feature distributions
        feature_plot_path = os.path.join(RESULTS_DIR, "graph_comparison_features.png")
        plot_feature_distributions(graphs, feature_plot_path)

    # Print advantages
    print_advantages_comparison()

    # Print use cases
    print_use_case_recommendations()

    # Compare complexity
    compare_complexity()

    print("\n" + "=" * 100)
    print("COMPARISON COMPLETED!")
    print("=" * 100)
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

