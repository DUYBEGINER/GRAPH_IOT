"""
Graph Visualization Script
Visualize graph structure cho GNN
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import os


def visualize_graph_structure(edge_index, node_features=None, node_labels=None,
                              max_nodes=100, output_path='graph_structure.png'):
    """
    Visualize graph structure

    Args:
        edge_index: PyG edge_index tensor (2, num_edges)
        node_features: Node feature matrix (optional)
        node_labels: Node labels (optional)
        max_nodes: Maximum nodes to visualize
        output_path: Output file path
    """
    print(f"🎨 Visualizing graph structure...")

    # Convert to numpy
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    # Sample nodes if too many
    num_nodes = edge_index.max() + 1
    if num_nodes > max_nodes:
        print(f"⚠️  Too many nodes ({num_nodes}), sampling {max_nodes} nodes")
        sample_nodes = np.random.choice(num_nodes, size=max_nodes, replace=False)
        # Filter edges
        mask = np.isin(edge_index[0], sample_nodes) & np.isin(edge_index[1], sample_nodes)
        edge_index = edge_index[:, mask]
        # Remap node indices
        node_map = {old: new for new, old in enumerate(sample_nodes)}
        edge_index = np.array([[node_map[e] for e in edge_index[0]],
                               [node_map[e] for e in edge_index[1]]])
        num_nodes = max_nodes

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)

    # Layout
    print(f"📐 Computing layout...")
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Node colors
    if node_labels is not None:
        if len(node_labels) > max_nodes:
            node_labels = node_labels[sample_nodes]
        node_colors = node_labels
        cmap = plt.cm.RdYlBu
    else:
        node_colors = 'lightblue'
        cmap = None

    # Plot
    plt.figure(figsize=(15, 12))

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                          node_size=100, alpha=0.8, cmap=cmap)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)

    plt.title(f'Graph Structure ({num_nodes} nodes, {len(edges)} edges)',
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    plt.close()


def visualize_node_embeddings(node_features, node_labels=None, method='tsne',
                              output_path='node_embeddings.png'):
    """
    Visualize node embeddings using dimensionality reduction

    Args:
        node_features: Node feature matrix (n_nodes, n_features)
        node_labels: Node labels (optional)
        method: 'tsne' or 'pca'
        output_path: Output file path
    """
    print(f"🎨 Visualizing node embeddings using {method.upper()}...")

    # Convert to numpy
    if isinstance(node_features, torch.Tensor):
        node_features = node_features.cpu().numpy()
    if isinstance(node_labels, torch.Tensor):
        node_labels = node_labels.cpu().numpy()

    # Sample if too many nodes
    if len(node_features) > 5000:
        indices = np.random.choice(len(node_features), size=5000, replace=False)
        node_features = node_features[indices]
        if node_labels is not None:
            node_labels = node_labels[indices]

    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2, random_state=42)

    embeddings_2d = reducer.fit_transform(node_features)

    # Plot
    plt.figure(figsize=(12, 10))

    if node_labels is not None:
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=node_labels, cmap='RdYlBu', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Label')
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                   alpha=0.6, s=20, c='lightblue')

    plt.title(f'Node Embeddings ({method.upper()})', fontsize=16, fontweight='bold')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    plt.close()


def visualize_graph_statistics(edge_index, node_labels=None, output_path='graph_stats.png'):
    """
    Visualize graph statistics (degree distribution, etc.)

    Args:
        edge_index: PyG edge_index tensor
        node_labels: Node labels (optional)
        output_path: Output file path
    """
    print(f"📊 Computing graph statistics...")

    # Convert to numpy
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    num_nodes = edge_index.max() + 1

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)

    # Compute statistics
    degrees = [G.degree(n) for n in G.nodes()]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Degree distribution
    axes[0, 0].hist(degrees, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Degree')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Degree Distribution')
    axes[0, 0].grid(alpha=0.3)

    # Degree statistics
    axes[0, 1].text(0.1, 0.9, f"Number of Nodes: {num_nodes:,}", fontsize=12, transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.8, f"Number of Edges: {len(edges):,}", fontsize=12, transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.7, f"Average Degree: {np.mean(degrees):.2f}", fontsize=12, transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.6, f"Min Degree: {min(degrees)}", fontsize=12, transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.5, f"Max Degree: {max(degrees)}", fontsize=12, transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.4, f"Median Degree: {np.median(degrees):.2f}", fontsize=12, transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.3, f"Density: {nx.density(G):.6f}", fontsize=12, transform=axes[0, 1].transAxes)
    axes[0, 1].set_title('Graph Statistics')
    axes[0, 1].axis('off')

    # Label distribution (if available)
    if node_labels is not None:
        if isinstance(node_labels, torch.Tensor):
            node_labels = node_labels.cpu().numpy()
        unique, counts = np.unique(node_labels, return_counts=True)
        axes[1, 0].bar(unique, counts, color='lightcoral', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Label')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Label Distribution')
        axes[1, 0].grid(alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No labels provided',
                       fontsize=14, ha='center', va='center',
                       transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')

    # Degree per label (if available)
    if node_labels is not None:
        degree_by_label = {}
        for node, label in enumerate(node_labels):
            if label not in degree_by_label:
                degree_by_label[label] = []
            degree_by_label[label].append(degrees[node])

        labels = sorted(degree_by_label.keys())
        degree_means = [np.mean(degree_by_label[l]) for l in labels]

        axes[1, 1].bar(labels, degree_means, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Label')
        axes[1, 1].set_ylabel('Average Degree')
        axes[1, 1].set_title('Average Degree per Label')
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No labels provided',
                       fontsize=14, ha='center', va='center',
                       transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    plt.close()


def demo_visualization():
    """Demo visualization với graph data"""

    print("\n" + "=" * 80)
    print("🎨 GRAPH VISUALIZATION DEMO")
    print("=" * 80 + "\n")

    # Check if graph exists
    graph_files = ['graph_binary.pt', 'graph_multi.pt']
    available_graphs = []

    for gf in graph_files:
        if os.path.exists(gf):
            available_graphs.append(gf)

    if not available_graphs:
        print("⚠️  No pre-built graph found. Creating sample graph...")

        # Load processed data
        import sys
        sys.path.append('')

        X = np.load('../processed_data/X_features.npy')[:1000]
        y = np.load('../processed_data/y_binary.npy')[:1000]

        print(f"✅ Loaded {len(X)} samples")

        # Build simple graph
        from inference_gnn import SimpleGraphBuilder
        builder = SimpleGraphBuilder(k_neighbors=8)
        edge_index = builder.build_knn_graph(X)

        node_features = torch.FloatTensor(X)
        node_labels = torch.LongTensor(y)

    else:
        # Load existing graph
        graph_path = available_graphs[0]
        print(f"📂 Loading graph from {graph_path}...")

        data = torch.load(graph_path, map_location='cpu', weights_only=False)
        edge_index = data.edge_index
        node_features = data.x
        node_labels = data.y if hasattr(data, 'y') else None

        print(f"✅ Graph loaded: {data.num_nodes} nodes, {data.num_edges} edges")

    # Create output directory
    output_dir = './visualization_results'
    os.makedirs(output_dir, exist_ok=True)

    # Visualizations
    print("\n📊 Creating visualizations...")

    # 1. Graph structure
    visualize_graph_structure(
        edge_index,
        node_features,
        node_labels,
        max_nodes=200,
        output_path=os.path.join(output_dir, 'graph_structure.png')
    )

    # 2. Node embeddings (t-SNE)
    visualize_node_embeddings(
        node_features,
        node_labels,
        method='tsne',
        output_path=os.path.join(output_dir, 'embeddings_tsne.png')
    )

    # 3. Node embeddings (PCA)
    visualize_node_embeddings(
        node_features,
        node_labels,
        method='pca',
        output_path=os.path.join(output_dir, 'embeddings_pca.png')
    )

    # 4. Graph statistics
    visualize_graph_statistics(
        edge_index,
        node_labels,
        output_path=os.path.join(output_dir, 'graph_statistics.png')
    )

    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETED!")
    print("=" * 80)
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    demo_visualization()

