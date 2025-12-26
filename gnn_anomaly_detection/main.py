"""
Main Pipeline Script for GNN Anomaly Detection
===============================================
Script này chạy toàn bộ pipeline từ preprocessing đến training và evaluation.

Usage:
    python main.py                    # Chạy toàn bộ pipeline
    python main.py --model GCN        # Chỉ định loại model
    python main.py --sample 100000    # Giới hạn số samples
    python main.py --skip-preprocess  # Bỏ qua bước preprocessing (dùng data đã có)
"""

import argparse
import os
import sys
import time
import torch

# Import các modules
import config
from data_preprocessing import preprocess_pipeline, load_preprocessed_data
from graph_construction import build_graph_from_splits, save_graph, load_graph, print_graph_info
from models import get_model, print_model_summary, count_parameters
from train import train, test, save_results
from evaluate import (plot_training_history, plot_confusion_matrix,
                      plot_all_curves, print_detailed_metrics,
                      create_summary_report, generate_classification_report)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GNN Anomaly Detection Pipeline')

    parser.add_argument('--model', type=str, default='GCN',
                       choices=['GCN', 'GAT', 'GraphSAGE'],
                       help='Loại GNN model (default: GCN)')

    parser.add_argument('--sample', type=int, default=None,
                       help='Số lượng samples để sử dụng (default: từ config)')

    parser.add_argument('--k-neighbors', type=int, default=None,
                       help='Số neighbors cho KNN graph (default: từ config)')

    parser.add_argument('--epochs', type=int, default=None,
                       help='Số epochs để train (default: từ config)')

    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: từ config)')

    parser.add_argument('--hidden-dim', type=int, default=None,
                       help='Hidden dimension size (default: từ config)')

    parser.add_argument('--skip-preprocess', action='store_true',
                       help='Bỏ qua preprocessing, dùng data đã có')

    parser.add_argument('--skip-graph', action='store_true',
                       help='Bỏ qua xây dựng graph, dùng graph đã có')

    parser.add_argument('--no-save', action='store_true',
                       help='Không lưu results và model')

    return parser.parse_args()


def main():
    """Main function chạy toàn bộ pipeline."""
    args = parse_args()

    print("\n" + "="*70)
    print("   GNN ANOMALY DETECTION PIPELINE")
    print("   Network Traffic Binary Classification")
    print("="*70)

    start_time = time.time()

    # =========================================================================
    # STEP 1: DATA PREPROCESSING
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)

    preprocessed_path = os.path.join(config.OUTPUT_DIR, 'preprocessed_data.pkl')

    if args.skip_preprocess and os.path.exists(preprocessed_path):
        print("[INFO] Loading existing preprocessed data...")
        preprocessed = load_preprocessed_data(preprocessed_path)
    else:
        sample_size = args.sample if args.sample else config.SAMPLE_SIZE
        preprocessed = preprocess_pipeline(sample_size=sample_size)

    # =========================================================================
    # STEP 2: GRAPH CONSTRUCTION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: GRAPH CONSTRUCTION")
    print("="*70)

    graph_path = os.path.join(config.OUTPUT_DIR, 'graph_data.pt')

    if args.skip_graph and os.path.exists(graph_path):
        print("[INFO] Loading existing graph...")
        data = load_graph(graph_path)
    else:
        k = args.k_neighbors if args.k_neighbors else config.K_NEIGHBORS
        data = build_graph_from_splits(
            preprocessed['X_train'], preprocessed['X_val'], preprocessed['X_test'],
            preprocessed['y_train'], preprocessed['y_val'], preprocessed['y_test'],
            k=k
        )

        if not args.no_save:
            save_graph(data, graph_path)

    # In thông tin graph
    print_graph_info(data)

    # =========================================================================
    # STEP 3: MODEL INITIALIZATION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: MODEL INITIALIZATION")
    print("="*70)

    input_dim = data.num_node_features
    hidden_dim = args.hidden_dim if args.hidden_dim else config.HIDDEN_DIM

    model = get_model(
        model_type=args.model,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=2,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )

    print_model_summary(model, input_dim)

    # =========================================================================
    # STEP 4: TRAINING
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: TRAINING")
    print("="*70)

    num_epochs = args.epochs if args.epochs else config.NUM_EPOCHS
    lr = args.lr if args.lr else config.LEARNING_RATE

    history = train(
        model=model,
        data=data,
        num_epochs=num_epochs,
        lr=lr,
        save_best=not args.no_save
    )

    # =========================================================================
    # STEP 5: TESTING
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: TESTING")
    print("="*70)

    test_metrics = test(model, data)

    # =========================================================================
    # STEP 6: EVALUATION & VISUALIZATION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: EVALUATION & VISUALIZATION")
    print("="*70)

    if not args.no_save:
        # Plot training history
        plot_training_history(history)

        # Plot confusion matrix
        plot_confusion_matrix(test_metrics['confusion_matrix'])

        # Plot ROC và PR curves
        plot_all_curves(model, data)

        # Save results
        save_results(history, test_metrics)

        # Model info cho report
        model_info = {
            'type': args.model,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': config.NUM_LAYERS,
            'num_params': count_parameters(model)
        }

        # Create summary report
        create_summary_report(history, test_metrics, model_info)

        # Generate classification report
        device = torch.device(config.DEVICE)
        model = model.to(device)
        data = data.to(device)
        model.eval()

        with torch.no_grad():
            out = model(data)
            pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
            y_true = data.y[data.test_mask].cpu().numpy()

        generate_classification_report(y_true, pred)

    # Print detailed metrics
    print_detailed_metrics(test_metrics)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("PIPELINE COMPLETED")
    print("="*70)
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nFinal Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {test_metrics['auc_roc']:.4f}")

    if not args.no_save:
        print(f"\nOutputs saved to: {config.OUTPUT_DIR}")
        print(f"Model saved to: {config.MODEL_DIR}")

    return history, test_metrics


if __name__ == "__main__":
    history, test_metrics = main()

