"""
Master Pipeline Script for IoT Anomaly Detection with GNN
Script chính để chạy toàn bộ quy trình: preprocess -> build graph -> train GNN
"""

import os
import sys
import time
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Pipeline steps to run
RUN_PREPROCESSING = True
RUN_GRAPH_BUILDING = True
RUN_TRAINING = True

# ============================================================================
# MASTER PIPELINE
# ============================================================================

def print_banner(text):
    """Print a fancy banner"""
    width = 80
    print("\n")
    print("╔" + "=" * (width - 2) + "╗")
    padding = (width - len(text) - 2) // 2
    print("║" + " " * padding + text + " " * (width - len(text) - padding - 2) + "║")
    print("╚" + "=" * (width - 2) + "╝")
    print("\n")


def run_step(step_name, script_name):
    """Run a pipeline step"""
    print_banner(f"STEP: {step_name}")

    print(f"Running: {script_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    start_time = time.time()

    try:
        # Import and run the script
        if script_name == 'preprocess_data.py':
            import preprocess_data
            preprocess_data.main()
        elif script_name == 'build_graph.py':
            import build_graph
            build_graph.build_graph_dataset()
        elif script_name == 'train_gnn.py':
            import train_gnn
            train_gnn.main()

        elapsed_time = time.time() - start_time

        print("-" * 80)
        print(f"✓ {step_name} completed successfully!")
        print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print("-" * 80)

        return True

    except Exception as e:
        elapsed_time = time.time() - start_time

        print("-" * 80)
        print(f"✗ {step_name} failed!")
        print(f"Error: {e}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print("-" * 80)

        import traceback
        traceback.print_exc()

        return False


def main():
    """Main pipeline execution"""

    print("\n" + "=" * 80)
    print(" " * 10 + "IOT ANOMALY DETECTION WITH GNN - MASTER PIPELINE")
    print("=" * 80)
    print(f"\nPipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPipeline steps:")
    print(f"  1. Data Preprocessing: {'YES' if RUN_PREPROCESSING else 'SKIP'}")
    print(f"  2. Graph Building: {'YES' if RUN_GRAPH_BUILDING else 'SKIP'}")
    print(f"  3. GNN Training: {'YES' if RUN_TRAINING else 'SKIP'}")
    print("\n" + "=" * 80)

    overall_start_time = time.time()
    results = {}

    # Step 1: Data Preprocessing
    if RUN_PREPROCESSING:
        success = run_step("DATA PREPROCESSING", "preprocess_data.py")
        results['preprocessing'] = success
        if not success:
            print("\n⚠ Preprocessing failed. Stopping pipeline.")
            return

    # Step 2: Graph Building
    if RUN_GRAPH_BUILDING:
        success = run_step("GRAPH CONSTRUCTION", "build_graph.py")
        results['graph_building'] = success
        if not success:
            print("\n⚠ Graph building failed. Stopping pipeline.")
            return

    # Step 3: GNN Training
    if RUN_TRAINING:
        success = run_step("GNN TRAINING", "train_gnn.py")
        results['training'] = success

    # Summary
    overall_elapsed = time.time() - overall_start_time

    print_banner("PIPELINE SUMMARY")

    print("Results:")
    for step, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {step.replace('_', ' ').title():20s}: {status}")

    print(f"\nTotal time: {overall_elapsed:.2f} seconds ({overall_elapsed/60:.2f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_success = all(results.values())

    if all_success:
        print("\n" + "=" * 80)
        print("🎉 ALL PIPELINE STEPS COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 80)
        print("\nYour GNN model is ready for IoT anomaly detection!")
        print("\nOutput locations:")
        print("  - Processed data: D:\\PROJECT\\Machine Learning\\IOT\\processed_data")
        print("  - Graph data: D:\\PROJECT\\Machine Learning\\IOT\\graph_data")
        print("  - Trained models: D:\\PROJECT\\Machine Learning\\IOT\\models")
        print("  - Results: D:\\PROJECT\\Machine Learning\\IOT\\results")
    else:
        print("\n" + "=" * 80)
        print("⚠ PIPELINE COMPLETED WITH ERRORS")
        print("=" * 80)
        print("\nPlease check the error messages above and fix the issues.")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

