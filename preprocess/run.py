"""
Unified preprocessing pipeline for CICIDS2018 dataset.

Runs the complete 3-step pipeline:
1. Clean raw CSV files
2. Split and scale data
3. Build graph (for GNN modes)

Usage:
    python preprocess/run.py --mode flow_gnn
    python preprocess/run.py --mode ip_gnn --format csv
    python preprocess/run.py --steps clean split  # Run only specific steps
"""

import argparse
import logging
import time
import yaml
from pathlib import Path
from typing import Dict, Any, List

# Import preprocessing modules
from clean import clean_all
from split_scale import make_split_manifest
from build_graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_pipeline(
    config: Dict[str, Any],
    steps: List[str] = None,
    output_format: str = "parquet"
):
    """
    Run the complete preprocessing pipeline.
    
    Args:
        config: Configuration dictionary
        steps: List of steps to run ['clean', 'split', 'graph']
                If None, runs all applicable steps
        output_format: Format for cleaned files ('parquet' or 'csv')
    """
    start_time = time.time()
    
    if steps is None:
        steps = ['clean', 'split', 'graph']
    
    mode = config['mode']
    output_dir = config['paths']['output_dir']
    
    logger.info("\n" + "=" * 100)
    logger.info("🚀 PREPROCESSING PIPELINE STARTED")
    logger.info("=" * 100)
    logger.info(f"🎯 Mode:          {mode}")
    logger.info(f"📁 Output dir:    {output_dir}")
    logger.info(f"⚙️  Steps:         {' → '.join(steps)}")
    logger.info(f"💾 Format:        {output_format}")
    logger.info("=" * 100 + "\n")
    
    results = {}
    
    # Step 1: Clean
    if 'clean' in steps:
        logger.info("\n" + "🔧" * 50)
        logger.info("STEP 1/3: DATA CLEANING")
        logger.info("🔧" * 50 + "\n")
        
        step_start = time.time()
        
        # Determine input path based on mode
        if mode == 'ip_gnn':
            input_path = config['paths']['input_file_ip']
        else:
            input_path = config['paths']['input_dir_flow']
        
        try:
            metadata = clean_all(input_path, output_dir, config, output_format)
            results['clean'] = {
                'success': True,
                'time': time.time() - step_start,
                'rows_processed': metadata['total_rows_output']
            }
            logger.info(f"✅ Cleaning completed in {results['clean']['time']:.2f}s\n")
        except Exception as e:
            logger.error(f"❌ Cleaning failed: {e}")
            results['clean'] = {'success': False, 'error': str(e)}
            return results
    
    # Step 2: Split and Scale
    if 'split' in steps:
        logger.info("\n" + "✂️" * 50)
        logger.info("STEP 2/3: SPLIT AND SCALE")
        logger.info("✂️" * 50 + "\n")
        
        step_start = time.time()
        
        try:
            manifest = make_split_manifest(output_dir, output_dir, config)
            results['split'] = {
                'success': True,
                'time': time.time() - step_start,
                'samples': manifest['total_samples'],
                'features': manifest['num_features']
            }
            logger.info(f"✅ Split and scale completed in {results['split']['time']:.2f}s\n")
        except Exception as e:
            logger.error(f"❌ Split and scale failed: {e}")
            results['split'] = {'success': False, 'error': str(e)}
            return results
    
    # Step 3: Build Graph (only for GNN modes)
    if 'graph' in steps:
        if mode in ['flow_gnn', 'ip_gnn']:
            logger.info("\n" + "🕸️" * 50)
            logger.info("STEP 3/3: GRAPH BUILDING")
            logger.info("🕸️" * 50 + "\n")
            
            step_start = time.time()
            
            try:
                # Tạo đường dẫn cho file graph output
                graph_output = Path(output_dir) / f"graph_{mode}.pt"
                graph_data = build_graph(output_dir, str(graph_output), config)
                results['graph'] = {
                    'success': True,
                    'time': time.time() - step_start,
                    'nodes': graph_data.num_nodes if hasattr(graph_data, 'num_nodes') else 'N/A',
                    'edges': graph_data.num_edges if hasattr(graph_data, 'num_edges') else 'N/A'
                }
                logger.info(f"✅ Graph building completed in {results['graph']['time']:.2f}s\n")
            except Exception as e:
                logger.error(f"❌ Graph building failed: {e}")
                results['graph'] = {'success': False, 'error': str(e)}
                return results
        else:
            logger.info(f"\n⏭️  Skipping graph building (mode={mode} doesn't require graphs)\n")
            results['graph'] = {'success': True, 'skipped': True}
    
    # Final summary
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 100)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 100)
    
    for step_name, result in results.items():
        if result.get('success'):
            if result.get('skipped'):
                logger.info(f"⏭️  {step_name.upper()}: Skipped")
            else:
                logger.info(f"✅ {step_name.upper()}: {result.get('time', 0):.2f}s")
        else:
            logger.info(f"❌ {step_name.upper()}: Failed - {result.get('error', 'Unknown')}")
    
    logger.info(f"\n⏱️  Total pipeline time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    logger.info(f"📁 All outputs saved to: {Path(output_dir).absolute()}")
    logger.info("=" * 100 + "\n")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline for CICIDS2018 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for flow_gnn mode
  python preprocess/run.py --mode flow_gnn
  
  # Run only cleaning and splitting
  python preprocess/run.py --steps clean split
  
  # Use custom config and CSV output
  python preprocess/run.py --config my_config.yaml --format csv
  
  # Override output directory
  python preprocess/run.py --output_dir my_output
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="preprocess/config.yaml",
        help="Path to configuration file (default: preprocess/config.yaml)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=['flow_gnn', 'ip_gnn', 'cnn_lstm'],
        help="Override mode from config (flow_gnn, ip_gnn, cnn_lstm)"
    )
    
    parser.add_argument(
        "--steps",
        type=str,
        nargs='+',
        choices=['clean', 'split', 'graph'],
        help="Specific steps to run (default: all steps)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=['parquet', 'csv'],
        help="Output format for cleaned files (default: parquet)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Override input directory from config (for flow_gnn mode)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"📋 Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.mode:
        config['mode'] = args.mode
        logger.info(f"🎯 Mode overridden to: {args.mode}")
    
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
        logger.info(f"📁 Output directory overridden to: {args.output_dir}")
    
    if args.input_dir:
        config['paths']['input_dir_flow'] = args.input_dir
        logger.info(f"📁 Input directory overridden to: {args.input_dir}")
    
    # Run pipeline
    try:
        results = run_pipeline(config, args.steps, args.format)
        
        # Check if all steps succeeded
        all_success = all(r.get('success', False) for r in results.values())
        
        if all_success:
            logger.info("🎉 All steps completed successfully!")
            return 0
        else:
            logger.error("⚠️ Some steps failed. Check logs above.")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
