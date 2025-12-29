#!/usr/bin/env python3
"""
Run preprocessing pipeline for CICIDS dataset.

Usage examples:
    # Run all steps with default config
    python run.py
    
    # Run only cleaning step
    python run.py --steps clean
    
    # Run cleaning and splitting (skip graph building)
    python run.py --steps clean split
    
    # Use custom config file
    python run.py --config custom_config.yaml
    
    # Override mode from config
    python run.py --mode ip_gnn
"""

import argparse
import logging
import yaml
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "preprocess/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_cleaning(config: dict):
    """Run data cleaning step."""
    from clean import clean_all
    
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA CLEANING")
    logger.info("="*80)
    
    # Select input path based on mode
    if config['mode'] == 'ip_gnn':
        input_path = config['paths']['input_file_ip']
        logger.info(f"Mode: ip_gnn - Using single file: {input_path}")
    else:
        input_path = config['paths']['input_dir_flow']
        logger.info(f"Mode: {config['mode']} - Using directory: {input_path}")
    
    metadata = clean_all(
        input_dir=input_path,
        output_dir=config['paths']['output_dir'],
        config=config,
        output_format="parquet"
    )
    
    logger.info(f"✓ Cleaned data saved to: {config['paths']['output_dir']}")
    return metadata


def run_split_scale(config: dict):
    """Run split and scaling step."""
    from split_scale import make_split_manifest
    
    logger.info("\n" + "="*80)
    logger.info("STEP 2: SPLIT & SCALE")
    logger.info("="*80)
    
    metadata = make_split_manifest(
        cleaned_dir=config['paths']['output_dir'],
        output_dir=config['paths']['output_dir'],
        config=config
    )
    
    logger.info(f"✓ Split data saved to: {config['paths']['output_dir']}")
    return metadata


def run_build_graph(config: dict):
    """Run graph building step (only for GNN modes)."""
    from build_graph import build_graph
    
    logger.info("\n" + "="*80)
    logger.info("STEP 3: BUILD GRAPH")
    logger.info("="*80)
    
    output_path = Path(config['paths']['output_dir']) / f"{config['mode']}_graph.pt"
    
    metadata = build_graph(
        data_dir=config['paths']['output_dir'],
        output_path=str(output_path),
        config=config
    )
    
    logger.info(f"✓ Graph saved to: {output_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument(
        '--config',
        type=str,
        default='preprocess/config.yaml',
        help='Path to config YAML file (default: preprocess/config.yaml)'
    )
    parser.add_argument(
        '--steps',
        type=str,
        nargs='+',
        default=['clean', 'split', 'graph'],
        choices=['clean', 'split', 'graph', 'all'],
        help='Steps to run (default: all)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['flow_gnn', 'ip_gnn', 'cnn_lstm'],
        help='Override mode from config'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override mode if specified
    if args.mode:
        config['mode'] = args.mode
    
    # Expand 'all' to all steps
    if 'all' in args.steps:
        args.steps = ['clean', 'split', 'graph']
    
    # Determine input path based on mode
    if config['mode'] == 'ip_gnn':
        input_display = config['paths']['input_file_ip']
    else:
        input_display = config['paths']['input_dir_flow']
    
    logger.info(f"\nPreprocessing Configuration:")
    logger.info(f"  Mode: {config['mode']}")
    logger.info(f"  Input: {input_display}")
    logger.info(f"  Output: {config['paths']['output_dir']}")
    logger.info(f"  Steps: {', '.join(args.steps)}")
    
    # Run selected steps
    try:
        if 'clean' in args.steps:
            run_cleaning(config)
        
        if 'split' in args.steps:
            run_split_scale(config)
        
        if 'graph' in args.steps:
            if config['mode'] in ['flow_gnn', 'ip_gnn']:
                run_build_graph(config)
            else:
                logger.info(f"\n⚠ Skipping graph building for mode: {config['mode']}")
        
        logger.info("\n" + "="*80)
        logger.info("✓ PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"\n✗ Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()
