"""Experiment runner for AutoML-Insight."""

import sys
import yaml
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_profile import DataProfiler
from core.preprocess import DataPreprocessor
from core.models_supervised import get_supervised_models
from core.models_clustering import get_clustering_models
from core.evaluate_cls import ClassificationEvaluator
from core.evaluate_clu import ClusteringEvaluator
from utils.seed_utils import set_seed
from utils.logging_utils import setup_logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_experiment(config_path: str):
    """
    Run AutoML experiment based on configuration.
    
    Args:
        config_path: Path to configuration YAML file
    """
    # Load config
    config = load_config(config_path)
    
    # Setup
    seed = config.get('random_seed', 42)
    set_seed(seed)
    logger = setup_logger()
    
    logger.info(f"Starting experiment: {config['experiment']['name']}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config['output']['results_dir']) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    data = pd.read_csv(config['data']['path'])
    
    task_type = config['experiment']['task']
    
    if task_type == 'classification':
        target_col = config['data']['target_column']
        X = data.drop(columns=[target_col])
        y = data[target_col]
    else:
        X = data
        y = None
    
    # Profile data
    logger.info("Profiling dataset...")
    profiler = DataProfiler()
    profile = profiler.profile_dataset(X, y)
    
    # Save profile
    with open(results_dir / 'profile.json', 'w') as f:
        json.dump(profile, f, indent=2)
    
    # Preprocess
    logger.info("Preprocessing...")
    preprocessor = DataPreprocessor()
    X_processed, y_processed = preprocessor.fit_transform(X, y)
    
    # Train models
    logger.info("Training models...")
    if task_type == 'classification':
        models = get_supervised_models(seed)
        evaluator = ClassificationEvaluator(
            n_folds=config['evaluation']['n_folds'],
            n_repeats=config['evaluation']['n_repeats']
        )
        
        results = {}
        for model_name, model in models.items():
            if model_name in config.get('models', models.keys()):
                logger.info(f"Training {model_name}...")
                result = evaluator.evaluate_model(model, X_processed, y_processed, model_name)
                results[model_name] = result
        
        # Get leaderboard
        leaderboard = evaluator.get_leaderboard('accuracy')
    else:
        models = get_clustering_models(seed)
        evaluator = ClusteringEvaluator()
        
        results = {}
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            labels = model.fit_predict(X_processed)
            result = evaluator.evaluate_model(model, X_processed, model_name, labels)
            results[model_name] = result
        
        leaderboard = evaluator.get_leaderboard('silhouette')
    
    # Save results
    logger.info("Saving results...")
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for model_name, result in results.items():
        results_serializable[model_name] = {
            k: v.tolist() if hasattr(v, 'tolist') else v
            for k, v in result.items()
            if k not in ['model', 'predictions', 'true_labels', 'labels']
        }
    
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    with open(results_dir / 'leaderboard.json', 'w') as f:
        json.dump(leaderboard, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*50)
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"\nTop 3 Models:")
    for idx, item in enumerate(leaderboard[:3], 1):
        logger.info(f"  {idx}. {item['model']}: {item['score']:.4f}")
    logger.info("="*50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run AutoML experiment')
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/configs/default.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == '__main__':
    main()
