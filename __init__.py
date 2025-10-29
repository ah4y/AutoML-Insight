"""
AutoML-Insight: Professional AutoML Platform
============================================

A comprehensive AutoML system for automated machine learning with:
- Dataset profiling and meta-feature extraction
- Multi-model training and evaluation
- Model explainability (SHAP)
- Meta-learning based recommendations
- Interactive Streamlit dashboard
- PDF report generation

Quick Start:
-----------
1. Install: pip install -r requirements.txt
2. Generate demo data: python generate_demo_data.py
3. Launch dashboard: streamlit run app/main.py

For more information, see README.md
"""

__version__ = "1.0.0"
__author__ = "AutoML-Insight Contributors"

from core import (
    data_profile,
    preprocess,
    models_supervised,
    models_clustering,
    evaluate_cls,
    evaluate_clu,
    visualize,
    explain,
    meta_selector,
    ensemble
)

from utils import seed_utils, logging_utils, metrics_utils

__all__ = [
    'data_profile',
    'preprocess',
    'models_supervised',
    'models_clustering',
    'evaluate_cls',
    'evaluate_clu',
    'visualize',
    'explain',
    'meta_selector',
    'ensemble',
    'seed_utils',
    'logging_utils',
    'metrics_utils'
]
