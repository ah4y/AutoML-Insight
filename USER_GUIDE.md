# AutoML-Insight User Guide

## Table of Contents
1. [Installation](#installation)
2. [Getting Started](#getting-started)
3. [Dashboard Usage](#dashboard-usage)
4. [CLI Experiments](#cli-experiments)
5. [API Reference](#api-reference)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- 4GB+ RAM recommended
- (Optional) CUDA-enabled GPU for PyTorch acceleration

### Step-by-Step Installation

```powershell
# 1. Clone or download the repository
cd AutoML-Insight

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Generate demo datasets
python generate_demo_data.py
```

### Quick Installation (Windows)
```powershell
.\quickstart.ps1
```

## Getting Started

### Launch Dashboard

```powershell
streamlit run app/main.py
```

The dashboard will open at http://localhost:8501

### Demo Mode

1. Check "🎮 Demo Mode" in sidebar
2. Select dataset: Iris or Wine
3. Choose task type
4. Click "🚀 Run AutoML"
5. Explore results in tabs

### Custom Dataset

Requirements:
- CSV format
- Clean column names (no special characters)
- Target column for classification
- Numeric or mixed features supported

Steps:
1. Upload CSV via sidebar
2. Select target column
3. Configure settings
4. Run AutoML

## Dashboard Usage

### Tab 1: Data Overview
- Dataset statistics
- Missing value analysis
- Correlation heatmap
- Feature distributions

### Tab 2: Models
- Model leaderboard with confidence intervals
- ROC curves (classification)
- Confusion matrix
- UMAP clusters (clustering)
- Elbow & silhouette curves

### Tab 3: Explainability
- SHAP feature importance
- Model-specific explanations
- Top feature rankings
- Permutation importance

### Tab 4: Recommendation
- Best model suggestion
- Performance metrics
- Statistical confidence
- Rationale and alternatives

### Tab 5: Report
- Generate PDF report
- Comprehensive results
- All visualizations
- Reproducibility metadata

## CLI Experiments

### Basic Experiment

```powershell
python experiments/run_experiment.py --config experiments/configs/default.yaml
```

### Custom Configuration

Create a YAML config:

```yaml
random_seed: 42

experiment:
  name: my_experiment
  task: classification

data:
  path: data/my_data.csv
  target_column: target

models:
  - LogisticRegression
  - RandomForest
  - XGBoost

evaluation:
  n_folds: 5
  n_repeats: 3

output:
  save_models: true
  results_dir: results/runs
```

Run:
```powershell
python experiments/run_experiment.py --config my_config.yaml
```

## API Reference

### Data Profiling

```python
from core.data_profile import DataProfiler

profiler = DataProfiler()
profile = profiler.profile_dataset(X, y)
meta_features = profiler.get_profile_vector()
```

### Preprocessing

```python
from core.preprocess import DataPreprocessor

preprocessor = DataPreprocessor()
X_transformed, y_transformed = preprocessor.fit_transform(X, y)
```

### Model Training

```python
from core.models_supervised import get_supervised_models

models = get_supervised_models(random_state=42)
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
```

### Evaluation

```python
from core.evaluate_cls import ClassificationEvaluator

evaluator = ClassificationEvaluator(n_folds=5, n_repeats=3)
results = evaluator.evaluate_model(model, X, y, 'ModelName')
leaderboard = evaluator.get_leaderboard('accuracy')
```

### Explainability

```python
from core.explain import ModelExplainer

explainer = ModelExplainer()
explanations = explainer.explain_model(model, X, feature_names)
top_features = explainer.get_top_features(explanations, top_n=10)
```

### Meta-Learning

```python
from core.meta_selector import MetaModelSelector

selector = MetaModelSelector()
recommendation = selector.get_recommendation_with_rationale(
    meta_features, evaluation_results
)
```

## Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'optuna'

**Solution:**
```powershell
pip install optuna
```

#### 2. Streamlit not found

**Solution:**
```powershell
pip install streamlit
```

#### 3. CUDA errors with PyTorch MLP

**Solution:**
- PyTorch will automatically use CPU if CUDA unavailable
- To force CPU: Set environment variable before running
```powershell
$env:CUDA_VISIBLE_DEVICES="-1"
streamlit run app/main.py
```

#### 4. Memory errors on large datasets

**Solution:**
- Reduce CV folds: `n_folds=3`
- Reduce repeats: `n_repeats=1`
- Sample data before profiling
- Disable some models in config

#### 5. PDF generation fails

**Solution:**
```powershell
pip install --upgrade reportlab weasyprint
```

For WeasyPrint on Windows, may need GTK3:
https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer

### Performance Tips

1. **Reduce computation time:**
   - Use fewer CV folds (3 instead of 5)
   - Reduce tuning trials (10 instead of 20)
   - Limit model selection

2. **Improve accuracy:**
   - Increase CV folds and repeats
   - Enable hyperparameter tuning
   - Use ensemble methods

3. **Better explainability:**
   - Increase SHAP sample size (up to 500)
   - Use Tree-based models for faster SHAP

## Configuration Reference

### app/config.yaml

```yaml
random_seed: 42              # Reproducibility seed

logging:
  level: INFO               # DEBUG, INFO, WARNING, ERROR
  log_dir: results/logs     # Log directory

preprocessing:
  numeric_imputation: median   # mean, median, most_frequent
  categorical_imputation: constant
  scaling: standard         # standard, minmax, robust
  encoding: onehot          # onehot, ordinal

training:
  n_folds: 5               # CV folds
  n_repeats: 3             # CV repeats
  test_size: 0.2           # Holdout test size

tuning:
  enabled: true            # Enable hyperparameter tuning
  n_trials: 20             # Optuna trials per model
  cv_folds: 3              # CV folds for tuning
  timeout: 300             # Timeout in seconds

evaluation:
  confidence_level: 0.95   # CI level
  statistical_tests: true  # Enable McNemar/Wilcoxon

explainability:
  shap_sample_size: 100    # Samples for SHAP
  top_features: 15         # Top N features to show

ensemble:
  enabled: true            # Create ensemble
  top_k_models: 3          # Models in ensemble
  calibration: true        # Calibrate probabilities
```

## Best Practices

### For Classification Tasks

1. **Small datasets (<1000 samples):**
   - Use LogisticRegression, KNN
   - Higher CV folds (5-10)
   - Enable ensemble

2. **Large datasets (>10000 samples):**
   - Use XGBoost, MLP
   - Fewer CV folds (3-5)
   - Enable hyperparameter tuning

3. **Imbalanced classes:**
   - Use stratified CV (automatic)
   - Monitor F1-score and ROC-AUC
   - Consider resampling (external)

### For Clustering Tasks

1. **Unknown K:**
   - Use KMeans with auto K
   - Check elbow and silhouette plots
   - Compare multiple methods

2. **Non-spherical clusters:**
   - Use DBSCAN or GMM
   - Tune epsilon/components
   - Visualize with UMAP

3. **Hierarchical structure:**
   - Use Agglomerative clustering
   - Try different linkages
   - Generate dendrograms (external)

## Support

- **Issues:** Open an issue on GitHub
- **Documentation:** See README.md
- **Examples:** Check experiments/configs/

## Version History

### v1.0.0 (2025-01-27)
- Initial release
- 7 supervised models
- 5 clustering models
- SHAP explainability
- Streamlit dashboard
- PDF reports
- Meta-learning recommendations

---

**Happy AutoML-ing! 🤖**
