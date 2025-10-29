# AutoML-Insight 🤖

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoML-Insight** is a professional, research-ready AutoML platform that automatically profiles datasets, trains and compares advanced ML models (ANNs, Ensembles, Clustering), visualizes metrics, and recommends the best approach using meta-learning and pattern recognition.

## ✨ Features

- 📊 **Automatic Dataset Profiling**: Comprehensive statistical analysis and meta-feature extraction
- 🤖 **Multi-Model Training**: 7+ supervised and 5+ unsupervised learning algorithms
- 📈 **Rigorous Evaluation**: Nested cross-validation with confidence intervals and statistical tests
- 🔍 **Model Explainability**: SHAP-based explanations, feature importance, and PDP
- 🎯 **Smart Recommendations**: Meta-learning engine with heuristic fallback
- 🔧 **Hyperparameter Tuning**: Optuna-based optimization with nested CV
- 📊 **Interactive Dashboard**: Streamlit UI with Plotly visualizations
- 📄 **PDF Reports**: Exportable analytical reports with all results
- 🧪 **Production-Ready**: Modular OOP design, logging, testing, and reproducibility

## 🏗️ Architecture

```
AutoML-Insight/
├── app/                    # Streamlit application
│   ├── main.py            # Entry point
│   ├── ui_dashboard.py    # Dashboard UI
│   ├── report_builder.py  # PDF report generation
│   └── config.yaml        # App configuration
├── core/                  # Core ML components
│   ├── data_profile.py    # Dataset profiling
│   ├── preprocess.py      # Data preprocessing
│   ├── models_supervised.py  # Supervised models
│   ├── models_clustering.py  # Clustering models
│   ├── tuning.py          # Hyperparameter tuning
│   ├── evaluate_cls.py    # Classification evaluation
│   ├── evaluate_clu.py    # Clustering evaluation
│   ├── visualize.py       # Visualization utilities
│   ├── explain.py         # Model explainability
│   ├── meta_selector.py   # Meta-learning selector
│   └── ensemble.py        # Ensemble methods
├── experiments/           # Experiment runners
│   ├── run_experiment.py  # CLI experiment runner
│   └── configs/           # Experiment configs
├── data/                  # Data storage
├── results/               # Output directory
│   ├── runs/             # Experiment results
│   ├── reports/          # PDF reports
│   └── logs/             # Log files
├── utils/                 # Utility functions
│   ├── seed_utils.py     # Reproducibility
│   ├── logging_utils.py  # Logging
│   └── metrics_utils.py  # Metrics computation
└── tests/                 # Unit tests
```

## 🚀 Quick Start

### Installation

```powershell
# Clone the repository
git clone https://github.com/yourusername/AutoML-Insight.git
cd AutoML-Insight

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Launch Dashboard

```powershell
streamlit run app/main.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Run Experiments (CLI)

```powershell
python experiments/run_experiment.py --config experiments/configs/default.yaml
```

## 📊 Usage

### Demo Mode

1. Launch the dashboard: `streamlit run app/main.py`
2. Check "🎮 Demo Mode" in the sidebar
3. Select Iris or Wine dataset
4. Choose task type (Classification/Clustering)
5. Click "🚀 Run AutoML"

### Custom Dataset

1. Upload your CSV file via the sidebar
2. Select target variable (for classification)
3. Configure settings (random seed, etc.)
4. Click "🚀 Run AutoML"
5. Explore results in multiple tabs

## 🤖 Supported Models

### Supervised Learning
- Logistic Regression
- Linear SVM
- RBF-SVM
- k-Nearest Neighbors
- Random Forest
- XGBoost
- Multi-Layer Perceptron (PyTorch)

### Unsupervised Learning
- KMeans (auto K selection)
- Gaussian Mixture Models
- DBSCAN (auto epsilon)
- Agglomerative Clustering
- Spectral Clustering

## 📈 Evaluation Metrics

### Classification
- Accuracy with 95% CI
- Macro F1-Score
- ROC-AUC (OVR)
- Log Loss
- Brier Score
- McNemar's Test
- Wilcoxon Test

### Clustering
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Cluster Stability

## 🔍 Explainability

- **SHAP Values**: Tree, Linear, and Kernel explainers
- **Feature Importance**: Native and permutation-based
- **Partial Dependence Plots**: Coming soon
- **Top Features Analysis**: Automated ranking

## 📄 Reports

Generate comprehensive PDF reports including:
- Dataset profile and statistics
- Model performance comparison
- Statistical significance tests
- Feature importance visualizations
- Recommendation rationale
- Metadata and reproducibility info

## 🧪 Testing

```powershell
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=core --cov=app --cov=utils
```

## 🛠️ Configuration

Edit `app/config.yaml` to customize:
- Cross-validation folds
- Hyperparameter tuning trials
- Clustering K range
- Visualization settings
- Report format options

## 📚 Academic References

- **Ensemble Learning**: Breiman (2001), Random Forests
- **XGBoost**: Chen & Guestrin (2016), XGBoost: A Scalable Tree Boosting System
- **SHAP**: Lundberg & Lee (2017), A Unified Approach to Interpreting Model Predictions
- **Meta-Learning**: Brazdil et al. (2009), Metalearning: Applications to Data Mining
- **Clustering Validation**: Rousseeuw (1987), Silhouettes: A Graphical Aid

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Ahmed Yassin(ah4y)** - Initial work

## 🙏 Acknowledgments

- scikit-learn team for excellent ML library
- Plotly for interactive visualizations
- Streamlit for rapid dashboard development
- SHAP contributors for explainability tools

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for the ML community**
