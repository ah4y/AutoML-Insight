# AutoML-Insight 🤖

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoML-Insight** is a professional AutoML platform with **AI-powered insights** that automatically profiles datasets, trains and compares 13+ ML models, provides intelligent recommendations, and delivers comprehensive analysis through an interactive Streamlit dashboard.

## ✨ Features

- 🧠 **AI-Powered Insights**: Dynamic analysis using Groq/OpenAI/Gemini LLMs at every workflow stage with intelligent rate limit handling and caching
- 📊 **Automatic Dataset Profiling**: Comprehensive statistical analysis and meta-feature extraction
- 🤖 **13+ ML Models**: 7 supervised (LogReg, SVM, RF, XGBoost, MLP) + 6 clustering algorithms with adaptive optimization
- 📈 **Rigorous Evaluation**: Nested cross-validation with confidence intervals and statistical tests
- 🔍 **Model Explainability**: SHAP-based explanations, feature importance, and interpretability with per-model caching
- 🎯 **Smart Recommendations**: Meta-learning engine with AI-enhanced deployment guidance and overfitting penalties
- 🔧 **Hyperparameter Tuning**: Optuna-based optimization with nested CV
- 📊 **Interactive Dashboard**: Streamlit UI with professional Plotly visualizations and real-time AI analysis
- 📄 **AI-Generated Reports**: Comprehensive reports with executive summaries and rule-based fallbacks
- 🌐 **Remote Execution**: Jupyter server integration for scalable cloud-based training
- 🛡️ **Intelligent Overfitting Detection**: Model-specific warnings with severity levels and actionable recommendations
- 📈 **Professional Clustering Visualizations**: 4 interactive tabs (PCA projection, distribution, silhouette, profiles)
- 🧪 **Production-Ready**: Modular OOP design, logging, testing, and reproducibility

## 🏗️ Architecture

```
AutoML-Insight/
├── app/                    # Streamlit application
│   ├── main.py            # Entry point
│   ├── ui_dashboard.py    # Dashboard UI with AI integration
│   ├── report_builder.py  # PDF report generation
│   └── config.yaml        # App configuration
├── core/                  # Core ML components
│   ├── ai_insights.py     # AI engine (Groq/OpenAI/Gemini)
│   ├── data_profile.py    # Dataset profiling
│   ├── preprocess.py      # Data preprocessing
│   ├── models_supervised.py  # 7 supervised models
│   ├── models_clustering.py  # 6 clustering algorithms
│   ├── tuning.py          # Hyperparameter tuning
│   ├── evaluate_cls.py    # Classification evaluation
│   ├── evaluate_clu.py    # Clustering evaluation
│   ├── visualize.py       # Plotly visualizations
│   ├── explain.py         # SHAP explainability
│   ├── meta_selector.py   # Meta-learning selector
│   └── ensemble.py        # Ensemble methods
├── experiments/           # Experiment runners
│   ├── run_experiment.py  # CLI experiment runner
│   └── configs/           # Experiment configs
├── data/                  # Demo datasets
├── docs/                  # Documentation
│   └── AI_SETUP.md       # AI integration guide
├── utils/                 # Utility functions
│   ├── jupyter_client.py # Remote execution
│   ├── cloud_executor.py # Cloud deployment
│   ├── seed_utils.py     # Reproducibility
│   ├── logging_utils.py  # Logging
│   └── metrics_utils.py  # Metrics computation
└── tests/                 # Unit tests
```

## 🚀 Quick Start

### Installation

```powershell
# Clone the repository
git clone https://github.com/ah4y/AutoML-Insight.git
cd AutoML-Insight

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### AI Setup (Optional but Recommended)

For AI-powered insights, create a `.env` file with your API keys:

```powershell
# Copy template
Copy-Item .env.example .env

# Edit .env and add your API keys
notepad .env
```

Supported providers:
- **Groq** (recommended, free): Get key at https://console.groq.com
- **OpenAI** (paid): Get key at https://platform.openai.com
- **Gemini** (free tier): Get key at https://makersuite.google.com

See `docs/AI_SETUP.md` for detailed instructions.

### Launch Dashboard

```powershell
streamlit run app/main.py
```

The dashboard will open at `http://localhost:8501`.

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
- Gaussian Mixture Models (GMM)
- DBSCAN (auto epsilon)
- Agglomerative Clustering
- Spectral Clustering
- Mean Shift

## 📈 Evaluation Metrics

### Classification
- Accuracy with 95% CI
- Train vs Test Performance (overfitting detection)
- Adjusted Score (with overfitting penalty)
- Macro F1-Score
- ROC-AUC (OVR)
- Log Loss
- Brier Score
- McNemar's Test
- Wilcoxon Test
- Interactive 4-panel performance dashboard

### Clustering
- Silhouette Score (overall and per-cluster)
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Cluster Stability
- Professional visualizations:
  - 2D PCA projection with variance explained
  - Cluster size distribution with balance assessment
  - Per-cluster silhouette analysis with quality metrics
  - Cluster profiles with centroids and radar charts

## 🔍 Explainability

- **SHAP Values**: Tree, Linear, and Kernel explainers
- **Feature Importance**: Native and permutation-based  
- **AI-Powered Interpretation**: Context-aware explanations of feature relationships
- **Top Features Analysis**: Automated ranking with business insights

## 📄 Reports

Generate comprehensive reports including:
- **AI-Generated Executive Summary**: Business-focused insights for stakeholders
- **Methodology Analysis**: Detailed explanation of algorithms and approach
- **Key Findings**: 4-5 critical discoveries from your data
- **Model Performance**: Statistical comparison with significance tests
- **Feature Analysis**: SHAP visualizations and AI interpretation
- **Deployment Recommendations**: Production-ready guidance and risk assessment
- **Limitations**: Honest assessment of model constraints
- PDF export with all visualizations and metadata

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

## 🧠 AI Features

### Dynamic Analysis Across All Workflow Stages

**Classification Tasks:**
- Performance assessment with context-aware insights
- Model comparison explaining why certain models excel
- Improvement suggestions based on actual results
- Deployment readiness evaluation
- Risk assessment and monitoring guidance
- **Intelligent overfitting detection** with model-specific warnings:
  - HIGH severity: Critical issues requiring immediate action
  - MEDIUM/LOW severity: Informational guidance for optimization
  - Smart thresholds based on model characteristics
  - Realistic perfect score detection (avoids false alarms)

**Clustering Tasks:**
- Cluster quality evaluation with honest assessments
- Cluster interpretation and business meaning
- Balance analysis and distribution insights
- Validation steps before deployment
- Use case recommendations
- **Professional visualizations** with 4 interactive tabs:
  - PCA 2D projection showing cluster separation
  - Size distribution with imbalance detection
  - Silhouette analysis revealing cluster quality
  - Cluster profiles with centroids and radar charts

**Robustness & Reliability:**
- **Intelligent rate limit handling**: Automatic fallback to rule-based insights
- **Response caching**: MD5-based caching reduces API calls
- **User-friendly error messages**: Clear guidance when AI unavailable
- **Graceful degradation**: App remains fully functional without AI

**For All Tasks:**
- AI-generated comprehensive reports with fallback mode
- Executive summaries for non-technical stakeholders
- Actionable next steps
- Honest limitations and caveats

## 📚 Academic References

- **Ensemble Learning**: Breiman (2001), Random Forests
- **XGBoost**: Chen & Guestrin (2016), XGBoost: A Scalable Tree Boosting System
- **SHAP**: Lundberg & Lee (2017), A Unified Approach to Interpreting Model Predictions
- **Meta-Learning**: Brazdil et al. (2009), Metalearning: Applications to Data Mining
- **Clustering Validation**: Rousseeuw (1987), Silhouettes: A Graphical Aid
- **LLM for Analysis**: Groq Llama 3.3 70B, OpenAI GPT-4, Google Gemini 1.5

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

-sochials: LinkdIN : https://www.linkedin.com/in/ahmed-yassin-052209374/

---

**Made with ❤️ for the ML community**
