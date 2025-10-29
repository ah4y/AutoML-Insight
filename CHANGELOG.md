# Changelog

All notable changes to AutoML-Insight will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive `config.yaml` with all project settings (general, preprocessing, models, CV, tuning, visualization, remote execution)
- `.copilot/main_prompt.json` for GitHub Copilot integration with comprehensive project documentation
- `.copilot/automl_insight_prompt.json` for AI-driven feature enhancements
- Adaptive cross-validation strategy (2-5 folds based on dataset size)
- Automatic task type detection (regression vs classification)
- Demo mode with locked target column to prevent user errors
- Class distribution validation before training
- Enhanced LabelEncoder logging with class mapping visualization

### Changed
- Updated XGBoost configuration to remove deprecated `use_label_encoder` parameter
- Improved preprocessing pipeline with two-stage feature selection
- Enhanced error messages with actionable solutions

### Fixed
- Cross-validation errors with small datasets (n_splits > class membership)
- Continuous target values being misidentified as classification tasks
- Non-contiguous class labels causing XGBoost and MLP failures
- Demo mode allowing selection of wrong target columns

### Deprecated
- None

### Removed
- Deprecated `use_label_encoder=False` parameter from XGBClassifier

### Security
- None

## [2.0.0] - 2025-01-XX

### Added
- Remote execution support (Jupyter Server and Google Colab)
- File-based Jupyter client for reliable remote execution
- Multi-mode execution: Local, Remote Jupyter, Google Colab
- Comprehensive logging with rotating file handlers
- Meta-learning based model recommendations
- SHAP explainability for all model types
- Interactive Plotly visualizations (ROC curves, confusion matrices, feature importance)
- 5 clustering algorithms with automatic parameter selection
- Ensemble methods (Voting, Stacking, Bagging)
- PyTorch-based MLP with GPU support

### Changed
- Redesigned UI dashboard with tabbed interface
- Improved preprocessing pipeline with intelligent feature selection
- Enhanced evaluation metrics with confidence intervals
- Optimized memory usage for large datasets (150K+ samples, 361K+ features)

### Fixed
- Memory errors with high-dimensional datasets
- Remote Jupyter connection timeout issues
- Clustering visualization errors

## [1.0.0] - 2024-12-XX

### Added
- Initial release
- 7 supervised learning models (Logistic Regression, Linear SVM, RBF-SVM, KNN, Random Forest, XGBoost, MLP)
- Basic classification and clustering support
- Streamlit dashboard with file upload
- Dataset profiling and statistics
- Basic model evaluation metrics
- Demo mode with Iris and Wine datasets
- CSV export of results

### Changed
- N/A (initial release)

### Fixed
- N/A (initial release)

---

## Release Notes

### Version 2.0.0 Highlights
- **Remote Execution**: Train models on powerful remote servers or Google Colab
- **Meta-Learning**: Intelligent model recommendations based on dataset characteristics
- **Explainability**: SHAP values for understanding model decisions
- **Production-Ready**: Comprehensive logging, error handling, and validation

### Version 1.0.0 Highlights
- **AutoML Pipeline**: Automated machine learning from data profiling to model evaluation
- **Multiple Models**: Compare 7+ supervised and 5+ unsupervised algorithms
- **Interactive UI**: Streamlit dashboard with real-time visualizations
- **Demo Datasets**: Pre-loaded Iris and Wine datasets for quick testing

---

## Migration Guides

### Upgrading from 1.x to 2.0

1. **New Dependencies**: Install updated requirements
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Configuration**: Create `config.yaml` if customizing settings (optional)

3. **Remote Execution**: New feature - see documentation for Jupyter/Colab setup

4. **API Changes**: None - backward compatible

### Breaking Changes
- None in 2.0.0 (fully backward compatible)

---

## Development Status

- ✅ **Active Development**: Regular updates and bug fixes
- ✅ **Production Ready**: Stable for research and production use
- ✅ **Community Driven**: Contributions welcome

---

## Contributors

Thanks to all contributors who have helped improve AutoML-Insight!

- **Lead Developer**: [Your Name]
- **Contributors**: [List contributors here]

---

## Roadmap

### Short-term (v2.1)
- [ ] Regression support (planned)
- [ ] PDF report generation (in progress)
- [ ] Experiment tracking with MLflow
- [ ] Model registry for versioning
- [ ] Unit test coverage >80%

### Medium-term (v2.5)
- [ ] Automated feature engineering
- [ ] Deep learning for images and text
- [ ] Time series forecasting
- [ ] Multi-GPU support
- [ ] Web API with FastAPI

### Long-term (v3.0)
- [ ] Neural architecture search (NAS)
- [ ] AutoML with reinforcement learning
- [ ] Distributed training with Dask/Ray
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Enterprise features (user management, audit logs)

---

For detailed information about each release, see [GitHub Releases](https://github.com/yourusername/AutoML-Insight/releases).
