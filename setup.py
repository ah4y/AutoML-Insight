"""Setup script for AutoML-Insight."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="automl-insight",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Professional AutoML platform with meta-learning and explainability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AutoML-Insight",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "torch>=2.0.0",
        "shap>=0.42.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "umap-learn>=0.5.3",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "streamlit>=1.25.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "reportlab>=4.0.0",
        "weasyprint>=59.0",
        "optuna>=3.3.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "automl-insight=app.main:main",
            "automl-experiment=experiments.run_experiment:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml"],
    },
)
