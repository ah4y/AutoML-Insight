"""Cloud execution utilities for training on remote resources."""

import os
import json
import psutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from utils.logging_utils import setup_logger

logger = setup_logger()


class CloudExecutor:
    """Execute AutoML training on cloud platforms."""
    
    @staticmethod
    def detect_environment() -> str:
        """
        Detect current execution environment.
        
        Returns:
            'colab', 'kaggle', 'jupyter', or 'local'
        """
        try:
            # Check for Google Colab
            import google.colab
            return 'colab'
        except ImportError:
            pass
        
        # Check for Kaggle
        if os.path.exists('/kaggle/input'):
            return 'kaggle'
        
        # Check for Jupyter
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                return 'jupyter'
        except ImportError:
            pass
        
        return 'local'
    
    @staticmethod
    def get_available_resources() -> Dict[str, Any]:
        """Get information about available computing resources."""
        resources = {
            'environment': CloudExecutor.detect_environment(),
            'cpu_count': psutil.cpu_count(),
            'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'ram_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'gpu_available': False,
            'gpu_name': None,
            'gpu_memory_gb': 0
        }
        
        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                resources['gpu_available'] = True
                resources['gpu_name'] = torch.cuda.get_device_name(0)
                resources['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        except ImportError:
            pass
        
        return resources
    
    @staticmethod
    def estimate_memory_required(n_samples: int, n_features: int) -> float:
        """
        Estimate memory required for training.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            Estimated memory in GB
        """
        # float64 = 8 bytes
        # Add overhead for preprocessing, model training (3x multiplier)
        memory_gb = (n_samples * n_features * 8 * 3) / (1024**3)
        return round(memory_gb, 2)
    
    @staticmethod
    def recommend_execution_mode(
        n_samples: int,
        n_features: int,
        available_ram_gb: float
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Recommend execution mode based on dataset size and resources.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            available_ram_gb: Available RAM in GB
            
        Returns:
            Tuple of (mode, reason, config)
            - mode: 'local' or 'cloud'
            - reason: Explanation
            - config: Recommended configuration
        """
        required_memory = CloudExecutor.estimate_memory_required(n_samples, n_features)
        
        config = {
            'required_memory_gb': required_memory,
            'available_memory_gb': available_ram_gb,
            'recommended_max_features': 1000,
            'can_run_locally': False
        }
        
        # Check if can run locally (need 1.5x more RAM than required for safety)
        if available_ram_gb >= required_memory * 1.5:
            # Can run locally
            config['can_run_locally'] = True
            
            # Adjust max features based on available RAM
            if available_ram_gb >= 16:
                config['recommended_max_features'] = 5000
            elif available_ram_gb >= 8:
                config['recommended_max_features'] = 2000
            else:
                config['recommended_max_features'] = 1000
            
            reason = f"✅ Your machine has {available_ram_gb:.1f} GB RAM available (need ~{required_memory:.1f} GB). Can run locally with {config['recommended_max_features']:,} features."
            return 'local', reason, config
        else:
            # Recommend cloud
            config['recommended_max_features'] = 10000  # Cloud can handle more
            
            reason = f"⚠️ Dataset needs ~{required_memory:.1f} GB RAM but only {available_ram_gb:.1f} GB available. Recommend cloud execution (Colab: 12GB RAM free, Kaggle: 30GB RAM free)."
            return 'cloud', reason, config
    
    @staticmethod
    def generate_colab_notebook(
        dataset_name: str,
        target_column: str,
        task_type: str,
        max_features: int = 5000
    ) -> str:
        """
        Generate a Google Colab notebook for remote execution.
        
        Args:
            dataset_name: Name of the dataset file
            target_column: Name of target variable
            task_type: 'Classification' or 'Clustering'
            max_features: Maximum features to use
            
        Returns:
            Notebook content as JSON string
        """
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# 🤖 AutoML-Insight - Cloud Execution\n",
                        f"\n**Task:** {task_type}\n",
                        f"**Target Column:** `{target_column}`\n",
                        f"**Max Features:** {max_features:,}\n",
                        "\n---\n",
                        "\n### 📋 Instructions:\n",
                        "1. **Enable GPU**: Runtime → Change runtime type → GPU (T4)\n",
                        "2. **Upload Dataset**: Run cell below and upload your CSV file\n",
                        "3. **Run All**: Runtime → Run all\n",
                        "4. **Download Results**: Last cell will download results.json\n",
                        "\n---"
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# 📦 Install Dependencies\n",
                        "!pip install -q scikit-learn==1.5.1 xgboost==2.1.1 optuna==3.6.1 shap==0.45.1 pandas numpy matplotlib seaborn\n",
                        "print('✅ Dependencies installed')"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# 📤 Upload Dataset\n",
                        "from google.colab import files\n",
                        "import pandas as pd\n",
                        "\n",
                        "print('📁 Please upload your CSV file:')\n",
                        "uploaded = files.upload()\n",
                        "dataset_file = list(uploaded.keys())[0]\n",
                        "\n",
                        "df = pd.read_csv(dataset_file)\n",
                        "print(f'\\n✅ Dataset loaded: {df.shape[0]:,} samples × {df.shape[1]} features')\n",
                        "df.head()"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# 🔍 Check GPU Resources\n",
                        "import torch\n",
                        "\n",
                        "print(f'🖥️  GPU Available: {torch.cuda.is_available()}')\n",
                        "if torch.cuda.is_available():\n",
                        "    print(f'📊 GPU Name: {torch.cuda.get_device_name(0)}')\n",
                        "    print(f'💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')\n",
                        "else:\n",
                        "    print('⚠️  No GPU detected. Go to Runtime → Change runtime type → GPU')"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# 🔧 Preprocessing\n",
                        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
                        "from sklearn.impute import SimpleImputer\n",
                        "from sklearn.feature_selection import SelectKBest, f_classif\n",
                        "import numpy as np\n",
                        "\n",
                        f"# Separate features and target\n",
                        f"X = df.drop(columns=['{target_column}'])\n",
                        f"y = df['{target_column}']\n",
                        "\n",
                        "# Remove constant features\n",
                        "constant_cols = X.columns[X.nunique() <= 1]\n",
                        "if len(constant_cols) > 0:\n",
                        "    print(f'Removing {len(constant_cols)} constant features')\n",
                        "    X = X.drop(columns=constant_cols)\n",
                        "\n",
                        "# Handle missing values\n",
                        "numeric_cols = X.select_dtypes(include=[np.number]).columns\n",
                        "if len(numeric_cols) > 0:\n",
                        "    imputer = SimpleImputer(strategy='median')\n",
                        "    X[numeric_cols] = imputer.fit_transform(X[numeric_cols])\n",
                        "\n",
                        "# Feature selection if needed\n",
                        f"max_features = {max_features}\n",
                        "if X.shape[1] > max_features:\n",
                        "    print(f'Selecting top {max_features:,} features from {X.shape[1]:,}')\n",
                        "    \n",
                        "    # Encode target for feature selection\n",
                        "    le = LabelEncoder()\n",
                        "    y_encoded = le.fit_transform(y)\n",
                        "    \n",
                        "    # Select best features\n",
                        "    selector = SelectKBest(f_classif, k=max_features)\n",
                        "    X_selected = selector.fit_transform(X, y_encoded)\n",
                        "    selected_features = X.columns[selector.get_support()].tolist()\n",
                        "    X = pd.DataFrame(X_selected, columns=selected_features)\n",
                        "\n",
                        "# Scale features\n",
                        "scaler = StandardScaler()\n",
                        "X_scaled = scaler.fit_transform(X)\n",
                        "\n",
                        "# Encode target\n",
                        "label_encoder = LabelEncoder()\n",
                        "y_encoded = label_encoder.fit_transform(y)\n",
                        "\n",
                        "print(f'\\n✅ Preprocessing complete: {X_scaled.shape[0]:,} × {X_scaled.shape[1]:,}')\n",
                        "print(f'📊 Classes: {label_encoder.classes_.tolist()}')"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# 🤖 Train Models\n",
                        "from sklearn.model_selection import cross_validate\n",
                        "from sklearn.linear_model import LogisticRegression\n",
                        "from sklearn.svm import SVC\n",
                        "from sklearn.neighbors import KNeighborsClassifier\n",
                        "from sklearn.ensemble import RandomForestClassifier\n",
                        "from xgboost import XGBClassifier\n",
                        "import warnings\n",
                        "warnings.filterwarnings('ignore')\n",
                        "\n",
                        "models = {\n",
                        "    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),\n",
                        "    'SVM-RBF': SVC(kernel='rbf', probability=True, random_state=42),\n",
                        "    'KNN': KNeighborsClassifier(n_neighbors=5),\n",
                        "    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),\n",
                        "    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss')\n",
                        "}\n",
                        "\n",
                        "results = {}\n",
                        "\n",
                        "for name, model in models.items():\n",
                        "    print(f'\\n🔄 Training {name}...')\n",
                        "    try:\n",
                        "        cv_results = cross_validate(\n",
                        "            model, X_scaled, y_encoded,\n",
                        "            cv=5,\n",
                        "            scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'],\n",
                        "            return_train_score=False,\n",
                        "            n_jobs=1\n",
                        "        )\n",
                        "        \n",
                        "        results[name] = {\n",
                        "            'accuracy': float(cv_results['test_accuracy'].mean()),\n",
                        "            'f1': float(cv_results['test_f1_weighted'].mean()),\n",
                        "            'precision': float(cv_results['test_precision_weighted'].mean()),\n",
                        "            'recall': float(cv_results['test_recall_weighted'].mean()),\n",
                        "        }\n",
                        "        \n",
                        "        print(f'✅ {name}: Accuracy = {results[name][\"accuracy\"]:.4f}')\n",
                        "        \n",
                        "    except Exception as e:\n",
                        "        print(f'❌ {name} failed: {str(e)}')\n",
                        "        results[name] = {'error': str(e)}\n",
                        "\n",
                        "print('\\n🎉 Training complete!')"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# 📊 Results Summary\n",
                        "import pandas as pd\n",
                        "\n",
                        "# Create results DataFrame\n",
                        "results_df = pd.DataFrame([\n",
                        "    {'Model': name, **metrics} \n",
                        "    for name, metrics in results.items() \n",
                        "    if 'error' not in metrics\n",
                        "]).sort_values('accuracy', ascending=False)\n",
                        "\n",
                        "print('\\n📈 Model Performance:')\n",
                        "print(results_df.to_string(index=False))\n",
                        "\n",
                        "# Best model\n",
                        "best_model = results_df.iloc[0]['Model']\n",
                        "best_acc = results_df.iloc[0]['accuracy']\n",
                        "print(f'\\n🏆 Best Model: {best_model} (Accuracy: {best_acc:.4f})')"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# 💾 Save and Download Results\n",
                        "from google.colab import files\n",
                        "import json\n",
                        "\n",
                        "# Prepare results\n",
                        "output = {\n",
                        "    'dataset_info': {\n",
                        "        'n_samples': int(df.shape[0]),\n",
                        "        'n_features': int(X_scaled.shape[1]),\n",
                        f"        'target_column': '{target_column}',\n",
                        "        'classes': label_encoder.classes_.tolist()\n",
                        "    },\n",
                        "    'results': results,\n",
                        "    'best_model': best_model,\n",
                        "    'best_accuracy': float(best_acc)\n",
                        "}\n",
                        "\n",
                        "# Save to file\n",
                        "with open('automl_results.json', 'w') as f:\n",
                        "    json.dump(output, f, indent=2)\n",
                        "\n",
                        "print('✅ Results saved!')\n",
                        "print('\\n📥 Downloading results...')\n",
                        "files.download('automl_results.json')\n",
                        "\n",
                        "print('\\n🎉 All done! Upload automl_results.json to AutoML-Insight app.')"
                    ],
                    "execution_count": None,
                    "outputs": []
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "accelerator": "GPU",
                "colab": {
                    "provenance": []
                }
            },
            "nbformat": 4,
            "nbformat_minor": 0
        }
        
        return json.dumps(notebook, indent=2)
    
    @staticmethod
    def save_notebook(notebook_content: str, output_path: str) -> str:
        """Save notebook to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write(notebook_content)
        
        logger.info(f"Saved notebook to: {output_path}")
        return str(path)
