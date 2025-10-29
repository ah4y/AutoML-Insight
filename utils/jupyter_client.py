"""Remote Jupyter Server Client for Cloud Execution."""

import requests
import json
import time
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import threading
from queue import Queue
import base64
import tempfile


class JupyterServerClient:
    """Client to connect and execute code on remote Jupyter servers."""
    
    def __init__(self, base_url: str, token: Optional[str] = None):
        """
        Initialize Jupyter client.
        
        Args:
            base_url: Base URL of Jupyter server (e.g., http://localhost:8888)
            token: Authentication token (if required)
        """
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Set up authentication
        if token:
            self.session.params = {'token': token}
    
    def test_connection(self) -> bool:
        """Test if server is reachable."""
        try:
            print(f"Testing connection to: {self.base_url}")
            print(f"Token provided: {'Yes' if self.token else 'No'}")
            
            response = self.session.get(f"{self.base_url}/api", timeout=5)
            
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.text[:200]}")
            
            return response.status_code == 200
        except Exception as e:
            print(f"Connection test failed with error: {type(e).__name__}: {e}")
            return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        try:
            response = self.session.get(f"{self.base_url}/api")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def execute_code_via_file(self, code: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Execute Python code by uploading and running a script.
        Most reliable method for Jupyter Server API.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary with execution results
        """
        try:
            import uuid
            script_id = str(uuid.uuid4())[:8]
            script_name = f"automl_exec_{script_id}.py"
            output_name = f"automl_output_{script_id}.json"
            
            # Wrap code to capture output
            wrapped_code = f'''
import json
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

stdout_buf = io.StringIO()
stderr_buf = io.StringIO()

result = {{
    "success": False,
    "outputs": [],
    "errors": []
}}

try:
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        exec("""
{code}
""")
    result["success"] = True
    result["outputs"] = [stdout_buf.getvalue()]
    if stderr_buf.getvalue():
        result["errors"] = [stderr_buf.getvalue()]
except Exception as e:
    result["success"] = False
    result["errors"] = [str(e)]
    import traceback
    result["errors"].append(traceback.format_exc())

# Save result
with open("{output_name}", "w") as f:
    json.dump(result, f)

print("Execution completed. Results saved to {output_name}")
'''
            
            # Save script locally
            temp_script = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
            temp_script.write(wrapped_code)
            temp_script.close()
            
            # Upload script
            print(f"Uploading script {script_name}...")
            if not self.upload_file(temp_script.name, script_name):
                os.unlink(temp_script.name)
                return {"success": False, "error": "Failed to upload script"}
            
            os.unlink(temp_script.name)
            
            # Create a notebook to run the script
            notebook_name = f"automl_runner_{script_id}.ipynb"
            notebook_content = {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [f"%run {script_name}"]
                    }
                ],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }
            
            # Upload notebook
            response = self.session.put(
                f"{self.base_url}/api/contents/{notebook_name}",
                json={
                    "type": "notebook",
                    "format": "json",
                    "content": notebook_content
                },
                timeout=30
            )
            
            if response.status_code not in [200, 201]:
                return {"success": False, "error": f"Failed to create notebook: {response.status_code}"}
            
            # Wait for execution (give it time to complete)
            print(f"Waiting for execution to complete...")
            time.sleep(5)  # Initial wait
            
            # Poll for output file
            max_attempts = timeout // 5
            for attempt in range(max_attempts):
                try:
                    # Try to download result
                    result_response = self.session.get(
                        f"{self.base_url}/api/contents/{output_name}",
                        timeout=10
                    )
                    
                    if result_response.status_code == 200:
                        data = result_response.json()
                        content = data.get('content', '')
                        
                        # Decode and parse result
                        import json
                        result = json.loads(content)
                        
                        # Cleanup
                        self._delete_file(script_name)
                        self._delete_file(notebook_name)
                        self._delete_file(output_name)
                        
                        return result
                except:
                    pass
                
                time.sleep(5)
            
            return {"success": False, "error": "Execution timeout - no results received"}
            
        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def _delete_file(self, filename: str):
        """Delete a file from Jupyter server."""
        try:
            self.session.delete(
                f"{self.base_url}/api/contents/{filename}",
                timeout=5
            )
        except:
            pass
    
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file to Jupyter server."""
        try:
            with open(local_path, 'rb') as f:
                content = f.read()
            
            # Convert to base64
            content_b64 = base64.b64encode(content).decode('utf-8')
            
            # Upload via Contents API
            response = self.session.put(
                f"{self.base_url}/api/contents/{remote_path}",
                json={
                    "type": "file",
                    "format": "base64",
                    "content": content_b64
                },
                timeout=60
            )
            return response.status_code in [200, 201]
        except Exception:
            return False
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file from Jupyter server."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/contents/{remote_path}",
                timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                content = data.get('content', '')
                
                # Decode base64
                file_bytes = base64.b64decode(content)
                
                with open(local_path, 'wb') as f:
                    f.write(file_bytes)
                return True
            return False
        except Exception:
            return False
    
    def install_packages(self, packages: List[str]) -> bool:
        """Install Python packages on remote server."""
        code = f"""
import subprocess
import sys

packages = {packages}
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q'] + packages,
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print(f"Successfully installed: {{', '.join(packages)}}")
else:
    print(f"Installation errors: {{result.stderr}}")
"""
        result = self.execute_code_via_file(code, timeout=180)
        return result.get('success', False)
    
    def shutdown_kernel(self):
        """Shutdown - no-op for file-based execution."""
        pass
    
    def __del__(self):
        """Cleanup on deletion."""
        pass


class ColabServerSetup:
    """Helper to set up Google Colab with ngrok tunnel."""
    
    @staticmethod
    def get_setup_code(ngrok_token: str) -> str:
        """Get code to run in Colab to expose Jupyter server."""
        return f'''# Run this code in Google Colab to expose Jupyter server
!pip install -q pyngrok

# Configure ngrok
from pyngrok import ngrok, conf
import time

conf.get_default().auth_token = "{ngrok_token}"

# Start Jupyter in background
import subprocess
import threading

def start_jupyter():
    subprocess.Popen([
        'jupyter', 'notebook',
        '--NotebookApp.token=',
        '--NotebookApp.password=',
        '--no-browser',
        '--port=8888',
        '--ip=0.0.0.0',
        '--allow-root'
    ])

thread = threading.Thread(target=start_jupyter)
thread.daemon = True
thread.start()

time.sleep(5)

# Create ngrok tunnel
public_url = ngrok.connect(8888, "http")
print("=" * 60)
print("✅ Jupyter Server is Ready!")
print("=" * 60)
print(f"Public URL: {{public_url}}")
print("=" * 60)
print("Copy the URL above and paste it in AutoML-Insight!")
print("=" * 60)
'''

    @staticmethod
    def get_colab_url() -> str:
        """Get Colab URL."""
        return "https://colab.research.google.com/"


class RemoteExecutor:
    """High-level executor for remote training."""
    
    def __init__(self, jupyter_client: JupyterServerClient):
        self.client = jupyter_client
        self.logs = []
    
    def execute_automl(
        self,
        data_path: str,
        target_col: Optional[str],
        task_type: str,
        random_seed: int = 42,
        max_features: int = 1000,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Execute AutoML pipeline on remote server.
        
        Args:
            data_path: Path to local CSV file
            target_col: Target column name (for classification)
            task_type: 'Classification' or 'Clustering'
            random_seed: Random seed for reproducibility
            max_features: Maximum features to use
            progress_callback: Function to call with progress updates
            
        Returns:
            Dictionary with training results
        """
        def log(msg: str, progress: int = None):
            self.logs.append(msg)
            if progress_callback:
                progress_callback(msg, progress)
        
        try:
            # Step 1: Upload dataset
            log("📤 Uploading dataset to remote server...", 10)
            if not self.client.upload_file(data_path, "automl_dataset.csv"):
                return {"success": False, "error": "Failed to upload dataset"}
            log("✅ Dataset uploaded", 20)
            
            # Step 2: Install dependencies
            log("📦 Installing dependencies...", 25)
            packages = ['scikit-learn', 'pandas', 'numpy', 'xgboost', 'optuna']
            if not self.client.install_packages(packages):
                log("⚠️ Package installation may have failed, continuing...")
            log("✅ Dependencies ready", 35)
            
            # Step 3: Execute training
            log("🤖 Training models on remote server...", 40)
            training_code = self._generate_training_code(
                target_col, task_type, random_seed, max_features
            )
            
            result = self.client.execute_code_via_file(training_code, timeout=600)
            
            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown error')
                errors = result.get('errors', [])
                full_error = f"{error_msg}\n" + '\n'.join(errors) if errors else error_msg
                return {"success": False, "error": full_error}
            
            log("✅ Training completed!", 70)
            
            # Step 4: Download results
            log("📥 Downloading results...", 80)
            local_results = "remote_results.json"
            if not self.client.download_file("automl_results.json", local_results):
                return {"success": False, "error": "Failed to download results"}
            
            # Load results
            with open(local_results, 'r') as f:
                results = json.load(f)
            
            log("✅ Results downloaded successfully!", 100)
            
            # Cleanup
            if os.path.exists(local_results):
                os.remove(local_results)
            
            return {"success": True, "results": results, "logs": self.logs}
            
        except Exception as e:
            return {"success": False, "error": str(e), "logs": self.logs}
    
    def _generate_training_code(
        self,
        target_col: Optional[str],
        task_type: str,
        random_seed: int,
        max_features: int
    ) -> str:
        """Generate optimized training code for remote execution."""
        
        code = f'''
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("AutoML-Insight Remote Execution")
print("=" * 60)

# Load data
print("Loading dataset...")
df = pd.read_csv('automl_dataset.csv')
print(f"Dataset shape: {{df.shape}}")

# Prepare data
{"X = df.drop('" + str(target_col) + "', axis=1)" if target_col else "X = df"}
{"y = df['" + str(target_col) + "']" if target_col else "y = None"}

# Preprocessing
print("Preprocessing...")
{"le = LabelEncoder(); y = le.fit_transform(y); print(f'Classes: {{le.classes_}}')" if target_col else ""}

# Feature selection - use only numeric features and limit count
X_numeric = X.select_dtypes(include=[np.number])
print(f"Numeric features: {{X_numeric.shape[1]}}")

# Limit features if too many
if X_numeric.shape[1] > {max_features}:
    print(f"Limiting to top {max_features} features by variance...")
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold()
    X_numeric = pd.DataFrame(
        selector.fit_transform(X_numeric),
        columns=X_numeric.columns[selector.get_support()]
    )
    if X_numeric.shape[1] > {max_features}:
        # Select top features by variance
        variances = X_numeric.var().sort_values(ascending=False)
        top_features = variances.head({max_features}).index
        X_numeric = X_numeric[top_features]
    print(f"Using {{X_numeric.shape[1]}} features")

# Fill missing values
X_numeric = X_numeric.fillna(X_numeric.median())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state={random_seed}
)

# Define models
models = {{
    'LogisticRegression': LogisticRegression(random_state={random_seed}, max_iter=1000, n_jobs=1),
    'RandomForest': RandomForestClassifier(random_state={random_seed}, n_estimators=100, max_depth=10, n_jobs=1),
    'XGBoost': XGBClassifier(random_state={random_seed}, n_estimators=100, max_depth=6, n_jobs=1, use_label_encoder=False, eval_metric='logloss'),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=1),
}}

# Train and evaluate
results = []
print("\\nTraining models...")
print("-" * 60)

for name, model in models.items():
    print(f"Training {{name}}...")
    try:
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({{
            'model': name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }})
        
        print(f"  ✅ {{name}}: Accuracy={{accuracy:.4f}}, F1={{f1:.4f}}")
        
    except Exception as e:
        print(f"  ❌ {{name}}: Failed - {{str(e)}}")
        results.append({{
            'model': name,
            'accuracy': 0.0,
            'error': str(e)
        }})

print("-" * 60)

# Find best model
best_result = max(results, key=lambda x: x.get('accuracy', 0))
print(f"\\n🏆 Best Model: {{best_result['model']}} (Accuracy: {{best_result['accuracy']:.4f}})")

# Save results
print("\\nSaving results...")
output = {{
    'results': results,
    'best_model': best_result['model'],
    'n_features': X_numeric.shape[1],
    'n_samples': len(df)
}}

with open('automl_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("✅ Results saved to automl_results.json")
print("=" * 60)
'''
        return code
