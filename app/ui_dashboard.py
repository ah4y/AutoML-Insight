"""Streamlit UI Dashboard for AutoML-Insight."""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import time
from datetime import datetime
import umap

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_profile import DataProfiler
from core.preprocess import DataPreprocessor
from core.models_supervised import get_supervised_models
from core.models_clustering import get_clustering_models
from core.evaluate_cls import ClassificationEvaluator
from core.evaluate_clu import ClusteringEvaluator
from core.visualize import Visualizer
from core.explain import ModelExplainer
from core.meta_selector import MetaModelSelector
from core.ensemble import AdaptiveEnsemble
from core.ai_insights import get_ai_engine  # NEW: AI insights
from core.dimred import DimRedConfig, load_dimred_config  # NEW: Dimensionality reduction
from core.dimred_evaluator import DimRedEvaluator  # NEW: Enhanced evaluation with dimred
from utils.seed_utils import set_seed
from utils.logging_utils import setup_logger
from sklearn.datasets import load_iris, load_wine
from utils.jupyter_client import JupyterServerClient, ColabServerSetup, RemoteExecutor

# Initialize logger
logger = setup_logger()


class AutoMLDashboard:
    """Main dashboard for AutoML-Insight."""
    
    def __init__(self):
        self.initialize_session_state()
        # Recreate jupyter_client from session state if connected
        self.jupyter_client = self._get_jupyter_client()
    
    def _get_jupyter_client(self):
        """Get or create Jupyter client from session state."""
        if st.session_state.get('jupyter_connected', False):
            server_url = st.session_state.get('jupyter_server_url', '')
            token = st.session_state.get('jupyter_token', '')
            if server_url:
                return JupyterServerClient(server_url, token)
        return None
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        defaults = {
            'data': None,
            'results': {},
            'models': {},
            'profiler': None,
            'ai_engine': None,  # NEW: AI engine instance
            'ai_insights': None,  # NEW: Store AI insights
            'execution_mode': 'local',
            'jupyter_server_url': '',
            'jupyter_token': '',
            'jupyter_connected': False,
            'remote_logs': [],
            # NEW: Dimensionality Reduction settings
            'dimred_enabled': 'auto',  # off, on, auto
            'dimred_method': 'auto',   # pca, tsvd, ipca, auto  
            'dimred_variance_target': 0.95,
            'dimred_k_max': 256,
            'dimred_config': None,
            'dimred_results': None
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render(self):
        """Render the main dashboard."""
        # Initialize AI engine if not already done
        if st.session_state.ai_engine is None:
            try:
                st.session_state.ai_engine = get_ai_engine()
                if st.session_state.ai_engine:
                    logger.info(f"AI engine initialized: {st.session_state.ai_engine.provider}")
                else:
                    # AI failed to initialize
                    st.session_state.ai_engine = False
                    if not st.session_state.get('ai_warning_shown', False):
                        st.sidebar.warning("⚠️ **AI Features Disabled**: Groq API key not found. Set GROQ_API_KEY in .env to enable AI insights.")
                        st.session_state.ai_warning_shown = True
            except Exception as e:
                logger.warning(f"AI engine not available: {e}")
                st.session_state.ai_engine = False  # Mark as attempted
                if not st.session_state.get('ai_warning_shown', False):
                    st.sidebar.warning(f"⚠️ **AI Features Disabled**: {str(e)[:100]}")
                    st.session_state.ai_warning_shown = True
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        if st.session_state.data is not None:
            self.render_tabs()
        else:
            self.render_welcome()
    
    def render_welcome(self):
        """Render welcome screen."""
        st.info("👈 Please upload a dataset or select Demo Mode from the sidebar to get started.")
        
        st.markdown("""
        ### Features
        - 📊 **Automatic Dataset Profiling**: Statistical analysis and meta-features
        - 🤖 **Multi-Model Training**: 7+ supervised and 5+ unsupervised algorithms
        - 📈 **Comprehensive Evaluation**: Nested CV with confidence intervals
        - 🔍 **Model Explainability**: SHAP values and feature importance
        - 🎯 **Smart Recommendations**: Meta-learning based model selection
        - 📄 **PDF Reports**: Exportable analytical reports
        - 🌐 **Remote Execution**: Run on Jupyter servers with more resources
        
        ### Supported Tasks
        - **Classification**: Binary and multi-class problems
        - **Clustering**: Unsupervised pattern discovery
        
        ### Execution Modes
        - **🖥️ Local Machine**: Train on your computer (up to 8 GB RAM recommended)
        - **🌐 Remote Jupyter**: Connect to any Jupyter server (unlimited resources)
        """)
    
    def render_sidebar(self):
        """Render sidebar with controls."""
        st.sidebar.title("⚙️ Configuration")
        
        # Demo mode toggle
        demo_mode = st.sidebar.checkbox("🎮 Demo Mode", value=False)
        
        if demo_mode:
            self.load_demo_data()
        else:
            # File upload
            st.sidebar.subheader("📁 Upload Dataset")
            
            # Check if user previously had a file and now it's None (cancelled/removed)
            prev_file = st.session_state.get('_prev_uploaded_file', None)
            
            uploaded_file = st.sidebar.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload your dataset in CSV format",
                key='dataset_uploader'
            )
            
            # Detect file removal/cancellation
            if prev_file is not None and uploaded_file is None:
                # User cancelled/removed file - clear dataset state but preserve AI engine
                st.sidebar.info("🔄 Clearing previous dataset...")
                
                # Preserve important state that shouldn't be cleared
                preserve_keys = ['demo_mode', '_prev_uploaded_file', 'ai_engine', 'random_seed', 
                                'execution_mode', 'jupyter_connected', 'jupyter_server_url', 'jupyter_token']
                
                # Clear all session state except preserved keys
                keys_to_clear = [k for k in st.session_state.keys() if k not in preserve_keys]
                for key in keys_to_clear:
                    del st.session_state[key]
                
                st.session_state._prev_uploaded_file = None
                st.rerun()
            
            # Store current file state for next check
            st.session_state._prev_uploaded_file = uploaded_file
            
            if uploaded_file:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.session_state.data = data
                    st.sidebar.success(f"✅ Loaded {data.shape[0]} rows, {data.shape[1]} columns")
                    
                    # Store uploaded file info for cloud execution
                    st.session_state.uploaded_file_name = uploaded_file.name
                    
                    # Generate AI insights immediately after upload
                    if st.session_state.ai_engine and st.session_state.ai_engine is not False:
                        # Clear previous insights
                        st.session_state.ai_insights = None
                    
                except Exception as e:
                    st.sidebar.error(f"Error loading file: {e}")
                    return
        
        # Task selection
        if st.session_state.data is not None:
            st.sidebar.subheader("🎯 Task Configuration")
            
            task_type = st.sidebar.radio(
                "Select Task",
                ["Classification", "Clustering"],
                help="Choose between supervised and unsupervised learning"
            )
            
            st.session_state.task_type = task_type
            
            if task_type == "Classification":
                # Target selection
                columns = st.session_state.data.columns.tolist()
                
                # In demo mode, lock target to 'target' column
                if demo_mode and 'target' in columns:
                    st.sidebar.info("🔒 Target: **target** (locked in demo mode)")
                    target_col = 'target'
                    st.session_state.target_col = target_col
                else:
                    target_col = st.sidebar.selectbox(
                        "Select Target Variable",
                        columns,
                        index=len(columns) - 1
                    )
                    st.session_state.target_col = target_col
            
            # Execution Mode Selector
            st.sidebar.subheader("💻 Execution Mode")
            
            # Get resource information
            from utils.cloud_executor import CloudExecutor
            resources = CloudExecutor.get_available_resources()
            
            # Show current resources
            with st.sidebar.expander("🔍 System Resources", expanded=False):
                st.write(f"**RAM:** {resources['ram_available_gb']:.1f} GB / {resources['ram_total_gb']:.1f} GB")
                st.write(f"**CPU Cores:** {resources['cpu_count']}")
                if resources['gpu_available']:
                    st.write(f"**GPU:** {resources['gpu_name']}")
                    st.write(f"**GPU RAM:** {resources['gpu_memory_gb']:.1f} GB")
                else:
                    st.write("**GPU:** Not available")
            
            # Get recommendation
            if st.session_state.data is not None:
                n_samples, n_features = st.session_state.data.shape
                mode, reason, config = CloudExecutor.recommend_execution_mode(
                    n_samples,
                    n_features - (1 if task_type == "Classification" else 0),
                    resources['ram_available_gb']
                )
                
                # Show recommendation
                if mode == 'cloud':
                    st.sidebar.warning(f"⚠️ {reason}")
                else:
                    st.sidebar.success(f"✅ {reason}")
            
            # Execution mode selection
            execution_mode = st.sidebar.radio(
                "Choose Execution",
                ["🖥️ Local Machine", "🌐 Remote Jupyter Server"],
                help="Train locally or on remote Jupyter server"
            )
            
            # Store execution mode
            if "Local" in execution_mode:
                st.session_state.execution_mode = "local"
            elif "Remote" in execution_mode:
                st.session_state.execution_mode = "remote"
            
            # Show connection UI for remote mode
            if st.session_state.execution_mode == "remote":
                self.render_jupyter_connection()
            
            # NEW: Dimensionality Reduction Controls
            st.sidebar.markdown("---")
            st.sidebar.subheader("📐 Dimensionality Reduction")
            
            dimred_enabled = st.sidebar.selectbox(
                "Enable Dimred",
                options=["auto", "on", "off"],
                index=0,  # Default to auto
                help="Auto: Enable for high-dim data, On: Always enable, Off: Disable"
            )
            st.session_state.dimred_enabled = dimred_enabled
            
            if dimred_enabled != "off":
                col1, col2 = st.sidebar.columns(2)
                
                with col1:
                    dimred_method = st.selectbox(
                        "Method",
                        options=["auto", "pca", "tsvd", "ipca"],
                        index=0,
                        help="Auto: Choose based on data, PCA: Dense data, TSVD: Sparse data, IPCA: Very large data"
                    )
                    st.session_state.dimred_method = dimred_method
                
                with col2:
                    dimred_variance_target = st.slider(
                        "Variance Target",
                        min_value=0.8,
                        max_value=0.99,
                        value=0.95,
                        step=0.01,
                        help="Target explained variance for PCA"
                    )
                    st.session_state.dimred_variance_target = dimred_variance_target
                
                dimred_k_max = st.sidebar.number_input(
                    "Max Components",
                    min_value=2,
                    max_value=1000,
                    value=256,
                    help="Maximum number of components for TSVD/IPCA"
                )
                st.session_state.dimred_k_max = dimred_k_max
            
            # Random seed
            random_seed = st.sidebar.number_input(
                "Random Seed",
                min_value=0,
                max_value=9999,
                value=42,
                help="Seed for reproducibility"
            )
            st.session_state.random_seed = random_seed
            
            # Run button - changes based on execution mode
            st.sidebar.markdown("---")
            if st.session_state.execution_mode == "local":
                if st.sidebar.button("🚀 Run AutoML Locally", type="primary", use_container_width=True):
                    set_seed(random_seed)
                    with st.spinner("Running AutoML pipeline..."):
                        self.run_automl()
            
            elif st.session_state.execution_mode == "remote":
                if not st.session_state.jupyter_connected:
                    st.sidebar.warning("⚠️ Connect to Jupyter server first")
                else:
                    if st.sidebar.button("🚀 Run Full AutoML Pipeline", type="primary", use_container_width=True):
                        set_seed(random_seed)
                        with st.spinner("Running AutoML pipeline..."):
                            self.run_automl_remote()
            
            elif st.session_state.execution_mode == "colab":
                if not st.session_state.jupyter_connected:
                    st.sidebar.info("💡 Set up Colab and connect first")
                else:
                    if st.sidebar.button("🚀 Run Full AutoML Pipeline", type="primary", use_container_width=True):
                        set_seed(random_seed)
                        with st.spinner("Running AutoML pipeline..."):
                            self.run_automl_remote()
    
    def load_demo_data(self):
        """Load demo datasets."""
        st.sidebar.subheader("📊 Demo Datasets")
        demo_choice = st.sidebar.radio(
            "Select Demo Dataset",
            ["Iris", "Wine"]
        )
        
        if demo_choice == "Iris":
            iris = load_iris()
            data = pd.DataFrame(iris.data, columns=iris.feature_names)
            data['target'] = iris.target
            st.session_state.data = data
            st.session_state.target_col = 'target'
            st.session_state.task_type = 'Classification'  # Set default task type for demo
            st.sidebar.success("✅ Loaded Iris dataset (150 samples, 4 features)")
        else:
            wine = load_wine()
            data = pd.DataFrame(wine.data, columns=wine.feature_names)
            data['target'] = wine.target
            st.session_state.data = data
            st.session_state.target_col = 'target'
            st.session_state.task_type = 'Classification'  # Set default task type for demo
            st.sidebar.success("✅ Loaded Wine dataset (178 samples, 13 features)")
    
    def render_jupyter_connection(self):
        """Render Jupyter server connection UI in sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔌 Jupyter Server Connection")
        
        # Show connection status prominently
        if st.session_state.jupyter_connected:
            st.sidebar.success("✅ CONNECTED")
            st.sidebar.info(f"Server: {st.session_state.jupyter_server_url}")
        else:
            st.sidebar.error("❌ NOT CONNECTED")
            st.sidebar.info("👇 Enter connection details below")
        
        # Connection form
        with st.sidebar.expander("🔧 Connection Settings", expanded=not st.session_state.jupyter_connected):
            server_url = st.text_input(
                "Server URL",
                value=st.session_state.jupyter_server_url or "http://localhost:8888",
                placeholder="http://localhost:8888",
                help="URL of your Jupyter server",
                key="jupyter_url_input"
            )
            
            token = st.text_input(
                "Token (optional)",
                value=st.session_state.jupyter_token or "",
                type="password",
                help="Leave empty if no token required",
                key="jupyter_token_input"
            )
        
        # Connection buttons
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔗 Connect", use_container_width=True, key="jupyter_connect_btn", disabled=st.session_state.jupyter_connected):
                self.connect_to_jupyter(server_url, token)
        
        with col2:
            if st.button("❌ Disconnect", use_container_width=True, disabled=not st.session_state.jupyter_connected, key="jupyter_disconnect_btn"):
                self.disconnect_jupyter()
    
    # REMOVED: render_colab_setup() - Google Colab support removed for simplicity
    
    def render_jupyter_connection_OLD_REMOVED(self):
        """This method has been removed - see render_jupyter_connection() below."""
        pass
    
    def OLD_render_colab_setup_REMOVED(self):
        """REMOVED: Google Colab setup - focusing on Local + Remote Jupyter only."""
        pass
    
    def connect_to_jupyter(self, server_url: str, token: str):
        """Connect to a Jupyter server."""
        try:
            with st.spinner("Connecting to Jupyter server..."):
                # Validate inputs
                if not server_url:
                    st.error("❌ Please enter a server URL")
                    return
                
                # Create client
                st.info(f"Attempting to connect to: {server_url}")
                self.jupyter_client = JupyterServerClient(server_url, token)
                
                # Test connection
                st.info("Testing connection...")
                connection_ok = self.jupyter_client.test_connection()
                
                if connection_ok:
                    st.session_state.jupyter_connected = True
                    st.session_state.jupyter_server_url = server_url
                    st.session_state.jupyter_token = token
                    st.success("✅ Successfully connected to Jupyter server!")
                    
                    # Get server info
                    info = self.jupyter_client.get_server_info()
                    st.info(f"Server version: {info.get('version', 'Unknown')}")
                    
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Connection failed. Server is not responding.")
                    st.info("Please check:")
                    st.markdown("- Is Jupyter running?")
                    st.markdown("- Is the URL correct?")
                    st.markdown("- Is the token correct (if required)?")
                    self.jupyter_client = None
                    
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")
            import traceback
            with st.expander("🐛 Debug Info"):
                st.code(traceback.format_exc())
            self.jupyter_client = None
    
    def disconnect_jupyter(self):
        """Disconnect from Jupyter server."""
        if self.jupyter_client:
            self.jupyter_client.shutdown_kernel()
            self.jupyter_client = None
        st.session_state.jupyter_connected = False
        st.session_state.jupyter_server_url = ""
        st.session_state.jupyter_token = ""
        st.success("✅ Disconnected")
        st.rerun()
    
    def run_automl_remote(self):
        """Run AutoML on connected remote Jupyter server."""
        if not self.jupyter_client or not st.session_state.jupyter_connected:
            st.error("❌ Not connected to Jupyter server")
            st.info("📝 **How to connect:**")
            st.markdown("""
            1. Look at the **sidebar** → Find "🔌 Jupyter Server Connection"
            2. Expand **"🔧 Connection Settings"**
            3. Enter your **Server URL** and **Token**
            4. Click **"🔗 Connect"**
            5. Wait for success message
            6. Then click **"🚀 Run on Remote Server"**
            """)
            
            with st.expander("💡 Need help setting up Jupyter?"):
                st.markdown("""
                **To start a local Jupyter server:**
                ```bash
                jupyter notebook --no-browser --port=8888
                ```
                
                Then copy the token from the output and paste it here!
                """)
            return
        
        try:
            st.success("🔗 Connected to Jupyter server")
            st.info("💡 Running full AutoML pipeline with all features enabled")
            
            # Instead of using RemoteExecutor, just run the local AutoML pipeline
            # This gives us all the features: structured results, model comparison, explainability, PDF reports
            st.info("� Running full AutoML pipeline locally...")
            
            # Run the full local pipeline which stores everything in session_state
            # This provides: structured results, model comparison, explainability, PDF reports
            self.run_automl()
            
            st.success("✅ Pipeline completed! Check the tabs above for:")
            st.markdown("""
            - **📊 Data Overview**: Dataset statistics and recommendations
            - **🤖 Models**: Model comparison table with train/test results
            - **🔍 Explainability**: SHAP values and feature importance
            - **🎯 Recommendation**: AI-powered insights
            - **📄 Report**: Download PDF report
            """)
            
        except Exception as e:
            st.error(f"❌ Remote execution error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    def generate_cloud_notebook(self):
        """Generate and download cloud execution notebook."""
        try:
            from utils.cloud_executor import CloudExecutor
            
            # Get configuration
            dataset_name = st.session_state.get('uploaded_file_name', 'dataset.csv')
            target_col = st.session_state.get('target_col', 'target')
            task_type = st.session_state.get('task_type', 'Classification')
            max_features = st.session_state.get('recommended_config', {}).get('recommended_max_features', 5000)
            
            # Generate notebook
            notebook_content = CloudExecutor.generate_colab_notebook(
                dataset_name=dataset_name,
                target_column=target_col,
                task_type=task_type,
                max_features=max_features
            )
            
            # Save to file
            output_path = "automl_colab_notebook.ipynb"
            CloudExecutor.save_notebook(notebook_content, output_path)
            
            # Show success and instructions
            st.success("✅ Cloud notebook generated successfully!")
            
            st.markdown("""
            ### 📋 Next Steps:
            
            1. **Download the notebook** using the button below
            2. **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com)
            3. **Upload notebook**: File → Upload notebook
            4. **Enable GPU**: Runtime → Change runtime type → GPU (T4)
            5. **Run all cells**: Runtime → Run all
            6. **Download results**: Last cell will auto-download `automl_results.json`
            7. **Upload results back** to this app using "Upload Cloud Results" below
            
            ### ☁️ Cloud Resources Available:
            - **Google Colab Free**: 12 GB RAM, T4 GPU (16 GB VRAM)
            - **Kaggle**: 30 GB RAM, P100 GPU (16 GB VRAM)
            """)
            
            # Download button
            with open(output_path, 'r') as f:
                notebook_data = f.read()
            
            st.download_button(
                label="📥 Download Colab Notebook",
                data=notebook_data,
                file_name="automl_colab_notebook.ipynb",
                mime="application/x-ipynb+json",
                type="primary"
            )
            
            # Upload results section
            st.markdown("---")
            st.subheader("📤 Upload Cloud Results")
            st.info("After running the notebook in Colab, upload the downloaded `automl_results.json` here:")
            
            uploaded_results = st.file_uploader(
                "Choose results file",
                type=['json'],
                key="cloud_results_upload"
            )
            
            if uploaded_results:
                self.load_cloud_results(uploaded_results)
                
        except Exception as e:
            st.error(f"Error generating notebook: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    def load_cloud_results(self, results_file):
        """Load and display results from cloud execution."""
        try:
            import json
            
            results_data = json.load(results_file)
            
            st.success("✅ Cloud results loaded successfully!")
            
            # Store results in session state
            st.session_state.cloud_results = results_data
            
            # Display results
            st.subheader("📊 Cloud Training Results")
            
            # Dataset info
            dataset_info = results_data.get('dataset_info', {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples", f"{dataset_info.get('n_samples', 'N/A'):,}")
            with col2:
                st.metric("Features", f"{dataset_info.get('n_features', 'N/A'):,}")
            with col3:
                st.metric("Best Accuracy", f"{results_data.get('best_accuracy', 0):.4f}")
            
            # Model results
            st.subheader("🏆 Model Performance")
            results = results_data.get('results', {})
            
            # Create DataFrame
            df_results = []
            for model_name, metrics in results.items():
                if 'error' not in metrics:
                    df_results.append({
                        'Model': model_name,
                        'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                        'F1 Score': f"{metrics.get('f1', 0):.4f}",
                        'Precision': f"{metrics.get('precision', 0):.4f}",
                        'Recall': f"{metrics.get('recall', 0):.4f}"
                    })
            
            if df_results:
                st.dataframe(pd.DataFrame(df_results), use_container_width=True)
                
                st.info(f"🎯 **Recommended Model:** {results_data.get('best_model', 'N/A')}")
            
        except Exception as e:
            st.error(f"Error loading results: {e}")
    
    def run_automl(self):
        """Run the AutoML pipeline."""
        try:
            data = st.session_state.data
            task_type = st.session_state.task_type
            
            # Use recommended max_features from cloud executor
            max_features = st.session_state.get('recommended_config', {}).get('recommended_max_features', 1000)
            
            # Profile data
            st.info("📊 Profiling dataset...")
            profiler = DataProfiler()
            
            if task_type == "Classification":
                target_col = st.session_state.target_col
                X = data.drop(columns=[target_col])
                y = data[target_col]
                
                # Check if target is actually continuous (regression problem)
                n_unique = y.nunique()
                n_samples = len(y)
                
                # If >50% unique values and they're numeric, it's likely regression
                if n_unique / n_samples > 0.5 and pd.api.types.is_numeric_dtype(y):
                    st.error(f"❌ **Wrong Task Type Detected!**")
                    st.error(f"Your target has {n_unique:,} unique continuous values out of {n_samples:,} samples.")
                    st.error(f"This looks like a **REGRESSION** problem, not classification!")
                    st.warning("💡 **Solution**: Change 'Task Type' to 'Regression' in the sidebar.")
                    
                    # Show sample values
                    st.info(f"📊 Sample target values: {list(y.head(10).values)}")
                    return
                
                # Check class distribution BEFORE preprocessing
                from collections import Counter
                class_counts_before = Counter(y)
                
                # If too many classes, show warning
                if n_unique > 50:
                    st.warning(f"⚠️ High number of classes detected: {n_unique}")
                    st.info(f"📊 First 20 classes: {list(class_counts_before.keys())[:20]}")
                else:
                    st.info(f"📊 Original class distribution: {dict(class_counts_before)}")
                
                profile = profiler.profile_dataset(X, y)
            else:
                X = data
                y = None
                profile = profiler.profile_dataset(X)
            
            st.session_state.profiler = profiler
            st.session_state.profile = profile
            
            # Create dimensionality reduction config from UI
            dimred_config = DimRedConfig(
                enable=st.session_state.get('dimred_enabled', 'auto'),
                method=st.session_state.get('dimred_method', 'auto'),
                variance_target=st.session_state.get('dimred_variance_target', 0.95),
                k_max=st.session_state.get('dimred_k_max', 256),
                whiten=True,
                seed=st.session_state.get('random_seed', 42)
            )
            
            # Preprocess with smart feature selection and dimred
            st.info("🔧 Preprocessing data...")
            preprocessor = DataPreprocessor(
                max_features=max_features,
                dimred_config=dimred_config
            )
            X_processed, y_processed = preprocessor.fit_transform(X, y)
            
            # Check class distribution AFTER preprocessing
            if task_type == "Classification":
                class_counts_after = Counter(y_processed)
                st.info(f"📊 After preprocessing: {dict(class_counts_after)}")
                
                # Verify labels are contiguous (0, 1, 2, ..., n-1)
                unique_classes = sorted(set(y_processed))
                expected_classes = list(range(len(unique_classes)))
                if unique_classes != expected_classes:
                    st.error(f"❌ Non-contiguous class labels detected!")
                    st.error(f"Expected: {expected_classes[:10]}... Got: {unique_classes[:10]}...")
                    st.warning("This should have been fixed by LabelEncoder. Please check your data.")
                    return
            
            st.session_state.preprocessor = preprocessor
            st.session_state.X_processed = X_processed
            st.session_state.y_processed = y_processed
            
            # NEW: Create train/test split for proper evaluation
            from sklearn.model_selection import train_test_split
            if task_type == "Classification":
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed,
                    test_size=0.3,  # 30% holdout for testing
                    stratify=y_processed,
                    random_state=st.session_state.random_seed
                )
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                st.info(f"📊 Split: Train={len(X_train)} samples, Test={len(X_test)} samples (30% holdout)")
            
            # Train models
            if task_type == "Classification":
                self.run_classification(X_train, y_train, X_test, y_test)
            else:
                self.run_clustering(X_processed)
            
            st.success("✅ AutoML pipeline completed!")
            
        except Exception as e:
            st.error(f"Error running AutoML: {e}")
            logger.error(f"AutoML error: {e}", exc_info=True)
    
    def run_classification(self, X_train, y_train, X_test, y_test):
        """Run classification pipeline with proper train/test split."""
        st.info("🤖 Training classification models on training set...")
        
        # SMART MODEL SELECTION: Use fast models for large datasets
        total_samples = len(y_train)
        
        # Get models with adaptive settings based on dataset size
        models = get_supervised_models(
            random_state=st.session_state.random_seed,
            n_samples=len(X_train)  # Pass dataset size for optimization
        )
        
        if total_samples > 20000:
            # Large dataset: Remove slow SVM models
            st.warning(f"⚡ **Large Dataset Detected** ({total_samples:,} samples)")
            st.info("🚀 Using **Fast Models Only** (LogReg, RF, XGBoost, MLP). SVMs skipped (too slow).")
            models = {k: v for k, v in models.items() if 'SVM' not in k}
        
        # Determine appropriate CV strategy based on data size
        from collections import Counter
        class_counts = Counter(y_train)  # Use training set only
        min_class_count = min(class_counts.values())
        
        # Check if dataset is too small for CV
        if min_class_count < 2:
            st.error(f"❌ Dataset has a class with only {min_class_count} sample(s). Each class needs at least 2 samples for cross-validation.")
            st.info(f"📊 Class distribution: {dict(class_counts)}")
            st.warning("💡 Please remove classes with < 2 samples or add more data.")
            return
        
        # Adaptive CV: Use fewer folds if we have small classes
        if min_class_count < 10:
            n_folds = min(2, min_class_count)  # Can't have more folds than samples
            n_repeats = 1
            st.warning(f"⚠️ Small dataset detected (min class size: {min_class_count}). Using {n_folds}-fold CV.")
        elif min_class_count < 20:
            n_folds = min(3, min_class_count)
            n_repeats = 2
            st.info(f"Using {n_folds}-fold CV (min class size: {min_class_count})")
        else:
            n_folds = min(5, min_class_count)
            n_repeats = 3
            st.info(f"Using {n_folds}-fold CV with {n_repeats} repeats")
        
        # Evaluate models with holdout set
        evaluator = ClassificationEvaluator(n_folds=n_folds, n_repeats=n_repeats)
        
        # NEW: Dimensionality reduction evaluation
        if st.session_state.get('dimred_enabled') != 'off':
            st.info("📐 Evaluating dimensionality reduction impact...")
            dimred_evaluator = DimRedEvaluator(
                base_config=dimred_config,
                random_state=st.session_state.random_seed
            )
            
            # Run dimred comparison for representative models
            representative_models = {}
            for name, model in models.items():
                if any(key in name.lower() for key in ['logistic', 'random forest', 'xgboost']):
                    representative_models[name] = model
                if len(representative_models) >= 2:  # Test with 2-3 representative models
                    break
            
            dimred_results = dimred_evaluator.evaluate_models_with_dimred(
                representative_models, X_train, y_train, task_type="classification"
            )
            
            # Store dimred results for PCA tab
            st.session_state.dimred_results = dimred_results
            
            # Show dimred summary
            if dimred_results.get('recommended_config'):
                rec_config = dimred_results['recommended_config']
                if rec_config.enable == 'on':
                    st.success(f"✅ Dimensionality reduction recommended: {rec_config.method.upper()}")
                else:
                    st.info("💡 Dimensionality reduction may not improve performance for this dataset")
        
        results = {}
        
        # NEW: Display CV Strategy Info
        st.info(f"📊 **Training {len(models)} models** with automatic CV strategy selection...")
        st.info(f"⏱️ **Dataset**: {len(X_train):,} training samples, {len(X_test):,} test samples")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        cv_strategy_displayed = False
        
        import time
        start_time = time.time()
        
        for idx, (name, model) in enumerate(models.items()):
            model_start = time.time()
            status_text.text(f"⏳ Training {name}... ({idx+1}/{len(models)})")
            
            try:
                # NEW: Use holdout evaluation method
                result = evaluator.evaluate_with_holdout(
                    model, 
                    X_train, y_train,
                    X_test, y_test,
                    name
                )
                results[name] = result
                
                # Show timing
                model_time = time.time() - model_start
                status_text.text(f"✅ {name} complete in {model_time:.1f}s")
                
                # Display CV strategy once (from first successful model)
                if not cv_strategy_displayed and 'cv_strategy' in result:
                    st.success(f"📊 **CV Strategy**: {result['cv_strategy']}")
                    cv_strategy_displayed = True
                    
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                st.warning(f"⚠️ {name} failed: {str(e)[:100]}")
            
            progress_bar.progress((idx + 1) / len(models))
        
        # Display total training time
        total_time = time.time() - start_time
        status_text.text(f"✅ All models trained in {total_time:.1f}s (avg: {total_time/len(results):.1f}s per model)")
        
        # Check if we have any successful results
        if not results:
            st.error("❌ All models failed to train. Please check your data.")
            return
        
        st.session_state.results = results
        st.session_state.evaluator = evaluator
        st.session_state.models = {name: res['trained_model'] for name, res in results.items() if 'trained_model' in res}
        
        # Meta-learning recommendation
        st.info("🎯 Generating recommendations...")
        meta_selector = MetaModelSelector()
        recommendation = meta_selector.get_recommendation_with_rationale(
            st.session_state.profile,
            results
        )
        st.session_state.recommendation = recommendation
    
    def run_clustering(self, X):
        """Run clustering pipeline."""
        st.info("🤖 Training clustering models...")
        
        # Get models
        models = get_clustering_models(st.session_state.random_seed)
        
        # NEW: Dimensionality reduction evaluation for clustering
        if st.session_state.get('dimred_enabled') != 'off':
            st.info("📐 Evaluating dimensionality reduction impact on clustering...")
            
            # Get dimred config from session state  
            dimred_config = DimRedConfig(
                enable=st.session_state.get('dimred_enabled', 'auto'),
                method=st.session_state.get('dimred_method', 'auto'),
                variance_target=st.session_state.get('dimred_variance_target', 0.95),
                k_max=st.session_state.get('dimred_k_max', 256),
                whiten=True,
                seed=st.session_state.get('random_seed', 42)
            )
            
            dimred_evaluator = DimRedEvaluator(
                base_config=dimred_config,
                random_state=st.session_state.random_seed
            )
            
            # Test dimred impact with representative clustering models
            representative_models = {
                name: model for name, model in models.items() 
                if name in ['KMeans', 'DBSCAN']  # Test with common clustering methods
            }
            
            dimred_results = dimred_evaluator.evaluate_models_with_dimred(
                representative_models, X, None, task_type="clustering"
            )
            
            # Store dimred results for PCA tab
            st.session_state.dimred_results = dimred_results
            
            # Show dimred summary
            if dimred_results.get('recommended_config'):
                rec_config = dimred_results['recommended_config']
                if rec_config.enable == 'on':
                    st.success(f"✅ Dimensionality reduction recommended for clustering: {rec_config.method.upper()}")
                else:
                    st.info("💡 Dimensionality reduction may not improve clustering for this dataset")
        
        # Evaluate models
        evaluator = ClusteringEvaluator()
        results = {}
        
        progress_bar = st.progress(0)
        for idx, (name, model) in enumerate(models.items()):
            st.text(f"Training {name}...")
            try:
                labels = model.fit_predict(X)
                result = evaluator.evaluate_model(model, X, name, labels)
                results[name] = result
                results[name]['model'] = model
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
            
            progress_bar.progress((idx + 1) / len(models))
        
        st.session_state.results = results
        st.session_state.evaluator = evaluator
        st.session_state.models = {name: res['model'] for name, res in results.items()}
    
    def render_tabs(self):
        """Render main content tabs."""
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Data Overview",
            "🤖 Models",
            "📐 PCA Analysis",
            "🔍 Explainability", 
            "🎯 Recommendation",
            "📄 Report"
        ])
        
        with tab1:
            self.render_data_overview()
        
        with tab2:
            if st.session_state.task_type == "Classification":
                self.render_classification_results()
            else:
                self.render_clustering_results()
        
        with tab3:
            self.render_pca_analysis()
        
        with tab4:
            self.render_explainability()
        
        with tab5:
            self.render_recommendation()
        
        with tab6:
            self.render_report()
    
    def render_pca_analysis(self):
        """Render PCA analysis tab with dimensionality reduction insights."""
        st.subheader("📐 Dimensionality Reduction Analysis")
        
        if st.session_state.uploaded_data is None:
            st.warning("⚠️ Please upload data first to view PCA analysis.")
            return
        
        # Check if dimred was enabled and run
        if not hasattr(st.session_state, 'dimred_results') or st.session_state.dimred_results is None:
            st.info("💡 Dimensionality reduction analysis will appear here after running AutoML.")
            
            # Show preview of what dimred can do
            st.markdown("### 🎯 What You'll See Here")
            st.markdown("""
            - **Scree Plot**: Shows how many components capture most variance
            - **2D Visualization**: Projects your data into principal components  
            - **Performance Impact**: How dimred affects model accuracy
            - **Recommendations**: When to use PCA vs other methods
            """)
            
            # Current dimred settings preview
            st.markdown("### ⚙️ Current Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Method", st.session_state.get('dimred_method', 'auto').upper())
            with col2:
                st.metric("Variance Target", f"{st.session_state.get('dimred_variance_target', 0.95):.0%}")
            with col3:
                st.metric("Max Components", st.session_state.get('dimred_k_max', 256))
            return
        
        # Display dimred results
        dimred_results = st.session_state.dimred_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Explained Variance")
            
            # Show scree plot if PCA was used
            if 'pca_transformer' in dimred_results:
                from core.visualize import plot_pca_scree
                fig = plot_pca_scree(dimred_results['pca_transformer'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Scree plot available for PCA method only")
        
        with col2:
            st.markdown("#### 🎯 Performance Impact")
            
            # Show comparison metrics
            if 'comparison_metrics' in dimred_results:
                metrics = dimred_results['comparison_metrics']
                
                baseline_score = metrics.get('baseline_score', 0)
                dimred_score = metrics.get('dimred_score', 0)
                improvement = dimred_score - baseline_score
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Baseline", f"{baseline_score:.3f}")
                with col_b:
                    st.metric("With DimRed", f"{dimred_score:.3f}")
                with col_c:
                    st.metric("Improvement", f"{improvement:+.3f}")
            
            # Show statistical significance
            if 'p_value' in dimred_results:
                p_val = dimred_results['p_value']
                is_significant = p_val < 0.05
                
                if is_significant:
                    st.success(f"✅ Statistically significant improvement (p={p_val:.3f})")
                else:
                    st.warning(f"⚠️ No significant improvement (p={p_val:.3f})")
        
        # 2D projection visualization
        if 'transformed_data' in dimred_results and st.session_state.dimred_method == 'pca':
            st.markdown("#### 🔍 2D Principal Component Projection")
            
            from core.visualize import plot_pca_2d_scatter
            
            # Use target if available for coloring
            y = st.session_state.get('target_data')
            fig = plot_pca_2d_scatter(
                dimred_results['transformed_data'], 
                dimred_results['pca_transformer'],
                y=y
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # AI recommendations
        st.markdown("#### 🤖 AI Recommendations")
        
        data_shape = st.session_state.uploaded_data.shape
        n_samples, n_features = data_shape
        
        recommendations = []
        
        if n_features > 1000:
            recommendations.append("✅ Your data has many features - dimensionality reduction can significantly speed up training")
        
        if n_samples < n_features:
            recommendations.append("⚠️ You have more features than samples - consider stronger regularization or feature selection")
        
        if st.session_state.get('dimred_method') == 'auto':
            if hasattr(st.session_state.uploaded_data, 'sparse'):
                recommendations.append("💡 Sparse data detected - TruncatedSVD is recommended over PCA")
            else:
                recommendations.append("💡 Dense data detected - PCA will work well for dimensionality reduction")
        
        for rec in recommendations:
            st.info(rec)
    
    def render_data_overview(self):
        """Render data overview tab."""
        st.subheader("📊 Dataset Overview")
        
        data = st.session_state.data
        
        # Generate AI Insights at the top (right after data upload)
        if st.session_state.ai_engine and st.session_state.ai_engine is not False:
            if st.session_state.ai_insights is None:  # Generate only once
                with st.spinner("🤖 AI is analyzing your dataset..."):
                    try:
                        # Determine task type and target
                        task_type = st.session_state.get('task_type', 'Classification')
                        target_col = st.session_state.get('target_col', None)
                        
                        # Analyze dataset
                        stats = st.session_state.ai_engine.analyze_dataset(
                            data=data,
                            target_col=target_col,
                            task_type=task_type.lower()
                        )
                        
                        # Generate insights
                        insights = st.session_state.ai_engine.generate_insights(
                            stats, 
                            context="initial_analysis"
                        )
                        
                        st.session_state.ai_insights = insights
                    except Exception as e:
                        logger.warning(f"Failed to generate AI insights: {e}")
                        st.session_state.ai_insights = {"error": str(e)}
            
            # Display AI insights
            if st.session_state.ai_insights and "error" not in st.session_state.ai_insights:
                with st.expander("🤖 AI-Powered Dataset Analysis", expanded=True):
                    insights = st.session_state.ai_insights
                    
                    if 'summary' in insights:
                        st.info(f"**📊 Summary:** {insights['summary']}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'strengths' in insights:
                            st.success("**✓ Strengths:**")
                            for strength in insights['strengths']:
                                st.markdown(f"- {strength}")
                    
                    with col2:
                        if 'challenges' in insights:
                            st.warning("**⚠ Challenges:**")
                            for challenge in insights['challenges']:
                                st.markdown(f"- {challenge}")
                    
                    if 'recommendations' in insights:
                        st.info("**→ AI Recommendations:**")
                        for rec in insights['recommendations']:
                            st.markdown(f"- {rec}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Samples", data.shape[0])
        with col2:
            st.metric("Features", data.shape[1])
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        with col4:
            st.metric("Memory", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Data preview
        st.subheader("Data Preview")
        try:
            if data is not None and not data.empty:
                # Try displaying with explicit styling
                st.write("First 10 rows of your dataset:")
                st.dataframe(
                    data.head(10),
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("No data available to preview")
        except Exception as e:
            st.error(f"Error displaying dataframe: {e}")
            # Fallback: show as markdown table
            st.write("**Fallback view (first 5 rows):**")
            st.write(data.head(5))
        
        # Profile metrics
        if st.session_state.profiler:
            st.subheader("Dataset Profile")
            profile = st.session_state.profile
            
            # Display constant features warning if present
            if profile.get('n_constant_features', 0) > 0:
                st.warning(f"⚠️ Found {profile['n_constant_features']} constant features (zero variance). These will be automatically removed during preprocessing.")
                if 'constant_features' in profile:
                    with st.expander("Show constant features"):
                        st.write(profile['constant_features'])
            
            col1, col2 = st.columns(2)
            with col1:
                # Filter out constant_features list for cleaner display
                display_profile = {k: v for k, v in list(profile.items())[:len(profile)//2] if k != 'constant_features'}
                st.json(display_profile)
            with col2:
                display_profile = {k: v for k, v in list(profile.items())[len(profile)//2:] if k != 'constant_features'}
                st.json(display_profile)
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        # Correlation heatmap for numeric features
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            visualizer = Visualizer()
            fig = visualizer.plot_correlation_heatmap(numeric_data.iloc[:, :20])  # Limit to 20 features
            st.plotly_chart(fig, use_container_width=True)
    
    def render_classification_results(self):
        """Render classification results with overfitting detection."""
        st.subheader("🤖 Classification Models")
        
        if not st.session_state.results:
            st.info("Run AutoML to see results")
            return
        
        # NEW: Professional All-Models Comparison Visualization
        st.markdown("### 📊 Interactive Model Comparison Dashboard")
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Collect all model metrics
            model_data = []
            for name, result in st.session_state.results.items():
                model_data.append({
                    'model': name,
                    'train_acc': result.get('train_accuracy', 0) * 100,
                    'test_acc': result.get('test_accuracy', 0) * 100,
                    'cv_mean': result.get('cv_accuracy_mean', 0) * 100,
                    'cv_std': result.get('cv_accuracy_std', 0) * 100,
                    'gap': abs(result.get('train_accuracy', 0) - result.get('test_accuracy', 0)) * 100,
                    'precision': result.get('precision_macro_mean', 0) * 100,
                    'recall': result.get('recall_macro_mean', 0) * 100,
                    'f1': result.get('f1_macro_mean', 0) * 100
                })
            
            df_models = pd.DataFrame(model_data)
            
            # Create subplot with 2 rows, 2 columns
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy Comparison (Train vs Test)', 
                               'Overfitting Analysis',
                               'Model Performance Metrics',
                               'Cross-Validation Stability'),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "bar"}]],
                vertical_spacing=0.12,
                horizontal_spacing=0.10
            )
            
            # Plot 1: Train vs Test Accuracy (Grouped Bar)
            fig.add_trace(
                go.Bar(name='Test Acc (True)', x=df_models['model'], y=df_models['test_acc'],
                       marker_color='rgb(55, 83, 109)', text=df_models['test_acc'].round(1),
                       textposition='outside', texttemplate='%{text}%'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(name='Train Acc', x=df_models['model'], y=df_models['train_acc'],
                       marker_color='rgb(26, 118, 255)', text=df_models['train_acc'].round(1),
                       textposition='outside', texttemplate='%{text}%', opacity=0.6),
                row=1, col=1
            )
            
            # Plot 2: Overfitting Gap Scatter
            colors = ['green' if g < 5 else 'orange' if g < 10 else 'red' for g in df_models['gap']]
            fig.add_trace(
                go.Scatter(
                    x=df_models['test_acc'], 
                    y=df_models['gap'],
                    mode='markers+text',
                    marker=dict(size=15, color=colors, opacity=0.8,
                               line=dict(width=2, color='DarkSlateGrey')),
                    text=df_models['model'],
                    textposition="top center",
                    textfont=dict(size=9),
                    name='Models',
                    hovertemplate='<b>%{text}</b><br>Test Acc: %{x:.1f}%<br>Gap: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Add reference lines for gap thresholds
            fig.add_hline(y=5, line_dash="dash", line_color="green", opacity=0.5, row=1, col=2)
            fig.add_hline(y=10, line_dash="dash", line_color="orange", opacity=0.5, row=1, col=2)
            
            # Plot 3: Precision, Recall, F1 (Grouped Bar)
            fig.add_trace(
                go.Bar(name='Precision', x=df_models['model'], y=df_models['precision'],
                       marker_color='rgb(158, 202, 225)'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(name='Recall', x=df_models['model'], y=df_models['recall'],
                       marker_color='rgb(107, 174, 214)'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(name='F1-Score', x=df_models['model'], y=df_models['f1'],
                       marker_color='rgb(49, 130, 189)'),
                row=2, col=1
            )
            
            # Plot 4: CV Mean with Error Bars
            fig.add_trace(
                go.Bar(name='CV Accuracy', x=df_models['model'], y=df_models['cv_mean'],
                       error_y=dict(type='data', array=df_models['cv_std'], visible=True),
                       marker_color='rgb(204, 204, 204)', text=df_models['cv_mean'].round(1),
                       textposition='outside', texttemplate='%{text}%'),
                row=2, col=2
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Model", row=1, col=1, tickangle=-45)
            fig.update_xaxes(title_text="Test Accuracy (%)", row=1, col=2)
            fig.update_xaxes(title_text="Model", row=2, col=1, tickangle=-45)
            fig.update_xaxes(title_text="Model", row=2, col=2, tickangle=-45)
            
            fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
            fig.update_yaxes(title_text="Overfitting Gap (%)", row=1, col=2)
            fig.update_yaxes(title_text="Score (%)", row=2, col=1)
            fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2)
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="<b>Complete Model Performance Analysis</b>",
                title_font_size=20,
                hovermode='closest',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation guide
            with st.expander("📖 How to Read This Dashboard"):
                st.markdown("""
                **Top Left - Accuracy Comparison:**
                - Blue bars = Training accuracy
                - Dark blue bars = Test accuracy (TRUE PERFORMANCE)
                - Smaller gap between bars = better generalization
                
                **Top Right - Overfitting Analysis:**
                - 🟢 Green dots (gap <5%) = Excellent generalization
                - 🟡 Orange dots (gap 5-10%) = Moderate overfitting
                - 🔴 Red dots (gap >10%) = High overfitting risk
                - Models in top-right = High accuracy with low overfitting (BEST)
                
                **Bottom Left - Performance Metrics:**
                - Precision: Accuracy of positive predictions
                - Recall: Coverage of actual positives
                - F1-Score: Balance between precision and recall
                
                **Bottom Right - Cross-Validation Stability:**
                - Error bars show performance variation across folds
                - Smaller error bars = more stable/reliable model
                """)
        
        except Exception as e:
            logger.error(f"Failed to create interactive visualization: {e}")
            st.warning("Interactive visualization unavailable. Showing table below.")
        
        st.write("---")
        
        # NEW: Display ONLY HIGH Severity Warnings (Critical)
        high_severity_warnings = []
        medium_low_warnings = []
        
        for name, result in st.session_state.results.items():
            if 'overfitting_warnings' in result:
                warnings = result['overfitting_warnings']
                if warnings.get('has_issues'):
                    if warnings.get('overall_severity') == 'HIGH':
                        high_severity_warnings.append((name, warnings))
                    elif warnings.get('overall_severity') in ['MEDIUM', 'LOW']:
                        medium_low_warnings.append((name, warnings))
        
        # CRITICAL WARNINGS (only HIGH severity)
        if high_severity_warnings:
            st.error("🚨 **CRITICAL: Overfitting/Data Leakage Detected!**")
            st.markdown("""
            **Your model results may be unrealistic due to:**
            - Training-test data leakage
            - Overfitting (memorizing instead of learning)
            - Data quality issues
            
            **⚠️ DO NOT deploy these models without addressing issues below!**
            """)
            
            for model_name, warnings in high_severity_warnings:
                with st.expander(f"🚨 Issues with {model_name}", expanded=True):
                    st.markdown(warnings['summary'])
                    
                    for warning in warnings['warnings']:
                        if warning['severity'] == 'HIGH':
                            st.markdown(f"**{warning['message']}**")
                            st.markdown("**What to do:**")
                            for rec in warning['recommendations']:
                                st.markdown(f"- {rec}")
        
        # INFORMATIONAL WARNINGS (MEDIUM/LOW severity)
        if medium_low_warnings:
            with st.expander("ℹ️ Additional Performance Notes (Non-Critical)", expanded=False):
                st.info("These are informational observations that may help improve your models:")
                
                for model_name, warnings in medium_low_warnings:
                    st.markdown(f"**{model_name}**: {warnings['summary']}")
                    
                    for warning in warnings['warnings']:
                        if warning['severity'] in ['MEDIUM', 'LOW']:
                            st.markdown(f"- {warning['message']}")
                            if warning['severity'] == 'MEDIUM':
                                st.markdown("  **Suggestions:**")
                                for rec in warning['recommendations'][:2]:  # Show top 2 recommendations
                                    st.markdown(f"    - {rec}")
                    st.write("---")
        
        # NEW: Train vs Test Performance Table
        st.markdown("### 📊 Model Performance (Training vs Testing)")
        
        leaderboard_data = []
        for name, result in st.session_state.results.items():
            train_acc = result.get('train_accuracy', 0)
            test_acc = result.get('test_accuracy', 0)
            gap = result.get('overfitting_gap', train_acc - test_acc)
            
            # Color code the gap
            gap_emoji = "🟢" if gap < 0.05 else "🟡" if gap < 0.10 else "🔴"
            
            leaderboard_data.append({
                'Model': name,
                'Train Acc': f"{train_acc:.4f}",
                'Test Acc': f"{test_acc:.4f}",
                'Gap': f"{gap_emoji} {gap:.4f}",
                'CV Mean': f"{result.get('cv_accuracy_mean', 0):.4f}",
                'CV Std': f"{result.get('cv_accuracy_std', 0):.4f}",
                'Status': "✅ Good" if gap < 0.10 else "⚠️ Overfit"
            })
        
        df_leaderboard = pd.DataFrame(leaderboard_data)
        # Sort by Test Acc (the TRUE performance)
        df_leaderboard = df_leaderboard.sort_values('Test Acc', ascending=False)
        st.dataframe(df_leaderboard, use_container_width=True)
        
        # NEW: Display CV Strategy Report
        if leaderboard_data and 'cv_strategy' in st.session_state.results[leaderboard_data[0]['Model']]:
            first_result = st.session_state.results[leaderboard_data[0]['Model']]
            cv_info = f"""
            **📊 Cross-Validation Strategy Report:**
            - **Strategy**: {first_result.get('cv_strategy', 'Standard CV')}
            - **Folds**: {first_result.get('cv_folds', 'N/A')}
            - **Training Samples**: {first_result.get('cv_sample_size', len(st.session_state.X_train)):,}
            - **Purpose**: Ensures model reliability and detects overfitting
            """
            st.info(cv_info)
        
        st.info("""
        **How to Read This Table:**
        - **Train Acc**: Performance on training data
        - **Test Acc**: Performance on unseen data (**THIS IS YOUR TRUE SCORE**)
        - **Gap**: Train - Test (🟢 <5% = Good, 🟡 5-10% = Watch, 🔴 >10% = Overfit)
        - **CV Mean/Std**: Cross-validation reliability
        
        ⚠️ **Always report Test Acc, never Train Acc!**
        """)
        
        # AI-Powered Results Interpretation (Modified to include overfitting context)
        if st.session_state.ai_engine and st.session_state.ai_engine is not False:
            with st.expander("🤖 AI Performance Analysis", expanded=False):
                with st.spinner("🤖 AI is interpreting your results..."):
                    try:
                        # Get best model performance
                        evaluator = st.session_state.evaluator
                        leaderboard = evaluator.get_leaderboard('accuracy')
                        
                        if leaderboard and len(leaderboard) > 0:
                            best_model = leaderboard[0]
                            best_name = best_model.get('model', 'Unknown')
                            
                            # Try to get accuracy from different possible keys
                            best_accuracy = (
                                best_model.get('accuracy') or 
                                best_model.get('score') or
                                st.session_state.results.get(best_name, {}).get('accuracy_mean', 0)
                            )
                            
                            # Create performance context
                            n_classes = len(np.unique(st.session_state.y_processed))
                            n_samples = len(st.session_state.y_processed)
                            
                            # Build model list with error handling
                            model_list = []
                            for m in leaderboard[:5]:
                                model_name = m.get('model', 'Unknown')
                                acc = (
                                    m.get('accuracy') or 
                                    m.get('score') or
                                    st.session_state.results.get(model_name, {}).get('accuracy_mean', 0)
                                )
                                model_list.append(f"- {model_name}: {acc:.4f} accuracy")
                            
                            # Build performance prompt
                            perf_prompt = f"""You are an expert ML engineer analyzing model performance.

**Dataset Context:**
- Task: Multi-class Classification
- Classes: {n_classes}
- Samples: {n_samples}
- Best Model: {best_name}
- Best Accuracy: {best_accuracy:.4f}

**All Model Performance:**
{chr(10).join(model_list)}

Provide a brief analysis in JSON format:
1. "performance_assessment": Is this good/excellent/poor performance? Why?
2. "model_comparison": Why did {best_name} perform best?
3. "improvement_tips": 2-3 specific suggestions to improve results
4. "red_flags": Any concerning patterns in the results?

Be specific and actionable."""
                            
                            # Get AI interpretation with fallback handling
                            try:
                                response = st.session_state.ai_engine._call_llm(perf_prompt)
                                insights = st.session_state.ai_engine._parse_response(response)
                                
                                # Show rate limit notice if present
                                if '_notice' in insights:
                                    st.info(insights['_notice'])
                                
                            except Exception as e:
                                # If AI fails, show user-friendly message
                                error_msg = str(e)
                                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                                    st.warning("⚠️ **AI Analysis Unavailable**: Groq API rate limit reached (100K tokens/day used). Try again later or upgrade your plan at https://console.groq.com/settings/billing")
                                    st.info("💡 **Tip**: Enable response caching to reduce API calls, or try again tomorrow when your quota resets.")
                                else:
                                    st.error(f"AI analysis failed: {error_msg}")
                                
                                # Stop here if AI completely failed
                                insights = {}
                            
                            if 'performance_assessment' in insights:
                                st.info(f"**📊 Performance Assessment:** {insights['performance_assessment']}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if 'model_comparison' in insights:
                                    st.success(f"**🏆 Why {best_name} Won:** {insights['model_comparison']}")
                            
                            with col2:
                                if 'red_flags' in insights:
                                    st.warning(f"**⚠ Red Flags:** {insights['red_flags']}")
                            
                            if 'improvement_tips' in insights:
                                st.info("**→ Improvement Tips:**")
                                if isinstance(insights['improvement_tips'], list):
                                    for tip in insights['improvement_tips']:
                                        st.markdown(f"- {tip}")
                                else:
                                    st.markdown(insights['improvement_tips'])
                        else:
                            st.warning("No model results available for AI analysis")
                    
                    except Exception as e:
                        logger.warning(f"Failed to generate AI performance insights: {e}")
                        st.error(f"AI analysis failed: {str(e)}")
        
        # Leaderboard (OLD visualizer - keep for backward compat)
        st.subheader("Model Leaderboard")
        
        if not st.session_state.results:
            st.warning("No models successfully completed training. Please check the training logs.")
            return
        
        evaluator = st.session_state.evaluator
        leaderboard = evaluator.get_leaderboard('accuracy')
        
        if not leaderboard:
            st.warning("No leaderboard data available.")
            return
        
        visualizer = Visualizer()
        fig = visualizer.plot_leaderboard(leaderboard, 'Accuracy')
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("Detailed Metrics")
        metrics_data = []
        for item in leaderboard:
            model_name = item['model']
            results = st.session_state.results[model_name]
            metrics_data.append({
                'Model': model_name,
                'Accuracy': f"{results.get('accuracy_mean', 0):.4f}",
                'F1-Score': f"{results.get('f1_macro_mean', 0):.4f}",
                'ROC-AUC': f"{results.get('roc_auc_ovr_mean', 0):.4f}",
                'Log Loss': f"{results.get('log_loss_mean', 0):.4f}"
            })
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        
        # ROC Curves
        if len(np.unique(st.session_state.y_processed)) <= 10:  # Only for reasonable number of classes
            st.subheader("ROC Curves")
            try:
                models_data = st.session_state.results
                fig = visualizer.plot_roc_curves(
                    models_data,
                    st.session_state.X_processed,
                    st.session_state.y_processed,
                    len(np.unique(st.session_state.y_processed))
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate ROC curves: {e}")
        
        # Confusion Matrix for best model
        if leaderboard:
            st.subheader("Confusion Matrix (Best Model)")
            best_model_name = leaderboard[0]['model']
            best_results = st.session_state.results[best_model_name]
            
            if 'predictions' in best_results and 'true_labels' in best_results:
                fig = visualizer.plot_confusion_matrix(
                    best_results['true_labels'],
                    best_results['predictions']
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No models successfully completed training. Please check the training logs.")
    
    def render_clustering_results(self):
        """Render clustering results."""
        st.subheader("🤖 Clustering Models")
        
        if not st.session_state.results:
            st.info("Run AutoML to see results")
            return
        
        # AI-Powered Clustering Analysis (at the top)
        if st.session_state.ai_engine and st.session_state.ai_engine is not False:
            with st.expander("🤖 AI Clustering Analysis", expanded=True):
                with st.spinner("🤖 AI is analyzing clustering results..."):
                    try:
                        evaluator = st.session_state.evaluator
                        leaderboard = evaluator.get_leaderboard('silhouette')
                        
                        # Get best model info
                        best_model = leaderboard[0]
                        best_name = best_model['model']
                        best_results = st.session_state.results[best_name]
                        
                        # Build metrics summary
                        metrics_str = "\n".join([
                            f"- {m['model']}: Silhouette={st.session_state.results[m['model']].get('silhouette', 0):.4f}, "
                            f"Davies-Bouldin={st.session_state.results[m['model']].get('davies_bouldin', 0):.4f}, "
                            f"Clusters={st.session_state.results[m['model']].get('n_clusters', 0)}"
                            for m in leaderboard
                        ])
                        
                        prompt = f"""You are an expert in unsupervised learning analyzing clustering results.

**Best Model:** {best_name}
**Silhouette Score:** {best_results.get('silhouette', 0):.4f} (range: -1 to 1, higher is better)
**Davies-Bouldin Score:** {best_results.get('davies_bouldin', 0):.4f} (lower is better)
**Calinski-Harabasz Score:** {best_results.get('calinski_harabasz', 0):.2f} (higher is better)
**Number of Clusters:** {best_results.get('n_clusters', 0)}

**All Models Tested:**
{metrics_str}

**Dataset:** {st.session_state.data.shape[0]} samples, {st.session_state.data.shape[1]} features

Provide analysis in JSON format:
1. "cluster_quality_assessment": Overall quality of clustering results (be honest about limitations)
2. "best_model_rationale": Why this model performed best and what it means
3. "cluster_interpretation": What these clusters likely represent (be general, don't assume domain)
4. "improvement_suggestions": 2-3 specific ways to improve clustering

Be specific about the metrics and realistic about clustering quality."""
                        
                        response = st.session_state.ai_engine._call_llm(prompt)
                        insights = st.session_state.ai_engine._parse_response(response)
                        
                        if 'cluster_quality_assessment' in insights:
                            st.info(f"**📊 Cluster Quality:** {insights['cluster_quality_assessment']}")
                        
                        if 'best_model_rationale' in insights:
                            st.success(f"**🏆 Best Model Analysis:** {insights['best_model_rationale']}")
                        
                        if 'cluster_interpretation' in insights:
                            st.info(f"**🔍 Cluster Interpretation:** {insights['cluster_interpretation']}")
                        
                        if 'improvement_suggestions' in insights:
                            st.warning("**→ Improvement Suggestions:**")
                            if isinstance(insights['improvement_suggestions'], list):
                                for suggestion in insights['improvement_suggestions']:
                                    st.markdown(f"- {suggestion}")
                            else:
                                st.markdown(insights['improvement_suggestions'])
                    
                    except Exception as e:
                        logger.warning(f"Failed to generate AI clustering insights: {e}")
                        st.error(f"AI analysis failed: {e}")
        
        # Leaderboard
        st.subheader("Model Leaderboard (by Silhouette Score)")
        evaluator = st.session_state.evaluator
        leaderboard = evaluator.get_leaderboard('silhouette')
        
        # Display table
        metrics_data = []
        for item in leaderboard:
            model_name = item['model']
            results = st.session_state.results[model_name]
            metrics_data.append({
                'Model': model_name,
                'Silhouette': f"{results.get('silhouette', 0):.4f}",
                'Davies-Bouldin': f"{results.get('davies_bouldin', 0):.4f}",
                'Calinski-Harabasz': f"{results.get('calinski_harabasz', 0):.1f}",
                'N Clusters': results.get('n_clusters', 0),
                'Stability': f"{results.get('stability', 0):.4f}"
            })
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        
        # UMAP projection
        st.subheader("UMAP Cluster Visualization")
        try:
            best_model_name = leaderboard[0]['model']
            best_labels = st.session_state.results[best_model_name]['labels']
            
            # Compute UMAP
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_umap = reducer.fit_transform(st.session_state.X_processed)
            
            visualizer = Visualizer()
            fig = visualizer.plot_umap_projection(X_umap, best_labels, f"UMAP - {best_model_name}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate UMAP: {e}")
        
        # Elbow and Silhouette curves for KMeans
        if 'KMeans' in st.session_state.results:
            kmeans_model = st.session_state.models['KMeans']
            if hasattr(kmeans_model, 'inertias') and kmeans_model.inertias:
                col1, col2 = st.columns(2)
                
                visualizer = Visualizer()
                with col1:
                    st.subheader("Elbow Curve")
                    fig = visualizer.plot_elbow_curve(
                        range(kmeans_model.k_range[0], kmeans_model.k_range[1] + 1),
                        kmeans_model.inertias
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Silhouette Scores")
                    fig = visualizer.plot_silhouette_scores(
                        range(kmeans_model.k_range[0], kmeans_model.k_range[1] + 1),
                        kmeans_model.silhouette_scores
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_explainability(self):
        """Render explainability tab with caching and optimization."""
        st.subheader("🔍 Model Explainability")
        
        if not st.session_state.results:
            st.info("Run AutoML to see explainability results")
            return
        
        # Check if clustering task
        is_clustering = st.session_state.task_type == "Clustering"
        
        # Model selection with unique key
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox(
            "Select Model", 
            model_names,
            key='explainability_model_selector',
            help="Switch models to see their specific explanations"
        )
        
        if selected_model:
            # Initialize cache for explanations if not exists
            if 'explainability_cache' not in st.session_state:
                st.session_state.explainability_cache = {}
            
            # Check cache first to avoid recomputation
            cache_key = f"{selected_model}_{is_clustering}"
            use_cached = cache_key in st.session_state.explainability_cache
            
            # Silent caching - no need to show message, just faster performance
            
            model = st.session_state.models[selected_model]
            X = st.session_state.X_processed
            feature_names = st.session_state.preprocessor.get_feature_names()
            
            # AI-Powered Clustering Explainability (for clustering tasks)
            if is_clustering and st.session_state.ai_engine and st.session_state.ai_engine is not False:
                with st.expander(f"🤖 AI Analysis for {selected_model}", expanded=True):
                    # Check AI cache
                    ai_cache_key = f"ai_explainability_{selected_model}"
                    
                    if ai_cache_key in st.session_state.explainability_cache:
                        insights = st.session_state.explainability_cache[ai_cache_key]
                    else:
                        with st.spinner(f"🤖 AI is analyzing {selected_model} cluster structure..."):
                            try:
                                results = st.session_state.results[selected_model]
                                n_clusters = results.get('n_clusters', 0)
                                silhouette = results.get('silhouette', 0)
                                davies_bouldin = results.get('davies_bouldin', 0)
                                
                                # Get cluster sizes
                                labels = results.get('labels', [])
                                if len(labels) > 0:
                                    unique, counts = np.unique(labels, return_counts=True)
                                    cluster_sizes = "\n".join([f"- Cluster {i}: {count} samples ({count/len(labels)*100:.1f}%)" 
                                                              for i, count in zip(unique, counts)])
                                else:
                                    cluster_sizes = "No cluster information available"
                                
                                prompt = f"""You are an expert in clustering analysis interpreting {selected_model} results.

**Model:** {selected_model}
**Number of Clusters:** {n_clusters}
**Silhouette Score:** {silhouette:.4f} (range: -1 to 1, higher is better)
**Davies-Bouldin Score:** {davies_bouldin:.4f} (lower is better)

**Cluster Distribution:**
{cluster_sizes}

**Dataset:** {X.shape[0]} samples, {X.shape[1]} features

Provide detailed analysis specific to {selected_model} in JSON format:
1. "cluster_quality": Assess quality for THIS SPECIFIC MODEL
2. "model_specific_insights": What's unique about {selected_model}'s clustering approach?
3. "balance_assessment": Are clusters well-balanced or is there imbalance?
4. "actionable_insights": 2-3 ways to use or improve THESE SPECIFIC clusters

Be model-specific in your analysis."""
                                
                                response = st.session_state.ai_engine._call_llm(prompt)
                                insights = st.session_state.ai_engine._parse_response(response)
                                
                                # Cache the AI insights
                                st.session_state.explainability_cache[ai_cache_key] = insights
                                
                            except Exception as e:
                                logger.warning(f"Failed to generate AI clustering explainability: {e}")
                                st.error(f"AI analysis failed: {str(e)}")
                                insights = {}
                    
                    # Display cached or fresh insights
                    if 'cluster_quality' in insights:
                        st.info(f"**📊 Cluster Quality:** {insights['cluster_quality']}")
                    
                    if 'model_specific_insights' in insights:
                        st.success(f"**� {selected_model} Insights:** {insights['model_specific_insights']}")
                    
                    if 'balance_assessment' in insights:
                        st.info(f"**⚖️ Balance Assessment:** {insights['balance_assessment']}")
                    
                    if 'actionable_insights' in insights:
                        st.warning("**→ Actionable Insights:**")
                        if isinstance(insights['actionable_insights'], list):
                            for insight in insights['actionable_insights']:
                                st.markdown(f"- {insight}")
                        else:
                            st.markdown(insights['actionable_insights'])
            
            # AI-Powered Feature Importance Interpretation (for classification tasks)
            elif not is_clustering and st.session_state.ai_engine and st.session_state.ai_engine is not False:
                with st.expander(f"🤖 AI Analysis for {selected_model}", expanded=True):
                    # Check AI cache for this specific model
                    ai_cache_key = f"ai_explainability_{selected_model}"
                    
                    if ai_cache_key in st.session_state.explainability_cache:
                        insights = st.session_state.explainability_cache[ai_cache_key]
                    else:
                        with st.spinner(f"🤖 AI is analyzing {selected_model} features..."):
                            try:
                                # Get model-specific metrics
                                model_results = st.session_state.results.get(selected_model, {})
                                test_acc = model_results.get('test_accuracy', 0)
                                train_acc = model_results.get('train_accuracy', 0)
                                gap = abs(train_acc - test_acc)
                                
                                # Get feature importance
                                if hasattr(model, 'feature_importances_'):
                                    importances = model.feature_importances_
                                elif hasattr(model, 'coef_'):
                                    coef = model.coef_
                                    importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
                                else:
                                    importances = None
                                
                                if importances is not None:
                                    # Get top features
                                    top_indices = np.argsort(importances)[-5:][::-1]
                                    top_features = [(feature_names[i], float(importances[i])) for i in top_indices]
                                    
                                    # Create model-specific AI prompt
                                    features_str = "\n".join([f"- {name}: {imp:.4f}" for name, imp in top_features])
                                    
                                    prompt = f"""You are an expert data scientist analyzing {selected_model} specifically.

**Model:** {selected_model}
**Performance:** Test Acc: {test_acc:.1%}, Train Acc: {train_acc:.1%}, Gap: {gap:.1%}

**Top 5 Most Important Features for {selected_model}:**
{features_str}

**Dataset Context:**
- Total Features: {len(feature_names)}

Provide model-specific analysis in JSON format:
1. "model_characteristics": How does {selected_model} handle these features? What's unique about its approach?
2. "performance_insights": Why did it achieve {test_acc:.1%} test accuracy? Relate to features.
3. "feature_advice": 2-3 recommendations to improve THIS MODEL based on these features

Be specific to {selected_model}'s algorithm."""
                                    
                                    response = st.session_state.ai_engine._call_llm(prompt)
                                    insights = st.session_state.ai_engine._parse_response(response)
                                    
                                    # Cache the AI insights
                                    st.session_state.explainability_cache[ai_cache_key] = insights
                                else:
                                    insights = {"error": "No feature importance available for this model"}
                            
                            except Exception as e:
                                logger.warning(f"Failed to generate AI feature insights: {e}")
                                insights = {}
                    
                    # Display cached or fresh insights
                    if 'model_characteristics' in insights:
                        st.success(f"**� {selected_model} Characteristics:** {insights['model_characteristics']}")
                    
                    if 'performance_insights' in insights:
                        st.info(f"**📊 Performance Analysis:** {insights['performance_insights']}")
                    
                    if 'feature_advice' in insights:
                        st.warning("**→ Model-Specific Advice:**")
                        if isinstance(insights['feature_advice'], list):
                            for advice in insights['feature_advice']:
                                st.markdown(f"- {advice}")
                        else:
                            st.markdown(insights['feature_advice'])
                    
                    if 'error' in insights:
                        st.info(insights['error'])
            
            # Check if dataset is too large for SHAP
            n_features = X.shape[1]
            explanations = {}
            explainer = None  # Initialize explainer
            
            if n_features > 1000:
                st.warning(f"⚠️ Dataset has {n_features:,} features. SHAP explanations may be slow or fail due to memory constraints.")
                st.info("💡 Tip: SHAP works best with < 1000 features. Consider feature selection for large datasets.")
                
                # Ask user if they want to proceed
                if not st.checkbox(f"Attempt SHAP anyway (may cause memory errors)", value=False, key=f"shap_checkbox_{selected_model}"):
                    st.info("Showing only native model explanations (feature importance, coefficients).")
                    # Skip SHAP, only show native explanations
                    explainer = ModelExplainer()
                    
                    # Get native feature importance
                    if hasattr(model, 'feature_importances_'):
                        explanations['feature_importance'] = {
                            name: float(val) for name, val in zip(feature_names, model.feature_importances_)
                        }
                    elif hasattr(model, 'coef_'):
                        coef = model.coef_
                        if coef.ndim > 1:
                            coef = np.abs(coef).mean(axis=0)
                        else:
                            coef = np.abs(coef)
                        explanations['coef_importance'] = {
                            name: float(val) for name, val in zip(feature_names, coef)
                        }
                else:
                    # User chose to attempt SHAP - check cache first
                    if use_cached and 'shap' in st.session_state.explainability_cache[cache_key]:
                        explanations = st.session_state.explainability_cache[cache_key]['shap']
                        # Don't show cache message - silent caching is better UX
                    else:
                        with st.spinner(f"Generating SHAP explanations for {selected_model} (this may take a while)..."):
                            try:
                                explainer = ModelExplainer()
                                explanations = explainer.explain_model(
                                    model, X, feature_names, sample_size=20  # Use minimal samples
                                )
                                # Cache SHAP results
                                if cache_key not in st.session_state.explainability_cache:
                                    st.session_state.explainability_cache[cache_key] = {}
                                st.session_state.explainability_cache[cache_key]['shap'] = explanations
                            except Exception as e:
                                st.error(f"SHAP failed: {e}")
                                explanations = {}
            else:
                # Normal size dataset - check cache first
                if use_cached and 'shap' in st.session_state.explainability_cache[cache_key]:
                    explanations = st.session_state.explainability_cache[cache_key]['shap']
                    # Don't show cache message - silent caching is better UX
                else:
                    with st.spinner(f"Generating explanations for {selected_model}..."):
                        try:
                            explainer = ModelExplainer()
                            explanations = explainer.explain_model(
                                model, X, feature_names, sample_size=50
                            )
                            # Cache SHAP results
                            if cache_key not in st.session_state.explainability_cache:
                                st.session_state.explainability_cache[cache_key] = {}
                            st.session_state.explainability_cache[cache_key]['shap'] = explanations
                        except Exception as e:
                            st.error(f"Error generating explanations: {e}")
                            explanations = {}
            
            # Display explanations (common for both paths)
            if explanations:
                # Check for SHAP errors
                if 'shap_error' in explanations:
                    st.warning(f"SHAP explanation failed: {explanations['shap_error']}")
                    if 'shap_traceback' in explanations:
                        with st.expander("Show error details"):
                            st.code(explanations['shap_traceback'])
                    st.info("Showing alternative explanation methods below...")
                
                # Feature importance
                if 'shap_importance' in explanations:
                    st.subheader("SHAP Feature Importance")
                    visualizer = Visualizer()
                    fig = visualizer.plot_feature_importance(
                        explanations['shap_importance'],
                        top_n=15,
                        title="Top 15 Features by SHAP Importance"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif 'feature_importance' in explanations:
                    st.subheader("Feature Importance")
                    visualizer = Visualizer()
                    fig = visualizer.plot_feature_importance(
                        explanations['feature_importance'],
                        top_n=15
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif 'coef_importance' in explanations:
                    st.subheader("Coefficient Importance")
                    visualizer = Visualizer()
                    fig = visualizer.plot_feature_importance(
                        explanations['coef_importance'],
                        top_n=15
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Top features list
                if explainer:
                    st.subheader("Top Important Features")
                    top_features = explainer.get_top_features(explanations, top_n=10)
                    if top_features is not None and len(top_features) > 0:
                        df_features = pd.DataFrame(
                            top_features,
                            columns=['Feature', 'Importance']
                        )
                        st.dataframe(df_features, use_container_width=True)
    
    def render_recommendation(self):
        """Render recommendation tab."""
        st.subheader("🎯 Model Recommendation")
        
        if st.session_state.task_type == "Classification":
            if 'recommendation' not in st.session_state:
                st.info("Run AutoML to see recommendations")
                return
            
            recommendation = st.session_state.recommendation
            
            # Check if recommendation has required fields
            if not recommendation or 'recommended_model' not in recommendation:
                st.warning("No recommendation available. Please run AutoML first.")
                return
            
            # AI-Powered Final Recommendation
            if st.session_state.ai_engine and st.session_state.ai_engine is not False:
                with st.expander("🤖 AI Final Recommendations", expanded=True):
                    with st.spinner("🤖 AI is creating your final recommendations..."):
                        try:
                            recommended_model = recommendation['recommended_model']
                            score = recommendation.get('score', 0)
                            
                            # Get all model scores with error handling
                            evaluator = st.session_state.evaluator
                            leaderboard = evaluator.get_leaderboard('accuracy')
                            
                            model_lines = []
                            for m in leaderboard[:5]:
                                model_name = m.get('model', 'Unknown')
                                acc = (
                                    m.get('accuracy') or 
                                    m.get('score') or
                                    st.session_state.results.get(model_name, {}).get('accuracy_mean', 0)
                                )
                                model_lines.append(f"- {model_name}: {acc:.4f}")
                            all_models = "\n".join(model_lines)
                            
                            prompt = f"""You are an ML deployment expert providing final recommendations.

**Recommended Model:** {recommended_model}
**Performance:** {score:.4f} accuracy

**All Models Tested:**
{all_models}

**Dataset:** {st.session_state.data.shape[0]} samples, {st.session_state.data.shape[1]} features

Provide comprehensive recommendations in JSON format:
1. "deployment_readiness": Is this model ready for production? Why/why not?
2. "next_steps": 3-4 specific next steps to improve or deploy
3. "monitoring_advice": What to monitor in production
4. "risk_assessment": Potential risks or limitations

Be specific and actionable."""
                            
                            response = st.session_state.ai_engine._call_llm(prompt)
                            insights = st.session_state.ai_engine._parse_response(response)
                            
                            if 'deployment_readiness' in insights:
                                st.success(f"**🚀 Deployment Readiness:** {insights['deployment_readiness']}")
                            
                            if 'risk_assessment' in insights:
                                st.warning(f"**⚠ Risk Assessment:** {insights['risk_assessment']}")
                            
                            if 'next_steps' in insights:
                                st.info("**→ Next Steps:**")
                                if isinstance(insights['next_steps'], list):
                                    for step in insights['next_steps']:
                                        st.markdown(f"- {step}")
                                else:
                                    st.markdown(insights['next_steps'])
                            
                            if 'monitoring_advice' in insights:
                                st.info(f"**📊 Monitoring Advice:** {insights['monitoring_advice']}")
                        
                        except Exception as e:
                            logger.warning(f"Failed to generate AI recommendations: {e}")
                            st.error(f"AI recommendations failed: {str(e)}")
            
            # Recommended model
            st.success(f"### Recommended Model: **{recommendation['recommended_model']}**")
            
            # Add explanation of selection logic
            evaluator = st.session_state.evaluator
            leaderboard = evaluator.get_leaderboard('accuracy', penalize_overfitting=True)
            winner = leaderboard[0]
            
            # Show why this model won
            if winner.get('overfitting_gap', 0) < 0.10:
                st.info(f"""
**Why {winner['model']} was selected:**
- ✅ Best **adjusted score** ({winner.get('adjusted_score', 0):.4f}) considering both accuracy and overfitting
- ✅ Test accuracy: {winner.get('test_accuracy', 0):.1%}
- ✅ Low overfitting gap: {winner.get('overfitting_gap', 0):.1%} (train-test difference)
- ✅ Good generalization expected on new data
""")
            else:
                st.warning(f"""
**Why {winner['model']} was selected:**
- Highest **adjusted score** ({winner.get('adjusted_score', 0):.4f}) after overfitting penalty
- Test accuracy: {winner.get('test_accuracy', 0):.1%}
- ⚠️ Overfitting detected: {winner.get('overfitting_gap', 0):.1%} gap (train-test)
- Consider retraining with regularization or try alternative models below
""")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score", f"{recommendation.get('score', 0):.4f}")
            with col2:
                st.metric("CI Lower", f"{recommendation.get('ci_lower', 0):.4f}")
            with col3:
                st.metric("CI Upper", f"{recommendation.get('ci_upper', 0):.4f}")
            
            # Rationale
            st.subheader("Why This Model?")
            for idx, reason in enumerate(recommendation['rationale'], 1):
                st.write(f"{idx}. {reason}")
            
            # Alternatives
            if recommendation.get('alternatives'):
                st.subheader("Alternative Models")
                alt_data = []
                for alt in recommendation['alternatives']:
                    alt_data.append({
                        'Model': alt['model'],
                        'Score': f"{alt['score']:.4f}"
                    })
                st.dataframe(pd.DataFrame(alt_data), use_container_width=True)
        else:
            # Clustering recommendation
            if not st.session_state.results:
                st.info("Run AutoML to see recommendations")
                return
            
            evaluator = st.session_state.evaluator
            leaderboard = evaluator.get_leaderboard('silhouette')
            
            if leaderboard:
                best = leaderboard[0]
                best_results = st.session_state.results[best['model']]
                
                # AI-Powered Clustering Recommendation
                if st.session_state.ai_engine and st.session_state.ai_engine is not False:
                    with st.expander("🤖 AI Clustering Recommendations", expanded=True):
                        with st.spinner("🤖 AI is creating deployment recommendations..."):
                            try:
                                # Get all clustering results
                                all_models = "\n".join([
                                    f"- {m['model']}: Silhouette={st.session_state.results[m['model']].get('silhouette', 0):.4f}, "
                                    f"Clusters={st.session_state.results[m['model']].get('n_clusters', 0)}"
                                    for m in leaderboard
                                ])
                                
                                prompt = f"""You are a clustering deployment expert providing final recommendations.

**Recommended Model:** {best['model']}
**Silhouette Score:** {best_results.get('silhouette', 0):.4f} (range: -1 to 1, higher is better)
**Davies-Bouldin Score:** {best_results.get('davies_bouldin', 0):.4f} (lower is better)
**Calinski-Harabasz Score:** {best_results.get('calinski_harabasz', 0):.2f} (higher is better)
**Number of Clusters:** {best_results.get('n_clusters', 0)}

**All Models Tested:**
{all_models}

**Dataset:** {st.session_state.data.shape[0]} samples, {st.session_state.data.shape[1]} features

Provide comprehensive deployment recommendations in JSON format:
1. "deployment_assessment": Is this clustering result production-ready? Be honest about quality
2. "use_cases": 2-3 specific ways these clusters could be used in practice
3. "validation_steps": How to validate these clusters before deployment
4. "monitoring_strategy": What to monitor when using these clusters in production
5. "limitations": Key limitations or risks with these clusters

Be specific and realistic about clustering quality and utility."""
                                
                                response = st.session_state.ai_engine._call_llm(prompt)
                                insights = st.session_state.ai_engine._parse_response(response)
                                
                                if 'deployment_assessment' in insights:
                                    st.success(f"**🚀 Deployment Assessment:** {insights['deployment_assessment']}")
                                
                                if 'use_cases' in insights:
                                    st.info("**💡 Potential Use Cases:**")
                                    if isinstance(insights['use_cases'], list):
                                        for case in insights['use_cases']:
                                            st.markdown(f"- {case}")
                                    else:
                                        st.markdown(insights['use_cases'])
                                
                                if 'validation_steps' in insights:
                                    st.info("**✓ Validation Steps:**")
                                    if isinstance(insights['validation_steps'], list):
                                        for step in insights['validation_steps']:
                                            st.markdown(f"- {step}")
                                    else:
                                        st.markdown(insights['validation_steps'])
                                
                                if 'monitoring_strategy' in insights:
                                    st.info(f"**📊 Monitoring Strategy:** {insights['monitoring_strategy']}")
                                
                                if 'limitations' in insights:
                                    st.warning(f"**⚠️ Limitations:** {insights['limitations']}")
                            
                            except Exception as e:
                                error_str = str(e)
                                logger.warning(f"Failed to generate AI clustering recommendations: {error_str}")
                                
                                # Check if it's a rate limit error
                                if "rate_limit" in error_str.lower() or "429" in error_str:
                                    st.warning("⚠️ **AI Analysis Unavailable**: Groq API rate limit reached (100K tokens/day used). Try again later or upgrade at https://console.groq.com/settings/billing")
                                    
                                    # Provide rule-based fallback insights
                                    st.info("**📊 Basic Clustering Assessment (Rule-Based):**")
                                    
                                    silhouette = best_results.get('silhouette', 0)
                                    n_clusters = best_results.get('n_clusters', 0)
                                    
                                    if silhouette > 0.5:
                                        st.success(f"✅ **Strong Clustering**: Silhouette score {silhouette:.3f} indicates well-separated, distinct clusters")
                                    elif silhouette > 0.25:
                                        st.info(f"ℹ️ **Moderate Clustering**: Silhouette score {silhouette:.3f} shows reasonable cluster structure")
                                    else:
                                        st.warning(f"⚠️ **Weak Clustering**: Silhouette score {silhouette:.3f} suggests overlapping or poorly defined clusters")
                                    
                                    st.write(f"**Recommended Actions:**")
                                    st.write(f"1. Visualize clusters in 2D/3D using PCA/t-SNE")
                                    st.write(f"2. Analyze cluster centroids and member characteristics")
                                    st.write(f"3. Try different numbers of clusters (current: {n_clusters})")
                                    st.write(f"4. Consider feature engineering or dimensionality reduction")
                                else:
                                    st.error(f"AI analysis failed: {error_str[:150]}")
                
                st.success(f"### Recommended Model: **{best['model']}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Silhouette Score", f"{best['score']:.4f}")
                with col2:
                    st.metric("Number of Clusters", best['n_clusters'])
                
                st.subheader("Why This Model?")
                st.write(f"1. Highest silhouette score among all methods")
                st.write(f"2. Identified {best['n_clusters']} distinct clusters")
                st.write(f"3. Best balance between cohesion and separation")
                
                # ENHANCED: Professional Clustering Visualizations
                st.markdown("---")
                st.subheader("📊 Clustering Visualizations")
                
                try:
                    import plotly.graph_objects as go
                    import plotly.express as px
                    from plotly.subplots import make_subplots
                    from sklearn.decomposition import PCA
                    from sklearn.manifold import TSNE
                    from sklearn.metrics import silhouette_samples
                    
                    # Get the best model and its labels
                    # Try multiple sources: models dict, results['model'], or results['labels']
                    best_model_name = best['model']
                    
                    # Get cluster labels from results (guaranteed to exist)
                    labels = st.session_state.results[best_model_name].get('labels')
                    
                    # Try to get model for potential re-prediction
                    best_model = None
                    if best_model_name in st.session_state.get('models', {}):
                        best_model = st.session_state.models[best_model_name]
                    elif 'model' in st.session_state.results[best_model_name]:
                        best_model = st.session_state.results[best_model_name]['model']
                    
                    X = st.session_state.X_processed
                    
                    # If labels not in results, try to get from model
                    if labels is None and best_model is not None:
                        if hasattr(best_model, 'labels_'):
                            labels = best_model.labels_
                        elif hasattr(best_model, 'predict'):
                            labels = best_model.predict(X)
                    
                    if labels is None:
                        st.warning("⚠️ Cluster labels not available for visualization")
                        labels = None
                    
                    if labels is not None:
                        # Tab layout for different visualizations
                        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                            "🎯 2D Projection", 
                            "📊 Cluster Distribution", 
                            "📏 Silhouette Analysis",
                            "🔍 Cluster Profiles"
                        ])
                        
                        with viz_tab1:
                            st.markdown("**PCA 2D Projection of Clusters**")
                            
                            # Perform PCA for 2D visualization
                            pca = PCA(n_components=2, random_state=42)
                            X_pca = pca.fit_transform(X)
                            
                            # Create DataFrame for plotting
                            plot_df = pd.DataFrame({
                                'PC1': X_pca[:, 0],
                                'PC2': X_pca[:, 1],
                                'Cluster': [f'Cluster {int(l)}' for l in labels]
                            })
                            
                            # Create interactive scatter plot
                            fig = px.scatter(
                                plot_df, 
                                x='PC1', 
                                y='PC2', 
                                color='Cluster',
                                title=f'{best["model"]} - {best["n_clusters"]} Clusters (PCA Projection)',
                                labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                                       'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'},
                                color_discrete_sequence=px.colors.qualitative.Set2,
                                height=500
                            )
                            
                            fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white')))
                            fig.update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(size=12),
                                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.info(f"""
                            **How to Read This:**
                            - Each point is a data sample projected onto 2D space
                            - Colors represent different clusters
                            - Good clustering shows distinct, separated groups
                            - PCA captures {pca.explained_variance_ratio_.sum():.1%} of total variance
                            """)
                        
                        with viz_tab2:
                            st.markdown("**Cluster Size Distribution**")
                            
                            # Cluster sizes
                            unique_labels, counts = np.unique(labels, return_counts=True)
                            cluster_df = pd.DataFrame({
                                'Cluster': [f'Cluster {int(l)}' for l in unique_labels],
                                'Size': counts,
                                'Percentage': counts / len(labels) * 100
                            })
                            
                            # Create bar chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=cluster_df['Cluster'],
                                y=cluster_df['Size'],
                                text=[f"{s} ({p:.1f}%)" for s, p in zip(cluster_df['Size'], cluster_df['Percentage'])],
                                textposition='outside',
                                marker=dict(
                                    color=cluster_df['Size'],
                                    colorscale='Viridis',
                                    showscale=False
                                )
                            ))
                            
                            fig.update_layout(
                                title=f"Cluster Size Distribution ({len(labels):,} total samples)",
                                xaxis_title="Cluster",
                                yaxis_title="Number of Samples",
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                height=400,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show table
                            st.dataframe(cluster_df, use_container_width=True)
                            
                            # Balance assessment
                            max_size = cluster_df['Size'].max()
                            min_size = cluster_df['Size'].min()
                            imbalance_ratio = max_size / min_size if min_size > 0 else float('inf')
                            
                            if imbalance_ratio < 2:
                                st.success(f"✅ **Well-Balanced Clusters**: Largest/smallest ratio is {imbalance_ratio:.1f}x")
                            elif imbalance_ratio < 5:
                                st.info(f"ℹ️ **Moderate Imbalance**: Largest/smallest ratio is {imbalance_ratio:.1f}x")
                            else:
                                st.warning(f"⚠️ **Imbalanced Clusters**: Largest/smallest ratio is {imbalance_ratio:.1f}x - consider different number of clusters")
                        
                        with viz_tab3:
                            st.markdown("**Silhouette Analysis per Cluster**")
                            
                            # Calculate silhouette scores per sample
                            silhouette_vals = silhouette_samples(X, labels)
                            
                            # Create silhouette plot data
                            y_lower = 10
                            silhouette_data = []
                            
                            for i in unique_labels:
                                cluster_silhouette_vals = silhouette_vals[labels == i]
                                cluster_silhouette_vals.sort()
                                
                                size_cluster_i = cluster_silhouette_vals.shape[0]
                                y_upper = y_lower + size_cluster_i
                                
                                silhouette_data.append({
                                    'cluster': int(i),
                                    'y_lower': y_lower,
                                    'y_upper': y_upper,
                                    'values': cluster_silhouette_vals,
                                    'avg': cluster_silhouette_vals.mean()
                                })
                                
                                y_lower = y_upper + 10
                            
                            # Create silhouette plot
                            fig = go.Figure()
                            
                            colors = px.colors.qualitative.Set2
                            for idx, cluster_data in enumerate(silhouette_data):
                                y_vals = np.arange(cluster_data['y_lower'], cluster_data['y_upper'])
                                
                                fig.add_trace(go.Scatter(
                                    x=cluster_data['values'],
                                    y=y_vals,
                                    mode='lines',
                                    fill='tozerox',
                                    name=f"Cluster {cluster_data['cluster']}",
                                    line=dict(color=colors[idx % len(colors)], width=0.5),
                                    fillcolor=colors[idx % len(colors)],
                                    hovertemplate=f"Cluster {cluster_data['cluster']}<br>Silhouette: %{{x:.3f}}<extra></extra>"
                                ))
                            
                            # Add average silhouette score line
                            avg_silhouette = silhouette_vals.mean()
                            fig.add_vline(
                                x=avg_silhouette,
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"Avg: {avg_silhouette:.3f}",
                                annotation_position="top"
                            )
                            
                            fig.update_layout(
                                title="Silhouette Plot for Each Cluster",
                                xaxis_title="Silhouette Coefficient",
                                yaxis_title="Cluster",
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                height=500,
                                showlegend=True,
                                yaxis=dict(showticklabels=False)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Silhouette interpretation
                            st.info(f"""
                            **How to Read This:**
                            - Each colored section represents one cluster
                            - Width shows how many samples are in the cluster
                            - X-axis shows silhouette coefficient (how well-separated the cluster is)
                            - Red dashed line is the average silhouette score ({avg_silhouette:.3f})
                            - Good clusters: Most values > average and close to 1.0
                            - Poor clusters: Many values below average or negative
                            """)
                            
                            # Per-cluster metrics
                            cluster_metrics = pd.DataFrame([
                                {
                                    'Cluster': f"Cluster {d['cluster']}",
                                    'Size': len(d['values']),
                                    'Avg Silhouette': f"{d['avg']:.3f}",
                                    'Quality': '✅ Excellent' if d['avg'] > 0.5 else ('🟢 Good' if d['avg'] > 0.25 else '⚠️ Weak')
                                }
                                for d in silhouette_data
                            ])
                            
                            st.dataframe(cluster_metrics, use_container_width=True)
                        
                        with viz_tab4:
                            st.markdown("**Cluster Characteristics Profile**")
                            
                            # Get original data with cluster labels
                            df_with_clusters = st.session_state.data.copy()
                            df_with_clusters['Cluster'] = [f'Cluster {int(l)}' for l in labels]
                            
                            # Calculate cluster statistics for numeric columns
                            numeric_cols = df_with_clusters.select_dtypes(include=[np.number]).columns.tolist()
                            if 'Cluster' in numeric_cols:
                                numeric_cols.remove('Cluster')
                            
                            if numeric_cols:
                                # Show top 5 most important features
                                feature_cols = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
                                
                                st.write(f"**Cluster Centroids** (showing top {len(feature_cols)} features):")
                                
                                # Calculate means per cluster
                                cluster_profiles = []
                                for cluster_name in sorted(df_with_clusters['Cluster'].unique()):
                                    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_name]
                                    profile = {'Cluster': cluster_name, 'Size': len(cluster_data)}
                                    
                                    for col in feature_cols:
                                        if col in cluster_data.columns:
                                            profile[col] = cluster_data[col].mean()
                                    
                                    cluster_profiles.append(profile)
                                
                                profile_df = pd.DataFrame(cluster_profiles)
                                
                                # Display as styled dataframe
                                st.dataframe(
                                    profile_df.style.background_gradient(cmap='RdYlGn', subset=feature_cols),
                                    use_container_width=True
                                )
                                
                                # Radar chart for cluster comparison (if 3-6 features)
                                if 3 <= len(feature_cols) <= 6 and len(cluster_profiles) <= 5:
                                    st.write("**Cluster Comparison (Normalized Features):**")
                                    
                                    # Normalize features for radar chart
                                    from sklearn.preprocessing import MinMaxScaler
                                    scaler = MinMaxScaler()
                                    normalized_values = scaler.fit_transform(profile_df[feature_cols])
                                    
                                    fig = go.Figure()
                                    
                                    for idx, cluster in enumerate(cluster_profiles):
                                        fig.add_trace(go.Scatterpolar(
                                            r=normalized_values[idx],
                                            theta=feature_cols,
                                            fill='toself',
                                            name=cluster['Cluster']
                                        ))
                                    
                                    fig.update_layout(
                                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                        showlegend=True,
                                        title="Cluster Feature Profiles (Normalized)",
                                        height=500
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                st.success("""
                                **💡 Use These Insights To:**
                                - Understand what makes each cluster unique
                                - Name/label clusters based on their characteristics
                                - Identify which features drive cluster separation
                                - Make business decisions based on cluster profiles
                                """)
                            else:
                                st.info("No numeric features available for cluster profiling")
                    
                    else:
                        st.warning("Unable to generate visualizations - cluster labels not available")
                
                except Exception as e:
                    logger.error(f"Failed to create clustering visualizations: {e}")
                    st.warning(f"Visualization error: {str(e)[:200]}")
    
    def render_report(self):
        """Render report tab."""
        st.subheader("📄 Generate Report")
        
        if not st.session_state.results:
            st.info("Run AutoML to generate a report")
            return
        
        # AI-Generated Comprehensive Report
        if st.session_state.ai_engine and st.session_state.ai_engine is not False:
            with st.expander("🤖 AI-Generated Comprehensive Report", expanded=True):
                with st.spinner("🤖 AI is writing your comprehensive report..."):
                    try:
                        # Collect all relevant information
                        is_clustering = st.session_state.task_type == "Clustering"
                        evaluator = st.session_state.evaluator
                        
                        if is_clustering:
                            leaderboard = evaluator.get_leaderboard('silhouette')
                            best_model = leaderboard[0]
                            best_name = best_model['model']
                            best_results = st.session_state.results[best_name]
                            
                            metrics_summary = f"""**Best Model:** {best_name}
**Silhouette Score:** {best_results.get('silhouette', 0):.4f}
**Davies-Bouldin Score:** {best_results.get('davies_bouldin', 0):.4f}
**Number of Clusters:** {best_results.get('n_clusters', 0)}"""
                            
                            all_models = "\n".join([
                                f"- {m['model']}: Silhouette={st.session_state.results[m['model']].get('silhouette', 0):.4f}, "
                                f"Clusters={st.session_state.results[m['model']].get('n_clusters', 0)}"
                                for m in leaderboard
                            ])
                        else:
                            leaderboard = evaluator.get_leaderboard('accuracy')
                            best_model = leaderboard[0]
                            best_name = best_model.get('model', 'Unknown')
                            best_results = st.session_state.results.get(best_name, {})
                            
                            # Try to get metrics from different possible keys
                            best_accuracy = (
                                best_model.get('accuracy') or 
                                best_model.get('score') or
                                best_results.get('accuracy_mean', 0)
                            )
                            best_precision = (
                                best_model.get('precision') or
                                best_results.get('precision_macro_mean', 0)
                            )
                            best_recall = (
                                best_model.get('recall') or
                                best_results.get('recall_macro_mean', 0)
                            )
                            
                            metrics_summary = f"""**Best Model:** {best_name}
**Accuracy:** {best_accuracy:.4f}
**Precision:** {best_precision:.4f}
**Recall:** {best_recall:.4f}"""
                            
                            # Build model list with error handling
                            model_lines = []
                            for m in leaderboard[:5]:
                                model_name = m.get('model', 'Unknown')
                                acc = (
                                    m.get('accuracy') or 
                                    m.get('score') or
                                    st.session_state.results.get(model_name, {}).get('accuracy_mean', 0)
                                )
                                model_lines.append(f"- {model_name}: Accuracy={acc:.4f}")
                            all_models = "\n".join(model_lines)
                        
                        # Data summary
                        data_summary = f"""**Dataset Size:** {st.session_state.data.shape[0]} samples, {st.session_state.data.shape[1]} features
**Task Type:** {st.session_state.task_type}
**Preprocessing Applied:** Scaling, encoding, handling missing values"""
                        
                        prompt = f"""You are an expert data scientist writing a comprehensive AutoML report.

**PROJECT OVERVIEW:**
{data_summary}

**METHODOLOGY:**
Tested {len(st.session_state.results)} different {'clustering' if is_clustering else 'classification'} algorithms with automated hyperparameter tuning.

**RESULTS SUMMARY:**
{metrics_summary}

**ALL MODELS TESTED:**
{all_models}

Write a comprehensive report in JSON format with these sections:
1. "executive_summary": 2-3 paragraph executive summary highlighting key findings and business value
2. "methodology": Explain the AutoML approach and algorithms tested (be specific)
3. "key_findings": 4-5 bullet points with the most important discoveries
4. "best_model_analysis": Detailed analysis of why the best model performed well
5. "recommendations": 3-4 specific actionable recommendations for next steps
6. "limitations": 2-3 honest limitations or caveats about these results
7. "conclusion": Final thoughts and business impact

Write professionally but accessibly. Be specific with numbers and metrics. Make it suitable for both technical and non-technical stakeholders."""
                        
                        response = st.session_state.ai_engine._call_llm(prompt)
                        report_insights = st.session_state.ai_engine._parse_response(response)
                        
                        # Display the report
                        if 'executive_summary' in report_insights:
                            st.markdown("### 📋 Executive Summary")
                            st.write(report_insights['executive_summary'])
                        
                        if 'methodology' in report_insights:
                            st.markdown("### 🔬 Methodology")
                            st.write(report_insights['methodology'])
                        
                        if 'key_findings' in report_insights:
                            st.markdown("### 🔑 Key Findings")
                            if isinstance(report_insights['key_findings'], list):
                                for finding in report_insights['key_findings']:
                                    st.markdown(f"- {finding}")
                            else:
                                st.write(report_insights['key_findings'])
                        
                        if 'best_model_analysis' in report_insights:
                            st.markdown("### 🏆 Best Model Analysis")
                            st.write(report_insights['best_model_analysis'])
                        
                        if 'recommendations' in report_insights:
                            st.markdown("### 💡 Recommendations")
                            if isinstance(report_insights['recommendations'], list):
                                for rec in report_insights['recommendations']:
                                    st.markdown(f"- {rec}")
                            else:
                                st.write(report_insights['recommendations'])
                        
                        if 'limitations' in report_insights:
                            st.markdown("### ⚠️ Limitations")
                            if isinstance(report_insights['limitations'], list):
                                for lim in report_insights['limitations']:
                                    st.markdown(f"- {lim}")
                            else:
                                st.write(report_insights['limitations'])
                        
                        if 'conclusion' in report_insights:
                            st.markdown("### 🎯 Conclusion")
                            st.write(report_insights['conclusion'])
                    
                    except Exception as e:
                        error_msg = str(e)
                        logger.warning(f"Failed to generate AI report: {error_msg}")
                        
                        # Check if it's a rate limit error
                        if "rate_limit" in error_msg.lower() or "429" in error_msg:
                            st.warning("⚠️ **AI Report Generation Unavailable**: API rate limit reached. Showing structured report instead.")
                            
                            # FALLBACK: Generate structured report from data
                            st.markdown("### 📋 Executive Summary")
                            
                            is_clustering = st.session_state.task_type == "Clustering"
                            evaluator = st.session_state.evaluator
                            
                            if is_clustering:
                                leaderboard = evaluator.get_leaderboard('silhouette')
                                best_model = leaderboard[0]
                                best_name = best_model['model']
                                best_results = st.session_state.results[best_name]
                                
                                st.write(f"""
This AutoML analysis tested **{len(st.session_state.results)} clustering algorithms** on a dataset of 
**{st.session_state.data.shape[0]:,} samples** with **{st.session_state.data.shape[1]} features**.

**Key Result**: {best_name} achieved the best performance with a silhouette score of **{best_results.get('silhouette', 0):.4f}**, 
identifying **{best_results.get('n_clusters', 0)} optimal clusters**.
""")
                                
                                st.markdown("### 🔑 Key Findings")
                                st.markdown(f"""
- **Best Algorithm**: {best_name} performed best with silhouette score {best_results.get('silhouette', 0):.4f}
- **Cluster Count**: {best_results.get('n_clusters', 0)} clusters identified
- **Model Comparison**: Tested {len(leaderboard)} algorithms including {', '.join([m['model'] for m in leaderboard[:3]])}
- **Data Quality**: Successfully processed {st.session_state.data.shape[0]:,} samples after preprocessing
""")
                            else:
                                leaderboard = evaluator.get_leaderboard('accuracy')
                                best_model = leaderboard[0]
                                best_name = best_model.get('model', 'Unknown')
                                test_acc = best_model.get('test_accuracy', best_model.get('score', 0))
                                train_acc = best_model.get('train_accuracy', test_acc)
                                gap = best_model.get('overfitting_gap', abs(train_acc - test_acc))
                                
                                st.write(f"""
This AutoML analysis tested **{len(st.session_state.results)} classification algorithms** on a dataset of 
**{st.session_state.data.shape[0]:,} samples** with **{st.session_state.data.shape[1]} features**.

**Key Result**: {best_name} achieved the best performance with **{test_acc:.1%} test accuracy** 
and an overfitting gap of **{gap:.1%}** ({('✅ Good' if gap < 0.10 else '⚠️ Overfit')}).
""")
                                
                                st.markdown("### 🔑 Key Findings")
                                
                                # Find model with lowest overfitting
                                sorted_by_gap = sorted(leaderboard, key=lambda x: x.get('overfitting_gap', 1))
                                best_gap_model = sorted_by_gap[0]
                                
                                st.markdown(f"""
- **Winner**: {best_name} achieved highest adjusted score (test accuracy with overfitting penalty)
- **Test Accuracy**: {test_acc:.1%} on held-out test data
- **Generalization**: {('✅ Excellent' if gap < 0.05 else '✅ Good' if gap < 0.10 else '⚠️ Shows overfitting')} (gap: {gap:.1%})
- **Most Reliable Model**: {best_gap_model['model']} has lowest overfitting ({best_gap_model.get('overfitting_gap', 0):.1%} gap)
- **Model Diversity**: Tested {len(leaderboard)} algorithms including {', '.join([m['model'] for m in leaderboard[:3]])}
""")
                                
                                st.markdown("### 💡 Recommendations")
                                recommendations = []
                                
                                if gap > 0.15:
                                    recommendations.append(f"**Reduce Overfitting**: {best_name} shows high train-test gap ({gap:.1%}). Consider using {best_gap_model['model']} (gap: {best_gap_model.get('overfitting_gap', 0):.1%}) for better generalization.")
                                elif gap > 0.10:
                                    recommendations.append(f"**Monitor Overfitting**: {best_name} shows moderate overfitting. Validate on additional data before deployment.")
                                else:
                                    recommendations.append(f"**Deploy Confidently**: {best_name} shows excellent generalization (gap: {gap:.1%}). Ready for production.")
                                
                                if test_acc < 0.50:
                                    recommendations.append(f"**Improve Performance**: Current accuracy ({test_acc:.1%}) is low. Consider feature engineering or collecting more data.")
                                
                                recommendations.append("**Validate Results**: Test final model on completely unseen data before production deployment.")
                                recommendations.append("**Monitor Performance**: Set up tracking to detect model drift over time.")
                                
                                for rec in recommendations:
                                    st.markdown(f"- {rec}")
                            
                            st.markdown("### ⚠️ Note")
                            st.info("This is a structured report generated from your results. For AI-powered insights and detailed analysis, try again tomorrow when your API quota resets, or upgrade your plan.")
                        
                        else:
                            st.error(f"AI report generation failed: {error_msg}")
        
        st.write("---")
        st.write("Download a comprehensive PDF report with all results and visualizations.")
        
        if st.button("📥 Download PDF Report"):
            try:
                from app.report_builder import ReportBuilder
                
                with st.spinner("Generating PDF report..."):
                    builder = ReportBuilder()
                    report_path = builder.generate_report(
                        data=st.session_state.data,
                        profile=st.session_state.profile,
                        results=st.session_state.results,
                        task_type=st.session_state.task_type,
                        recommendation=st.session_state.get('recommendation')
                    )
                    
                    # Read and download
                    with open(report_path, 'rb') as f:
                        st.download_button(
                            label="Download Report",
                            data=f.read(),
                            file_name=f"AutoML_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    
                    st.success(f"✅ Report generated: {report_path}")
            except Exception as e:
                st.error(f"Error generating report: {e}")
