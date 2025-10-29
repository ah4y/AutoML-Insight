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
from utils.seed_utils import set_seed
from utils.logging_utils import setup_logger
from sklearn.datasets import load_iris, load_wine
from utils.jupyter_client import JupyterServerClient, ColabServerSetup, RemoteExecutor

# Initialize logger
logger = setup_logger()


class AutoMLDashboard:
    """Main dashboard for AutoML-Insight."""
    
    def __init__(self):
        self.jupyter_client = None
        self.initialize_session_state()
    
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
            'remote_logs': []
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
            except Exception as e:
                logger.warning(f"AI engine not available: {e}")
                st.session_state.ai_engine = False  # Mark as attempted
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        if st.session_state.data is not None:
            self.render_tabs()
        else:
            self.render_welcome()
    
    def render_welcome(self):
        """Render welcome screen."""
        # Show Colab setup instructions if triggered
        if st.session_state.get('show_colab_instructions') and st.session_state.get('ngrok_token_input'):
            st.title("☁️ Google Colab Setup Instructions")
            
            setup_code = ColabServerSetup.get_setup_code(st.session_state.ngrok_token_input)
            
            st.markdown("""
            ### 📋 Steps to Connect to Google Colab:
            
            1. **Open Google Colab**: Click the button below
            2. **Create a new notebook**
            3. **Copy and paste** the code below into the first cell
            4. **Run the cell** (Shift+Enter)
            5. **Copy the Public URL** that appears
            6. **Paste it in the sidebar** and click Connect
            """)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🔗 Open Colab", type="primary"):
                    st.markdown(f"[Open Google Colab]({ColabServerSetup.get_colab_url()})")
            
            st.markdown("### 📝 Setup Code:")
            st.code(setup_code, language='python')
            
            if st.button("✖️ Close Instructions"):
                st.session_state.show_colab_instructions = False
                st.rerun()
            
            return
        
        # Normal welcome screen
        st.info("👈 Please upload a dataset or select Demo Mode from the sidebar to get started.")
        
        st.markdown("""
        ### Features
        - 📊 **Automatic Dataset Profiling**: Statistical analysis and meta-features
        - 🤖 **Multi-Model Training**: 7+ supervised and 5+ unsupervised algorithms
        - 📈 **Comprehensive Evaluation**: Nested CV with confidence intervals
        - 🔍 **Model Explainability**: SHAP values and feature importance
        - 🎯 **Smart Recommendations**: Meta-learning based model selection
        - 📄 **PDF Reports**: Exportable analytical reports
        - 🌐 **Remote Execution**: Run on Jupyter servers or Google Colab
        
        ### Supported Tasks
        - **Classification**: Binary and multi-class problems
        - **Clustering**: Unsupervised pattern discovery
        
        ### Execution Modes
        - **🖥️ Local Machine**: Train on your computer (limited by RAM)
        - **🌐 Remote Jupyter**: Connect to any Jupyter server
        - **☁️ Google Colab**: Free 12 GB RAM + T4 GPU
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
            uploaded_file = st.sidebar.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload your dataset in CSV format"
            )
            
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
                ["🖥️ Local Machine", "🌐 Remote Jupyter Server", "☁️ Google Colab Setup"],
                help="Train locally, on remote server, or via Colab"
            )
            
            # Store execution mode
            if "Local" in execution_mode:
                st.session_state.execution_mode = "local"
            elif "Remote" in execution_mode:
                st.session_state.execution_mode = "remote"
            elif "Colab" in execution_mode:
                st.session_state.execution_mode = "colab"
            
            # Show connection UI for remote/colab modes
            if st.session_state.execution_mode == "remote":
                self.render_jupyter_connection()
            elif st.session_state.execution_mode == "colab":
                self.render_colab_setup()
            
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
                    if st.sidebar.button("� Run on Remote Server", type="primary", use_container_width=True):
                        set_seed(random_seed)
                        self.run_automl_remote()
            
            elif st.session_state.execution_mode == "colab":
                if not st.session_state.jupyter_connected:
                    st.sidebar.info("💡 Set up Colab and connect first")
                else:
                    if st.sidebar.button("� Run on Google Colab", type="primary", use_container_width=True):
                        set_seed(random_seed)
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
        st.sidebar.markdown("### 🔌 Jupyter Server")
        
        # Show connection status prominently
        if st.session_state.jupyter_connected:
            st.sidebar.success("✅ CONNECTED")
        else:
            st.sidebar.warning("⚠️ Not Connected")
        
        # Connection form
        server_url = st.sidebar.text_input(
            "Server URL",
            value=st.session_state.jupyter_server_url or "http://localhost:8888",
            placeholder="http://localhost:8888",
            help="URL of your Jupyter server",
            key="jupyter_url_input"
        )
        
        token = st.sidebar.text_input(
            "Token (optional)",
            value=st.session_state.jupyter_token or "",
            type="password",
            help="Leave empty if no token required",
            key="jupyter_token_input"
        )
        
        # Connection buttons
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔗 Connect", use_container_width=True, key="jupyter_connect_btn"):
                self.connect_to_jupyter(server_url, token)
        
        with col2:
            if st.button("❌ Disconnect", use_container_width=True, disabled=not st.session_state.jupyter_connected, key="jupyter_disconnect_btn"):
                self.disconnect_jupyter()
    
    def render_colab_setup(self):
        """Render Google Colab setup instructions in sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ☁️ Colab Setup")
        
        # Get ngrok token
        ngrok_token = st.sidebar.text_input(
            "Ngrok Token",
            type="password",
            help="Get free token from ngrok.com",
            key="ngrok_token_input"
        )
        
        if ngrok_token:
            # Generate setup code
            setup_code = ColabServerSetup.get_setup_code(ngrok_token)
            
            # Show instructions in main area (not sidebar)
            if 'show_colab_instructions' not in st.session_state:
                st.session_state.show_colab_instructions = True
            
            # Button to show instructions
            if st.sidebar.button("📋 Show Setup Instructions", use_container_width=True):
                st.session_state.show_colab_instructions = True
            
            # URL input
            st.sidebar.markdown("**Paste Colab URL:**")
            colab_url = st.sidebar.text_input(
                "Public URL",
                placeholder="https://xxxx.ngrok.io",
                key="colab_url_input"
            )
            
            if colab_url and st.sidebar.button("🔗 Connect", use_container_width=True):
                self.connect_to_jupyter(colab_url, "")
        else:
            st.sidebar.info("👉 Enter ngrok token to begin")
    
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
            return
        
        try:
            # Save dataset to temp file
            import tempfile
            temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            st.session_state.data.to_csv(temp_csv.name, index=False)
            temp_csv.close()
            
            # Get configuration
            target_col = st.session_state.get('target_col')
            task_type = st.session_state.get('task_type', 'Classification')
            random_seed = st.session_state.get('random_seed', 42)
            max_features = st.session_state.get('recommended_config', {}).get('recommended_max_features', 1000)
            
            # Create executor
            executor = RemoteExecutor(self.jupyter_client)
            
            # Progress container
            progress_container = st.container()
            status_text = progress_container.empty()
            progress_bar = progress_container.progress(0)
            log_container = progress_container.expander("📋 Execution Logs", expanded=True)
            log_area = log_container.empty()
            
            # Progress callback
            def update_progress(message, progress):
                status_text.info(message)
                if progress is not None:
                    progress_bar.progress(progress)
                # Update logs
                st.session_state.remote_logs.append(message)
                log_area.code('\n'.join(st.session_state.remote_logs[-15:]))
            
            # Execute remotely
            result = executor.execute_automl(
                data_path=temp_csv.name,
                target_col=target_col,
                task_type=task_type,
                random_seed=random_seed,
                max_features=max_features,
                progress_callback=update_progress
            )
            
            # Cleanup temp file
            import os
            os.unlink(temp_csv.name)
            
            # Check result
            if result.get('success'):
                status_text.success("🎉 Remote training completed successfully!")
                progress_bar.progress(100)
                
                # Store results
                results_data = result.get('results', {})
                st.session_state.remote_training_results = results_data
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Training Results")
                
                # Model leaderboard
                model_results = results_data.get('results', [])
                if model_results:
                    df_results = pd.DataFrame(model_results)
                    st.dataframe(df_results.sort_values('accuracy', ascending=False), use_container_width=True)
                    
                    # Best model info
                    best_model = results_data.get('best_model', 'N/A')
                    best_accuracy = max([r.get('accuracy', 0) for r in model_results])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🏆 Best Model", best_model)
                    with col2:
                        st.metric("🎯 Best Accuracy", f"{best_accuracy:.4f}")
                    
                    # Dataset info
                    st.markdown("**Dataset Info:**")
                    st.write(f"- Samples: {results_data.get('n_samples', 'N/A'):,}")
                    st.write(f"- Features Used: {results_data.get('n_features', 'N/A'):,}")
                
            else:
                error_msg = result.get('error', 'Unknown error')
                status_text.error(f"❌ Remote training failed: {error_msg}")
                
                # Show logs
                logs = result.get('logs', [])
                if logs:
                    with st.expander("📋 Error Logs"):
                        st.code('\n'.join(logs))
        
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
            
            # Preprocess with smart feature selection
            st.info("🔧 Preprocessing data...")
            preprocessor = DataPreprocessor(max_features=max_features)
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
            
            # Train models
            if task_type == "Classification":
                self.run_classification(X_processed, y_processed)
            else:
                self.run_clustering(X_processed)
            
            st.success("✅ AutoML pipeline completed!")
            
        except Exception as e:
            st.error(f"Error running AutoML: {e}")
            logger.error(f"AutoML error: {e}", exc_info=True)
    
    def run_classification(self, X, y):
        """Run classification pipeline."""
        st.info("🤖 Training classification models...")
        
        # Get models
        models = get_supervised_models(st.session_state.random_seed)
        
        # Determine appropriate CV strategy based on data size
        from collections import Counter
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        total_samples = len(y)
        
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
        
        # Evaluate models
        evaluator = ClassificationEvaluator(n_folds=n_folds, n_repeats=n_repeats)
        results = {}
        
        progress_bar = st.progress(0)
        for idx, (name, model) in enumerate(models.items()):
            st.text(f"Training {name}...")
            try:
                result = evaluator.evaluate_model(model, X, y, name)
                results[name] = result
                # Train a fresh model on all data for explainability later
                from sklearn.base import clone
                fresh_model = clone(model)
                fresh_model.fit(X, y)
                results[name]['trained_model'] = fresh_model
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                st.warning(f"⚠️ {name} failed: {str(e)[:100]}")
            
            progress_bar.progress((idx + 1) / len(models))
        
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview",
            "🤖 Models",
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
            self.render_explainability()
        
        with tab4:
            self.render_recommendation()
        
        with tab5:
            self.render_report()
    
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
        """Render classification results."""
        st.subheader("🤖 Classification Models")
        
        if not st.session_state.results:
            st.info("Run AutoML to see results")
            return
        
        # AI-Powered Results Interpretation
        if st.session_state.ai_engine and st.session_state.ai_engine is not False:
            with st.expander("🤖 AI Performance Analysis", expanded=True):
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
                            
                            # Get AI interpretation
                            response = st.session_state.ai_engine._call_llm(perf_prompt)
                            insights = st.session_state.ai_engine._parse_response(response)
                            
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
        
        # Leaderboard
        st.subheader("Model Leaderboard")
        evaluator = st.session_state.evaluator
        leaderboard = evaluator.get_leaderboard('accuracy')
        
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
        """Render explainability tab."""
        st.subheader("🔍 Model Explainability")
        
        if not st.session_state.results:
            st.info("Run AutoML to see explainability results")
            return
        
        # Check if clustering task
        is_clustering = st.session_state.task_type == "Clustering"
        
        # Model selection
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("Select Model", model_names)
        
        if selected_model:
            model = st.session_state.models[selected_model]
            X = st.session_state.X_processed
            feature_names = st.session_state.preprocessor.get_feature_names()
            
            # AI-Powered Clustering Explainability (for clustering tasks)
            if is_clustering and st.session_state.ai_engine and st.session_state.ai_engine is not False:
                with st.expander("🤖 AI Cluster Analysis", expanded=True):
                    with st.spinner("🤖 AI is analyzing cluster structure..."):
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

Provide detailed analysis in JSON format:
1. "cluster_quality": Assess the quality of these clusters based on metrics
2. "cluster_characteristics": What makes each cluster distinct? (be general)
3. "balance_assessment": Are clusters well-balanced or is there imbalance?
4. "actionable_insights": 2-3 ways to use or improve these clusters

Be specific about the model and metrics."""
                            
                            response = st.session_state.ai_engine._call_llm(prompt)
                            insights = st.session_state.ai_engine._parse_response(response)
                            
                            if 'cluster_quality' in insights:
                                st.info(f"**📊 Cluster Quality:** {insights['cluster_quality']}")
                            
                            if 'cluster_characteristics' in insights:
                                st.success(f"**🔍 Cluster Characteristics:** {insights['cluster_characteristics']}")
                            
                            if 'balance_assessment' in insights:
                                st.info(f"**⚖️ Balance Assessment:** {insights['balance_assessment']}")
                            
                            if 'actionable_insights' in insights:
                                st.warning("**→ Actionable Insights:**")
                                if isinstance(insights['actionable_insights'], list):
                                    for insight in insights['actionable_insights']:
                                        st.markdown(f"- {insight}")
                                else:
                                    st.markdown(insights['actionable_insights'])
                        
                        except Exception as e:
                            logger.warning(f"Failed to generate AI clustering explainability: {e}")
                            st.error(f"AI analysis failed: {e}")
            
            # AI-Powered Feature Importance Interpretation (for classification tasks)
            elif not is_clustering and st.session_state.ai_engine and st.session_state.ai_engine is not False:
                with st.expander("🤖 AI Feature Analysis", expanded=True):
                    with st.spinner("🤖 AI is analyzing feature importance..."):
                        try:
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
                                
                                # Create AI prompt
                                features_str = "\n".join([f"- {name}: {imp:.4f}" for name, imp in top_features])
                                
                                prompt = f"""You are an expert data scientist analyzing feature importance for {selected_model}.

**Top 5 Most Important Features:**
{features_str}

**Dataset Context:**
- Total Features: {len(feature_names)}
- Model: {selected_model}

Provide analysis in JSON format:
1. "key_insights": What do these top features tell us about the problem?
2. "feature_relationships": Any interesting patterns or relationships?
3. "actionable_advice": 2-3 specific recommendations based on these features

Be concise and domain-agnostic (don't assume you know the domain)."""
                                
                                response = st.session_state.ai_engine._call_llm(prompt)
                                insights = st.session_state.ai_engine._parse_response(response)
                                
                                if 'key_insights' in insights:
                                    st.info(f"**🔑 Key Insights:** {insights['key_insights']}")
                                
                                if 'feature_relationships' in insights:
                                    st.success(f"**🔗 Feature Relationships:** {insights['feature_relationships']}")
                                
                                if 'actionable_advice' in insights:
                                    st.info("**→ Actionable Advice:**")
                                    if isinstance(insights['actionable_advice'], list):
                                        for advice in insights['actionable_advice']:
                                            st.markdown(f"- {advice}")
                                    else:
                                        st.markdown(insights['actionable_advice'])
                        
                        except Exception as e:
                            logger.warning(f"Failed to generate AI feature insights: {e}")
            
            # Check if dataset is too large for SHAP
            n_features = X.shape[1]
            if n_features > 1000:
                st.warning(f"⚠️ Dataset has {n_features:,} features. SHAP explanations may be slow or fail due to memory constraints.")
                st.info("💡 Tip: SHAP works best with < 1000 features. Consider feature selection for large datasets.")
                
                # Ask user if they want to proceed
                if not st.checkbox(f"Attempt SHAP anyway (may cause memory errors)", value=False):
                    st.info("Showing only native model explanations (feature importance, coefficients).")
                    # Skip SHAP, only show native explanations
                    explanations = {}
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
                    # User chose to attempt SHAP
                    with st.spinner(f"Generating SHAP explanations (this may take a while)..."):
                        try:
                            explainer = ModelExplainer()
                            explanations = explainer.explain_model(
                                model, X, feature_names, sample_size=20  # Use minimal samples
                            )
                        except Exception as e:
                            st.error(f"SHAP failed: {e}")
                            explanations = {}
            else:
                # Normal size dataset
                with st.spinner(f"Generating explanations for {selected_model}..."):
                    try:
                        explainer = ModelExplainer()
                        explanations = explainer.explain_model(
                            model, X, feature_names, sample_size=50
                        )
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
                                logger.warning(f"Failed to generate AI clustering recommendations: {e}")
                                st.error(f"AI analysis failed: {e}")
                
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
                        logger.warning(f"Failed to generate AI report: {e}")
                        st.error(f"AI report generation failed: {e}")
        
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
