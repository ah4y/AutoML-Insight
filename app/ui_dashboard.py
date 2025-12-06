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
from core.ai_insights import get_ai_engine  # Standard AI insights  
from core.ai_insights_enhanced import get_enhanced_ai_engine, EnhancedDatasetStatistics  # Enhanced AI insights
from core.dimred import DimRedConfig, load_dimred_config  # NEW: Dimensionality reduction
from core.dimred_evaluator import DimRedEvaluator  # NEW: Enhanced evaluation with dimred
from core.advanced_optimization import AdvancedHyperparameterOptimizer, AutoMLPipeline  # NEW: Professional optimization
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
            'ai_engine': None,  # Standard AI engine instance
            'enhanced_ai_engine': None,  # Enhanced AI engine instance
            'ai_insights': None,  # Store AI insights
            'execution_mode': 'local',
            'jupyter_server_url': '',
            'jupyter_token': '',
            'jupyter_connected': False,
            'remote_logs': [],
            # NEW: Configuration system
            'show_configuration': False,
            'config_analyzed': False,
            'dataset_config': {},
            'optimization_config': {
                'time_minutes': 15,
                'max_trials': 100,
                'include_ensemble': True,
                'advanced_features': []
            },
            # NEW: App Stage Management
            'app_stage': 'welcome',  # welcome, configure, results
            # NEW: Feature Engineering state
            'show_feature_engineering': False,
            'feature_engineering_applied': False,
            'selected_columns': None,
            'ai_analysis': None,
            # NEW: Dimensionality Reduction settings
            'dimred_enabled': 'auto',  # off, on, auto
            'dimred_method': 'auto',   # pca, tsvd, ipca, auto  
            'dimred_variance_target': 0.95,
            'dimred_k_max': 256,
            'dimred_config': None,
            'dimred_results': None,
            # NEW: Class filtering settings
            'enable_class_filter': False,
            'min_class_samples': 5,
            # Random seed for reproducibility - user configurable
            'random_seed': 42  # Initial default, will be overridden by user config
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render(self):
        """Render the main dashboard with 3-stage navigation."""
        # Set page config for wide layout
        if not st.session_state.get('page_config_set', False):
            st.set_page_config(
                page_title="AutoML-Insight", 
                layout="wide", 
                page_icon="🤖",
                initial_sidebar_state="expanded"
            )
            st.session_state.page_config_set = True
        
        # Handle configuration dashboard separately
        if st.session_state.get('show_configuration', False):
            self.render_configuration_dashboard()
            return
            
        # Initialize AI engines if not already done
        if st.session_state.ai_engine is None:
            try:
                st.session_state.ai_engine = get_ai_engine()
                st.session_state.enhanced_ai_engine = get_enhanced_ai_engine()  # Enhanced AI engine
                
                if st.session_state.enhanced_ai_engine:
                    logger.info(f"Enhanced AI engine initialized: {st.session_state.enhanced_ai_engine.provider}")
                elif st.session_state.ai_engine:
                    logger.info(f"Standard AI engine initialized: {st.session_state.ai_engine.provider}")
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
        
        # 3-Stage Navigation System
        app_stage = st.session_state.app_stage
        
        if app_stage == 'welcome':
            self.render_welcome_stage()
        elif app_stage == 'configure':
            self.render_configuration_stage()
        elif app_stage == 'results':
            self.render_results_stage()
        else:
            # Fallback to welcome
            st.session_state.app_stage = 'welcome'
            self.render_welcome_stage()
    
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
    
    def render_welcome_stage(self):
        """Render the welcome stage with app introduction and upload."""
        # Main container for better layout
        st.markdown("""
        <style>
        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            margin: 0 auto;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # App header with proper centering
        st.markdown("""
        <div style='text-align: center; margin: 2rem auto; max-width: 1200px;'>
            <h1 style='color: #1f77b4; font-size: 3rem; margin-bottom: 0;'>🤖 AutoML-Insight</h1>
            <p style='font-size: 1.3rem; color: #666; margin-top: 0.5rem;'>
                Professional AutoML Platform for Automated Model Selection, Training & Explainability
            </p>
            <div style='color: #999; font-size: 1rem; margin-top: 1rem;'>Step 1 of 3 - Dataset Upload & Feature Engineering</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase with better spacing
        st.markdown("<div style='max-width: 1200px; margin: 0 auto;'>", unsafe_allow_html=True)
        st.markdown("### ✨ **What AutoML-Insight Does for You**")
        
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;'>
                <h3 style='margin: 0; font-size: 1.3rem;'>🧠 AI-Powered Analysis</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Dataset insights, recommendations, and optimization strategies powered by LLM</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;'>
                <h3 style='margin: 0; font-size: 1.3rem;'>⚙️ Smart Configuration</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Dataset-aware configuration with automatic parameter optimization</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;'>
                <h3 style='margin: 0; font-size: 1.3rem;'>🚀 Professional Results</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Enterprise-grade AutoML with explainability and comprehensive reports</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close container
        
        st.markdown("---")
        
        # Professional Upload section with better layout
        st.markdown("<div style='max-width: 1200px; margin: 0 auto;'>", unsafe_allow_html=True)
        st.markdown("### 📁 **Get Started - Upload Your Dataset**")
        st.markdown("*Upload your CSV file or try our demo dataset to begin your AutoML journey*")
        
        # Upload area with better organization
        upload_col1, upload_col2 = st.columns([3, 2], gap="large")
        
        with upload_col1:
            # Demo mode toggle with better styling
            demo_mode = st.checkbox(
                "🎮 **Use Demo Dataset (Credit Card Fraud Detection)**", 
                value=False, 
                help="Try with sample credit card fraud detection dataset - perfect for testing AutoML features"
            )
            
            if demo_mode:
                st.markdown("**🎯 Demo Dataset Features:**")
                st.markdown("""
                - 💳 Real credit card transaction data
                - 🎯 Binary classification (fraud vs normal)
                - 📊 30 features with PCA transformations
                - 🔢 284,807 samples with class imbalance
                """)
                
                if st.button("🚀 **Load Demo Dataset**", type="primary", use_container_width=True):
                    with st.spinner("📥 Loading demo dataset..."):
                        self.load_demo_data()
                        st.success("✅ Demo dataset loaded successfully!")
                        st.session_state.app_stage = 'configure'
                        st.rerun()
            else:
                # File uploader with better styling
                st.markdown("**📂 Upload Your CSV File:**")
                uploaded_file = st.file_uploader(
                    "Choose your CSV file",
                    type=['csv'],
                    help="Upload your dataset in CSV format to begin AutoML analysis",
                    key='main_uploader',
                    label_visibility="collapsed"
                )
                
                if uploaded_file is not None:
                    try:
                        with st.spinner("🔄 Processing your dataset..."):
                            data = pd.read_csv(uploaded_file)
                            st.session_state.data = data
                            st.session_state.uploaded_file_name = uploaded_file.name
                            
                            # Clear previous AI insights
                            st.session_state.ai_insights = None
                        
                        st.success(f"✅ **Successfully loaded:** {data.shape[0]:,} rows × {data.shape[1]} columns!")
                        
                        # Show dataset overview and AI analyzer
                        st.markdown("---")
                        self._render_dataset_overview_and_analyzer(data)
                        
                    except Exception as e:
                        st.error(f"❌ **Error loading file:** {e}")
                        st.info("💡 **Tip:** Make sure your file is a valid CSV with proper headers")
        
        with upload_col2:
            # Enhanced requirements and tips
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #1f77b4;'>
                <h4 style='margin-top: 0; color: #1f77b4; display: flex; align-items: center;'>
                    📋 Dataset Requirements
                </h4>
                <ul style='margin: 0; padding-left: 1.2rem; line-height: 1.6;'>
                    <li><strong>Format:</strong> CSV files only</li>
                    <li><strong>Headers:</strong> Clean column names</li>
                    <li><strong>Data:</strong> Numeric/categorical values</li>
                    <li><strong>Size:</strong> < 100MB recommended</li>
                    <li><strong>Quality:</strong> Minimal missing values preferred</li>
                </ul>
                <br>
                <h4 style='color: #28a745; margin-bottom: 0.5rem;'>🎯 Supported ML Tasks</h4>
                <div style='display: flex; flex-direction: column; gap: 0.5rem;'>
                    <div style='background: rgba(40, 167, 69, 0.1); padding: 0.5rem; border-radius: 5px;'>
                        <strong>🔍 Classification:</strong> Predict categories/labels
                    </div>
                    <div style='background: rgba(23, 162, 184, 0.1); padding: 0.5rem; border-radius: 5px;'>
                        <strong>🕸️ Clustering:</strong> Discover hidden patterns
                    </div>
                </div>
                <br>
                <div style='background: rgba(255, 193, 7, 0.1); padding: 0.8rem; border-radius: 5px; border-left: 3px solid #ffc107;'>
                    <strong>💡 Pro Tip:</strong> Clean data = Better models!<br>
                    Use our Feature Engineering tools to optimize your dataset.
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_dataset_overview_and_analyzer(self, data):
        """Render dataset overview and AI analyzer after upload."""
        st.markdown("---")
        st.markdown("<div style='max-width: 1200px; margin: 0 auto;'>", unsafe_allow_html=True)
        
        # Professional dataset statistics with enhanced metrics
        st.markdown("#### 📈 **Dataset Overview**")
        
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        
        with col1:
            st.metric(
                label="📊 Total Rows", 
                value=f"{len(data):,}",
                help="Number of samples/observations in your dataset"
            )
        with col2:
            st.metric(
                label="📐 Total Columns", 
                value=len(data.columns),
                help="Number of features/variables in your dataset"
            )
        with col3:
            numeric_cols = data.select_dtypes(include=[np.number]).shape[1]
            categorical_cols = data.select_dtypes(include=['object', 'category']).shape[1]
            st.metric(
                label="🔢 Numeric Features", 
                value=f"{numeric_cols}",
                delta=f"{categorical_cols} categorical",
                help="Distribution of numeric vs categorical features"
            )
        with col4:
            missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric(
                label="❓ Missing Data", 
                value=f"{missing_percentage:.1f}%",
                delta=f"{memory_mb:.1f} MB",
                help="Percentage of missing values and dataset size"
            )
        
        # Enhanced Data Analysis Tabs with better organization
        st.markdown("### 📊 **Comprehensive Dataset Analysis**")
        st.markdown("*Explore your data through multiple analytical perspectives*")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 **Preview & Info**", 
            "📊 **Distribution Analysis**", 
            "🔗 **Correlation Matrix**", 
            "📈 **Data Quality Assessment**"
        ])
        
        with tab1:
            st.subheader("📋 Dataset Preview")
            st.dataframe(data.head(10), use_container_width=True, height=300)
            
            st.subheader("📊 Column Information Summary")
            
            # Create columns info with better formatting
            info_data = []
            for col in data.columns:
                dtype = str(data[col].dtype)
                null_count = data[col].isnull().sum()
                null_pct = (null_count / len(data)) * 100
                unique_count = data[col].nunique()
                
                # Add color indicators for data types
                if 'int' in dtype or 'float' in dtype:
                    type_icon = "🔢"
                elif 'object' in dtype:
                    type_icon = "📝"
                elif 'datetime' in dtype:
                    type_icon = "📅"
                else:
                    type_icon = "❓"
                
                info_data.append({
                    'Column': f"{type_icon} {col}",
                    'Data Type': dtype,
                    'Non-Null Count': f"{len(data) - null_count:,}",
                    'Missing (%)': f"{null_pct:.1f}%",
                    'Unique Values': f"{unique_count:,}"
                })
            
            # Display as a styled dataframe
            info_df = pd.DataFrame(info_data)
            st.dataframe(
                info_df, 
                use_container_width=True,
                height=min(400, len(info_df) * 35 + 50)  # Dynamic height based on rows
            )
        
        with tab2:
            st.subheader("Data Distribution Analysis")
            
            # Numeric columns analysis
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.markdown("**Numeric Columns Distribution**")
                
                for i in range(0, len(numeric_cols), 2):
                    cols = st.columns(2)
                    for j, col_name in enumerate(numeric_cols[i:i+2]):
                        with cols[j]:
                            try:
                                import plotly.express as px
                                fig = px.histogram(data, x=col_name, title=f"Distribution of {col_name}")
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            except:
                                st.write(f"**{col_name}** - Basic Statistics:")
                                st.write(data[col_name].describe())
            
            # Categorical columns analysis
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                st.markdown("**Categorical Columns Distribution**")
                
                for col_name in categorical_cols[:4]:  # Limit to first 4
                    value_counts = data[col_name].value_counts().head(10)
                    try:
                        import plotly.express as px
                        fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                   title=f"Top Values in {col_name}")
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.write(f"**{col_name}** - Top 10 Values:")
                        st.write(value_counts)
        
        with tab3:
            st.subheader("Correlation Analysis")
            
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                try:
                    import plotly.express as px
                    corr_matrix = numeric_data.corr()
                    fig = px.imshow(corr_matrix, 
                                  title="Feature Correlation Matrix",
                                  aspect="auto",
                                  color_continuous_scale="RdBu_r")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show strongest correlations
                    st.markdown("**Strongest Correlations:**")
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.5:  # Strong correlation threshold
                                corr_pairs.append({
                                    'Feature 1': corr_matrix.columns[i],
                                    'Feature 2': corr_matrix.columns[j],
                                    'Correlation': f"{corr_val:.3f}"
                                })
                    
                    if corr_pairs:
                        st.dataframe(pd.DataFrame(corr_pairs), use_container_width=True)
                    else:
                        st.info("No strong correlations (>0.5) found between features.")
                        
                except Exception as e:
                    st.write("Correlation matrix:")
                    st.dataframe(numeric_data.corr(), use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")
        
        with tab4:
            st.subheader("Data Quality Assessment")
            
            # Missing data analysis
            missing_data = data.isnull().sum()
            missing_pct = (missing_data / len(data)) * 100
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing Percentage': missing_pct.values
            }).sort_values('Missing Count', ascending=False)
            
            if missing_df['Missing Count'].sum() > 0:
                st.markdown("**Missing Data Analysis:**")
                try:
                    import plotly.express as px
                    fig = px.bar(missing_df[missing_df['Missing Count'] > 0], 
                               x='Column', y='Missing Percentage',
                               title="Missing Data by Column")
                    fig.update_layout(height=400, xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            else:
                st.success("✅ No missing data detected!")
            
            # Data types summary
            st.markdown("**Data Types Summary:**")
            dtype_counts = data.dtypes.value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Column Types:**")
                for dtype, count in dtype_counts.items():
                    st.write(f"• {dtype}: {count} columns")
            
            with col2:
                # Potential issues
                issues = []
                
                # Check for columns with single value
                for col in data.columns:
                    if data[col].nunique() == 1:
                        issues.append(f"Column '{col}' has only one unique value")
                
                # Check for high cardinality categorical columns
                for col in data.select_dtypes(include=['object']).columns:
                    if data[col].nunique() > len(data) * 0.8:
                        issues.append(f"Column '{col}' has very high cardinality ({data[col].nunique()} unique values)")
                
                if issues:
                    st.warning("**Potential Issues:**")
                    for issue in issues[:5]:  # Show top 5 issues
                        st.write(f"• {issue}")
                else:
                    st.success("✅ No obvious data quality issues detected")
        
        # Professional AI and Feature Engineering Section
        st.markdown("---")
        st.markdown("### 🧠 **Intelligent Dataset Analysis & Feature Engineering**")
        st.markdown("*Leverage AI insights to optimize your dataset before training*")
        
        # Action buttons in professional layout
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
        
        with col1:
            ai_button_text = "🔄 Re-analyze with AI" if st.session_state.get('ai_analysis') else "🔍 Analyze Dataset with AI"
            if st.button(ai_button_text, type="secondary", use_container_width=True, help="Get AI-powered insights about your dataset"):
                with st.spinner("🤖 AI is analyzing your dataset..."):
                    try:
                        analysis = self._generate_ai_dataset_analysis(data)
                        st.session_state.ai_analysis = analysis
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ AI Analysis failed: {e}")
        
        with col2:
            fe_button_text = "🔄 Modify Features" if st.session_state.get('feature_engineering_applied') else "🛠️ Feature Engineering"
            if st.button(fe_button_text, type="secondary", use_container_width=True, help="Edit and transform your dataset"):
                st.session_state.show_feature_engineering = True
                st.rerun()
        
        with col3:
            # Status indicator
            ai_status = "✅ Complete" if st.session_state.get('ai_analysis') else "⏳ Pending"
            fe_status = "✅ Applied" if st.session_state.get('feature_engineering_applied') else "⏳ None"
            st.markdown(f"**Analysis:** {ai_status}")
            st.markdown(f"**Engineering:** {fe_status}")
        
        # Display AI Analysis if available
        if st.session_state.get('ai_analysis') or st.session_state.get('ai_insights'):
            # Check both possible sources of AI insights
            analysis = st.session_state.get('ai_analysis') or st.session_state.get('ai_insights')
            
            with st.expander("🎯 **AI Insights & Recommendations**", expanded=True):
                if isinstance(analysis, dict):
                    # Handle structured analysis
                    # Task type recommendation
                    if 'task_recommendation' in analysis:
                        task_rec = analysis['task_recommendation']
                        st.success(f"**Recommended Task:** {task_rec['task']} ({task_rec['confidence']:.0%} confidence)")
                        st.info(f"**Reasoning:** {task_rec['reasoning']}")
                    
                    # Target column suggestions
                    if 'target_suggestions' in analysis:
                        st.markdown("**🎯 Potential Target Columns:**")
                        for suggestion in analysis['target_suggestions'][:3]:
                            st.write(f"• `{suggestion['column']}` - {suggestion['reasoning']}")
                    
                    # Enhanced AI insights display
                    if isinstance(analysis, dict) and any(key.startswith(('dataset_', 'key_', 'critical_')) for key in analysis.keys()):
                        self._display_enhanced_ai_insights(analysis)
                
                elif isinstance(analysis, str):
                    # Handle simple string analysis
                    st.markdown(analysis)
                    
                else:
                    # Fallback for any other format
                    st.info("AI analysis completed. Results may be displayed in other sections.")
        else:
            # Show placeholder when no AI analysis is available
            with st.expander("🎯 **AI Insights & Recommendations**", expanded=False):
                st.info("Click 'Analyze Dataset with AI' to get intelligent insights about your data.")
        
        # Feature Engineering Section
        if st.session_state.get('show_feature_engineering', False):
            self._render_feature_engineering_section(data)
        
        # Professional Navigation Section
        st.markdown("---")
        self._render_step1_navigation()
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close dataset overview container
        
        # Continue to configuration button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("➡️ Continue to Configuration", type="primary", use_container_width=True):
                st.session_state.app_stage = 'configure'
                st.rerun()
    
    def _generate_ai_dataset_analysis(self, data):
        """Generate AI-powered dataset analysis and recommendations."""
        try:
            # Try enhanced AI analysis first
            if st.session_state.get('enhanced_ai_engine'):
                try:
                    # Generate comprehensive insights using enhanced AI
                    insights = st.session_state.enhanced_ai_engine.generate_comprehensive_insights(
                        data, context="initial_analysis"
                    )
                    if insights:
                        return insights
                except Exception as e:
                    logger.warning(f"Enhanced AI analysis failed: {e}")
            
            # Fallback to basic analysis
            return self._generate_basic_dataset_analysis(data)
            
        except Exception as e:
            logger.error(f"AI dataset analysis failed: {e}")
            return self._generate_basic_dataset_analysis(data)
    
    def _generate_basic_dataset_analysis(self, data):
        """Generate basic dataset analysis as fallback."""
        try:
            # Basic dataset characteristics
            n_rows, n_cols = data.shape
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            missing_data = data.isnull().sum().sum()
            
            # Simple heuristics for task recommendation
            analysis = {
                'task_recommendation': {
                    'task': 'Classification',
                    'confidence': 0.8,
                    'reasoning': 'Dataset appears suitable for classification based on column types and structure'
                },
                'target_suggestions': [],
                'quality_issues': []
            }
            
            # Suggest potential target columns
            for col in data.columns:
                if data[col].dtype == 'object':
                    unique_ratio = data[col].nunique() / len(data)
                    if 0.02 <= unique_ratio <= 0.3:  # Good target candidate ratio
                        analysis['target_suggestions'].append({
                            'column': col,
                            'reasoning': f'Categorical column with {data[col].nunique()} unique values - good for classification'
                        })
                elif data[col].dtype in ['int64', 'float64']:
                    if data[col].nunique() < 20:  # Discrete numeric - classification
                        analysis['target_suggestions'].append({
                            'column': col,
                            'reasoning': f'Numeric column with {data[col].nunique()} unique values - could be classification target'
                        })
                    else:  # Continuous - regression
                        analysis['target_suggestions'].append({
                            'column': col,
                            'reasoning': f'Continuous numeric column - suitable for regression target'
                        })
            
            # Check for quality issues
            if missing_data > 0:
                missing_pct = (missing_data / (n_rows * n_cols)) * 100
                if missing_pct > 10:
                    analysis['quality_issues'].append(f'Dataset has {missing_pct:.1f}% missing data - consider imputation')
            
            # Check for high cardinality
            for col in categorical_cols:
                if data[col].nunique() > n_rows * 0.8:
                    analysis['quality_issues'].append(f'Column {col} has very high cardinality - consider encoding strategies')
            
            # Check for single-value columns
            for col in data.columns:
                if data[col].nunique() == 1:
                    analysis['quality_issues'].append(f'Column {col} has only one unique value - consider removing')
            
            return analysis
            
        except Exception as e:
            logger.error(f"Basic dataset analysis failed: {e}")
            return {'task_recommendation': {'task': 'Analysis', 'confidence': 0.5, 'reasoning': 'Basic analysis completed'}}
    
    def _display_enhanced_ai_insights(self, insights):
        """Display enhanced AI insights in organized format."""
        if not insights:
            st.info("No AI insights available.")
            return
            
        # Dataset Overview
        if 'dataset_overview' in insights:
            st.markdown("#### 📊 **Dataset Overview**")
            overview = insights['dataset_overview']
            if isinstance(overview, dict):
                col1, col2 = st.columns(2)
                with col1:
                    if 'summary' in overview:
                        st.write(overview['summary'])
                with col2:
                    if 'recommendations' in overview:
                        for rec in overview['recommendations'][:3]:
                            st.write(f"• {rec}")
            else:
                st.write(overview)
        
        # Key Strengths
        if 'key_strengths' in insights:
            st.markdown("#### ✅ **Dataset Strengths**")
            strengths = insights['key_strengths']
            if isinstance(strengths, list):
                for strength in strengths:
                    st.success(f"✓ {strength}")
            else:
                st.success(strengths)
        
        # Critical Challenges
        if 'critical_challenges' in insights:
            st.markdown("#### ⚠️ **Areas for Improvement**")
            challenges = insights['critical_challenges']
            if isinstance(challenges, list):
                for challenge in challenges:
                    st.warning(f"⚠ {challenge}")
            else:
                st.warning(challenges)
        
        # Preprocessing Strategy
        if 'preprocessing_strategy' in insights:
            st.markdown("#### 🔧 **Recommended Preprocessing**")
            strategy = insights['preprocessing_strategy']
            if isinstance(strategy, list):
                for step in strategy:
                    st.info(f"🔧 {step}")
            else:
                st.info(strategy)
        
        # Model Recommendations
        if 'recommended_models' in insights:
            st.markdown("#### 🤖 **Recommended Models**")
            models = insights['recommended_models']
            if isinstance(models, list):
                for model in models:
                    st.write(f"🎯 {model}")
            else:
                st.write(models)
        
        # Features overview
        st.markdown("---")
        st.markdown("### 🔧 **Advanced Features**")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("""
            **🤖 Automated Machine Learning:**
            - Multi-model comparison (7+ algorithms)
            - Automatic hyperparameter optimization
            - Cross-validation with confidence intervals
            - Ensemble model creation
            
            **📊 Professional Analysis:**
            - Statistical data profiling
            - Feature importance analysis
            - Model performance visualization
            - Overfitting detection
            """)
            
        with feature_col2:
            st.markdown("""
            **🧠 AI-Powered Insights:**
            - Dataset quality assessment
            - Model recommendation engine
            - Configuration optimization
            - Intelligent explanations
            
            **🔍 Explainability & Reports:**
            - SHAP value analysis
            - Feature impact visualization
            - Comprehensive PDF reports
            - Executive summary generation
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
            
            # Class filtering for classification
            if task_type == "Classification":
                st.sidebar.subheader("🎯 Class Filtering")
                
                enable_class_filter = st.sidebar.checkbox(
                    "Auto-remove rare classes",
                    value=False,
                    help="Automatically remove classes with too few samples before training"
                )
                st.session_state.enable_class_filter = enable_class_filter
                
                if enable_class_filter:
                    min_class_samples = st.sidebar.number_input(
                        "Min samples per class",
                        min_value=2,
                        max_value=100,
                        value=5,
                        help="Classes with fewer samples will be removed automatically"
                    )
                    st.session_state.min_class_samples = min_class_samples
                else:
                    st.session_state.min_class_samples = 2  # Default minimum for CV
            
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
                
                # Configuration link
                st.sidebar.markdown("---")
                if st.sidebar.button("⚙️ Advanced Configuration", use_container_width=True):
                    st.session_state.show_configuration = True
                    st.rerun()
            
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
    
    # Google Colab support removed for simplicity - focusing on Local + Remote Jupyter only
    
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
        st.info("🔍 DEBUG: run_automl() started")
        
        try:
            data = st.session_state.data
            task_type = st.session_state.task_type
            
            st.info(f"🔍 DEBUG: Task type: {task_type}, Data shape: {data.shape}")
            st.info(f"⚙️ Using your configured random seed: {st.session_state.random_seed}")
            st.info(f"⚙️ Dimensionality reduction: {st.session_state.get('dimred_enabled', 'auto')} mode")
            
            # Create dimensionality reduction config from UI (moved to top for scope)
            dimred_config = DimRedConfig(
                enable=st.session_state.get('dimred_enabled', 'auto'),
                method=st.session_state.get('dimred_method', 'auto'),
                variance_target=st.session_state.get('dimred_variance_target', 0.95),
                k_max=st.session_state.get('dimred_k_max', 256),
                whiten=True,
                seed=st.session_state.random_seed  # Use actual user-configured value
            )
            
            # Use user-configured max_features or recommended default
            max_features = st.session_state.get('recommended_config', {}).get('recommended_max_features', 1000)
            
            # Override with user's explicit configuration if available
            user_max_features = st.session_state.get('dataset_config', {}).get('max_features')
            if user_max_features:
                max_features = user_max_features
                st.info(f"⚙️ Using your configured max features: {max_features}")
            else:
                st.info(f"⚙️ Using recommended max features: {max_features}")
            
            # Profile data
            st.info("📊 Profiling dataset...")
            profiler = DataProfiler()
            
            st.info("🔍 DEBUG: DataProfiler created, starting classification check")
            
            if task_type == "Classification":
                target_col = st.session_state.target_col
                X = data.drop(columns=[target_col])
                y = data[target_col]
                
                # Check if target is actually continuous (regression problem)
                n_unique = y.nunique()
                n_samples = len(y)
                
                st.info(f"🔍 DEBUG: Target check - Unique values: {n_unique}, Samples: {n_samples}, Ratio: {n_unique/n_samples:.3f}")
                
                # CRITICAL: Check if target looks like an ID column
                if n_unique > 100 and n_unique > n_samples * 0.8:
                    st.error(f"❌ **Invalid Target Column Detected!**")
                    st.error(f"Target column has {n_unique:,} unique values out of {n_samples:,} samples.")
                    st.error(f"This appears to be an **ID column**, not a classification target!")
                    st.warning("💡 **Solution**: Select a different column with 2-20 unique categories (e.g., 'diagnosis', 'class', 'label')")
                    st.session_state.results = {}
                    st.session_state.automl_error = f"Target column appears to be an ID column with {n_unique} unique values"
                    return
                
                # If >50% unique values and they're numeric, it's likely regression
                import pandas as pandas_api
                if n_unique / n_samples > 0.5 and pandas_api.api.types.is_numeric_dtype(y):
                    st.warning(f"⚠️ **Potential Task Type Issue Detected!**")
                    st.warning(f"Your target has {n_unique:,} unique continuous values out of {n_samples:,} samples.")
                    st.warning(f"This looks like it might be a **REGRESSION** problem, not classification!")
                    st.info("💡 **Continuing with classification anyway**. Consider changing 'Task Type' to 'Regression' for better results.")
                    
                    # Show sample values but DON'T return - continue with classification
                    st.info(f"📊 Sample target values: {list(y.head(10).values)}")
                
                # Check class distribution BEFORE preprocessing
                from collections import Counter
                class_counts_before = Counter(y)
                
                # If too many classes, show warning
                if n_unique > 50:
                    st.warning(f"⚠️ High number of classes detected: {n_unique}")
                    # Show class range instead of specific values for many classes
                    min_class = min(class_counts_before.keys()) if class_counts_before.keys() else 0
                    max_class = max(class_counts_before.keys()) if class_counts_before.keys() else 0
                    st.info(f"📊 Class range: {min_class} to {max_class} (showing stats instead of all values)")
                elif n_unique <= 20:
                    st.info(f"📊 Original class distribution: {dict(class_counts_before)}")
                else:
                    # For 21-50 classes, show just the counts
                    st.info(f"📊 Classes: {n_unique} total (range: {min(class_counts_before.keys())} to {max(class_counts_before.keys())})")
                
                profile = profiler.profile_dataset(X, y)
            else:
                X = data
                y = None
                profile = profiler.profile_dataset(X)
            
            st.session_state.profiler = profiler
            st.session_state.profile = profile
            
            # Preprocess with smart feature selection and dimred
            with st.spinner("🔧 Preprocessing data..."):
                preprocessor = DataPreprocessor(
                    max_features=max_features,
                    dimred_config=dimred_config
                )
                X_processed, y_processed = preprocessor.fit_transform(X, y)
            
            # Check class distribution AFTER preprocessing
            if task_type == "Classification":
                class_counts_after = Counter(y_processed)
                unique_classes_after = len(class_counts_after)
                total_samples_after = len(y_processed)
                
                # Critical: Check if train-test split is possible
                min_class_count = min(class_counts_after.values())
                classes_with_one_sample = sum(1 for count in class_counts_after.values() if count == 1)
                
                if min_class_count < 2:
                    st.error(f"""
🚨 **Cannot Proceed with AutoML - Insufficient Samples per Class**

**After preprocessing:** {unique_classes_after:,} classes, {total_samples_after:,} samples

**Problem:** {classes_with_one_sample:,} classes have only 1 sample each.
**Requirement:** Each class needs ≥2 samples for train-test split.

**Why this happened:**
- Original data had many rare classes
- Preprocessing may have filtered some samples
- Some classes became even rarer

**Solutions:**
1. **Aggregate rare classes**: Group similar classes together
2. **Filter minimum samples**: Remove classes with <5 samples  
3. **Check if regression**: Is this actually a continuous prediction problem?
4. **Collect more data**: Increase samples for rare classes

**Current Class Distribution (worst 10):**
{dict(sorted(class_counts_after.items(), key=lambda x: x[1])[:10])}
                    """)
                    return
                
                # Check if test split will be too large
                test_size = 0.25
                expected_test_samples = int(total_samples_after * test_size)
                
                if expected_test_samples >= unique_classes_after:
                    st.warning(f"""
⚠️ **Large number of classes detected**

- Classes: {unique_classes_after:,}
- Samples: {total_samples_after:,}  
- Expected test size: {expected_test_samples:,}

This may cause slow training. Consider filtering to top frequent classes.
                    """)
                
                # Only show distribution details if reasonable number of classes
                if unique_classes_after <= 20:
                    st.info(f"📊 Final class distribution: {dict(class_counts_after)}")
                elif unique_classes_after <= 100:
                    st.info(f"📊 Classes: {unique_classes_after} total, samples per class: {min_class_count}-{max(class_counts_after.values())}")
                else:
                    st.info(f"📊 Classes: {unique_classes_after:,} total, min samples per class: {min_class_count}")
                
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
                # Auto-enable class filtering for high-cardinality datasets
                unique_classes_count = len(set(y_processed))
                total_samples_count = len(y_processed)
                
                # Auto-enable filtering if we have too many classes
                auto_filter_needed = (
                    unique_classes_count > 1000 or  # More than 1000 classes
                    unique_classes_count > total_samples_count * 0.1  # More than 10% unique classes
                )
                
                if auto_filter_needed and not hasattr(st.session_state, 'enable_class_filter'):
                    st.session_state.enable_class_filter = True
                    st.session_state.min_class_samples = max(2, total_samples_count // unique_classes_count)
                    st.warning(f"""
🛠️ **Auto-Filter Enabled**
                    
Detected high-cardinality target ({unique_classes_count:,} classes for {total_samples_count:,} samples).
Auto-filtering classes with <{st.session_state.min_class_samples} samples to prevent train-test split errors.
                    """)
                
                # Apply proactive class filtering if enabled
                if hasattr(st.session_state, 'enable_class_filter') and st.session_state.enable_class_filter:
                    min_samples = st.session_state.get('min_class_samples', 5)
                    original_class_counts = Counter(y_processed)
                    
                    # Filter out classes with insufficient samples
                    valid_classes = {class_label: count for class_label, count in original_class_counts.items() 
                                   if count >= min_samples}
                    
                    if len(valid_classes) >= 2:  # Need at least 2 classes
                        # Filter the data - FIX: Use np.isin for numpy arrays
                        if isinstance(y_processed, pd.Series):
                            mask = y_processed.isin(valid_classes.keys())
                        else:
                            mask = np.isin(y_processed, list(valid_classes.keys()))
                        
                        X_filtered = X_processed[mask]
                        y_filtered = y_processed[mask]
                        
                        # Update the processed data
                        X_processed = X_filtered
                        y_processed = y_filtered
                        st.session_state.X_processed = X_processed
                        st.session_state.y_processed = y_processed
                        
                        # Show filtering results
                        removed_classes = len(original_class_counts) - len(valid_classes)
                        if removed_classes > 0:
                            st.success(f"🛠️ **Auto-Filter Applied:** Removed {removed_classes:,} rare classes with <{min_samples} samples")
                            st.info(f"📊 **Filtered Dataset:** {len(valid_classes):,} classes, {len(X_processed):,} samples remaining")
                    else:
                        st.error(f"❌ After filtering, only {len(valid_classes)} classes would remain. Disabling filter.")
                        st.session_state.enable_class_filter = False
                
                # Check if stratification is possible (each class has at least 2 samples)
                class_counts = Counter(y_processed)
                min_class_count = min(class_counts.values())
                use_stratify = min_class_count >= 2
                
                if use_stratify:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y_processed,
                        test_size=0.3,  # 30% holdout for testing
                        stratify=y_processed,
                        random_state=st.session_state.random_seed  # Use user preference
                    )
                else:
                    st.warning(f"⚠️ Some classes have only 1 sample. Using random split instead of stratified split.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y_processed,
                        test_size=0.3,
                        random_state=st.session_state.random_seed  # Use user preference
                    )
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                # Create raw data splits with same random state for DimRedEvaluator
                # IMPORTANT: Use the same class filter logic for raw data if classes were filtered
                X_raw_for_split = X
                y_raw_for_split = y
                
                # If we filtered classes in processed data, filter raw data too
                if hasattr(st.session_state, 'enable_class_filter') and st.session_state.enable_class_filter:
                    # Use the same valid classes from the processed data
                    processed_classes = set(y_processed)
                    if isinstance(y, pd.Series):
                        raw_mask = y.isin(processed_classes)
                    else:
                        raw_mask = np.isin(y, list(processed_classes))
                    X_raw_for_split = X[raw_mask]
                    y_raw_for_split = y[raw_mask]
                
                # Check if raw data stratification is possible
                raw_class_counts = Counter(y_raw_for_split)
                raw_min_class_count = min(raw_class_counts.values()) if raw_class_counts else 0
                raw_use_stratify = raw_min_class_count >= 2
                
                if raw_use_stratify:
                    X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
                        X_raw_for_split, y_raw_for_split,
                        test_size=0.3,
                        stratify=y_raw_for_split,
                        random_state=st.session_state.random_seed  # Use user preference
                    )
                else:
                    X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
                        X_raw_for_split, y_raw_for_split,
                        test_size=0.3,
                        random_state=st.session_state.random_seed  # Use user preference
                    )
                
                st.info(f"📊 Split: Train={len(X_train)} samples, Test={len(X_test)} samples (30% holdout)")
            
            # Train models
            if task_type == "Classification":
                st.info("🚀 Starting Classification...")
                st.info("🔍 DEBUG: About to call run_classification()")
                self.run_classification(X_train, y_train, X_test, y_test, dimred_config, preprocessor, X_raw_train, y_raw_train)
                st.info("🔍 DEBUG: run_classification() completed")
            else:
                st.info("🚀 Starting Clustering...")
                st.info("🔍 DEBUG: About to call run_clustering()")
                self.run_clustering(X_processed, preprocessor, dimred_config)
                st.info("🔍 DEBUG: run_clustering() completed")
            
            st.success("✅ AutoML pipeline completed!")
            
        except Exception as e:
            st.error(f"Error running AutoML: {e}")
            logger.error(f"AutoML error: {e}", exc_info=True)
    
    def run_professional_automl(self, optimization_time_minutes=15, max_trials=100, 
                              include_ensemble=True, advanced_features=None):
        """
        Run Professional AutoML with advanced hyperparameter optimization.
        AI Engineer-level approach with Optuna optimization.
        
        Args:
            optimization_time_minutes: Time limit for hyperparameter optimization
            max_trials: Maximum trials per model
            include_ensemble: Whether to create ensemble models
            advanced_features: List of advanced features to enable
        """
        if advanced_features is None:
            advanced_features = []
        
        # Clear any existing session state that might cause conflicts
        if hasattr(st.session_state, 'professional_results'):
            delattr(st.session_state, 'professional_results')
        
        try:
            data = st.session_state.data
            task_type = st.session_state.task_type
            
            # Get basic task mapping
            if task_type == "Classification":
                ml_task = "classification"
                target_col = st.session_state.target_col
                X = data.drop(columns=[target_col])
                y = data[target_col]
                
                # Check for regression disguised as classification
                n_unique = y.nunique()
                n_samples = len(y)
                
                if n_unique / n_samples > 0.5:
                    import pandas as pandas_api
                    if pandas_api.api.types.is_numeric_dtype(y):
                        st.error("❌ **Wrong Task Type!** This appears to be regression, not classification.")
                        return
                        
            elif task_type == "Regression":
                ml_task = "regression"
                target_col = st.session_state.target_col
                X = data.drop(columns=[target_col])
                y = data[target_col]
            else:
                ml_task = "clustering"
                X = data
                y = None
            
            # Professional preprocessing with advanced feature engineering
            st.info("🔧 Professional preprocessing with advanced feature engineering...")
            
            # Create dimensionality reduction config
            dimred_config = DimRedConfig(
                enable='auto',
                method='auto',
                variance_target=0.95,
                k_max=256,
                whiten=True,
                seed=st.session_state.random_seed
            )
            
            preprocessor = DataPreprocessor(
                max_features=1000,
                dimred_config=dimred_config
            )
            X_processed, y_processed = preprocessor.fit_transform(X, y)
            
            # Initialize Professional AutoML Pipeline
            st.info("🤖 Initializing Professional AutoML Pipeline...")
            
            professional_pipeline = AutoMLPipeline(
                task_type=ml_task,
                optimization_time_minutes=optimization_time_minutes,
                random_state=st.session_state.random_seed
            )
            
            # Get model candidates based on task type and dataset size
            n_features = X_processed.shape[1] if hasattr(X_processed, 'shape') else len(X_processed.columns)
            model_candidates = self._get_professional_model_candidates(
                ml_task, len(X_processed), n_features
            )
            
            # Display optimization progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Analyzing dataset characteristics...")
            progress_bar.progress(0.1)
            
            # Advanced dataset analysis
            dataset_stats = self._analyze_dataset_professionally(X_processed, y_processed, ml_task)
            
            # Show dataset insights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Samples", f"{len(X_processed):,}")
            with col2:
                n_features = X_processed.shape[1] if hasattr(X_processed, 'shape') else len(X_processed.columns)
                st.metric("📐 Features", n_features)
            with col3:
                if ml_task != 'clustering':
                    unique_targets = len(np.unique(y_processed))
                    st.metric("🎯 Classes/Range" if ml_task == 'classification' else "🎯 Target Range", 
                             f"{unique_targets:,}" if ml_task == 'classification' else f"{y_processed.min():.2f}-{y_processed.max():.2f}")
            
            # Professional optimization with progress tracking
            status_text.text("⚙️ Running hyperparameter optimization...")
            progress_bar.progress(0.3)
            
            # Record start time for optimization tracking
            optimization_start_time = time.time()
            
            # Run professional AutoML pipeline
            results = professional_pipeline.run_advanced_automl(
                X_processed, y_processed, 
                model_candidates=model_candidates,
                include_ensemble=include_ensemble and ml_task != 'clustering'
            )
            
            # Validate results before proceeding
            if results is None:
                st.error("❌ Professional AutoML failed to generate results")
                progress_bar.empty()
                status_text.empty()
                return
            
            # Check if optimization was successful
            if not results.get('individual_models'):
                st.warning("⚠️ No models were successfully optimized. Using fallback results.")
                # Create minimal results for display
                results = {
                    'individual_models': {'RandomForest': {'best_score': -1000, 'optimization_failed': True}},
                    'ensemble_models': None,
                    'optimization_summary': 'Optimization failed - using fallback mode',
                    'dataset_info': {'n_samples': len(X_processed), 'n_features': X_processed.shape[1], 'task_type': ml_task}
                }
                
            progress_bar.progress(0.9)
            status_text.text("📊 Finalizing results...")
            
            # Store results in session state for display on Results page
            st.session_state.professional_results = results
            st.session_state.professional_pipeline = professional_pipeline
            st.session_state.dataset_stats = dataset_stats
            st.session_state.optimization_time = time.time() - optimization_start_time
            
            # Final completion steps
            progress_bar.progress(1.0)
            status_text.text("✅ Professional AutoML completed!")
            time.sleep(0.5)  # Brief pause to show completion
            
            # Clear progress indicators BEFORE navigation
            progress_bar.empty()
            status_text.empty()
            
            # Navigate to Results page
            st.session_state.app_stage = 'results'
            st.success("🏆 **Professional AutoML Pipeline Complete!**")
            st.info("📊 **Redirecting to Results page...**")
            
            # Force rerun to navigate to results
            st.rerun()
            
        except Exception as e:
            # Enhanced error handling with cleanup
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()
                
            st.error(f"❌ Professional AutoML Error: {str(e)}")
            
            # Provide detailed error information for troubleshooting
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'dataset_shape': f"{data.shape[0]}x{data.shape[1]}" if 'data' in locals() else 'Unknown',
                'task_type': task_type if 'task_type' in locals() else 'Unknown'
            }
            
            with st.expander("🔍 Error Details (for troubleshooting)"):
                st.json(error_details)
                
            import traceback
            st.error(f"Technical Details: {traceback.format_exc()}")
            
            # Reset session state to prevent stuck state
            if hasattr(st.session_state, 'professional_results'):
                delattr(st.session_state, 'professional_results')
            
            st.info("💡 **Suggested Actions:**")
            st.write("1. Try reducing the dataset size")
            st.write("2. Check for data quality issues")  
            st.write("3. Use Standard AutoML mode instead")
            st.write("4. Ensure all required packages are installed")
    
    def _get_professional_model_candidates(self, task_type, n_samples, n_features):
        """Get professional model candidates with intelligent selection."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.mixture import GaussianMixture
        
        try:
            from xgboost import XGBClassifier, XGBRegressor
            xgb_available = True
        except ImportError:
            xgb_available = False
        
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            lgb_available = True
        except ImportError:
            lgb_available = False
        
        if task_type == "classification":
            candidates = [
                ('RandomForest', RandomForestClassifier(random_state=st.session_state.random_seed)),
                ('LogisticRegression', LogisticRegression(random_state=st.session_state.random_seed, max_iter=1000))
            ]
            
            # Add advanced models if available
            if xgb_available:
                candidates.append(('XGBoost', XGBClassifier(random_state=st.session_state.random_seed, eval_metric='logloss')))
            if lgb_available:
                candidates.append(('LightGBM', LGBMClassifier(random_state=st.session_state.random_seed, verbose=-1)))
            
            # Add SVM for smaller datasets
            if n_samples <= 10000:
                candidates.append(('SVM', SVC(random_state=st.session_state.random_seed, probability=True)))
            
            # Add MLP for appropriate dataset sizes
            if 1000 <= n_samples <= 50000:
                candidates.append(('MLP', MLPClassifier(random_state=st.session_state.random_seed, max_iter=1000)))
            
            # Add KNN for smaller datasets
            if n_samples <= 20000:
                candidates.append(('KNN', KNeighborsClassifier()))
                
        elif task_type == "regression":
            candidates = [
                ('RandomForest', RandomForestRegressor(random_state=st.session_state.random_seed)),
                ('LinearRegression', LinearRegression())
            ]
            
            if xgb_available:
                candidates.append(('XGBoost', XGBRegressor(random_state=st.session_state.random_seed)))
            if lgb_available:
                candidates.append(('LightGBM', LGBMRegressor(random_state=st.session_state.random_seed, verbose=-1)))
            
            if n_samples <= 10000:
                candidates.append(('SVR', SVR()))
            
            if 1000 <= n_samples <= 50000:
                candidates.append(('MLP', MLPRegressor(random_state=st.session_state.random_seed, max_iter=1000)))
            
            if n_samples <= 20000:
                candidates.append(('KNN', KNeighborsRegressor()))
                
        else:  # clustering
            candidates = [
                ('KMeans', KMeans(random_state=st.session_state.random_seed)),
                ('GaussianMixture', GaussianMixture(random_state=st.session_state.random_seed))
            ]
            
            # Add DBSCAN for smaller datasets
            if n_samples <= 10000:
                candidates.append(('DBSCAN', DBSCAN()))
        
        return candidates
    
    def _analyze_dataset_professionally(self, X, y, task_type):
        """Professional dataset analysis for optimization insights."""
        # Handle both pandas DataFrames and numpy arrays
        if hasattr(X, 'memory_usage'):
            # DataFrame case
            memory_mb = X.memory_usage(deep=True).sum() / 1024 / 1024
            missing_values = X.isnull().sum().sum()
            numeric_features = X.select_dtypes(include=[np.number]).shape[1]
            categorical_features = X.select_dtypes(include=['object']).shape[1]
        else:
            # Numpy array case
            memory_mb = X.nbytes / 1024 / 1024
            missing_values = np.isnan(X).sum() if np.issubdtype(X.dtype, np.number) else 0
            numeric_features = X.shape[1]  # Assume all numeric for processed arrays
            categorical_features = 0
        
        stats = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'memory_usage_mb': memory_mb,
            'missing_values': missing_values,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features
        }
        
        if task_type != 'clustering':
            stats['target_type'] = str(y.dtype)
            if task_type == 'classification':
                stats['n_classes'] = len(np.unique(y))
                stats['class_balance'] = (np.bincount(y) / len(y)).std()  # Higher = more imbalanced
            else:  # regression
                stats['target_range'] = y.max() - y.min()
                stats['target_std'] = y.std()
        
        # Data complexity analysis - handle both DataFrame and numpy array
        if hasattr(X, 'corr'):
            # DataFrame case
            stats['feature_correlation_avg'] = abs(X.corr()).mean().mean() if X.shape[1] > 1 else 0
            stats['sparsity'] = (X == 0).mean().mean()
        else:
            # Numpy array case
            if X.shape[1] > 1:
                # Calculate correlation matrix using numpy
                corr_matrix = np.corrcoef(X, rowvar=False)
                # Handle NaN values that might occur with constant features
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                # Get average absolute correlation (excluding diagonal)
                mask = np.ones_like(corr_matrix, dtype=bool)
                np.fill_diagonal(mask, False)
                stats['feature_correlation_avg'] = np.abs(corr_matrix[mask]).mean() if mask.sum() > 0 else 0
            else:
                stats['feature_correlation_avg'] = 0
            stats['sparsity'] = (X == 0).mean()
        
        return stats
    
    def _display_professional_results(self, results, advanced_features):
        """Display professional AutoML results with advanced insights."""
        st.header("🏆 Professional AutoML Results")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        individual_results = results['individual_models']
        n_models = len(individual_results)
        
        # Calculate total improvement
        total_improvement = sum(r['improvement'] for r in individual_results.values())
        avg_improvement = total_improvement / n_models if n_models > 0 else 0
        
        # Get best model
        best_model_info = None
        best_score = -float('inf')
        
        for name, result in individual_results.items():
            if result['best_score'] > best_score:
                best_score = result['best_score']
                best_model_info = (name, result)
        
        with col1:
            st.metric("🤖 Models Optimized", n_models)
        with col2:
            st.metric("📈 Avg Improvement", f"{avg_improvement:.4f}")
        with col3:
            st.metric("🏆 Best Score", f"{best_score:.4f}")
        with col4:
            total_time = sum(r['optimization_time'] for r in individual_results.values())
            st.metric("⏱️ Total Time", f"{total_time:.1f}s")
        
        # Best Model Details
        if best_model_info:
            st.subheader(f"🥇 Best Model: {best_model_info[0]}")
            best_result = best_model_info[1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎯 Optimized Score", f"{best_result['best_score']:.4f}")
                st.metric("📊 Baseline Score", f"{best_result['baseline_score']:.4f}")
            with col2:
                st.metric("⬆️ Improvement", f"+{best_result['improvement']:.4f}")
                st.metric("📈 Improvement %", f"{best_result['improvement_percent']:+.1f}%")
            
            # Show best parameters
            st.subheader("⚙️ Optimized Parameters")
            best_params_df = pd.DataFrame([
                {'Parameter': param, 'Value': str(value)} 
                for param, value in best_result['best_params'].items()
            ])
            st.dataframe(best_params_df, use_container_width=True)
        
        # Model Comparison Table
        st.subheader("📊 Model Performance Comparison")
        
        comparison_data = []
        for name, result in individual_results.items():
            comparison_data.append({
                'Model': name,
                'Baseline Score': f"{result['baseline_score']:.4f}",
                'Optimized Score': f"{result['best_score']:.4f}",
                'Improvement': f"+{result['improvement']:.4f}",
                'Improvement %': f"{result['improvement_percent']:+.1f}%",
                'Time (s)': f"{result['optimization_time']:.1f}",
                'Trials': result['n_trials']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Ensemble Results
        if results['ensemble_models']:
            st.subheader("🎭 Ensemble Model Results")
            ensemble_result = results['ensemble_models']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎭 Ensemble Score", f"{ensemble_result['ensemble_score']:.4f}")
                st.metric("🥇 Best Individual", f"{ensemble_result['best_individual_score']:.4f}")
            with col2:
                ensemble_improvement = ensemble_result['ensemble_improvement']
                st.metric("🚀 Ensemble Boost", f"+{ensemble_improvement:.4f}")
                
                if ensemble_improvement > 0:
                    st.success("✅ Ensemble outperforms individual models!")
                else:
                    st.info("ℹ️ Individual models perform better")
        
        # Advanced Features Status
        if advanced_features:
            st.subheader("🔧 Advanced Features")
            for feature in advanced_features:
                if feature == "Early Stopping":
                    st.success("✅ Early Stopping: Enabled for faster convergence")
                elif feature == "Multi-objective":
                    st.info("🎯 Multi-objective: Balancing accuracy and efficiency")
                elif feature == "Automated Feature Engineering":
                    st.success("⚙️ Feature Engineering: Advanced preprocessing applied")
        
        # Optimization Summary
        st.subheader("📋 Optimization Summary")
        summary = results['optimization_summary']
        st.text(summary)
        
        # Professional Insights
        self._show_professional_insights(results)
    
    def _show_professional_insights(self, results):
        """Show professional AI engineer insights and recommendations."""
        st.subheader("🧠 AI Engineer Insights")
        
        individual_results = results['individual_models']
        dataset_info = results['dataset_info']
        
        insights = []
        
        # Performance insights
        best_model = max(individual_results.items(), key=lambda x: x[1]['best_score'])
        worst_model = min(individual_results.items(), key=lambda x: x[1]['best_score'])
        
        score_gap = best_model[1]['best_score'] - worst_model[1]['best_score']
        
        if score_gap > 0.1:
            insights.append(f"🎯 **Significant Model Differences**: {best_model[0]} outperforms {worst_model[0]} by {score_gap:.3f}. Consider ensemble approaches.")
        
        # Optimization insights
        high_improvement_models = [name for name, res in individual_results.items() if res['improvement_percent'] > 10]
        if high_improvement_models:
            insights.append(f"⚡ **High Optimization Impact**: {', '.join(high_improvement_models)} showed >10% improvement from hyperparameter tuning.")
        
        # Dataset-specific insights
        n_samples = dataset_info['n_samples']
        n_features = dataset_info['n_features']
        
        if n_samples < 1000:
            insights.append("⚠️ **Small Dataset Warning**: Consider regularization, cross-validation, and simpler models to avoid overfitting.")
        elif n_samples > 100000:
            insights.append("🚀 **Large Dataset Advantage**: Consider deep learning models and advanced ensemble methods for potential performance gains.")
        
        if n_features > n_samples:
            insights.append("📐 **High-Dimensional Data**: Feature selection and dimensionality reduction are critical. Consider L1 regularization.")
        
        # Model-specific recommendations
        if 'RandomForest' in individual_results and individual_results['RandomForest']['best_score'] > 0.8:
            insights.append("🌲 **Tree-Based Success**: Random Forest performs well. Consider XGBoost, LightGBM, or CatBoost for potential improvements.")
        
        if 'SVM' in individual_results and individual_results['SVM']['optimization_time'] > 60:
            insights.append("⏱️ **SVM Performance**: SVM optimization took significant time. For larger datasets, consider faster alternatives.")
        
        # Ensemble insights
        if results['ensemble_models'] and results['ensemble_models']['ensemble_improvement'] > 0:
            insights.append("🎭 **Ensemble Success**: Ensemble models show improvement. Consider stacking with meta-learners for advanced performance.")
        
        # Display insights
        for i, insight in enumerate(insights, 1):
            st.markdown(f"{i}. {insight}")
        
        if not insights:
            st.info("💡 All models performed within expected ranges. Consider feature engineering or domain-specific preprocessing for further improvements.")
        
        # Professional recommendations
        st.subheader("🔧 Next Steps - Professional ML Engineering")
        
        recommendations = [
            "**Feature Engineering**: Create domain-specific features, polynomial interactions, or time-based features",
            "**Advanced Validation**: Implement time-series aware splits, group-based validation, or stratified sampling",
            "**Model Calibration**: Apply Platt scaling or isotonic regression for probability calibration", 
            "**Uncertainty Quantification**: Implement prediction intervals or confidence estimation",
            "**Production Optimization**: Consider model compression, quantization, or knowledge distillation",
            "**Monitoring Setup**: Implement drift detection, performance monitoring, and automated retraining"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Advanced techniques section
        with st.expander("🎓 Advanced ML Engineering Techniques"):
            st.markdown("""
            **For Classification:**
            - Multi-label classification with label powerset or binary relevance
            - Cost-sensitive learning for imbalanced datasets
            - Conformal prediction for uncertainty quantification
            
            **For Regression:**
            - Multi-output regression with target transformations
            - Quantile regression for risk assessment
            - Bayesian optimization for expensive function approximation
            
            **For Clustering:**
            - Hierarchical clustering with optimal number selection
            - Density-based clustering for arbitrary shapes
            - Semi-supervised clustering with constraints
            
            **Cross-Domain:**
            - Transfer learning from pre-trained models
            - Meta-learning for few-shot scenarios
            - Automated machine learning (AutoML) pipelines
            """)
    
    def run_classification(self, X_train, y_train, X_test, y_test, dimred_config, preprocessor, X_raw_train, y_raw_train):
        """Run classification pipeline with proper train/test split."""
        st.info("🤖 Training classification models on training set...")
        
        # DEBUG: Track method entry
        st.info("🔍 DEBUG: run_classification() started")
        
        # SMART MODEL SELECTION: Use fast models for large datasets
        total_samples = len(y_train)
        
        # Get models with adaptive settings based on dataset size
        models = get_supervised_models(
            random_state=st.session_state.random_seed,  # Use user preference
            n_samples=len(X_train)  # Pass dataset size for optimization
        )
        
        # DEBUG: Show all available models before filtering
        st.info(f"🔍 DEBUG: All available models from get_supervised_models(): {list(models.keys())}")
        
        # Check if SVM should be available for this dataset size
        if total_samples <= 20000:
            st.info(f"🔍 DEBUG: Dataset size ({total_samples}) allows SVM (threshold: 20000)")
        else:
            st.info(f"🔍 DEBUG: Dataset size ({total_samples}) would normally exclude SVM")
        
        # Apply user's model selection if configured
        selected_models = st.session_state.get('selected_models')
        if selected_models:
            st.info(f"🔍 DEBUG: User selected models: {selected_models}")
            
            # Map UI model names to actual implementation names
            model_mapping = {
                'SVM': ['LinearSVM', 'RBF-SVM'],  # UI shows 'SVM', but we have 'LinearSVM' and 'RBF-SVM'
                'SVR': ['LinearSVR', 'RBF-SVR']   # Similar for regression
            }
            
            # Expand selection to include all variants
            expanded_selection = []
            for selected in selected_models:
                if selected in model_mapping:
                    expanded_selection.extend(model_mapping[selected])
                else:
                    expanded_selection.append(selected)
            
            st.info(f"🔍 DEBUG: Expanded selection: {expanded_selection}")
            
            # Filter models to only include user-selected ones (or their variants)
            models = {name: model for name, model in models.items() if name in expanded_selection}
            st.info(f"⚙️ Using your selected models: {list(models.keys())}")
            
            # Check if any selected models are missing
            missing_models = [m for m in expanded_selection if m not in models]
            if missing_models:
                st.warning(f"⚠️ Some selected models are not available: {missing_models}")
        else:
            st.info(f"⚙️ Using all available models: {list(models.keys())}")
        
        # DEBUG: Check models loaded
        st.info(f"🔍 DEBUG: Loaded {len(models)} models: {list(models.keys())}")
        
        if total_samples > 20000:
            # Large dataset: Remove slow SVM models
            st.warning(f"⚡ **Large Dataset Detected** ({total_samples:,} samples)")
            st.info("🚀 Using **Fast Models Only** (LogReg, RF, XGBoost, MLP). SVMs skipped (too slow).")
            models = {k: v for k, v in models.items() if 'SVM' not in k}
        
        # Determine appropriate CV strategy based on data size
        from collections import Counter
        class_counts = Counter(y_train)  # Use training set only
        min_class_count = min(class_counts.values())
        
        # DEBUG: Class distribution check
        st.info(f"🔍 DEBUG: Class counts - Total classes: {len(class_counts)}, Min samples per class: {min_class_count}")
        
        # Check if dataset is too small for CV
        if min_class_count < 2:
            st.error(f"❌ Dataset has a class with only {min_class_count} sample(s). Each class needs at least 2 samples for cross-validation.")
            # Show concise class distribution summary instead of full dict
            classes_with_one = sum(1 for count in class_counts.values() if count == 1)
            classes_with_few = sum(1 for count in class_counts.values() if count < 5)
            st.info(f"📊 Class summary: {len(class_counts)} total classes, {classes_with_one} with 1 sample, {classes_with_few} with <5 samples")
            
            # Create a unique key for the button to avoid caching issues
            button_key = f"auto_fix_button_{len(class_counts)}_{classes_with_one}"
            
            # Offer automatic class filtering
            st.warning("💡 **Solution Options:**")
            col1, col2 = st.columns(2)
            
            with col1:
                min_samples_threshold = st.number_input(
                    "🎯 Minimum samples per class:", 
                    min_value=2, max_value=50, value=5,
                    key=f"threshold_input_{len(class_counts)}",
                    help="Classes with fewer samples will be removed"
                )
            
            with col2:
                if st.button("🛠️ Auto-Fix Dataset", type="primary", key=button_key):
                    with st.spinner("🔄 Applying class filter..."):
                        # Filter out classes with insufficient samples
                        valid_classes = {class_label: count for class_label, count in class_counts.items() 
                                       if count >= min_samples_threshold}
                        
                        if len(valid_classes) < 2:
                            st.error(f"❌ After filtering, only {len(valid_classes)} classes remain. Need at least 2 classes.")
                            st.stop()  # Stop execution here
                        
                        # Filter the training data - FIX: Use np.isin for numpy arrays
                        if isinstance(y_train, pd.Series):
                            mask_train = y_train.isin(valid_classes.keys())
                        else:
                            mask_train = np.isin(y_train, list(valid_classes.keys()))
                        
                        X_train_filtered = X_train[mask_train]
                        y_train_filtered = y_train[mask_train]
                        
                        # Filter the test data - FIX: Use np.isin for numpy arrays
                        if isinstance(y_test, pd.Series):
                            mask_test = y_test.isin(valid_classes.keys())
                        else:
                            mask_test = np.isin(y_test, list(valid_classes.keys()))
                        X_test_filtered = X_test[mask_test]
                        y_test_filtered = y_test[mask_test]
                        
                        # CRITICAL: Update session state with filtered data
                        st.session_state.X_train = X_train_filtered
                        st.session_state.X_test = X_test_filtered
                        st.session_state.y_train = y_train_filtered
                        st.session_state.y_test = y_test_filtered
                        
                        # Also update the processed data in session state
                        from sklearn.model_selection import train_test_split
                        X_processed_combined = pd.concat([X_train_filtered, X_test_filtered])
                        y_processed_combined = pd.concat([y_train_filtered, y_test_filtered])
                        st.session_state.X_processed = X_processed_combined
                        st.session_state.y_processed = y_processed_combined
                        
                        # Show results
                        removed_classes = len(class_counts) - len(valid_classes)
                        removed_samples_train = len(X_train) - len(X_train_filtered)
                        removed_samples_test = len(X_test) - len(X_test_filtered)
                        
                        st.success(f"✅ **Dataset Auto-Fixed!**")
                        st.info(f"📊 **Results:**\n"
                               f"- Removed {removed_classes:,} classes with <{min_samples_threshold} samples\n"
                               f"- Kept {len(valid_classes):,} classes\n" 
                               f"- Removed {removed_samples_train:,} training samples ({removed_samples_train/(len(X_train)+0.001)*100:.1f}%)\n"
                               f"- Removed {removed_samples_test:,} test samples ({removed_samples_test/(len(X_test)+0.001)*100:.1f}%)\n"
                               f"- New training size: {len(X_train_filtered):,} samples\n"
                               f"- New test size: {len(X_test_filtered):,} samples")
                        
                        st.success("🔄 **Page will refresh to continue with filtered data...**")
                        time.sleep(2)  # Give user time to read the results
                        st.rerun()  # This will restart and use the updated session state data
                    
            st.info("📌 **Manual alternatives:**\n"
                   "- Remove rare classes from your data before upload\n"
                   "- Combine similar classes into broader categories\n"
                   "- Use clustering instead of classification")
            
            # AUTO-FIX: Remove classes with 1 sample and continue
            st.warning("🔄 **Auto-applying fix to continue AutoML...**")
            valid_classes = {class_label: count for class_label, count in class_counts.items() if count >= 2}
            
            if len(valid_classes) < 2:
                st.error(f"❌ After auto-filtering, only {len(valid_classes)} classes remain. Cannot continue.")
                return
                
            # Apply the filter automatically
            if isinstance(y_train, pd.Series):
                mask_train = y_train.isin(valid_classes.keys())
                mask_test = y_test.isin(valid_classes.keys())
            else:
                mask_train = np.isin(y_train, list(valid_classes.keys()))
                mask_test = np.isin(y_test, list(valid_classes.keys()))
            
            X_train = X_train[mask_train]
            y_train = y_train[mask_train]
            X_test = X_test[mask_test]
            y_test = y_test[mask_test]
            
            removed_classes = len(class_counts) - len(valid_classes)
            st.success(f"✅ Auto-filtered: Removed {removed_classes} classes with <2 samples. Continuing with {len(valid_classes)} classes.")
            
            # Update class counts for the adaptive CV logic below
            class_counts = {k: v for k, v in class_counts.items() if k in valid_classes}
            min_class_count = min(class_counts.values())
            
            # DEBUG: After auto-fix
            st.info(f"🔍 DEBUG: After auto-fix - Classes: {len(class_counts)}, Min samples: {min_class_count}")

        # Get user-configured CV folds with safety constraints
        user_cv_folds = st.session_state.get('advanced_config', {}).get('validation', {}).get('cv_folds', 5)
        
        # Adaptive CV: Respect user preference but ensure data safety
        if min_class_count < 2:
            n_folds = 2
            n_repeats = 1
            st.warning(f"⚠️ Very small classes detected. Using minimum 2-fold CV for safety.")
        elif min_class_count < user_cv_folds:
            n_folds = min_class_count  # Can't have more folds than samples per class
            n_repeats = 1 if n_folds <= 3 else 2
            st.warning(f"⚠️ Small classes detected (min: {min_class_count}). Using {n_folds}-fold CV instead of your configured {user_cv_folds} folds.")
        else:
            n_folds = user_cv_folds  # Use user preference
            n_repeats = 1 if n_folds >= 10 else (2 if n_folds >= 5 else 3)
            st.info(f"⚙️ Using your configured {n_folds}-fold CV with {n_repeats} repeats")
        
        # Evaluate models with holdout set
        evaluator = ClassificationEvaluator(n_folds=n_folds, n_repeats=n_repeats)
        
        # DEBUG: Evaluator created
        st.info(f"🔍 DEBUG: Evaluator created with {n_folds}-fold CV, {n_repeats} repeats")
        
        # NEW: Dimensionality reduction evaluation
        if st.session_state.get('dimred_enabled') != 'off':
            st.info("📐 Evaluating dimensionality reduction impact...")
            dimred_evaluator = DimRedEvaluator(
                preprocessor=preprocessor,
                dimred_config=dimred_config,
                random_state=st.session_state.random_seed
            )
            
            # Run dimred comparison for representative models
            representative_models = {}
            for name, model in models.items():
                if any(key in name.lower() for key in ['logistic', 'random forest', 'xgboost']):
                    representative_models[name] = model
                if len(representative_models) >= 2:  # Test with 2-3 representative models
                    break
            
            dimred_results = dimred_evaluator.evaluate_classification_with_dimred(
                representative_models, X_raw_train, y_raw_train, task_type="classification"
            )
            
            # DEBUG: Show what's actually in dimred_results
            st.info(f"🔍 DEBUG: Dimred evaluation completed. Keys: {list(dimred_results.keys()) if dimred_results else 'None'}")
            if dimred_results:
                for key, value in dimred_results.items():
                    st.info(f"🔍 DEBUG: dimred_results['{key}'] type: {type(value)}")
                    if hasattr(value, 'explained_variance_ratio_'):
                        st.info(f"🔍 DEBUG: Found PCA-like object in '{key}' with {len(value.explained_variance_ratio_)} components")
            
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
        
        # DEBUG: Starting model training loop
        st.info(f"🔍 DEBUG: Starting training loop for {len(models)} models")
        
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
            st.session_state.results = {}
            st.session_state.automl_error = "All models failed to train"
            return
        
        st.session_state.results = results
        st.session_state.evaluator = evaluator
        st.session_state.models = {name: res['trained_model'] for name, res in results.items() if 'trained_model' in res}
        
        # Debug logging
        st.success(f"✅ Classification completed! Stored {len(results)} result sections.")
        st.info(f"🔍 Results keys: {list(results.keys())}")
        
        # Meta-learning recommendation
        st.info("🎯 Generating recommendations...")
        meta_selector = MetaModelSelector()
        recommendation = meta_selector.get_recommendation_with_rationale(
            st.session_state.profile,
            results
        )
        st.session_state.recommendation = recommendation
        
        # Transition to results stage
        st.session_state.app_stage = 'results'
        st.info(f"📊 Transitioning to results stage. Current stage: {st.session_state.app_stage}")
    
    def run_clustering(self, X, preprocessor, dimred_config):
        """Run clustering pipeline with optimizations for large datasets."""
        
        st.info("🤖 Training clustering models...")
        
        # PERFORMANCE OPTIMIZATION: Handle large datasets
        n_samples, n_features = X.shape
        max_samples_for_clustering = 50000  # Reasonable limit for clustering
        
        X_for_clustering = X
        use_sampling = False
        
        if n_samples > max_samples_for_clustering:
            use_sampling = True
            st.warning(f"""
⚡ **Large Dataset Optimization**

Your dataset has **{n_samples:,} samples** which is very large for clustering.

**Performance Optimization Applied:**
- Using **{max_samples_for_clustering:,} randomly sampled** points for clustering
- This maintains data distribution while ensuring reasonable computation time
- Final cluster assignments will be predicted for all data points
            """)
            
            # Stratified sampling to maintain data distribution
            np.random.seed(st.session_state.random_seed)
            sample_indices = np.random.choice(n_samples, max_samples_for_clustering, replace=False)
            X_for_clustering = X[sample_indices]
            
            st.info(f"📊 **Clustering Sample:** {X_for_clustering.shape[0]:,} samples × {X_for_clustering.shape[1]} features")
        
        # Get models with optimizations for dataset size
        models = get_clustering_models(st.session_state.random_seed, n_samples=n_samples)
        
        # Remove slow models for large datasets
        if n_samples > 20000:
            slow_models = ['DBSCAN', 'AgglomerativeClustering']
            models = {k: v for k, v in models.items() if k not in slow_models}
            st.info(f"🚀 **Fast Models Only:** Removed slow algorithms (DBSCAN, Agglomerative) for large dataset")
        
        # NEW: Hybrid dimensionality reduction evaluation for clustering
        if st.session_state.get('dimred_enabled') != 'off' and n_samples < 100000:
            st.info("📐 Smart evaluation: Dimensionality reduction impact on clustering...")
            
            # Progressive evaluation with time limits
            dimred_result = self._evaluate_dimred_hybrid(X_for_clustering, st.session_state.random_seed)
            
            # Store results
            st.session_state.dimred_results = dimred_result
            
            # Show recommendations
            if dimred_result.get('recommended'):
                st.success(f"✅ {dimred_result['recommendation']}")
                st.info(f"📊 {dimred_result['details']}")
            else:
                st.info(f"💡 {dimred_result['recommendation']}")
                if dimred_result.get('reason'):
                    st.caption(f"Reason: {dimred_result['reason']}")
                    
        else:
            if n_samples >= 100000:
                st.info("📐 Skipping dimensionality reduction evaluation for very large datasets")
            st.session_state.dimred_results = {
                'recommended': False,
                'recommendation': "Auto mode: Will decide per model",
                'method': 'auto'
            }
        
        # Evaluate models
        evaluator = ClusteringEvaluator()
        results = {}
        
        progress_bar = st.progress(0)
        for idx, (name, model) in enumerate(models.items()):
            st.text(f"Training {name}...")
            try:
                # Fit on sampled data
                labels_sample = model.fit_predict(X_for_clustering)
                
                # If we used sampling, predict on full dataset for final results
                if use_sampling:
                    try:
                        # For models that support predict
                        if hasattr(model, 'predict'):
                            labels = model.predict(X)
                        else:
                            # For models like DBSCAN that don't have predict, use the sampled results
                            labels = labels_sample
                    except:
                        # Fallback to sampled labels
                        labels = labels_sample
                else:
                    labels = labels_sample
                
                # Evaluate on the appropriate dataset
                eval_X = X if not use_sampling or hasattr(model, 'predict') else X_for_clustering
                result = evaluator.evaluate_model(model, eval_X, name, labels if not use_sampling or hasattr(model, 'predict') else labels_sample)
                
                results[name] = result
                results[name]['model'] = model
                
                if use_sampling and hasattr(model, 'predict'):
                    st.success(f"✅ {name}: Trained on {X_for_clustering.shape[0]:,} samples, evaluated on full {n_samples:,} samples")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                st.warning(f"⚠️ {name} failed: {str(e)[:100]}...")
            
            progress_bar.progress((idx + 1) / len(models))
        
        st.session_state.results = results
        st.session_state.evaluator = evaluator
        st.session_state.models = {name: res['model'] for name, res in results.items()}
        
        # Debug logging  
        st.success(f"✅ Clustering completed! Stored {len(results)} result sections.")
        st.info(f"🔍 Results keys: {list(results.keys())}")
        
        # Transition to results stage
        st.session_state.app_stage = 'results'
        st.info(f"📊 Transitioning to results stage. Current stage: {st.session_state.app_stage}")
    
    def _evaluate_dimred_hybrid(self, X, random_state, max_time_seconds=30):
        """
        Hybrid dimensionality reduction evaluation with progressive complexity.
        
        Args:
            X: Feature matrix for clustering
            random_state: Random seed
            max_time_seconds: Maximum time to spend on evaluation
            
        Returns:
            Dictionary with recommendation results
        """
        import time
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA
        
        start_time = time.time()
        n_samples = len(X)
        
        try:
            # Phase 1: Fast preliminary check (5 seconds max, 5K samples)
            sample_size_fast = min(5000, n_samples)
            if n_samples > sample_size_fast:
                indices_fast = np.random.choice(n_samples, sample_size_fast, replace=False)
                X_fast = X[indices_fast]
            else:
                X_fast = X
                
            # Quick 3-cluster test
            result_fast = self._evaluate_pca_clustering_phase(X_fast, random_state, phase="fast")
            
            elapsed = time.time() - start_time
            
            # If fast phase shows promise and we have time, do medium evaluation
            if (elapsed < 15 and 
                result_fast.get('improvement', 0) > 0.03 and 
                n_samples > 10000):
                
                # Phase 2: Medium evaluation (remaining time, 20K samples)
                sample_size_medium = min(20000, n_samples)
                indices_medium = np.random.choice(n_samples, sample_size_medium, replace=False)
                X_medium = X[indices_medium]
                
                result_medium = self._evaluate_pca_clustering_phase(X_medium, random_state, phase="medium")
                elapsed = time.time() - start_time
                
                # If still promising and time allows, do comprehensive evaluation
                if (elapsed < 25 and 
                    result_medium.get('improvement', 0) > 0.05 and 
                    n_samples > 30000):
                    
                    # Phase 3: Comprehensive evaluation
                    sample_size_full = min(50000, n_samples)
                    indices_full = np.random.choice(n_samples, sample_size_full, replace=False)
                    X_full = X[indices_full]
                    
                    return self._evaluate_pca_clustering_phase(X_full, random_state, phase="comprehensive")
                
                return result_medium
            
            return result_fast
            
        except Exception as e:
            return {
                'recommended': False,
                'recommendation': f"Evaluation failed: {str(e)[:50]}...",
                'method': 'baseline',
                'reason': 'Error during hybrid evaluation'
            }
    
    def _evaluate_pca_clustering_phase(self, X, random_state, phase="fast"):
        """
        Single phase of PCA clustering evaluation.
        
        Args:
            X: Feature matrix
            random_state: Random seed
            phase: Evaluation phase (fast/medium/comprehensive)
            
        Returns:
            Dictionary with phase results
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA
        
        # Progressive complexity based on phase
        if phase == "fast":
            k_range = [3]
            n_init = 3
            pca_components = 0.95
        elif phase == "medium":
            k_range = [2, 3, 4]
            n_init = 5
            pca_components = [0.95, 0.90]
        else:  # comprehensive
            k_range = [2, 3, 4, 5]
            n_init = 10
            pca_components = [0.99, 0.95, 0.90, 0.85]
        
        best_baseline_score = -1
        best_pca_score = -1
        best_k = 3
        best_pca_components = 0.95
        
        # Test different k values
        for k in k_range:
            # Baseline KMeans
            try:
                kmeans_base = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=100)
                labels_base = kmeans_base.fit_predict(X)
                if len(set(labels_base)) > 1:
                    score_base = silhouette_score(X, labels_base)
                    if score_base > best_baseline_score:
                        best_baseline_score = score_base
                        best_k = k
            except:
                continue
                
            # Test PCA variants
            if isinstance(pca_components, list):
                pca_components_list = pca_components
            else:
                pca_components_list = [pca_components]
                
            for pca_comp in pca_components_list:
                try:
                    pca = PCA(n_components=pca_comp, random_state=random_state)
                    X_pca = pca.fit_transform(X)
                    
                    kmeans_pca = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=100)
                    labels_pca = kmeans_pca.fit_predict(X_pca)
                    
                    if len(set(labels_pca)) > 1:
                        score_pca = silhouette_score(X_pca, labels_pca)
                        if score_pca > best_pca_score:
                            best_pca_score = score_pca
                            best_pca_components = pca_comp
                except:
                    continue
        
        # Calculate improvement
        improvement = best_pca_score - best_baseline_score
        
        # Make recommendation based on phase and improvement
        if phase == "fast":
            threshold = 0.03
        elif phase == "medium":
            threshold = 0.05
        else:  # comprehensive
            threshold = 0.02  # Lower threshold for comprehensive evaluation
            
        if improvement > threshold:
            return {
                'recommended': True,
                'recommendation': f"PCA recommended (retains {best_pca_components*100:.0f}% variance)",
                'method': 'pca',
                'improvement': improvement,
                'details': f"Silhouette: {best_baseline_score:.3f} → {best_pca_score:.3f} (+{improvement:.3f})",
                'best_k': best_k,
                'pca_components': best_pca_components,
                'phase': phase
            }
        else:
            return {
                'recommended': False,
                'recommendation': f"Original features preferred ({phase} evaluation)",
                'method': 'baseline',
                'improvement': improvement,
                'details': f"Silhouette: baseline {best_baseline_score:.3f} vs PCA {best_pca_score:.3f}",
                'phase': phase,
                'reason': f"Improvement (+{improvement:.3f}) below threshold ({threshold})"
            }
    
    def render_tabs(self):
        """Render main content tabs."""
        # Determine which tabs to show based on available results
        professional_results = st.session_state.get('professional_results')
        standard_results = st.session_state.get('results')
        
        if professional_results and standard_results:
            # Show all tabs including Professional
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Data Overview",
                "🤖 Models",
                "🔥 Professional AutoML",
                "📐 PCA Analysis",
                "🔍 Explainability", 
                "🎯 Recommendation",
                "📄 Report"
            ])
        elif professional_results:
            # Show professional-focused tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Data Overview",
                "🔥 Professional AutoML",
                "📐 PCA Analysis", 
                "🔍 Explainability",
                "🎯 Recommendation",
                "📄 Report",
                "🎯 Insights"  # Professional insights tab
            ])
        else:
            # Standard tabs
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
        
        # Professional AutoML tab
        if professional_results:
            with tab2 if not standard_results else tab3:
                self.render_professional_automl_tab()
        
        # Standard Models tab (if available)
        if standard_results:
            with tab2:
                if st.session_state.task_type == "Classification":
                    self.render_classification_results()
                else:
                    self.render_clustering_results()
        
        # Adjust tab indices based on what's available
        if professional_results and standard_results:
            # Both available: Data, Models, Professional, PCA, Explain, Recommend, Report
            pca_tab, explain_tab, recommend_tab, report_tab = tab4, tab5, tab6, tab7
        elif professional_results:
            # Professional only: Data, Professional, PCA, Explain, Recommend, Report, Insights
            pca_tab, explain_tab, recommend_tab, report_tab = tab3, tab4, tab5, tab6
        else:
            # Standard only: Data, Models, PCA, Explain, Recommend, Report
            pca_tab, explain_tab, recommend_tab, report_tab = tab3, tab4, tab5, tab6
        
        with pca_tab:
            self.render_pca_analysis()
        
        with explain_tab:
            self.render_explainability()
        
        with recommend_tab:
            self.render_recommendation()
        
        with report_tab:
            self.render_report()
    
    def render_professional_automl_tab(self):
        """Render dedicated Professional AutoML results tab."""
        st.markdown("### 🔥 **Professional AutoML Pipeline Results**")
        
        professional_results = st.session_state.get('professional_results')
        dataset_stats = st.session_state.get('dataset_stats')
        
        if not professional_results:
            st.warning("⚠️ No Professional AutoML results available.")
            return
        
        # Professional results overview
        st.markdown("---")
        self._display_professional_results(professional_results, st.session_state.get('advanced_features', []))
        
        # Additional professional insights if available
        if dataset_stats:
            st.markdown("---")
            st.markdown("### 📊 **Advanced Dataset Analysis**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💾 Memory Usage", f"{dataset_stats.get('memory_usage_mb', 0):.1f} MB")
            with col2:
                st.metric("🔢 Numeric Features", dataset_stats.get('numeric_features', 0))
            with col3:
                st.metric("📝 Categorical Features", dataset_stats.get('categorical_features', 0))
            with col4:
                st.metric("❓ Missing Values", dataset_stats.get('missing_values', 0))
            
            # Data complexity metrics
            if 'feature_correlation_avg' in dataset_stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔗 Avg Correlation", f"{dataset_stats['feature_correlation_avg']:.3f}")
                with col2:
                    st.metric("🕳️ Sparsity", f"{dataset_stats['sparsity']:.3f}")
                with col3:
                    if 'class_balance' in dataset_stats:
                        st.metric("⚖️ Class Balance", f"{dataset_stats['class_balance']:.3f}")
    
    def render_pca_analysis(self):
        """Render PCA analysis tab with dimensionality reduction insights."""
        st.subheader("📐 Dimensionality Reduction Analysis")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload data first to view PCA analysis.")
            return
        
        # Check if dimred was enabled and run
        if not hasattr(st.session_state, 'dimred_results') or st.session_state.dimred_results is None:
            st.info("💡 Run PCA analysis on your current dataset:")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                # PCA configuration
                st.markdown("#### ⚙️ PCA Configuration")
                variance_target = st.slider(
                    "Variance Target", 
                    min_value=0.8, max_value=0.99, value=0.95, step=0.01,
                    help="How much variance to preserve"
                )
                max_components = st.number_input(
                    "Max Components", 
                    min_value=2, max_value=50, value=10,
                    help="Maximum number of components"
                )
                
                if st.button("🔍 Analyze with PCA", type="primary"):
                    with st.spinner("Running PCA analysis..."):
                        try:
                            from core.dimred import make_dimred, DimRedConfig
                            from core.visualize import plot_pca_scree, plot_pca_2d_scatter
                            
                            # Get data
                            if hasattr(st.session_state, 'X_processed') and st.session_state.X_processed is not None:
                                X_data = st.session_state.X_processed
                                y_data = st.session_state.get('y_processed', None)
                            else:
                                X_data = st.session_state.data.select_dtypes(include=[np.number])
                                y_data = None
                            
                            # Run PCA analysis
                            dimred_config = DimRedConfig(enable='on', method='pca', variance_target=variance_target)
                            pca = make_dimred(
                                is_sparse_after_ohe=False,
                                n_features=X_data.shape[1], 
                                n_samples=X_data.shape[0],
                                cfg=dimred_config
                            )
                            
                            if pca is None:
                                st.error("❌ PCA not applicable to this dataset (insufficient features or samples)")
                                return
                            
                            # Validate data before transformation
                            if X_data.shape[1] < 2:
                                st.error("❌ PCA requires at least 2 features. Current dataset has only 1 feature.")
                                return
                            
                            # Fit and transform
                            X_transformed = pca.fit_transform(X_data)
                            
                            # Create and store results
                            st.session_state.dimred_results = {
                                'pca_transformer': pca,
                                'X_transformed': X_transformed,
                                'explained_variance_ratio': pca.explained_variance_ratio_,
                                'n_components': pca.n_components_,
                                'method': 'pca'
                            }
                            
                            st.success("✅ PCA analysis completed!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ PCA analysis failed: {e}")
                            logger.error(f"PCA analysis error: {e}")
            
            with col2:
                st.markdown("#### 🎯 What You'll See")
                st.markdown("""
                - **Scree Plot**: Explained variance by component
                - **2D Visualization**: Data projected to 2D
                - **Component Analysis**: PCA loadings and importance
                - **Variance Metrics**: How much information is preserved
                """)
            return
        
        # Display dimred results
        dimred_results = st.session_state.dimred_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Explained Variance")
            
            # Debug: Show what's in dimred_results
            st.write(f"🔍 DEBUG: Dimred results keys: {list(dimred_results.keys()) if dimred_results else 'None'}")
            
            # Check for PCA results in various possible locations
            pca_transformer = None
            if dimred_results:
                # Check direct pca_transformer
                if 'pca_transformer' in dimred_results:
                    pca_transformer = dimred_results['pca_transformer']
                # Check in recommended config
                elif 'recommended_config' in dimred_results:
                    rec_config = dimred_results['recommended_config']
                    if hasattr(rec_config, 'method') and rec_config.method == 'pca':
                        # Look for PCA transformer in other keys
                        for key, value in dimred_results.items():
                            if hasattr(value, 'explained_variance_ratio_'):
                                pca_transformer = value
                                break
                # Check all values for PCA-like objects
                if not pca_transformer:
                    for key, value in dimred_results.items():
                        if hasattr(value, 'explained_variance_ratio_'):
                            pca_transformer = value
                            st.info(f"Found PCA transformer in key: {key}")
                            break
            
            # Show scree plot if PCA was used
            if pca_transformer and hasattr(pca_transformer, 'explained_variance_ratio_'):
                from core.visualize import plot_pca_scree
                try:
                    fig = plot_pca_scree(pca_transformer)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating scree plot: {e}")
                    # Fallback to simple visualization
                    explained_var = pca_transformer.explained_variance_ratio_
                    cumsum_var = np.cumsum(explained_var)
                    
                    st.write(f"**Components**: {len(explained_var)}")
                    st.write(f"**Total Variance Explained**: {cumsum_var[-1]:.1%}")
                    
                    # Show top components
                    for i in range(min(5, len(explained_var))):
                        st.write(f"PC{i+1}: {explained_var[i]:.1%}")
            else:
                st.info("Scree plot available for PCA method only")
        
        with col2:
            st.markdown("#### 🎯 PCA Insights")
            
            if 'pca_transformer' in dimred_results:
                pca = dimred_results['pca_transformer']
                
                # Show key metrics
                st.metric("Components Used", dimred_results.get('n_components', 'N/A'))
                st.metric("Variance Preserved", f"{np.sum(pca.explained_variance_ratio_):.1%}")
                st.metric("Original Features", pca.n_features_in_ if hasattr(pca, 'n_features_in_') else 'N/A')
                
                # Show 2D projection if available
                if 'X_transformed' in dimred_results and dimred_results['X_transformed'].shape[1] >= 2:
                    st.markdown("#### 🎨 2D Projection")
                    try:
                        from core.visualize import plot_pca_2d_scatter
                        y_data = st.session_state.get('y_processed', None)
                        
                        # Handle different data types for y
                        if y_data is not None:
                            if len(np.unique(y_data)) > 20:
                                y_data = None  # Too many classes, don't color
                        
                        # Get explained variance ratio for proper axis labels
                        explained_variance = pca.explained_variance_ratio_ if hasattr(pca, 'explained_variance_ratio_') else None
                        
                        fig_2d = plot_pca_2d_scatter(
                            dimred_results['X_transformed'][:, :2], 
                            y_data,
                            "PCA 2D Projection",
                            explained_variance
                        )
                        st.plotly_chart(fig_2d, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating 2D plot: {e}")
            else:
                # Generate AI-powered PCA recommendations
                if st.session_state.ai_engine and hasattr(st.session_state, 'X_processed') and st.session_state.X_processed is not None:
                    st.markdown("#### 🤖 AI PCA Analysis")
                    
                    try:
                        n_features = st.session_state.X_processed.shape[1] 
                        n_samples = st.session_state.X_processed.shape[0]
                        
                        # Create quick statistics for AI analysis
                        from core.ai_insights import DatasetStatistics
                        pca_stats = DatasetStatistics(
                            n_samples=n_samples,
                            n_features=n_features,
                            n_numeric=n_features,  # PCA is for numeric data
                            n_categorical=0,
                            target_type=st.session_state.get('task_type', 'classification').lower(),
                            n_classes=None,
                            class_balance=None,
                            missing_rate=0.0,  # Processed data has no missing values
                            feature_correlations=[],
                            top_features=[],
                            data_quality_score=95.0
                        )
                        
                        # Generate PCA-specific insights using AI
                        pca_prompt = f"""
You are analyzing a dataset for PCA (Principal Component Analysis). 

Dataset: {n_samples:,} samples, {n_features} features
Task: {st.session_state.get('task_type', 'Classification')}
Context: User is considering PCA for dimensionality reduction

Provide PCA-specific recommendations in JSON format:
1. "recommendation": Overall PCA recommendation (Highly Recommended/Recommended/Optional/Not Recommended)
2. "reasoning": 2-3 sentences explaining why PCA fits this dataset
3. "optimal_components": Suggested number of components to retain
4. "expected_benefits": 2-3 specific benefits PCA will provide
5. "considerations": Any important considerations or warnings

Be specific to this dataset size and dimensionality."""

                        pca_response = st.session_state.ai_engine._call_llm(pca_prompt)
                        pca_insights = st.session_state.ai_engine._parse_response(pca_response)
                        
                        # Display AI-generated PCA insights
                        if 'recommendation' in pca_insights:
                            if 'Highly Recommended' in pca_insights['recommendation']:
                                st.success(f"🎯 **{pca_insights['recommendation']}**")
                            elif 'Recommended' in pca_insights['recommendation']:
                                st.info(f"📊 **{pca_insights['recommendation']}**") 
                            else:
                                st.warning(f"⚠️ **{pca_insights['recommendation']}**")
                        
                        if 'reasoning' in pca_insights:
                            st.write(f"**💡 AI Analysis:** {pca_insights['reasoning']}")
                        
                        if 'optimal_components' in pca_insights:
                            st.metric("🎯 Suggested Components", pca_insights['optimal_components'])
                        
                        if 'expected_benefits' in pca_insights:
                            st.write("**✅ Expected Benefits:**")
                            benefits = pca_insights['expected_benefits']
                            if isinstance(benefits, list):
                                for benefit in benefits:
                                    st.success(f"• {benefit}")
                            else:
                                st.success(f"• {benefits}")
                        
                        if 'considerations' in pca_insights:
                            st.write("**⚠️ Important Considerations:**")
                            considerations = pca_insights['considerations']
                            if isinstance(considerations, list):
                                for consideration in considerations:
                                    st.warning(f"• {consideration}")
                            else:
                                st.warning(f"• {considerations}")
                    
                    except Exception as e:
                        # Fallback to basic recommendations if AI fails
                        st.markdown("#### 📊 PCA Recommendations")
                        logger.warning(f"AI PCA insights failed: {e}")
                        
                        if n_features > 50:
                            st.success("✅ High-dimensional data - PCA strongly recommended")
                        elif n_features > 20:
                            st.warning("⚠️ Moderate dimensions - PCA may help")
                        else:
                            st.info("ℹ️ Low dimensions - PCA optional")
                        
                        if n_samples < 1000:
                            st.info("📊 Small dataset - Consider fewer components")
                        elif n_samples > 10000:
                            st.info("📊 Large dataset - PCA will be efficient")
                else:
                    # Basic fallback when no AI engine or data
                    st.markdown("#### 📊 PCA Recommendations")
                    st.info("💡 Upload data and enable AI for personalized PCA insights")

    
    def render_data_overview(self):
        """Render data overview tab."""
        st.subheader("📊 Dataset Overview")
        
        data = st.session_state.data
        
        # Generate Comprehensive AI Insights at the top (right after data upload)
        ai_engine = st.session_state.enhanced_ai_engine or st.session_state.ai_engine
        
        # AI Insights Control Panel
        if ai_engine and ai_engine is not False:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if st.session_state.enhanced_ai_engine:
                    st.success("🧠 **Enhanced AI Analysis Active** - Comprehensive dataset insights")
                elif st.session_state.ai_engine:
                    st.info("🤖 **Standard AI Analysis Active** - Basic dataset insights")
            
            with col2:
                refresh_insights = st.button("🔄 Refresh Analysis", help="Re-analyze dataset with AI")
            
            with col3:
                analysis_depth = st.selectbox(
                    "Analysis Depth",
                    ["initial_analysis", "advanced_insights", "model_selection"],
                    help="Choose analysis focus"
                )
            
            # Generate insights (initially or on refresh)
            if st.session_state.ai_insights is None or refresh_insights:
                with st.spinner("🤖 Enhanced AI is performing comprehensive dataset analysis..."):
                    try:
                        # Determine task type and target
                        task_type = st.session_state.get('task_type', 'Classification')
                        target_col = st.session_state.get('target_col', None)
                        
                        # Use enhanced AI engine if available
                        if st.session_state.enhanced_ai_engine:
                            # Enhanced comprehensive analysis
                            stats = st.session_state.enhanced_ai_engine.analyze_dataset_comprehensive(
                                data=data,
                                target_col=target_col,
                                task_type=task_type.lower()
                            )
                            
                            insights = st.session_state.enhanced_ai_engine.generate_comprehensive_insights(
                                stats=stats,
                                context=analysis_depth
                            )
                        else:
                            # Fallback to standard AI engine
                            stats = st.session_state.ai_engine.analyze_dataset(
                                data=data,
                                target_col=target_col,
                                task_type=task_type.lower()
                            )
                            
                            insights = st.session_state.ai_engine.generate_insights(
                                stats=stats,
                                context=analysis_depth
                            )
                        
                        st.session_state.ai_insights = insights
                        if refresh_insights:
                            st.success("✅ Analysis refreshed successfully!")
                        
                    except Exception as e:
                        logger.warning(f"AI insights generation failed: {e}")
                        # Fallback to basic insights
                        insights = self._generate_basic_insights(data, target_col, task_type)
                        st.session_state.ai_insights = insights
                        st.error(f"AI analysis failed. Using fallback analysis. Error: {str(e)[:100]}...")
            
            # Display Comprehensive AI Insights
            if st.session_state.ai_insights and "error" not in st.session_state.ai_insights:
                insights = st.session_state.ai_insights
                
                # Check insight source and quality
                is_ai_generated = insights.get('_source', 'unknown') == 'ai'
                is_enhanced = insights.get('_source', 'unknown') == 'enhanced_rules' or st.session_state.enhanced_ai_engine
                quality_score = insights.get('_quality_score', 0)
                
                # Header based on insight type
                if is_ai_generated and is_enhanced:
                    with st.expander("🧠 **Enhanced AI-Powered Dataset Analysis** ⭐", expanded=True):
                        st.success("🎯 **Powered by Enhanced AI** - Comprehensive analysis with deep insights")
                elif is_ai_generated:
                    with st.expander("🤖 **AI-Generated Dataset Analysis** ✨", expanded=True):
                        st.success("🎯 **Powered by AI** - Dynamic analysis from your LLM")
                else:
                    with st.expander("📊 **Comprehensive Dataset Analysis** (Enhanced Rules)", expanded=True):
                        if '_notice' in insights:
                            st.warning(insights['_notice'])
                        else:
                            st.info("ℹ️ Using enhanced rule-based analysis. Enable AI for dynamic insights.")
                
                # Display quality score
                if quality_score > 0:
                    if quality_score >= 80:
                        st.success(f"📊 **Data Quality Score: {quality_score:.0f}/100** - Excellent")
                    elif quality_score >= 60:
                        st.warning(f"📊 **Data Quality Score: {quality_score:.0f}/100** - Good")
                    else:
                        st.error(f"📊 **Data Quality Score: {quality_score:.0f}/100** - Needs Improvement")
                
                # Dataset Overview (Enhanced)
                if 'dataset_overview' in insights:
                    st.markdown("### 📋 **Dataset Overview**")
                    st.markdown(insights['dataset_overview'])
                    st.markdown("---")
                
                # Main Analysis in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # Strengths (Enhanced)
                    if 'key_strengths' in insights or 'strengths' in insights:
                        st.markdown("#### ✅ **Key Strengths**")
                        strengths = insights.get('key_strengths', insights.get('strengths', []))
                        if isinstance(strengths, list):
                            for strength in strengths:
                                st.success(f"• {strength}")
                        else:
                            st.success(f"• {strengths}")
                
                with col2:
                    # Challenges (Enhanced)
                    if 'critical_challenges' in insights or 'challenges' in insights:
                        st.markdown("#### ⚠️ **Critical Challenges**")
                        challenges = insights.get('critical_challenges', insights.get('challenges', []))
                        if isinstance(challenges, list):
                            for challenge in challenges:
                                st.warning(f"• {challenge}")
                        else:
                            st.warning(f"• {challenges}")
                
                # Data Quality Assessment (Enhanced)
                if 'data_quality_assessment' in insights:
                    st.markdown("#### 🔍 **Data Quality Assessment**")
                    st.info(insights['data_quality_assessment'])
                
                # Preprocessing Strategy (Enhanced)
                if 'preprocessing_strategy' in insights:
                    st.markdown("#### 🔧 **Preprocessing Strategy**")
                    prep_steps = insights['preprocessing_strategy']
                    if isinstance(prep_steps, list):
                        for i, step in enumerate(prep_steps, 1):
                            st.markdown(f"**{i}.** {step}")
                    else:
                        st.markdown(prep_steps)
                
                # Model Recommendations (Enhanced)
                if 'recommended_models' in insights:
                    st.markdown("#### 🎯 **Recommended Models**")
                    models = insights['recommended_models']
                    if isinstance(models, list):
                        for model in models:
                            st.success(f"🤖 {model}")
                    else:
                        st.success(f"🤖 {models}")
                
                # Feature Engineering (Enhanced)
                if 'feature_engineering_opportunities' in insights:
                    st.markdown("#### ⚙️ **Feature Engineering Opportunities**")
                    feature_eng = insights['feature_engineering_opportunities']
                    if isinstance(feature_eng, list):
                        for opp in feature_eng:
                            st.info(f"🛠️ {opp}")
                    else:
                        st.info(f"🛠️ {feature_eng}")
                
                # Statistical Insights (Enhanced)
                if 'statistical_insights' in insights:
                    st.markdown("#### 📈 **Statistical Insights**")
                    stats_insights = insights['statistical_insights']
                    if isinstance(stats_insights, list):
                        for stat in stats_insights:
                            st.markdown(f"• {stat}")
                    else:
                        st.markdown(stats_insights)
                
                # Risk Assessment (Enhanced)
                if 'risk_factors' in insights:
                    st.markdown("#### ⚠️ **Risk Factors**")
                    risks = insights['risk_factors']
                    if isinstance(risks, list):
                        for risk in risks:
                            st.error(f"🚨 {risk}")
                    else:
                        st.error(f"🚨 {risks}")
                
                # Performance Expectations (Enhanced)
                if 'expected_performance' in insights:
                    st.markdown("#### 🎯 **Performance Expectations**")
                    st.success(insights['expected_performance'])
                
                # Next Steps (Enhanced)
                if 'next_steps' in insights:
                    st.markdown("#### 🚀 **Next Steps**")
                    steps = insights['next_steps']
                    if isinstance(steps, list):
                        for i, step in enumerate(steps, 1):
                            st.markdown(f"**{i}.** {step}")
                    else:
                        st.markdown(steps)
                
                # Additional advanced insights (if available)
                advanced_fields = [
                    'tier1_models', 'tier2_models', 'avoid_models', 
                    'hyperparameter_priorities', 'validation_strategy',
                    'advanced_statistical_analysis', 'feature_space_analysis',
                    'domain_specific_recommendations'
                ]
                
                for field in advanced_fields:
                    if field in insights:
                        st.markdown(f"#### 🔬 **{field.replace('_', ' ').title()}**")
                        value = insights[field]
                        if isinstance(value, list):
                            for item in value:
                                st.markdown(f"• {item}")
                        else:
                            st.markdown(value)
                
                st.markdown("---")
        
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
            
            # Add comprehensive visualizations for classification models
            if not is_clustering:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### 📊 Feature Importance")
                    try:
                        # Get feature importance based on model type
                        if hasattr(model, 'feature_importances_'):
                            # Tree-based models
                            importances = model.feature_importances_
                            importance_type = "Built-in Feature Importance"
                        elif hasattr(model, 'coef_'):
                            # Linear models
                            coef = model.coef_
                            importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
                            importance_type = "Coefficient Magnitude"
                        else:
                            # Use permutation importance for other models
                            from sklearn.inspection import permutation_importance
                            perm_result = permutation_importance(
                                model, X[:100], model.predict(X[:100]),  # Use subset for speed
                                n_repeats=3, random_state=42, n_jobs=1
                            )
                            importances = perm_result.importances_mean
                            importance_type = "Permutation Importance"
                        
                        # Create importance plot
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=True).tail(15)  # Top 15
                        
                        import plotly.express as px
                        fig_importance = px.bar(
                            importance_df, 
                            x='Importance', 
                            y='Feature',
                            title=f"{importance_type} - {selected_model}",
                            orientation='h'
                        )
                        fig_importance.update_layout(height=400)
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"Could not generate feature importance plot: {e}")
                
                with col2:
                    st.markdown("#### 🎯 Performance Breakdown")
                    try:
                        # Show model performance metrics in detail
                        results = st.session_state.results.get(selected_model, {})
                        
                        metrics_data = []
                        if 'accuracy_mean' in results:
                            metrics_data.append(['Accuracy', f"{results['accuracy_mean']:.3f} ± {results.get('accuracy_std', 0):.3f}"])
                        if 'precision_mean' in results:
                            metrics_data.append(['Precision', f"{results['precision_mean']:.3f} ± {results.get('precision_std', 0):.3f}"])
                        if 'recall_mean' in results:
                            metrics_data.append(['Recall', f"{results['recall_mean']:.3f} ± {results.get('recall_std', 0):.3f}"])
                        if 'f1_mean' in results:
                            metrics_data.append(['F1-Score', f"{results['f1_mean']:.3f} ± {results.get('f1_std', 0):.3f}"])
                        
                        if metrics_data:
                            metrics_df = pd.DataFrame(metrics_data, columns=['Metric', 'Value'])
                            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                        
                        # Add training vs test comparison
                        if 'test_accuracy' in results and 'train_accuracy' in results:
                            train_acc = results['train_accuracy']
                            test_acc = results['test_accuracy']
                            gap = train_acc - test_acc
                            
                            st.metric("Overfitting Gap", f"{gap:.3f}", 
                                     delta=f"{'Good' if gap < 0.05 else 'Warning' if gap < 0.1 else 'Poor'}")
                        
                    except Exception as e:
                        st.warning(f"Could not generate performance breakdown: {e}")
            
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
                                    # For models without native importance (KNN, MLP, RBF-SVM), use permutation
                                    try:
                                        # Use a small sample for faster computation
                                        perm_sample_size = min(100, X.shape[0])
                                        if X.shape[0] > perm_sample_size:
                                            perm_indices = np.random.choice(X.shape[0], perm_sample_size, replace=False)
                                            X_perm = X[perm_indices]
                                        else:
                                            X_perm = X
                                        
                                        y_perm = model.predict(X_perm)
                                        
                                        from sklearn.inspection import permutation_importance
                                        perm_result = permutation_importance(
                                            model, X_perm, y_perm,
                                            n_repeats=3,
                                            random_state=42,
                                            n_jobs=1
                                        )
                                        importances = perm_result.importances_mean
                                    except Exception as perm_error:
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

**Top 5 Most Important Features for {selected_model} (Permutation Importance):**
{features_str}

**Dataset Context:**
- Total Features: {len(feature_names)}
- Note: {selected_model} doesn't provide native feature importance, so permutation importance was used

Provide model-specific analysis in JSON format:
1. "model_characteristics": How does {selected_model} work? What makes it unique for this problem?
2. "performance_insights": Why did it achieve {test_acc:.1%} test accuracy? What are its strengths/weaknesses?
3. "feature_advice": 2-3 recommendations to improve THIS MODEL or use these insights

Be specific to {selected_model}'s algorithm and explain why permutation importance is appropriate."""
                                    
                                    response = st.session_state.ai_engine._call_llm(prompt)
                                    insights = st.session_state.ai_engine._parse_response(response)
                                    
                                    # Cache the AI insights
                                    st.session_state.explainability_cache[ai_cache_key] = insights
                                else:
                                    # No importance available - provide model-specific analysis anyway
                                    prompt = f"""You are an expert data scientist analyzing {selected_model} specifically.

**Model:** {selected_model}
**Performance:** Test Acc: {test_acc:.1%}, Train Acc: {train_acc:.1%}, Gap: {gap:.1%}
**Dataset:** {X.shape[0]} samples, {X.shape[1]} features

**Note:** Feature importance unavailable for this model type - provide general analysis.

Provide model-specific analysis in JSON format:
1. "model_characteristics": How does {selected_model} work? What are its key properties?
2. "performance_insights": Why did it achieve {test_acc:.1%} test accuracy? What makes it suitable/unsuitable for this problem?
3. "model_advice": 2-3 recommendations for using or improving {selected_model} performance

Focus on the algorithm's behavior rather than specific features."""
                                    
                                    response = st.session_state.ai_engine._call_llm(prompt)
                                    insights = st.session_state.ai_engine._parse_response(response)
                                    
                                    # Cache the AI insights
                                    st.session_state.explainability_cache[ai_cache_key] = insights
                            
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
                    
                    if 'model_advice' in insights:
                        st.warning("**→ Model Recommendations:**")
                        if isinstance(insights['model_advice'], list):
                            for advice in insights['model_advice']:
                                st.markdown(f"- {advice}")
                        else:
                            st.markdown(insights['model_advice'])
                    
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
                        # Models without native importance - use permutation importance
                        try:
                            with st.spinner(f"Computing permutation importance for {selected_model}..."):
                                perm_sample_size = min(200, X.shape[0])
                                if X.shape[0] > perm_sample_size:
                                    perm_indices = np.random.choice(X.shape[0], perm_sample_size, replace=False)
                                    X_perm = X[perm_indices]
                                else:
                                    X_perm = X
                                
                                y_perm = model.predict(X_perm)
                                
                                from sklearn.inspection import permutation_importance
                                perm_result = permutation_importance(
                                    model, X_perm, y_perm,
                                    n_repeats=5,
                                    random_state=42,
                                    n_jobs=1
                                )
                                
                                explanations['permutation_importance'] = {
                                    name: float(val) for name, val in zip(feature_names, perm_result.importances_mean)
                                }
                                explanations['permutation_std'] = {
                                    name: float(val) for name, val in zip(feature_names, perm_result.importances_std)
                                }
                        except Exception as perm_error:
                            st.warning(f"Permutation importance failed: {str(perm_error)}")
                            explanations['permutation_error'] = str(perm_error)
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
                
                elif 'permutation_importance' in explanations:
                    st.subheader("Permutation Feature Importance")
                    st.info("💡 Permutation importance shows how much model performance decreases when each feature is randomly shuffled")
                    visualizer = Visualizer()
                    fig = visualizer.plot_feature_importance(
                        explanations['permutation_importance'],
                        top_n=15,
                        title="Top 15 Features by Permutation Importance"
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
        """Render recommendation tab with comprehensive visualizations."""
        st.subheader("🎯 AI-Powered Model Recommendations")
        
        if st.session_state.task_type == "Classification":
            if 'recommendation' not in st.session_state:
                st.info("Run AutoML to see recommendations")
                return
            
            recommendation = st.session_state.recommendation
            
            # Check if recommendation has required fields
            if not recommendation or 'recommended_model' not in recommendation:
                st.warning("No recommendation available. Please run AutoML first.")
                return
            
            # Create comprehensive recommendation dashboard
            col1, col2, col3 = st.columns([1, 1, 1])
            
            # Model Performance Comparison Chart
            with col1:
                st.markdown("#### 📊 Model Performance Comparison")
                try:
                    evaluator = st.session_state.evaluator
                    leaderboard = evaluator.get_leaderboard('accuracy')
                    
                    if leaderboard:
                        models = [item['model'] for item in leaderboard[:5]]
                        scores = [item.get('score', item.get('accuracy', 0)) for item in leaderboard[:5]]
                        
                        # Create performance comparison chart
                        import plotly.express as px
                        import plotly.graph_objects as go
                        
                        fig = go.Figure(data=[
                            go.Bar(x=models, y=scores, 
                                  marker_color=['gold' if m == recommendation['recommended_model'] 
                                               else 'lightblue' for m in models],
                                  text=[f'{s:.3f}' for s in scores],
                                  textposition='auto')
                        ])
                        fig.update_layout(
                            title="Model Accuracy Comparison",
                            xaxis_title="Models",
                            yaxis_title="Accuracy",
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate performance chart: {e}")
            
            # Recommended Model Details
            with col2:
                st.markdown("#### 🎆 Recommended Model Details")
                recommended_model = recommendation['recommended_model']
                score = recommendation.get('score', 0)
                
                st.metric("Best Model", recommended_model)
                st.metric("Accuracy", f"{score:.3f}")
                
                # Model characteristics
                model_info = {
                    'LogisticRegression': {'Type': 'Linear', 'Speed': 'Fast', 'Interpretability': 'High'},
                    'RandomForest': {'Type': 'Ensemble', 'Speed': 'Medium', 'Interpretability': 'Medium'},
                    'XGBoost': {'Type': 'Boosting', 'Speed': 'Fast', 'Interpretability': 'Low'},
                    'LinearSVM': {'Type': 'Linear SVM', 'Speed': 'Medium', 'Interpretability': 'Medium'},
                    'RBF-SVM': {'Type': 'Non-linear SVM', 'Speed': 'Slow', 'Interpretability': 'Low'},
                    'KNN': {'Type': 'Instance-based', 'Speed': 'Slow', 'Interpretability': 'Medium'},
                    'MLP': {'Type': 'Neural Network', 'Speed': 'Medium', 'Interpretability': 'Low'}
                }
                
                if recommended_model in model_info:
                    info = model_info[recommended_model]
                    st.json(info)
            
            # Model Comparison Radar Chart
            with col3:
                st.markdown("#### 🕵️ Model Trade-offs Analysis")
                try:
                    # Create radar chart comparing models on different aspects
                    model_aspects = {
                        'LogisticRegression': {'Accuracy': 4, 'Speed': 5, 'Interpretability': 5, 'Complexity': 2},
                        'RandomForest': {'Accuracy': 5, 'Speed': 3, 'Interpretability': 3, 'Complexity': 3},
                        'XGBoost': {'Accuracy': 5, 'Speed': 4, 'Interpretability': 2, 'Complexity': 4},
                        'LinearSVM': {'Accuracy': 4, 'Speed': 3, 'Interpretability': 3, 'Complexity': 3},
                        'RBF-SVM': {'Accuracy': 4, 'Speed': 2, 'Interpretability': 2, 'Complexity': 4},
                        'KNN': {'Accuracy': 3, 'Speed': 2, 'Interpretability': 4, 'Complexity': 1},
                        'MLP': {'Accuracy': 4, 'Speed': 3, 'Interpretability': 1, 'Complexity': 5}
                    }
                    
                    if recommended_model in model_aspects:
                        aspects = model_aspects[recommended_model]
                        
                        import plotly.graph_objects as go
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=list(aspects.values()),
                            theta=list(aspects.keys()),
                            fill='toself',
                            name=recommended_model,
                            line_color='rgb(106, 81, 163)'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 5])
                            ),
                            height=300,
                            title="Model Characteristics"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.warning(f"Could not generate trade-offs chart: {e}")
            
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
    
    def _generate_basic_insights(self, data, target_col=None, task_type="Classification"):
        """Generate comprehensive dataset insights when AI engine is not available."""
        try:
            import pandas as pd
            from collections import Counter
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            datetime_cols = data.select_dtypes(include=['datetime64']).columns
            
            # Detailed analysis
            n_rows, n_cols = data.shape
            missing_values = data.isnull().sum()
            missing_cols = missing_values[missing_values > 0]
            missing_pct = (missing_values.sum() / (n_rows * n_cols)) * 100
            
            # Memory usage
            memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            
            strengths = []
            challenges = []
            recommendations = []
            
            # Dataset size analysis
            if n_rows < 100:
                challenges.append(f"Small dataset ({n_rows:,} samples) - risk of overfitting")
                recommendations.append("Consider data augmentation or simpler models")
            elif n_rows < 1000:
                strengths.append(f"Moderate dataset size ({n_rows:,} samples) - good for prototyping")
                recommendations.append("Cross-validation crucial for reliable estimates")
            elif n_rows < 10000:
                strengths.append(f"Good dataset size ({n_rows:,} samples) - sufficient for most models")
            elif n_rows < 100000:
                strengths.append(f"Large dataset ({n_rows:,} samples) - enables complex models")
                recommendations.append("Consider gradient boosting or neural networks")
            else:
                strengths.append(f"Very large dataset ({n_rows:,} samples) - excellent for deep learning")
                recommendations.append("Use batch processing and distributed training")
            
            # Feature analysis
            if len(numeric_cols) == 0:
                challenges.append("No numerical features detected")
                recommendations.append("Feature engineering needed for numerical representations")
            else:
                strengths.append(f"{len(numeric_cols)} numerical features available")
                
                # Check for potential issues in numerical data
                for col in numeric_cols:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        # Check for constant features
                        if col_data.nunique() == 1:
                            challenges.append(f"Constant feature detected: {col}")
                        # Check for extreme outliers
                        elif col_data.std() > 0:
                            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                            extreme_outliers = (z_scores > 5).sum()
                            if extreme_outliers > n_rows * 0.05:  # More than 5% extreme outliers
                                challenges.append(f"Many extreme outliers in {col} ({extreme_outliers} values)")
            
            if len(categorical_cols) == 0:
                strengths.append("No categorical encoding needed")
            else:
                strengths.append(f"{len(categorical_cols)} categorical features")
                # Check cardinality
                high_cardinality = []
                for col in categorical_cols:
                    unique_vals = data[col].nunique()
                    if unique_vals > n_rows * 0.5:  # High cardinality
                        high_cardinality.append(f"{col} ({unique_vals} unique)")
                
                if high_cardinality:
                    challenges.append(f"High cardinality features: {', '.join(high_cardinality)}")
                    recommendations.append("Consider target encoding or feature hashing")
            
            # Missing data analysis
            if missing_pct < 1:
                strengths.append("Excellent data quality - minimal missing values")
            elif missing_pct < 5:
                strengths.append("Good data quality - few missing values")
            elif missing_pct < 15:
                challenges.append(f"Some missing data ({missing_pct:.1f}% of all values)")
                recommendations.append("Imputation strategy needed")
            else:
                challenges.append(f"Significant missing data ({missing_pct:.1f}% of all values)")
                recommendations.append("Consider advanced imputation or feature removal")
            
            if len(missing_cols) > 0:
                worst_missing = missing_cols.nlargest(3)
                missing_details = [f"{col}: {count} ({count/n_rows*100:.1f}%)" for col, count in worst_missing.items()]
                challenges.append(f"Top missing features: {', '.join(missing_details)}")
            
            # Target variable analysis (if available)
            if target_col and target_col in data.columns:
                target_data = data[target_col].dropna()
                unique_targets = target_data.nunique()
                
                if task_type == "Classification":
                    class_counts = Counter(target_data)
                    
                    if unique_targets == 2:
                        strengths.append("Binary classification - well-suited for many algorithms")
                        # Check balance
                        min_class, max_class = min(class_counts.values()), max(class_counts.values())
                        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
                        
                        if imbalance_ratio > 10:
                            challenges.append(f"Severe class imbalance (ratio: {imbalance_ratio:.1f}:1)")
                            recommendations.append("Use SMOTE, class weights, or threshold tuning")
                        elif imbalance_ratio > 3:
                            challenges.append(f"Moderate class imbalance (ratio: {imbalance_ratio:.1f}:1)")
                            recommendations.append("Consider class balancing techniques")
                        else:
                            strengths.append("Well-balanced binary classification")
                    
                    elif unique_targets <= 10:
                        strengths.append(f"Multi-class classification ({unique_targets} classes)")
                        # Check if any classes are very rare
                        rare_classes = [label for label, count in class_counts.items() if count < 10]
                        if rare_classes:
                            challenges.append(f"{len(rare_classes)} classes with <10 samples")
                            recommendations.append("Consider class consolidation or stratified sampling")
                    
                    elif unique_targets <= 50:
                        challenges.append(f"Many classes ({unique_targets}) - complex classification")
                        recommendations.append("Consider hierarchical classification or ensemble methods")
                    
                    else:
                        challenges.append(f"Excessive classes ({unique_targets:,}) - likely regression problem")
                        recommendations.append("Verify task type - this looks like regression data")
                
                # Target distribution analysis
                target_missing = data[target_col].isnull().sum()
                if target_missing > 0:
                    challenges.append(f"Missing target values: {target_missing} ({target_missing/n_rows*100:.1f}%)")
                    recommendations.append("Remove or impute missing target values")
            
            # Dimensionality analysis
            if n_cols > n_rows:
                challenges.append(f"More features ({n_cols}) than samples ({n_rows}) - curse of dimensionality")
                recommendations.append("Essential: Feature selection or dimensionality reduction")
            elif n_cols > n_rows * 0.1:
                recommendations.append("Consider dimensionality reduction (PCA/UMAP) for efficiency")
            
            if n_cols > 100:
                recommendations.append("High-dimensional data - PCA strongly recommended")
            elif n_cols > 50:
                recommendations.append("Moderate dimensionality - PCA may help performance")
            
            # Memory and performance insights
            if memory_mb > 1000:  # > 1GB
                challenges.append(f"Large memory footprint ({memory_mb:.1f} MB)")
                recommendations.append("Consider data sampling or distributed processing")
            
            # Final summary
            if len(challenges) == 0:
                summary = f"🎯 Excellent dataset for {task_type.lower()} - {n_rows:,} samples, {n_cols} features"
            elif len(challenges) <= 2:
                summary = f"📊 Good dataset for {task_type.lower()} - {n_rows:,} samples with minor issues"
            else:
                summary = f"⚠️ Dataset needs preprocessing - {n_rows:,} samples with several challenges"
            
            # Ensure we have some content
            if not strengths:
                strengths = ["Dataset loaded successfully"]
            if not challenges:
                challenges = ["No major data quality issues detected"]
            if not recommendations:
                recommendations = ["Dataset appears ready for modeling"]
            
            return {
                "summary": summary,
                "strengths": strengths,
                "challenges": challenges,
                "recommendations": recommendations
            }
            
        except Exception as e:
            # Fallback for any errors
            return {
                "summary": f"Basic analysis completed for {data.shape[0]:,} samples",
                "strengths": ["Data loaded successfully"],
                "challenges": [f"Analysis error: {str(e)[:100]}..."],
                "recommendations": ["Check data format and verify column types"]
            }
    
    def render_configuration_stage(self):
        """Render the unified configuration stage."""
        # Add main container for better centering
        st.markdown("<div style='max-width: 1200px; margin: 0 auto;'>", unsafe_allow_html=True)
        
        # Header with navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("← Back to Welcome", key="back_to_welcome"):
                st.session_state.app_stage = 'welcome'
                st.rerun()
        
        with col2:
            st.markdown("<h2 style='text-align: center; margin: 0;'>⚙️ Configure AutoML Pipeline</h2>", unsafe_allow_html=True)
        
        with col3:
            # Progress indicator
            st.markdown("<div style='text-align: right; color: #666;'>Step 2 of 3 - Configuration</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dataset overview
        if st.session_state.data is not None:
            data = st.session_state.data
            
            # Quick dataset summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Dataset", f"{data.shape[0]:,} × {data.shape[1]}")
            with col2:
                memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("💾 Size", f"{memory_mb:.1f} MB")
            with col3:
                missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
                st.metric("❓ Missing", f"{missing_pct:.1f}%")
            with col4:
                if hasattr(self, '_get_file_name'):
                    filename = st.session_state.get('uploaded_file_name', 'Unknown')
                    st.metric("📁 File", filename.replace('.csv', ''))
        
        # Unified configuration tabs
        config_tab1, config_tab2, config_tab3, config_tab4 = st.tabs([
            "🎯 Task & Analysis", 
            "🤖 Model Selection", 
            "⚙️ Optimization", 
            "🔧 Advanced Settings"
        ])
        
        with config_tab1:
            self._render_unified_task_tab()
        
        with config_tab2:
            self._render_unified_model_tab()
            
        with config_tab3:
            self._render_unified_optimization_tab()
            
        with config_tab4:
            self._render_unified_advanced_tab()
        
        # Final execution section
        st.markdown("---")
        st.markdown("### 🚀 **Execute AutoML Pipeline**")
        
        # Pre-flight check
        ready_checks = self._check_configuration_readiness()
        
        if ready_checks['ready']:
            # Show current configuration summary
            st.markdown("### 🎯 **Ready to Execute AutoML**")
            
            # Display configuration summary
            config_info = []
            if hasattr(st.session_state, 'selected_models') and st.session_state.selected_models:
                config_info.append(f"📊 **Models**: {', '.join(st.session_state.selected_models[:3])}{'...' if len(st.session_state.selected_models) > 3 else ''}")
            
            if hasattr(st.session_state, 'dimred_enabled') and st.session_state.dimred_enabled != 'auto':
                config_info.append(f"📉 **PCA**: {st.session_state.dimred_enabled}")
            
            if hasattr(st.session_state, 'advanced_config') and st.session_state.advanced_config:
                if st.session_state.advanced_config.get('validation', {}).get('cv_folds'):
                    config_info.append(f"🔄 **CV Folds**: {st.session_state.advanced_config['validation']['cv_folds']}")
            
            if hasattr(st.session_state, 'optimization_config') and st.session_state.optimization_config:
                opt_config = st.session_state.optimization_config
                if opt_config.get('time_minutes', 0) > 5:
                    config_info.append(f"⏰ **Optimization**: {opt_config['time_minutes']} min")
            
            if config_info:
                st.info(" | ".join(config_info))
            else:
                st.info("📝 **Configuration**: Using default settings")
            
            # Single unified run button
            if st.button("🚀 **Run AutoML with My Configuration**", type="primary", use_container_width=True, key="run_unified_automl"):
                # Use intelligent mode selection
                mode = self._determine_execution_mode()
                st.info(f"🔥 **Executing {mode.title()} AutoML** with your configured preferences...")
                self._execute_automl(mode)
        else:
            st.error("❌ Configuration incomplete. Please complete the required settings above.")
            for issue in ready_checks['issues']:
                st.warning(f"• {issue}")
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close configuration container
    
    def _determine_execution_mode(self):
        """Intelligently determine whether to use standard or professional mode based on user configuration."""
        # Check for professional mode indicators
        professional_indicators = [
            # Advanced model selection
            hasattr(st.session_state, 'selected_models') and len(st.session_state.get('selected_models', [])) > 3,
            # Custom optimization settings
            hasattr(st.session_state, 'optimization_config') and st.session_state.optimization_config.get('time_minutes', 0) > 5,
            # Advanced dimensionality reduction
            hasattr(st.session_state, 'dimred_enabled') and st.session_state.dimred_enabled not in ['auto', 'off'],
            # Custom validation settings
            hasattr(st.session_state, 'advanced_config') and st.session_state.advanced_config.get('validation', {}).get('cv_folds', 5) != 5,
            # Large dataset size
            hasattr(st.session_state, 'data') and len(st.session_state.data) > 1000
        ]
        
        return 'professional' if any(professional_indicators) else 'standard'
    
    def render_results_stage(self):
        """Render the results stage with all analysis tabs."""
        # Header with navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("← Back to Configure", key="back_to_configure"):
                st.session_state.app_stage = 'configure'
                st.rerun()
        
        with col2:
            st.markdown("<h2 style='text-align: center; margin: 0;'>📊 AutoML Results</h2>", unsafe_allow_html=True)
        
        with col3:
            # New run button
            if st.button("🔄 New Analysis", key="new_analysis"):
                # Reset to welcome but keep data
                st.session_state.app_stage = 'configure'
                st.session_state.results = None
                st.session_state.evaluator = None
                st.rerun()
        
        st.markdown("---")
        
        # Results content using existing tabs system
        results = st.session_state.get('results')
        professional_results = st.session_state.get('professional_results')
        
        if (results and len(results) > 0) or professional_results:
            self.render_tabs()
        else:
            st.error("❌ No results available. Please run AutoML analysis first.")
            
            # Show specific error if available
            error_msg = st.session_state.get('automl_error')
            if error_msg:
                st.error(f"Error details: {error_msg}")
            
            if st.button("↩️ Back to Configuration", key="back_to_config_from_results"):
                st.session_state.app_stage = 'configure'
                st.rerun()
    
    def _render_unified_task_tab(self):
        """Render unified task selection and dataset analysis."""
        st.header("🎯 Task Selection & Dataset Analysis")
        
        # Task type selection
        st.subheader("📋 Select Machine Learning Task")
        
        task_type = st.radio(
            "Choose your analysis type:",
            ["Classification", "Clustering"],
            help="Classification: Predict categories | Clustering: Find patterns"
        )
        
        st.session_state.task_type = task_type
        
        if task_type == "Classification":
            # Target selection
            columns = st.session_state.data.columns.tolist()
            target_col = st.selectbox(
                "Select Target Column (what you want to predict)",
                options=columns,
                help="Choose the column that contains the values you want to predict"
            )
            st.session_state.target_col = target_col
            
            if target_col:
                # Quick target analysis
                col1, col2 = st.columns(2)
                with col1:
                    unique_values = st.session_state.data[target_col].nunique()
                    st.metric("🎯 Unique Classes", unique_values)
                with col2:
                    missing_target = st.session_state.data[target_col].isnull().sum()
                    st.metric("❓ Missing in Target", missing_target)
                
                # Show class distribution
                if unique_values < 20:  # Only for reasonable number of classes
                    st.markdown("##### 📊 Class Distribution")
                    class_counts = st.session_state.data[target_col].value_counts()
                    st.bar_chart(class_counts)
        
        # Auto-analyze dataset
        if not st.session_state.get('config_analyzed', False) or st.button("🔄 Re-analyze Dataset", key="reanalyze"):
            with st.spinner("🧠 Analyzing dataset characteristics..."):
                dataset_analysis = self._analyze_dataset_for_config(st.session_state.data)
                st.session_state.dataset_config = dataset_analysis
                st.session_state.config_analyzed = True
                st.rerun()
        
        # Display analysis results
        if st.session_state.get('dataset_config'):
            self._display_dataset_analysis(st.session_state.dataset_config)
    
    def _render_unified_model_tab(self):
        """Render unified model selection."""
        self._render_model_selection_tab()
    
    def _render_unified_optimization_tab(self):
        """Render unified optimization settings."""
        self._render_optimization_tab()
    
    def _render_unified_advanced_tab(self):
        """Render unified advanced settings."""
        self._render_advanced_tab()
    
    def _check_configuration_readiness(self):
        """Check if configuration is ready for execution."""
        issues = []
        
        # Check dataset
        if st.session_state.data is None:
            issues.append("No dataset loaded")
        
        # Check task type
        if not st.session_state.get('task_type'):
            issues.append("Task type not selected")
        
        # Check target for classification
        if st.session_state.get('task_type') == 'Classification' and not st.session_state.get('target_col'):
            issues.append("Target column not selected for classification")
        
        return {
            'ready': len(issues) == 0,
            'issues': issues
        }
    
    def _execute_automl(self, mode='standard'):
        """Execute AutoML and transition to results."""
        st.info("🔍 DEBUG: _execute_automl() started")
        try:
            with st.spinner(f"🚀 Running {mode.title()} AutoML..."):
                st.info(f"🔍 DEBUG: About to run {mode} AutoML")
                if mode == 'professional':
                    opt_config = st.session_state.optimization_config
                    self.run_professional_automl(
                        optimization_time_minutes=opt_config['time_minutes'],
                        max_trials=opt_config['max_trials'],
                        include_ensemble=opt_config['include_ensemble'],
                        advanced_features=opt_config.get('advanced_features', [])
                    )
                else:
                    self.run_automl()
                
                st.info("🔍 DEBUG: AutoML method completed")
                
                # CRITICAL: Validate results were stored
                results = st.session_state.get('results')
                professional_results = st.session_state.get('professional_results')
                st.info(f"🔍 DEBUG: Results after AutoML: {results is not None}")
                st.info(f"🔍 DEBUG: Professional Results: {professional_results is not None}")
                
                if results:
                    st.info(f"🔍 DEBUG: Results keys: {list(results.keys())}")
                    
                # Check if we have any results (standard or professional)
                has_results = (results and len(results) > 0) or professional_results
                
                if not has_results:
                    st.error("❌ AutoML execution failed - no results generated")
                    st.warning("Please check your data and target column selection")
                    error_msg = st.session_state.get('automl_error', 'Unknown error')
                    st.error(f"Error details: {error_msg}")
                    return  # Don't transition to results
                
                # Transition to results
                st.session_state.app_stage = 'results'
                st.info("🔍 DEBUG: Transitioning to results stage")
                st.success("✅ AutoML completed successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ AutoML execution failed: {e}")
            st.info(f"🔍 DEBUG: Exception in _execute_automl: {e}")
            logger.error(f"AutoML execution error: {e}")

    def render_configuration_dashboard(self):
        """Render the comprehensive configuration dashboard."""
        # Add main container with CSS
        st.markdown("""
        <style>
        .main .block-container {
            max-width: 1400px;
            padding: 2rem 1rem;
            margin: 0 auto;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("⚙️ AutoML Configuration Dashboard")
        st.markdown("### Dataset-Aware Intelligent Configuration System")
        
        # Back button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("← Back to Dashboard", width="stretch"):
                st.session_state.show_configuration = False
                st.rerun()
        
        # Main configuration tabs
        config_tab1, config_tab2, config_tab3, config_tab4, config_tab5 = st.tabs([
            "📊 Dataset Analysis", 
            "🤖 Model Selection", 
            "⚙️ Optimization", 
            "🔧 Advanced", 
            "🚀 Execute"
        ])
        
        with config_tab1:
            self._render_dataset_analysis_tab()
        
        with config_tab2:
            self._render_model_selection_tab()
            
        with config_tab3:
            self._render_optimization_tab()
            
        with config_tab4:
            self._render_advanced_tab()
            
        with config_tab5:
            self._render_execution_tab()
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close configuration dashboard container
    
    def _render_dataset_analysis_tab(self):
        """Render dataset analysis and configuration tab."""
        st.header("📊 Dataset Analysis & Intelligent Configuration")
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload a dataset first to enable intelligent configuration.")
            st.info("💡 Upload your dataset from the main dashboard, then return to configuration.")
            return
        
        data = st.session_state.data
        
        # Dataset overview
        st.subheader("📈 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📋 Samples", f"{len(data):,}")
        with col2:
            st.metric("📐 Features", len(data.columns))
        with col3:
            memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("💾 Memory", f"{memory_mb:.1f} MB")
        with col4:
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            st.metric("❓ Missing %", f"{missing_pct:.1f}%")
        
        # Auto-analyze dataset if not done
        if not st.session_state.get('config_analyzed', False) or st.button("🔄 Re-analyze Dataset"):
            with st.spinner("🧠 Analyzing dataset characteristics..."):
                dataset_analysis = self._analyze_dataset_for_config(data)
                st.session_state.dataset_config = dataset_analysis
                st.session_state.config_analyzed = True
                st.rerun()
        
        if st.session_state.get('dataset_config'):
            self._display_dataset_analysis(st.session_state.dataset_config)
    
    def _analyze_dataset_for_config(self, data):
        """Comprehensive dataset analysis for configuration recommendations."""
        analysis = {
            'basic_stats': {},
            'data_quality': {},
            'complexity': {},
            'recommendations': {},
            'auto_config': {}
        }
        
        n_samples, n_features = data.shape
        
        # Basic statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        
        analysis['basic_stats'] = {
            'n_samples': n_samples,
            'n_features': n_features,
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'datetime_features': len(datetime_cols),
            'memory_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'sparsity': (data == 0).mean().mean() if len(numeric_cols) > 0 else 0
        }
        
        # Data quality analysis
        missing_values = data.isnull().sum()
        analysis['data_quality'] = {
            'missing_features': len(missing_values[missing_values > 0]),
            'missing_percentage': (missing_values.sum() / (n_samples * n_features)) * 100,
            'duplicate_rows': data.duplicated().sum(),
            'constant_features': len([col for col in data.columns if data[col].nunique() <= 1]),
            'high_cardinality_features': len([col for col in categorical_cols if data[col].nunique() > n_samples * 0.1])
        }
        
        # Complexity analysis
        feature_to_sample_ratio = n_features / n_samples
        analysis['complexity'] = {
            'feature_to_sample_ratio': feature_to_sample_ratio,
            'is_high_dimensional': feature_to_sample_ratio > 0.1,
            'is_sparse': analysis['basic_stats']['sparsity'] > 0.1,
            'is_large_dataset': n_samples > 50000,
            'is_wide_dataset': n_features > 1000,
            'estimated_training_time': self._estimate_training_time(n_samples, n_features)
        }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_config_recommendations(analysis)
        
        # Auto-configuration
        analysis['auto_config'] = self._generate_auto_config(analysis)
        
        return analysis
    
    def _estimate_training_time(self, n_samples, n_features):
        """Estimate training time based on dataset characteristics."""
        # Rough estimation based on experience
        base_time = 0.5  # minutes
        
        # Scale with samples (log scale)
        sample_factor = np.log10(max(n_samples, 100)) / 4
        
        # Scale with features
        feature_factor = n_features / 100
        
        # Estimated time per model
        time_per_model = base_time * sample_factor * feature_factor
        
        return {
            'per_model_minutes': max(0.1, min(time_per_model, 30)),  # Cap between 0.1 and 30 minutes
            'total_pipeline_minutes': max(2, min(time_per_model * 8, 120)),  # 8 models, cap at 2 hours
            'category': 'fast' if time_per_model < 2 else 'medium' if time_per_model < 10 else 'slow'
        }
    
    def _generate_config_recommendations(self, analysis):
        """Generate intelligent configuration recommendations."""
        recommendations = {
            'preprocessing': {},
            'model_selection': {},
            'optimization': {},
            'validation': {},
            'warnings': []
        }
        
        stats = analysis['basic_stats']
        quality = analysis['data_quality']
        complexity = analysis['complexity']
        
        # Preprocessing recommendations
        if complexity['is_high_dimensional']:
            recommendations['preprocessing']['dimensionality_reduction'] = {
                'enable': True,
                'method': 'tsvd' if complexity['is_sparse'] else 'pca',
                'variance_target': 0.95,
                'reason': f"High dimensionality ({stats['n_features']} features vs {stats['n_samples']} samples)"
            }
        
        if quality['missing_percentage'] > 5:
            recommendations['preprocessing']['imputation'] = {
                'strategy': 'advanced',
                'reason': f"Significant missing data ({quality['missing_percentage']:.1f}%)"
            }
        
        if complexity['is_sparse']:
            recommendations['preprocessing']['scaling'] = {
                'method': 'robust',
                'reason': f"Sparse data detected ({stats['sparsity']*100:.1f}% zeros)"
            }
        
        # Model selection recommendations
        if stats['n_samples'] < 1000:
            recommendations['model_selection']['focus'] = {
                'models': ['LogisticRegression', 'SVM', 'KNN'],
                'avoid': ['MLP', 'XGBoost'],
                'reason': 'Small dataset - prefer simpler models'
            }
        elif stats['n_samples'] > 50000:
            recommendations['model_selection']['focus'] = {
                'models': ['XGBoost', 'RandomForest', 'MLP'],
                'avoid': ['SVM'],
                'reason': 'Large dataset - use scalable algorithms'
            }
        
        # Optimization recommendations
        training_time = complexity['estimated_training_time']
        if training_time['category'] == 'slow':
            recommendations['optimization']['strategy'] = {
                'time_limit': 30,
                'trials_per_model': 50,
                'early_stopping': True,
                'reason': f"Large dataset detected - limit optimization time"
            }
        elif training_time['category'] == 'fast':
            recommendations['optimization']['strategy'] = {
                'time_limit': 60,
                'trials_per_model': 200,
                'comprehensive': True,
                'reason': f"Small/medium dataset - enable comprehensive optimization"
            }
        
        # Validation recommendations
        if stats['n_samples'] < 500:
            recommendations['validation']['strategy'] = {
                'cv_folds': 10,
                'repeats': 3,
                'reason': 'Small dataset - use more rigorous validation'
            }
        elif stats['n_samples'] > 10000:
            recommendations['validation']['strategy'] = {
                'cv_folds': 3,
                'repeats': 1,
                'reason': 'Large dataset - faster validation sufficient'
            }
        
        # Generate warnings
        if complexity['feature_to_sample_ratio'] > 1:
            recommendations['warnings'].append({
                'type': 'critical',
                'message': f"More features ({stats['n_features']}) than samples ({stats['n_samples']}) - curse of dimensionality!",
                'suggestion': 'Feature selection or dimensionality reduction essential'
            })
        
        if quality['missing_percentage'] > 20:
            recommendations['warnings'].append({
                'type': 'warning',
                'message': f"High missing data percentage ({quality['missing_percentage']:.1f}%)",
                'suggestion': 'Consider data collection improvement or advanced imputation'
            })
        
        return recommendations
    
    def _generate_auto_config(self, analysis):
        """Generate automatic configuration based on analysis."""
        stats = analysis['basic_stats']
        complexity = analysis['complexity']
        recommendations = analysis['recommendations']
        
        config = {
            'preprocessing': {
                'max_features': min(1000, stats['n_features']),
                'scaling': 'standard',
                'imputation_numeric': 'median',
                'imputation_categorical': 'most_frequent'
            },
            'dimensionality_reduction': {
                'enable': 'auto',
                'method': 'auto',
                'variance_target': 0.95
            },
            'model_selection': {
                'include_models': ['RandomForest', 'LogisticRegression', 'XGBoost', 'SVM', 'MLP'],
                'exclude_models': []
            },
            'optimization': {
                'time_minutes': 15,
                'max_trials': 100,
                'include_ensemble': True,
                'early_stopping': True
            },
            'validation': {
                'cv_folds': 5,
                'test_size': 0.2,
                'stratified': True
            }
        }
        
        # Apply intelligent adjustments
        if stats['n_samples'] > 50000:
            config['model_selection']['exclude_models'].append('SVM')
            config['optimization']['time_minutes'] = 30
        
        if stats['n_samples'] < 1000:
            config['model_selection']['exclude_models'].extend(['XGBoost', 'MLP'])
            config['validation']['cv_folds'] = 10
        
        if complexity['is_high_dimensional']:
            config['dimensionality_reduction']['enable'] = 'on'
            if complexity['is_sparse']:
                config['dimensionality_reduction']['method'] = 'tsvd'
            else:
                config['dimensionality_reduction']['method'] = 'pca'
        
        return config
    
    def _display_dataset_analysis(self, analysis):
        """Display the dataset analysis results."""
        st.subheader("🧠 AI Analysis Results")
        
        # Display warnings first
        if analysis['recommendations']['warnings']:
            st.warning("⚠️ **Critical Issues Detected:**")
            for warning in analysis['recommendations']['warnings']:
                if warning['type'] == 'critical':
                    st.error(f"🚨 {warning['message']}")
                    st.info(f"💡 **Recommendation**: {warning['suggestion']}")
                else:
                    st.warning(f"⚠️ {warning['message']}")
                    st.info(f"💡 **Suggestion**: {warning['suggestion']}")
        
        # Data characteristics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Data Characteristics")
            stats = analysis['basic_stats']
            complexity = analysis['complexity']
            
            characteristics = []
            if complexity['is_large_dataset']:
                characteristics.append("🔵 Large Dataset")
            elif stats['n_samples'] < 1000:
                characteristics.append("🟡 Small Dataset")
            else:
                characteristics.append("🟢 Medium Dataset")
            
            if complexity['is_high_dimensional']:
                characteristics.append("📐 High-Dimensional")
            
            if complexity['is_sparse']:
                characteristics.append("🕳️ Sparse Data")
                
            if complexity['is_wide_dataset']:
                characteristics.append("↔️ Wide Dataset")
            
            for char in characteristics:
                st.markdown(f"- {char}")
        
        with col2:
            st.markdown("#### ⏱️ Estimated Training Time")
            training_time = complexity['estimated_training_time']
            
            if training_time['category'] == 'fast':
                st.success(f"🚀 Fast: ~{training_time['per_model_minutes']:.1f} min/model")
            elif training_time['category'] == 'medium':
                st.info(f"⏳ Medium: ~{training_time['per_model_minutes']:.1f} min/model")
            else:
                st.warning(f"🐌 Slow: ~{training_time['per_model_minutes']:.1f} min/model")
            
            st.caption(f"Total pipeline: ~{training_time['total_pipeline_minutes']:.0f} minutes")
        
        # AI Recommendations
        st.subheader("🎯 AI Configuration Recommendations")
        
        recommendations = analysis['recommendations']
        
        if recommendations['model_selection'].get('focus'):
            focus = recommendations['model_selection']['focus']
            st.success(f"🤖 **Recommended Models**: {', '.join(focus['models'])}")
            st.caption(f"📝 Reason: {focus['reason']}")
            
            if focus.get('avoid'):
                st.warning(f"⚠️ **Avoid**: {', '.join(focus['avoid'])}")
        
        if recommendations['optimization'].get('strategy'):
            strategy = recommendations['optimization']['strategy']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Time Limit", f"{strategy['time_limit']} min")
            with col2:
                st.metric("🔄 Trials/Model", strategy.get('trials_per_model', 100))
            with col3:
                st.metric("⚡ Early Stop", "✅" if strategy.get('early_stopping', True) else "❌")
        
        # Auto-configuration preview
        st.subheader("⚙️ Recommended Configuration")
        auto_config = analysis['auto_config']
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.markdown("##### Preprocessing")
            st.code(f"""
Scaling: {auto_config['preprocessing']['scaling']}
Max Features: {auto_config['preprocessing']['max_features']:,}
Imputation: {auto_config['preprocessing']['imputation_numeric']}
            """.strip())
            
            st.markdown("##### Validation")
            st.code(f"""
CV Folds: {auto_config['validation']['cv_folds']}
Test Size: {auto_config['validation']['test_size']}
Stratified: {auto_config['validation']['stratified']}
            """.strip())
        
        with config_col2:
            st.markdown("##### Model Selection")
            included = [m for m in auto_config['model_selection']['include_models'] 
                       if m not in auto_config['model_selection']['exclude_models']]
            st.code(f"""
Included: {', '.join(included)}
Excluded: {', '.join(auto_config['model_selection']['exclude_models']) or 'None'}
            """.strip())
            
            st.markdown("##### Optimization")
            st.code(f"""
Time: {auto_config['optimization']['time_minutes']} minutes
Trials: {auto_config['optimization']['max_trials']}
Ensemble: {auto_config['optimization']['include_ensemble']}
            """.strip())
        
        # Apply recommendations button
        if st.button("✨ Apply AI Recommendations", type="primary", use_container_width=True):
            # Apply ALL auto config recommendations to session state
            auto_config = analysis['auto_config']
            
            # Update optimization config
            st.session_state.optimization_config.update(auto_config['optimization'])
            
            # Update model selection
            st.session_state.selected_models = [
                model for model in auto_config['model_selection']['include_models']
                if model not in auto_config['model_selection']['exclude_models']
            ]
            
            # Update advanced config
            st.session_state.advanced_config = {
                'preprocessing': auto_config['preprocessing'],
                'dimensionality_reduction': auto_config['dimensionality_reduction'],
                'validation': auto_config['validation'],
                'performance': {
                    'n_jobs': 1,
                    'enable_caching': True,
                    'memory_limit_gb': 8,
                    'gpu_enabled': False
                }
            }
            
            # Mark as applied
            st.session_state.ai_recommendations_applied = True
            
            st.success("✅ AI recommendations applied to all configuration tabs! Check other tabs to review and customize.")
            time.sleep(1)
            st.rerun()
    
    def _render_model_selection_tab(self):
        """Render model selection configuration tab."""
        st.header("🤖 Model Selection Configuration")
        
        # Get available models based on task type
        task_type = st.session_state.get('task_type', 'Classification')
        
        st.subheader("📋 Available Models")
        
        if task_type == "Classification":
            all_models = {
                'LogisticRegression': {'complexity': 'Low', 'speed': 'Fast', 'interpretability': 'High'},
                'SVM': {'complexity': 'Medium', 'speed': 'Medium', 'interpretability': 'Medium'},
                'RandomForest': {'complexity': 'Medium', 'speed': 'Fast', 'interpretability': 'Medium'},
                'XGBoost': {'complexity': 'High', 'speed': 'Fast', 'interpretability': 'Low'},
                'MLP': {'complexity': 'High', 'speed': 'Medium', 'interpretability': 'Low'},
                'KNN': {'complexity': 'Low', 'speed': 'Slow', 'interpretability': 'Medium'}
            }
        elif task_type == "Regression":
            all_models = {
                'LinearRegression': {'complexity': 'Low', 'speed': 'Fast', 'interpretability': 'High'},
                'SVR': {'complexity': 'Medium', 'speed': 'Medium', 'interpretability': 'Medium'},
                'RandomForest': {'complexity': 'Medium', 'speed': 'Fast', 'interpretability': 'Medium'},
                'XGBoost': {'complexity': 'High', 'speed': 'Fast', 'interpretability': 'Low'},
                'MLP': {'complexity': 'High', 'speed': 'Medium', 'interpretability': 'Low'},
                'KNN': {'complexity': 'Low', 'speed': 'Slow', 'interpretability': 'Medium'}
            }
        else:  # Clustering
            all_models = {
                'KMeans': {'complexity': 'Low', 'speed': 'Fast', 'interpretability': 'High'},
                'DBSCAN': {'complexity': 'Medium', 'speed': 'Medium', 'interpretability': 'Medium'},
                'GaussianMixture': {'complexity': 'Medium', 'speed': 'Medium', 'interpretability': 'Medium'},
                'AgglomerativeClustering': {'complexity': 'High', 'speed': 'Slow', 'interpretability': 'High'}
            }
        
        # Display model selection interface
        selected_models = st.multiselect(
            "Select Models to Include",
            options=list(all_models.keys()),
            default=list(all_models.keys()),
            help="Choose which models to include in the AutoML pipeline"
        )
        
        # Display model characteristics
        if selected_models:
            st.subheader("📊 Selected Models Characteristics")
            model_df = []
            for model in selected_models:
                char = all_models[model]
                model_df.append({
                    'Model': model,
                    'Complexity': char['complexity'],
                    'Speed': char['speed'],
                    'Interpretability': char['interpretability']
                })
            
            df = pd.DataFrame(model_df)
            st.dataframe(df, use_container_width=True)
        
        # Smart model selection
        st.subheader("🧠 Smart Model Selection")
        
        if st.session_state.get('dataset_config'):
            stats = st.session_state.dataset_config['basic_stats']
            n_samples = stats['n_samples']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🎯 Optimize for Accuracy", use_container_width=True):
                    if n_samples < 1000:
                        selected = ['LogisticRegression', 'SVM', 'RandomForest']
                    else:
                        selected = ['XGBoost', 'RandomForest', 'MLP']
                    st.info(f"Selected: {', '.join(selected)}")
            
            with col2:
                if st.button("⚡ Optimize for Speed", use_container_width=True):
                    if n_samples < 10000:
                        selected = ['LogisticRegression', 'RandomForest']
                    else:
                        selected = ['RandomForest', 'XGBoost']
                    st.info(f"Selected: {', '.join(selected)}")
        
        # Store selection
        st.session_state.selected_models = selected_models
    
    def _render_optimization_tab(self):
        """Render optimization configuration tab."""
        st.header("⚙️ Optimization Configuration")
        
        # Load current config
        opt_config = st.session_state.optimization_config
        
        st.subheader("⏱️ Time & Resource Allocation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time_minutes = st.slider(
                "Optimization Time (minutes)",
                min_value=5, max_value=120, 
                value=opt_config['time_minutes'],
                help="Total time allocated for hyperparameter optimization"
            )
            
            max_trials = st.slider(
                "Maximum Trials per Model",
                min_value=20, max_value=500,
                value=opt_config['max_trials'],
                help="Maximum number of hyperparameter combinations to try"
            )
        
        with col2:
            include_ensemble = st.checkbox(
                "🎭 Include Ensemble Models",
                value=opt_config['include_ensemble'],
                help="Create ensemble models from optimized base models"
            )
            
            early_stopping = st.checkbox(
                "⚡ Enable Early Stopping",
                value=opt_config.get('early_stopping', True),
                help="Stop optimization early if no improvement is observed"
            )
        
        # Advanced optimization features
        st.subheader("🔧 Advanced Optimization Features")
        
        advanced_features = st.multiselect(
            "Select Advanced Features",
            options=[
                "Multi-objective Optimization",
                "Automated Feature Engineering", 
                "Model Calibration",
                "Uncertainty Quantification",
                "Cross-validation Strategy Optimization"
            ],
            default=opt_config.get('advanced_features', []),
            help="Enable advanced ML engineering features"
        )
        
        # Optimization strategy
        st.subheader("📈 Optimization Strategy")
        
        strategy = st.radio(
            "Choose optimization focus",
            options=["Balanced", "Accuracy-focused", "Speed-focused", "Interpretability-focused"],
            help="Select the primary optimization objective"
        )
        
        # Dataset-aware recommendations
        if st.session_state.get('dataset_config'):
            stats = st.session_state.dataset_config['basic_stats']
            complexity = st.session_state.dataset_config['complexity']
            
            st.info("💡 **AI Recommendation based on your dataset:**")
            
            if stats['n_samples'] < 1000:
                st.warning("Small dataset detected - recommend shorter optimization time with more thorough validation")
                rec_time, rec_trials = 10, 50
            elif stats['n_samples'] > 50000:
                st.success("Large dataset detected - can afford longer optimization for better results")
                rec_time, rec_trials = 45, 200
            else:
                st.info("Medium dataset - balanced optimization recommended")
                rec_time, rec_trials = 20, 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Recommended Time", f"{rec_time} min")
            with col2:
                st.metric("🔄 Recommended Trials", rec_trials)
        
        # Update configuration
        st.session_state.optimization_config.update({
            'time_minutes': time_minutes,
            'max_trials': max_trials,
            'include_ensemble': include_ensemble,
            'early_stopping': early_stopping,
            'advanced_features': advanced_features,
            'strategy': strategy
        })
    
    def _render_advanced_tab(self):
        """Render advanced configuration tab."""
        st.header("🔧 Advanced Configuration")
        
        # Preprocessing settings
        st.subheader("🛠️ Data Preprocessing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Scaling & Normalization")
            scaling_method = st.selectbox(
                "Scaling Method",
                options=["standard", "minmax", "robust", "none"],
                index=0,
                help="Choose feature scaling method"
            )
            
            imputation_numeric = st.selectbox(
                "Numeric Imputation",
                options=["median", "mean", "most_frequent"],
                index=0,
                help="Strategy for filling missing numeric values"
            )
            
        with col2:
            st.markdown("##### Feature Engineering")
            max_features = st.number_input(
                "Maximum Features",
                min_value=10, max_value=10000,
                value=1000,
                help="Limit features for memory optimization"
            )
            
            remove_low_variance = st.checkbox(
                "Remove Low Variance Features",
                value=True,
                help="Remove features with very low variance"
            )
        
        # Dimensionality Reduction
        st.subheader("📐 Dimensionality Reduction")
        
        dimred_enable = st.radio(
            "Enable Dimensionality Reduction",
            options=["auto", "on", "off"],
            index=0,
            help="Auto: Enable for high-dimensional datasets"
        )
        
        if dimred_enable != "off":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                dimred_method = st.selectbox(
                    "Method",
                    options=["auto", "pca", "tsvd", "ipca"],
                    index=0,
                    help="Dimensionality reduction method"
                )
            
            with col2:
                variance_target = st.slider(
                    "Variance Target",
                    min_value=0.8, max_value=0.99,
                    value=0.95, step=0.01,
                    help="Target explained variance ratio"
                )
            
            with col3:
                k_max = st.number_input(
                    "Max Components",
                    min_value=10, max_value=1000,
                    value=256,
                    help="Maximum number of components"
                )
        
        # Validation Strategy
        st.subheader("✅ Validation Strategy")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cv_folds = st.slider(
                "CV Folds",
                min_value=3, max_value=20,
                value=5,
                help="Number of cross-validation folds"
            )
        
        with col2:
            test_size = st.slider(
                "Test Size",
                min_value=0.1, max_value=0.4,
                value=0.2, step=0.05,
                help="Proportion of data for testing"
            )
        
        with col3:
            random_seed = st.number_input(
                "Random Seed",
                min_value=0, max_value=9999,
                value=42,
                help="Seed for reproducibility"
            )
        
        # Performance & Memory
        st.subheader("🚀 Performance & Memory")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_jobs = st.slider(
                "Parallel Jobs",
                min_value=-1, max_value=16,
                value=1,
                help="-1 uses all available cores"
            )
            
            enable_caching = st.checkbox(
                "Enable Caching",
                value=True,
                help="Cache intermediate results for faster re-runs"
            )
        
        with col2:
            memory_limit_gb = st.slider(
                "Memory Limit (GB)",
                min_value=1, max_value=32,
                value=8,
                help="Maximum memory usage limit"
            )
            
            gpu_enabled = st.checkbox(
                "Enable GPU (if available)",
                value=False,
                help="Use GPU acceleration for compatible models"
            )
        
        # Store advanced configuration
        advanced_config = {
            'preprocessing': {
                'scaling_method': scaling_method,
                'imputation_numeric': imputation_numeric,
                'max_features': max_features,
                'remove_low_variance': remove_low_variance
            },
            'dimensionality_reduction': {
                'enable': dimred_enable,
                'method': dimred_method if dimred_enable != "off" else None,
                'variance_target': variance_target if dimred_enable != "off" else None,
                'k_max': k_max if dimred_enable != "off" else None
            },
            'validation': {
                'cv_folds': cv_folds,
                'test_size': test_size,
                'random_seed': random_seed
            },
            'performance': {
                'n_jobs': n_jobs,
                'enable_caching': enable_caching,
                'memory_limit_gb': memory_limit_gb,
                'gpu_enabled': gpu_enabled
            }
        }
        
        st.session_state.advanced_config = advanced_config
    
    def _render_execution_tab(self):
        """Render execution configuration and launch tab."""
        st.header("🚀 Execute AutoML Pipeline")
        
        # Configuration summary
        st.subheader("📋 Configuration Summary")
        
        # Create summary based on current configuration
        summary_data = []
        
        if st.session_state.get('optimization_config'):
            opt_config = st.session_state.optimization_config
            summary_data.extend([
                ("⏱️ Optimization Time", f"{opt_config['time_minutes']} minutes"),
                ("🔄 Max Trials", f"{opt_config['max_trials']} per model"),
                ("🎭 Ensemble", "Enabled" if opt_config['include_ensemble'] else "Disabled"),
                ("⚡ Early Stopping", "Enabled" if opt_config.get('early_stopping', True) else "Disabled")
            ])
        
        if st.session_state.get('selected_models'):
            summary_data.append(("🤖 Selected Models", f"{len(st.session_state.selected_models)} models"))
        
        # Display summary in a nice format
        if summary_data:
            col1, col2 = st.columns(2)
            
            for i, (key, value) in enumerate(summary_data):
                if i % 2 == 0:
                    col1.metric(key, value)
                else:
                    col2.metric(key, value)
        
        # Final checks and warnings
        st.subheader("🔍 Pre-flight Checks")
        
        checks_passed = 0
        total_checks = 3
        
        # Check 1: Dataset loaded
        if st.session_state.data is not None:
            st.success("✅ Dataset loaded and ready")
            checks_passed += 1
        else:
            st.error("❌ No dataset loaded")
        
        # Check 2: Task type selected
        if st.session_state.get('task_type'):
            st.success(f"✅ Task type: {st.session_state.task_type}")
            checks_passed += 1
        else:
            st.error("❌ Task type not selected")
        
        # Check 3: Target column (for supervised tasks)
        if st.session_state.get('task_type') in ['Classification', 'Regression']:
            if st.session_state.get('target_col'):
                st.success(f"✅ Target column: {st.session_state.target_col}")
                checks_passed += 1
            else:
                st.error("❌ Target column not selected")
        else:
            st.success("✅ Clustering task - no target needed")
            checks_passed += 1
        
        # Progress bar for readiness
        progress = checks_passed / total_checks
        st.progress(progress)
        st.caption(f"Readiness: {checks_passed}/{total_checks} checks passed")
        
        # Launch buttons
        st.subheader("🎯 Execute Pipeline")
        
        # Single unified run button that uses all user preferences
        if st.button("🚀 **Execute AutoML with My Preferences**", type="primary", use_container_width=True, key="unified_run_1"):
            if checks_passed >= 2:
                st.session_state.show_configuration = False
                
                # Determine execution mode using intelligent logic
                mode = self._determine_execution_mode()
                
                if mode == 'professional':
                    opt_config = st.session_state.optimization_config
                    with st.spinner("🔥 Running Professional AutoML with your configuration..."):
                        self.run_professional_automl(
                            optimization_time_minutes=opt_config['time_minutes'],
                            max_trials=opt_config['max_trials'],
                            include_ensemble=opt_config['include_ensemble'],
                            advanced_features=opt_config.get('advanced_features', [])
                        )
                else:
                    with st.spinner("⚡ Running Standard AutoML with your settings..."):
                        self.run_automl()
                st.rerun()
            else:
                st.error("Please complete required configuration first")
        
        if st.session_state.get('selected_models'):
            summary_data.append(("🤖 Selected Models", f"{len(st.session_state.selected_models)} models"))
        
        if st.session_state.get('advanced_config'):
            adv_config = st.session_state.advanced_config
            summary_data.extend([
                ("📐 Dimensionality Reduction", adv_config['dimensionality_reduction']['enable'].title()),
                ("✅ CV Folds", str(adv_config['validation']['cv_folds'])),
                ("🎲 Random Seed", str(adv_config['validation']['random_seed']))
            ])
        
        # Display summary in a nice format
        if summary_data:
            col1, col2 = st.columns(2)
            
            for i, (key, value) in enumerate(summary_data):
                if i % 2 == 0:
                    col1.metric(key, value)
                else:
                    col2.metric(key, value)
        
        # Estimated runtime
        if st.session_state.get('dataset_config'):
            training_time = st.session_state.dataset_config['complexity']['estimated_training_time']
            
            st.info(f"🕒 **Estimated Total Runtime**: ~{training_time['total_pipeline_minutes']:.0f} minutes")
            
            if training_time['category'] == 'slow':
                st.warning("⚠️ This configuration may take significant time. Consider reducing optimization time or number of models.")
            elif training_time['category'] == 'fast':
                st.success("🚀 This configuration should complete quickly!")
        
        # Final checks and warnings
        st.subheader("🔍 Pre-flight Checks")
        
        checks_passed = 0
        total_checks = 4
        
        # Check 1: Dataset loaded
        if st.session_state.data is not None:
            st.success("✅ Dataset loaded and ready")
            checks_passed += 1
        else:
            st.error("❌ No dataset loaded")
        
        # Check 2: Task type selected
        if st.session_state.get('task_type'):
            st.success(f"✅ Task type: {st.session_state.task_type}")
            checks_passed += 1
        else:
            st.error("❌ Task type not selected")
        
        # Check 3: Target column (for supervised tasks)
        if st.session_state.get('task_type') in ['Classification', 'Regression']:
            if st.session_state.get('target_col'):
                st.success(f"✅ Target column: {st.session_state.target_col}")
                checks_passed += 1
            else:
                st.error("❌ Target column not selected")
        else:
            st.success("✅ Clustering task - no target needed")
            checks_passed += 1
        
        # Check 4: Models selected
        if st.session_state.get('selected_models'):
            st.success(f"✅ Models selected: {len(st.session_state.selected_models)}")
            checks_passed += 1
        else:
            st.warning("⚠️ No specific models selected - will use defaults")
            checks_passed += 1
        
        # Progress bar for readiness
        progress = checks_passed / total_checks
        st.progress(progress)
        st.caption(f"Readiness: {checks_passed}/{total_checks} checks passed")
        
        # Launch buttons
        st.subheader("🎯 Execute Pipeline")
        
        # Single unified execution button
        if st.button("🚀 **Launch AutoML with All My Settings**", type="primary", use_container_width=True, key="unified_run_2"):
            st.info("🔍 DEBUG: Unified AutoML button clicked!")
            st.info(f"🔍 DEBUG: checks_passed = {checks_passed}")
            if checks_passed >= 3:
                st.session_state.show_configuration = False
                
                # Smart mode selection using helper method
                mode = self._determine_execution_mode()
                
                if mode == 'professional':
                    st.info("🔍 DEBUG: Using Professional AutoML mode based on configuration")
                    opt_config = st.session_state.optimization_config
                    with st.spinner("🔥 Running Professional AutoML with your preferences..."):
                        self.run_professional_automl(
                            optimization_time_minutes=opt_config['time_minutes'],
                            max_trials=opt_config['max_trials'],
                            include_ensemble=opt_config['include_ensemble'],
                            advanced_features=opt_config.get('advanced_features', [])
                        )
                else:
                    st.info("🔍 DEBUG: Using Standard AutoML mode")
                    with st.spinner("⚡ Running Standard AutoML with your settings..."):
                        self.run_automl()
                st.rerun()
            else:
                st.error("Please complete required configuration first")
        
        # Export configuration button (separate from run button)
        if st.button("📊 Export Configuration", use_container_width=True, key="export_config"):
                config_export = {
                    'dataset_info': st.session_state.get('dataset_config', {}),
                    'optimization': st.session_state.optimization_config,
                    'selected_models': st.session_state.get('selected_models', []),
                    'advanced': st.session_state.get('advanced_config', {}),
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                config_json = json.dumps(config_export, indent=2)
                st.download_button(
                    label="💾 Download Config JSON",
                    data=config_json,
                    file_name=f"automl_config_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    def _render_feature_engineering_section(self, data):
        """Render comprehensive feature engineering section."""
        st.markdown("### 🛠️ **Feature Engineering & Data Preparation**")
        st.markdown("*Edit your dataset before ML configuration. Changes will be applied to the pipeline.*")
        
        # Feature Engineering Tabs
        fe_tab1, fe_tab2, fe_tab3, fe_tab4, fe_tab5 = st.tabs([
            "📊 Column Selection", 
            "🧹 Data Cleaning", 
            "🔄 Transformations",
            "➕ Feature Creation",
            "💾 Export Changes"
        ])
        
        with fe_tab1:
            self._render_column_selection_tab(data)
        
        with fe_tab2:
            self._render_data_cleaning_tab(data)
            
        with fe_tab3:
            self._render_transformations_tab(data)
            
        with fe_tab4:
            self._render_feature_creation_tab(data)
            
        with fe_tab5:
            self._render_export_changes_tab(data)
    
    def _render_column_selection_tab(self, data):
        """Render column selection and removal interface."""
        st.subheader("📊 Column Selection & Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Select columns to keep in your dataset:**")
            
            # Initialize selected columns in session state
            if 'selected_columns' not in st.session_state:
                st.session_state.selected_columns = list(data.columns)
            
            # Column selection interface
            all_columns = list(data.columns)
            selected_cols = st.multiselect(
                "Choose columns for ML pipeline:",
                options=all_columns,
                default=st.session_state.selected_columns,
                help="Select only the columns you want to use for machine learning"
            )
            st.session_state.selected_columns = selected_cols
            
            if len(selected_cols) != len(all_columns):
                removed_cols = [col for col in all_columns if col not in selected_cols]
                st.warning(f"**Removing {len(removed_cols)} columns:** {', '.join(removed_cols[:5])}{'...' if len(removed_cols) > 5 else ''}")
            
        with col2:
            st.markdown("**Column Information:**")
            for col in data.columns:
                status = "✅" if col in st.session_state.selected_columns else "❌"
                missing_pct = (data[col].isnull().sum() / len(data)) * 100
                st.write(f"{status} `{col}` - {data[col].dtype} ({missing_pct:.1f}% missing)")
        
        # Preview of selected data
        if st.session_state.selected_columns:
            st.markdown("**Preview of selected columns:**")
            preview_data = data[st.session_state.selected_columns]
            st.dataframe(preview_data.head(), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Selected Columns", len(st.session_state.selected_columns))
            with col2:
                st.metric("Rows", len(preview_data))
            with col3:
                memory_mb = preview_data.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Memory", f"{memory_mb:.1f} MB")
    
    def _render_data_cleaning_tab(self, data):
        """Render data cleaning options."""
        st.subheader("🧹 Data Cleaning & Missing Value Handling")
        
        # Missing data analysis
        selected_cols = st.session_state.get('selected_columns', list(data.columns))
        clean_data = data[selected_cols] if selected_cols else data
        
        missing_info = clean_data.isnull().sum()
        if missing_info.sum() > 0:
            st.markdown("**Missing Data Summary:**")
            missing_df = pd.DataFrame({
                'Column': missing_info.index,
                'Missing Count': missing_info.values,
                'Missing %': (missing_info.values / len(clean_data)) * 100
            }).sort_values('Missing Count', ascending=False)
            
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            
            # Missing value handling options
            st.markdown("**Missing Value Strategy:**")
            
            col1, col2 = st.columns(2)
            with col1:
                missing_strategy = st.selectbox(
                    "Choose strategy for numeric columns:",
                    ["mean", "median", "mode", "drop_rows", "forward_fill"],
                    help="How to handle missing values in numeric columns"
                )
                st.session_state.missing_numeric_strategy = missing_strategy
            
            with col2:
                missing_categorical = st.selectbox(
                    "Choose strategy for categorical columns:",
                    ["mode", "unknown", "drop_rows"],
                    help="How to handle missing values in categorical columns"
                )
                st.session_state.missing_categorical_strategy = missing_categorical
        else:
            st.success("✅ No missing values detected in selected columns!")
        
        # Duplicate detection
        st.markdown("**Duplicate Detection:**")
        duplicates = clean_data.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates} duplicate rows ({duplicates/len(clean_data)*100:.1f}%)")
            remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
            st.session_state.remove_duplicates = remove_duplicates
        else:
            st.success("✅ No duplicate rows found!")
        
        # Outlier detection for numeric columns
        numeric_cols = clean_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.markdown("**Outlier Detection:**")
            
            outlier_method = st.selectbox(
                "Outlier detection method:",
                ["iqr", "z_score", "none"],
                help="Method to detect and handle outliers"
            )
            
            if outlier_method != "none":
                outlier_threshold = st.slider(
                    "Outlier sensitivity:", 
                    min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                    help="Lower values = more sensitive to outliers"
                )
                st.session_state.outlier_method = outlier_method
                st.session_state.outlier_threshold = outlier_threshold
    
    def _render_transformations_tab(self, data):
        """Render data transformation options."""
        st.subheader("🔄 Data Transformations")
        
        selected_cols = st.session_state.get('selected_columns', list(data.columns))
        clean_data = data[selected_cols] if selected_cols else data
        
        # Scaling options for numeric data
        numeric_cols = clean_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.markdown("**Numeric Data Scaling:**")
            
            scaling_method = st.selectbox(
                "Choose scaling method:",
                ["none", "standard", "minmax", "robust", "quantile"],
                help="Scaling method for numeric features"
            )
            st.session_state.scaling_method = scaling_method
            
            if scaling_method != "none":
                st.info(f"Will apply {scaling_method} scaling to numeric columns: {', '.join(numeric_cols)}")
        
        # Encoding options for categorical data
        categorical_cols = clean_data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.markdown("**Categorical Data Encoding:**")
            
            encoding_method = st.selectbox(
                "Choose encoding method:",
                ["auto", "onehot", "label", "target"],
                help="Encoding method for categorical features"
            )
            st.session_state.encoding_method = encoding_method
            
            # Show cardinality info
            st.markdown("**Categorical Column Cardinality:**")
            cardinality_info = []
            for col in categorical_cols:
                unique_count = clean_data[col].nunique()
                cardinality_info.append({
                    'Column': col,
                    'Unique Values': unique_count,
                    'Recommended': "One-Hot" if unique_count <= 10 else "Label/Target"
                })
            
            st.dataframe(pd.DataFrame(cardinality_info), use_container_width=True)
        
        # Advanced transformations
        st.markdown("**Advanced Transformations:**")
        
        col1, col2 = st.columns(2)
        with col1:
            log_transform = st.multiselect(
                "Apply log transformation to columns:",
                options=list(numeric_cols),
                help="Useful for highly skewed numeric data"
            )
            st.session_state.log_transform_cols = log_transform
        
        with col2:
            polynomial_features = st.checkbox(
                "Create polynomial features",
                help="Generate interaction and polynomial terms"
            )
            if polynomial_features:
                poly_degree = st.slider("Polynomial degree:", 2, 3, 2)
                st.session_state.polynomial_degree = poly_degree
            st.session_state.create_polynomial = polynomial_features
    
    def _render_feature_creation_tab(self, data):
        """Render feature creation interface."""
        st.subheader("➕ Feature Creation & Engineering")
        
        selected_cols = st.session_state.get('selected_columns', list(data.columns))
        clean_data = data[selected_cols] if selected_cols else data
        
        # Date/time feature extraction
        datetime_cols = clean_data.select_dtypes(include=['datetime64']).columns
        date_like_cols = [col for col in clean_data.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if len(datetime_cols) > 0 or len(date_like_cols) > 0:
            st.markdown("**Date/Time Feature Engineering:**")
            
            potential_date_cols = list(datetime_cols) + [col for col in date_like_cols if col not in datetime_cols]
            
            selected_date_cols = st.multiselect(
                "Extract features from date columns:",
                options=potential_date_cols,
                help="Extract year, month, day, weekday, etc. from date columns"
            )
            
            if selected_date_cols:
                date_features = st.multiselect(
                    "Select date features to create:",
                    ["year", "month", "day", "weekday", "quarter", "is_weekend"],
                    default=["year", "month", "weekday"]
                )
                st.session_state.date_features = {
                    'columns': selected_date_cols,
                    'features': date_features
                }
        
        # Text feature extraction
        text_cols = [col for col in clean_data.select_dtypes(include=['object']).columns 
                    if clean_data[col].str.len().mean() > 20]  # Likely text columns
        
        if len(text_cols) > 0:
            st.markdown("**Text Feature Engineering:**")
            
            selected_text_cols = st.multiselect(
                "Extract features from text columns:",
                options=text_cols,
                help="Extract length, word count, etc. from text columns"
            )
            
            if selected_text_cols:
                text_features = st.multiselect(
                    "Select text features to create:",
                    ["length", "word_count", "sentiment", "contains_numbers"],
                    default=["length", "word_count"]
                )
                st.session_state.text_features = {
                    'columns': selected_text_cols,
                    'features': text_features
                }
        
        # Mathematical combinations
        numeric_cols = clean_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            st.markdown("**Mathematical Feature Combinations:**")
            
            create_ratios = st.checkbox(
                "Create ratio features",
                help="Create ratios between numeric columns"
            )
            
            if create_ratios:
                ratio_pairs = st.multiselect(
                    "Select column pairs for ratios:",
                    options=[(f"{col1}/{col2}") for col1 in numeric_cols for col2 in numeric_cols if col1 != col2][:20],
                    help="Select pairs of columns to create ratio features"
                )
                st.session_state.ratio_features = ratio_pairs
        
        # Custom feature formula
        st.markdown("**Custom Feature Formula:**")
        custom_formula = st.text_input(
            "Create custom feature (e.g., 'col1 + col2 * 2'):",
            help="Use column names and basic math operators"
        )
        if custom_formula:
            custom_name = st.text_input("Feature name:", value="custom_feature")
            st.session_state.custom_feature = {
                'formula': custom_formula,
                'name': custom_name
            }
    
    def _render_export_changes_tab(self, data):
        """Render export and apply changes interface."""
        st.subheader("💾 Apply & Export Changes")
        
        # Show summary of all pending changes
        st.markdown("**Summary of Pending Changes:**")
        changes_summary = []
        
        # Column selection changes
        selected_cols = st.session_state.get('selected_columns', list(data.columns))
        if len(selected_cols) != len(data.columns):
            changes_summary.append(f"• Column selection: {len(selected_cols)}/{len(data.columns)} columns selected")
        
        # Data cleaning changes
        if st.session_state.get('missing_numeric_strategy'):
            changes_summary.append(f"• Missing values: {st.session_state.missing_numeric_strategy} for numeric")
        
        if st.session_state.get('remove_duplicates'):
            changes_summary.append("• Remove duplicate rows")
        
        # Transformation changes
        if st.session_state.get('scaling_method', 'none') != 'none':
            changes_summary.append(f"• Scaling: {st.session_state.scaling_method}")
        
        if st.session_state.get('encoding_method', 'auto') != 'auto':
            changes_summary.append(f"• Encoding: {st.session_state.encoding_method}")
        
        # Feature creation changes
        if st.session_state.get('date_features'):
            changes_summary.append("• Date feature extraction")
        
        if st.session_state.get('create_polynomial'):
            changes_summary.append("• Polynomial features")
        
        if changes_summary:
            for change in changes_summary:
                st.write(change)
        else:
            st.info("No changes pending. Select modifications in the tabs above.")
        
        # Apply changes button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Apply Changes to Dataset", type="primary", use_container_width=True):
                with st.spinner("Applying feature engineering changes..."):
                    try:
                        processed_data = self._apply_feature_engineering_changes(data)
                        st.session_state.data = processed_data
                        st.session_state.feature_engineering_applied = True
                        st.success(f"✅ Changes applied! New dataset: {processed_data.shape[0]} × {processed_data.shape[1]}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error applying changes: {e}")
        
        with col2:
            if st.button("📁 Export Modified Dataset", use_container_width=True):
                try:
                    processed_data = self._apply_feature_engineering_changes(data)
                    csv = processed_data.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"modified_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Error exporting: {e}")
        
        # Reset button
        if st.button("🔄 Reset All Changes", help="Clear all feature engineering settings"):
            self._reset_feature_engineering_settings()
            st.success("✅ All changes reset!")
            st.rerun()
    
    def _apply_feature_engineering_changes(self, data):
        """Apply all feature engineering changes to the dataset."""
        # Start with original data
        processed_data = data.copy()
        
        # 1. Column selection
        selected_cols = st.session_state.get('selected_columns', list(data.columns))
        if selected_cols:
            processed_data = processed_data[selected_cols]
        
        # 2. Handle missing values
        missing_numeric = st.session_state.get('missing_numeric_strategy')
        missing_categorical = st.session_state.get('missing_categorical_strategy')
        
        if missing_numeric:
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            if missing_numeric == 'mean':
                processed_data[numeric_cols] = processed_data[numeric_cols].fillna(processed_data[numeric_cols].mean())
            elif missing_numeric == 'median':
                processed_data[numeric_cols] = processed_data[numeric_cols].fillna(processed_data[numeric_cols].median())
            elif missing_numeric == 'mode':
                for col in numeric_cols:
                    processed_data[col].fillna(processed_data[col].mode().iloc[0] if not processed_data[col].mode().empty else 0, inplace=True)
        
        if missing_categorical:
            categorical_cols = processed_data.select_dtypes(include=['object', 'category']).columns
            if missing_categorical == 'mode':
                for col in categorical_cols:
                    processed_data[col].fillna(processed_data[col].mode().iloc[0] if not processed_data[col].mode().empty else 'Unknown', inplace=True)
            elif missing_categorical == 'unknown':
                processed_data[categorical_cols] = processed_data[categorical_cols].fillna('Unknown')
        
        # 3. Remove duplicates
        if st.session_state.get('remove_duplicates'):
            processed_data = processed_data.drop_duplicates()
        
        # 4. Create date features
        if st.session_state.get('date_features'):
            date_config = st.session_state.date_features
            for col in date_config['columns']:
                if col in processed_data.columns:
                    # Convert to datetime if not already
                    processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
                    
                    for feature in date_config['features']:
                        if feature == 'year':
                            processed_data[f'{col}_year'] = processed_data[col].dt.year
                        elif feature == 'month':
                            processed_data[f'{col}_month'] = processed_data[col].dt.month
                        elif feature == 'day':
                            processed_data[f'{col}_day'] = processed_data[col].dt.day
                        elif feature == 'weekday':
                            processed_data[f'{col}_weekday'] = processed_data[col].dt.dayofweek
                        elif feature == 'quarter':
                            processed_data[f'{col}_quarter'] = processed_data[col].dt.quarter
                        elif feature == 'is_weekend':
                            processed_data[f'{col}_is_weekend'] = (processed_data[col].dt.dayofweek >= 5).astype(int)
        
        # 5. Create text features
        if st.session_state.get('text_features'):
            text_config = st.session_state.text_features
            for col in text_config['columns']:
                if col in processed_data.columns:
                    for feature in text_config['features']:
                        if feature == 'length':
                            processed_data[f'{col}_length'] = processed_data[col].astype(str).str.len()
                        elif feature == 'word_count':
                            processed_data[f'{col}_word_count'] = processed_data[col].astype(str).str.split().str.len()
                        elif feature == 'contains_numbers':
                            processed_data[f'{col}_has_numbers'] = processed_data[col].astype(str).str.contains(r'\d').astype(int)
        
        # 6. Log transformations
        log_cols = st.session_state.get('log_transform_cols', [])
        for col in log_cols:
            if col in processed_data.columns:
                # Add small constant to avoid log(0)
                processed_data[f'{col}_log'] = np.log(processed_data[col] + 1)
        
        # 7. Ratio features
        if st.session_state.get('ratio_features'):
            for ratio in st.session_state.ratio_features:
                if '/' in ratio:
                    col1, col2 = ratio.split('/')
                    if col1 in processed_data.columns and col2 in processed_data.columns:
                        processed_data[f'{col1}_to_{col2}_ratio'] = processed_data[col1] / (processed_data[col2] + 1e-8)  # Avoid division by zero
        
        return processed_data
    
    def _reset_feature_engineering_settings(self):
        """Reset all feature engineering settings."""
        fe_keys = [
            'selected_columns', 'missing_numeric_strategy', 'missing_categorical_strategy',
            'remove_duplicates', 'outlier_method', 'outlier_threshold', 'scaling_method',
            'encoding_method', 'log_transform_cols', 'create_polynomial', 'polynomial_degree',
            'date_features', 'text_features', 'ratio_features', 'custom_feature'
        ]
        
        for key in fe_keys:
            if key in st.session_state:
                del st.session_state[key]
    
    def _render_step1_navigation(self):
        """Render Step 1 navigation controls with professional layout."""
        st.markdown("### 🚀 **Ready to Proceed?**")
        
        # Professional status dashboard
        if st.session_state.data is not None:
            data = st.session_state.data
            
            # Create status cards
            st.markdown("#### 📊 **Current Dataset Status**")
            
            col1, col2, col3, col4 = st.columns(4, gap="medium")
            
            with col1:
                st.metric(
                    label="📊 Dataset Size", 
                    value=f"{data.shape[0]:,} rows",
                    delta=f"{data.shape[1]} columns",
                    help="Current dataset dimensions"
                )
            
            with col2:
                memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric(
                    label="💾 Memory Usage", 
                    value=f"{memory_mb:.1f} MB",
                    help="Dataset memory footprint"
                )
            
            with col3:
                fe_applied = st.session_state.get('feature_engineering_applied', False)
                fe_status = "✅ Applied" if fe_applied else "⏳ None"
                fe_delta = "Modified" if fe_applied else "Original"
                st.metric(
                    label="🛠️ Feature Engineering", 
                    value=fe_status,
                    delta=fe_delta,
                    help="Feature engineering status"
                )
            
            with col4:
                ai_analyzed = st.session_state.get('ai_analysis') is not None
                ai_status = "✅ Complete" if ai_analyzed else "⏳ Pending"
                ai_delta = "Analyzed" if ai_analyzed else "Not analyzed"
                st.metric(
                    label="🧠 AI Analysis", 
                    value=ai_status,
                    delta=ai_delta,
                    help="AI analysis status"
                )
        
        st.markdown("---")
        
        # Professional navigation buttons
        col1, col2, col3 = st.columns([1, 1, 2], gap="medium")
        
        with col1:
            if st.button("🔄 Reset Dataset", help="Go back to original uploaded dataset", use_container_width=True):
                # Reset to original state
                self._reset_feature_engineering_settings()
                st.session_state.show_feature_engineering = False
                st.session_state.ai_analysis = None
                st.session_state.feature_engineering_applied = False
                st.success("✅ Reset to original dataset")
                st.rerun()
        
        with col2:
            if st.button("📁 Export Dataset", help="Download current dataset as CSV", use_container_width=True):
                if st.session_state.data is not None:
                    csv = st.session_state.data.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"prepared_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        with col3:
            # Only allow proceeding if we have data
            if st.session_state.data is not None:
                if st.button("➡️ **Continue to ML Configuration**", type="primary", use_container_width=True):
                    st.session_state.app_stage = 'configure'
                    st.rerun()
            else:
                st.button("➡️ Upload Dataset First", disabled=True, use_container_width=True, help="Please upload a dataset before proceeding")


# Main entry point for Streamlit
def main():
    """Main function to run the AutoML Dashboard."""
    # Initialize and run dashboard
    dashboard = AutoMLDashboard()
    dashboard.render()


if __name__ == "__main__":
    main()
