"""Streamlit UI Dashboard for AutoML-Insight."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import umap

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# NEW: Modular tab imports
from app.tabs import (
    DataOverviewTab,
    ExplainabilityTab,
    ModelsTab,
    PCAAnalysisTab,
    ProfessionalAutoMLTab,
    RecommendationTab,
    ReportTab,
)
from core.advanced_optimization import AutoMLPipeline  # NEW: Professional optimization
from core.ai_insights import get_ai_engine  # Standard AI insights
from core.ai_insights_enhanced import get_enhanced_ai_engine  # Enhanced AI insights
from core.data_profile import DataProfiler
from core.dimred import DimRedConfig  # NEW: Dimensionality reduction
from core.dimred_evaluator import DimRedEvaluator  # NEW: Enhanced evaluation with dimred
from core.evaluate_cls import ClassificationEvaluator
from core.evaluate_clu import ClusteringEvaluator
from core.meta_selector import MetaModelSelector
from core.preprocess import DataPreprocessor
from utils.cache_utils import (  # NEW: Caching utilities
    CachedDataLoader,
    cached_data_profile,
    cached_get_models,
    cached_preprocess,
    clear_cache,
    hash_dataframe,
)
from utils.logging_utils import setup_logger
from utils.seed_utils import set_seed

# Initialize logger
logger = setup_logger()


class AutoMLDashboard:
    """Main dashboard for AutoML-Insight."""

    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        defaults = {
            "data": None,
            "results": {},
            "models": {},
            "profiler": None,
            "ai_engine": None,  # Standard AI engine instance
            "enhanced_ai_engine": None,  # Enhanced AI engine instance
            "ai_insights": None,  # Store AI insights
            "config_analyzed": False,
            "dataset_config": {},
            "optimization_config": {
                "time_minutes": 15,
                "max_trials": 100,
                "include_ensemble": True,
                "advanced_features": [],
            },
            # NEW: App Stage Management
            "app_stage": "welcome",  # welcome, configure, results
            # NEW: Feature Engineering state
            "show_feature_engineering": False,
            "feature_engineering_applied": False,
            "selected_columns": None,
            "ai_analysis": None,
            # NEW: Dimensionality Reduction settings
            "dimred_enabled": "auto",  # off, on, auto
            "dimred_method": "auto",  # pca, tsvd, ipca, auto
            "dimred_variance_target": 0.95,
            "dimred_k_max": 256,
            "dimred_config": None,
            "dimred_results": None,
            # NEW: Class filtering settings
            "enable_class_filter": False,
            "min_class_samples": 5,
            # Random seed for reproducibility - user configurable
            "random_seed": 42,  # Initial default, will be overridden by user config
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def render(self):
        """Render the main dashboard with 3-stage navigation."""
        # Set page config for wide layout
        if not st.session_state.get("page_config_set", False):
            st.set_page_config(
                page_title="AutoML-Insight", layout="wide", page_icon="🤖", initial_sidebar_state="expanded"
            )
            st.session_state.page_config_set = True

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
                    if not st.session_state.get("ai_warning_shown", False):
                        st.sidebar.warning(
                            "⚠️ **AI Features Disabled**: Groq API key not found. Set GROQ_API_KEY in .env to enable AI insights."
                        )
                        st.session_state.ai_warning_shown = True
            except Exception as e:
                logger.warning(f"AI engine not available: {e}")
                st.session_state.ai_engine = False  # Mark as attempted
                if not st.session_state.get("ai_warning_shown", False):
                    st.sidebar.warning(f"⚠️ **AI Features Disabled**: {str(e)[:100]}")
                    st.session_state.ai_warning_shown = True

        # 3-Stage Navigation System
        app_stage = st.session_state.app_stage

        if app_stage == "welcome":
            self.render_welcome_stage()
        elif app_stage == "configure":
            self.render_configuration_stage()
        elif app_stage == "results":
            self.render_results_stage()
        else:
            # Fallback to welcome
            st.session_state.app_stage = "welcome"
            self.render_welcome_stage()

    def render_welcome_stage(self):
        """Render the welcome stage with app introduction and upload."""
        # Main container for better layout
        st.markdown(
            """
        <style>
        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            margin: 0 auto;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Stage indicator (the app title/tagline are already rendered once in app/main.py)
        st.markdown(
            """
        <div style='text-align: center; margin: 1rem auto 2rem; max-width: 1200px;'>
            <div style='color: #999; font-size: 1rem;'>Step 1 of 3 - Dataset Upload & Feature Engineering</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Feature showcase with better spacing
        st.markdown("<div style='max-width: 1200px; margin: 0 auto;'>", unsafe_allow_html=True)
        st.markdown("### ✨ **What AutoML-Insight Does for You**")

        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            st.markdown(
                """
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;'>
                <h3 style='margin: 0; font-size: 1.3rem;'>🧠 AI-Powered Analysis</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Dataset insights, recommendations, and optimization strategies powered by LLM</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;'>
                <h3 style='margin: 0; font-size: 1.3rem;'>⚙️ Smart Configuration</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Dataset-aware configuration with automatic parameter optimization</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;'>
                <h3 style='margin: 0; font-size: 1.3rem;'>🚀 Professional Results</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Enterprise-grade AutoML with explainability and comprehensive reports</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

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
                "🎮 **Use a Demo Dataset**",
                value=False,
                help="Try a built-in classic dataset - perfect for testing AutoML features",
            )

            if demo_mode:
                demo_choice = st.radio(
                    "Choose a demo dataset:", ["Iris (150 samples, 4 features)", "Wine (178 samples, 13 features)"]
                )
                st.markdown("**🎯 Demo Dataset Features:**")
                if demo_choice.startswith("Iris"):
                    st.markdown("""
                    - 🌸 Classic flower measurements dataset
                    - 🎯 3-class classification (species)
                    - 📊 4 numeric features
                    - 🔢 150 samples
                    """)
                else:
                    st.markdown("""
                    - 🍷 Wine chemical analysis dataset
                    - 🎯 3-class classification (cultivar)
                    - 📊 13 numeric features
                    - 🔢 178 samples
                    """)

                if st.button("🚀 **Load Demo Dataset**", type="primary", width="stretch"):
                    with st.spinner("📥 Loading demo dataset..."):
                        dataset_name = "iris" if demo_choice.startswith("Iris") else "wine"
                        data, target_col = CachedDataLoader.load_demo_dataset(dataset_name)
                        st.session_state.data = data
                        st.session_state.target_col = target_col
                        st.session_state.task_type = "Classification"
                        st.session_state.uploaded_file_name = f"demo_{dataset_name}.csv"
                        st.session_state.ai_insights = None
                        st.success(f"✅ {demo_choice.split(' (')[0]} dataset loaded successfully!")
                        st.session_state.app_stage = "configure"
                        st.rerun()
            else:
                # File uploader with better styling
                st.markdown("**📂 Upload Your CSV File:**")
                uploaded_file = st.file_uploader(
                    "Choose your CSV file",
                    type=["csv"],
                    help="Upload your dataset in CSV format to begin AutoML analysis",
                    key="main_uploader",
                    label_visibility="collapsed",
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
            st.markdown(
                """
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
            """,
                unsafe_allow_html=True,
            )

    def _render_dataset_overview_and_analyzer(self, data):
        """Render dataset overview and AI analyzer after upload."""
        st.markdown("---")
        st.markdown("<div style='max-width: 1200px; margin: 0 auto;'>", unsafe_allow_html=True)

        # Professional dataset statistics with enhanced metrics
        st.markdown("#### 📈 **Dataset Overview**")

        col1, col2, col3, col4 = st.columns(4, gap="medium")

        with col1:
            st.metric(
                label="📊 Total Rows", value=f"{len(data):,}", help="Number of samples/observations in your dataset"
            )
        with col2:
            st.metric(
                label="📐 Total Columns", value=len(data.columns), help="Number of features/variables in your dataset"
            )
        with col3:
            numeric_cols = data.select_dtypes(include=[np.number]).shape[1]
            categorical_cols = data.select_dtypes(include=["object", "category"]).shape[1]
            st.metric(
                label="🔢 Numeric Features",
                value=f"{numeric_cols}",
                delta=f"{categorical_cols} categorical",
                help="Distribution of numeric vs categorical features",
            )
        with col4:
            missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric(
                label="❓ Missing Data",
                value=f"{missing_percentage:.1f}%",
                delta=f"{memory_mb:.1f} MB",
                help="Percentage of missing values and dataset size",
            )

        # Enhanced Data Analysis Tabs with better organization
        st.markdown("### 📊 **Comprehensive Dataset Analysis**")
        st.markdown("*Explore your data through multiple analytical perspectives*")

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📋 **Preview & Info**",
                "📊 **Distribution Analysis**",
                "🔗 **Correlation Matrix**",
                "📈 **Data Quality Assessment**",
            ]
        )

        with tab1:
            st.subheader("📋 Dataset Preview")
            st.dataframe(data.head(10), width="stretch", height=300)

            st.subheader("📊 Column Information Summary")

            # Create columns info with better formatting
            info_data = []
            for col in data.columns:
                dtype = str(data[col].dtype)
                null_count = data[col].isnull().sum()
                null_pct = (null_count / len(data)) * 100
                unique_count = data[col].nunique()

                # Add color indicators for data types
                if "int" in dtype or "float" in dtype:
                    type_icon = "🔢"
                elif "object" in dtype:
                    type_icon = "📝"
                elif "datetime" in dtype:
                    type_icon = "📅"
                else:
                    type_icon = "❓"

                info_data.append(
                    {
                        "Column": f"{type_icon} {col}",
                        "Data Type": dtype,
                        "Non-Null Count": f"{len(data) - null_count:,}",
                        "Missing (%)": f"{null_pct:.1f}%",
                        "Unique Values": f"{unique_count:,}",
                    }
                )

            # Display as a styled dataframe
            info_df = pd.DataFrame(info_data)
            st.dataframe(
                info_df, width="stretch", height=min(400, len(info_df) * 35 + 50)  # Dynamic height based on rows
            )

        with tab2:
            st.subheader("Data Distribution Analysis")

            # Numeric columns analysis
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.markdown("**Numeric Columns Distribution**")

                for i in range(0, len(numeric_cols), 2):
                    cols = st.columns(2)
                    for j, col_name in enumerate(numeric_cols[i : i + 2]):
                        with cols[j]:
                            try:
                                import plotly.express as px

                                fig = px.histogram(data, x=col_name, title=f"Distribution of {col_name}")
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, width="stretch")
                            except Exception as e:
                                logger.debug(f"Histogram render failed for '{col_name}': {e}")
                                st.write(f"**{col_name}** - Basic Statistics:")
                                st.write(data[col_name].describe())

            # Categorical columns analysis
            categorical_cols = data.select_dtypes(include=["object", "category"]).columns
            if len(categorical_cols) > 0:
                st.markdown("**Categorical Columns Distribution**")

                for col_name in categorical_cols[:4]:  # Limit to first 4
                    value_counts = data[col_name].value_counts().head(10)
                    try:
                        import plotly.express as px

                        fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Top Values in {col_name}")
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, width="stretch")
                    except Exception as e:
                        logger.debug(f"Bar chart render failed for '{col_name}': {e}")
                        st.write(f"**{col_name}** - Top 10 Values:")
                        st.write(value_counts)

        with tab3:
            st.subheader("Correlation Analysis")

            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                try:
                    import plotly.express as px

                    corr_matrix = numeric_data.corr()
                    fig = px.imshow(
                        corr_matrix, title="Feature Correlation Matrix", aspect="auto", color_continuous_scale="RdBu_r"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width="stretch")

                    # Show strongest correlations
                    st.markdown("**Strongest Correlations:**")
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i + 1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.5:  # Strong correlation threshold
                                corr_pairs.append(
                                    {
                                        "Feature 1": corr_matrix.columns[i],
                                        "Feature 2": corr_matrix.columns[j],
                                        "Correlation": f"{corr_val:.3f}",
                                    }
                                )

                    if corr_pairs:
                        st.dataframe(pd.DataFrame(corr_pairs), width="stretch")
                    else:
                        st.info("No strong correlations (>0.5) found between features.")

                except Exception as e:
                    logger.debug(f"Correlation pairs table failed, falling back to raw matrix: {e}")
                    st.write("Correlation matrix:")
                    st.dataframe(numeric_data.corr(), width="stretch")
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")

        with tab4:
            st.subheader("Data Quality Assessment")

            # Missing data analysis
            missing_data = data.isnull().sum()
            missing_pct = (missing_data / len(data)) * 100
            missing_df = pd.DataFrame(
                {
                    "Column": missing_data.index,
                    "Missing Count": missing_data.values,
                    "Missing Percentage": missing_pct.values,
                }
            ).sort_values("Missing Count", ascending=False)

            if missing_df["Missing Count"].sum() > 0:
                st.markdown("**Missing Data Analysis:**")
                try:
                    import plotly.express as px

                    fig = px.bar(
                        missing_df[missing_df["Missing Count"] > 0],
                        x="Column",
                        y="Missing Percentage",
                        title="Missing Data by Column",
                    )
                    fig.update_layout(height=400, xaxis_tickangle=45)
                    st.plotly_chart(fig, width="stretch")
                except Exception as e:
                    logger.debug(f"Missing-data chart render failed: {e}")
                    st.dataframe(missing_df[missing_df["Missing Count"] > 0], width="stretch")
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
                for col in data.select_dtypes(include=["object"]).columns:
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
            ai_button_text = (
                "🔄 Re-analyze with AI" if st.session_state.get("ai_analysis") else "🔍 Analyze Dataset with AI"
            )
            if st.button(
                ai_button_text, type="secondary", width="stretch", help="Get AI-powered insights about your dataset"
            ):
                with st.spinner("🤖 AI is analyzing your dataset..."):
                    try:
                        analysis = self._generate_ai_dataset_analysis(data)
                        st.session_state.ai_analysis = analysis
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ AI Analysis failed: {e}")

        with col2:
            fe_button_text = (
                "🔄 Modify Features"
                if st.session_state.get("feature_engineering_applied")
                else "🛠️ Feature Engineering"
            )
            if st.button(fe_button_text, type="secondary", width="stretch", help="Edit and transform your dataset"):
                st.session_state.show_feature_engineering = True
                st.rerun()

        with col3:
            # Status indicator
            ai_status = "✅ Complete" if st.session_state.get("ai_analysis") else "⏳ Pending"
            fe_status = "✅ Applied" if st.session_state.get("feature_engineering_applied") else "⏳ None"
            st.markdown(f"**Analysis:** {ai_status}")
            st.markdown(f"**Engineering:** {fe_status}")

        # Display AI Analysis if available
        if st.session_state.get("ai_analysis") or st.session_state.get("ai_insights"):
            # Check both possible sources of AI insights
            analysis = st.session_state.get("ai_analysis") or st.session_state.get("ai_insights")

            with st.expander("🎯 **AI Insights & Recommendations**", expanded=True):
                if isinstance(analysis, dict):
                    # Task type recommendation
                    if "task_recommendation" in analysis:
                        task_rec = analysis["task_recommendation"]
                        st.success(
                            f"**Recommended Task:** {task_rec['task']} ({task_rec['confidence']:.0%} confidence)"
                        )
                        st.info(f"**Reasoning:** {task_rec['reasoning']}")

                    # Dataset Overview
                    if "dataset_overview" in analysis:
                        st.markdown("---")
                        st.markdown("### 📊 **Dataset Overview**")
                        overview = analysis["dataset_overview"]

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📝 Samples", f"{overview.get('samples', 0):,}")
                            st.metric("📊 Features", f"{overview.get('features', 0)}")
                        with col2:
                            st.metric("🔢 Numeric", f"{overview.get('numeric_features', 0)}")
                            st.metric("📝 Categorical", f"{overview.get('categorical_features', 0)}")
                        with col3:
                            st.metric("❓ Missing", overview.get("missing_percentage", 0))
                            st.metric("📋 Duplicates", overview.get("duplicate_rows", 0))
                        with col4:
                            st.metric("💾 Memory", f"{overview.get('memory_mb', 0)} MB")
                            st.metric("📅 DateTime", f"{overview.get('datetime_features', 0)}")

                    # Data Quality
                    if "data_quality" in analysis:
                        st.markdown("---")
                        st.markdown("### 🔍 **Data Quality Assessment**")
                        quality = analysis["data_quality"]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Missing Data:** {quality.get('missing_data', 'None')}")
                            st.write(f"**Duplicates:** {quality.get('duplicates', 'None')}")
                        with col2:
                            correlations = quality.get("high_correlations", "None")
                            if isinstance(correlations, list) and correlations:
                                st.write("**High Correlations:**")
                                for corr in correlations[:3]:
                                    st.write(f"  • {corr}")
                            else:
                                st.write(f"**High Correlations:** {correlations}")

                    # Target column suggestions
                    if "target_suggestions" in analysis and analysis["target_suggestions"]:
                        st.markdown("---")
                        st.markdown("### 🎯 **Potential Target Columns**")
                        for i, suggestion in enumerate(analysis["target_suggestions"][:5], 1):
                            st.write(f"**{i}. `{suggestion['column']}`**")
                            st.write(f"   {suggestion['reasoning']}")
                            st.write("")

                    # Quality Issues
                    if "quality_issues" in analysis and analysis["quality_issues"]:
                        st.markdown("---")
                        st.markdown("### ⚠️ **Quality Issues Detected**")
                        for issue in analysis["quality_issues"]:
                            if "⚠️" in issue or "ℹ️" in issue:
                                st.warning(issue)
                            else:
                                st.info(issue)

                    # Recommendations
                    if "recommendations" in analysis and analysis["recommendations"]:
                        st.markdown("---")
                        st.markdown("### 💡 **Recommendations**")
                        for rec in analysis["recommendations"]:
                            if "✅" in rec:
                                st.success(rec)
                            else:
                                st.info(f"💡 {rec}")

                    # Enhanced AI insights display (for enhanced analysis)
                    if isinstance(analysis, dict) and any(
                        key.startswith(("key_", "critical_")) for key in analysis.keys()
                    ):
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
        if st.session_state.get("show_feature_engineering", False):
            self._render_feature_engineering_section(data)

        # Professional Navigation Section
        st.markdown("---")
        self._render_step1_navigation()

        st.markdown("</div>", unsafe_allow_html=True)  # Close dataset overview container

    def _generate_ai_dataset_analysis(self, data):
        """Generate AI-powered dataset analysis and recommendations."""
        try:
            # Try enhanced AI analysis first
            if st.session_state.get("enhanced_ai_engine"):
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
        """Generate comprehensive intelligent dataset analysis based on actual data characteristics."""
        try:
            # Deep dataset analysis
            n_rows, n_cols = data.shape
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
            datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()
            missing_data = data.isnull().sum().sum()

            # Calculate detailed statistics
            missing_pct = (missing_data / (n_rows * n_cols)) * 100 if n_rows * n_cols > 0 else 0
            duplicate_rows = data.duplicated().sum()

            # Analyze correlations for numeric data
            correlations = []
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr().abs()
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        corr_val = corr_matrix.iloc[i, j]
                        if corr_val > 0.7:
                            correlations.append(f"{numeric_cols[i]} ↔ {numeric_cols[j]} ({corr_val:.2f})")

            # Intelligent task recommendation
            task_type = "Classification"
            confidence = 0.5
            reasoning = ""

            classification_score = 0
            regression_score = 0
            clustering_score = 0

            # Detailed column analysis
            target_candidates = []
            for col in data.columns:
                unique_count = data[col].nunique()
                unique_ratio = unique_count / len(data)

                # Classification indicators
                if data[col].dtype == "object" or data[col].dtype.name == "category":
                    if 2 <= unique_count <= 50 and unique_ratio < 0.5:
                        classification_score += 3
                        if unique_count <= 20:
                            target_candidates.append(
                                {
                                    "column": col,
                                    "type": "classification",
                                    "unique_values": unique_count,
                                    "reasoning": f"Categorical with {unique_count} classes ({unique_ratio:.1%} unique) - ideal for classification",
                                    "score": 10 - unique_count,  # Prefer fewer classes
                                }
                            )
                elif data[col].dtype in ["int64", "int32"]:
                    if 2 <= unique_count <= 20:
                        classification_score += 2
                        target_candidates.append(
                            {
                                "column": col,
                                "type": "classification",
                                "unique_values": unique_count,
                                "reasoning": f"Integer with {unique_count} discrete values - suitable for classification",
                                "score": 8 - unique_count,
                            }
                        )
                    elif unique_count > 50 and unique_ratio > 0.5:
                        regression_score += 1
                        target_candidates.append(
                            {
                                "column": col,
                                "type": "regression",
                                "unique_values": unique_count,
                                "reasoning": f"Integer with {unique_count} values ({unique_ratio:.1%} unique) - suitable for regression",
                                "score": min(10, int(unique_ratio * 10)),
                            }
                        )

                # Regression indicators
                if data[col].dtype in ["float64", "float32"]:
                    if unique_ratio > 0.8:
                        regression_score += 2
                        target_candidates.append(
                            {
                                "column": col,
                                "type": "regression",
                                "unique_values": unique_count,
                                "reasoning": f"Continuous numeric ({unique_ratio:.1%} unique values) - ideal for regression",
                                "score": int(unique_ratio * 10),
                            }
                        )

            # Clustering indicators
            if len(categorical_cols) == 0 and len(numeric_cols) >= 3:
                clustering_score += 2
            if n_cols >= 5 and all(data[col].dtype in ["float64", "int64", "int32", "float32"] for col in data.columns):
                clustering_score += 1

            # Determine task type
            scores = {
                "Classification": classification_score,
                "Regression": regression_score,
                "Clustering": clustering_score,
            }

            max_score = max(scores.values())
            if max_score > 0:
                task_type = max(scores, key=scores.get)
                total_score = sum(scores.values())
                confidence = min(0.95, 0.5 + (max_score / max(1, total_score)) * 0.45)

            # Generate detailed reasoning
            if task_type == "Classification":
                cat_count = len(categorical_cols)
                discrete_count = len([col for col in data.columns if data[col].nunique() <= 20])
                reasoning = f"Dataset has {cat_count} categorical columns and {discrete_count} columns with discrete values, indicating classification task"
            elif task_type == "Regression":
                cont_count = len([col for col in data.columns if data[col].dtype in ["float64", "float32"]])
                high_card = len([col for col in numeric_cols if data[col].nunique() > 50])
                reasoning = f"Dataset has {cont_count} continuous numeric columns and {high_card} high-cardinality features, suggesting regression task"
            else:
                reasoning = f"Dataset has {len(numeric_cols)} numeric features with no clear target column, suitable for clustering analysis"

            # Sort and select top target candidates
            target_candidates.sort(key=lambda x: x["score"], reverse=True)
            top_targets = target_candidates[:5]  # Top 5 candidates

            # Build comprehensive analysis
            analysis = {
                "task_recommendation": {"task": task_type, "confidence": confidence, "reasoning": reasoning},
                "target_suggestions": [{"column": t["column"], "reasoning": t["reasoning"]} for t in top_targets],
                "dataset_overview": {
                    "samples": n_rows,
                    "features": n_cols,
                    "numeric_features": len(numeric_cols),
                    "categorical_features": len(categorical_cols),
                    "datetime_features": len(datetime_cols),
                    "missing_percentage": round(missing_pct, 2),
                    "duplicate_rows": duplicate_rows,
                    "memory_mb": round(data.memory_usage(deep=True).sum() / 1024**2, 2),
                },
                "data_quality": {
                    "missing_data": f"{missing_pct:.1f}%" if missing_pct > 0 else "None",
                    "duplicates": (
                        f"{duplicate_rows} rows ({duplicate_rows/n_rows*100:.1f}%)" if duplicate_rows > 0 else "None"
                    ),
                    "high_correlations": correlations[:5] if correlations else "No strong correlations detected",
                },
                "quality_issues": [],
                "recommendations": [],
            }

            # Detailed quality checks
            if missing_pct > 10:
                analysis["quality_issues"].append(
                    f"⚠️ High missing data ({missing_pct:.1f}%) - imputation or removal needed"
                )
                analysis["recommendations"].append("Apply SimpleImputer or KNNImputer for missing values")

            if duplicate_rows > n_rows * 0.05:
                analysis["quality_issues"].append(
                    f"⚠️ {duplicate_rows} duplicate rows detected ({duplicate_rows/n_rows*100:.1f}%)"
                )
                analysis["recommendations"].append("Remove duplicate rows before training")

            # Check for high cardinality categorical
            for col in categorical_cols:
                if data[col].nunique() > 50:
                    analysis["quality_issues"].append(
                        f'⚠️ Column "{col}" has {data[col].nunique()} categories - high cardinality'
                    )
                    analysis["recommendations"].append(
                        f'Consider target encoding or grouping rare categories for "{col}"'
                    )

            # Check for constant columns
            constant_cols = [col for col in data.columns if data[col].nunique() == 1]
            if constant_cols:
                analysis["quality_issues"].append(
                    f'⚠️ {len(constant_cols)} constant columns: {", ".join(constant_cols[:3])}'
                )
                analysis["recommendations"].append("Remove constant columns - they provide no information")

            # Check for skewed distributions
            if len(numeric_cols) > 0:
                from scipy import stats as scipy_stats

                skewed_cols = []
                for col in numeric_cols:
                    skewness = scipy_stats.skew(data[col].dropna())
                    if abs(skewness) > 1:
                        skewed_cols.append(f"{col} (skew={skewness:.2f})")

                if skewed_cols:
                    analysis["quality_issues"].append(f'ℹ️ Skewed distributions in: {", ".join(skewed_cols[:3])}')
                    analysis["recommendations"].append(
                        "Consider log transformation or power transformation for skewed features"
                    )

            # Add positive insights
            if missing_pct < 5:
                analysis["recommendations"].append("✅ Low missing data - dataset is clean")
            if len(correlations) == 0:
                analysis["recommendations"].append("✅ No multicollinearity issues detected")
            if n_rows > 1000:
                analysis["recommendations"].append("✅ Good sample size for model training")

            return analysis

        except Exception as e:
            logger.error(f"Comprehensive dataset analysis failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {
                "task_recommendation": {
                    "task": "Analysis",
                    "confidence": 0.5,
                    "reasoning": "Analysis completed with limited information",
                },
                "target_suggestions": [],
                "quality_issues": [f"Analysis error: {str(e)}"],
            }

    def _display_enhanced_ai_insights(self, insights):
        """Display enhanced AI insights in organized format."""
        if not insights:
            st.info("No AI insights available.")
            return

        # Dataset Overview
        if "dataset_overview" in insights:
            st.markdown("#### 📊 **Dataset Overview**")
            overview = insights["dataset_overview"]
            if isinstance(overview, dict):
                col1, col2 = st.columns(2)
                with col1:
                    if "summary" in overview:
                        st.write(overview["summary"])
                with col2:
                    if "recommendations" in overview:
                        for rec in overview["recommendations"][:3]:
                            st.write(f"• {rec}")
            else:
                st.write(overview)

        # Key Strengths
        if "key_strengths" in insights:
            st.markdown("#### ✅ **Dataset Strengths**")
            strengths = insights["key_strengths"]
            if isinstance(strengths, list):
                for strength in strengths:
                    st.success(f"✓ {strength}")
            else:
                st.success(strengths)

        # Critical Challenges
        if "critical_challenges" in insights:
            st.markdown("#### ⚠️ **Areas for Improvement**")
            challenges = insights["critical_challenges"]
            if isinstance(challenges, list):
                for challenge in challenges:
                    st.warning(f"⚠ {challenge}")
            else:
                st.warning(challenges)

        # Preprocessing Strategy
        if "preprocessing_strategy" in insights:
            st.markdown("#### 🔧 **Recommended Preprocessing**")
            strategy = insights["preprocessing_strategy"]
            if isinstance(strategy, list):
                for step in strategy:
                    st.info(f"🔧 {step}")
            else:
                st.info(strategy)

        # Model Recommendations
        if "recommended_models" in insights:
            st.markdown("#### 🤖 **Recommended Models**")
            models = insights["recommended_models"]
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

    def run_automl(self):
        """Run the AutoML pipeline."""
        logger.debug("run_automl() started")

        try:
            data = st.session_state.data
            task_type = st.session_state.task_type

            logger.debug(f"Task type: {task_type}, Data shape: {data.shape}")
            st.info(f"⚙️ Using your configured random seed: {st.session_state.random_seed}")
            st.info(f"⚙️ Dimensionality reduction: {st.session_state.get('dimred_enabled', 'auto')} mode")

            # Create dimensionality reduction config from UI (moved to top for scope)
            dimred_config = DimRedConfig(
                enable=st.session_state.get("dimred_enabled", "auto"),
                method=st.session_state.get("dimred_method", "auto"),
                variance_target=st.session_state.get("dimred_variance_target", 0.95),
                k_max=st.session_state.get("dimred_k_max", 256),
                whiten=True,
                seed=st.session_state.random_seed,  # Use actual user-configured value
            )

            # Use user-configured max_features or recommended default
            max_features = st.session_state.get("recommended_config", {}).get("recommended_max_features", 1000)

            # Override with user's explicit configuration if available
            user_max_features = st.session_state.get("dataset_config", {}).get("max_features")
            if user_max_features:
                max_features = user_max_features
                st.info(f"⚙️ Using your configured max features: {max_features}")
            else:
                st.info(f"⚙️ Using recommended max features: {max_features}")

            # Profile data with caching
            st.info("📊 Profiling dataset...")
            profiler = DataProfiler()

            if task_type == "Classification":
                target_col = st.session_state.target_col
                X = data.drop(columns=[target_col])
                y = data[target_col]

                # Check if target is actually continuous (regression problem)
                n_unique = y.nunique()
                n_samples = len(y)

                # CRITICAL: Check if target looks like an ID column
                if n_unique > 100 and n_unique > n_samples * 0.8:
                    st.error("❌ **Invalid Target Column Detected!**")
                    st.error(f"Target column has {n_unique:,} unique values out of {n_samples:,} samples.")
                    st.error("This appears to be an **ID column**, not a classification target!")
                    st.warning(
                        "💡 **Solution**: Select a different column with 2-20 unique categories (e.g., 'diagnosis', 'class', 'label')"
                    )
                    st.session_state.results = {}
                    st.session_state.automl_error = (
                        f"Target column appears to be an ID column with {n_unique} unique values"
                    )
                    return

                # If >50% unique values and they're numeric, it's likely regression
                import pandas as pandas_api

                if n_unique / n_samples > 0.5 and pandas_api.api.types.is_numeric_dtype(y):
                    st.warning("⚠️ **Potential Task Type Issue Detected!**")
                    st.warning(f"Your target has {n_unique:,} unique continuous values out of {n_samples:,} samples.")
                    st.warning("This looks like it might be a **REGRESSION** problem, not classification!")
                    st.info(
                        "💡 **Continuing with classification anyway**. Consider changing 'Task Type' to 'Regression' for better results."
                    )

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
                    st.info(
                        f"📊 Classes: {n_unique} total (range: {min(class_counts_before.keys())} to {max(class_counts_before.keys())})"
                    )

                # Use cached profiling
                data_hash = hash_dataframe(data)
                profile = cached_data_profile(profiler, data_hash, X, y)
            else:
                X = data
                y = None
                # Use cached profiling
                data_hash = hash_dataframe(data)
                profile = cached_data_profile(profiler, data_hash, X)

            st.session_state.profiler = profiler
            st.session_state.profile = profile

            # Preprocess with smart feature selection and dimred (with caching)
            with st.spinner("🔧 Preprocessing data..."):
                preprocessor = DataPreprocessor(max_features=max_features, dimred_config=dimred_config)
                # Use cached preprocessing
                data_hash = hash_dataframe(data)
                X_processed, y_processed = cached_preprocess(preprocessor, data_hash, X, y)

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
                    st.info(
                        f"📊 Classes: {unique_classes_after} total, samples per class: {min_class_count}-{max(class_counts_after.values())}"
                    )
                else:
                    st.info(f"📊 Classes: {unique_classes_after:,} total, min samples per class: {min_class_count}")

                # Verify labels are contiguous (0, 1, 2, ..., n-1)
                unique_classes = sorted(set(y_processed))
                expected_classes = list(range(len(unique_classes)))
                if unique_classes != expected_classes:
                    st.error("❌ Non-contiguous class labels detected!")
                    st.error(f"Expected: {expected_classes[:10]}... Got: {unique_classes[:10]}...")
                    st.warning("This should have been fixed by LabelEncoder. Please check your data.")
                    return

            st.session_state.preprocessor = preprocessor
            st.session_state.X_processed = X_processed
            st.session_state.y_processed = y_processed

            # CRITICAL: Validate processed data doesn't have NaN values
            if isinstance(X_processed, pd.DataFrame):
                nan_count = X_processed.isnull().sum().sum()
            else:
                nan_count = np.isnan(X_processed).sum()

            if nan_count > 0:
                st.error(f"❌ Preprocessing failed: {nan_count} NaN values detected in processed data!")
                st.error("This should not happen. The preprocessor should handle all missing values.")
                st.info("💡 Applying emergency imputation...")

                # Emergency imputation
                from sklearn.impute import SimpleImputer

                imputer = SimpleImputer(strategy="median")
                if isinstance(X_processed, pd.DataFrame):
                    X_processed = pd.DataFrame(
                        imputer.fit_transform(X_processed), columns=X_processed.columns, index=X_processed.index
                    )
                else:
                    X_processed = imputer.fit_transform(X_processed)

                st.session_state.X_processed = X_processed
                st.success("✅ Emergency imputation completed")

            # NEW: Create train/test split for proper evaluation
            from sklearn.model_selection import train_test_split

            if task_type == "Classification":
                # Auto-enable class filtering for high-cardinality datasets
                unique_classes_count = len(set(y_processed))
                total_samples_count = len(y_processed)

                # Auto-enable filtering if we have too many classes
                auto_filter_needed = (
                    unique_classes_count > 1000  # More than 1000 classes
                    or unique_classes_count > total_samples_count * 0.1  # More than 10% unique classes
                )

                if auto_filter_needed and not hasattr(st.session_state, "enable_class_filter"):
                    st.session_state.enable_class_filter = True
                    st.session_state.min_class_samples = max(2, total_samples_count // unique_classes_count)
                    st.warning(f"""
🛠️ **Auto-Filter Enabled**

Detected high-cardinality target ({unique_classes_count:,} classes for {total_samples_count:,} samples).
Auto-filtering classes with <{st.session_state.min_class_samples} samples to prevent train-test split errors.
                    """)

                # Apply proactive class filtering if enabled
                valid_raw_labels = None
                if hasattr(st.session_state, "enable_class_filter") and st.session_state.enable_class_filter:
                    min_samples = st.session_state.get("min_class_samples", 5)
                    original_class_counts = Counter(y_processed)

                    # Filter out classes with insufficient samples
                    valid_classes = {
                        class_label: count
                        for class_label, count in original_class_counts.items()
                        if count >= min_samples
                    }

                    if len(valid_classes) >= 2:  # Need at least 2 classes
                        # Filter the data - FIX: Use np.isin for numpy arrays
                        if isinstance(y_processed, pd.Series):
                            mask = y_processed.isin(valid_classes.keys())
                        else:
                            mask = np.isin(y_processed, list(valid_classes.keys()))

                        X_filtered = X_processed[mask]
                        y_filtered = y_processed[mask]

                        # Removing classes leaves gaps in the encoded label space (e.g.
                        # dropping class 20 out of 0..21 leaves 0..19,21) which models like
                        # XGBoost reject as non-contiguous. Re-fit the label encoder on the
                        # surviving raw labels so it's contiguous again, and keep
                        # preprocessor.label_encoder in sync. Also remember the surviving
                        # raw labels so the raw-data split below filters on raw values
                        # instead of comparing raw values against encoded integers.
                        if preprocessor.label_encoder is not None:
                            raw_labels_filtered = preprocessor.label_encoder.inverse_transform(y_filtered)
                            valid_raw_labels = set(preprocessor.label_encoder.inverse_transform(list(valid_classes.keys())))
                            from sklearn.preprocessing import LabelEncoder as _LabelEncoder

                            preprocessor.label_encoder = _LabelEncoder()
                            y_filtered = preprocessor.label_encoder.fit_transform(raw_labels_filtered)

                        # Update the processed data
                        X_processed = X_filtered
                        y_processed = y_filtered
                        st.session_state.X_processed = X_processed
                        st.session_state.y_processed = y_processed

                        # Show filtering results
                        removed_classes = len(original_class_counts) - len(valid_classes)
                        if removed_classes > 0:
                            st.success(
                                f"🛠️ **Auto-Filter Applied:** Removed {removed_classes:,} rare classes with <{min_samples} samples"
                            )
                            st.info(
                                f"📊 **Filtered Dataset:** {len(valid_classes):,} classes, {len(X_processed):,} samples remaining"
                            )
                    else:
                        st.error(
                            f"❌ After filtering, only {len(valid_classes)} classes would remain. Disabling filter."
                        )
                        st.session_state.enable_class_filter = False

                # Check if stratification is possible (each class has at least 2 samples)
                class_counts = Counter(y_processed)
                min_class_count = min(class_counts.values())
                use_stratify = min_class_count >= 2

                if use_stratify:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed,
                        y_processed,
                        test_size=0.3,  # 30% holdout for testing
                        stratify=y_processed,
                        random_state=st.session_state.random_seed,  # Use user preference
                    )
                else:
                    st.warning("⚠️ Some classes have only 1 sample. Using random split instead of stratified split.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed,
                        y_processed,
                        test_size=0.3,
                        random_state=st.session_state.random_seed,  # Use user preference
                    )
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test

                # Create raw data splits with same random state for DimRedEvaluator
                # IMPORTANT: Use the same class filter logic for raw data if classes were filtered
                X_raw_for_split = X
                y_raw_for_split = y

                # If we filtered classes in processed data, filter raw data too - using the
                # raw label values captured above, not y_processed's encoded integers (which
                # live in a different, re-encoded label space and would filter incorrectly).
                if valid_raw_labels is not None:
                    if isinstance(y, pd.Series):
                        raw_mask = y.isin(valid_raw_labels)
                    else:
                        raw_mask = np.isin(y, list(valid_raw_labels))
                    X_raw_for_split = X[raw_mask]
                    y_raw_for_split = y[raw_mask]

                # Check if raw data stratification is possible
                raw_class_counts = Counter(y_raw_for_split)
                raw_min_class_count = min(raw_class_counts.values()) if raw_class_counts else 0
                raw_use_stratify = raw_min_class_count >= 2

                if raw_use_stratify:
                    X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
                        X_raw_for_split,
                        y_raw_for_split,
                        test_size=0.3,
                        stratify=y_raw_for_split,
                        random_state=st.session_state.random_seed,  # Use user preference
                    )
                else:
                    X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
                        X_raw_for_split,
                        y_raw_for_split,
                        test_size=0.3,
                        random_state=st.session_state.random_seed,  # Use user preference
                    )

                st.info(f"📊 Split: Train={len(X_train)} samples, Test={len(X_test)} samples (30% holdout)")

            # Train models
            if task_type == "Classification":
                st.info("🚀 Starting Classification...")
                logger.debug("About to call run_classification()")
                self.run_classification(
                    X_train, y_train, X_test, y_test, dimred_config, preprocessor, X_raw_train, y_raw_train
                )
                logger.debug("run_classification() completed")
            else:
                st.info("🚀 Starting Clustering...")
                logger.debug("About to call run_clustering()")
                self.run_clustering(X_processed, preprocessor, dimred_config)
                logger.debug("run_clustering() completed")

            st.success("✅ AutoML pipeline completed!")

        except Exception as e:
            st.error(f"Error running AutoML: {e}")
            logger.error(f"AutoML error: {e}", exc_info=True)

    def run_professional_automl(
        self, optimization_time_minutes=15, max_trials=100, include_ensemble=True, advanced_features=None
    ):
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
        if hasattr(st.session_state, "professional_results"):
            delattr(st.session_state, "professional_results")

        try:
            data = st.session_state.data
            task_type = st.session_state.task_type

            # Get basic task mapping
            if task_type == "Classification":
                ml_task = "classification"
                target_col = st.session_state.target_col
                X = data.drop(columns=[target_col])
                y = data[target_col]

                # Check for potential regression disguised as classification
                n_unique = y.nunique()
                n_samples = len(y)

                if n_unique / n_samples > 0.5:
                    import pandas as pandas_api

                    if pandas_api.api.types.is_numeric_dtype(y):
                        st.warning(
                            f"⚠️ **Warning:** Target has {n_unique} unique values ({n_unique/n_samples:.1%} of samples). This looks like a continuous variable - consider using Regression instead of Classification for better results."
                        )
                elif n_unique > 100:
                    st.warning(
                        f"⚠️ **Warning:** Target has {n_unique} classes. Classification with many classes may be challenging and slow."
                    )

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
                enable="auto",
                method="auto",
                variance_target=0.95,
                k_max=256,
                whiten=True,
                seed=st.session_state.random_seed,
            )

            preprocessor = DataPreprocessor(max_features=1000, dimred_config=dimred_config)
            X_processed, y_processed = preprocessor.fit_transform(X, y)

            # Initialize Professional AutoML Pipeline
            st.info("🤖 Initializing Professional AutoML Pipeline...")

            professional_pipeline = AutoMLPipeline(
                task_type=ml_task,
                optimization_time_minutes=optimization_time_minutes,
                random_state=st.session_state.random_seed,
            )

            # Get model candidates based on task type and dataset size
            n_features = X_processed.shape[1] if hasattr(X_processed, "shape") else len(X_processed.columns)
            model_candidates = self._get_professional_model_candidates(ml_task, len(X_processed), n_features)

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
                n_features = X_processed.shape[1] if hasattr(X_processed, "shape") else len(X_processed.columns)
                st.metric("📐 Features", n_features)
            with col3:
                if ml_task != "clustering":
                    unique_targets = len(np.unique(y_processed))
                    st.metric(
                        "🎯 Classes/Range" if ml_task == "classification" else "🎯 Target Range",
                        (
                            f"{unique_targets:,}"
                            if ml_task == "classification"
                            else f"{y_processed.min():.2f}-{y_processed.max():.2f}"
                        ),
                    )

            # Professional optimization with progress tracking
            status_text.text("⚙️ Running hyperparameter optimization...")
            progress_bar.progress(0.3)

            # Record start time for optimization tracking
            optimization_start_time = time.time()

            # Run professional AutoML pipeline
            results = professional_pipeline.run_advanced_automl(
                X_processed,
                y_processed,
                model_candidates=model_candidates,
                include_ensemble=include_ensemble and ml_task != "clustering",
            )

            # Validate results before proceeding
            if results is None:
                st.error("❌ Professional AutoML failed to generate results")
                progress_bar.empty()
                status_text.empty()
                return

            # Check if optimization was successful
            if not results.get("individual_models"):
                st.warning("⚠️ No models were successfully optimized. Using fallback results.")
                # Create minimal results for display
                results = {
                    "individual_models": {"RandomForest": {"best_score": -1000, "optimization_failed": True}},
                    "ensemble_models": None,
                    "optimization_summary": "Optimization failed - using fallback mode",
                    "dataset_info": {
                        "n_samples": len(X_processed),
                        "n_features": X_processed.shape[1],
                        "task_type": ml_task,
                    },
                }

            progress_bar.progress(0.9)
            status_text.text("📊 Finalizing results...")

            # Store results in session state for display on Results page
            st.session_state.professional_results = results
            st.session_state.professional_pipeline = professional_pipeline
            st.session_state.dataset_stats = dataset_stats
            st.session_state.optimization_time = time.time() - optimization_start_time

            # Store error logs for debugging
            if hasattr(professional_pipeline, "optimizer") and hasattr(professional_pipeline.optimizer, "error_log"):
                st.session_state.optimization_errors = professional_pipeline.optimizer.error_log
            else:
                st.session_state.optimization_errors = []

            # CRITICAL: Populate st.session_state.models for other tabs
            # Extract trained models from professional results
            if results and "individual_models" in results:
                trained_models = {}
                for model_name, model_result in results["individual_models"].items():
                    # The key is 'model' in the optimization results
                    if "model" in model_result:
                        trained_models[model_name] = model_result["model"]

                if trained_models:
                    st.session_state.models = trained_models
                    st.info(f"✅ Stored {len(trained_models)} trained models for analysis")
                else:
                    st.warning("⚠️ No trained models found in results")

            # Store processed data and preprocessor for explainability and recommendations
            st.session_state.X_processed = X_processed
            st.session_state.y_processed = y_processed
            st.session_state.preprocessor = preprocessor
            st.session_state.task_performed = ml_task

            # Final completion steps
            progress_bar.progress(1.0)
            status_text.text("✅ Professional AutoML completed!")
            time.sleep(0.5)  # Brief pause to show completion

            # Clear progress indicators BEFORE navigation
            progress_bar.empty()
            status_text.empty()

            # Navigate to Results page
            st.session_state.app_stage = "results"
            st.success("🏆 **Professional AutoML Pipeline Complete!**")
            st.info("📊 **Redirecting to Results page...**")

            # Force rerun to navigate to results
            st.rerun()

        except Exception as e:
            # Enhanced error handling with cleanup
            if "progress_bar" in locals():
                progress_bar.empty()
            if "status_text" in locals():
                status_text.empty()

            st.error(f"❌ Professional AutoML Error: {str(e)}")

            # Provide detailed error information for troubleshooting
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "dataset_shape": f"{data.shape[0]}x{data.shape[1]}" if "data" in locals() else "Unknown",
                "task_type": task_type if "task_type" in locals() else "Unknown",
            }

            with st.expander("🔍 Error Details (for troubleshooting)"):
                st.json(error_details)

            import traceback

            st.error(f"Technical Details: {traceback.format_exc()}")

            # Reset session state to prevent stuck state
            if hasattr(st.session_state, "professional_results"):
                delattr(st.session_state, "professional_results")

            st.info("💡 **Suggested Actions:**")
            st.write("1. Try reducing the dataset size")
            st.write("2. Check for data quality issues")
            st.write("3. Use Standard AutoML mode instead")
            st.write("4. Ensure all required packages are installed")

    def _get_professional_model_candidates(self, task_type, n_samples, n_features):
        """Get professional model candidates with intelligent selection."""
        from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.mixture import GaussianMixture
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.svm import SVC, SVR

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
                ("RandomForest", RandomForestClassifier(random_state=st.session_state.random_seed)),
                ("LogisticRegression", LogisticRegression(random_state=st.session_state.random_seed, max_iter=1000)),
            ]

            # Add advanced models if available
            if xgb_available:
                candidates.append(
                    ("XGBoost", XGBClassifier(random_state=st.session_state.random_seed, eval_metric="logloss"))
                )
            if lgb_available:
                candidates.append(("LightGBM", LGBMClassifier(random_state=st.session_state.random_seed, verbose=-1)))

            # Add SVM for smaller datasets
            if n_samples <= 10000:
                candidates.append(("SVM", SVC(random_state=st.session_state.random_seed, probability=True)))

            # Add MLP for appropriate dataset sizes
            if 1000 <= n_samples <= 50000:
                candidates.append(("MLP", MLPClassifier(random_state=st.session_state.random_seed, max_iter=1000)))

            # Add KNN for smaller datasets
            if n_samples <= 20000:
                candidates.append(("KNN", KNeighborsClassifier()))

        elif task_type == "regression":
            candidates = [
                ("RandomForest", RandomForestRegressor(random_state=st.session_state.random_seed)),
                ("LinearRegression", LinearRegression()),
            ]

            if xgb_available:
                candidates.append(("XGBoost", XGBRegressor(random_state=st.session_state.random_seed)))
            if lgb_available:
                candidates.append(("LightGBM", LGBMRegressor(random_state=st.session_state.random_seed, verbose=-1)))

            if n_samples <= 10000:
                candidates.append(("SVR", SVR()))

            if 1000 <= n_samples <= 50000:
                candidates.append(("MLP", MLPRegressor(random_state=st.session_state.random_seed, max_iter=1000)))

            if n_samples <= 20000:
                candidates.append(("KNN", KNeighborsRegressor()))

        else:  # clustering
            candidates = [
                ("KMeans", KMeans(random_state=st.session_state.random_seed)),
                ("GaussianMixture", GaussianMixture(random_state=st.session_state.random_seed)),
            ]

            # Add DBSCAN for smaller datasets
            if n_samples <= 10000:
                candidates.append(("DBSCAN", DBSCAN()))

            # Add Agglomerative for smaller datasets (O(n^2), too slow otherwise)
            if n_samples <= 10000:
                candidates.append(("AgglomerativeClustering", AgglomerativeClustering()))

        # Honor the user's Model Selection tab choice instead of always training every
        # size-eligible candidate - mirrors the filtering already done in run_classification.
        selected_models = st.session_state.get("selected_models")
        if selected_models:
            filtered = [c for c in candidates if c[0] in selected_models]
            if filtered:
                candidates = filtered

        return candidates

    def _analyze_dataset_professionally(self, X, y, task_type):
        """Professional dataset analysis for optimization insights."""
        # Handle both pandas DataFrames and numpy arrays
        if hasattr(X, "memory_usage"):
            # DataFrame case
            memory_mb = X.memory_usage(deep=True).sum() / 1024 / 1024
            missing_values = X.isnull().sum().sum()
            numeric_features = X.select_dtypes(include=[np.number]).shape[1]
            categorical_features = X.select_dtypes(include=["object"]).shape[1]
        else:
            # Numpy array case
            memory_mb = X.nbytes / 1024 / 1024
            missing_values = np.isnan(X).sum() if np.issubdtype(X.dtype, np.number) else 0
            numeric_features = X.shape[1]  # Assume all numeric for processed arrays
            categorical_features = 0

        stats = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "memory_usage_mb": memory_mb,
            "missing_values": missing_values,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        }

        if task_type != "clustering":
            stats["target_type"] = str(y.dtype)
            if task_type == "classification":
                stats["n_classes"] = len(np.unique(y))
                stats["class_balance"] = (np.bincount(y) / len(y)).std()  # Higher = more imbalanced
            else:  # regression
                stats["target_range"] = y.max() - y.min()
                stats["target_std"] = y.std()

        # Data complexity analysis - handle both DataFrame and numpy array
        if hasattr(X, "corr"):
            # DataFrame case
            stats["feature_correlation_avg"] = abs(X.corr()).mean().mean() if X.shape[1] > 1 else 0
            stats["sparsity"] = (X == 0).mean().mean()
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
                stats["feature_correlation_avg"] = np.abs(corr_matrix[mask]).mean() if mask.sum() > 0 else 0
            else:
                stats["feature_correlation_avg"] = 0
            stats["sparsity"] = (X == 0).mean()

        return stats

    def run_classification(
        self, X_train, y_train, X_test, y_test, dimred_config, preprocessor, X_raw_train, y_raw_train
    ):
        """Run classification pipeline with proper train/test split."""
        st.info("🤖 Training classification models on training set...")

        # SMART MODEL SELECTION: Use fast models for large datasets (with caching)
        total_samples = len(y_train)

        # Get models with adaptive settings based on dataset size (cached)
        models = cached_get_models(task_type="classification", random_seed=st.session_state.random_seed)

        # First, handle large dataset optimization (if no user selection)
        selected_models = st.session_state.get("selected_models")
        if total_samples > 20000 and not selected_models:
            # Large dataset: Remove slow SVM models (only if user hasn't made a selection)
            st.warning(f"⚡ **Large Dataset Detected** ({total_samples:,} samples)")
            st.info("🚀 Using **Fast Models Only** (LogReg, RF, XGBoost, MLP). SVMs skipped (too slow).")
            models = {k: v for k, v in models.items() if "SVM" not in k}

        # Apply user's model selection if configured (this takes priority)
        if selected_models:

            # Map UI model names to actual implementation names
            model_mapping = {
                "SVM": ["LinearSVM", "RBF-SVM"],  # UI shows 'SVM', but we have 'LinearSVM' and 'RBF-SVM'
                "SVR": ["LinearSVR", "RBF-SVR"],  # Similar for regression
            }

            # Expand selection to include all variants
            expanded_selection = []
            for selected in selected_models:
                if selected in model_mapping:
                    expanded_selection.extend(model_mapping[selected])
                else:
                    expanded_selection.append(selected)

            # Filter models to only include user-selected ones (or their variants)
            models = {name: model for name, model in models.items() if name in expanded_selection}
            st.info(f"⚙️ Using your selected models: {list(models.keys())}")

            # Check if any selected models are missing
            missing_models = [m for m in expanded_selection if m not in models]
            if missing_models:
                st.warning(f"⚠️ Some selected models are not available: {missing_models}")
        else:
            st.info(f"⚙️ Using all available models: {list(models.keys())}")

        # Determine appropriate CV strategy based on data size
        from collections import Counter

        class_counts = Counter(y_train)  # Use training set only
        min_class_count = min(class_counts.values())

        # Check if dataset is too small for CV
        if min_class_count < 2:
            st.error(
                f"❌ Dataset has a class with only {min_class_count} sample(s). Each class needs at least 2 samples for cross-validation."
            )
            # Show concise class distribution summary instead of full dict
            classes_with_one = sum(1 for count in class_counts.values() if count == 1)
            classes_with_few = sum(1 for count in class_counts.values() if count < 5)
            st.info(
                f"📊 Class summary: {len(class_counts)} total classes, {classes_with_one} with 1 sample, {classes_with_few} with <5 samples"
            )

            # Create a unique key for the button to avoid caching issues
            button_key = f"auto_fix_button_{len(class_counts)}_{classes_with_one}"

            # Offer automatic class filtering
            st.warning("💡 **Solution Options:**")
            col1, col2 = st.columns(2)

            with col1:
                min_samples_threshold = st.number_input(
                    "🎯 Minimum samples per class:",
                    min_value=2,
                    max_value=50,
                    value=5,
                    key=f"threshold_input_{len(class_counts)}",
                    help="Classes with fewer samples will be removed",
                )

            with col2:
                if st.button("🛠️ Auto-Fix Dataset", type="primary", key=button_key):
                    with st.spinner("🔄 Applying class filter..."):
                        # Filter out classes with insufficient samples
                        valid_classes = {
                            class_label: count
                            for class_label, count in class_counts.items()
                            if count >= min_samples_threshold
                        }

                        if len(valid_classes) < 2:
                            st.error(
                                f"❌ After filtering, only {len(valid_classes)} classes remain. Need at least 2 classes."
                            )
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

                        X_processed_combined = pd.concat([X_train_filtered, X_test_filtered])
                        y_processed_combined = pd.concat([y_train_filtered, y_test_filtered])
                        st.session_state.X_processed = X_processed_combined
                        st.session_state.y_processed = y_processed_combined

                        # Show results
                        removed_classes = len(class_counts) - len(valid_classes)
                        removed_samples_train = len(X_train) - len(X_train_filtered)
                        removed_samples_test = len(X_test) - len(X_test_filtered)

                        st.success("✅ **Dataset Auto-Fixed!**")
                        st.info(
                            f"📊 **Results:**\n"
                            f"- Removed {removed_classes:,} classes with <{min_samples_threshold} samples\n"
                            f"- Kept {len(valid_classes):,} classes\n"
                            f"- Removed {removed_samples_train:,} training samples ({removed_samples_train/(len(X_train)+0.001)*100:.1f}%)\n"
                            f"- Removed {removed_samples_test:,} test samples ({removed_samples_test/(len(X_test)+0.001)*100:.1f}%)\n"
                            f"- New training size: {len(X_train_filtered):,} samples\n"
                            f"- New test size: {len(X_test_filtered):,} samples"
                        )

                        st.success("🔄 **Page will refresh to continue with filtered data...**")
                        time.sleep(2)  # Give user time to read the results
                        st.rerun()  # This will restart and use the updated session state data

            st.info(
                "📌 **Manual alternatives:**\n"
                "- Remove rare classes from your data before upload\n"
                "- Combine similar classes into broader categories\n"
                "- Use clustering instead of classification"
            )

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
            st.success(
                f"✅ Auto-filtered: Removed {removed_classes} classes with <2 samples. Continuing with {len(valid_classes)} classes."
            )

            # Update class counts for the adaptive CV logic below
            class_counts = {k: v for k, v in class_counts.items() if k in valid_classes}
            min_class_count = min(class_counts.values())

        # Get user-configured CV folds with safety constraints
        user_cv_folds = st.session_state.get("advanced_config", {}).get("validation", {}).get("cv_folds", 5)

        # Adaptive CV: Respect user preference but ensure data safety
        if min_class_count < 2:
            n_folds = 2
            n_repeats = 1
            st.warning("⚠️ Very small classes detected. Using minimum 2-fold CV for safety.")
        elif min_class_count < user_cv_folds:
            n_folds = min_class_count  # Can't have more folds than samples per class
            n_repeats = 1 if n_folds <= 3 else 2
            st.warning(
                f"⚠️ Small classes detected (min: {min_class_count}). Using {n_folds}-fold CV instead of your configured {user_cv_folds} folds."
            )
        else:
            n_folds = user_cv_folds  # Use user preference
            n_repeats = 1 if n_folds >= 10 else (2 if n_folds >= 5 else 3)
            st.info(f"⚙️ Using your configured {n_folds}-fold CV with {n_repeats} repeats")

        # Evaluate models with holdout set
        evaluator = ClassificationEvaluator(n_folds=n_folds, n_repeats=n_repeats)

        # NEW: Dimensionality reduction evaluation
        if st.session_state.get("dimred_enabled") != "off":
            st.info("📐 Evaluating dimensionality reduction impact...")
            dimred_evaluator = DimRedEvaluator(
                preprocessor=preprocessor, dimred_config=dimred_config, random_state=st.session_state.random_seed
            )

            # Run dimred comparison for representative models
            representative_models = {}
            for name, model in models.items():
                if any(key in name.lower() for key in ["logistic", "random forest", "xgboost"]):
                    representative_models[name] = model
                if len(representative_models) >= 2:  # Test with 2-3 representative models
                    break

            dimred_results = dimred_evaluator.evaluate_classification_with_dimred(
                representative_models, X_raw_train, y_raw_train, task_type="classification"
            )

            # Store dimred results for PCA tab
            st.session_state.dimred_results = dimred_results

            # Show dimred summary
            if dimred_results.get("recommended_config"):
                rec_config = dimred_results["recommended_config"]
                if rec_config.enable == "on":
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
                result = evaluator.evaluate_with_holdout(model, X_train, y_train, X_test, y_test, name)
                results[name] = result

                # Show timing
                model_time = time.time() - model_start
                status_text.text(f"✅ {name} complete in {model_time:.1f}s")

                # Display CV strategy once (from first successful model)
                if not cv_strategy_displayed and "cv_strategy" in result:
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
        st.session_state.models = {
            name: res["trained_model"] for name, res in results.items() if "trained_model" in res
        }

        # Debug logging
        st.success(f"✅ Classification completed! Stored {len(results)} result sections.")
        st.info(f"🔍 Results keys: {list(results.keys())}")

        # Meta-learning recommendation
        st.info("🎯 Generating recommendations...")
        meta_selector = MetaModelSelector()
        recommendation = meta_selector.get_recommendation_with_rationale(st.session_state.profile, results)
        st.session_state.recommendation = recommendation

        # Transition to results stage
        st.session_state.app_stage = "results"
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

            st.info(
                f"📊 **Clustering Sample:** {X_for_clustering.shape[0]:,} samples × {X_for_clustering.shape[1]} features"
            )

        # Get models with optimizations for dataset size (cached)
        models = cached_get_models(task_type="clustering", random_seed=st.session_state.random_seed)

        # Apply user's model selection if configured (this takes priority)
        selected_models = st.session_state.get("selected_models")
        if selected_models:
            # Map UI model names to actual implementation names for clustering
            model_mapping = {
                "GaussianMixture": ["GMM"],  # UI might show different names
                "AgglomerativeClustering": ["Agglomerative"],
            }

            # Expand selection to include all variants
            expanded_selection = []
            for selected in selected_models:
                if selected in model_mapping:
                    expanded_selection.extend(model_mapping[selected])
                else:
                    expanded_selection.append(selected)

            # Filter models to only include user-selected ones
            models = {name: model for name, model in models.items() if name in expanded_selection}
            st.info(f"⚙️ Using your selected models: {list(models.keys())}")

            # Check if any selected models are missing
            missing_models = [m for m in expanded_selection if m not in models]
            if missing_models:
                st.warning(f"⚠️ Some selected models are not available: {missing_models}")
        else:
            # Remove slow models for large datasets (only if user hasn't made a selection)
            if n_samples > 20000:
                slow_models = ["DBSCAN", "Agglomerative"]
                models = {k: v for k, v in models.items() if k not in slow_models}
                st.info("🚀 **Fast Models Only:** Removed slow algorithms (DBSCAN, Agglomerative) for large dataset")
            else:
                st.info(f"⚙️ Using all available models: {list(models.keys())}")

        # NEW: Hybrid dimensionality reduction evaluation for clustering
        if st.session_state.get("dimred_enabled") != "off" and n_samples < 100000:
            st.info("📐 Smart evaluation: Dimensionality reduction impact on clustering...")

            # Progressive evaluation with time limits
            dimred_result = self._evaluate_dimred_hybrid(X_for_clustering, st.session_state.random_seed)

            # Store results
            st.session_state.dimred_results = dimred_result

            # Show recommendations
            if dimred_result.get("recommended"):
                st.success(f"✅ {dimred_result['recommendation']}")
                st.info(f"📊 {dimred_result['details']}")
            else:
                st.info(f"💡 {dimred_result['recommendation']}")
                if dimred_result.get("reason"):
                    st.caption(f"Reason: {dimred_result['reason']}")

        else:
            if n_samples >= 100000:
                st.info("📐 Skipping dimensionality reduction evaluation for very large datasets")
            st.session_state.dimred_results = {
                "recommended": False,
                "recommendation": "Auto mode: Will decide per model",
                "method": "auto",
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
                        if hasattr(model, "predict"):
                            labels = model.predict(X)
                        else:
                            # For models like DBSCAN that don't have predict, use the sampled results
                            labels = labels_sample
                    except Exception as e:
                        logger.debug(f"Model predict failed, falling back to sampled labels: {e}")
                        labels = labels_sample
                else:
                    labels = labels_sample

                # Evaluate on the appropriate dataset
                eval_X = X if not use_sampling or hasattr(model, "predict") else X_for_clustering
                result = evaluator.evaluate_model(
                    model, eval_X, name, labels if not use_sampling or hasattr(model, "predict") else labels_sample
                )

                results[name] = result
                results[name]["model"] = model

                if use_sampling and hasattr(model, "predict"):
                    st.success(
                        f"✅ {name}: Trained on {X_for_clustering.shape[0]:,} samples, evaluated on full {n_samples:,} samples"
                    )

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                st.warning(f"⚠️ {name} failed: {str(e)[:100]}...")

            progress_bar.progress((idx + 1) / len(models))

        st.session_state.results = results
        st.session_state.evaluator = evaluator
        st.session_state.models = {name: res["model"] for name, res in results.items()}

        # Debug logging
        st.success(f"✅ Clustering completed! Stored {len(results)} result sections.")
        st.info(f"🔍 Results keys: {list(results.keys())}")

        # Transition to results stage
        st.session_state.app_stage = "results"
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
            if elapsed < 15 and result_fast.get("improvement", 0) > 0.03 and n_samples > 10000:

                # Phase 2: Medium evaluation (remaining time, 20K samples)
                sample_size_medium = min(20000, n_samples)
                indices_medium = np.random.choice(n_samples, sample_size_medium, replace=False)
                X_medium = X[indices_medium]

                result_medium = self._evaluate_pca_clustering_phase(X_medium, random_state, phase="medium")
                elapsed = time.time() - start_time

                # If still promising and time allows, do comprehensive evaluation
                if elapsed < 25 and result_medium.get("improvement", 0) > 0.05 and n_samples > 30000:

                    # Phase 3: Comprehensive evaluation
                    sample_size_full = min(50000, n_samples)
                    indices_full = np.random.choice(n_samples, sample_size_full, replace=False)
                    X_full = X[indices_full]

                    return self._evaluate_pca_clustering_phase(X_full, random_state, phase="comprehensive")

                return result_medium

            return result_fast

        except Exception as e:
            return {
                "recommended": False,
                "recommendation": f"Evaluation failed: {str(e)[:50]}...",
                "method": "baseline",
                "reason": "Error during hybrid evaluation",
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
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score

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
            except Exception as e:
                logger.debug(f"Baseline clustering evaluation skipped for k={k}: {e}")
                continue

            # Test PCA variants
            if isinstance(pca_components, list):
                pca_components_list = pca_components
            else:
                pca_components_list = [pca_components]

            for pca_comp in pca_components_list:
                try:
                    # Handle missing values before PCA
                    from sklearn.impute import SimpleImputer

                    X_for_pca = X.copy()
                    if pd.DataFrame(X_for_pca).isnull().any().any():
                        imputer = SimpleImputer(strategy="median")
                        X_for_pca = imputer.fit_transform(X_for_pca)

                    pca = PCA(n_components=pca_comp, random_state=random_state)
                    X_pca = pca.fit_transform(X_for_pca)

                    kmeans_pca = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=100)
                    labels_pca = kmeans_pca.fit_predict(X_pca)

                    if len(set(labels_pca)) > 1:
                        score_pca = silhouette_score(X_pca, labels_pca)
                        if score_pca > best_pca_score:
                            best_pca_score = score_pca
                            best_pca_components = pca_comp
                except Exception as e:
                    logger.debug(f"PCA clustering evaluation skipped for {pca_comp} components: {e}")
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
                "recommended": True,
                "recommendation": f"PCA recommended (retains {best_pca_components*100:.0f}% variance)",
                "method": "pca",
                "improvement": improvement,
                "details": f"Silhouette: {best_baseline_score:.3f} → {best_pca_score:.3f} (+{improvement:.3f})",
                "best_k": best_k,
                "pca_components": best_pca_components,
                "phase": phase,
            }
        else:
            return {
                "recommended": False,
                "recommendation": f"Original features preferred ({phase} evaluation)",
                "method": "baseline",
                "improvement": improvement,
                "details": f"Silhouette: baseline {best_baseline_score:.3f} vs PCA {best_pca_score:.3f}",
                "phase": phase,
                "reason": f"Improvement (+{improvement:.3f}) below threshold ({threshold})",
            }

    def render_tabs(self):
        """Render main content tabs."""
        # Determine which tabs to show based on available results
        professional_results = st.session_state.get("professional_results")
        standard_results = st.session_state.get("results")

        if professional_results and standard_results:
            # Show all tabs including Professional
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                [
                    "📊 Data Overview",
                    "🤖 Models",
                    "🔥 Professional AutoML",
                    "📐 PCA Analysis",
                    "🔍 Explainability",
                    "🎯 Recommendation",
                    "📄 Report",
                ]
            )
        elif professional_results:
            # Show professional-focused tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                [
                    "📊 Data Overview",
                    "🔥 Professional AutoML",
                    "📐 PCA Analysis",
                    "🔍 Explainability",
                    "🎯 Recommendation",
                    "📄 Report",
                    "🎯 Insights",  # Professional insights tab
                ]
            )
        else:
            # Standard tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                [
                    "📊 Data Overview",
                    "🤖 Models",
                    "📐 PCA Analysis",
                    "🔍 Explainability",
                    "🎯 Recommendation",
                    "📄 Report",
                ]
            )

        # Initialize tab modules
        data_overview_tab = DataOverviewTab()
        models_tab = ModelsTab()
        professional_tab = ProfessionalAutoMLTab()
        pca_tab_module = PCAAnalysisTab()
        explainability_tab = ExplainabilityTab()
        recommendation_tab = RecommendationTab()
        report_tab_module = ReportTab()

        with tab1:
            data_overview_tab.render()

        # Professional AutoML tab
        if professional_results:
            with tab2 if not standard_results else tab3:
                professional_tab.render()

        # Standard Models tab (if available)
        if standard_results:
            with tab2:
                models_tab.render()

        # Adjust tab indices based on what's available
        if professional_results and standard_results:
            # Both available: Data, Models, Professional, PCA, Explain, Recommend, Report
            pca_tab, explain_tab, recommend_tab, report_tab = tab4, tab5, tab6, tab7
            insights_tab = None  # No insights tab when both are available
        elif professional_results:
            # Professional only: Data, Professional, PCA, Explain, Recommend, Report, Insights
            pca_tab, explain_tab, recommend_tab, report_tab, insights_tab = tab3, tab4, tab5, tab6, tab7
        else:
            # Standard only: Data, Models, PCA, Explain, Recommend, Report
            pca_tab, explain_tab, recommend_tab, report_tab = tab3, tab4, tab5, tab6
            insights_tab = None

        with pca_tab:
            pca_tab_module.render()

        with explain_tab:
            explainability_tab.render()

        with recommend_tab:
            recommendation_tab.render()

        with report_tab:
            report_tab_module.render()

        # Render insights tab for professional results
        if insights_tab is not None and professional_results:
            with insights_tab:
                professional_tab.render_insights()

    def render_configuration_stage(self):
        """Render the unified configuration stage."""
        # Add main container for better centering
        st.markdown("<div style='max-width: 1200px; margin: 0 auto;'>", unsafe_allow_html=True)

        # Header with navigation
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("← Back to Welcome", key="back_to_welcome"):
                st.session_state.app_stage = "welcome"
                st.rerun()

        with col2:
            st.markdown(
                "<h2 style='text-align: center; margin: 0;'>⚙️ Configure AutoML Pipeline</h2>", unsafe_allow_html=True
            )

        with col3:
            # Progress indicator
            st.markdown(
                "<div style='text-align: right; color: #666;'>Step 2 of 3 - Configuration</div>", unsafe_allow_html=True
            )

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
                if hasattr(self, "_get_file_name"):
                    filename = st.session_state.get("uploaded_file_name", "Unknown")
                    st.metric("📁 File", filename.replace(".csv", ""))

        # Unified configuration tabs
        config_tab1, config_tab2, config_tab3, config_tab4 = st.tabs(
            ["🎯 Task & Analysis", "🤖 Model Selection", "⚙️ Optimization", "🔧 Advanced Settings"]
        )

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

        if ready_checks["ready"]:
            # Show current configuration summary
            st.markdown("### 🎯 **Ready to Execute AutoML**")

            # Display configuration summary
            config_info = []
            if hasattr(st.session_state, "selected_models") and st.session_state.selected_models:
                config_info.append(
                    f"📊 **Models**: {', '.join(st.session_state.selected_models[:3])}{'...' if len(st.session_state.selected_models) > 3 else ''}"
                )

            if hasattr(st.session_state, "dimred_enabled") and st.session_state.dimred_enabled != "auto":
                config_info.append(f"📉 **PCA**: {st.session_state.dimred_enabled}")

            if hasattr(st.session_state, "advanced_config") and st.session_state.advanced_config:
                if st.session_state.advanced_config.get("validation", {}).get("cv_folds"):
                    config_info.append(f"🔄 **CV Folds**: {st.session_state.advanced_config['validation']['cv_folds']}")

            if hasattr(st.session_state, "optimization_config") and st.session_state.optimization_config:
                opt_config = st.session_state.optimization_config
                if opt_config.get("time_minutes", 0) > 5:
                    config_info.append(f"⏰ **Optimization**: {opt_config['time_minutes']} min")

            if config_info:
                st.info(" | ".join(config_info))
            else:
                st.info("📝 **Configuration**: Using default settings")

            # Single unified run button
            if st.button(
                "🚀 **Run AutoML with My Configuration**", type="primary", width="stretch", key="run_unified_automl"
            ):
                # Use intelligent mode selection
                mode = self._determine_execution_mode()
                st.info(f"🔥 **Executing {mode.title()} AutoML** with your configured preferences...")
                self._execute_automl(mode)
        else:
            st.error("❌ Configuration incomplete. Please complete the required settings above.")
            for issue in ready_checks["issues"]:
                st.warning(f"• {issue}")

        st.markdown("</div>", unsafe_allow_html=True)  # Close configuration container

    def _determine_execution_mode(self):
        """Intelligently determine whether to use standard or professional mode based on user configuration."""
        # Check for professional mode indicators
        professional_indicators = [
            # Advanced model selection
            hasattr(st.session_state, "selected_models") and len(st.session_state.get("selected_models", [])) > 3,
            # Custom optimization settings
            hasattr(st.session_state, "optimization_config")
            and st.session_state.optimization_config.get("time_minutes", 0) > 5,
            # Advanced dimensionality reduction
            hasattr(st.session_state, "dimred_enabled") and st.session_state.dimred_enabled not in ["auto", "off"],
            # Custom validation settings
            hasattr(st.session_state, "advanced_config")
            and st.session_state.advanced_config.get("validation", {}).get("cv_folds", 5) != 5,
            # Large dataset size
            hasattr(st.session_state, "data") and len(st.session_state.data) > 1000,
        ]

        return "professional" if any(professional_indicators) else "standard"

    def render_results_stage(self):
        """Render the results stage with all analysis tabs."""
        # Header with navigation
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("← Back to Configure", key="back_to_configure"):
                st.session_state.app_stage = "configure"
                st.rerun()

        with col2:
            st.markdown("<h2 style='text-align: center; margin: 0;'>📊 AutoML Results</h2>", unsafe_allow_html=True)

        with col3:
            # New run button
            if st.button("🔄 New Analysis", key="new_analysis"):
                # Reset to welcome but keep data
                st.session_state.app_stage = "configure"
                st.session_state.results = None
                st.session_state.evaluator = None
                st.rerun()

        st.markdown("---")

        # Results content using existing tabs system
        results = st.session_state.get("results")
        professional_results = st.session_state.get("professional_results")

        if (results and len(results) > 0) or professional_results:
            self.render_tabs()
        else:
            st.error("❌ No results available. Please run AutoML analysis first.")

            # Show specific error if available
            error_msg = st.session_state.get("automl_error")
            if error_msg:
                st.error(f"Error details: {error_msg}")

            if st.button("↩️ Back to Configuration", key="back_to_config_from_results"):
                st.session_state.app_stage = "configure"
                st.rerun()

    def _guess_target_column(self, data):
        """Heuristically guess the most likely classification target column.

        Prefers low-cardinality columns (few distinct values relative to sample count)
        and conventionally-named target columns, instead of defaulting to column 0.
        """
        n_samples = len(data)
        max_classes = min(20, max(2, int(n_samples * 0.5)))
        candidates = []

        for col in data.columns:
            n_unique = data[col].nunique()
            if n_unique < 2 or n_unique > max_classes:
                continue
            score = -n_unique  # fewer classes scores higher
            if col.strip().lower() in ("target", "label", "class", "y", "outcome", "diagnosis"):
                score += 100
            candidates.append((score, col))

        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        return data.columns[-1]

    def _render_unified_task_tab(self):
        """Render unified task selection and dataset analysis."""
        st.header("🎯 Task Selection & Dataset Analysis")

        # Task type selection
        st.subheader("📋 Select Machine Learning Task")

        task_type = st.radio(
            "Choose your analysis type:",
            ["Classification", "Clustering"],
            help="Classification: Predict categories | Clustering: Find patterns",
        )

        st.session_state.task_type = task_type

        if task_type == "Classification":
            # Target selection - default to the previous choice, or a dataset-driven guess
            # (low-cardinality / conventionally-named column) instead of always column 0.
            columns = st.session_state.data.columns.tolist()
            default_target = st.session_state.get("target_col")
            if default_target not in columns:
                default_target = self._guess_target_column(st.session_state.data)
            default_index = columns.index(default_target) if default_target in columns else len(columns) - 1

            target_col = st.selectbox(
                "Select Target Column (what you want to predict)",
                options=columns,
                index=default_index,
                help="Choose the column that contains the values you want to predict",
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
        if not st.session_state.get("config_analyzed", False) or st.button("🔄 Re-analyze Dataset", key="reanalyze"):
            with st.spinner("🧠 Analyzing dataset characteristics..."):
                dataset_analysis = self._analyze_dataset_for_config(st.session_state.data)
                st.session_state.dataset_config = dataset_analysis
                st.session_state.config_analyzed = True
                st.rerun()

        # Display analysis results
        if st.session_state.get("dataset_config"):
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
        if not st.session_state.get("task_type"):
            issues.append("Task type not selected")

        # Check target for classification
        if st.session_state.get("task_type") == "Classification" and not st.session_state.get("target_col"):
            issues.append("Target column not selected for classification")

        return {"ready": len(issues) == 0, "issues": issues}

    def _execute_automl(self, mode="standard"):
        """Execute AutoML and transition to results."""
        try:
            with st.spinner(f"🚀 Running {mode.title()} AutoML..."):
                if mode == "professional":
                    opt_config = st.session_state.optimization_config
                    self.run_professional_automl(
                        optimization_time_minutes=opt_config["time_minutes"],
                        max_trials=opt_config["max_trials"],
                        include_ensemble=opt_config["include_ensemble"],
                        advanced_features=opt_config.get("advanced_features", []),
                    )
                else:
                    self.run_automl()

                # CRITICAL: Validate results were stored
                results = st.session_state.get("results")
                professional_results = st.session_state.get("professional_results")

                # Check if we have any results (standard or professional)
                has_results = (results and len(results) > 0) or professional_results

                if not has_results:
                    st.error("❌ AutoML execution failed - no results generated")
                    st.warning("Please check your data and target column selection")
                    error_msg = st.session_state.get("automl_error", "Unknown error")
                    st.error(f"Error details: {error_msg}")
                    return  # Don't transition to results

                # Transition to results
                st.session_state.app_stage = "results"
                st.success("✅ AutoML completed successfully!")
                st.rerun()

        except Exception as e:
            st.error(f"❌ AutoML execution failed: {e}")
            logger.error(f"AutoML execution error: {e}")

    def _analyze_dataset_for_config(self, data):
        """Comprehensive dataset analysis for configuration recommendations."""
        analysis = {"basic_stats": {}, "data_quality": {}, "complexity": {}, "recommendations": {}, "auto_config": {}}

        n_samples, n_features = data.shape

        # Basic statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        datetime_cols = data.select_dtypes(include=["datetime64"]).columns

        analysis["basic_stats"] = {
            "n_samples": n_samples,
            "n_features": n_features,
            "numeric_features": len(numeric_cols),
            "categorical_features": len(categorical_cols),
            "datetime_features": len(datetime_cols),
            "memory_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
            "sparsity": (data == 0).mean().mean() if len(numeric_cols) > 0 else 0,
        }

        # Data quality analysis
        missing_values = data.isnull().sum()
        analysis["data_quality"] = {
            "missing_features": len(missing_values[missing_values > 0]),
            "missing_percentage": (missing_values.sum() / (n_samples * n_features)) * 100,
            "duplicate_rows": data.duplicated().sum(),
            "constant_features": len([col for col in data.columns if data[col].nunique() <= 1]),
            "high_cardinality_features": len(
                [col for col in categorical_cols if data[col].nunique() > n_samples * 0.1]
            ),
        }

        # Complexity analysis
        feature_to_sample_ratio = n_features / n_samples
        analysis["complexity"] = {
            "feature_to_sample_ratio": feature_to_sample_ratio,
            "is_high_dimensional": feature_to_sample_ratio > 0.1,
            "is_sparse": analysis["basic_stats"]["sparsity"] > 0.1,
            "is_large_dataset": n_samples > 50000,
            "is_wide_dataset": n_features > 1000,
            "estimated_training_time": self._estimate_training_time(n_samples, n_features),
        }

        # Generate recommendations
        analysis["recommendations"] = self._generate_config_recommendations(analysis)

        # Auto-configuration
        analysis["auto_config"] = self._generate_auto_config(analysis)

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
            "per_model_minutes": max(0.1, min(time_per_model, 30)),  # Cap between 0.1 and 30 minutes
            "total_pipeline_minutes": max(2, min(time_per_model * 8, 120)),  # 8 models, cap at 2 hours
            "category": "fast" if time_per_model < 2 else "medium" if time_per_model < 10 else "slow",
        }

    def _generate_config_recommendations(self, analysis):
        """Generate intelligent configuration recommendations."""
        recommendations = {
            "preprocessing": {},
            "model_selection": {},
            "optimization": {},
            "validation": {},
            "warnings": [],
        }

        stats = analysis["basic_stats"]
        quality = analysis["data_quality"]
        complexity = analysis["complexity"]

        # Preprocessing recommendations
        if complexity["is_high_dimensional"]:
            recommendations["preprocessing"]["dimensionality_reduction"] = {
                "enable": True,
                "method": "tsvd" if complexity["is_sparse"] else "pca",
                "variance_target": 0.95,
                "reason": f"High dimensionality ({stats['n_features']} features vs {stats['n_samples']} samples)",
            }

        if quality["missing_percentage"] > 5:
            recommendations["preprocessing"]["imputation"] = {
                "strategy": "advanced",
                "reason": f"Significant missing data ({quality['missing_percentage']:.1f}%)",
            }

        if complexity["is_sparse"]:
            recommendations["preprocessing"]["scaling"] = {
                "method": "robust",
                "reason": f"Sparse data detected ({stats['sparsity']*100:.1f}% zeros)",
            }

        # Model selection recommendations
        if stats["n_samples"] < 1000:
            recommendations["model_selection"]["focus"] = {
                "models": ["LogisticRegression", "SVM", "KNN"],
                "avoid": ["MLP", "XGBoost"],
                "reason": "Small dataset - prefer simpler models",
            }
        elif stats["n_samples"] > 50000:
            recommendations["model_selection"]["focus"] = {
                "models": ["XGBoost", "RandomForest", "MLP"],
                "avoid": ["SVM"],
                "reason": "Large dataset - use scalable algorithms",
            }

        # Optimization recommendations
        training_time = complexity["estimated_training_time"]
        if training_time["category"] == "slow":
            recommendations["optimization"]["strategy"] = {
                "time_limit": 30,
                "trials_per_model": 50,
                "early_stopping": True,
                "reason": "Large dataset detected - limit optimization time",
            }
        elif training_time["category"] == "fast":
            recommendations["optimization"]["strategy"] = {
                "time_limit": 60,
                "trials_per_model": 200,
                "comprehensive": True,
                "reason": "Small/medium dataset - enable comprehensive optimization",
            }

        # Validation recommendations
        if stats["n_samples"] < 500:
            recommendations["validation"]["strategy"] = {
                "cv_folds": 10,
                "repeats": 3,
                "reason": "Small dataset - use more rigorous validation",
            }
        elif stats["n_samples"] > 10000:
            recommendations["validation"]["strategy"] = {
                "cv_folds": 3,
                "repeats": 1,
                "reason": "Large dataset - faster validation sufficient",
            }

        # Generate warnings
        if complexity["feature_to_sample_ratio"] > 1:
            recommendations["warnings"].append(
                {
                    "type": "critical",
                    "message": f"More features ({stats['n_features']}) than samples ({stats['n_samples']}) - curse of dimensionality!",
                    "suggestion": "Feature selection or dimensionality reduction essential",
                }
            )

        if quality["missing_percentage"] > 20:
            recommendations["warnings"].append(
                {
                    "type": "warning",
                    "message": f"High missing data percentage ({quality['missing_percentage']:.1f}%)",
                    "suggestion": "Consider data collection improvement or advanced imputation",
                }
            )

        return recommendations

    def _generate_auto_config(self, analysis):
        """Generate automatic configuration based on analysis."""
        stats = analysis["basic_stats"]
        complexity = analysis["complexity"]
        analysis["recommendations"]

        config = {
            "preprocessing": {
                "max_features": min(1000, stats["n_features"]),
                "scaling": "standard",
                "imputation_numeric": "median",
                "imputation_categorical": "most_frequent",
            },
            "dimensionality_reduction": {"enable": "auto", "method": "auto", "variance_target": 0.95},
            "model_selection": {
                "include_models": ["RandomForest", "LogisticRegression", "XGBoost", "SVM", "MLP"],
                "exclude_models": [],
            },
            "optimization": {"time_minutes": 15, "max_trials": 100, "include_ensemble": True, "early_stopping": True},
            "validation": {"cv_folds": 5, "test_size": 0.2, "stratified": True},
        }

        # Apply intelligent adjustments
        if stats["n_samples"] > 50000:
            config["model_selection"]["exclude_models"].append("SVM")
            config["optimization"]["time_minutes"] = 30

        if stats["n_samples"] < 1000:
            config["model_selection"]["exclude_models"].extend(["XGBoost", "MLP"])
            config["validation"]["cv_folds"] = 10

        if complexity["is_high_dimensional"]:
            config["dimensionality_reduction"]["enable"] = "on"
            if complexity["is_sparse"]:
                config["dimensionality_reduction"]["method"] = "tsvd"
            else:
                config["dimensionality_reduction"]["method"] = "pca"

        return config

    def _display_dataset_analysis(self, analysis):
        """Display the dataset analysis results."""
        st.subheader("🧠 AI Analysis Results")

        # Display warnings first
        if analysis["recommendations"]["warnings"]:
            st.warning("⚠️ **Critical Issues Detected:**")
            for warning in analysis["recommendations"]["warnings"]:
                if warning["type"] == "critical":
                    st.error(f"🚨 {warning['message']}")
                    st.info(f"💡 **Recommendation**: {warning['suggestion']}")
                else:
                    st.warning(f"⚠️ {warning['message']}")
                    st.info(f"💡 **Suggestion**: {warning['suggestion']}")

        # Data characteristics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Data Characteristics")
            stats = analysis["basic_stats"]
            complexity = analysis["complexity"]

            characteristics = []
            if complexity["is_large_dataset"]:
                characteristics.append("🔵 Large Dataset")
            elif stats["n_samples"] < 1000:
                characteristics.append("🟡 Small Dataset")
            else:
                characteristics.append("🟢 Medium Dataset")

            if complexity["is_high_dimensional"]:
                characteristics.append("📐 High-Dimensional")

            if complexity["is_sparse"]:
                characteristics.append("🕳️ Sparse Data")

            if complexity["is_wide_dataset"]:
                characteristics.append("↔️ Wide Dataset")

            for char in characteristics:
                st.markdown(f"- {char}")

        with col2:
            st.markdown("#### ⏱️ Estimated Training Time")
            training_time = complexity["estimated_training_time"]

            if training_time["category"] == "fast":
                st.success(f"🚀 Fast: ~{training_time['per_model_minutes']:.1f} min/model")
            elif training_time["category"] == "medium":
                st.info(f"⏳ Medium: ~{training_time['per_model_minutes']:.1f} min/model")
            else:
                st.warning(f"🐌 Slow: ~{training_time['per_model_minutes']:.1f} min/model")

            st.caption(f"Total pipeline: ~{training_time['total_pipeline_minutes']:.0f} minutes")

        # AI Recommendations
        st.subheader("🎯 AI Configuration Recommendations")

        recommendations = analysis["recommendations"]

        if recommendations["model_selection"].get("focus"):
            focus = recommendations["model_selection"]["focus"]
            st.success(f"🤖 **Recommended Models**: {', '.join(focus['models'])}")
            st.caption(f"📝 Reason: {focus['reason']}")

            if focus.get("avoid"):
                st.warning(f"⚠️ **Avoid**: {', '.join(focus['avoid'])}")

        if recommendations["optimization"].get("strategy"):
            strategy = recommendations["optimization"]["strategy"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Time Limit", f"{strategy['time_limit']} min")
            with col2:
                st.metric("🔄 Trials/Model", strategy.get("trials_per_model", 100))
            with col3:
                st.metric("⚡ Early Stop", "✅" if strategy.get("early_stopping", True) else "❌")

        # Auto-configuration preview
        st.subheader("⚙️ Recommended Configuration")
        auto_config = analysis["auto_config"]

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
            included = [
                m
                for m in auto_config["model_selection"]["include_models"]
                if m not in auto_config["model_selection"]["exclude_models"]
            ]
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
        if st.button("✨ Apply AI Recommendations", type="primary", width="stretch"):
            # Apply ALL auto config recommendations to session state
            auto_config = analysis["auto_config"]

            # Update optimization config
            st.session_state.optimization_config.update(auto_config["optimization"])

            # Update model selection
            st.session_state.selected_models = [
                model
                for model in auto_config["model_selection"]["include_models"]
                if model not in auto_config["model_selection"]["exclude_models"]
            ]

            # Update advanced config
            st.session_state.advanced_config = {
                "preprocessing": auto_config["preprocessing"],
                "dimensionality_reduction": auto_config["dimensionality_reduction"],
                "validation": auto_config["validation"],
                "performance": {"n_jobs": 1, "enable_caching": True, "memory_limit_gb": 8, "gpu_enabled": False},
            }

            # Mark as applied
            st.session_state.ai_recommendations_applied = True

            st.success(
                "✅ AI recommendations applied to all configuration tabs! Check other tabs to review and customize."
            )
            time.sleep(1)
            st.rerun()

    def _render_model_selection_tab(self):
        """Render model selection configuration tab."""
        st.header("🤖 Model Selection Configuration")

        # Get available models based on task type
        task_type = st.session_state.get("task_type", "Classification")

        st.subheader("📋 Available Models")

        if task_type == "Classification":
            all_models = {
                "LogisticRegression": {"complexity": "Low", "speed": "Fast", "interpretability": "High"},
                "SVM": {"complexity": "Medium", "speed": "Medium", "interpretability": "Medium"},
                "RandomForest": {"complexity": "Medium", "speed": "Fast", "interpretability": "Medium"},
                "XGBoost": {"complexity": "High", "speed": "Fast", "interpretability": "Low"},
                "MLP": {"complexity": "High", "speed": "Medium", "interpretability": "Low"},
                "KNN": {"complexity": "Low", "speed": "Slow", "interpretability": "Medium"},
            }
        elif task_type == "Regression":
            all_models = {
                "LinearRegression": {"complexity": "Low", "speed": "Fast", "interpretability": "High"},
                "SVR": {"complexity": "Medium", "speed": "Medium", "interpretability": "Medium"},
                "RandomForest": {"complexity": "Medium", "speed": "Fast", "interpretability": "Medium"},
                "XGBoost": {"complexity": "High", "speed": "Fast", "interpretability": "Low"},
                "MLP": {"complexity": "High", "speed": "Medium", "interpretability": "Low"},
                "KNN": {"complexity": "Low", "speed": "Slow", "interpretability": "Medium"},
            }
        else:  # Clustering
            all_models = {
                "KMeans": {"complexity": "Low", "speed": "Fast", "interpretability": "High"},
                "DBSCAN": {"complexity": "Medium", "speed": "Medium", "interpretability": "Medium"},
                "GaussianMixture": {"complexity": "Medium", "speed": "Medium", "interpretability": "Medium"},
                "AgglomerativeClustering": {"complexity": "High", "speed": "Slow", "interpretability": "High"},
            }

        # Display model selection interface
        # Default to whatever was previously selected (e.g. via "Apply AI Recommendations"
        # or the "Optimize for X" buttons below) instead of always resetting to "all models".
        valid_model_keys = list(all_models.keys())
        previous_selection = [m for m in st.session_state.get("selected_models") or [] if m in valid_model_keys]
        # No explicit `key=` here: Streamlit derives this widget's identity from its
        # `default` value when unkeyed, so recomputing `default` from session_state on
        # each rerun (e.g. after "Optimize for X" or "Apply AI Recommendations" changes
        # selected_models) correctly refreshes the widget. A fixed key would instead make
        # Streamlit ignore `default` after the first mount and keep the stale selection.
        selected_models = st.multiselect(
            "Select Models to Include",
            options=valid_model_keys,
            default=previous_selection or valid_model_keys,
            help="Choose which models to include in the AutoML pipeline",
        )

        # Display model characteristics
        if selected_models:
            st.subheader("📊 Selected Models Characteristics")
            model_df = []
            for model in selected_models:
                char = all_models[model]
                model_df.append(
                    {
                        "Model": model,
                        "Complexity": char["complexity"],
                        "Speed": char["speed"],
                        "Interpretability": char["interpretability"],
                    }
                )

            df = pd.DataFrame(model_df)
            st.dataframe(df, width="stretch")

        # Smart model selection
        st.subheader("🧠 Smart Model Selection")

        if st.session_state.get("dataset_config"):
            stats = st.session_state.dataset_config["basic_stats"]
            n_samples = stats["n_samples"]

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🎯 Optimize for Accuracy", width="stretch"):
                    if n_samples < 1000:
                        selected = ["LogisticRegression", "SVM", "RandomForest"]
                    else:
                        selected = ["XGBoost", "RandomForest", "MLP"]
                    st.session_state.selected_models = [m for m in selected if m in valid_model_keys]
                    st.rerun()

            with col2:
                if st.button("⚡ Optimize for Speed", width="stretch"):
                    if n_samples < 10000:
                        selected = ["LogisticRegression", "RandomForest"]
                    else:
                        selected = ["RandomForest", "XGBoost"]
                    st.session_state.selected_models = [m for m in selected if m in valid_model_keys]
                    st.rerun()

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
                min_value=5,
                max_value=120,
                value=opt_config["time_minutes"],
                help="Total time allocated for hyperparameter optimization",
            )

            max_trials = st.slider(
                "Maximum Trials per Model",
                min_value=20,
                max_value=500,
                value=opt_config["max_trials"],
                help="Maximum number of hyperparameter combinations to try",
            )

        with col2:
            include_ensemble = st.checkbox(
                "🎭 Include Ensemble Models",
                value=opt_config["include_ensemble"],
                help="Create ensemble models from optimized base models",
            )

            early_stopping = st.checkbox(
                "⚡ Enable Early Stopping",
                value=opt_config.get("early_stopping", True),
                help="Stop optimization early if no improvement is observed",
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
                "Cross-validation Strategy Optimization",
            ],
            default=opt_config.get("advanced_features", []),
            help="Enable advanced ML engineering features",
        )

        # Optimization strategy
        st.subheader("📈 Optimization Strategy")

        strategy = st.radio(
            "Choose optimization focus",
            options=["Balanced", "Accuracy-focused", "Speed-focused", "Interpretability-focused"],
            help="Select the primary optimization objective",
        )

        # Dataset-aware recommendations
        if st.session_state.get("dataset_config"):
            stats = st.session_state.dataset_config["basic_stats"]
            st.session_state.dataset_config["complexity"]

            st.info("💡 **AI Recommendation based on your dataset:**")

            if stats["n_samples"] < 1000:
                st.warning("Small dataset detected - recommend shorter optimization time with more thorough validation")
                rec_time, rec_trials = 10, 50
            elif stats["n_samples"] > 50000:
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
        st.session_state.optimization_config.update(
            {
                "time_minutes": time_minutes,
                "max_trials": max_trials,
                "include_ensemble": include_ensemble,
                "early_stopping": early_stopping,
                "advanced_features": advanced_features,
                "strategy": strategy,
            }
        )

    def _render_advanced_tab(self):
        """Render advanced configuration tab."""
        st.header("🔧 Advanced Configuration")

        # Read back any previously-applied config (e.g. from "Apply AI Recommendations")
        # so widgets don't silently reset to hardcoded defaults on every rerun.
        prev = st.session_state.get("advanced_config") or {}
        prev_prep = prev.get("preprocessing", {})
        prev_dimred = prev.get("dimensionality_reduction", {})
        prev_valid = prev.get("validation", {})
        prev_perf = prev.get("performance", {})

        def _idx(options, value, default_idx=0):
            return options.index(value) if value in options else default_idx

        # Preprocessing settings
        st.subheader("🛠️ Data Preprocessing")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Scaling & Normalization")
            scaling_options = ["standard", "minmax", "robust", "none"]
            scaling_method = st.selectbox(
                "Scaling Method",
                options=scaling_options,
                index=_idx(scaling_options, prev_prep.get("scaling_method") or prev_prep.get("scaling")),
                help="Choose feature scaling method",
            )

            imputation_options = ["median", "mean", "most_frequent"]
            imputation_numeric = st.selectbox(
                "Numeric Imputation",
                options=imputation_options,
                index=_idx(imputation_options, prev_prep.get("imputation_numeric")),
                help="Strategy for filling missing numeric values",
            )

        with col2:
            st.markdown("##### Feature Engineering")
            max_features = st.number_input(
                "Maximum Features",
                min_value=10,
                max_value=10000,
                value=min(10000, max(10, prev_prep.get("max_features", 1000))),
                help="Limit features for memory optimization",
            )

            remove_low_variance = st.checkbox(
                "Remove Low Variance Features",
                value=prev_prep.get("remove_low_variance", True),
                help="Remove features with very low variance",
            )

        # Dimensionality Reduction
        st.subheader("📐 Dimensionality Reduction")

        dimred_options = ["auto", "on", "off"]
        dimred_enable = st.radio(
            "Enable Dimensionality Reduction",
            options=dimred_options,
            index=_idx(dimred_options, prev_dimred.get("enable")),
            help="Auto: Enable for high-dimensional datasets",
        )

        if dimred_enable != "off":
            col1, col2, col3 = st.columns(3)

            with col1:
                dimred_method_options = ["auto", "pca", "tsvd", "ipca"]
                dimred_method = st.selectbox(
                    "Method",
                    options=dimred_method_options,
                    index=_idx(dimred_method_options, prev_dimred.get("method")),
                    help="Dimensionality reduction method",
                )

            with col2:
                variance_target = st.slider(
                    "Variance Target",
                    min_value=0.8,
                    max_value=0.99,
                    value=prev_dimred.get("variance_target", 0.95),
                    step=0.01,
                    help="Target explained variance ratio",
                )

            with col3:
                k_max = st.number_input(
                    "Max Components",
                    min_value=10,
                    max_value=1000,
                    value=prev_dimred.get("k_max", 256),
                    help="Maximum number of components",
                )

        # Validation Strategy
        st.subheader("✅ Validation Strategy")

        col1, col2, col3 = st.columns(3)

        with col1:
            cv_folds = st.slider(
                "CV Folds", min_value=3, max_value=20, value=prev_valid.get("cv_folds", 5),
                help="Number of cross-validation folds",
            )

        with col2:
            test_size = st.slider(
                "Test Size",
                min_value=0.1,
                max_value=0.4,
                value=prev_valid.get("test_size", 0.2),
                step=0.05,
                help="Proportion of data for testing",
            )

        with col3:
            random_seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=9999,
                value=prev_valid.get("random_seed", st.session_state.get("random_seed", 42)),
                help="Seed for reproducibility",
            )

        # Performance & Memory
        st.subheader("🚀 Performance & Memory")

        col1, col2 = st.columns(2)

        with col1:
            n_jobs = st.slider(
                "Parallel Jobs",
                min_value=-1,
                max_value=16,
                value=prev_perf.get("n_jobs", 1),
                help="-1 uses all available cores",
            )

            enable_caching = st.checkbox(
                "Enable Caching",
                value=prev_perf.get("enable_caching", True),
                help="Cache intermediate results for faster re-runs",
            )

        with col2:
            memory_limit_gb = st.slider(
                "Memory Limit (GB)",
                min_value=1,
                max_value=32,
                value=prev_perf.get("memory_limit_gb", 8),
                help="Maximum memory usage limit",
            )

            gpu_enabled = st.checkbox(
                "Enable GPU (if available)",
                value=prev_perf.get("gpu_enabled", False),
                help="Use GPU acceleration for compatible models",
            )

        # Store advanced configuration
        advanced_config = {
            "preprocessing": {
                "scaling_method": scaling_method,
                "imputation_numeric": imputation_numeric,
                "max_features": max_features,
                "remove_low_variance": remove_low_variance,
            },
            "dimensionality_reduction": {
                "enable": dimred_enable,
                "method": dimred_method if dimred_enable != "off" else None,
                "variance_target": variance_target if dimred_enable != "off" else None,
                "k_max": k_max if dimred_enable != "off" else None,
            },
            "validation": {"cv_folds": cv_folds, "test_size": test_size, "random_seed": random_seed},
            "performance": {
                "n_jobs": n_jobs,
                "enable_caching": enable_caching,
                "memory_limit_gb": memory_limit_gb,
                "gpu_enabled": gpu_enabled,
            },
        }

        st.session_state.advanced_config = advanced_config

    def _render_feature_engineering_section(self, data):
        """Render comprehensive feature engineering section."""
        st.markdown("### 🛠️ **Feature Engineering & Data Preparation**")
        st.markdown("*Edit your dataset before ML configuration. Changes will be applied to the pipeline.*")

        # Feature Engineering Tabs
        fe_tab1, fe_tab2, fe_tab3, fe_tab4, fe_tab5 = st.tabs(
            [
                "📊 Column Selection",
                "🧹 Data Cleaning",
                "🔄 Transformations",
                "➕ Feature Creation",
                "💾 Export Changes",
            ]
        )

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
            if "selected_columns" not in st.session_state:
                st.session_state.selected_columns = list(data.columns)

            # Column selection interface
            all_columns = list(data.columns)
            selected_cols = st.multiselect(
                "Choose columns for ML pipeline:",
                options=all_columns,
                default=st.session_state.selected_columns,
                help="Select only the columns you want to use for machine learning",
            )
            st.session_state.selected_columns = selected_cols

            if len(selected_cols) != len(all_columns):
                removed_cols = [col for col in all_columns if col not in selected_cols]
                st.warning(
                    f"**Removing {len(removed_cols)} columns:** {', '.join(removed_cols[:5])}{'...' if len(removed_cols) > 5 else ''}"
                )

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
            st.dataframe(preview_data.head(), width="stretch")

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
        selected_cols = st.session_state.get("selected_columns", list(data.columns))
        clean_data = data[selected_cols] if selected_cols else data

        missing_info = clean_data.isnull().sum()
        if missing_info.sum() > 0:
            st.markdown("**Missing Data Summary:**")
            missing_df = pd.DataFrame(
                {
                    "Column": missing_info.index,
                    "Missing Count": missing_info.values,
                    "Missing %": (missing_info.values / len(clean_data)) * 100,
                }
            ).sort_values("Missing Count", ascending=False)

            st.dataframe(missing_df[missing_df["Missing Count"] > 0], width="stretch")

            # Missing value handling options
            st.markdown("**Missing Value Strategy:**")

            col1, col2 = st.columns(2)
            with col1:
                missing_strategy = st.selectbox(
                    "Choose strategy for numeric columns:",
                    ["mean", "median", "mode", "drop_rows", "forward_fill"],
                    help="How to handle missing values in numeric columns",
                )
                st.session_state.missing_numeric_strategy = missing_strategy

            with col2:
                missing_categorical = st.selectbox(
                    "Choose strategy for categorical columns:",
                    ["mode", "unknown", "drop_rows"],
                    help="How to handle missing values in categorical columns",
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
                "Outlier detection method:", ["iqr", "z_score", "none"], help="Method to detect and handle outliers"
            )

            if outlier_method != "none":
                outlier_threshold = st.slider(
                    "Outlier sensitivity:",
                    min_value=1.0,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="Lower values = more sensitive to outliers",
                )
                st.session_state.outlier_method = outlier_method
                st.session_state.outlier_threshold = outlier_threshold

    def _render_transformations_tab(self, data):
        """Render data transformation options."""
        st.subheader("🔄 Data Transformations")

        selected_cols = st.session_state.get("selected_columns", list(data.columns))
        clean_data = data[selected_cols] if selected_cols else data

        # Scaling options for numeric data
        numeric_cols = clean_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.markdown("**Numeric Data Scaling:**")

            scaling_method = st.selectbox(
                "Choose scaling method:",
                ["none", "standard", "minmax", "robust", "quantile"],
                help="Scaling method for numeric features",
            )
            st.session_state.scaling_method = scaling_method

            if scaling_method != "none":
                st.info(f"Will apply {scaling_method} scaling to numeric columns: {', '.join(numeric_cols)}")

        # Encoding options for categorical data
        categorical_cols = clean_data.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            st.markdown("**Categorical Data Encoding:**")

            encoding_method = st.selectbox(
                "Choose encoding method:",
                ["auto", "onehot", "label", "target"],
                help="Encoding method for categorical features",
            )
            st.session_state.encoding_method = encoding_method

            # Show cardinality info
            st.markdown("**Categorical Column Cardinality:**")
            cardinality_info = []
            for col in categorical_cols:
                unique_count = clean_data[col].nunique()
                cardinality_info.append(
                    {
                        "Column": col,
                        "Unique Values": unique_count,
                        "Recommended": "One-Hot" if unique_count <= 10 else "Label/Target",
                    }
                )

            st.dataframe(pd.DataFrame(cardinality_info), width="stretch")

        # Advanced transformations
        st.markdown("**Advanced Transformations:**")

        col1, col2 = st.columns(2)
        with col1:
            log_transform = st.multiselect(
                "Apply log transformation to columns:",
                options=list(numeric_cols),
                help="Useful for highly skewed numeric data",
            )
            st.session_state.log_transform_cols = log_transform

        with col2:
            polynomial_features = st.checkbox(
                "Create polynomial features", help="Generate interaction and polynomial terms"
            )
            if polynomial_features:
                poly_degree = st.slider("Polynomial degree:", 2, 3, 2)
                st.session_state.polynomial_degree = poly_degree
            st.session_state.create_polynomial = polynomial_features

    def _render_feature_creation_tab(self, data):
        """Render feature creation interface."""
        st.subheader("➕ Feature Creation & Engineering")

        selected_cols = st.session_state.get("selected_columns", list(data.columns))
        clean_data = data[selected_cols] if selected_cols else data

        # Date/time feature extraction
        datetime_cols = clean_data.select_dtypes(include=["datetime64"]).columns
        date_like_cols = [col for col in clean_data.columns if "date" in col.lower() or "time" in col.lower()]

        if len(datetime_cols) > 0 or len(date_like_cols) > 0:
            st.markdown("**Date/Time Feature Engineering:**")

            potential_date_cols = list(datetime_cols) + [col for col in date_like_cols if col not in datetime_cols]

            selected_date_cols = st.multiselect(
                "Extract features from date columns:",
                options=potential_date_cols,
                help="Extract year, month, day, weekday, etc. from date columns",
            )

            if selected_date_cols:
                date_features = st.multiselect(
                    "Select date features to create:",
                    ["year", "month", "day", "weekday", "quarter", "is_weekend"],
                    default=["year", "month", "weekday"],
                )
                st.session_state.date_features = {"columns": selected_date_cols, "features": date_features}

        # Text feature extraction
        text_cols = [
            col for col in clean_data.select_dtypes(include=["object"]).columns if clean_data[col].str.len().mean() > 20
        ]  # Likely text columns

        if len(text_cols) > 0:
            st.markdown("**Text Feature Engineering:**")

            selected_text_cols = st.multiselect(
                "Extract features from text columns:",
                options=text_cols,
                help="Extract length, word count, etc. from text columns",
            )

            if selected_text_cols:
                text_features = st.multiselect(
                    "Select text features to create:",
                    ["length", "word_count", "sentiment", "contains_numbers"],
                    default=["length", "word_count"],
                )
                st.session_state.text_features = {"columns": selected_text_cols, "features": text_features}

        # Mathematical combinations
        numeric_cols = clean_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            st.markdown("**Mathematical Feature Combinations:**")

            create_ratios = st.checkbox("Create ratio features", help="Create ratios between numeric columns")

            if create_ratios:
                ratio_pairs = st.multiselect(
                    "Select column pairs for ratios:",
                    options=[(f"{col1}/{col2}") for col1 in numeric_cols for col2 in numeric_cols if col1 != col2][:20],
                    help="Select pairs of columns to create ratio features",
                )
                st.session_state.ratio_features = ratio_pairs

        # Custom feature formula
        st.markdown("**Custom Feature Formula:**")
        custom_formula = st.text_input(
            "Create custom feature (e.g., 'col1 + col2 * 2'):", help="Use column names and basic math operators"
        )
        if custom_formula:
            custom_name = st.text_input("Feature name:", value="custom_feature")
            st.session_state.custom_feature = {"formula": custom_formula, "name": custom_name}

    def _render_export_changes_tab(self, data):
        """Render export and apply changes interface."""
        st.subheader("💾 Apply & Export Changes")

        # Show summary of all pending changes
        st.markdown("**Summary of Pending Changes:**")
        changes_summary = []

        # Column selection changes
        selected_cols = st.session_state.get("selected_columns", list(data.columns))
        if len(selected_cols) != len(data.columns):
            changes_summary.append(f"• Column selection: {len(selected_cols)}/{len(data.columns)} columns selected")

        # Data cleaning changes
        if st.session_state.get("missing_numeric_strategy"):
            changes_summary.append(f"• Missing values: {st.session_state.missing_numeric_strategy} for numeric")

        if st.session_state.get("remove_duplicates"):
            changes_summary.append("• Remove duplicate rows")

        # Transformation changes
        if st.session_state.get("scaling_method", "none") != "none":
            changes_summary.append(f"• Scaling: {st.session_state.scaling_method}")

        if st.session_state.get("encoding_method", "auto") != "auto":
            changes_summary.append(f"• Encoding: {st.session_state.encoding_method}")

        # Feature creation changes
        if st.session_state.get("date_features"):
            changes_summary.append("• Date feature extraction")

        if st.session_state.get("create_polynomial"):
            changes_summary.append("• Polynomial features")

        if changes_summary:
            for change in changes_summary:
                st.write(change)
        else:
            st.info("No changes pending. Select modifications in the tabs above.")

        # Apply changes button
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Apply Changes to Dataset", type="primary", width="stretch"):
                with st.spinner("Applying feature engineering changes..."):
                    try:
                        processed_data = self._apply_feature_engineering_changes(data)
                        st.session_state.data = processed_data
                        st.session_state.feature_engineering_applied = True
                        st.success(
                            f"✅ Changes applied! New dataset: {processed_data.shape[0]} × {processed_data.shape[1]}"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error applying changes: {e}")

        with col2:
            if st.button("📁 Export Modified Dataset", width="stretch"):
                try:
                    processed_data = self._apply_feature_engineering_changes(data)
                    csv = processed_data.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"modified_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
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
        selected_cols = st.session_state.get("selected_columns", list(data.columns))
        if selected_cols:
            processed_data = processed_data[selected_cols]

        # 2. Handle missing values
        missing_numeric = st.session_state.get("missing_numeric_strategy")
        missing_categorical = st.session_state.get("missing_categorical_strategy")

        if missing_numeric:
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            if missing_numeric == "mean":
                processed_data[numeric_cols] = processed_data[numeric_cols].fillna(processed_data[numeric_cols].mean())
            elif missing_numeric == "median":
                processed_data[numeric_cols] = processed_data[numeric_cols].fillna(
                    processed_data[numeric_cols].median()
                )
            elif missing_numeric == "mode":
                for col in numeric_cols:
                    processed_data[col].fillna(
                        processed_data[col].mode().iloc[0] if not processed_data[col].mode().empty else 0, inplace=True
                    )

        if missing_categorical:
            categorical_cols = processed_data.select_dtypes(include=["object", "category"]).columns
            if missing_categorical == "mode":
                for col in categorical_cols:
                    processed_data[col].fillna(
                        processed_data[col].mode().iloc[0] if not processed_data[col].mode().empty else "Unknown",
                        inplace=True,
                    )
            elif missing_categorical == "unknown":
                processed_data[categorical_cols] = processed_data[categorical_cols].fillna("Unknown")

        # 3. Remove duplicates
        if st.session_state.get("remove_duplicates"):
            processed_data = processed_data.drop_duplicates()

        # 4. Create date features
        if st.session_state.get("date_features"):
            date_config = st.session_state.date_features
            for col in date_config["columns"]:
                if col in processed_data.columns:
                    # Convert to datetime if not already
                    processed_data[col] = pd.to_datetime(processed_data[col], errors="coerce")

                    for feature in date_config["features"]:
                        if feature == "year":
                            processed_data[f"{col}_year"] = processed_data[col].dt.year
                        elif feature == "month":
                            processed_data[f"{col}_month"] = processed_data[col].dt.month
                        elif feature == "day":
                            processed_data[f"{col}_day"] = processed_data[col].dt.day
                        elif feature == "weekday":
                            processed_data[f"{col}_weekday"] = processed_data[col].dt.dayofweek
                        elif feature == "quarter":
                            processed_data[f"{col}_quarter"] = processed_data[col].dt.quarter
                        elif feature == "is_weekend":
                            processed_data[f"{col}_is_weekend"] = (processed_data[col].dt.dayofweek >= 5).astype(int)

        # 5. Create text features
        if st.session_state.get("text_features"):
            text_config = st.session_state.text_features
            for col in text_config["columns"]:
                if col in processed_data.columns:
                    for feature in text_config["features"]:
                        if feature == "length":
                            processed_data[f"{col}_length"] = processed_data[col].astype(str).str.len()
                        elif feature == "word_count":
                            processed_data[f"{col}_word_count"] = processed_data[col].astype(str).str.split().str.len()
                        elif feature == "contains_numbers":
                            processed_data[f"{col}_has_numbers"] = (
                                processed_data[col].astype(str).str.contains(r"\d").astype(int)
                            )

        # 6. Log transformations
        log_cols = st.session_state.get("log_transform_cols", [])
        for col in log_cols:
            if col in processed_data.columns:
                # Add small constant to avoid log(0)
                processed_data[f"{col}_log"] = np.log(processed_data[col] + 1)

        # 7. Ratio features
        if st.session_state.get("ratio_features"):
            for ratio in st.session_state.ratio_features:
                if "/" in ratio:
                    col1, col2 = ratio.split("/")
                    if col1 in processed_data.columns and col2 in processed_data.columns:
                        processed_data[f"{col1}_to_{col2}_ratio"] = processed_data[col1] / (
                            processed_data[col2] + 1e-8
                        )  # Avoid division by zero

        return processed_data

    def _reset_feature_engineering_settings(self):
        """Reset all feature engineering settings."""
        fe_keys = [
            "selected_columns",
            "missing_numeric_strategy",
            "missing_categorical_strategy",
            "remove_duplicates",
            "outlier_method",
            "outlier_threshold",
            "scaling_method",
            "encoding_method",
            "log_transform_cols",
            "create_polynomial",
            "polynomial_degree",
            "date_features",
            "text_features",
            "ratio_features",
            "custom_feature",
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
                    help="Current dataset dimensions",
                )

            with col2:
                memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric(label="💾 Memory Usage", value=f"{memory_mb:.1f} MB", help="Dataset memory footprint")

            with col3:
                fe_applied = st.session_state.get("feature_engineering_applied", False)
                fe_status = "✅ Applied" if fe_applied else "⏳ None"
                fe_delta = "Modified" if fe_applied else "Original"
                st.metric(
                    label="🛠️ Feature Engineering", value=fe_status, delta=fe_delta, help="Feature engineering status"
                )

            with col4:
                ai_analyzed = st.session_state.get("ai_analysis") is not None
                ai_status = "✅ Complete" if ai_analyzed else "⏳ Pending"
                ai_delta = "Analyzed" if ai_analyzed else "Not analyzed"
                st.metric(label="🧠 AI Analysis", value=ai_status, delta=ai_delta, help="AI analysis status")

        st.markdown("---")

        # Professional navigation buttons
        col1, col2, col3 = st.columns([1, 1, 2], gap="medium")

        with col1:
            if st.button("🔄 Reset Dataset", help="Go back to original uploaded dataset", width="stretch"):
                # Reset to original state
                self._reset_feature_engineering_settings()
                st.session_state.show_feature_engineering = False
                st.session_state.ai_analysis = None
                st.session_state.feature_engineering_applied = False
                st.success("✅ Reset to original dataset")
                st.rerun()

        with col2:
            if st.session_state.data is not None:
                st.download_button(
                    "📁 Export Dataset",
                    data=st.session_state.data.to_csv(index=False),
                    file_name=f"prepared_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download current dataset as CSV",
                    width="stretch",
                )

        with col3:
            # Only allow proceeding if we have data
            if st.session_state.data is not None:
                if st.button("➡️ **Continue to ML Configuration**", type="primary", width="stretch"):
                    st.session_state.app_stage = "configure"
                    st.rerun()
            else:
                st.button(
                    "➡️ Upload Dataset First",
                    disabled=True,
                    width="stretch",
                    help="Please upload a dataset before proceeding",
                )


# Main entry point for Streamlit
def main():
    """Main function to run the AutoML Dashboard."""
    # Initialize and run dashboard
    dashboard = AutoMLDashboard()
    dashboard.render()


if __name__ == "__main__":
    main()
