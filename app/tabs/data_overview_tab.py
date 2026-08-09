"""
Data Overview Tab - Display dataset statistics and visualizations.
"""

import logging

import numpy as np
import streamlit as st

from core.visualize import Visualizer

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class DataOverviewTab(BaseTab):
    """Tab for displaying dataset overview and statistics."""

    def render(self) -> None:
        """Render the data overview tab."""
        if not self.require_data():
            return

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
                    help="Choose analysis focus",
                )

            # Generate insights (initially or on refresh)
            if st.session_state.ai_insights is None or refresh_insights:
                with st.spinner("🤖 Enhanced AI is performing comprehensive dataset analysis..."):
                    try:
                        # Determine task type and target
                        task_type = st.session_state.get("task_type", "Classification")
                        target_col = st.session_state.get("target_col", None)

                        # Use enhanced AI engine if available
                        if st.session_state.enhanced_ai_engine:
                            # Enhanced comprehensive analysis
                            stats = st.session_state.enhanced_ai_engine.analyze_dataset_comprehensive(
                                data=data, target_col=target_col, task_type=task_type.lower()
                            )

                            insights = st.session_state.enhanced_ai_engine.generate_comprehensive_insights(
                                stats=stats, context=analysis_depth
                            )
                        else:
                            # Fallback to standard AI engine
                            stats = st.session_state.ai_engine.analyze_dataset(
                                data=data, target_col=target_col, task_type=task_type.lower()
                            )

                            insights = st.session_state.ai_engine.generate_insights(stats=stats, context=analysis_depth)

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
                is_ai_generated = insights.get("_source", "unknown") == "ai"
                is_enhanced = (
                    insights.get("_source", "unknown") == "enhanced_rules" or st.session_state.enhanced_ai_engine
                )
                quality_score = insights.get("_quality_score", 0)

                # Header based on insight type
                if is_ai_generated and is_enhanced:
                    with st.expander("🧠 **Enhanced AI-Powered Dataset Analysis** ⭐", expanded=True):
                        st.success("🎯 **Powered by Enhanced AI** - Comprehensive analysis with deep insights")
                elif is_ai_generated:
                    with st.expander("🤖 **AI-Generated Dataset Analysis** ✨", expanded=True):
                        st.success("🎯 **Powered by AI** - Dynamic analysis from your LLM")
                else:
                    with st.expander("📊 **Comprehensive Dataset Analysis** (Enhanced Rules)", expanded=True):
                        if "_notice" in insights:
                            st.warning(insights["_notice"])
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
                if "dataset_overview" in insights:
                    st.markdown("### 📋 **Dataset Overview**")
                    st.markdown(insights["dataset_overview"])
                    st.markdown("---")

                # Main Analysis in columns
                col1, col2 = st.columns(2)

                with col1:
                    # Strengths (Enhanced)
                    if "key_strengths" in insights or "strengths" in insights:
                        st.markdown("#### ✅ **Key Strengths**")
                        strengths = insights.get("key_strengths", insights.get("strengths", []))
                        if isinstance(strengths, list):
                            for strength in strengths:
                                st.success(f"• {strength}")
                        else:
                            st.success(f"• {strengths}")

                with col2:
                    # Challenges (Enhanced)
                    if "critical_challenges" in insights or "challenges" in insights:
                        st.markdown("#### ⚠️ **Critical Challenges**")
                        challenges = insights.get("critical_challenges", insights.get("challenges", []))
                        if isinstance(challenges, list):
                            for challenge in challenges:
                                st.warning(f"• {challenge}")
                        else:
                            st.warning(f"• {challenges}")

                # Data Quality Assessment (Enhanced)
                if "data_quality_assessment" in insights:
                    st.markdown("#### 🔍 **Data Quality Assessment**")
                    st.info(insights["data_quality_assessment"])

                # Preprocessing Strategy (Enhanced)
                if "preprocessing_strategy" in insights:
                    st.markdown("#### 🔧 **Preprocessing Strategy**")
                    prep_steps = insights["preprocessing_strategy"]
                    if isinstance(prep_steps, list):
                        for i, step in enumerate(prep_steps, 1):
                            st.markdown(f"**{i}.** {step}")
                    else:
                        st.markdown(prep_steps)

                # Model Recommendations (Enhanced)
                if "recommended_models" in insights:
                    st.markdown("#### 🎯 **Recommended Models**")
                    models = insights["recommended_models"]
                    if isinstance(models, list):
                        for model in models:
                            st.success(f"🤖 {model}")
                    else:
                        st.success(f"🤖 {models}")

                # Feature Engineering (Enhanced)
                if "feature_engineering_opportunities" in insights:
                    st.markdown("#### ⚙️ **Feature Engineering Opportunities**")
                    feature_eng = insights["feature_engineering_opportunities"]
                    if isinstance(feature_eng, list):
                        for opp in feature_eng:
                            st.info(f"🛠️ {opp}")
                    else:
                        st.info(f"🛠️ {feature_eng}")

                # Statistical Insights (Enhanced)
                if "statistical_insights" in insights:
                    st.markdown("#### 📈 **Statistical Insights**")
                    stats_insights = insights["statistical_insights"]
                    if isinstance(stats_insights, list):
                        for stat in stats_insights:
                            st.markdown(f"• {stat}")
                    else:
                        st.markdown(stats_insights)

                # Risk Assessment (Enhanced)
                if "risk_factors" in insights:
                    st.markdown("#### ⚠️ **Risk Factors**")
                    risks = insights["risk_factors"]
                    if isinstance(risks, list):
                        for risk in risks:
                            st.error(f"🚨 {risk}")
                    else:
                        st.error(f"🚨 {risks}")

                # Performance Expectations (Enhanced)
                if "expected_performance" in insights:
                    st.markdown("#### 🎯 **Performance Expectations**")
                    st.success(insights["expected_performance"])

                # Next Steps (Enhanced)
                if "next_steps" in insights:
                    st.markdown("#### 🚀 **Next Steps**")
                    steps = insights["next_steps"]
                    if isinstance(steps, list):
                        for i, step in enumerate(steps, 1):
                            st.markdown(f"**{i}.** {step}")
                    else:
                        st.markdown(steps)

                # Additional advanced insights (if available)
                advanced_fields = [
                    "tier1_models",
                    "tier2_models",
                    "avoid_models",
                    "hyperparameter_priorities",
                    "validation_strategy",
                    "advanced_statistical_analysis",
                    "feature_space_analysis",
                    "domain_specific_recommendations",
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
                st.dataframe(data.head(10), width="stretch", height=400)
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
            if profile.get("n_constant_features", 0) > 0:
                st.warning(
                    f"⚠️ Found {profile['n_constant_features']} constant features (zero variance). These will be automatically removed during preprocessing."
                )
                if "constant_features" in profile:
                    with st.expander("Show constant features"):
                        st.write(profile["constant_features"])

            col1, col2 = st.columns(2)
            with col1:
                # Filter out constant_features list for cleaner display
                display_profile = {
                    k: v for k, v in list(profile.items())[: len(profile) // 2] if k != "constant_features"
                }
                st.json(display_profile)
            with col2:
                display_profile = {
                    k: v for k, v in list(profile.items())[len(profile) // 2 :] if k != "constant_features"
                }
                st.json(display_profile)

        # Visualizations
        st.subheader("Data Visualizations")

        # Correlation heatmap for numeric features
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            visualizer = Visualizer()
            fig = visualizer.plot_correlation_heatmap(numeric_data.iloc[:, :20])  # Limit to 20 features
            st.plotly_chart(fig, width="stretch")

    def _generate_basic_insights(self, data, target_col=None, task_type="Classification"):
        """Generate comprehensive dataset insights when AI engine is not available."""
        try:
            from collections import Counter

            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=["object"]).columns
            datetime_cols = data.select_dtypes(include=["datetime64"]).columns

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
                        imbalance_ratio = max_class / min_class if min_class > 0 else float("inf")

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
                "recommendations": recommendations,
            }

        except Exception as e:
            # Fallback for any errors
            return {
                "summary": f"Basic analysis completed for {data.shape[0]:,} samples",
                "strengths": ["Data loaded successfully"],
                "challenges": [f"Analysis error: {str(e)[:100]}..."],
                "recommendations": ["Check data format and verify column types"],
            }
