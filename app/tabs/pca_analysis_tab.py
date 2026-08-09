"""
PCA Analysis Tab - Display dimensionality reduction results.
"""

import logging

import numpy as np
import pandas as pd
import streamlit as st

from core.dimred import DimRedConfig, make_dimred

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class PCAAnalysisTab(BaseTab):
    """Tab for displaying PCA analysis results."""

    def render(self) -> None:
        """Render the PCA analysis tab."""
        st.subheader("📐 Dimensionality Reduction Analysis")

        if st.session_state.data is None:
            st.warning("⚠️ Please upload data first to view PCA analysis.")
            return

        # Check if dimred was enabled and run
        if not hasattr(st.session_state, "dimred_results") or st.session_state.dimred_results is None:
            st.info("💡 Run PCA analysis on your current dataset:")

            col1, col2 = st.columns([1, 1])
            with col1:
                # PCA configuration
                st.markdown("#### ⚙️ PCA Configuration")
                variance_target = st.slider(
                    "Variance Target",
                    min_value=0.8,
                    max_value=0.99,
                    value=0.95,
                    step=0.01,
                    help="How much variance to preserve",
                )
                max_components = st.number_input(
                    "Max Components", min_value=2, max_value=50, value=10, help="Maximum number of components"
                )

                if st.button("🔍 Analyze with PCA", type="primary"):
                    with st.spinner("Running PCA analysis..."):
                        try:
                            # Get data
                            if hasattr(st.session_state, "X_processed") and st.session_state.X_processed is not None:
                                X_data = st.session_state.X_processed
                                y_data = st.session_state.get("y_processed", None)
                            else:
                                X_data = st.session_state.data.select_dtypes(include=[np.number])
                                y_data = None

                            # Validate data before transformation
                            if X_data.shape[1] < 2:
                                st.error("❌ PCA requires at least 2 features. Current dataset has only 1 feature.")
                                return

                            # Handle missing values before PCA
                            from sklearn.impute import SimpleImputer

                            if pd.DataFrame(X_data).isnull().any().any():
                                st.info("⚠️ Missing values detected. Imputing with median strategy...")
                                imputer = SimpleImputer(strategy="median")
                                X_data = imputer.fit_transform(X_data)

                            # Run PCA analysis
                            dimred_config = DimRedConfig(
                                enable="on", method="pca", variance_target=variance_target, k_max=int(max_components)
                            )
                            pca = make_dimred(
                                is_sparse_after_ohe=False,
                                n_features=X_data.shape[1] if hasattr(X_data, "shape") else len(X_data[0]),
                                n_samples=X_data.shape[0] if hasattr(X_data, "shape") else len(X_data),
                                cfg=dimred_config,
                            )

                            if pca is None:
                                st.error("❌ PCA not applicable to this dataset (insufficient features or samples)")
                                return

                            # Fit and transform
                            X_transformed = pca.fit_transform(X_data)

                            # Create and store results
                            st.session_state.dimred_results = {
                                "pca_transformer": pca,
                                "X_transformed": X_transformed,
                                "explained_variance_ratio": pca.explained_variance_ratio_,
                                "n_components": pca.n_components_,
                                "method": "pca",
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

            # Check for PCA results in various possible locations
            pca_transformer = None
            if dimred_results:
                # Check direct pca_transformer
                if "pca_transformer" in dimred_results:
                    pca_transformer = dimred_results["pca_transformer"]
                # Check in recommended config
                elif "recommended_config" in dimred_results:
                    rec_config = dimred_results["recommended_config"]
                    if hasattr(rec_config, "method") and rec_config.method == "pca":
                        # Look for PCA transformer in other keys
                        for key, value in dimred_results.items():
                            if hasattr(value, "explained_variance_ratio_"):
                                pca_transformer = value
                                break
                # Check all values for PCA-like objects
                if not pca_transformer:
                    for key, value in dimred_results.items():
                        if hasattr(value, "explained_variance_ratio_"):
                            pca_transformer = value
                            st.info(f"Found PCA transformer in key: {key}")
                            break

            # Show scree plot if PCA was used
            if pca_transformer and hasattr(pca_transformer, "explained_variance_ratio_"):
                from core.visualize import plot_pca_scree

                try:
                    fig = plot_pca_scree(pca_transformer)
                    st.plotly_chart(fig, width="stretch")
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

            if "pca_transformer" in dimred_results:
                pca = dimred_results["pca_transformer"]

                # Show key metrics
                st.metric("Components Used", dimred_results.get("n_components", "N/A"))
                st.metric("Variance Preserved", f"{np.sum(pca.explained_variance_ratio_):.1%}")
                st.metric("Original Features", pca.n_features_in_ if hasattr(pca, "n_features_in_") else "N/A")

                # Show 2D projection if available
                if "X_transformed" in dimred_results and dimred_results["X_transformed"].shape[1] >= 2:
                    st.markdown("#### 🎨 2D Projection")
                    try:
                        from core.visualize import plot_pca_2d_scatter

                        y_data = st.session_state.get("y_processed", None)

                        # Handle different data types for y
                        if y_data is not None:
                            if len(np.unique(y_data)) > 20:
                                y_data = None  # Too many classes, don't color

                        # Get explained variance ratio for proper axis labels
                        explained_variance = (
                            pca.explained_variance_ratio_ if hasattr(pca, "explained_variance_ratio_") else None
                        )

                        fig_2d = plot_pca_2d_scatter(
                            dimred_results["X_transformed"][:, :2], y_data, "PCA 2D Projection", explained_variance
                        )
                        st.plotly_chart(fig_2d, width="stretch")
                    except Exception as e:
                        st.error(f"Error creating 2D plot: {e}")
            else:
                # Generate AI-powered PCA recommendations
                if (
                    st.session_state.ai_engine
                    and hasattr(st.session_state, "X_processed")
                    and st.session_state.X_processed is not None
                ):
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
                            target_type=st.session_state.get("task_type", "classification").lower(),
                            n_classes=None,
                            class_balance=None,
                            missing_rate=0.0,  # Processed data has no missing values
                            feature_correlations=[],
                            top_features=[],
                            data_quality_score=95.0,
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
                        if "recommendation" in pca_insights:
                            if "Highly Recommended" in pca_insights["recommendation"]:
                                st.success(f"🎯 **{pca_insights['recommendation']}**")
                            elif "Recommended" in pca_insights["recommendation"]:
                                st.info(f"📊 **{pca_insights['recommendation']}**")
                            else:
                                st.warning(f"⚠️ **{pca_insights['recommendation']}**")

                        if "reasoning" in pca_insights:
                            st.write(f"**💡 AI Analysis:** {pca_insights['reasoning']}")

                        if "optimal_components" in pca_insights:
                            st.metric("🎯 Suggested Components", pca_insights["optimal_components"])

                        if "expected_benefits" in pca_insights:
                            st.write("**✅ Expected Benefits:**")
                            benefits = pca_insights["expected_benefits"]
                            if isinstance(benefits, list):
                                for benefit in benefits:
                                    st.success(f"• {benefit}")
                            else:
                                st.success(f"• {benefits}")

                        if "considerations" in pca_insights:
                            st.write("**⚠️ Important Considerations:**")
                            considerations = pca_insights["considerations"]
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
