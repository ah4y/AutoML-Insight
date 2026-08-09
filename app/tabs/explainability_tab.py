"""
Explainability Tab - Display SHAP values and feature importance.
"""

import logging

import numpy as np
import pandas as pd
import streamlit as st

from core.explain import ModelExplainer
from core.visualize import Visualizer

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class ExplainabilityTab(BaseTab):
    """Tab for displaying model explainability results."""

    def render(self) -> None:
        """Render the explainability tab."""
        st.subheader("🔍 Model Explainability")

        # Check for both standard and professional results
        has_results = st.session_state.results or st.session_state.get("professional_results")

        if not has_results:
            st.info("Run AutoML to see explainability results")
            return

        # Check if we have models to explain
        if not st.session_state.get("models"):
            st.warning("⚠️ No trained models available for explainability analysis.")
            st.info("💡 Models may have failed during training. Check the console logs for errors.")
            return

        # Check if clustering task
        is_clustering = st.session_state.task_type == "Clustering"

        # Model selection with unique key
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox(
            "Select Model",
            model_names,
            key="explainability_model_selector",
            help="Switch models to see their specific explanations",
        )

        if selected_model:
            # Initialize cache for explanations if not exists
            if "explainability_cache" not in st.session_state:
                st.session_state.explainability_cache = {}

            # Check cache first to avoid recomputation
            cache_key = f"{selected_model}_{is_clustering}"
            use_cached = cache_key in st.session_state.explainability_cache

            # Silent caching - no need to show message, just faster performance

            model = st.session_state.models[selected_model]

            # Check if X_processed and preprocessor exist
            if not hasattr(st.session_state, "X_processed") or st.session_state.X_processed is None:
                st.warning("⚠️ Processed data not available.")
                st.info(
                    "💡 Explainability requires the preprocessed dataset. Please run AutoML again or ensure data preprocessing was completed."
                )

                # Try to show basic model information instead
                st.markdown("#### 📊 Model Information")
                st.write(f"**Model Type:** {type(model).__name__}")
                if hasattr(model, "get_params"):
                    st.write("**Parameters:**")
                    st.json(model.get_params())
                return

            if not hasattr(st.session_state, "preprocessor") or st.session_state.preprocessor is None:
                st.warning("⚠️ Preprocessor not available.")
                st.info("💡 Feature names cannot be determined without the preprocessor.")
                return

            X = st.session_state.X_processed
            feature_names = st.session_state.preprocessor.get_feature_names()

            # Add comprehensive visualizations for classification models
            if not is_clustering:
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("#### 📊 Feature Importance")
                    try:
                        # Get feature importance based on model type
                        if hasattr(model, "feature_importances_"):
                            # Tree-based models
                            importances = model.feature_importances_
                            importance_type = "Built-in Feature Importance"
                        elif hasattr(model, "coef_"):
                            # Linear models
                            coef = model.coef_
                            importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
                            importance_type = "Coefficient Magnitude"
                        else:
                            # Use permutation importance for other models
                            from sklearn.inspection import permutation_importance

                            perm_result = permutation_importance(
                                model,
                                X[:100],
                                model.predict(X[:100]),  # Use subset for speed
                                n_repeats=3,
                                random_state=42,
                                n_jobs=1,
                            )
                            importances = perm_result.importances_mean
                            importance_type = "Permutation Importance"

                        # Create importance plot
                        importance_df = (
                            pd.DataFrame({"Feature": feature_names, "Importance": importances})
                            .sort_values("Importance", ascending=True)
                            .tail(15)
                        )  # Top 15

                        import plotly.express as px

                        fig_importance = px.bar(
                            importance_df,
                            x="Importance",
                            y="Feature",
                            title=f"{importance_type} - {selected_model}",
                            orientation="h",
                        )
                        fig_importance.update_layout(height=400)
                        st.plotly_chart(fig_importance, width="stretch")

                    except Exception as e:
                        st.warning(f"Could not generate feature importance plot: {e}")

                with col2:
                    st.markdown("#### 🎯 Performance Breakdown")
                    try:
                        # Show model performance metrics in detail
                        results = st.session_state.results.get(selected_model, {})

                        metrics_data = []
                        if "accuracy_mean" in results:
                            metrics_data.append(
                                ["Accuracy", f"{results['accuracy_mean']:.3f} ± {results.get('accuracy_std', 0):.3f}"]
                            )
                        if "precision_mean" in results:
                            metrics_data.append(
                                [
                                    "Precision",
                                    f"{results['precision_mean']:.3f} ± {results.get('precision_std', 0):.3f}",
                                ]
                            )
                        if "recall_mean" in results:
                            metrics_data.append(
                                ["Recall", f"{results['recall_mean']:.3f} ± {results.get('recall_std', 0):.3f}"]
                            )
                        if "f1_mean" in results:
                            metrics_data.append(
                                ["F1-Score", f"{results['f1_mean']:.3f} ± {results.get('f1_std', 0):.3f}"]
                            )

                        if metrics_data:
                            metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])
                            st.dataframe(metrics_df, width="stretch", hide_index=True)

                        # Add training vs test comparison
                        if "test_accuracy" in results and "train_accuracy" in results:
                            train_acc = results["train_accuracy"]
                            test_acc = results["test_accuracy"]
                            gap = train_acc - test_acc

                            st.metric(
                                "Overfitting Gap",
                                f"{gap:.3f}",
                                delta=f"{'Good' if gap < 0.05 else 'Warning' if gap < 0.1 else 'Poor'}",
                            )

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
                                n_clusters = results.get("n_clusters", 0)
                                silhouette = results.get("silhouette", 0)
                                davies_bouldin = results.get("davies_bouldin", 0)

                                # Get cluster sizes
                                labels = results.get("labels", [])
                                if len(labels) > 0:
                                    unique, counts = np.unique(labels, return_counts=True)
                                    cluster_sizes = "\n".join(
                                        [
                                            f"- Cluster {i}: {count} samples ({count/len(labels)*100:.1f}%)"
                                            for i, count in zip(unique, counts)
                                        ]
                                    )
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
                    if "cluster_quality" in insights:
                        st.info(f"**📊 Cluster Quality:** {insights['cluster_quality']}")

                    if "model_specific_insights" in insights:
                        st.success(f"**� {selected_model} Insights:** {insights['model_specific_insights']}")

                    if "balance_assessment" in insights:
                        st.info(f"**⚖️ Balance Assessment:** {insights['balance_assessment']}")

                    if "actionable_insights" in insights:
                        st.warning("**→ Actionable Insights:**")
                        if isinstance(insights["actionable_insights"], list):
                            for insight in insights["actionable_insights"]:
                                st.markdown(f"- {insight}")
                        else:
                            st.markdown(insights["actionable_insights"])

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
                                # Get model-specific metrics. Standard-mode results have a
                                # train/test split; Professional (Optuna) mode doesn't store one,
                                # so fall back to its tuned CV score instead of silently reporting 0%.
                                model_results = st.session_state.results.get(selected_model, {})
                                test_acc = model_results.get("test_accuracy")
                                train_acc = model_results.get("train_accuracy")
                                if test_acc is None or train_acc is None:
                                    prof_models = st.session_state.get("professional_results", {}).get(
                                        "individual_models", {}
                                    )
                                    score = prof_models.get(selected_model, {}).get("best_score", 0)
                                    test_acc = score if test_acc is None else test_acc
                                    train_acc = score if train_acc is None else train_acc
                                gap = abs(train_acc - test_acc)

                                # Get feature importance
                                if hasattr(model, "feature_importances_"):
                                    importances = model.feature_importances_
                                elif hasattr(model, "coef_"):
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
                                            model, X_perm, y_perm, n_repeats=3, random_state=42, n_jobs=1
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
                    if "model_characteristics" in insights:
                        st.success(f"**� {selected_model} Characteristics:** {insights['model_characteristics']}")

                    if "performance_insights" in insights:
                        st.info(f"**📊 Performance Analysis:** {insights['performance_insights']}")

                    if "feature_advice" in insights:
                        st.warning("**→ Model-Specific Advice:**")
                        if isinstance(insights["feature_advice"], list):
                            for advice in insights["feature_advice"]:
                                st.markdown(f"- {advice}")
                        else:
                            st.markdown(insights["feature_advice"])

                    if "model_advice" in insights:
                        st.warning("**→ Model Recommendations:**")
                        if isinstance(insights["model_advice"], list):
                            for advice in insights["model_advice"]:
                                st.markdown(f"- {advice}")
                        else:
                            st.markdown(insights["model_advice"])

                    if "error" in insights:
                        st.info(insights["error"])

            # Check if dataset is too large for SHAP
            n_features = X.shape[1]
            explanations = {}
            explainer = None  # Initialize explainer

            if n_features > 1000:
                st.warning(
                    f"⚠️ Dataset has {n_features:,} features. SHAP explanations may be slow or fail due to memory constraints."
                )
                st.info("💡 Tip: SHAP works best with < 1000 features. Consider feature selection for large datasets.")

                # Ask user if they want to proceed
                if not st.checkbox(
                    "Attempt SHAP anyway (may cause memory errors)", value=False, key=f"shap_checkbox_{selected_model}"
                ):
                    st.info("Showing only native model explanations (feature importance, coefficients).")
                    # Skip SHAP, only show native explanations
                    explainer = ModelExplainer()

                    # Get native feature importance
                    if hasattr(model, "feature_importances_"):
                        explanations["feature_importance"] = {
                            name: float(val) for name, val in zip(feature_names, model.feature_importances_)
                        }
                    elif hasattr(model, "coef_"):
                        coef = model.coef_
                        if coef.ndim > 1:
                            coef = np.abs(coef).mean(axis=0)
                        else:
                            coef = np.abs(coef)
                        explanations["coef_importance"] = {name: float(val) for name, val in zip(feature_names, coef)}
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
                                    model, X_perm, y_perm, n_repeats=5, random_state=42, n_jobs=1
                                )

                                explanations["permutation_importance"] = {
                                    name: float(val) for name, val in zip(feature_names, perm_result.importances_mean)
                                }
                                explanations["permutation_std"] = {
                                    name: float(val) for name, val in zip(feature_names, perm_result.importances_std)
                                }
                        except Exception as perm_error:
                            st.warning(f"Permutation importance failed: {str(perm_error)}")
                            explanations["permutation_error"] = str(perm_error)
                else:
                    # User chose to attempt SHAP - check cache first
                    if use_cached and "shap" in st.session_state.explainability_cache[cache_key]:
                        explanations = st.session_state.explainability_cache[cache_key]["shap"]
                        # Don't show cache message - silent caching is better UX
                    else:
                        with st.spinner(
                            f"Generating SHAP explanations for {selected_model} (this may take a while)..."
                        ):
                            try:
                                explainer = ModelExplainer()
                                explanations = explainer.explain_model(
                                    model, X, feature_names, sample_size=20  # Use minimal samples
                                )
                                # Cache SHAP results
                                if cache_key not in st.session_state.explainability_cache:
                                    st.session_state.explainability_cache[cache_key] = {}
                                st.session_state.explainability_cache[cache_key]["shap"] = explanations
                            except Exception as e:
                                st.error(f"SHAP failed: {e}")
                                explanations = {}
            else:
                # Normal size dataset - check cache first
                if use_cached and "shap" in st.session_state.explainability_cache[cache_key]:
                    explanations = st.session_state.explainability_cache[cache_key]["shap"]
                    # Don't show cache message - silent caching is better UX
                else:
                    with st.spinner(f"Generating explanations for {selected_model}..."):
                        try:
                            explainer = ModelExplainer()
                            explanations = explainer.explain_model(model, X, feature_names, sample_size=50)
                            # Cache SHAP results
                            if cache_key not in st.session_state.explainability_cache:
                                st.session_state.explainability_cache[cache_key] = {}
                            st.session_state.explainability_cache[cache_key]["shap"] = explanations
                        except Exception as e:
                            st.error(f"Error generating explanations: {e}")
                            explanations = {}

            # Display explanations (common for both paths)
            if explanations:
                # Check for SHAP errors
                if "shap_error" in explanations:
                    st.warning(f"SHAP explanation failed: {explanations['shap_error']}")
                    if "shap_traceback" in explanations:
                        with st.expander("Show error details"):
                            st.code(explanations["shap_traceback"])
                    st.info("Showing alternative explanation methods below...")

                # Feature importance
                if "shap_importance" in explanations:
                    st.subheader("SHAP Feature Importance")
                    visualizer = Visualizer()
                    fig = visualizer.plot_feature_importance(
                        explanations["shap_importance"], top_n=15, title="Top 15 Features by SHAP Importance"
                    )
                    st.plotly_chart(fig, width="stretch")

                elif "feature_importance" in explanations:
                    st.subheader("Feature Importance")
                    visualizer = Visualizer()
                    fig = visualizer.plot_feature_importance(explanations["feature_importance"], top_n=15)
                    st.plotly_chart(fig, width="stretch")

                elif "coef_importance" in explanations:
                    st.subheader("Coefficient Importance")
                    visualizer = Visualizer()
                    fig = visualizer.plot_feature_importance(explanations["coef_importance"], top_n=15)
                    st.plotly_chart(fig, width="stretch")

                elif "permutation_importance" in explanations:
                    st.subheader("Permutation Feature Importance")
                    st.info(
                        "💡 Permutation importance shows how much model performance decreases when each feature is randomly shuffled"
                    )
                    visualizer = Visualizer()
                    fig = visualizer.plot_feature_importance(
                        explanations["permutation_importance"],
                        top_n=15,
                        title="Top 15 Features by Permutation Importance",
                    )
                    st.plotly_chart(fig, width="stretch")

                # Top features list
                if explainer:
                    st.subheader("Top Important Features")
                    top_features = explainer.get_top_features(explanations, top_n=10)
                    if top_features is not None and len(top_features) > 0:
                        df_features = pd.DataFrame(top_features, columns=["Feature", "Importance"])
                        st.dataframe(df_features, width="stretch")
