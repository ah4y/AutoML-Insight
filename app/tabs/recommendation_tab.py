"""
Recommendation Tab - Display meta-learning recommendations.
"""

import logging

import numpy as np
import pandas as pd
import streamlit as st

from core.meta_selector import MetaModelSelector

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class RecommendationTab(BaseTab):
    """Tab for displaying model recommendations."""

    def render(self) -> None:
        """Render the recommendation tab."""
        st.subheader("🎯 AI-Powered Model Recommendations")

        # Check for both standard and professional results
        has_results = st.session_state.results or st.session_state.get("professional_results")

        if not has_results:
            st.info("Run AutoML to see recommendations")
            return

        # For clustering tasks, show professional results summary
        if st.session_state.task_type == "Clustering":
            professional_results = st.session_state.get("professional_results")
            if professional_results and "individual_models" in professional_results:
                st.markdown("### 🎯 Clustering Model Recommendations")

                # Get best model
                best_model = None
                best_score = -float("inf")
                for name, result in professional_results["individual_models"].items():
                    if result.get("best_score", -1000) > best_score:
                        best_score = result.get("best_score", -1000)
                        best_model = (name, result)

                if best_model:
                    st.success(f"✅ **Recommended Model:** {best_model[0]}")
                    st.metric("Silhouette Score", f"{best_score:.4f}")

                    st.markdown("#### ⚙️ Best Parameters")
                    params_df = pd.DataFrame(
                        [{"Parameter": k, "Value": str(v)} for k, v in best_model[1].get("best_params", {}).items()]
                    )
                    st.dataframe(params_df, width="stretch")

                    st.markdown("#### 📊 All Models Comparison")
                    comparison_data = []
                    for name, result in professional_results["individual_models"].items():
                        comparison_data.append(
                            {
                                "Model": name,
                                "Score": f"{result.get('best_score', 0):.4f}",
                                "Improvement": f"{result.get('improvement_percent', 0):+.1f}%",
                            }
                        )
                    st.dataframe(pd.DataFrame(comparison_data), width="stretch")
            else:
                st.info("Run Professional AutoML to see clustering recommendations")
            return

        if st.session_state.task_type == "Classification":
            # Check if we have any results to base recommendations on
            if not st.session_state.get("results") and not st.session_state.get("professional_results"):
                st.info("Recommendations will be generated after successful model training")
                return

            # Try to get recommendation, or generate one if missing
            recommendation = st.session_state.get("recommendation")

            # If no recommendation but we have results, generate it
            if not recommendation and st.session_state.get("results"):
                try:
                    meta_selector = MetaModelSelector()
                    recommendation = meta_selector.get_recommendation_with_rationale(
                        st.session_state.get("profile"), st.session_state.results
                    )
                    st.session_state.recommendation = recommendation
                except Exception as e:
                    st.error(f"Failed to generate recommendation: {e}")
                    return

            # For professional results without standard recommendation
            if not recommendation and st.session_state.get("professional_results"):
                st.markdown("### 🎯 Professional AutoML Recommendations")
                self._render_professional_recommendations()
                return

            # Check if recommendation has required fields
            if not recommendation or "recommended_model" not in recommendation:
                st.warning("No recommendation available. Model training may have failed.")
                return

            # Create comprehensive recommendation dashboard
            col1, col2, col3 = st.columns([1, 1, 1])

            # Model Performance Comparison Chart
            with col1:
                st.markdown("#### 📊 Model Performance Comparison")
                try:
                    evaluator = st.session_state.evaluator
                    leaderboard = evaluator.get_leaderboard("accuracy")

                    if leaderboard:
                        models = [item["model"] for item in leaderboard[:5]]
                        scores = [item.get("score", item.get("accuracy", 0)) for item in leaderboard[:5]]

                        # Create performance comparison chart
                        import plotly.express as px
                        import plotly.graph_objects as go

                        fig = go.Figure(
                            data=[
                                go.Bar(
                                    x=models,
                                    y=scores,
                                    marker_color=[
                                        "gold" if m == recommendation["recommended_model"] else "lightblue"
                                        for m in models
                                    ],
                                    text=[f"{s:.3f}" for s in scores],
                                    textposition="auto",
                                )
                            ]
                        )
                        fig.update_layout(
                            title="Model Accuracy Comparison", xaxis_title="Models", yaxis_title="Accuracy", height=350
                        )
                        st.plotly_chart(fig, width="stretch")
                except Exception as e:
                    st.warning(f"Could not generate performance chart: {e}")

            # Recommended Model Details
            with col2:
                st.markdown("#### 🎆 Recommended Model Details")
                recommended_model = recommendation["recommended_model"]
                score = recommendation.get("score", 0)

                st.metric("Best Model", recommended_model)
                st.metric("Accuracy", f"{score:.3f}")

                # Model characteristics
                model_info = {
                    "LogisticRegression": {"Type": "Linear", "Speed": "Fast", "Interpretability": "High"},
                    "RandomForest": {"Type": "Ensemble", "Speed": "Medium", "Interpretability": "Medium"},
                    "XGBoost": {"Type": "Boosting", "Speed": "Fast", "Interpretability": "Low"},
                    "LinearSVM": {"Type": "Linear SVM", "Speed": "Medium", "Interpretability": "Medium"},
                    "RBF-SVM": {"Type": "Non-linear SVM", "Speed": "Slow", "Interpretability": "Low"},
                    "KNN": {"Type": "Instance-based", "Speed": "Slow", "Interpretability": "Medium"},
                    "MLP": {"Type": "Neural Network", "Speed": "Medium", "Interpretability": "Low"},
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
                        "LogisticRegression": {"Accuracy": 4, "Speed": 5, "Interpretability": 5, "Complexity": 2},
                        "RandomForest": {"Accuracy": 5, "Speed": 3, "Interpretability": 3, "Complexity": 3},
                        "XGBoost": {"Accuracy": 5, "Speed": 4, "Interpretability": 2, "Complexity": 4},
                        "LinearSVM": {"Accuracy": 4, "Speed": 3, "Interpretability": 3, "Complexity": 3},
                        "RBF-SVM": {"Accuracy": 4, "Speed": 2, "Interpretability": 2, "Complexity": 4},
                        "KNN": {"Accuracy": 3, "Speed": 2, "Interpretability": 4, "Complexity": 1},
                        "MLP": {"Accuracy": 4, "Speed": 3, "Interpretability": 1, "Complexity": 5},
                    }

                    if recommended_model in model_aspects:
                        aspects = model_aspects[recommended_model]

                        import plotly.graph_objects as go

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatterpolar(
                                r=list(aspects.values()),
                                theta=list(aspects.keys()),
                                fill="toself",
                                name=recommended_model,
                                line_color="rgb(106, 81, 163)",
                            )
                        )

                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                            height=300,
                            title="Model Characteristics",
                        )
                        st.plotly_chart(fig, width="stretch")

                except Exception as e:
                    st.warning(f"Could not generate trade-offs chart: {e}")

            # AI-Powered Final Recommendation
            if st.session_state.ai_engine and st.session_state.ai_engine is not False:
                with st.expander("🤖 AI Final Recommendations", expanded=True):
                    with st.spinner("🤖 AI is creating your final recommendations..."):
                        try:
                            recommended_model = recommendation["recommended_model"]
                            score = recommendation.get("score", 0)

                            # Get all model scores with error handling
                            evaluator = st.session_state.evaluator
                            leaderboard = evaluator.get_leaderboard("accuracy")

                            model_lines = []
                            for m in leaderboard[:5]:
                                model_name = m.get("model", "Unknown")
                                acc = (
                                    m.get("accuracy")
                                    or m.get("score")
                                    or st.session_state.results.get(model_name, {}).get("accuracy_mean", 0)
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

                            if "deployment_readiness" in insights:
                                st.success(f"**🚀 Deployment Readiness:** {insights['deployment_readiness']}")

                            if "risk_assessment" in insights:
                                st.warning(f"**⚠ Risk Assessment:** {insights['risk_assessment']}")

                            if "next_steps" in insights:
                                st.info("**→ Next Steps:**")
                                if isinstance(insights["next_steps"], list):
                                    for step in insights["next_steps"]:
                                        st.markdown(f"- {step}")
                                else:
                                    st.markdown(insights["next_steps"])

                            if "monitoring_advice" in insights:
                                st.info(f"**📊 Monitoring Advice:** {insights['monitoring_advice']}")

                        except Exception as e:
                            logger.warning(f"Failed to generate AI recommendations: {e}")
                            st.error(f"AI recommendations failed: {str(e)}")

            # Recommended model
            st.success(f"### Recommended Model: **{recommendation['recommended_model']}**")

            # Add explanation of selection logic
            evaluator = st.session_state.evaluator
            leaderboard = evaluator.get_leaderboard("accuracy", penalize_overfitting=True)
            winner = leaderboard[0]

            # Show why this model won
            if winner.get("overfitting_gap", 0) < 0.10:
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
            for idx, reason in enumerate(recommendation["rationale"], 1):
                st.write(f"{idx}. {reason}")

            # Alternatives
            if recommendation.get("alternatives"):
                st.subheader("Alternative Models")
                alt_data = []
                for alt in recommendation["alternatives"]:
                    alt_data.append({"Model": alt["model"], "Score": f"{alt['score']:.4f}"})
                st.dataframe(pd.DataFrame(alt_data), width="stretch")
        else:
            # Clustering recommendation
            if not st.session_state.results:
                st.info("Run AutoML to see recommendations")
                return

            evaluator = st.session_state.evaluator
            leaderboard = evaluator.get_leaderboard("silhouette")

            if leaderboard:
                best = leaderboard[0]
                best_results = st.session_state.results[best["model"]]

                # AI-Powered Clustering Recommendation
                if st.session_state.ai_engine and st.session_state.ai_engine is not False:
                    with st.expander("🤖 AI Clustering Recommendations", expanded=True):
                        with st.spinner("🤖 AI is creating deployment recommendations..."):
                            try:
                                # Get all clustering results
                                all_models = "\n".join(
                                    [
                                        f"- {m['model']}: Silhouette={st.session_state.results[m['model']].get('silhouette', 0):.4f}, "
                                        f"Clusters={st.session_state.results[m['model']].get('n_clusters', 0)}"
                                        for m in leaderboard
                                    ]
                                )

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

                                if "deployment_assessment" in insights:
                                    st.success(f"**🚀 Deployment Assessment:** {insights['deployment_assessment']}")

                                if "use_cases" in insights:
                                    st.info("**💡 Potential Use Cases:**")
                                    if isinstance(insights["use_cases"], list):
                                        for case in insights["use_cases"]:
                                            st.markdown(f"- {case}")
                                    else:
                                        st.markdown(insights["use_cases"])

                                if "validation_steps" in insights:
                                    st.info("**✓ Validation Steps:**")
                                    if isinstance(insights["validation_steps"], list):
                                        for step in insights["validation_steps"]:
                                            st.markdown(f"- {step}")
                                    else:
                                        st.markdown(insights["validation_steps"])

                                if "monitoring_strategy" in insights:
                                    st.info(f"**📊 Monitoring Strategy:** {insights['monitoring_strategy']}")

                                if "limitations" in insights:
                                    st.warning(f"**⚠️ Limitations:** {insights['limitations']}")

                            except Exception as e:
                                error_str = str(e)
                                logger.warning(f"Failed to generate AI clustering recommendations: {error_str}")

                                # Check if it's a rate limit error
                                if "rate_limit" in error_str.lower() or "429" in error_str:
                                    st.warning(
                                        "⚠️ **AI Analysis Unavailable**: Groq API rate limit reached (100K tokens/day used). Try again later or upgrade at https://console.groq.com/settings/billing"
                                    )

                                    # Provide rule-based fallback insights
                                    st.info("**📊 Basic Clustering Assessment (Rule-Based):**")

                                    silhouette = best_results.get("silhouette", 0)
                                    n_clusters = best_results.get("n_clusters", 0)

                                    if silhouette > 0.5:
                                        st.success(
                                            f"✅ **Strong Clustering**: Silhouette score {silhouette:.3f} indicates well-separated, distinct clusters"
                                        )
                                    elif silhouette > 0.25:
                                        st.info(
                                            f"ℹ️ **Moderate Clustering**: Silhouette score {silhouette:.3f} shows reasonable cluster structure"
                                        )
                                    else:
                                        st.warning(
                                            f"⚠️ **Weak Clustering**: Silhouette score {silhouette:.3f} suggests overlapping or poorly defined clusters"
                                        )

                                    st.write("**Recommended Actions:**")
                                    st.write("1. Visualize clusters in 2D/3D using PCA/t-SNE")
                                    st.write("2. Analyze cluster centroids and member characteristics")
                                    st.write(f"3. Try different numbers of clusters (current: {n_clusters})")
                                    st.write("4. Consider feature engineering or dimensionality reduction")
                                else:
                                    st.error(f"AI analysis failed: {error_str[:150]}")

                st.success(f"### Recommended Model: **{best['model']}**")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Silhouette Score", f"{best['score']:.4f}")
                with col2:
                    st.metric("Number of Clusters", best["n_clusters"])

                st.subheader("Why This Model?")
                st.write("1. Highest silhouette score among all methods")
                st.write(f"2. Identified {best['n_clusters']} distinct clusters")
                st.write("3. Best balance between cohesion and separation")

                # ENHANCED: Professional Clustering Visualizations
                st.markdown("---")
                st.subheader("📊 Clustering Visualizations")

                try:
                    import plotly.express as px
                    import plotly.graph_objects as go
                    from sklearn.decomposition import PCA
                    from sklearn.metrics import silhouette_samples

                    # Get the best model and its labels
                    # Try multiple sources: models dict, results['model'], or results['labels']
                    best_model_name = best["model"]

                    # Get cluster labels from results (guaranteed to exist)
                    labels = st.session_state.results[best_model_name].get("labels")

                    # Try to get model for potential re-prediction
                    best_model = None
                    if best_model_name in st.session_state.get("models", {}):
                        best_model = st.session_state.models[best_model_name]
                    elif "model" in st.session_state.results[best_model_name]:
                        best_model = st.session_state.results[best_model_name]["model"]

                    X = st.session_state.X_processed

                    # If labels not in results, try to get from model
                    if labels is None and best_model is not None:
                        if hasattr(best_model, "labels_"):
                            labels = best_model.labels_
                        elif hasattr(best_model, "predict"):
                            labels = best_model.predict(X)

                    if labels is None:
                        st.warning("⚠️ Cluster labels not available for visualization")
                        labels = None

                    if labels is not None:
                        # Tab layout for different visualizations
                        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(
                            [
                                "🎯 2D Projection",
                                "📊 Cluster Distribution",
                                "📏 Silhouette Analysis",
                                "🔍 Cluster Profiles",
                            ]
                        )

                        with viz_tab1:
                            st.markdown("**PCA 2D Projection of Clusters**")

                            # Handle missing values before PCA
                            from sklearn.impute import SimpleImputer

                            X_for_viz = X.copy()
                            if pd.DataFrame(X_for_viz).isnull().any().any():
                                imputer = SimpleImputer(strategy="median")
                                X_for_viz = imputer.fit_transform(X_for_viz)

                            # Perform PCA for 2D visualization
                            pca = PCA(n_components=2, random_state=42)
                            X_pca = pca.fit_transform(X_for_viz)

                            # Create DataFrame for plotting
                            plot_df = pd.DataFrame(
                                {
                                    "PC1": X_pca[:, 0],
                                    "PC2": X_pca[:, 1],
                                    "Cluster": [f"Cluster {int(lbl)}" for lbl in labels],
                                }
                            )

                            # Create interactive scatter plot
                            fig = px.scatter(
                                plot_df,
                                x="PC1",
                                y="PC2",
                                color="Cluster",
                                title=f'{best["model"]} - {best["n_clusters"]} Clusters (PCA Projection)',
                                labels={
                                    "PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
                                    "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
                                },
                                color_discrete_sequence=px.colors.qualitative.Set2,
                                height=500,
                            )

                            fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="white")))
                            fig.update_layout(
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                font=dict(size=12),
                                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                            )

                            st.plotly_chart(fig, width="stretch")

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
                            cluster_df = pd.DataFrame(
                                {
                                    "Cluster": [f"Cluster {int(lbl)}" for lbl in unique_labels],
                                    "Size": counts,
                                    "Percentage": counts / len(labels) * 100,
                                }
                            )

                            # Create bar chart
                            fig = go.Figure()

                            fig.add_trace(
                                go.Bar(
                                    x=cluster_df["Cluster"],
                                    y=cluster_df["Size"],
                                    text=[
                                        f"{s} ({p:.1f}%)" for s, p in zip(cluster_df["Size"], cluster_df["Percentage"])
                                    ],
                                    textposition="outside",
                                    marker=dict(color=cluster_df["Size"], colorscale="Viridis", showscale=False),
                                )
                            )

                            fig.update_layout(
                                title=f"Cluster Size Distribution ({len(labels):,} total samples)",
                                xaxis_title="Cluster",
                                yaxis_title="Number of Samples",
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                height=400,
                                showlegend=False,
                            )

                            st.plotly_chart(fig, width="stretch")

                            # Show table
                            st.dataframe(cluster_df, width="stretch")

                            # Balance assessment
                            max_size = cluster_df["Size"].max()
                            min_size = cluster_df["Size"].min()
                            imbalance_ratio = max_size / min_size if min_size > 0 else float("inf")

                            if imbalance_ratio < 2:
                                st.success(
                                    f"✅ **Well-Balanced Clusters**: Largest/smallest ratio is {imbalance_ratio:.1f}x"
                                )
                            elif imbalance_ratio < 5:
                                st.info(f"ℹ️ **Moderate Imbalance**: Largest/smallest ratio is {imbalance_ratio:.1f}x")
                            else:
                                st.warning(
                                    f"⚠️ **Imbalanced Clusters**: Largest/smallest ratio is {imbalance_ratio:.1f}x - consider different number of clusters"
                                )

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

                                silhouette_data.append(
                                    {
                                        "cluster": int(i),
                                        "y_lower": y_lower,
                                        "y_upper": y_upper,
                                        "values": cluster_silhouette_vals,
                                        "avg": cluster_silhouette_vals.mean(),
                                    }
                                )

                                y_lower = y_upper + 10

                            # Create silhouette plot
                            fig = go.Figure()

                            colors = px.colors.qualitative.Set2
                            for idx, cluster_data in enumerate(silhouette_data):
                                y_vals = np.arange(cluster_data["y_lower"], cluster_data["y_upper"])

                                fig.add_trace(
                                    go.Scatter(
                                        x=cluster_data["values"],
                                        y=y_vals,
                                        mode="lines",
                                        fill="tozerox",
                                        name=f"Cluster {cluster_data['cluster']}",
                                        line=dict(color=colors[idx % len(colors)], width=0.5),
                                        fillcolor=colors[idx % len(colors)],
                                        hovertemplate=f"Cluster {cluster_data['cluster']}<br>Silhouette: %{{x:.3f}}<extra></extra>",
                                    )
                                )

                            # Add average silhouette score line
                            avg_silhouette = silhouette_vals.mean()
                            fig.add_vline(
                                x=avg_silhouette,
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"Avg: {avg_silhouette:.3f}",
                                annotation_position="top",
                            )

                            fig.update_layout(
                                title="Silhouette Plot for Each Cluster",
                                xaxis_title="Silhouette Coefficient",
                                yaxis_title="Cluster",
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                height=500,
                                showlegend=True,
                                yaxis=dict(showticklabels=False),
                            )

                            st.plotly_chart(fig, width="stretch")

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
                            cluster_metrics = pd.DataFrame(
                                [
                                    {
                                        "Cluster": f"Cluster {d['cluster']}",
                                        "Size": len(d["values"]),
                                        "Avg Silhouette": f"{d['avg']:.3f}",
                                        "Quality": (
                                            "✅ Excellent"
                                            if d["avg"] > 0.5
                                            else ("🟢 Good" if d["avg"] > 0.25 else "⚠️ Weak")
                                        ),
                                    }
                                    for d in silhouette_data
                                ]
                            )

                            st.dataframe(cluster_metrics, width="stretch")

                        with viz_tab4:
                            st.markdown("**Cluster Characteristics Profile**")

                            # Get original data with cluster labels
                            df_with_clusters = st.session_state.data.copy()
                            df_with_clusters["Cluster"] = [f"Cluster {int(lbl)}" for lbl in labels]

                            # Calculate cluster statistics for numeric columns
                            numeric_cols = df_with_clusters.select_dtypes(include=[np.number]).columns.tolist()
                            if "Cluster" in numeric_cols:
                                numeric_cols.remove("Cluster")

                            if numeric_cols:
                                # Show top 5 most important features
                                feature_cols = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols

                                st.write(f"**Cluster Centroids** (showing top {len(feature_cols)} features):")

                                # Calculate means per cluster
                                cluster_profiles = []
                                for cluster_name in sorted(df_with_clusters["Cluster"].unique()):
                                    cluster_data = df_with_clusters[df_with_clusters["Cluster"] == cluster_name]
                                    profile = {"Cluster": cluster_name, "Size": len(cluster_data)}

                                    for col in feature_cols:
                                        if col in cluster_data.columns:
                                            profile[col] = cluster_data[col].mean()

                                    cluster_profiles.append(profile)

                                profile_df = pd.DataFrame(cluster_profiles)

                                # Display as styled dataframe
                                st.dataframe(
                                    profile_df.style.background_gradient(cmap="RdYlGn", subset=feature_cols),
                                    width="stretch",
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
                                        fig.add_trace(
                                            go.Scatterpolar(
                                                r=normalized_values[idx],
                                                theta=feature_cols,
                                                fill="toself",
                                                name=cluster["Cluster"],
                                            )
                                        )

                                    fig.update_layout(
                                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                        showlegend=True,
                                        title="Cluster Feature Profiles (Normalized)",
                                        height=500,
                                    )

                                    st.plotly_chart(fig, width="stretch")

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

    def _render_professional_recommendations(self):
        """Render recommendations for professional AutoML results."""
        professional_results = st.session_state.get("professional_results")

        if not professional_results or "individual_models" not in professional_results:
            st.warning("No professional results available")
            return

        # Get best model
        best_model_name = None
        best_score = -float("inf")

        for name, result in professional_results["individual_models"].items():
            score = result.get("best_score", -1000)
            if score > best_score:
                best_score = score
                best_model_name = name

        if not best_model_name or best_score <= -1000:
            st.error("All models failed during training. Cannot generate recommendations.")
            return

        # Display recommendations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🏆 Recommended Model")
            st.success(f"**{best_model_name}**")
            st.metric("Best Score", f"{best_score:.4f}")

            best_result = professional_results["individual_models"][best_model_name]
            st.metric("Improvement", f"{best_result.get('improvement_percent', 0):+.1f}%")
            st.metric("Training Time", f"{best_result.get('optimization_time', 0):.1f}s")

        with col2:
            st.markdown("### 📊 Why This Model?")

            reasons = []
            if best_score > 0.9:
                reasons.append("✅ Excellent performance (>90% accuracy)")
            elif best_score > 0.8:
                reasons.append("✅ Good performance (>80% accuracy)")

            if best_result.get("improvement_percent", 0) > 5:
                reasons.append(
                    f"✅ Significant optimization improvement ({best_result.get('improvement_percent', 0):.1f}%)"
                )

            if best_result.get("optimization_time", 0) < 60:
                reasons.append("⚡ Fast training (<1 minute)")

            if reasons:
                for reason in reasons:
                    st.write(reason)
            else:
                st.write("This model showed the best performance among all tested models.")

        st.markdown("### 📈 All Models Performance")

        comparison_data = []
        for name, result in professional_results["individual_models"].items():
            comparison_data.append(
                {
                    "Model": name,
                    "Score": result.get("best_score", -1000),
                    "Improvement": f"{result.get('improvement_percent', 0):+.1f}%",
                    "Training Time (s)": f"{result.get('optimization_time', 0):.1f}",
                    "Recommended": "🏆" if name == best_model_name else "",
                }
            )

        df = pd.DataFrame(comparison_data)
        df = df.sort_values("Score", ascending=False)
        st.dataframe(df, width="stretch")

        st.markdown("### 🚀 Deployment Recommendations")
        recommendations = [
            "**Model Export:** Save the trained model using joblib or pickle",
            "**API Development:** Create a REST API using Flask or FastAPI",
            "**Monitoring:** Implement prediction logging and performance tracking",
            "**Testing:** Validate on additional hold-out data before production",
        ]
        for rec in recommendations:
            st.write(f"• {rec}")
