"""
Professional AutoML Tab - Display advanced optimization results.
"""

import logging

import pandas as pd
import streamlit as st

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class ProfessionalAutoMLTab(BaseTab):
    """Tab for displaying professional AutoML results."""

    def render(self) -> None:
        """Render the professional AutoML tab."""
        if not self.has_professional_results():
            self.show_warning("Professional AutoML results not available. Run Professional AutoML first.")
            return

        st.markdown("### 🔥 **Professional AutoML Pipeline Results**")

        professional_results = st.session_state.get("professional_results")
        dataset_stats = st.session_state.get("dataset_stats")

        if not professional_results:
            st.warning("⚠️ No Professional AutoML results available.")
            return

        # Professional results overview
        st.markdown("---")
        self._display_professional_results(professional_results, st.session_state.get("advanced_features", []))

        # Additional professional insights if available
        if dataset_stats:
            st.markdown("---")
            st.markdown("### 📊 **Advanced Dataset Analysis**")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💾 Memory Usage", f"{dataset_stats.get('memory_usage_mb', 0):.1f} MB")
            with col2:
                st.metric("🔢 Numeric Features", dataset_stats.get("numeric_features", 0))
            with col3:
                st.metric("📝 Categorical Features", dataset_stats.get("categorical_features", 0))
            with col4:
                st.metric("❓ Missing Values", dataset_stats.get("missing_values", 0))

            # Data complexity metrics
            if "feature_correlation_avg" in dataset_stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔗 Avg Correlation", f"{dataset_stats['feature_correlation_avg']:.3f}")
                with col2:
                    st.metric("🕳️ Sparsity", f"{dataset_stats['sparsity']:.3f}")
                with col3:
                    if "class_balance" in dataset_stats:
                        st.metric("⚖️ Class Balance", f"{dataset_stats['class_balance']:.3f}")

    def _display_professional_results(self, results, advanced_features):
        """Display professional AutoML results with advanced insights."""
        st.header("🏆 Professional AutoML Results")

        # Check if all models failed
        individual_results = results.get("individual_models", {})
        if not individual_results:
            st.error("❌ No models were successfully trained!")
            st.warning("💡 Check the console/terminal for detailed error messages.")
            return

        # Check if all models returned failure scores (-1000)
        all_failed = all(result.get("best_score", 0) <= -1000 for result in individual_results.values())
        if all_failed:
            st.error("❌ All models failed during training!")
            st.warning("⚠️ All models returned failure scores (-1000). This typically indicates:")
            st.write("1. **Data issues**: NaN values, infinite values, or invalid data")
            st.write("2. **Target variable issues**: Incorrect target column or data type mismatch")
            st.write("3. **Insufficient data**: Not enough samples for cross-validation")
            st.info(
                "💡 **Solution**: Check the console/terminal for detailed error logs showing the exact failure reason."
            )

            # Show debug info
            with st.expander("🔍 Debug Information"):
                st.write("**Dataset Info:**")
                st.json(results.get("dataset_info", {}))
                st.write("**Model Results:**")
                for model_name, result in individual_results.items():
                    st.write(f"- {model_name}: Score = {result.get('best_score', 'N/A')}")

                # Show captured error logs
                if st.session_state.get("optimization_errors"):
                    st.write("**Error Logs:**")
                    st.code("\n".join(st.session_state.optimization_errors[-10:]), language="text")  # Last 10 errors
            return

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        individual_results = results["individual_models"]
        n_models = len(individual_results)

        # Calculate total improvement
        total_improvement = sum(r["improvement"] for r in individual_results.values())
        avg_improvement = total_improvement / n_models if n_models > 0 else 0

        # Get best model
        best_model_info = None
        best_score = -float("inf")

        for name, result in individual_results.items():
            if result["best_score"] > best_score:
                best_score = result["best_score"]
                best_model_info = (name, result)

        with col1:
            st.metric("🤖 Models Optimized", n_models)
        with col2:
            st.metric("📈 Avg Improvement", f"{avg_improvement:.4f}")
        with col3:
            st.metric("🏆 Best Score", f"{best_score:.4f}")
        with col4:
            total_time = sum(r["optimization_time"] for r in individual_results.values())
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
            best_params_df = pd.DataFrame(
                [{"Parameter": param, "Value": str(value)} for param, value in best_result["best_params"].items()]
            )
            st.dataframe(best_params_df, width="stretch")

        # Model Comparison Table
        st.subheader("📊 Model Performance Comparison")

        comparison_data = []
        for name, result in individual_results.items():
            comparison_data.append(
                {
                    "Model": name,
                    "Baseline Score": f"{result['baseline_score']:.4f}",
                    "Optimized Score": f"{result['best_score']:.4f}",
                    "Improvement": f"+{result['improvement']:.4f}",
                    "Improvement %": f"{result['improvement_percent']:+.1f}%",
                    "Time (s)": f"{result['optimization_time']:.1f}",
                    "Trials": result["n_trials"],
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, width="stretch")

        # Ensemble Results
        if results["ensemble_models"]:
            st.subheader("🎭 Ensemble Model Results")
            ensemble_result = results["ensemble_models"]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎭 Ensemble Score", f"{ensemble_result['ensemble_score']:.4f}")
                st.metric("🥇 Best Individual", f"{ensemble_result['best_individual_score']:.4f}")
            with col2:
                ensemble_improvement = ensemble_result["ensemble_improvement"]
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
        summary = results["optimization_summary"]
        st.text(summary)

        # Professional Insights
        self._show_professional_insights(results)

    def _show_professional_insights(self, results):
        """Show professional AI engineer insights and recommendations."""
        st.subheader("🧠 AI Engineer Insights")

        individual_results = results["individual_models"]
        dataset_info = results["dataset_info"]

        insights = []

        # Performance insights
        best_model = max(individual_results.items(), key=lambda x: x[1]["best_score"])
        worst_model = min(individual_results.items(), key=lambda x: x[1]["best_score"])

        score_gap = best_model[1]["best_score"] - worst_model[1]["best_score"]

        if score_gap > 0.1:
            insights.append(
                f"🎯 **Significant Model Differences**: {best_model[0]} outperforms {worst_model[0]} by {score_gap:.3f}. Consider ensemble approaches."
            )

        # Optimization insights
        high_improvement_models = [name for name, res in individual_results.items() if res["improvement_percent"] > 10]
        if high_improvement_models:
            insights.append(
                f"⚡ **High Optimization Impact**: {', '.join(high_improvement_models)} showed >10% improvement from hyperparameter tuning."
            )

        # Dataset-specific insights
        n_samples = dataset_info["n_samples"]
        n_features = dataset_info["n_features"]

        if n_samples < 1000:
            insights.append(
                "⚠️ **Small Dataset Warning**: Consider regularization, cross-validation, and simpler models to avoid overfitting."
            )
        elif n_samples > 100000:
            insights.append(
                "🚀 **Large Dataset Advantage**: Consider deep learning models and advanced ensemble methods for potential performance gains."
            )

        if n_features > n_samples:
            insights.append(
                "📐 **High-Dimensional Data**: Feature selection and dimensionality reduction are critical. Consider L1 regularization."
            )

        # Model-specific recommendations
        if "RandomForest" in individual_results and individual_results["RandomForest"]["best_score"] > 0.8:
            insights.append(
                "🌲 **Tree-Based Success**: Random Forest performs well. Consider XGBoost, LightGBM, or CatBoost for potential improvements."
            )

        if "SVM" in individual_results and individual_results["SVM"]["optimization_time"] > 60:
            insights.append(
                "⏱️ **SVM Performance**: SVM optimization took significant time. For larger datasets, consider faster alternatives."
            )

        # Ensemble insights
        if results["ensemble_models"] and results["ensemble_models"]["ensemble_improvement"] > 0:
            insights.append(
                "🎭 **Ensemble Success**: Ensemble models show improvement. Consider stacking with meta-learners for advanced performance."
            )

        # Display insights
        for i, insight in enumerate(insights, 1):
            st.markdown(f"{i}. {insight}")

        if not insights:
            st.info(
                "💡 All models performed within expected ranges. Consider feature engineering or domain-specific preprocessing for further improvements."
            )

        # Professional recommendations
        st.subheader("🔧 Next Steps - Professional ML Engineering")

        recommendations = [
            "**Feature Engineering**: Create domain-specific features, polynomial interactions, or time-based features",
            "**Advanced Validation**: Implement time-series aware splits, group-based validation, or stratified sampling",
            "**Model Calibration**: Apply Platt scaling or isotonic regression for probability calibration",
            "**Uncertainty Quantification**: Implement prediction intervals or confidence estimation",
            "**Production Optimization**: Consider model compression, quantization, or knowledge distillation",
            "**Monitoring Setup**: Implement drift detection, performance monitoring, and automated retraining",
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

    def render_insights(self) -> None:
        """Render professional ML insights and recommendations."""
        st.markdown("### 🎯 **Professional ML Insights**")

        professional_results = st.session_state.get("professional_results")

        if not professional_results:
            st.info("🎯 Professional insights will be displayed here after running Professional AutoML.")
            return

        # Check if we have individual model results
        if "individual_models" not in professional_results or not professional_results["individual_models"]:
            st.warning("⚠️ No model results available for insights.")
            return

        # Display AI-generated insights
        st.markdown("#### 🧠 **AI Engineer Insights**")

        # Get model scores for analysis
        model_scores = {}
        for name, result in professional_results["individual_models"].items():
            if "best_score" in result and result["best_score"] > -1000:
                model_scores[name] = result["best_score"]

        if model_scores:
            best_model = max(model_scores, key=model_scores.get)
            worst_model = min(model_scores, key=model_scores.get)
            score_diff = model_scores[best_model] - model_scores[worst_model]

            insights = []

            # Model performance insights
            if score_diff > 0.1:
                insights.append(
                    f"🎯 **Significant Model Differences:** {best_model} outperforms {worst_model} by {score_diff:.3f}. Consider ensemble approaches."
                )
            elif score_diff < 0.02:
                insights.append(
                    "⚖️ **Similar Performance:** Models perform similarly. Consider simplicity and inference speed."
                )

            # Dataset size insights
            dataset_info = professional_results.get("dataset_info", {})
            n_samples = dataset_info.get("n_samples", 0)
            n_features = dataset_info.get("n_features", 0)

            if n_samples < 1000:
                insights.append(
                    "⚠️ **Small Dataset Warning:** Consider regularization, cross-validation, and simpler models to avoid overfitting."
                )
            elif n_samples > 100000:
                insights.append(
                    "📈 **Large Dataset:** Consider using gradient boosting, deep learning, or distributed training approaches."
                )

            if n_features > n_samples:
                insights.append(
                    "⚠️ **High Dimensionality:** Features exceed samples. Apply feature selection or dimensionality reduction."
                )

            # Tree-based model insights
            tree_models = [name for name in model_scores.keys() if "Forest" in name or "XGB" in name or "LGBM" in name]
            if tree_models and any(model_scores[m] == max(model_scores.values()) for m in tree_models):
                insights.append(
                    "🌲 **Tree-Based Success:** Random Forest performs well. Consider XGBoost, LightGBM, or CatBoost for potential improvements."
                )

            for insight in insights:
                st.info(insight)
        else:
            st.warning("⚠️ No successful model scores available for insights.")

        # Next steps recommendations
        st.markdown("#### 🔧 **Next Steps - Professional ML Engineering**")

        recommendations = [
            "**Feature Engineering:** Create domain-specific features, polynomial interactions, or time-based features",
            "**Advanced Validation:** Implement time-series aware splits, group-based validation, or stratified sampling",
            "**Model Calibration:** Apply Platt scaling or isotonic regression for probability calibration",
            "**Uncertainty Quantification:** Implement prediction intervals or confidence estimation",
            "**Production Optimization:** Consider model compression, quantization, or knowledge distillation",
            "**Monitoring Setup:** Implement drift detection, performance monitoring, and automated retraining",
        ]

        for rec in recommendations:
            st.write(f"• {rec}")

        # Advanced techniques
        st.markdown("#### 🎓 **Advanced ML Engineering Techniques**")

        task_type = professional_results.get("dataset_info", {}).get("task_type", "classification")

        if task_type == "classification":
            st.markdown("""
**For Classification:**
- Multi-label classification with label powerset or binary relevance
- Cost-sensitive learning for imbalanced datasets
- Conformal prediction for uncertainty quantification
            """)
        elif task_type == "regression":
            st.markdown("""
**For Regression:**
- Multi-output regression with target transformations
- Quantile regression for risk assessment
- Bayesian optimization for expensive function approximation
            """)
        else:
            st.markdown("""
**For Clustering:**
- Hierarchical clustering with optimal number selection
- Density-based clustering for arbitrary shapes
- Semi-supervised clustering with constraints
            """)

        st.markdown("""
**Cross-Domain:**
- Transfer learning from pre-trained models
- Meta-learning for few-shot scenarios
- Automated machine learning (AutoML) pipelines
        """)

        # Add visualizations for insights
        st.markdown("#### 📈 Model Performance Visualizations")

        if model_scores:
            # Create performance comparison chart
            import plotly.graph_objects as go

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(model_scores.keys()),
                        y=list(model_scores.values()),
                        marker_color="lightblue",
                        text=[f"{score:.4f}" for score in model_scores.values()],
                        textposition="auto",
                    )
                ]
            )

            fig.update_layout(
                title="Model Performance Comparison", xaxis_title="Model", yaxis_title="Score", height=400
            )

            st.plotly_chart(fig, width="stretch")

            # Training time comparison if available
            training_times = {}
            for name, result in professional_results["individual_models"].items():
                if "optimization_time" in result:
                    training_times[name] = result["optimization_time"]

            if training_times:
                st.markdown("#### ⏱️ Training Time Analysis")

                fig2 = go.Figure(
                    data=[
                        go.Bar(
                            x=list(training_times.keys()),
                            y=list(training_times.values()),
                            marker_color="lightcoral",
                            text=[f"{t:.1f}s" for t in training_times.values()],
                            textposition="auto",
                        )
                    ]
                )

                fig2.update_layout(
                    title="Training Time Comparison", xaxis_title="Model", yaxis_title="Time (seconds)", height=400
                )

                st.plotly_chart(fig2, width="stretch")
