"""
Models Tab - Display standard model results (Classification/Clustering).
"""

import logging

import numpy as np
import pandas as pd
import streamlit as st
import umap

from core.background_analyzer import get_background_result_safe
from core.statistical_tests import StatisticalModelComparator
from core.visualize import Visualizer
from utils.visualization_helpers import (
    plot_bias_variance_comparison,
    plot_calibration_curve,
    plot_confidence_histogram,
    plot_cv_stability_boxplot,
    plot_learning_curve,
    plot_pairwise_pvalues_heatmap,
)

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class ModelsTab(BaseTab):
    """Tab for displaying standard model results."""

    def render(self) -> None:
        """Render the models tab."""
        if not self.require_results():
            return

        task_type = self.get_session_data("task_type", "Classification")

        if task_type == "Classification":
            self.render_classification_results()
        else:
            self.render_clustering_results()

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
                model_data.append(
                    {
                        "model": name,
                        "train_acc": result.get("train_accuracy", 0) * 100,
                        "test_acc": result.get("test_accuracy", 0) * 100,
                        "cv_mean": result.get("cv_accuracy_mean", 0) * 100,
                        "cv_std": result.get("cv_accuracy_std", 0) * 100,
                        "gap": abs(result.get("train_accuracy", 0) - result.get("test_accuracy", 0)) * 100,
                        "precision": result.get("precision_macro_mean", 0) * 100,
                        "recall": result.get("recall_macro_mean", 0) * 100,
                        "f1": result.get("f1_macro_mean", 0) * 100,
                    }
                )

            df_models = pd.DataFrame(model_data)

            # Create subplot with 2 rows, 2 columns
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Accuracy Comparison (Train vs Test)",
                    "Overfitting Analysis",
                    "Model Performance Metrics",
                    "Cross-Validation Stability",
                ),
                specs=[[{"type": "bar"}, {"type": "scatter"}], [{"type": "bar"}, {"type": "bar"}]],
                vertical_spacing=0.12,
                horizontal_spacing=0.10,
            )

            # Plot 1: Train vs Test Accuracy (Grouped Bar)
            fig.add_trace(
                go.Bar(
                    name="Test Acc (True)",
                    x=df_models["model"],
                    y=df_models["test_acc"],
                    marker_color="rgb(55, 83, 109)",
                    text=df_models["test_acc"].round(1),
                    textposition="outside",
                    texttemplate="%{text}%",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(
                    name="Train Acc",
                    x=df_models["model"],
                    y=df_models["train_acc"],
                    marker_color="rgb(26, 118, 255)",
                    text=df_models["train_acc"].round(1),
                    textposition="outside",
                    texttemplate="%{text}%",
                    opacity=0.6,
                ),
                row=1,
                col=1,
            )

            # Plot 2: Overfitting Gap Scatter
            colors = ["green" if g < 5 else "orange" if g < 10 else "red" for g in df_models["gap"]]
            fig.add_trace(
                go.Scatter(
                    x=df_models["test_acc"],
                    y=df_models["gap"],
                    mode="markers+text",
                    marker=dict(size=15, color=colors, opacity=0.8, line=dict(width=2, color="DarkSlateGrey")),
                    text=df_models["model"],
                    textposition="top center",
                    textfont=dict(size=9),
                    name="Models",
                    hovertemplate="<b>%{text}</b><br>Test Acc: %{x:.1f}%<br>Gap: %{y:.1f}%<extra></extra>",
                ),
                row=1,
                col=2,
            )

            # Add reference lines for gap thresholds
            fig.add_hline(y=5, line_dash="dash", line_color="green", opacity=0.5, row=1, col=2)
            fig.add_hline(y=10, line_dash="dash", line_color="orange", opacity=0.5, row=1, col=2)

            # Plot 3: Precision, Recall, F1 (Grouped Bar)
            fig.add_trace(
                go.Bar(
                    name="Precision", x=df_models["model"], y=df_models["precision"], marker_color="rgb(158, 202, 225)"
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Bar(name="Recall", x=df_models["model"], y=df_models["recall"], marker_color="rgb(107, 174, 214)"),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Bar(name="F1-Score", x=df_models["model"], y=df_models["f1"], marker_color="rgb(49, 130, 189)"),
                row=2,
                col=1,
            )

            # Plot 4: CV Mean with Error Bars
            fig.add_trace(
                go.Bar(
                    name="CV Accuracy",
                    x=df_models["model"],
                    y=df_models["cv_mean"],
                    error_y=dict(type="data", array=df_models["cv_std"], visible=True),
                    marker_color="rgb(204, 204, 204)",
                    text=df_models["cv_mean"].round(1),
                    textposition="outside",
                    texttemplate="%{text}%",
                ),
                row=2,
                col=2,
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
                hovermode="closest",
                template="plotly_white",
            )

            st.plotly_chart(fig, width="stretch")

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
            if "overfitting_warnings" in result:
                warnings = result["overfitting_warnings"]
                if warnings.get("has_issues"):
                    if warnings.get("overall_severity") == "HIGH":
                        high_severity_warnings.append((name, warnings))
                    elif warnings.get("overall_severity") in ["MEDIUM", "LOW"]:
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
                    st.markdown(warnings["summary"])

                    for warning in warnings["warnings"]:
                        if warning["severity"] == "HIGH":
                            st.markdown(f"**{warning['message']}**")
                            st.markdown("**What to do:**")
                            for rec in warning["recommendations"]:
                                st.markdown(f"- {rec}")

        # INFORMATIONAL WARNINGS (MEDIUM/LOW severity)
        if medium_low_warnings:
            with st.expander("ℹ️ Additional Performance Notes (Non-Critical)", expanded=False):
                st.info("These are informational observations that may help improve your models:")

                for model_name, warnings in medium_low_warnings:
                    st.markdown(f"**{model_name}**: {warnings['summary']}")

                    for warning in warnings["warnings"]:
                        if warning["severity"] in ["MEDIUM", "LOW"]:
                            st.markdown(f"- {warning['message']}")
                            if warning["severity"] == "MEDIUM":
                                st.markdown("  **Suggestions:**")
                                for rec in warning["recommendations"][:2]:  # Show top 2 recommendations
                                    st.markdown(f"    - {rec}")
                    st.write("---")

        # NEW: Train vs Test Performance Table
        st.markdown("### 📊 Model Performance (Training vs Testing)")

        leaderboard_data = []
        for name, result in st.session_state.results.items():
            train_acc = result.get("train_accuracy", 0)
            test_acc = result.get("test_accuracy", 0)
            gap = result.get("overfitting_gap", train_acc - test_acc)

            # Color code the gap
            gap_emoji = "🟢" if gap < 0.05 else "🟡" if gap < 0.10 else "🔴"

            leaderboard_data.append(
                {
                    "Model": name,
                    "Train Acc": f"{train_acc:.4f}",
                    "Test Acc": f"{test_acc:.4f}",
                    "Gap": f"{gap_emoji} {gap:.4f}",
                    "CV Mean": f"{result.get('cv_accuracy_mean', 0):.4f}",
                    "CV Std": f"{result.get('cv_accuracy_std', 0):.4f}",
                    "Status": "✅ Good" if gap < 0.10 else "⚠️ Overfit",
                }
            )

        df_leaderboard = pd.DataFrame(leaderboard_data)
        # Sort by Test Acc (the TRUE performance)
        df_leaderboard = df_leaderboard.sort_values("Test Acc", ascending=False)
        st.dataframe(df_leaderboard, width="stretch")

        # NEW: Display CV Strategy Report
        if leaderboard_data and "cv_strategy" in st.session_state.results[leaderboard_data[0]["Model"]]:
            first_result = st.session_state.results[leaderboard_data[0]["Model"]]
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
                        leaderboard = evaluator.get_leaderboard("accuracy")

                        if leaderboard and len(leaderboard) > 0:
                            best_model = leaderboard[0]
                            best_name = best_model.get("model", "Unknown")

                            # Try to get accuracy from different possible keys
                            best_accuracy = (
                                best_model.get("accuracy")
                                or best_model.get("score")
                                or st.session_state.results.get(best_name, {}).get("accuracy_mean", 0)
                            )

                            # Create performance context
                            n_classes = len(np.unique(st.session_state.y_processed))
                            n_samples = len(st.session_state.y_processed)

                            # Build model list with error handling
                            model_list = []
                            for m in leaderboard[:5]:
                                model_name = m.get("model", "Unknown")
                                acc = (
                                    m.get("accuracy")
                                    or m.get("score")
                                    or st.session_state.results.get(model_name, {}).get("accuracy_mean", 0)
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
                                if "_notice" in insights:
                                    st.info(insights["_notice"])

                            except Exception as e:
                                # If AI fails, show user-friendly message
                                error_msg = str(e)
                                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                                    st.warning(
                                        "⚠️ **AI Analysis Unavailable**: Groq API rate limit reached (100K tokens/day used). Try again later or upgrade your plan at https://console.groq.com/settings/billing"
                                    )
                                    st.info(
                                        "💡 **Tip**: Enable response caching to reduce API calls, or try again tomorrow when your quota resets."
                                    )
                                else:
                                    st.error(f"AI analysis failed: {error_msg}")

                                # Stop here if AI completely failed
                                insights = {}

                            if "performance_assessment" in insights:
                                st.info(f"**📊 Performance Assessment:** {insights['performance_assessment']}")

                            col1, col2 = st.columns(2)
                            with col1:
                                if "model_comparison" in insights:
                                    st.success(f"**🏆 Why {best_name} Won:** {insights['model_comparison']}")

                            with col2:
                                if "red_flags" in insights:
                                    st.warning(f"**⚠ Red Flags:** {insights['red_flags']}")

                            if "improvement_tips" in insights:
                                st.info("**→ Improvement Tips:**")
                                if isinstance(insights["improvement_tips"], list):
                                    for tip in insights["improvement_tips"]:
                                        st.markdown(f"- {tip}")
                                else:
                                    st.markdown(insights["improvement_tips"])
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
        leaderboard = evaluator.get_leaderboard("accuracy")

        if not leaderboard:
            st.warning("No leaderboard data available.")
            return

        visualizer = Visualizer()
        fig = visualizer.plot_leaderboard(leaderboard, "Accuracy")
        st.plotly_chart(fig, width="stretch")

        # Detailed metrics table
        st.subheader("Detailed Metrics")
        metrics_data = []
        for item in leaderboard:
            model_name = item["model"]
            results = st.session_state.results[model_name]
            metrics_data.append(
                {
                    "Model": model_name,
                    "Accuracy": f"{results.get('accuracy_mean', 0):.4f}",
                    "F1-Score": f"{results.get('f1_macro_mean', 0):.4f}",
                    "ROC-AUC": f"{results.get('roc_auc_ovr_mean', 0):.4f}",
                    "Log Loss": f"{results.get('log_loss_mean', 0):.4f}",
                }
            )

        st.dataframe(pd.DataFrame(metrics_data), width="stretch")

        # ROC Curves
        if len(np.unique(st.session_state.y_processed)) <= 10:  # Only for reasonable number of classes
            st.subheader("ROC Curves")
            try:
                models_data = st.session_state.results
                fig = visualizer.plot_roc_curves(
                    models_data,
                    st.session_state.X_processed,
                    st.session_state.y_processed,
                    len(np.unique(st.session_state.y_processed)),
                )
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.warning(f"Could not generate ROC curves: {e}")

        # Confusion Matrix for best model
        if leaderboard:
            st.subheader("Confusion Matrix (Best Model)")
            best_model_name = leaderboard[0]["model"]
            best_results = st.session_state.results[best_model_name]

            if "predictions" in best_results and "true_labels" in best_results:
                fig = visualizer.plot_confusion_matrix(best_results["true_labels"], best_results["predictions"])
                st.plotly_chart(fig, width="stretch")
        else:
            st.warning("No models successfully completed training. Please check the training logs.")

        # Advanced Evaluation Sections
        if st.session_state.results:
            self.render_advanced_evaluation_sections(st.session_state.results)

    def render_advanced_evaluation_sections(self, results_dict):
        """Render advanced evaluation sections with collapsible expanders.

        Args:
            results_dict: Dictionary mapping model names to their results
        """
        st.markdown("---")
        st.markdown("## 📊 Advanced Model Evaluation")

        # Section 1: Extended Performance Metrics
        with st.expander("📊 Extended Performance Metrics", expanded=False):
            st.markdown("### Comprehensive Metrics Beyond Accuracy")

            metrics_data = []
            for model_name, results in results_dict.items():
                row = {
                    "Model": model_name,
                    "Accuracy": results.get("test_accuracy", 0),
                    "Balanced_Acc": results.get("balanced_accuracy", None),
                    "MCC": results.get("matthews_corrcoef", None),
                    "Cohen_Kappa": results.get("cohen_kappa", None),
                    "Jaccard": results.get("jaccard_score", None),
                    "Hamming_Loss": results.get("hamming_loss", None),
                }
                metrics_data.append(row)

            df_metrics = pd.DataFrame(metrics_data)

            # Style the dataframe with color gradients
            styled_df = (
                df_metrics.style.background_gradient(
                    subset=["Accuracy", "Balanced_Acc", "MCC", "Cohen_Kappa", "Jaccard"], cmap="RdYlGn", vmin=0, vmax=1
                )
                .background_gradient(subset=["Hamming_Loss"], cmap="RdYlGn_r", vmin=0, vmax=1)
                .format(
                    {
                        "Accuracy": "{:.4f}",
                        "Balanced_Acc": "{:.4f}",
                        "MCC": "{:.4f}",
                        "Cohen_Kappa": "{:.4f}",
                        "Jaccard": "{:.4f}",
                        "Hamming_Loss": "{:.4f}",
                    }
                )
            )

            st.dataframe(styled_df, width="stretch")

            st.info("""
            **Metric Explanations:**
            - **Balanced Accuracy**: Average recall per class (good for imbalanced data)
            - **MCC**: Matthews Correlation Coefficient (-1 to 1, accounts for all confusion matrix cells)
            - **Cohen's Kappa**: Agreement between predictions and truth (accounts for chance)
            - **Jaccard Score**: Intersection over union of predictions
            - **Hamming Loss**: Fraction of incorrect predictions (lower is better)
            """)

        # Section 2: Bias-Variance Analysis
        with st.expander("🎯 Bias-Variance Analysis", expanded=False):
            st.markdown("### Understanding Model Complexity")

            bv_data = {}
            for model_name, results in results_dict.items():
                future = results.get("bias_variance_future")
                bv_result = get_background_result_safe(
                    future, timeout=2.0, placeholder_msg="⏳ Bias-variance analysis running in background..."
                )
                if bv_result:
                    bv_data[model_name] = bv_result

            if bv_data:
                # Display metrics in columns
                for model_name, bv_result in bv_data.items():
                    st.markdown(f"#### {model_name}")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Bias²", f"{bv_result['bias_squared']:.4f}")

                    with col2:
                        st.metric("Variance", f"{bv_result['variance']:.4f}")

                    with col3:
                        ratio = bv_result["bias_variance_ratio"]
                        st.metric("Ratio", f"{ratio:.2f}")

                    with col4:
                        interpretation = bv_result["interpretation"]
                        if "High bias" in interpretation:
                            st.error(interpretation)
                        elif "High variance" in interpretation:
                            st.warning(interpretation)
                        else:
                            st.success(interpretation)

                    st.markdown("---")

                # Comparison chart
                if len(bv_data) > 1:
                    fig = plot_bias_variance_comparison(bv_data)
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("⏳ Bias-variance analysis is running in the background. Refresh to see results.")

            st.info("""
            **Understanding Bias-Variance:**
            - **High Bias (Ratio > 2)**: Model is too simple (underfitting). Try more complex models.
            - **High Variance (Ratio < 0.5)**: Model is too complex (overfitting). Try regularization or simpler models.
            - **Good Balance**: Model complexity is appropriate for the data.
            """)

        # Section 3: Learning Curves
        with st.expander("📈 Learning Curves", expanded=False):
            st.markdown("### Performance vs Training Set Size")

            for model_name, results in results_dict.items():
                future = results.get("learning_curves_future")
                lc_result = get_background_result_safe(
                    future, timeout=2.0, placeholder_msg="⏳ Learning curves analysis running in background..."
                )

                if lc_result and "error" not in lc_result:
                    st.markdown(f"#### {model_name}")
                    fig = plot_learning_curve(lc_result)
                    st.plotly_chart(fig, width="stretch")

                    # Interpretation
                    train_final = lc_result["train_scores_mean"][-1]
                    test_final = lc_result["test_scores_mean"][-1]
                    gap = train_final - test_final

                    if gap > 0.1:
                        st.warning(f"⚠️ Large gap ({gap:.3f}) suggests overfitting. More data may not help.")
                    elif test_final < 0.7 and gap < 0.05:
                        st.info("ℹ️ Both curves are low. Model may need more features or complexity.")
                    else:
                        st.success(f"✅ Curves look good! Gap = {gap:.3f}")

                    st.markdown("---")

            st.info("""
            **Interpreting Learning Curves:**
            - **Converging curves**: Model has learned the pattern well
            - **Large gap**: Overfitting - model memorizes training data
            - **Both curves low**: Underfitting - model needs more capacity
            - **Increasing test score**: More data would likely help
            """)

        # Section 4: Calibration & Confidence Analysis
        with st.expander("🔬 Calibration & Confidence Analysis", expanded=False):
            st.markdown("### Probability Calibration Quality")

            for model_name, results in results_dict.items():
                calibration = results.get("calibration")
                confidence_analysis = results.get("confidence_analysis")

                if calibration:
                    st.markdown(f"#### {model_name}")

                    # Display ECE
                    ece = calibration.get("ece")
                    if ece is not None:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("Expected Calibration Error (ECE)", f"{ece:.4f}")
                            if ece < 0.05:
                                st.success("Well calibrated")
                            elif ece < 0.10:
                                st.warning("Moderately calibrated")
                            else:
                                st.error("Poorly calibrated")

                        with col2:
                            # Calibration curve for first class
                            fig_cal = plot_calibration_curve(calibration, class_idx=0)
                            st.plotly_chart(fig_cal, width="stretch")

                    # Confidence analysis
                    if confidence_analysis:
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Avg Confidence (Correct)", f"{confidence_analysis['avg_confidence_correct']:.3f}"
                            )

                        with col2:
                            st.metric(
                                "Avg Confidence (Incorrect)", f"{confidence_analysis['avg_confidence_incorrect']:.3f}"
                            )

                        with col3:
                            separation = confidence_analysis["confidence_separation"]
                            st.metric("Confidence Separation", f"{separation:.3f}")
                            if separation > 0.15:
                                st.success("Good separation")
                            else:
                                st.warning("Poor separation")

                        # Display confidence histogram
                        fig_conf = plot_confidence_histogram(confidence_analysis)
                        st.plotly_chart(fig_conf, width="stretch")

                        # High confidence errors
                        high_conf_errors = confidence_analysis["high_confidence_errors"]
                        if high_conf_errors > 0:
                            st.warning(
                                f"⚠️ {high_conf_errors} high-confidence errors detected. "
                                "Model is overconfident on some mistakes."
                            )

                    st.markdown("---")

            st.info("""
            **Calibration Quality:**
            - **Well-calibrated**: Predicted probabilities match actual frequencies
            - **ECE < 0.05**: Excellent calibration
            - **High confidence separation**: Model knows when it's uncertain
            - **High-confidence errors**: Model is overconfident - consider calibration techniques
            """)

        # Section 5: Detailed Confusion Matrix Analysis
        with st.expander("🧮 Detailed Confusion Matrix Analysis", expanded=False):
            st.markdown("### Per-Class Performance Breakdown")

            for model_name, results in results_dict.items():
                confusion_analysis = results.get("confusion_analysis")

                if confusion_analysis:
                    st.markdown(f"#### {model_name}")

                    # Per-class metrics table
                    per_class_data = []
                    for class_name in confusion_analysis["per_class_precision"].keys():
                        row = {
                            "Class": class_name,
                            "Precision": confusion_analysis["per_class_precision"][class_name],
                            "Recall": confusion_analysis["per_class_recall"][class_name],
                            "F1-Score": confusion_analysis["per_class_f1"][class_name],
                        }
                        per_class_data.append(row)

                    df_per_class = pd.DataFrame(per_class_data)
                    styled_df = df_per_class.style.background_gradient(
                        subset=["Precision", "Recall", "F1-Score"], cmap="RdYlGn", vmin=0, vmax=1
                    ).format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"})

                    st.dataframe(styled_df, width="stretch")

                    # Top misclassification patterns
                    if confusion_analysis["misclassification_patterns"]:
                        st.markdown("##### Top Misclassification Patterns")

                        misc_df = pd.DataFrame(confusion_analysis["misclassification_patterns"][:5])
                        st.dataframe(misc_df, width="stretch")

                    st.markdown("---")

        # Section 6: Statistical Significance Tests
        with st.expander("📉 Statistical Significance Tests", expanded=False):
            st.markdown("### Are Performance Differences Real?")

            comparator = StatisticalModelComparator()

            # Pairwise tests
            pairwise_results = comparator.compute_pairwise_tests(results_dict)

            if pairwise_results and pairwise_results["significant_differences"]:
                st.markdown("#### Significant Differences Found")

                for diff in pairwise_results["significant_differences"]:
                    st.success(
                        f"**{diff['winner']}** significantly outperforms **{diff['loser']}** "
                        f"(p = {diff['p_value']:.4f}) - {diff['interpretation']}"
                    )

                # Heatmap of p-values
                fig_heatmap = plot_pairwise_pvalues_heatmap(pairwise_results)
                st.plotly_chart(fig_heatmap, width="stretch")
            else:
                st.info("No statistically significant differences found between models (p ≥ 0.05)")

            # Friedman test (if 3+ models)
            if len(results_dict) >= 3:
                friedman_result = comparator.compute_friedman_test(results_dict)

                if friedman_result:
                    st.markdown("#### Friedman Test (Overall Comparison)")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Test Statistic", f"{friedman_result['statistic']:.4f}")
                    with col2:
                        st.metric("P-Value", f"{friedman_result['p_value']:.4f}")

                    st.info(friedman_result["interpretation"])

                    # Model rankings
                    st.markdown("##### Model Rankings (Lower is Better)")
                    rankings_df = pd.DataFrame(friedman_result["sorted_rankings"], columns=["Model", "Mean Rank"])
                    st.dataframe(rankings_df, width="stretch")

            st.info("""
            **Statistical Testing:**
            - **p < 0.05**: Statistically significant difference
            - **p ≥ 0.05**: No significant difference (could be due to chance)
            - **Friedman Test**: Non-parametric test for comparing 3+ models
            - **Lower rank**: Better average performance across CV folds
            """)

        # Section 7: Cross-Validation Stability
        with st.expander("🔄 Cross-Validation Stability", expanded=False):
            st.markdown("### How Reliable Are These Results?")

            # Prepare CV scores
            cv_scores_dict = {}
            stability_data = []

            for model_name, results in results_dict.items():
                cv_scores = results.get("cv_scores")
                if cv_scores and len(cv_scores) > 1:
                    cv_scores_dict[model_name] = cv_scores

                    stability = comparator.compute_cv_stability(cv_scores)
                    if stability:
                        stability_data.append(
                            {
                                "Model": model_name,
                                "Mean CV Score": stability["mean"],
                                "Std Dev": stability["std"],
                                "Stability Index": stability["stability_index"],
                                "Interpretation": stability["interpretation"],
                            }
                        )

            if stability_data:
                # Stability metrics table
                df_stability = pd.DataFrame(stability_data)
                styled_df = df_stability.style.background_gradient(
                    subset=["Stability Index"], cmap="RdYlGn", vmin=0, vmax=1
                ).format({"Mean CV Score": "{:.4f}", "Std Dev": "{:.4f}", "Stability Index": "{:.4f}"})

                st.dataframe(styled_df, width="stretch")

                # Box plot
                fig_boxplot = plot_cv_stability_boxplot(cv_scores_dict)
                st.plotly_chart(fig_boxplot, width="stretch")

            st.info("""
            **Stability Metrics:**
            - **High Stability (>0.9)**: Very consistent performance - reliable model
            - **Moderate Stability (0.7-0.9)**: Some variability but acceptable
            - **Low Stability (<0.7)**: High variability - results may not be reliable
            - **Lower Std Dev**: More stable predictions across different data splits
            """)

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
                        leaderboard = evaluator.get_leaderboard("silhouette")

                        # Get best model info
                        best_model = leaderboard[0]
                        best_name = best_model["model"]
                        best_results = st.session_state.results[best_name]

                        # Build metrics summary
                        metrics_str = "\n".join(
                            [
                                f"- {m['model']}: Silhouette={st.session_state.results[m['model']].get('silhouette', 0):.4f}, "
                                f"Davies-Bouldin={st.session_state.results[m['model']].get('davies_bouldin', 0):.4f}, "
                                f"Clusters={st.session_state.results[m['model']].get('n_clusters', 0)}"
                                for m in leaderboard
                            ]
                        )

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

                        if "cluster_quality_assessment" in insights:
                            st.info(f"**📊 Cluster Quality:** {insights['cluster_quality_assessment']}")

                        if "best_model_rationale" in insights:
                            st.success(f"**🏆 Best Model Analysis:** {insights['best_model_rationale']}")

                        if "cluster_interpretation" in insights:
                            st.info(f"**🔍 Cluster Interpretation:** {insights['cluster_interpretation']}")

                        if "improvement_suggestions" in insights:
                            st.warning("**→ Improvement Suggestions:**")
                            if isinstance(insights["improvement_suggestions"], list):
                                for suggestion in insights["improvement_suggestions"]:
                                    st.markdown(f"- {suggestion}")
                            else:
                                st.markdown(insights["improvement_suggestions"])

                    except Exception as e:
                        logger.warning(f"Failed to generate AI clustering insights: {e}")
                        st.error(f"AI analysis failed: {e}")

        # Leaderboard
        st.subheader("Model Leaderboard (by Silhouette Score)")
        evaluator = st.session_state.evaluator
        leaderboard = evaluator.get_leaderboard("silhouette")

        # Display table
        metrics_data = []
        for item in leaderboard:
            model_name = item["model"]
            results = st.session_state.results[model_name]
            metrics_data.append(
                {
                    "Model": model_name,
                    "Silhouette": f"{results.get('silhouette', 0):.4f}",
                    "Davies-Bouldin": f"{results.get('davies_bouldin', 0):.4f}",
                    "Calinski-Harabasz": f"{results.get('calinski_harabasz', 0):.1f}",
                    "N Clusters": results.get("n_clusters", 0),
                    "Stability": f"{results.get('stability', 0):.4f}",
                }
            )

        st.dataframe(pd.DataFrame(metrics_data), width="stretch")

        # UMAP projection
        st.subheader("UMAP Cluster Visualization")
        try:
            best_model_name = leaderboard[0]["model"]
            best_labels = st.session_state.results[best_model_name]["labels"]

            # Compute UMAP
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_umap = reducer.fit_transform(st.session_state.X_processed)

            visualizer = Visualizer()
            fig = visualizer.plot_umap_projection(X_umap, best_labels, f"UMAP - {best_model_name}")
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(f"Could not generate UMAP: {e}")

        # Elbow and Silhouette curves for KMeans
        if "KMeans" in st.session_state.results:
            kmeans_model = st.session_state.models["KMeans"]
            if hasattr(kmeans_model, "inertias") and kmeans_model.inertias:
                col1, col2 = st.columns(2)

                visualizer = Visualizer()
                with col1:
                    st.subheader("Elbow Curve")
                    fig = visualizer.plot_elbow_curve(
                        range(kmeans_model.k_range[0], kmeans_model.k_range[1] + 1), kmeans_model.inertias
                    )
                    st.plotly_chart(fig, width="stretch")

                with col2:
                    st.subheader("Silhouette Scores")
                    fig = visualizer.plot_silhouette_scores(
                        range(kmeans_model.k_range[0], kmeans_model.k_range[1] + 1), kmeans_model.silhouette_scores
                    )
                    st.plotly_chart(fig, width="stretch")
