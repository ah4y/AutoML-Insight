"""
Report Tab - Generate and download PDF reports.
"""

import logging
from datetime import datetime

import streamlit as st

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class ReportTab(BaseTab):
    """Tab for generating and downloading reports."""

    def render(self) -> None:
        """Render the report tab."""
        st.subheader("📄 Generate Report")

        # Check for both standard and professional results
        has_results = st.session_state.results or st.session_state.get("professional_results")

        if not has_results:
            st.info("Run AutoML to generate a report")
            return

        # Check if models actually trained successfully
        if not st.session_state.get("models"):
            st.warning("⚠️ No trained models available for report generation.")
            st.info("💡 Models may have failed during training. Check the AutoML results for details.")
            return

        # AI-Generated Comprehensive Report
        if st.session_state.ai_engine and st.session_state.ai_engine is not False:
            with st.expander("🤖 AI-Generated Comprehensive Report", expanded=True):
                with st.spinner("🤖 AI is writing your comprehensive report..."):
                    try:
                        # Collect all relevant information
                        is_clustering = st.session_state.task_type == "Clustering"

                        # Handle both standard and professional results
                        professional_results = st.session_state.get("professional_results")
                        standard_results = st.session_state.get("results")
                        evaluator = st.session_state.get("evaluator")

                        # Use professional results if available
                        if professional_results and "individual_models" in professional_results:
                            # Get best model from professional results
                            best_name = None
                            best_score = -float("inf")
                            for name, result in professional_results["individual_models"].items():
                                if result.get("best_score", -1000) > best_score:
                                    best_score = result.get("best_score", -1000)
                                    best_name = name

                            if is_clustering:
                                metrics_summary = f"""**Best Model:** {best_name}
**Silhouette Score:** {best_score:.4f}
**Optimization Time:** {professional_results['individual_models'][best_name].get('optimization_time', 0):.1f}s
**Trials:** {professional_results['individual_models'][best_name].get('n_trials', 0)}"""

                                all_models = "\n".join(
                                    [
                                        f"- {name}: Score={result.get('best_score', 0):.4f}, Improvement={result.get('improvement_percent', 0):+.1f}%"
                                        for name, result in professional_results["individual_models"].items()
                                    ]
                                )
                            else:
                                metrics_summary = f"""**Best Model:** {best_name}
**Score:** {best_score:.4f}
**Improvement:** {professional_results['individual_models'][best_name].get('improvement_percent', 0):+.1f}%
**Optimization Time:** {professional_results['individual_models'][best_name].get('optimization_time', 0):.1f}s"""

                                all_models = "\n".join(
                                    [
                                        f"- {name}: Score={result.get('best_score', 0):.4f}"
                                        for name, result in professional_results["individual_models"].items()
                                    ]
                                )

                        elif evaluator and standard_results:
                            # Use standard results with evaluator
                            if is_clustering:
                                leaderboard = evaluator.get_leaderboard("silhouette")
                                best_model = leaderboard[0]
                                best_name = best_model["model"]
                                best_results = standard_results[best_name]

                                metrics_summary = f"""**Best Model:** {best_name}
**Silhouette Score:** {best_results.get('silhouette', 0):.4f}
**Davies-Bouldin Score:** {best_results.get('davies_bouldin', 0):.4f}
**Number of Clusters:** {best_results.get('n_clusters', 0)}"""

                                all_models = "\n".join(
                                    [
                                        f"- {m['model']}: Silhouette={standard_results[m['model']].get('silhouette', 0):.4f}, "
                                        f"Clusters={standard_results[m['model']].get('n_clusters', 0)}"
                                        for m in leaderboard
                                    ]
                                )
                            else:
                                leaderboard = evaluator.get_leaderboard("accuracy")
                                best_model = leaderboard[0]
                                best_name = best_model.get("model", "Unknown")
                                best_results = standard_results.get(best_name, {})

                                # Try to get metrics from different possible keys
                                best_accuracy = (
                                    best_model.get("accuracy")
                                    or best_model.get("score")
                                    or best_results.get("accuracy_mean", 0)
                                )
                                best_precision = best_model.get("precision") or best_results.get(
                                    "precision_macro_mean", 0
                                )
                                best_recall = best_model.get("recall") or best_results.get("recall_macro_mean", 0)

                                metrics_summary = f"""**Best Model:** {best_name}
**Accuracy:** {best_accuracy:.4f}
**Precision:** {best_precision:.4f}
**Recall:** {best_recall:.4f}"""

                                # Build model list with error handling
                                model_lines = []
                                for m in leaderboard[:5]:
                                    model_name = m.get("model", "Unknown")
                                    acc = (
                                        m.get("accuracy")
                                        or m.get("score")
                                        or standard_results.get(model_name, {}).get("accuracy_mean", 0)
                                    )
                                    model_lines.append(f"- {model_name}: Accuracy={acc:.4f}")
                                all_models = "\n".join(model_lines)
                        else:
                            st.warning("No evaluator or results available for report generation")
                            return

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
                        if "executive_summary" in report_insights:
                            st.markdown("### 📋 Executive Summary")
                            st.write(report_insights["executive_summary"])

                        if "methodology" in report_insights:
                            st.markdown("### 🔬 Methodology")
                            st.write(report_insights["methodology"])

                        if "key_findings" in report_insights:
                            st.markdown("### 🔑 Key Findings")
                            if isinstance(report_insights["key_findings"], list):
                                for finding in report_insights["key_findings"]:
                                    st.markdown(f"- {finding}")
                            else:
                                st.write(report_insights["key_findings"])

                        if "best_model_analysis" in report_insights:
                            st.markdown("### 🏆 Best Model Analysis")
                            st.write(report_insights["best_model_analysis"])

                        if "recommendations" in report_insights:
                            st.markdown("### 💡 Recommendations")
                            if isinstance(report_insights["recommendations"], list):
                                for rec in report_insights["recommendations"]:
                                    st.markdown(f"- {rec}")
                            else:
                                st.write(report_insights["recommendations"])

                        if "limitations" in report_insights:
                            st.markdown("### ⚠️ Limitations")
                            if isinstance(report_insights["limitations"], list):
                                for lim in report_insights["limitations"]:
                                    st.markdown(f"- {lim}")
                            else:
                                st.write(report_insights["limitations"])

                        if "conclusion" in report_insights:
                            st.markdown("### 🎯 Conclusion")
                            st.write(report_insights["conclusion"])

                    except Exception as e:
                        error_msg = str(e)
                        logger.warning(f"Failed to generate AI report: {error_msg}")

                        # Check if it's a rate limit error
                        if "rate_limit" in error_msg.lower() or "429" in error_msg:
                            st.warning(
                                "⚠️ **AI Report Generation Unavailable**: API rate limit reached. Showing structured report instead."
                            )

                            # FALLBACK: Generate structured report from data
                            st.markdown("### 📋 Executive Summary")

                            is_clustering = st.session_state.task_type == "Clustering"
                            evaluator = st.session_state.evaluator

                            if is_clustering:
                                leaderboard = evaluator.get_leaderboard("silhouette")
                                best_model = leaderboard[0]
                                best_name = best_model["model"]
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
                                leaderboard = evaluator.get_leaderboard("accuracy")
                                best_model = leaderboard[0]
                                best_name = best_model.get("model", "Unknown")
                                test_acc = best_model.get("test_accuracy", best_model.get("score", 0))
                                train_acc = best_model.get("train_accuracy", test_acc)
                                gap = best_model.get("overfitting_gap", abs(train_acc - test_acc))

                                st.write(f"""
This AutoML analysis tested **{len(st.session_state.results)} classification algorithms** on a dataset of
**{st.session_state.data.shape[0]:,} samples** with **{st.session_state.data.shape[1]} features**.

**Key Result**: {best_name} achieved the best performance with **{test_acc:.1%} test accuracy**
and an overfitting gap of **{gap:.1%}** ({('✅ Good' if gap < 0.10 else '⚠️ Overfit')}).
""")

                                st.markdown("### 🔑 Key Findings")

                                # Find model with lowest overfitting
                                sorted_by_gap = sorted(leaderboard, key=lambda x: x.get("overfitting_gap", 1))
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
                                    recommendations.append(
                                        f"**Reduce Overfitting**: {best_name} shows high train-test gap ({gap:.1%}). Consider using {best_gap_model['model']} (gap: {best_gap_model.get('overfitting_gap', 0):.1%}) for better generalization."
                                    )
                                elif gap > 0.10:
                                    recommendations.append(
                                        f"**Monitor Overfitting**: {best_name} shows moderate overfitting. Validate on additional data before deployment."
                                    )
                                else:
                                    recommendations.append(
                                        f"**Deploy Confidently**: {best_name} shows excellent generalization (gap: {gap:.1%}). Ready for production."
                                    )

                                if test_acc < 0.50:
                                    recommendations.append(
                                        f"**Improve Performance**: Current accuracy ({test_acc:.1%}) is low. Consider feature engineering or collecting more data."
                                    )

                                recommendations.append(
                                    "**Validate Results**: Test final model on completely unseen data before production deployment."
                                )
                                recommendations.append(
                                    "**Monitor Performance**: Set up tracking to detect model drift over time."
                                )

                                for rec in recommendations:
                                    st.markdown(f"- {rec}")

                            st.markdown("### ⚠️ Note")
                            st.info(
                                "This is a structured report generated from your results. For AI-powered insights and detailed analysis, try again tomorrow when your API quota resets, or upgrade your plan."
                            )

                        else:
                            st.error(f"AI report generation failed: {error_msg}")

        st.write("---")
        st.write("Download a comprehensive PDF report with all results and visualizations.")

        if st.button("📥 Download PDF Report"):
            try:
                from app.report_builder import ReportBuilder

                with st.spinner("Generating PDF report..."):
                    # Professional (Optuna) mode never populates st.session_state.results -
                    # its results live in professional_results with a different shape. Build
                    # a report-compatible dict from it instead of silently passing {} and
                    # rendering a report that says "Models Evaluated: 0".
                    report_results = st.session_state.results
                    if not report_results:
                        prof_models = st.session_state.get("professional_results", {}).get("individual_models", {})
                        if st.session_state.task_type == "Classification":
                            report_results = {
                                name: {"accuracy_mean": res.get("best_score", 0)} for name, res in prof_models.items()
                            }
                        else:
                            report_results = {
                                name: {"silhouette": res.get("best_score", 0)} for name, res in prof_models.items()
                            }

                    builder = ReportBuilder()
                    report_path = builder.generate_report(
                        data=st.session_state.data,
                        profile=st.session_state.get("profile") or {},
                        results=report_results,
                        task_type=st.session_state.task_type,
                        recommendation=st.session_state.get("recommendation"),
                    )

                    # Read and download
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="Download Report",
                            data=f.read(),
                            file_name=f"AutoML_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                        )

                    st.success(f"✅ Report generated: {report_path}")
            except Exception as e:
                st.error(f"Error generating report: {e}")
