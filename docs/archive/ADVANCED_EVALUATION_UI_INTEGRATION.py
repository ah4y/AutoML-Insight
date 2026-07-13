"""
UI Dashboard Integration Code for Advanced Evaluation Sections

Add this code to app/ui_dashboard.py in the render_classification_results method
after the existing model comparison visualization (around line 3690).

This implements all the collapsible expander sections for advanced evaluation.
"""

import streamlit as st
import pandas as pd
from core.background_analyzer import get_background_result_safe
from core.statistical_tests import StatisticalModelComparator
from utils.visualization_helpers import (
    plot_learning_curve, plot_calibration_curve, plot_confidence_histogram,
    plot_bias_variance_comparison, plot_pairwise_pvalues_heatmap,
    plot_cv_stability_boxplot
)


def render_advanced_evaluation_sections(results_dict):
    """
    Render all advanced evaluation sections with collapsible expanders.
    
    Args:
        results_dict: Dictionary mapping model names to their results
    """
    st.markdown("---")
    st.markdown("## 📊 Advanced Model Evaluation")
    
    # ============================================================================
    # Section 1: Extended Performance Metrics
    # ============================================================================
    with st.expander("📊 Extended Performance Metrics", expanded=False):
        st.markdown("### Comprehensive Metrics Beyond Accuracy")
        
        metrics_data = []
        for model_name, results in results_dict.items():
            row = {
                'Model': model_name,
                'Accuracy': results.get('test_accuracy', 0),
                'Balanced_Acc': results.get('balanced_accuracy', None),
                'MCC': results.get('matthews_corrcoef', None),
                'Cohen_Kappa': results.get('cohen_kappa', None),
                'Jaccard': results.get('jaccard_score', None),
                'Hamming_Loss': results.get('hamming_loss', None)
            }
            metrics_data.append(row)
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Style the dataframe with color gradients
        styled_df = df_metrics.style.background_gradient(
            subset=['Accuracy', 'Balanced_Acc', 'MCC', 'Cohen_Kappa', 'Jaccard'],
            cmap='RdYlGn',
            vmin=0, vmax=1
        ).background_gradient(
            subset=['Hamming_Loss'],
            cmap='RdYlGn_r',
            vmin=0, vmax=1
        ).format({
            'Accuracy': '{:.4f}',
            'Balanced_Acc': '{:.4f}',
            'MCC': '{:.4f}',
            'Cohen_Kappa': '{:.4f}',
            'Jaccard': '{:.4f}',
            'Hamming_Loss': '{:.4f}'
        })
        
        st.dataframe(styled_df, width='stretch')
        
        st.info("""
        **Metric Explanations:**
        - **Balanced Accuracy**: Average recall per class (good for imbalanced data)
        - **MCC**: Matthews Correlation Coefficient (-1 to 1, accounts for all confusion matrix cells)
        - **Cohen's Kappa**: Agreement between predictions and truth (accounts for chance)
        - **Jaccard Score**: Intersection over union of predictions
        - **Hamming Loss**: Fraction of incorrect predictions (lower is better)
        """)
    
    # ============================================================================
    # Section 2: Bias-Variance Analysis
    # ============================================================================
    with st.expander("🎯 Bias-Variance Analysis", expanded=False):
        st.markdown("### Understanding Model Complexity")
        
        bv_data = {}
        for model_name, results in results_dict.items():
            future = results.get('bias_variance_future')
            bv_result = get_background_result_safe(
                future, 
                timeout=2.0,
                placeholder_msg="⏳ Bias-variance analysis running in background..."
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
                    ratio = bv_result['bias_variance_ratio']
                    st.metric("Ratio", f"{ratio:.2f}")
                
                with col4:
                    # Interpretation with color coding
                    interpretation = bv_result['interpretation']
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
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("⏳ Bias-variance analysis is running in the background. Refresh to see results.")
        
        st.info("""
        **Understanding Bias-Variance:**
        - **High Bias (Ratio > 2)**: Model is too simple (underfitting). Try more complex models.
        - **High Variance (Ratio < 0.5)**: Model is too complex (overfitting). Try regularization or simpler models.
        - **Good Balance**: Model complexity is appropriate for the data.
        """)
    
    # ============================================================================
    # Section 3: Learning Curves
    # ============================================================================
    with st.expander("📈 Learning Curves", expanded=False):
        st.markdown("### Performance vs Training Set Size")
        
        for model_name, results in results_dict.items():
            future = results.get('learning_curves_future')
            lc_result = get_background_result_safe(
                future,
                timeout=2.0,
                placeholder_msg="⏳ Learning curves analysis running in background..."
            )
            
            if lc_result and 'error' not in lc_result:
                st.markdown(f"#### {model_name}")
                fig = plot_learning_curve(lc_result)
                st.plotly_chart(fig, width='stretch')
                
                # Interpretation
                train_final = lc_result['train_scores_mean'][-1]
                test_final = lc_result['test_scores_mean'][-1]
                gap = train_final - test_final
                
                if gap > 0.1:
                    st.warning(f"⚠️ Large gap ({gap:.3f}) suggests overfitting. More data may not help.")
                elif test_final < 0.7 and gap < 0.05:
                    st.info(f"ℹ️ Both curves are low. Model may need more features or complexity.")
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
    
    # ============================================================================
    # Section 4: Calibration & Confidence Analysis
    # ============================================================================
    with st.expander("🔬 Calibration & Confidence Analysis", expanded=False):
        st.markdown("### Probability Calibration Quality")
        
        for model_name, results in results_dict.items():
            calibration = results.get('calibration')
            confidence_analysis = results.get('confidence_analysis')
            
            if calibration:
                st.markdown(f"#### {model_name}")
                
                # Display ECE
                ece = calibration.get('ece')
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
                        st.plotly_chart(fig_cal, width='stretch')
                
                # Confidence analysis
                if confidence_analysis:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Avg Confidence (Correct)",
                            f"{confidence_analysis['avg_confidence_correct']:.3f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Avg Confidence (Incorrect)",
                            f"{confidence_analysis['avg_confidence_incorrect']:.3f}"
                        )
                    
                    with col3:
                        separation = confidence_analysis['confidence_separation']
                        st.metric("Confidence Separation", f"{separation:.3f}")
                        if separation > 0.15:
                            st.success("Good separation")
                        else:
                            st.warning("Poor separation")
                    
                    # Display confidence histogram
                    fig_conf = plot_confidence_histogram(confidence_analysis)
                    st.plotly_chart(fig_conf, width='stretch')
                    
                    # High confidence errors
                    high_conf_errors = confidence_analysis['high_confidence_errors']
                    if high_conf_errors > 0:
                        st.warning(f"⚠️ {high_conf_errors} high-confidence errors detected. "
                                 "Model is overconfident on some mistakes.")
                
                st.markdown("---")
        
        st.info("""
        **Calibration Quality:**
        - **Well-calibrated**: Predicted probabilities match actual frequencies
        - **ECE < 0.05**: Excellent calibration
        - **High confidence separation**: Model knows when it's uncertain
        - **High-confidence errors**: Model is overconfident - consider calibration techniques
        """)
    
    # ============================================================================
    # Section 5: Detailed Confusion Matrix Analysis
    # ============================================================================
    with st.expander("🧮 Detailed Confusion Matrix Analysis", expanded=False):
        st.markdown("### Per-Class Performance Breakdown")
        
        for model_name, results in results_dict.items():
            confusion_analysis = results.get('confusion_analysis')
            
            if confusion_analysis:
                st.markdown(f"#### {model_name}")
                
                # Per-class metrics table
                per_class_data = []
                for class_name in confusion_analysis['per_class_precision'].keys():
                    row = {
                        'Class': class_name,
                        'Precision': confusion_analysis['per_class_precision'][class_name],
                        'Recall': confusion_analysis['per_class_recall'][class_name],
                        'F1-Score': confusion_analysis['per_class_f1'][class_name]
                    }
                    per_class_data.append(row)
                
                df_per_class = pd.DataFrame(per_class_data)
                styled_df = df_per_class.style.background_gradient(
                    subset=['Precision', 'Recall', 'F1-Score'],
                    cmap='RdYlGn',
                    vmin=0, vmax=1
                ).format({
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1-Score': '{:.4f}'
                })
                
                st.dataframe(styled_df, width='stretch')
                
                # Top misclassification patterns
                if confusion_analysis['misclassification_patterns']:
                    st.markdown("##### Top Misclassification Patterns")
                    
                    misc_df = pd.DataFrame(
                        confusion_analysis['misclassification_patterns'][:5]
                    )
                    st.dataframe(misc_df, width='stretch')
                
                st.markdown("---")
    
    # ============================================================================
    # Section 6: Statistical Significance Tests
    # ============================================================================
    with st.expander("📉 Statistical Significance Tests", expanded=False):
        st.markdown("### Are Performance Differences Real?")
        
        comparator = StatisticalModelComparator()
        
        # Pairwise tests
        pairwise_results = comparator.compute_pairwise_tests(results_dict)
        
        if pairwise_results and pairwise_results['significant_differences']:
            st.markdown("#### Significant Differences Found")
            
            for diff in pairwise_results['significant_differences']:
                st.success(
                    f"**{diff['winner']}** significantly outperforms **{diff['loser']}** "
                    f"(p = {diff['p_value']:.4f}) - {diff['interpretation']}"
                )
            
            # Heatmap of p-values
            fig_heatmap = plot_pairwise_pvalues_heatmap(pairwise_results)
            st.plotly_chart(fig_heatmap, width='stretch')
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
                
                st.info(friedman_result['interpretation'])
                
                # Model rankings
                st.markdown("##### Model Rankings (Lower is Better)")
                rankings_df = pd.DataFrame(
                    friedman_result['sorted_rankings'],
                    columns=['Model', 'Mean Rank']
                )
                st.dataframe(rankings_df, width='stretch')
        
        st.info("""
        **Statistical Testing:**
        - **p < 0.05**: Statistically significant difference
        - **p ≥ 0.05**: No significant difference (could be due to chance)
        - **Friedman Test**: Non-parametric test for comparing 3+ models
        - **Lower rank**: Better average performance across CV folds
        """)
    
    # ============================================================================
    # Section 7: Cross-Validation Stability
    # ============================================================================
    with st.expander("🔄 Cross-Validation Stability", expanded=False):
        st.markdown("### How Reliable Are These Results?")
        
        # Prepare CV scores
        cv_scores_dict = {}
        stability_data = []
        
        for model_name, results in results_dict.items():
            cv_scores = results.get('cv_scores')
            if cv_scores and len(cv_scores) > 1:
                cv_scores_dict[model_name] = cv_scores
                
                stability = comparator.compute_cv_stability(cv_scores)
                if stability:
                    stability_data.append({
                        'Model': model_name,
                        'Mean CV Score': stability['mean'],
                        'Std Dev': stability['std'],
                        'Stability Index': stability['stability_index'],
                        'Interpretation': stability['interpretation']
                    })
        
        if stability_data:
            # Stability metrics table
            df_stability = pd.DataFrame(stability_data)
            styled_df = df_stability.style.background_gradient(
                subset=['Stability Index'],
                cmap='RdYlGn',
                vmin=0, vmax=1
            ).format({
                'Mean CV Score': '{:.4f}',
                'Std Dev': '{:.4f}',
                'Stability Index': '{:.4f}'
            })
            
            st.dataframe(styled_df, width='stretch')
            
            # Box plot
            fig_boxplot = plot_cv_stability_boxplot(cv_scores_dict)
            st.plotly_chart(fig_boxplot, width='stretch')
        
        st.info("""
        **Stability Metrics:**
        - **High Stability (>0.9)**: Very consistent performance - reliable model
        - **Moderate Stability (0.7-0.9)**: Some variability but acceptable
        - **Low Stability (<0.7)**: High variability - results may not be reliable
        - **Lower Std Dev**: More stable predictions across different data splits
        """)


# ============================================================================
# Helper function for background results
# ============================================================================
def _get_background_result(future, result_key, placeholder_msg='Analysis running...'):
    """Safely retrieve background analysis result."""
    if future is None:
        return None
    if future.done():
        try:
            return future.result(timeout=1)
        except Exception as e:
            st.warning(f'Analysis failed: {e}')
            return None
    else:
        st.info(placeholder_msg)
        return None


# ============================================================================
# Integration Instructions
# ============================================================================
"""
To integrate into app/ui_dashboard.py:

1. Add imports at the top of the file
2. In render_classification_results method, after line ~3690 (after existing visualizations), add:

    # Advanced Evaluation Sections
    render_advanced_evaluation_sections(results)

3. That's it! All sections will render as collapsible expanders.
"""
