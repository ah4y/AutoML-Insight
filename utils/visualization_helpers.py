"""Visualization Helpers for Advanced Model Evaluation.

This module provides reusable plotting functions for advanced evaluation metrics
including learning curves, calibration curves, confidence histograms, and more.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Any


# Color scheme constants
TRAIN_COLOR = '#1f77b4'  # Blue
TEST_COLOR = '#ff7f0e'   # Orange
GOOD_COLOR = '#2ca02c'   # Green
BAD_COLOR = '#d62728'    # Red
NEUTRAL_COLOR = '#9467bd' # Purple


def plot_learning_curve(learning_curve_data: Dict) -> go.Figure:
    """Plot learning curves showing performance vs training set size.
    
    Args:
        learning_curve_data: Dictionary with train_sizes, train_scores_mean,
                            train_scores_std, test_scores_mean, test_scores_std
    
    Returns:
        Plotly Figure object
    """
    if not learning_curve_data or 'error' in learning_curve_data:
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text="Learning curve data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
    
    train_sizes = learning_curve_data['train_sizes']
    train_mean = learning_curve_data['train_scores_mean']
    train_std = learning_curve_data['train_scores_std']
    test_mean = learning_curve_data['test_scores_mean']
    test_std = learning_curve_data['test_scores_std']
    
    fig = go.Figure()
    
    # Training score with confidence band
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color=TRAIN_COLOR, width=2),
        marker=dict(size=8)
    ))
    
    # Training confidence band
    fig.add_trace(go.Scatter(
        x=train_sizes + train_sizes[::-1],
        y=(np.array(train_mean) + np.array(train_std)).tolist() + 
          (np.array(train_mean) - np.array(train_std))[::-1].tolist(),
        fill='toself',
        fillcolor=f'rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Test score with confidence band
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=test_mean,
        mode='lines+markers',
        name='Cross-validation Score',
        line=dict(color=TEST_COLOR, width=2),
        marker=dict(size=8)
    ))
    
    # Test confidence band
    fig.add_trace(go.Scatter(
        x=train_sizes + train_sizes[::-1],
        y=(np.array(test_mean) + np.array(test_std)).tolist() + 
          (np.array(test_mean) - np.array(test_std))[::-1].tolist(),
        fill='toself',
        fillcolor=f'rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title='Learning Curves',
        xaxis_title='Training Set Size',
        yaxis_title='Score (Accuracy)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_calibration_curve(calibration_data: Dict, class_idx: int = 0) -> go.Figure:
    """Plot calibration curve for a specific class.
    
    Args:
        calibration_data: Dictionary with calibration metrics
        class_idx: Index of class to plot (default: 0)
    
    Returns:
        Plotly Figure object
    """
    if not calibration_data or 'per_class' not in calibration_data:
        fig = go.Figure()
        fig.add_annotation(
            text="Calibration data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
    
    class_data = calibration_data['per_class'][class_idx]
    prob_true = [p for p in class_data['prob_true'] if p is not None]
    prob_pred = [p for p in class_data['prob_pred'] if p is not None]
    
    fig = go.Figure()
    
    # Calibration curve
    fig.add_trace(go.Scatter(
        x=prob_pred,
        y=prob_true,
        mode='lines+markers',
        name=f'Class {class_idx}',
        line=dict(color=NEUTRAL_COLOR, width=2),
        marker=dict(size=8)
    ))
    
    # Perfect calibration line (diagonal)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    ece = calibration_data.get('ece', None)
    title_text = 'Calibration Curve'
    if ece is not None:
        title_text += f' (ECE: {ece:.4f})'
    
    fig.update_layout(
        title=title_text,
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_confidence_histogram(confidence_analysis: Dict) -> go.Figure:
    """Plot histogram of prediction confidence split by correctness.
    
    Args:
        confidence_analysis: Dictionary with confidence metrics
    
    Returns:
        Plotly Figure object
    """
    if not confidence_analysis:
        fig = go.Figure()
        fig.add_annotation(
            text="Confidence analysis not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
    
    # Create sample data for visualization (placeholder)
    # In practice, you'd need to pass the actual confidence values
    fig = go.Figure()
    
    avg_correct = confidence_analysis['avg_confidence_correct']
    avg_incorrect = confidence_analysis['avg_confidence_incorrect']
    
    fig.add_trace(go.Bar(
        x=['Correct Predictions', 'Incorrect Predictions'],
        y=[avg_correct, avg_incorrect],
        marker_color=[GOOD_COLOR, BAD_COLOR],
        text=[f'{avg_correct:.3f}', f'{avg_incorrect:.3f}'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Average Prediction Confidence',
        yaxis_title='Average Confidence',
        template='plotly_white',
        height=350,
        showlegend=False
    )
    
    return fig


def plot_bias_variance_comparison(models_bv_data: Dict[str, Dict]) -> go.Figure:
    """Plot bias-variance comparison across multiple models.
    
    Args:
        models_bv_data: Dictionary mapping model names to bias-variance dictionaries
    
    Returns:
        Plotly Figure object
    """
    model_names = []
    bias_values = []
    variance_values = []
    
    for model_name, bv_data in models_bv_data.items():
        if bv_data and 'bias_squared' in bv_data and bv_data['bias_squared'] is not None:
            model_names.append(model_name)
            bias_values.append(bv_data['bias_squared'])
            variance_values.append(bv_data['variance'])
    
    if not model_names:
        fig = go.Figure()
        fig.add_annotation(
            text="Bias-variance data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Bias²',
        x=model_names,
        y=bias_values,
        marker_color=TRAIN_COLOR
    ))
    
    fig.add_trace(go.Bar(
        name='Variance',
        x=model_names,
        y=variance_values,
        marker_color=TEST_COLOR
    ))
    
    fig.update_layout(
        title='Bias-Variance Decomposition',
        xaxis_title='Model',
        yaxis_title='Error Component',
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_pairwise_pvalues_heatmap(pairwise_tests: Dict) -> go.Figure:
    """Plot heatmap of pairwise statistical test p-values.
    
    Args:
        pairwise_tests: Dictionary with pairwise test results
    
    Returns:
        Plotly Figure object
    """
    if not pairwise_tests or 'pairwise_ttest' not in pairwise_tests:
        fig = go.Figure()
        fig.add_annotation(
            text="Statistical test data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
    
    ttest_results = pairwise_tests['pairwise_ttest']
    
    # Extract unique model names
    model_names = set()
    for (model1, model2) in ttest_results.keys():
        model_names.add(model1)
        model_names.add(model2)
    model_names = sorted(list(model_names))
    
    # Create matrix
    n = len(model_names)
    matrix = np.ones((n, n))  # Default 1.0 (no difference)
    
    for (model1, model2), pval in ttest_results.items():
        if pval is not None:
            i = model_names.index(model1)
            j = model_names.index(model2)
            matrix[i, j] = pval
            matrix[j, i] = pval
    
    # Diagonal is NaN (model compared to itself)
    np.fill_diagonal(matrix, np.nan)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=model_names,
        y=model_names,
        colorscale='RdYlGn_r',
        zmid=0.05,  # Center at significance threshold
        text=np.round(matrix, 4),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="p-value")
    ))
    
    fig.update_layout(
        title='Pairwise Statistical Test P-Values<br>(p < 0.05 = significant)',
        template='plotly_white',
        height=500,
        xaxis_title='Model',
        yaxis_title='Model'
    )
    
    return fig


def plot_cv_stability_boxplot(cv_scores_dict: Dict[str, List[float]]) -> go.Figure:
    """Plot box plot of cross-validation score distributions.
    
    Args:
        cv_scores_dict: Dictionary mapping model names to lists of CV scores
    
    Returns:
        Plotly Figure object
    """
    if not cv_scores_dict:
        fig = go.Figure()
        fig.add_annotation(
            text="CV scores not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
    
    fig = go.Figure()
    
    for model_name, scores in cv_scores_dict.items():
        if scores and len(scores) > 0:
            fig.add_trace(go.Box(
                y=scores,
                name=model_name,
                boxmean='sd'  # Show mean and standard deviation
            ))
    
    fig.update_layout(
        title='Cross-Validation Score Stability',
        yaxis_title='CV Score (Accuracy)',
        template='plotly_white',
        height=450,
        showlegend=False
    )
    
    return fig
