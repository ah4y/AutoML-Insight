"""Visualization utilities for AutoML-Insight."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix
)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Optional, Any
import warnings
from utils.logging_utils import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


class Visualizer:
    """Create comprehensive visualizations for ML results."""
    
    def __init__(self):
        self.figures = {}
    
    def plot_leaderboard(
        self,
        leaderboard: List[Dict[str, Any]],
        metric_name: str = 'Accuracy',
        title: str = 'Model Leaderboard'
    ) -> go.Figure:
        """
        Create interactive leaderboard plot with confidence intervals.
        
        Args:
            leaderboard: List of model results
            metric_name: Name of the metric
            title: Plot title
            
        Returns:
            Plotly figure
        """
        models = [item['model'] for item in leaderboard]
        scores = [item['score'] for item in leaderboard]
        ci_lower = [item.get('ci_lower', item['score']) for item in leaderboard]
        ci_upper = [item.get('ci_upper', item['score']) for item in leaderboard]
        
        # Compute error bars
        error_y = [upper - score for score, upper in zip(scores, ci_upper)]
        error_y_minus = [score - lower for score, lower in zip(scores, ci_lower)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=models,
            y=scores,
            error_y=dict(
                type='data',
                symmetric=False,
                array=error_y,
                arrayminus=error_y_minus
            ),
            marker_color='steelblue',
            text=[f'{s:.4f}' for s in scores],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Model',
            yaxis_title=metric_name,
            height=500,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def plot_roc_curves(
        self,
        models_data: Dict[str, Dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int
    ) -> go.Figure:
        """
        Plot ROC curves for multiple models.
        
        Args:
            models_data: Dictionary of model results
            X: Feature matrix
            y: Target variable
            n_classes: Number of classes
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Binarize labels for multi-class
        if n_classes > 2:
            y_bin = label_binarize(y, classes=np.unique(y))
        else:
            y_bin = y
        
        for model_name, data in models_data.items():
            if 'model' not in data:
                continue
            
            model = data['model']
            
            try:
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    y_score = model.predict_proba(X)
                else:
                    continue
                
                # Compute ROC for each class
                if n_classes == 2:
                    fpr, tpr, _ = roc_curve(y_bin, y_score[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{model_name} (AUC={roc_auc:.3f})',
                        line=dict(width=2)
                    ))
                else:
                    # Macro-average ROC
                    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
                    roc_auc = auc(fpr_micro, tpr_micro)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr_micro, y=tpr_micro,
                        mode='lines',
                        name=f'{model_name} (AUC={roc_auc:.3f})',
                        line=dict(width=2)
                    ))
            except Exception as e:
                continue
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            
        Returns:
            Plotly figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_norm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title='Proportion')
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='True',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def plot_calibration_curve(
        self,
        models_data: Dict[str, Dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray
    ) -> go.Figure:
        """
        Plot calibration curves for binary classification.
        
        Args:
            models_data: Dictionary of model results
            X: Feature matrix
            y: Target variable (binary)
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for model_name, data in models_data.items():
            if 'model' not in data:
                continue
            
            model = data['model']
            
            try:
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X)[:, 1]
                    
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y, y_prob, n_bins=10, strategy='uniform'
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=mean_predicted_value,
                        y=fraction_of_positives,
                        mode='lines+markers',
                        name=model_name,
                        line=dict(width=2)
                    ))
            except (ValueError, AttributeError, IndexError) as e:
                logger.warning(f"Failed to compute calibration curve for {model_name}: {e}")
                continue
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='Calibration Curves',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        top_n: int = 15,
        title: str = 'Feature Importance'
    ) -> go.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            importance_dict: Dictionary of feature importances
            top_n: Number of top features to show
            title: Plot title
            
        Returns:
            Plotly figure
        """
        import numpy as np
        
        # Ensure we're working with a clean dictionary
        # Force convert all values to Python floats immediately
        importance_dict_clean = {}
        
        # Use list() to avoid iterator issues during conversion
        for k in list(importance_dict.keys()):
            v = importance_dict[k]
            # Convert to float, handling various types
            try:
                # First, just try to convert directly to float
                importance_dict_clean[k] = float(v)
            except (ValueError, TypeError):
                # If that fails, try using item() method for numpy types
                try:
                    importance_dict_clean[k] = float(v.item())
                except (ValueError, TypeError, AttributeError):
                    # If that fails, try indexing
                    try:
                        importance_dict_clean[k] = float(v[0])
                    except (ValueError, TypeError, IndexError, AttributeError) as e:
                        logger.warning(f"Could not convert feature importance for {k}: {type(v)} - {e}. Setting to 0.0")
                        importance_dict_clean[k] = 0.0
        
        # Sort and select top N
        sorted_features = sorted(
            importance_dict_clean.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        features, importances = zip(*sorted_features)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(importances),
            y=list(features),
            orientation='h',
            marker_color='coral'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def plot_elbow_curve(
        self,
        k_range: range,
        inertias: List[float]
    ) -> go.Figure:
        """
        Plot elbow curve for KMeans.
        
        Args:
            k_range: Range of k values
            inertias: Inertia values
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            marker=dict(size=8, color='steelblue'),
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title='Elbow Curve for KMeans',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Inertia',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def plot_silhouette_scores(
        self,
        k_range: range,
        silhouette_scores: List[float]
    ) -> go.Figure:
        """
        Plot silhouette scores for different k values.
        
        Args:
            k_range: Range of k values
            silhouette_scores: Silhouette scores
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=silhouette_scores,
            mode='lines+markers',
            marker=dict(size=8, color='green'),
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title='Silhouette Scores by Number of Clusters',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Silhouette Score',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def plot_umap_projection(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        title: str = 'UMAP Projection'
    ) -> go.Figure:
        """
        Plot 2D UMAP projection of clusters.
        
        Args:
            X: UMAP-transformed features (2D)
            labels: Cluster labels
            title: Plot title
            
        Returns:
            Plotly figure
        """
        df = pd.DataFrame({
            'UMAP1': X[:, 0],
            'UMAP2': X[:, 1],
            'Cluster': labels.astype(str)
        })
        
        fig = px.scatter(
            df,
            x='UMAP1',
            y='UMAP2',
            color='Cluster',
            title=title,
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_correlation_heatmap(
        self,
        data: pd.DataFrame,
        title: str = 'Feature Correlation Heatmap'
    ) -> go.Figure:
        """
        Plot correlation heatmap.
        
        Args:
            data: DataFrame with numeric features
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Calculate correlation
        corr = data.corr()
        
        # Replace NaN with 0 (for constant features with zero variance)
        corr = corr.fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 8},
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def plot_pca_scree(
        self,
        explained_variance_ratio: np.ndarray,
        n_components_selected: int = None,
        title: str = 'PCA Scree Plot'
    ) -> go.Figure:
        """
        Create scree plot showing explained variance per component.
        
        Args:
            explained_variance_ratio: Array of explained variance ratios
            n_components_selected: Number of components selected (for marking)
            title: Plot title
            
        Returns:
            Plotly figure with scree plot
        """
        # Ensure we have a proper numpy array
        explained_variance_ratio = np.asarray(explained_variance_ratio)
        n_components = len(explained_variance_ratio)
        components = np.arange(1, n_components + 1)
        
        # Create subplots for individual and cumulative variance
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Individual Variance', 'Cumulative Variance'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Individual explained variance
        fig.add_trace(
            go.Bar(
                x=components,
                y=explained_variance_ratio,
                name='Individual',
                marker_color='steelblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Cumulative explained variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        fig.add_trace(
            go.Scatter(
                x=components,
                y=cumulative_variance,
                mode='lines+markers',
                name='Cumulative',
                line=dict(color='red', width=3),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # Mark selected components if provided
        if n_components_selected is not None and n_components_selected > 0:
            # Ensure n_components_selected is an integer
            n_comp_int = int(n_components_selected) if not isinstance(n_components_selected, int) else n_components_selected
            
            # Ensure it's within valid range
            if n_comp_int > 0 and n_comp_int <= len(cumulative_variance):
                # Mark on individual plot
                fig.add_vline(
                    x=n_comp_int,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Selected: {n_comp_int}",
                    row=1, col=1
                )
                
                # Mark on cumulative plot
                selected_variance = cumulative_variance[n_comp_int - 1]
                fig.add_hline(
                    y=selected_variance,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"{selected_variance:.1%} variance",
                    row=1, col=2
                )
        
        # Update layout
        fig.update_xaxes(title_text="Principal Component", row=1, col=1)
        fig.update_xaxes(title_text="Principal Component", row=1, col=2)
        fig.update_yaxes(title_text="Explained Variance Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Explained Variance", row=1, col=2)
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_pca_2d_scatter(
        self,
        X_pca: np.ndarray,
        y: Optional[np.ndarray] = None,
        explained_variance_ratio: Optional[np.ndarray] = None,
        title: str = 'PCA 2D Projection'
    ) -> go.Figure:
        """
        Create 2D scatter plot of first two principal components.
        
        Args:
            X_pca: PCA-transformed data (n_samples, n_components)
            y: Optional labels for coloring
            explained_variance_ratio: Explained variance ratios for axis labels
            title: Plot title
            
        Returns:
            Plotly figure with 2D scatter plot
        """
        # Prepare axis labels
        if explained_variance_ratio is not None:
            # Ensure it's a numpy array
            evr = np.asarray(explained_variance_ratio)
            if evr.ndim > 0 and len(evr) >= 2:
                x_var = float(evr[0]) * 100
                y_var = float(evr[1]) * 100
                x_label = f'PC1 ({x_var:.1f}% variance)'
                y_label = f'PC2 ({y_var:.1f}% variance)'
            else:
                x_label = 'PC1'
                y_label = 'PC2'
        else:
            x_label = 'PC1'
            y_label = 'PC2'
        
        # Create DataFrame for plotting
        # Ensure X_pca is a proper numpy array
        X_pca = np.asarray(X_pca)
        if X_pca.ndim == 1:
            X_pca = X_pca.reshape(-1, 1)
            
        plot_data = {
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1] if X_pca.shape[1] > 1 else np.zeros(len(X_pca))
        }
        
        # Add color information if labels provided
        if y is not None:
            if len(np.unique(y)) <= 20:  # Discrete labels
                plot_data['Label'] = y.astype(str)
                color_col = 'Label'
                color_discrete_map = None
            else:  # Continuous values
                plot_data['Value'] = y
                color_col = 'Value'
                color_discrete_map = None
        else:
            color_col = None
            color_discrete_map = None
        
        df = pd.DataFrame(plot_data)
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='PC1',
            y='PC2',
            color=color_col,
            title=title,
            labels={'PC1': x_label, 'PC2': y_label},
            template='plotly_white',
            height=500
        )
        
        # Update marker properties
        fig.update_traces(
            marker=dict(size=6, opacity=0.7, line=dict(width=0.5, color='white'))
        )
        
        return fig
    
    def plot_dimred_comparison_leaderboard(
        self,
        leaderboard: List[Dict[str, Any]],
        metric_name: str = 'Accuracy',
        title: str = 'Model Comparison: With vs Without Dimensionality Reduction'
    ) -> go.Figure:
        """
        Create leaderboard plot that shows dimred comparison information.
        
        Args:
            leaderboard: List with dimred comparison metadata
            metric_name: Name of the metric being plotted
            title: Plot title
            
        Returns:
            Plotly figure with enhanced leaderboard
        """
        models = [item['model'] for item in leaderboard]
        scores = [item['score'] for item in leaderboard]
        ci_lower = [item.get('ci_lower', item['score']) for item in leaderboard]
        ci_upper = [item.get('ci_upper', item['score']) for item in leaderboard]
        
        # Extract dimred information
        uses_dimred = [item.get('uses_dimred', False) for item in leaderboard]
        dimred_methods = [item.get('dimred_method', 'none') for item in leaderboard]
        
        # Create colors based on dimred usage
        colors = ['orange' if used else 'steelblue' for used in uses_dimred]
        
        # Error bars
        error_y = dict(
            type='data',
            array=[ci_upper[i] - scores[i] for i in range(len(scores))],
            arrayminus=[scores[i] - ci_lower[i] for i in range(len(scores))],
            visible=True
        )
        
        # Create hover text with dimred information
        hover_text = []
        for i, item in enumerate(leaderboard):
            text = f"Model: {models[i]}<br>"
            text += f"{metric_name}: {scores[i]:.4f}<br>"
            text += f"CI: [{ci_lower[i]:.4f}, {ci_upper[i]:.4f}]<br>"
            text += f"Uses DimRed: {uses_dimred[i]}<br>"
            if uses_dimred[i]:
                text += f"Method: {dimred_methods[i]}<br>"
                n_components = item.get('n_components', 0)
                if n_components > 0:
                    text += f"Components: {n_components}<br>"
            
            # Add comparison info if available
            comparison = item.get('comparison', {})
            if isinstance(comparison, dict) and 'improvement' in comparison:
                improvement = comparison['improvement']
                text += f"Improvement: {improvement:+.4f}<br>"
                if comparison.get('is_significant', False):
                    text += "Significant: Yes"
                else:
                    text += "Significant: No"
            
            hover_text.append(text)
        
        fig = go.Figure(data=go.Bar(
            x=scores,
            y=models,
            orientation='h',
            error_x=error_y,
            marker_color=colors,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=metric_name,
            yaxis_title='Model',
            template='plotly_white',
            height=max(400, len(models) * 30),
            showlegend=False
        )
        
        # Add annotation explaining colors
        fig.add_annotation(
            text="🟠 With DimRed | 🔵 Without DimRed",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1
        )
        
        return fig


# Standalone wrapper functions for backward compatibility
def plot_pca_scree(pca_transformer, title: str = 'PCA Scree Plot') -> go.Figure:
    """
    Create scree plot from a fitted PCA transformer.
    
    Args:
        pca_transformer: Fitted PCA transformer with explained_variance_ratio_
        title: Plot title
        
    Returns:
        Plotly figure with scree plot
    """
    if not hasattr(pca_transformer, 'explained_variance_ratio_'):
        raise ValueError("PCA transformer must have explained_variance_ratio_ attribute")
    
    visualizer = Visualizer()
    return visualizer.plot_pca_scree(
        explained_variance_ratio=pca_transformer.explained_variance_ratio_,
        n_components_selected=getattr(pca_transformer, 'n_components', None),
        title=title
    )


def plot_pca_2d_scatter(X_pca: np.ndarray, y_labels: np.ndarray = None, title: str = 'PCA 2D Projection', explained_variance_ratio: np.ndarray = None) -> go.Figure:
    """
    Create 2D scatter plot from PCA-transformed data.
    
    Args:
        X_pca: PCA-transformed data (n_samples, n_components)
        y_labels: Optional labels for coloring
        title: Plot title
        explained_variance_ratio: Optional explained variance ratios for axis labels
        
    Returns:
        Plotly figure with 2D scatter plot
    """
    visualizer = Visualizer()
    return visualizer.plot_pca_2d_scatter(X_pca, y_labels, explained_variance_ratio, title)
