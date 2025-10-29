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
            except:
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
                    except:
                        print(f"WARNING: Could not convert {k}: {type(v)} - setting to 0.0")
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
