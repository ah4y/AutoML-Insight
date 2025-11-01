"""Supervised learning models."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Any, Dict


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """
    PyTorch-based Multi-Layer Perceptron with advanced features.
    Includes dropout, early stopping, and learning rate scheduling.
    """
    
    def __init__(
        self,
        hidden_layers: tuple = (128, 64),
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 10,
        random_state: int = 42
    ):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
    
    def _build_model(self):
        """Build the neural network architecture."""
        layers = []
        input_size = self.n_features_
        
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, self.n_classes_))
        
        return nn.Sequential(*layers)
    
    def fit(self, X, y):
        """Train the MLP classifier."""
        torch.manual_seed(self.random_state)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        # Build model
        self.model = self._build_model().to(self.device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = torch.softmax(outputs, dim=1)
        
        return probas.cpu().numpy()
    
    def predict(self, X):
        """Predict class labels."""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


def get_supervised_models(random_state: int = 42, n_samples: int = None) -> Dict[str, Any]:
    """
    Get a dictionary of supervised learning models.
    Auto-optimizes model settings based on dataset size.
    
    Args:
        random_state: Random seed for reproducibility
        n_samples: Number of samples in dataset (for optimization)
        
    Returns:
        Dictionary mapping model names to model instances
    """
    # Adaptive settings based on dataset size
    if n_samples is not None and n_samples > 10000:
        # Large dataset optimizations
        svm_max_iter = 500  # Reduce iterations for speed
        rf_estimators = 50   # Fewer trees
        xgb_estimators = 50
        rf_max_depth = 8
    elif n_samples is not None and n_samples > 5000:
        svm_max_iter = 800
        rf_estimators = 75
        xgb_estimators = 75
        rf_max_depth = 10
    else:
        # Standard settings for smaller datasets
        svm_max_iter = 1000
        rf_estimators = 100
        xgb_estimators = 100
        rf_max_depth = 10
    
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            n_jobs=1
        ),
        'LinearSVM': SVC(
            kernel='linear',
            probability=True,
            random_state=random_state,
            max_iter=svm_max_iter,  # Adaptive
            cache_size=1000  # Larger cache for speed
        ),
        'RBF-SVM': SVC(
            kernel='rbf',
            probability=True,
            gamma='scale',
            random_state=random_state,
            max_iter=svm_max_iter,  # Adaptive
            cache_size=1000
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=rf_estimators,  # Adaptive
            max_depth=rf_max_depth,  # Adaptive
            random_state=random_state,
            n_jobs=1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=xgb_estimators,  # Adaptive
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=1,
            eval_metric='logloss'
        ),
        'MLP': MLPClassifier(
            hidden_layers=(128, 64),
            dropout=0.3,
            learning_rate=0.001,
            max_epochs=100,
            patience=10,
            random_state=random_state
        )
    }
    
    return models
