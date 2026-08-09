# 🚨 Overfitting Detection & Prevention System

## Problem Identified

**Current Issue**: Your app shows 99%+ accuracy because it's evaluating models on the **same data used for training**, causing data leakage.

**Root Cause** (Line 801 in `ui_dashboard.py`):
```python
result = evaluator.evaluate_model(model, X, y, name)  # ❌ Uses ALL data
```

The evaluator does cross-validation on the full dataset, but the model is also trained on the full dataset separately, making results unrealistic.

---

## 🔧 Solution: Multi-Layer Validation System

### 1. **Proper Train/Test Split with Holdout Set**

Split data into:
- **Training Set (70%)**: For model training and cross-validation
- **Holdout Test Set (30%)**: Never seen by models during training
- **Final evaluation**: Report performance on holdout set

### 2. **Overfitting Detection System**

Add automatic warnings when:
- Train accuracy >> Test accuracy (gap > 10%)
- All models achieve > 95% accuracy
- Test set too small (< 100 samples)
- Perfect scores on imbalanced data

### 3. **User Guidance System**

Provide actionable recommendations when overfitting detected.

---

## 📝 Implementation Plan

### **File 1: Create Overfitting Detector**
`core/overfitting_detector.py`

```python
"""
Overfitting Detection and User Guidance System.
Detects unrealistic model performance and provides actionable recommendations.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class OverfittingWarning:
    """Container for overfitting warning details."""
    severity: str  # 'HIGH', 'MEDIUM', 'LOW'
    warning_type: str
    message: str
    recommendations: List[str]


class OverfittingDetector:
    """
    Detects overfitting and data leakage issues.
    Provides user-friendly guidance for resolving issues.
    """
    
    def __init__(self):
        self.warnings = []
    
    def detect_overfitting(
        self,
        train_scores: Dict[str, float],
        test_scores: Dict[str, float],
        cv_scores: Dict[str, List[float]],
        dataset_info: Dict[str, Any]
    ) -> List[OverfittingWarning]:
        """
        Comprehensive overfitting detection.
        
        Args:
            train_scores: Metrics on training set
            test_scores: Metrics on test/holdout set
            cv_scores: Cross-validation scores
            dataset_info: Dataset characteristics
            
        Returns:
            List of overfitting warnings
        """
        self.warnings = []
        
        # Check 1: Train vs Test Gap
        self._check_train_test_gap(train_scores, test_scores)
        
        # Check 2: Unrealistic Perfect Scores
        self._check_perfect_scores(test_scores, dataset_info)
        
        # Check 3: CV Score Variance
        self._check_cv_variance(cv_scores)
        
        # Check 4: Small Test Set
        self._check_test_set_size(dataset_info)
        
        # Check 5: All Models Too Similar
        self._check_model_similarity(test_scores)
        
        # Check 6: Imbalanced Data with Perfect Scores
        self._check_imbalanced_perfect(test_scores, dataset_info)
        
        return self.warnings
    
    def _check_train_test_gap(
        self, 
        train_scores: Dict[str, float], 
        test_scores: Dict[str, float]
    ):
        """Check if training accuracy significantly exceeds test accuracy."""
        for metric in ['accuracy', 'f1_macro']:
            if metric in train_scores and metric in test_scores:
                gap = train_scores[metric] - test_scores[metric]
                
                if gap > 0.15:  # 15% gap
                    self.warnings.append(OverfittingWarning(
                        severity='HIGH',
                        warning_type='TRAIN_TEST_GAP',
                        message=f"⚠️ **Severe Overfitting Detected**: Training {metric} ({train_scores[metric]:.2%}) is {gap:.1%} higher than test {metric} ({test_scores[metric]:.2%}).",
                        recommendations=[
                            "Your model memorized the training data instead of learning patterns",
                            "Collect more diverse training data (aim for 10x current size)",
                            "Use regularization: Set max_depth=5 for trees, C=0.1 for SVM",
                            "Try simpler models: LogisticRegression instead of ensemble methods",
                            "Apply feature selection: Use only top 10-20 most important features"
                        ]
                    ))
                elif gap > 0.10:  # 10% gap
                    self.warnings.append(OverfittingWarning(
                        severity='MEDIUM',
                        warning_type='TRAIN_TEST_GAP',
                        message=f"⚠️ **Moderate Overfitting**: Training {metric} is {gap:.1%} higher than test.",
                        recommendations=[
                            "Consider using cross-validation for more reliable estimates",
                            "Try regularization to prevent overfitting",
                            "Reduce model complexity (fewer features or simpler models)"
                        ]
                    ))
    
    def _check_perfect_scores(
        self, 
        test_scores: Dict[str, float],
        dataset_info: Dict[str, Any]
    ):
        """Check for unrealistic perfect or near-perfect scores."""
        accuracy = test_scores.get('accuracy', 0)
        n_samples = dataset_info.get('n_samples', 1000)
        
        if accuracy > 0.99 and n_samples < 500:
            self.warnings.append(OverfittingWarning(
                severity='HIGH',
                warning_type='PERFECT_SCORE_SMALL_DATA',
                message=f"🚨 **Data Leakage Suspected**: {accuracy:.1%} accuracy on {n_samples} samples is unrealistic for most real-world problems.",
                recommendations=[
                    "**Verify your train/test split is correct** - Check that test data wasn't used in training",
                    "**Check for data leakage** - Look for features that directly reveal the target (e.g., 'customer_churned' predicting churn)",
                    "**Try a different random seed** - If accuracy drops significantly, original split was lucky",
                    "**Collect more data** - Small datasets lead to unstable results",
                    "**Simplify the problem** - Your task might be too easy or data too clean"
                ]
            ))
        elif accuracy > 0.95:
            self.warnings.append(OverfittingWarning(
                severity='MEDIUM',
                warning_type='VERY_HIGH_SCORE',
                message=f"⚠️ **Unusually High Accuracy**: {accuracy:.1%} is excellent but verify it's genuine.",
                recommendations=[
                    "Test on completely new data from a different time period or source",
                    "Check if your problem is unusually easy (e.g., spam detection with obvious keywords)",
                    "Verify feature engineering didn't introduce target leakage"
                ]
            ))
    
    def _check_cv_variance(self, cv_scores: Dict[str, List[float]]):
        """Check if cross-validation scores have suspiciously low variance."""
        for metric, scores in cv_scores.items():
            if len(scores) >= 3:
                std = np.std(scores)
                mean = np.mean(scores)
                
                if std < 0.01 and mean > 0.9:  # Less than 1% variation with high scores
                    self.warnings.append(OverfittingWarning(
                        severity='MEDIUM',
                        warning_type='LOW_CV_VARIANCE',
                        message=f"🤔 **Suspiciously Consistent Scores**: {metric} has {std:.3f} standard deviation across folds.",
                        recommendations=[
                            "Very low variance suggests the problem might be too easy",
                            "Check if all folds are identical (data duplication issue)",
                            "Verify stratification is working correctly"
                        ]
                    ))
    
    def _check_test_set_size(self, dataset_info: Dict[str, Any]):
        """Check if test set is too small for reliable evaluation."""
        n_test = dataset_info.get('n_test_samples', 0)
        n_classes = dataset_info.get('n_classes', 2)
        
        min_samples_per_class = n_test / n_classes if n_classes > 0 else n_test
        
        if n_test < 30:
            self.warnings.append(OverfittingWarning(
                severity='HIGH',
                warning_type='SMALL_TEST_SET',
                message=f"⚠️ **Test Set Too Small**: Only {n_test} samples in test set is insufficient.",
                recommendations=[
                    "Collect more data - aim for at least 100 test samples",
                    "Use cross-validation instead of single train/test split",
                    "Results are highly unreliable with this sample size"
                ]
            ))
        elif min_samples_per_class < 10:
            self.warnings.append(OverfittingWarning(
                severity='MEDIUM',
                warning_type='FEW_SAMPLES_PER_CLASS',
                message=f"⚠️ **Imbalanced Test Set**: Only ~{min_samples_per_class:.0f} samples per class.",
                recommendations=[
                    "Use stratified sampling to ensure each class is represented",
                    "Consider oversampling minority classes",
                    "Aim for at least 30 samples per class in test set"
                ]
            ))
    
    def _check_model_similarity(self, test_scores: Dict[str, float]):
        """Check if all models achieve suspiciously similar scores."""
        # This would need multiple model scores, simplified for now
        pass
    
    def _check_imbalanced_perfect(
        self,
        test_scores: Dict[str, float],
        dataset_info: Dict[str, Any]
    ):
        """Check for perfect scores on imbalanced data."""
        accuracy = test_scores.get('accuracy', 0)
        class_balance = dataset_info.get('class_balance', {})
        
        if class_balance:
            # Calculate imbalance ratio
            counts = list(class_balance.values())
            if len(counts) >= 2:
                imbalance_ratio = max(counts) / min(counts)
                
                if accuracy > 0.95 and imbalance_ratio > 3:
                    self.warnings.append(OverfittingWarning(
                        severity='HIGH',
                        warning_type='IMBALANCED_PERFECT',
                        message=f"🚨 **Misleading Accuracy on Imbalanced Data**: {accuracy:.1%} accuracy with {imbalance_ratio:.1f}:1 class imbalance.",
                        recommendations=[
                            f"Accuracy is misleading - a dummy model predicting majority class would get {max(counts):.1%}",
                            "Use F1-score, Precision, or Recall instead of accuracy",
                            "Check confusion matrix - model might just predict majority class",
                            "Apply SMOTE or class weighting to handle imbalance",
                            "Consider using balanced_accuracy_score"
                        ]
                    ))
    
    def get_user_guidance(self) -> Dict[str, Any]:
        """
        Generate user-friendly guidance document.
        
        Returns:
            Dictionary with warnings, severity, and recommendations
        """
        if not self.warnings:
            return {
                'has_issues': False,
                'message': "✅ No overfitting issues detected. Your model performance looks realistic.",
                'severity': 'NONE'
            }
        
        # Determine overall severity
        severities = [w.severity for w in self.warnings]
        overall_severity = 'HIGH' if 'HIGH' in severities else 'MEDIUM' if 'MEDIUM' in severities else 'LOW'
        
        return {
            'has_issues': True,
            'overall_severity': overall_severity,
            'warning_count': len(self.warnings),
            'warnings': [
                {
                    'severity': w.severity,
                    'type': w.warning_type,
                    'message': w.message,
                    'recommendations': w.recommendations
                }
                for w in self.warnings
            ],
            'summary': self._generate_summary()
        }
    
    def _generate_summary(self) -> str:
        """Generate executive summary of issues."""
        high_count = sum(1 for w in self.warnings if w.severity == 'HIGH')
        medium_count = sum(1 for w in self.warnings if w.severity == 'MEDIUM')
        
        if high_count > 0:
            return f"🚨 **Critical Issues Found**: {high_count} high-severity and {medium_count} medium-severity overfitting warnings. Your model performance is likely unrealistic. Please review recommendations below."
        elif medium_count > 0:
            return f"⚠️ **Moderate Concerns**: {medium_count} potential overfitting issues detected. Results may not generalize well to new data."
        else:
            return "✅ Minor concerns detected. Review recommendations for best practices."


def create_overfitting_report(
    train_accuracy: float,
    test_accuracy: float,
    cv_scores: List[float],
    dataset_size: int,
    n_classes: int,
    class_balance: Dict[str, float]
) -> Dict[str, Any]:
    """
    Convenience function to create overfitting report.
    
    Example usage:
        report = create_overfitting_report(
            train_accuracy=0.99,
            test_accuracy=0.99,
            cv_scores=[0.98, 0.99, 0.99, 0.98, 0.99],
            dataset_size=1372,
            n_classes=2,
            class_balance={0: 0.56, 1: 0.44}
        )
    """
    detector = OverfittingDetector()
    
    warnings = detector.detect_overfitting(
        train_scores={'accuracy': train_accuracy},
        test_scores={'accuracy': test_accuracy},
        cv_scores={'accuracy': cv_scores},
        dataset_info={
            'n_samples': dataset_size,
            'n_test_samples': int(dataset_size * 0.3),
            'n_classes': n_classes,
            'class_balance': class_balance
        }
    )
    
    return detector.get_user_guidance()
```

### **File 2: Update Classification Evaluator**
`core/evaluate_cls.py` - Add train/test split support

Add this method to `ClassificationEvaluator`:

```python
def evaluate_with_holdout(
    self,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """
    Evaluate with proper train/test split.
    
    Args:
        model: Untrained model instance
        X_train, y_train: Training data
        X_test, y_test: Held-out test data
        model_name: Model identifier
        
    Returns:
        Dictionary with train scores, test scores, and overfitting warnings
    """
    from core.overfitting_detector import OverfittingDetector
    
    # 1. Cross-validation on training set only
    cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
    cv_scores = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring={'accuracy': 'accuracy', 'f1_macro': 'f1_macro'},
        return_estimator=False
    )
    
    # 2. Train on full training set
    model.fit(X_train, y_train)
    
    # 3. Evaluate on training set (to detect overfitting)
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    
    # 4. Evaluate on test set (true performance)
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    # 5. Detect overfitting
    detector = OverfittingDetector()
    warnings = detector.detect_overfitting(
        train_scores={'accuracy': train_accuracy, 'f1_macro': train_f1},
        test_scores={'accuracy': test_accuracy, 'f1_macro': test_f1},
        cv_scores={'accuracy': cv_scores['test_accuracy'].tolist()},
        dataset_info={
            'n_samples': len(X_train) + len(X_test),
            'n_test_samples': len(X_test),
            'n_classes': len(np.unique(y_train)),
            'class_balance': {
                i: np.sum(y_test == i) / len(y_test) 
                for i in np.unique(y_test)
            }
        }
    )
    
    overfitting_report = detector.get_user_guidance()
    
    return {
        'model_name': model_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_accuracy_mean': np.mean(cv_scores['test_accuracy']),
        'cv_accuracy_std': np.std(cv_scores['test_accuracy']),
        'overfitting_warnings': overfitting_report,
        'trained_model': model,
        'test_predictions': y_test_pred,
        'test_true': y_test
    }
```

### **File 3: Update Dashboard**
`app/ui_dashboard.py` - Fix line 801 and add warnings

**Change 1** (around line 727):
```python
# OLD:
X_processed, y_processed = preprocessor.fit_transform(X, y)

# NEW: Split into train/test AFTER preprocessing
from sklearn.model_selection import train_test_split
X_processed, y_processed = preprocessor.fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed, 
    test_size=0.3,  # 30% holdout
    stratify=y_processed,
    random_state=st.session_state.random_seed
)
st.session_state.X_train = X_train
st.session_state.X_test = X_test
st.session_state.y_train = y_train
st.session_state.y_test = y_test
```

**Change 2** (around line 801):
```python
# OLD:
result = evaluator.evaluate_model(model, X, y, name)

# NEW: Use holdout evaluation
result = evaluator.evaluate_with_holdout(
    model, 
    st.session_state.X_train,
    st.session_state.y_train,
    st.session_state.X_test,
    st.session_state.y_test,
    name
)
```

**Change 3** (in render_classification_results):
```python
# Add overfitting warnings display
def render_classification_results(self):
    st.subheader("🤖 Classification Models")
    
    if not st.session_state.results:
        st.warning("No results yet. Train models first.")
        return
    
    # NEW: Check for overfitting warnings
    high_severity_warnings = []
    for name, result in st.session_state.results.items():
        if 'overfitting_warnings' in result:
            warnings = result['overfitting_warnings']
            if warnings.get('has_issues') and warnings.get('overall_severity') == 'HIGH':
                high_severity_warnings.append((name, warnings))
    
    # Display critical warnings at the top
    if high_severity_warnings:
        st.error("🚨 **Critical Overfitting Issues Detected**")
        
        for model_name, warnings in high_severity_warnings:
            with st.expander(f"⚠️ Issues with {model_name}", expanded=True):
                st.markdown(warnings['summary'])
                
                for warning in warnings['warnings']:
                    if warning['severity'] == 'HIGH':
                        st.markdown(f"**{warning['message']}**")
                        st.markdown("**What to do:**")
                        for rec in warning['recommendations']:
                            st.markdown(f"- {rec}")
    
    # Show leaderboard with train/test comparison
    st.markdown("### 📊 Model Leaderboard")
    
    leaderboard_data = []
    for name, result in st.session_state.results.items():
        leaderboard_data.append({
            'Model': name,
            'Train Acc': f"{result.get('train_accuracy', 0):.3f}",
            'Test Acc': f"{result.get('test_accuracy', 0):.3f}",
            'Overfitting Gap': f"{result.get('train_accuracy', 0) - result.get('test_accuracy', 0):.3f}",
            'CV Mean': f"{result.get('cv_accuracy_mean', 0):.3f}",
            'CV Std': f"{result.get('cv_accuracy_std', 0):.3f}"
        })
    
    df_leaderboard = pd.DataFrame(leaderboard_data)
    st.dataframe(df_leaderboard, use_container_width=True)
    
    # Add explanation
    st.info("""
    **How to Read This Table:**
    - **Train Acc**: Performance on training data (should be high)
    - **Test Acc**: Performance on unseen test data (TRUE performance metric)
    - **Overfitting Gap**: Train - Test (should be < 0.10)
    - **CV Mean/Std**: Cross-validation scores (reliability indicator)
    
    ⚠️ If Gap > 0.10, your model is overfitting!
    """)
```

---

## 📚 User Documentation to Add

Create `docs/OVERFITTING_GUIDE.md`:

```markdown
# Understanding Model Performance in AutoML-Insight

## 🎯 What is Overfitting?

Overfitting occurs when your model memorizes the training data instead of learning general patterns. It's like a student memorizing answers without understanding concepts.

## 🚨 Red Flags

### 1. **Training Accuracy >> Test Accuracy**
- Train: 99% | Test: 75% → **SEVERE OVERFITTING**
- Train: 92% | Test: 85% → **MILD OVERFITTING**
- Train: 87% | Test: 85% → **HEALTHY**

### 2. **All Models Achieve >95% Accuracy**
Suggests data leakage or problem too easy.

### 3. **Perfect Scores on Small Datasets**
99% accuracy with <500 samples is usually unrealistic.

## 🔧 How to Fix Overfitting

### **Option 1: Collect More Data** (BEST)
- Aim for 10x your current dataset size
- Ensure data diversity

### **Option 2: Simplify Models**
- Use LogisticRegression instead of XGBoost
- Reduce max_depth for tree models
- Use fewer features

### **Option 3: Add Regularization**
- For trees: `max_depth=5`, `min_samples_split=10`
- For SVM: `C=0.1`
- For neural nets: `alpha=0.01`

### **Option 4: Feature Selection**
- Keep only top 10-20 features
- Remove highly correlated features

### **Option 5: Cross-Validation**
- Use 5-fold or 10-fold CV
- Check consistency across folds

## 📊 Interpreting Your Results

### Good Results
```
Model: RandomForest
Train Acc: 0.87
Test Acc: 0.85
Gap: 0.02 ✅
```

### Overfitting
```
Model: RandomForest
Train Acc: 0.99
Test Acc: 0.76
Gap: 0.23 ❌
```

## 🎓 When to Trust Your Model

1. ✅ Train/Test gap < 10%
2. ✅ Test accuracy reasonable for your domain
3. ✅ CV std dev < 5%
4. ✅ Test set has 100+ samples
5. ✅ Results validated on new data

## 💡 Real-World Expectations

| Task | Realistic Accuracy |
|------|-------------------|
| Email Spam | 90-98% |
| Medical Diagnosis | 75-90% |
| Customer Churn | 70-85% |
| Fraud Detection | 85-95% |
| Image Recognition | 90-99% |

If your accuracy is higher, verify it's genuine!
```

---

## 🎯 Summary

1. **Root Cause**: App evaluates on all data, causing data leakage
2. **Fix**: Implement proper train/test split with holdout set
3. **Detection**: Add overfitting detector to warn users
4. **Guidance**: Provide actionable recommendations
5. **Education**: Help users understand realistic performance

This will make your app production-grade and trustworthy! 🚀
