# Contributing to AutoML-Insight 🤝

Thank you for considering contributing to AutoML-Insight! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, gender, gender identity and expression, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, or nationality.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/AutoML-Insight.git
   cd AutoML-Insight
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/AutoML-Insight.git
   ```

### Set Up Development Environment

1. Create a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. Install pre-commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

## Development Workflow

### Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/fixes

### Make Changes

1. Write your code following [Coding Standards](#coding-standards)
2. Add tests for new functionality
3. Update documentation as needed
4. Run tests locally: `pytest tests/`

### Commit Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add new clustering algorithm"
git commit -m "fix: resolve CV error with small datasets"
git commit -m "docs: update README with new examples"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

### Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line Length**: 120 characters maximum (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings
- **Imports**: Organized into sections (standard library, third-party, local)

### Type Hints

Use type hints for all function signatures:

```python
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    hyperparameters: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a machine learning model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_name: Name of the model to train
        hyperparameters: Optional hyperparameter dictionary
        
    Returns:
        Tuple of (trained_model, metrics_dict)
        
    Raises:
        ValueError: If model_name is not recognized
    """
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for predictions.
    
    This function computes accuracy, precision, recall, and F1-score
    for binary or multiclass classification problems.
    
    Args:
        y_true: Ground truth labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        
    Returns:
        Dictionary containing:
            - 'accuracy': Overall accuracy score
            - 'precision': Macro-averaged precision
            - 'recall': Macro-averaged recall
            - 'f1_score': Macro-averaged F1 score
            
    Raises:
        ValueError: If y_true and y_pred have different lengths
        
    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> print(metrics['accuracy'])
        0.8
    """
    pass
```

### Code Organization

- **One class per file** (for major components)
- **Group related functions** in modules
- **Use meaningful names**: `train_classifier` not `tc`
- **Keep functions small**: Ideally < 50 lines
- **Avoid magic numbers**: Use named constants

### Error Handling

Always use proper error handling:

```python
def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from file."""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")
            
        df = pd.read_csv(filepath)
        
        if df.empty:
            raise ValueError("Dataset is empty")
            
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty or corrupted: {filepath}")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise
```

### Logging

Use the logging module consistently:

```python
import logging

logger = logging.getLogger(__name__)

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process the input data."""
    logger.info(f"Processing data with shape {data.shape}")
    
    try:
        # Processing logic
        logger.debug("Applying preprocessing steps")
        processed = preprocess(data)
        logger.info("Data processing completed successfully")
        return processed
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}", exc_info=True)
        raise
```

## Testing Guidelines

### Test Structure

Tests are located in the `tests/` directory:

```
tests/
├── __init__.py
├── test_preprocess.py
├── test_models.py
├── test_evaluate.py
├── test_integration.py
└── fixtures/
    └── sample_data.py
```

### Writing Tests

Use pytest and follow this structure:

```python
import pytest
import pandas as pd
import numpy as np
from core.preprocess import DataPreprocessor

class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        preprocessor = DataPreprocessor()
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        X_transformed = preprocessor.fit_transform(X, y)
        
        assert X_transformed.shape[0] == len(X)
        assert not X_transformed.isnull().any().any()
    
    def test_invalid_input(self):
        """Test error handling for invalid input."""
        preprocessor = DataPreprocessor()
        
        with pytest.raises(ValueError, match="Input must be a DataFrame"):
            preprocessor.fit_transform(None, None)
```

### Test Coverage

- Aim for **>80% code coverage**
- Test edge cases and error conditions
- Include integration tests for complete workflows

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=core --cov=app --cov=utils --cov-report=html

# Run specific test file
pytest tests/test_preprocess.py

# Run specific test
pytest tests/test_preprocess.py::TestDataPreprocessor::test_fit_transform
```

## Documentation

### Code Documentation

- All public functions/methods must have docstrings
- Complex logic should have inline comments
- Update README.md for user-facing changes
- Update CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/)

### Documentation Updates

When adding features, update:
1. Function/class docstrings
2. README.md (if user-facing)
3. CHANGELOG.md
4. Type hints
5. Examples (if applicable)

## Pull Request Process

### Before Submitting

Checklist:
- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts with main branch
- [ ] Commit messages follow conventions

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Testing
Describe testing done

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] No lint errors
```

### Review Process

1. Maintainer will review within 3-5 days
2. Address review comments
3. Once approved, maintainer will merge
4. Delete your feature branch after merge

## Issue Guidelines

### Reporting Bugs

Use the bug report template:

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen

**Screenshots**
If applicable

**Environment:**
- OS: [e.g., Windows 11]
- Python version: [e.g., 3.11.5]
- Package versions: [e.g., streamlit 1.50.0]

**Additional context**
Any other relevant information
```

### Requesting Features

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
Description of the problem

**Describe the solution you'd like**
Clear description of desired feature

**Describe alternatives considered**
Alternative solutions considered

**Additional context**
Any other relevant information
```

## Development Tips

### Useful Commands

```powershell
# Format code with black
black core/ app/ utils/

# Sort imports with isort
isort core/ app/ utils/

# Check type hints with mypy
mypy core/ app/ utils/

# Run linter
flake8 core/ app/ utils/

# Generate coverage report
pytest tests/ --cov=. --cov-report=html
```

### Debugging

- Use logging instead of print statements
- Set `logging.basicConfig(level=logging.DEBUG)` for verbose output
- Use breakpoints with `import pdb; pdb.set_trace()`
- Check logs in `results/logs/`

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## Questions?

If you have questions:
1. Check existing issues
2. Review documentation
3. Ask in Discussions tab
4. Create a new issue with "question" label

## Thank You!

Your contributions make AutoML-Insight better for everyone. We appreciate your time and effort!

---

**Happy Coding! 🚀**
