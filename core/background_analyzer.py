"""Background Analysis Manager for Expensive Computations.

This module manages background execution of computationally expensive analyses
using ThreadPoolExecutor to prevent UI blocking in Streamlit applications.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, Future
import streamlit as st

logger = logging.getLogger(__name__)


class BackgroundAnalysisManager:
    """Manage background execution of expensive analyses.
    
    This class uses ThreadPoolExecutor to run computationally expensive analyses
    in background threads, preventing Streamlit UI from blocking. Results are
    stored in session_state for retrieval.
    
    Attributes:
        executor: ThreadPoolExecutor for background tasks
        max_workers: Maximum number of concurrent workers
    """
    
    def __init__(self, max_workers=3):
        """Initialize the BackgroundAnalysisManager.
        
        Args:
            max_workers: Maximum number of concurrent background tasks (default: 3)
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
        logger.info(f"BackgroundAnalysisManager initialized with {max_workers} workers")
    
    def schedule_analysis(self, analysis_fn, *args, **kwargs):
        """Schedule an analysis function to run in the background.
        
        Args:
            analysis_fn: Function to execute in background
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Future object that can be used to retrieve results
        """
        try:
            logger.info(f"Scheduling background analysis: {analysis_fn.__name__}")
            future = self.executor.submit(analysis_fn, *args, **kwargs)
            return future
        except Exception as e:
            logger.error(f"Error scheduling analysis: {e}")
            return None
    
    def get_result(self, future, timeout=None):
        """Get result from a background analysis.
        
        Args:
            future: Future object from schedule_analysis
            timeout: Maximum time to wait in seconds (None = no timeout)
            
        Returns:
            Analysis result if complete, None if not ready or error occurred
        """
        if future is None:
            return None
        
        try:
            if self.is_complete(future):
                result = future.result(timeout=timeout)
                logger.info("Background analysis result retrieved successfully")
                return result
            else:
                logger.debug("Background analysis not yet complete")
                return None
        except TimeoutError:
            logger.warning(f"Timeout waiting for background analysis result")
            return None
        except Exception as e:
            logger.error(f"Error retrieving background analysis result: {e}")
            return None
    
    def is_complete(self, future):
        """Check if a background analysis has completed.
        
        Args:
            future: Future object from schedule_analysis
            
        Returns:
            True if analysis is complete, False otherwise
        """
        if future is None:
            return False
        return future.done()
    
    def shutdown(self, wait=True):
        """Shutdown the executor.
        
        Args:
            wait: If True, wait for all tasks to complete before shutting down
        """
        logger.info("Shutting down BackgroundAnalysisManager")
        self.executor.shutdown(wait=wait)


def get_background_result_safe(future, timeout=1.0, placeholder_msg="⏳ Analysis running in background..."):
    """Safely retrieve background analysis result with user feedback.
    
    This is a convenience function for Streamlit apps to check and display
    background analysis results with appropriate user feedback.
    
    Args:
        future: Future object from BackgroundAnalysisManager.schedule_analysis
        timeout: Maximum time to wait for result in seconds (default: 1.0)
        placeholder_msg: Message to display if analysis is still running
        
    Returns:
        Analysis result if complete, None if not ready
    """
    if future is None:
        return None
    
    if not future.done():
        st.info(placeholder_msg)
        return None
    
    try:
        result = future.result(timeout=timeout)
        return result
    except TimeoutError:
        st.info(placeholder_msg)
        return None
    except Exception as e:
        st.warning(f"Analysis failed: {str(e)}")
        logger.error(f"Background analysis error: {e}")
        return None


def schedule_bias_variance_analysis(model, X_train, y_train, X_test, y_test, n_bootstrap=30):
    """Convenience function to schedule bias-variance analysis in background.
    
    Args:
        model: Trained model to analyze
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_bootstrap: Number of bootstrap samples (default: 30)
        
    Returns:
        Future object for retrieving results
    """
    try:
        from core.bias_variance_analyzer import BiasVarianceAnalyzer
        
        # Initialize manager if not in session state
        if 'bg_manager' not in st.session_state:
            st.session_state.bg_manager = BackgroundAnalysisManager()
        
        analyzer = BiasVarianceAnalyzer(n_bootstrap=n_bootstrap)
        future = st.session_state.bg_manager.schedule_analysis(
            analyzer.compute_bias_variance_decomposition,
            model, X_train, y_train, X_test, y_test
        )
        
        logger.info("Bias-variance analysis scheduled in background")
        return future
        
    except Exception as e:
        logger.error(f"Error scheduling bias-variance analysis: {e}")
        return None


def schedule_learning_curves_analysis(model, X, y, train_sizes=None, cv=5):
    """Convenience function to schedule learning curves analysis in background.
    
    Args:
        model: Model to analyze
        X: Feature matrix
        y: Labels
        train_sizes: Training sizes to evaluate (default: [0.1, 0.3, 0.5, 0.7, 0.9])
        cv: Number of CV folds (default: 5)
        
    Returns:
        Future object for retrieving results
    """
    try:
        from core.bias_variance_analyzer import BiasVarianceAnalyzer
        
        # Initialize manager if not in session state
        if 'bg_manager' not in st.session_state:
            st.session_state.bg_manager = BackgroundAnalysisManager()
        
        analyzer = BiasVarianceAnalyzer()
        future = st.session_state.bg_manager.schedule_analysis(
            analyzer.compute_learning_curves,
            model, X, y, train_sizes, cv
        )
        
        logger.info("Learning curves analysis scheduled in background")
        return future
        
    except Exception as e:
        logger.error(f"Error scheduling learning curves analysis: {e}")
        return None
