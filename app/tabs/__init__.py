"""
Tab modules for AutoML-Insight Dashboard.

Each tab is a self-contained module responsible for rendering
a specific section of the dashboard.
"""

from .data_overview_tab import DataOverviewTab
from .models_tab import ModelsTab
from .professional_automl_tab import ProfessionalAutoMLTab
from .pca_analysis_tab import PCAAnalysisTab
from .explainability_tab import ExplainabilityTab
from .recommendation_tab import RecommendationTab
from .report_tab import ReportTab

__all__ = [
    'DataOverviewTab',
    'ModelsTab',
    'ProfessionalAutoMLTab',
    'PCAAnalysisTab',
    'ExplainabilityTab',
    'RecommendationTab',
    'ReportTab',
]
