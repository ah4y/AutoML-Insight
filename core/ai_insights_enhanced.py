"""
Enhanced AI-Powered Insights Engine for AutoML.

This module provides comprehensive, dynamic, dataset-specific insights using LLMs.
Supports multiple providers: Groq (fast), OpenAI (powerful), Gemini (alternative).
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Global cache for AI responses
_ai_response_cache: Dict[str, str] = {}


@dataclass
class EnhancedDatasetStatistics:
    """Container for comprehensive dataset statistics to send to LLM."""

    # Basic info
    n_samples: int
    n_features: int
    n_numeric: int
    n_categorical: int
    target_type: str
    n_classes: Optional[int]
    class_balance: Optional[Dict[str, float]]
    missing_rate: float
    feature_correlations: List[Tuple[str, str, float]]
    top_features: List[str]
    data_quality_score: float

    # Enhanced metrics
    outlier_rate: float
    skewness_summary: Dict[str, float]  # high_skew_features -> skewness values
    variance_summary: Dict[str, int]  # low/high variance feature counts
    categorical_cardinality: Dict[str, int]  # feature -> unique values
    data_types_detailed: Dict[str, str]  # feature -> detailed type
    correlation_with_target: List[Tuple[str, float]]  # top correlated with target
    feature_importance_proxy: Dict[str, float]  # simple importance estimates
    potential_issues: List[str]  # detected data issues
    domain_insights: Dict[str, Any]  # domain-specific patterns
    dataset_complexity: str  # simple, moderate, complex
    memory_usage_mb: float


class EnhancedAIInsightEngine:
    """
    Enhanced AI-powered analysis engine for AutoML.

    Generates comprehensive, dynamic insights based on dataset characteristics using LLMs.
    Never static - every insight is tailored to your specific data.
    """

    def __init__(self, provider: str = "groq", model: Optional[str] = None, temperature: float = 0.3):
        """
        Initialize AI insight engine.

        Args:
            provider: LLM provider ('groq', 'openai', 'gemini')
            model: Specific model name (uses defaults if None)
            temperature: Response creativity (0.0-1.0, lower = more deterministic)
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)

        # Load API keys from environment
        self.api_keys = {
            "groq": os.getenv("GROQ_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
        }

        # Default models for each provider
        self.default_models = {
            "groq": "llama-3.3-70b-versatile",  # Fast and capable (updated model)
            "openai": "gpt-4o-mini",  # Cost-effective GPT-4
            "gemini": "gemini-1.5-flash",  # Fast Gemini
        }

        self.model = model or self.default_models.get(self.provider)

        # Initialize the appropriate client
        self._init_client()

        self.logger.info(f"Enhanced AI Insight Engine initialized: {self.provider}/{self.model}")

    def _init_client(self) -> None:
        """Initialize LLM client based on provider."""
        api_key = self.api_keys.get(self.provider)

        if not api_key:
            raise ValueError(
                f"API key for {self.provider} not found. " f"Please set {self.provider.upper()}_API_KEY in .env file"
            )

        if self.provider == "groq":
            from groq import Groq

            self.client = Groq(api_key=api_key)

        elif self.provider == "openai":
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)

        elif self.provider == "gemini":
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def analyze_dataset_comprehensive(
        self, data: pd.DataFrame, target_col: Optional[str] = None, task_type: str = "classification"
    ) -> EnhancedDatasetStatistics:
        """
        Extract comprehensive statistics and insights from dataset.

        Args:
            data: Input DataFrame
            target_col: Target column name
            task_type: 'classification' or 'regression'

        Returns:
            EnhancedDatasetStatistics object with comprehensive metrics
        """
        n_samples, n_features = data.shape

        # Separate features and target
        if target_col and target_col in data.columns:
            features_data = data.drop(columns=[target_col])
            target_data = data[target_col]
        else:
            features_data = data
            target_data = None

        # Feature types
        numeric_cols = features_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = features_data.select_dtypes(exclude=[np.number]).columns.tolist()

        # Enhanced type analysis
        data_types_detailed = {}
        for col in features_data.columns:
            dtype = str(features_data[col].dtype)
            unique_count = features_data[col].nunique()
            if dtype.startswith("int") or dtype.startswith("float"):
                if unique_count <= 10:
                    data_types_detailed[col] = f"Numeric-Discrete ({unique_count} values)"
                else:
                    data_types_detailed[col] = f"Numeric-Continuous ({unique_count} values)"
            else:
                data_types_detailed[col] = f"Categorical ({unique_count} categories)"

        # Target analysis
        n_classes = None
        class_balance = None
        if target_data is not None:
            if task_type == "classification":
                n_classes = target_data.nunique()
                value_counts = target_data.value_counts(normalize=True)
                class_balance = {str(k): float(v) for k, v in value_counts.head(10).items()}  # Limit to top 10

        # Missing values analysis
        missing_rate = float(data.isnull().sum().sum() / (n_samples * n_features))

        # Outlier detection (using IQR method)
        outlier_count = 0
        if numeric_cols:
            for col in numeric_cols:
                try:
                    Q1 = features_data[col].quantile(0.25)
                    Q3 = features_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_count += ((features_data[col] < lower_bound) | (features_data[col] > upper_bound)).sum()
                except Exception as e:
                    self.logger.debug(f"Outlier detection skipped for column '{col}': {e}")
                    continue

        outlier_rate = float(outlier_count / n_samples) if n_samples > 0 else 0

        # Skewness analysis
        skewness_summary = {}
        if numeric_cols:
            for col in numeric_cols:
                try:
                    skew = features_data[col].skew()
                    if not pd.isna(skew) and abs(skew) > 1:  # Only include highly skewed features
                        skewness_summary[col] = float(skew)
                except Exception as e:
                    self.logger.debug(f"Skewness calculation skipped for column '{col}': {e}")
                    continue

        # Variance analysis
        variance_summary = {}
        if numeric_cols:
            try:
                variances = features_data[numeric_cols].var()
                # Normalize variances to 0-1 scale for comparison
                if len(variances) > 0 and variances.max() > 0:
                    var_normalized = variances / variances.max()

                    # Low variance features (< 0.1 of max variance)
                    low_var = var_normalized[var_normalized < 0.1]
                    if not low_var.empty:
                        variance_summary["low_variance"] = len(low_var)

                    # High variance features (> 0.8 of max variance)
                    high_var = var_normalized[var_normalized > 0.8]
                    if not high_var.empty:
                        variance_summary["high_variance"] = len(high_var)
            except Exception as e:
                self.logger.debug(f"Variance analysis failed, continuing without it: {e}")
                variance_summary = {}

        # Categorical cardinality analysis
        categorical_cardinality = {}
        for col in categorical_cols:
            try:
                cardinality = features_data[col].nunique()
                if cardinality > 1:  # Exclude constant features
                    categorical_cardinality[col] = cardinality
            except Exception as e:
                self.logger.debug(f"Cardinality calculation skipped for column '{col}': {e}")
                continue

        # Feature correlations (top 5)
        correlations = []
        if len(numeric_cols) > 1:
            try:
                corr_matrix = features_data[numeric_cols].corr().abs()
                # Get upper triangle
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                # Find top correlations
                corr_pairs = []
                for col in upper.columns:
                    for row in upper.index:
                        val = upper.loc[row, col]
                        if not pd.isna(val):
                            corr_pairs.append((col, row, float(val)))
                correlations = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:5]
            except Exception as e:
                self.logger.debug(f"Feature correlation analysis failed, continuing without it: {e}")
                correlations = []

        # Correlation with target
        correlation_with_target = []
        if target_data is not None and task_type == "regression" and numeric_cols:
            try:
                target_corrs = []
                for col in numeric_cols:
                    corr = features_data[col].corr(target_data)
                    if not pd.isna(corr):
                        target_corrs.append((col, float(abs(corr))))
                correlation_with_target = sorted(target_corrs, key=lambda x: x[1], reverse=True)[:5]
            except Exception as e:
                self.logger.debug(f"Target correlation analysis failed, continuing without it: {e}")
                correlation_with_target = []

        # Simple feature importance proxy (variance for numeric, cardinality for categorical)
        feature_importance_proxy = {}
        if numeric_cols:
            try:
                variances = features_data[numeric_cols].var()
                max_var = variances.max() if len(variances) > 0 and variances.max() > 0 else 1
                for col in numeric_cols:
                    if not pd.isna(variances[col]):
                        feature_importance_proxy[col] = float(variances[col] / max_var)
            except Exception as e:
                self.logger.debug(f"Numeric feature-importance proxy failed, continuing without it: {e}")

        for col in categorical_cols:
            try:
                # Higher cardinality = potentially more informative
                cardinality = features_data[col].nunique()
                max_cardinality = max(categorical_cardinality.values()) if categorical_cardinality else 1
                feature_importance_proxy[col] = float(min(cardinality / max_cardinality, 1.0))
            except Exception as e:
                self.logger.debug(f"Categorical feature-importance proxy skipped for column '{col}': {e}")
                continue

        # Top features (by importance proxy)
        top_features = []
        if feature_importance_proxy:
            top_features_sorted = sorted(feature_importance_proxy.items(), key=lambda x: x[1], reverse=True)[:5]
            top_features = [feat[0] for feat in top_features_sorted]

        # Detect potential issues
        potential_issues = []

        if missing_rate > 0.1:
            potential_issues.append(f"High missing data rate ({missing_rate:.1%})")

        if outlier_rate > 0.05:
            potential_issues.append(f"High outlier rate ({outlier_rate:.1%})")

        if len(skewness_summary) > len(numeric_cols) * 0.5:
            potential_issues.append(f"Many skewed features ({len(skewness_summary)} highly skewed)")

        if variance_summary.get("low_variance", 0) > 3:
            potential_issues.append(f"Multiple low-variance features ({variance_summary['low_variance']})")

        high_cardinality_cats = [col for col, card in categorical_cardinality.items() if card > n_samples * 0.1]
        if high_cardinality_cats:
            potential_issues.append(f"High-cardinality categorical features: {', '.join(high_cardinality_cats[:3])}")

        if class_balance and len(class_balance) > 1:
            imbalance_ratio = max(class_balance.values()) / min(class_balance.values())
            if imbalance_ratio > 10:
                potential_issues.append(f"Severe class imbalance (ratio {imbalance_ratio:.1f}:1)")

        # Domain insights (pattern detection)
        domain_insights = {}

        # Time series detection
        time_cols = [col for col in data.columns if "time" in col.lower() or "date" in col.lower()]
        if time_cols:
            domain_insights["temporal_features"] = time_cols

        # ID/identifier detection
        id_cols = [col for col in data.columns if "id" in col.lower() and data[col].nunique() > n_samples * 0.8]
        if id_cols:
            domain_insights["identifier_features"] = id_cols

        # Text feature detection
        text_cols = []
        for col in categorical_cols:
            try:
                avg_length = features_data[col].astype(str).str.len().mean()
                if avg_length > 20:  # Likely text if average length > 20 characters
                    text_cols.append(col)
            except Exception as e:
                self.logger.debug(f"Text-feature detection skipped for column '{col}': {e}")
                continue
        if text_cols:
            domain_insights["text_features"] = text_cols

        # Dataset complexity assessment
        complexity_score = 0
        if n_features > 50:
            complexity_score += 1
        if n_samples < 1000:
            complexity_score += 1
        if len(categorical_cols) > n_features * 0.5:
            complexity_score += 1
        if missing_rate > 0.1:
            complexity_score += 1
        if len(skewness_summary) > 5:
            complexity_score += 1

        if complexity_score >= 3:
            dataset_complexity = "complex"
        elif complexity_score >= 1:
            dataset_complexity = "moderate"
        else:
            dataset_complexity = "simple"

        # Memory usage
        try:
            memory_usage_mb = float(data.memory_usage(deep=True).sum() / (1024 * 1024))
        except Exception as e:
            self.logger.debug(f"Memory usage calculation failed: {e}")
            memory_usage_mb = 0.0

        # Data quality score (0-100)
        quality_score = 100.0
        quality_score -= missing_rate * 30  # Penalize missing data
        quality_score -= outlier_rate * 20  # Penalize outliers
        quality_score -= min(len(potential_issues) * 10, 30)  # Penalize issues

        if class_balance and len(class_balance) > 1:
            imbalance_ratio = max(class_balance.values()) / min(class_balance.values())
            if imbalance_ratio > 10:
                quality_score -= 20
            elif imbalance_ratio > 5:
                quality_score -= 10

        return EnhancedDatasetStatistics(
            # Basic info
            n_samples=n_samples,
            n_features=n_features - (1 if target_col else 0),
            n_numeric=len(numeric_cols),
            n_categorical=len(categorical_cols),
            target_type=task_type,
            n_classes=n_classes,
            class_balance=class_balance,
            missing_rate=missing_rate,
            feature_correlations=correlations,
            top_features=top_features,
            data_quality_score=max(0, quality_score),
            # Enhanced metrics
            outlier_rate=outlier_rate,
            skewness_summary=skewness_summary,
            variance_summary=variance_summary,
            categorical_cardinality=categorical_cardinality,
            data_types_detailed=data_types_detailed,
            correlation_with_target=correlation_with_target,
            feature_importance_proxy=feature_importance_proxy,
            potential_issues=potential_issues,
            domain_insights=domain_insights,
            dataset_complexity=dataset_complexity,
            memory_usage_mb=memory_usage_mb,
        )

    def generate_comprehensive_insights(
        self, stats: EnhancedDatasetStatistics, context: str = "initial_analysis"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive AI-powered insights based on dataset statistics.
        Falls back to enhanced rule-based insights if AI fails.

        Args:
            stats: Enhanced dataset statistics object
            context: Analysis context ('initial_analysis', 'model_selection', 'advanced_insights')

        Returns:
            Dictionary with comprehensive insights
        """
        prompt = self._build_comprehensive_prompt(stats, context)

        try:
            response = self._call_llm_with_retry(prompt)
            insights = self._parse_ai_response(response)
            insights["_source"] = "ai"  # Mark as AI-generated
            insights["_quality_score"] = stats.data_quality_score
            return insights

        except Exception as e:
            error_msg = str(e)
            self.logger.warning(f"AI insights failed: {error_msg}")

            # Use enhanced rule-based fallback
            insights = self._generate_enhanced_fallback_insights(stats)
            insights["_source"] = "enhanced_rules"  # Mark as enhanced rule-based
            insights["_quality_score"] = stats.data_quality_score

            # Add informative message for user
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                insights["_notice"] = (
                    "⚠️ AI rate limit reached. Showing enhanced rule-based analysis. Upgrade your API plan or try again later for AI insights."
                )
            else:
                insights["_notice"] = "⚠️ AI analysis unavailable. Showing comprehensive rule-based analysis."

            return insights

    def _build_comprehensive_prompt(self, stats: EnhancedDatasetStatistics, context: str) -> str:
        """Build comprehensive prompt for LLM based on statistics and context."""

        # Build comprehensive dataset profile
        base_info = f"""COMPREHENSIVE DATASET ANALYSIS REQUEST

DATASET OVERVIEW:
- Samples: {stats.n_samples:,}
- Features: {stats.n_features} ({stats.n_numeric} numeric, {stats.n_categorical} categorical)
- Task Type: {stats.target_type}
- Complexity Level: {stats.dataset_complexity}
- Memory Usage: {stats.memory_usage_mb:.1f} MB
- Data Quality Score: {stats.data_quality_score:.1f}/100
- Outlier Rate: {stats.outlier_rate:.1%}
- Missing Data Rate: {stats.missing_rate:.1%}"""

        # Add target information
        if stats.n_classes:
            base_info += f"\n- Classes: {stats.n_classes}"
            if stats.class_balance:
                # Show class distribution for imbalanced datasets
                top_classes = list(stats.class_balance.items())[:3]
                class_info = ", ".join([f"{k}:{v:.1%}" for k, v in top_classes])
                base_info += f"\n- Class Distribution: {class_info}"

        # Add comprehensive data quality analysis
        if stats.potential_issues:
            base_info += "\n\nDATA QUALITY ISSUES DETECTED:"
            for i, issue in enumerate(stats.potential_issues[:6], 1):  # Limit to top 6 issues
                base_info += f"\n{i}. {issue}"

        # Add feature insights
        if stats.top_features:
            base_info += f"\n\nTOP FEATURES (by importance): {', '.join(stats.top_features[:5])}"

        # Add statistical insights
        if stats.skewness_summary:
            skewed_features = list(stats.skewness_summary.keys())[:3]
            base_info += f"\n\nHIGHLY SKEWED FEATURES: {', '.join(skewed_features)}"

        # Add correlation insights
        if stats.feature_correlations:
            top_corr = stats.feature_correlations[0]  # Highest correlation
            base_info += f"\n\nSTRONGEST FEATURE CORRELATION: {top_corr[0]} ↔ {top_corr[1]} (r={top_corr[2]:.3f})"

        if stats.correlation_with_target:
            top_target_corr = stats.correlation_with_target[0]
            base_info += f"\nSTRONGEST TARGET CORRELATION: {top_target_corr[0]} (r={top_target_corr[1]:.3f})"

        # Add domain insights
        if stats.domain_insights:
            insights_summary = []
            if "temporal_features" in stats.domain_insights:
                insights_summary.append(f"Time series features: {len(stats.domain_insights['temporal_features'])}")
            if "text_features" in stats.domain_insights:
                insights_summary.append(f"Text features: {len(stats.domain_insights['text_features'])}")
            if "identifier_features" in stats.domain_insights:
                insights_summary.append(f"ID columns: {len(stats.domain_insights['identifier_features'])}")

            if insights_summary:
                base_info += f"\n\nDOMAIN PATTERNS: {', '.join(insights_summary)}"

        # Context-specific prompts
        if context == "initial_analysis":
            prompt = base_info + """\n\nProvide comprehensive initial analysis in JSON format:
{
  "dataset_overview": "2-3 sentence expert summary of dataset characteristics and ML suitability",
  "key_strengths": ["List 3-4 major dataset advantages for ML"],
  "critical_challenges": ["List 3-4 main obstacles and data issues"],
  "data_quality_assessment": "Detailed assessment of data quality with specific numerical thresholds",
  "preprocessing_strategy": ["List 5-6 specific preprocessing steps with technical details"],
  "recommended_models": ["List 4-5 ML algorithms best suited for this data with brief rationale"],
  "feature_engineering_opportunities": ["List 3-4 specific feature engineering techniques"],
  "statistical_insights": ["List 3-4 key statistical observations about the data"],
  "risk_factors": ["List 3 main risks that could impact model performance"],
  "expected_performance": "Realistic ML performance expectations with confidence level and reasoning",
  "next_steps": ["List 3-4 immediate action items for data scientist"]
}

Focus on actionable, dataset-specific insights with technical depth."""

        elif context == "model_selection":
            prompt = base_info + """\n\nProvide expert model selection guidance in JSON format:
{
  "tier1_models": ["Top 3 algorithms with highest success probability and rationale"],
  "tier2_models": ["2-3 alternative algorithms worth testing"],
  "avoid_models": ["1-2 algorithms to avoid with technical reasoning"],
  "hyperparameter_priorities": ["Key hyperparameters to focus optimization on"],
  "validation_strategy": "Optimal cross-validation approach with fold recommendations",
  "ensemble_opportunities": ["Specific ensemble techniques that would work well"],
  "computational_considerations": "Runtime and memory expectations",
  "performance_metrics": ["Most appropriate evaluation metrics for this dataset"]
}"""

        elif context == "advanced_insights":
            prompt = base_info + """\n\nProvide advanced data science insights in JSON format:
{
  "advanced_statistical_analysis": ["Deep statistical properties and their ML implications"],
  "feature_space_analysis": "Assessment of dimensionality, manifold structure, and reduction needs",
  "distribution_insights": ["Key insights about feature and target distributions"],
  "correlation_analysis": ["Advanced correlation patterns and their significance"],
  "domain_specific_recommendations": ["Domain expertise-based suggestions"],
  "advanced_preprocessing": ["Sophisticated preprocessing techniques beyond basics"],
  "model_interpretability": ["Strategies for explaining model decisions with this data"],
  "production_deployment": ["Key considerations for productionizing models"],
  "data_augmentation": ["Potential data augmentation strategies if applicable"],
  "research_directions": ["Advanced techniques or research areas relevant to this dataset"]
}"""
        else:
            prompt = base_info + "\n\nProvide comprehensive insights about this dataset in structured JSON format."

        # Ensure reasonable length while keeping comprehensive
        if len(prompt) > 2500:
            # Trim less critical sections while keeping core analysis
            prompt = (
                prompt[:2200]
                + "...\n\nProvide comprehensive analysis based on the dataset characteristics shown above."
            )

        return prompt

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Call the LLM API with retry logic for rate limits."""
        # Check cache first
        cache_key = hashlib.md5(f"{self.provider}:{self.model}:{prompt}".encode()).hexdigest()
        if cache_key in _ai_response_cache:
            self.logger.info("Using cached AI response")
            return _ai_response_cache[cache_key]

        last_exception = None

        for attempt in range(max_retries):
            try:
                response_text = self._make_llm_api_call(prompt)

                # Cache successful response
                _ai_response_cache[cache_key] = response_text

                # Limit cache size (keep last 100 entries)
                if len(_ai_response_cache) > 100:
                    oldest_key = next(iter(_ai_response_cache))
                    del _ai_response_cache[oldest_key]

                return response_text

            except Exception as e:
                error_str = str(e)
                last_exception = e

                # Check if it's a rate limit error
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    wait_time = self._extract_wait_time_from_error(error_str)

                    if wait_time and wait_time > 0:
                        self.logger.warning(
                            f"Rate limit hit. Waiting {wait_time:.0f}s before retry {attempt + 1}/{max_retries}"
                        )

                        # Only wait if it's reasonable (< 5 minutes) and not last attempt
                        if wait_time < 300 and attempt < max_retries - 1:
                            time.sleep(wait_time)
                            continue

                    # Rate limit too long or last attempt
                    self.logger.error(f"Rate limit exceeded: {error_str}")
                    raise Exception("Rate limit exceeded. Please try again later or upgrade your API plan.")

                # For other errors, retry with exponential backoff
                if attempt < max_retries - 1:
                    backoff = (2**attempt) * 2  # 2s, 4s, 8s
                    self.logger.warning(f"API call failed: {error_str}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    self.logger.error(f"API call failed after {max_retries} attempts: {error_str}")
                    raise

        raise last_exception or Exception("API call failed")

    def _extract_wait_time_from_error(self, error_message: str) -> Optional[float]:
        """Extract wait time from rate limit error message."""
        import re

        # Look for patterns like "3m29.952s" or "5m33s" or "30s"
        pattern = r"(?:(\d+)m)?(\d+(?:\.\d+)?)s"
        match = re.search(pattern, error_message)

        if match:
            minutes = int(match.group(1)) if match.group(1) else 0
            seconds = float(match.group(2))
            return minutes * 60 + seconds

        return None

    def _make_llm_api_call(self, prompt: str) -> str:
        """Make the actual LLM API call (no retry logic)."""

        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data scientist and ML engineer with deep expertise in dataset analysis, statistical insights, and machine learning. Always provide comprehensive, actionable insights in valid JSON format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=3000,
            )
            return response.choices[0].message.content

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data scientist and ML engineer with deep expertise in dataset analysis, statistical insights, and machine learning. Always provide comprehensive, actionable insights in valid JSON format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=3000,
            )
            return response.choices[0].message.content

        elif self.provider == "gemini":
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": 3000,
                },
            )
            return response.text

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()

            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON response, returning as text")
            return {"raw_response": response}

    def _generate_enhanced_fallback_insights(self, stats: EnhancedDatasetStatistics) -> Dict[str, Any]:
        """Generate comprehensive rule-based insights without AI (enhanced fallback)."""
        insights = {}

        # Dataset overview
        overview_parts = []
        overview_parts.append(f"Dataset with {stats.n_samples:,} samples and {stats.n_features} features")

        if stats.n_classes:
            overview_parts.append(f"for {stats.n_classes}-class {stats.target_type}")

        complexity_desc = {
            "simple": "straightforward structure suitable for most ML algorithms",
            "moderate": "moderate complexity requiring careful preprocessing",
            "complex": "high complexity demanding advanced techniques",
        }
        overview_parts.append(complexity_desc.get(stats.dataset_complexity, "unknown complexity"))

        insights["dataset_overview"] = ". ".join(overview_parts) + "."

        # Key strengths
        strengths = []
        if stats.n_samples >= 10000:
            strengths.append(f"Large sample size ({stats.n_samples:,}) enables complex model training")
        elif stats.n_samples >= 1000:
            strengths.append(f"Good sample size ({stats.n_samples:,}) for reliable model validation")

        if stats.data_quality_score >= 85:
            strengths.append(f"Excellent data quality (score: {stats.data_quality_score:.0f}/100)")
        elif stats.data_quality_score >= 70:
            strengths.append(f"Good data quality (score: {stats.data_quality_score:.0f}/100)")

        if stats.missing_rate < 0.05:
            strengths.append(f"Minimal missing data ({stats.missing_rate:.1%}) reduces preprocessing burden")

        if stats.outlier_rate < 0.02:
            strengths.append("Low outlier rate indicates clean, consistent data")

        if len(stats.feature_correlations) > 0 and stats.feature_correlations[0][2] < 0.9:
            strengths.append("Features show diverse patterns without excessive multicollinearity")

        insights["key_strengths"] = strengths if strengths else ["Dataset structure is suitable for machine learning"]

        # Critical challenges
        challenges = []

        if stats.missing_rate > 0.15:
            challenges.append(
                f"Very high missing data rate ({stats.missing_rate:.1%}) requires robust imputation strategy"
            )
        elif stats.missing_rate > 0.05:
            challenges.append(f"Moderate missing data ({stats.missing_rate:.1%}) needs careful handling")

        if stats.outlier_rate > 0.1:
            challenges.append(f"High outlier rate ({stats.outlier_rate:.1%}) may skew model performance")

        if stats.class_balance:
            max_ratio = max(stats.class_balance.values())
            min_ratio = min(stats.class_balance.values())
            imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float("inf")

            if imbalance_ratio > 20:
                challenges.append(
                    f"Severe class imbalance (ratio {imbalance_ratio:.1f}:1) requires specialized techniques"
                )
            elif imbalance_ratio > 10:
                challenges.append(
                    f"Significant class imbalance (ratio {imbalance_ratio:.1f}:1) needs balancing strategies"
                )

        if len(stats.skewness_summary) > stats.n_numeric * 0.6:
            challenges.append(f"Many highly skewed features ({len(stats.skewness_summary)}) require transformation")

        if stats.n_features > 100 and stats.n_samples < stats.n_features * 10:
            challenges.append(f"High dimensionality ({stats.n_features} features) vs samples may cause overfitting")

        high_cardinality = [f for f, c in stats.categorical_cardinality.items() if c > stats.n_samples * 0.1]
        if high_cardinality:
            challenges.append("High-cardinality categorical features need encoding optimization")

        insights["critical_challenges"] = challenges if challenges else ["No major data quality issues detected"]

        # Data quality assessment
        quality_issues = []
        if stats.data_quality_score < 60:
            quality_issues.append("Poor data quality requiring extensive preprocessing")
        elif stats.data_quality_score < 80:
            quality_issues.append("Moderate data quality with several issues to address")
        else:
            quality_issues.append("Good overall data quality with minor preprocessing needs")

        quality_details = []
        if stats.missing_rate > 0.1:
            quality_details.append(f"missing data ({stats.missing_rate:.1%})")
        if stats.outlier_rate > 0.05:
            quality_details.append(f"outliers ({stats.outlier_rate:.1%})")
        if len(stats.potential_issues) > 3:
            quality_details.append(f"multiple data issues ({len(stats.potential_issues)})")

        if quality_details:
            quality_issues.append(f"Main concerns: {', '.join(quality_details)}")

        insights["data_quality_assessment"] = ". ".join(quality_issues)

        # Preprocessing strategy
        preprocessing = []

        if stats.missing_rate > 0.01:
            if stats.missing_rate > 0.1:
                preprocessing.append("Apply multiple imputation (MICE) for missing values")
            else:
                preprocessing.append("Use median/mode imputation for missing values")

        if stats.outlier_rate > 0.05:
            preprocessing.append("Apply outlier detection and treatment (IQR or Z-score methods)")

        if len(stats.skewness_summary) > 0:
            preprocessing.append("Apply power transformations (Box-Cox/Yeo-Johnson) for skewed features")

        if stats.n_numeric > 0:
            preprocessing.append("Standardize/normalize numeric features for consistent scales")

        if stats.categorical_cardinality:
            high_card_features = [f for f, c in stats.categorical_cardinality.items() if c > 20]
            if high_card_features:
                preprocessing.append("Apply target encoding or embedding for high-cardinality categoricals")
            else:
                preprocessing.append("Use one-hot encoding for categorical features")

        if stats.feature_correlations and stats.feature_correlations[0][2] > 0.95:
            preprocessing.append("Remove highly correlated features to reduce multicollinearity")

        insights["preprocessing_strategy"] = preprocessing

        # Recommended models
        models = []

        # Based on dataset size and complexity
        if stats.n_samples < 1000:
            models.append("Logistic Regression/Linear models (robust for small datasets)")
            models.append("Random Forest (handles overfitting well)")
            models.append("Support Vector Machine (effective in high dimensions)")
        elif stats.n_samples < 10000:
            models.append("Random Forest (excellent baseline performance)")
            models.append("Gradient Boosting (XGBoost/LightGBM) for best accuracy")
            models.append("Neural Networks (if sufficient preprocessing)")
        else:
            models.append("Gradient Boosting (XGBoost/LightGBM) for production performance")
            models.append("Deep Neural Networks for complex patterns")
            models.append("Ensemble methods combining multiple algorithms")

        # Special considerations
        if stats.class_balance and len(stats.class_balance) > 10:
            models.append("Tree-based models (handle many classes naturally)")

        if stats.n_categorical > stats.n_numeric:
            models.append("CatBoost (optimized for categorical features)")

        insights["recommended_models"] = models

        # Feature engineering opportunities
        feature_eng = []

        if stats.domain_insights.get("temporal_features"):
            feature_eng.append("Extract temporal features (day, month, seasonality) from date columns")

        if stats.domain_insights.get("text_features"):
            feature_eng.append("Apply TF-IDF or word embeddings for text features")

        if len(stats.feature_correlations) > 0:
            feature_eng.append("Create polynomial/interaction features from top correlated pairs")

        if stats.n_numeric > 5:
            feature_eng.append("Apply dimensionality reduction (PCA) if high correlation exists")

        feature_eng.append("Create binned versions of continuous variables for non-linear relationships")

        if stats.categorical_cardinality:
            feature_eng.append("Generate frequency-based features from categorical variables")

        insights["feature_engineering_opportunities"] = feature_eng

        # Statistical insights
        statistical = []

        if stats.skewness_summary:
            skewed_count = len(stats.skewness_summary)
            statistical.append(f"{skewed_count} features show high skewness requiring transformation")

        if stats.variance_summary.get("low_variance", 0) > 0:
            statistical.append(
                f"{stats.variance_summary['low_variance']} features have low variance and may be uninformative"
            )

        if stats.feature_correlations:
            max_corr = stats.feature_correlations[0][2]
            statistical.append(
                f"Maximum feature correlation is {max_corr:.3f}, indicating {'high' if max_corr > 0.8 else 'moderate' if max_corr > 0.5 else 'low'} multicollinearity"
            )

        if stats.correlation_with_target:
            max_target_corr = stats.correlation_with_target[0][1]
            statistical.append(
                f"Strongest target correlation is {max_target_corr:.3f}, suggesting {'strong' if max_target_corr > 0.7 else 'moderate' if max_target_corr > 0.3 else 'weak'} linear relationships"
            )

        insights["statistical_insights"] = statistical if statistical else ["Standard statistical properties observed"]

        # Risk factors
        risks = []

        if stats.n_samples < 500:
            risks.append("Small dataset size increases overfitting risk")

        if stats.n_features > stats.n_samples * 0.1:
            risks.append("High dimensionality relative to samples may cause curse of dimensionality")

        if stats.data_quality_score < 70:
            risks.append("Poor data quality may lead to unreliable model performance")

        if not risks:
            risks.append("Low risk factors identified for this dataset")

        insights["risk_factors"] = risks

        # Expected performance
        if stats.data_quality_score >= 80 and stats.n_samples >= 1000:
            performance = "High performance expected (accuracy >85%) due to good data quality and sufficient samples"
        elif stats.data_quality_score >= 60 and stats.n_samples >= 500:
            performance = "Moderate performance expected (accuracy 70-85%) with careful preprocessing"
        else:
            performance = "Performance may be limited due to data quality or size constraints"

        insights["expected_performance"] = performance

        # Next steps
        next_steps = []
        next_steps.append("Perform exploratory data analysis to validate statistical assumptions")
        next_steps.append("Implement preprocessing pipeline based on identified issues")
        next_steps.append("Start with simple baseline models before trying complex algorithms")
        next_steps.append("Set up robust cross-validation strategy for reliable performance estimates")

        insights["next_steps"] = next_steps

        return insights


def get_enhanced_ai_engine(provider: Optional[str] = None) -> Optional[EnhancedAIInsightEngine]:
    """
    Factory function to create enhanced AI engine with environment-based configuration.

    Args:
        provider: Override default provider from environment

    Returns:
        EnhancedAIInsightEngine instance or None if AI is disabled
    """
    # Check if AI insights are enabled
    if os.getenv("ENABLE_AI_INSIGHTS", "true").lower() != "true":
        return None

    provider = provider or os.getenv("DEFAULT_LLM_PROVIDER", "groq")
    temperature = float(os.getenv("AI_TEMPERATURE", "0.3"))

    try:
        return EnhancedAIInsightEngine(provider=provider, temperature=temperature)
    except Exception as e:
        logger.error(f"Failed to initialize enhanced AI engine: {e}")
        return None
