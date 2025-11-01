"""
AI-Powered Insights Engine for AutoML.

This module provides dynamic, dataset-specific insights using LLMs.
Supports multiple providers: Groq (fast), OpenAI (powerful), Gemini (alternative).
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Global cache for AI responses
_ai_response_cache: Dict[str, Dict[str, Any]] = {}


@dataclass
class DatasetStatistics:
    """Container for dataset statistics to send to LLM."""
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


class AIInsightEngine:
    """
    AI-powered analysis engine for AutoML.
    
    Generates dynamic insights based on dataset characteristics using LLMs.
    Never static - every insight is tailored to your specific data.
    """
    
    def __init__(
        self,
        provider: str = "groq",
        model: Optional[str] = None,
        temperature: float = 0.3
    ):
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
            'groq': os.getenv('GROQ_API_KEY'),
            'openai': os.getenv('OPENAI_API_KEY'),
            'gemini': os.getenv('GEMINI_API_KEY'),
        }
        
        # Default models for each provider
        self.default_models = {
            'groq': 'llama-3.3-70b-versatile',  # Fast and capable (updated model)
            'openai': 'gpt-4o-mini',  # Cost-effective GPT-4
            'gemini': 'gemini-1.5-flash',  # Fast Gemini
        }
        
        self.model = model or self.default_models.get(self.provider)
        
        # Initialize the appropriate client
        self._init_client()
        
        self.logger.info(f"AI Insight Engine initialized: {self.provider}/{self.model}")
    
    def _init_client(self) -> None:
        """Initialize LLM client based on provider."""
        api_key = self.api_keys.get(self.provider)
        
        if not api_key:
            raise ValueError(
                f"API key for {self.provider} not found. "
                f"Please set {self.provider.upper()}_API_KEY in .env file"
            )
        
        if self.provider == 'groq':
            from groq import Groq
            self.client = Groq(api_key=api_key)
            
        elif self.provider == 'openai':
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            
        elif self.provider == 'gemini':
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model)
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def analyze_dataset(
        self,
        data: pd.DataFrame,
        target_col: Optional[str] = None,
        task_type: str = "classification"
    ) -> DatasetStatistics:
        """
        Extract comprehensive statistics from dataset.
        
        Args:
            data: Input DataFrame
            target_col: Target column name
            task_type: 'classification' or 'regression'
            
        Returns:
            DatasetStatistics object with all metrics
        """
        n_samples, n_features = data.shape
        
        # Separate features and target
        if target_col and target_col in data.columns:
            X = data.drop(columns=[target_col])
            y = data[target_col]
        else:
            X = data
            y = None
        
        # Feature types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Target analysis
        n_classes = None
        class_balance = None
        if y is not None and task_type == "classification":
            n_classes = y.nunique()
            value_counts = y.value_counts(normalize=True)
            class_balance = {str(k): float(v) for k, v in value_counts.items()}
        
        # Missing values
        missing_rate = float(data.isnull().sum().sum() / (n_samples * n_features))
        
        # Feature correlations (top 5)
        correlations = []
        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr().abs()
            # Get upper triangle
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            # Find top correlations
            corr_pairs = [
                (col, row, float(upper.loc[row, col]))
                for col in upper.columns
                for row in upper.index
                if not pd.isna(upper.loc[row, col])
            ]
            correlations = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:5]
        
        # Top features (by variance for numeric, by cardinality for categorical)
        top_features = []
        if numeric_cols:
            variances = X[numeric_cols].var().sort_values(ascending=False)
            top_features.extend(variances.head(3).index.tolist())
        
        # Data quality score (0-100)
        quality_score = 100.0
        quality_score -= missing_rate * 30  # Penalize missing data
        if class_balance and len(class_balance) > 1:
            # Penalize severe class imbalance
            imbalance = max(class_balance.values()) / min(class_balance.values())
            if imbalance > 10:
                quality_score -= 20
            elif imbalance > 5:
                quality_score -= 10
        
        return DatasetStatistics(
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
            data_quality_score=max(0, quality_score)
        )
    
    def generate_insights(
        self,
        stats: DatasetStatistics,
        context: str = "initial_analysis"
    ) -> Dict[str, str]:
        """
        Generate AI-powered insights based on dataset statistics.
        Falls back to rule-based insights if AI fails.
        
        Args:
            stats: Dataset statistics object
            context: Analysis context ('initial_analysis', 'model_selection', 'results_interpretation')
            
        Returns:
            Dictionary with different types of insights
        """
        prompt = self._build_prompt(stats, context)
        
        try:
            response = self._call_llm(prompt)
            insights = self._parse_response(response)
            insights['_source'] = 'ai'  # Mark as AI-generated
            return insights
            
        except Exception as e:
            error_msg = str(e)
            self.logger.warning(f"AI insights failed: {error_msg}")
            
            # Use rule-based fallback
            insights = self._fallback_insights(stats)
            insights['_source'] = 'rules'  # Mark as rule-based
            
            # Add informative message for user
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                insights['_notice'] = "⚠️ AI rate limit reached. Showing rule-based recommendations. Upgrade your API plan or try again later for AI insights."
            else:
                insights['_notice'] = "⚠️ AI analysis unavailable. Showing rule-based recommendations."
            
            return insights
    
    def _build_prompt(self, stats: DatasetStatistics, context: str) -> str:
        """Build prompt for LLM based on statistics and context."""
        
        base_info = f"""You are an expert data scientist analyzing a machine learning dataset.

Dataset Overview:
- Samples: {stats.n_samples:,}
- Features: {stats.n_features} ({stats.n_numeric} numeric, {stats.n_categorical} categorical)
- Task: {stats.target_type}
- Classes: {stats.n_classes if stats.n_classes else 'N/A'}
- Missing Data: {stats.missing_rate:.1%}
- Data Quality Score: {stats.data_quality_score:.1f}/100

"""
        
        if stats.class_balance:
            base_info += f"\nClass Distribution:\n"
            for cls, ratio in stats.class_balance.items():
                base_info += f"  - Class {cls}: {ratio:.1%}\n"
        
        if stats.feature_correlations:
            base_info += f"\nTop Feature Correlations:\n"
            for feat1, feat2, corr in stats.feature_correlations[:3]:
                base_info += f"  - {feat1} ↔ {feat2}: {corr:.3f}\n"
        
        if context == "initial_analysis":
            prompt = base_info + """
Provide a brief, actionable analysis in JSON format with these keys:
1. "summary": 2-3 sentence overview of the dataset
2. "strengths": List 2-3 positive aspects
3. "challenges": List 2-3 potential issues or challenges
4. "recommendations": List 3-4 specific preprocessing or modeling recommendations

Be concise and data-driven. Focus on actionable insights."""

        elif context == "model_selection":
            prompt = base_info + """
Recommend the best ML models for this dataset in JSON format:
1. "top_models": List 3 recommended models with brief justification
2. "avoid_models": List 1-2 models to avoid with reasons
3. "hyperparameter_focus": Key hyperparameters to tune

Consider dataset size, feature types, and class balance."""

        elif context == "results_interpretation":
            prompt = base_info + """
Provide interpretation guidance in JSON format:
1. "key_metrics": Which metrics matter most for this dataset and why
2. "expected_performance": Realistic performance expectations
3. "red_flags": What results would indicate problems

Be specific to this dataset's characteristics."""

        else:
            prompt = base_info + "\nProvide key insights about this dataset."
        
        return prompt
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """
        Call the LLM API with retry logic for rate limits.
        
        Args:
            prompt: The prompt to send to LLM
            max_retries: Maximum retry attempts
            
        Returns:
            Response text from LLM
            
        Raises:
            Exception: If all retries fail
        """
        # Check cache first
        cache_key = hashlib.md5(f"{self.provider}:{self.model}:{prompt}".encode()).hexdigest()
        if cache_key in _ai_response_cache:
            self.logger.info("Using cached AI response")
            return _ai_response_cache[cache_key]
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response_text = self._make_llm_call(prompt)
                
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
                    # Extract wait time from error message
                    wait_time = self._extract_wait_time(error_str)
                    
                    if wait_time and wait_time > 0:
                        self.logger.warning(f"Rate limit hit. Waiting {wait_time:.0f}s before retry {attempt + 1}/{max_retries}")
                        
                        # Only wait if it's reasonable (< 5 minutes) and not last attempt
                        if wait_time < 300 and attempt < max_retries - 1:
                            time.sleep(wait_time)
                            continue
                    
                    # Rate limit too long or last attempt - return fallback
                    self.logger.error(f"Rate limit exceeded: {error_str}")
                    raise Exception(f"Rate limit exceeded. Please try again later or upgrade your API plan.")
                
                # For other errors, retry with exponential backoff
                if attempt < max_retries - 1:
                    backoff = (2 ** attempt) * 2  # 2s, 4s, 8s
                    self.logger.warning(f"API call failed: {error_str}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    self.logger.error(f"API call failed after {max_retries} attempts: {error_str}")
                    raise
        
        # Should never reach here, but just in case
        raise last_exception or Exception("API call failed")
    
    def _extract_wait_time(self, error_message: str) -> Optional[float]:
        """Extract wait time from rate limit error message."""
        import re
        
        # Look for patterns like "3m29.952s" or "5m33s" or "30s"
        pattern = r'(?:(\d+)m)?(\d+(?:\.\d+)?)s'
        match = re.search(pattern, error_message)
        
        if match:
            minutes = int(match.group(1)) if match.group(1) else 0
            seconds = float(match.group(2))
            return minutes * 60 + seconds
        
        return None
    
    def _make_llm_call(self, prompt: str) -> str:
        """Make the actual LLM API call (no retry logic)."""
        
        if self.provider == 'groq':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert data scientist providing concise, actionable insights. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000,
            )
            return response.choices[0].message.content
            
        elif self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert data scientist providing concise, actionable insights. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000,
            )
            return response.choices[0].message.content
            
        elif self.provider == 'gemini':
            response = self.client.generate_content(
                prompt,
                generation_config={
                    'temperature': self.temperature,
                    'max_output_tokens': 2000,
                }
            )
            return response.text
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
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
    
    def _fallback_insights(self, stats: DatasetStatistics) -> Dict[str, str]:
        """Generate rule-based insights without AI (fallback for rate limits/errors)."""
        insights = {}
        
        # Summary
        summary_parts = []
        summary_parts.append(f"Dataset with {stats.n_samples:,} samples and {stats.n_features} features")
        
        if stats.n_classes:
            summary_parts.append(f"{stats.n_classes}-class {stats.target_type} problem")
        
        insights["summary"] = ". ".join(summary_parts) + "."
        
        # Strengths
        strengths = []
        if stats.n_samples >= 1000:
            strengths.append(f"Good sample size ({stats.n_samples:,} samples) for reliable training")
        if stats.data_quality_score >= 80:
            strengths.append(f"High data quality (score: {stats.data_quality_score:.0f}/100)")
        if stats.missing_rate < 0.05:
            strengths.append(f"Low missing data ({stats.missing_rate:.1%})")
        
        if strengths:
            insights["strengths"] = strengths
        
        # Challenges
        challenges = []
        if stats.missing_rate > 0.1:
            challenges.append(f"High missing data rate ({stats.missing_rate:.1%}) - consider imputation strategies")
        
        if stats.class_balance:
            max_ratio = max(stats.class_balance.values())
            min_ratio = min(stats.class_balance.values())
            imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')
            
            if imbalance_ratio > 10:
                challenges.append(f"Severe class imbalance (ratio {imbalance_ratio:.1f}:1) - use SMOTE or class weights")
            elif imbalance_ratio > 5:
                challenges.append(f"Moderate class imbalance (ratio {imbalance_ratio:.1f}:1) - monitor minority class performance")
        
        if stats.n_features > 100:
            challenges.append(f"High dimensionality ({stats.n_features} features) - consider feature selection")
        
        if stats.n_samples < 500:
            challenges.append(f"Small dataset ({stats.n_samples} samples) - risk of overfitting, use cross-validation")
        
        if challenges:
            insights["challenges"] = challenges
        
        # Recommendations
        recommendations = []
        
        # Based on dataset size
        if stats.n_samples < 500:
            recommendations.append("Use k-fold cross-validation (5-10 folds) for reliable evaluation")
            recommendations.append("Try simpler models (LogisticRegression, RandomForest) to avoid overfitting")
        elif stats.n_samples > 10000:
            recommendations.append("Large dataset allows complex models (XGBoost, Neural Networks)")
            recommendations.append("Consider using early stopping to save training time")
        
        # Based on feature types
        if stats.n_categorical > stats.n_numeric:
            recommendations.append("Many categorical features - ensure proper encoding (one-hot/label encoding)")
        
        # Based on class balance
        if stats.class_balance and imbalance_ratio > 5:
            recommendations.append("Use stratified splits and balanced accuracy metrics")
            recommendations.append("Consider SMOTE oversampling or class_weight='balanced'")
        
        # Based on missing data
        if stats.missing_rate > 0.1:
            recommendations.append("Apply robust imputation (median for numeric, mode for categorical)")
        
        # Feature correlation
        if stats.feature_correlations and len(stats.feature_correlations) > 0:
            high_corr = [c for c in stats.feature_correlations if c[2] > 0.9]
            if high_corr:
                recommendations.append(f"Remove highly correlated features ({len(high_corr)} pairs with r>0.9)")
        
        # Ensure at least 3 recommendations
        if len(recommendations) < 3:
            recommendations.append("Start with tree-based models (RandomForest, XGBoost) for good baseline")
            recommendations.append("Monitor train vs test performance to detect overfitting")
        
        insights["recommendations"] = recommendations
        
        # Data quality assessment
        if stats.data_quality_score < 60:
            insights["data_quality"] = f"⚠️ Data quality score is low ({stats.data_quality_score:.0f}/100). Address missing data and class imbalance before training."
        elif stats.data_quality_score < 80:
            insights["data_quality"] = f"Data quality is moderate ({stats.data_quality_score:.0f}/100). Some preprocessing recommended."
        else:
            insights["data_quality"] = f"✅ Good data quality ({stats.data_quality_score:.0f}/100)."
        
        return insights


def get_ai_engine(provider: Optional[str] = None) -> Optional[AIInsightEngine]:
    """
    Factory function to create AI engine with environment-based configuration.
    
    Args:
        provider: Override default provider from environment
        
    Returns:
        AIInsightEngine instance or None if AI is disabled
    """
    # Check if AI insights are enabled
    if os.getenv('ENABLE_AI_INSIGHTS', 'true').lower() != 'true':
        return None
    
    provider = provider or os.getenv('DEFAULT_LLM_PROVIDER', 'groq')
    temperature = float(os.getenv('AI_TEMPERATURE', '0.3'))
    
    try:
        return AIInsightEngine(provider=provider, temperature=temperature)
    except Exception as e:
        logger.error(f"Failed to initialize AI engine: {e}")
        return None
