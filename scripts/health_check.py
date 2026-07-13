"""Comprehensive Health Check for AutoML-Insight."""
import sys
import os
import io
import traceback

# Force UTF-8 output to avoid Windows cp1252 encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

results = {"pass": [], "fail": [], "warn": []}

def check(name, fn):
    try:
        fn()
        results["pass"].append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        results["fail"].append((name, str(e)))
        print(f"  [FAIL] {name}: {e}")

def warn_check(name, fn):
    try:
        fn()
        results["pass"].append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        results["warn"].append((name, str(e)))
        print(f"  [WARN] {name}: {e}")

print("=" * 70)
print("  AutoML-Insight Health Check")
print("=" * 70)

# -- 1. DEPENDENCY CHECKS --
print("\n--- 1. Core Dependencies ---")

check("numpy", lambda: __import__("numpy"))
check("pandas", lambda: __import__("pandas"))
check("scikit-learn", lambda: __import__("sklearn"))
check("xgboost", lambda: __import__("xgboost"))
check("streamlit", lambda: __import__("streamlit"))
check("plotly", lambda: __import__("plotly"))
check("matplotlib", lambda: __import__("matplotlib"))
check("seaborn", lambda: __import__("seaborn"))
check("scipy", lambda: __import__("scipy"))
check("yaml (pyyaml)", lambda: __import__("yaml"))
check("joblib", lambda: __import__("joblib"))
check("tqdm", lambda: __import__("tqdm"))

print("\n--- 2. AI/ML Dependencies ---")
warn_check("shap", lambda: __import__("shap"))
warn_check("optuna", lambda: __import__("optuna"))
warn_check("umap-learn", lambda: __import__("umap"))
warn_check("torch", lambda: __import__("torch"))
warn_check("statsmodels", lambda: __import__("statsmodels"))

print("\n--- 3. AI/LLM Dependencies ---")
warn_check("groq", lambda: __import__("groq"))
warn_check("openai", lambda: __import__("openai"))
warn_check("google-generativeai", lambda: __import__("google.generativeai"))
warn_check("langchain", lambda: __import__("langchain"))
warn_check("langchain-groq", lambda: __import__("langchain_groq"))
warn_check("python-dotenv", lambda: __import__("dotenv"))

print("\n--- 4. Utility Dependencies ---")
warn_check("reportlab", lambda: __import__("reportlab"))
warn_check("weasyprint", lambda: __import__("weasyprint"))
warn_check("psutil", lambda: __import__("psutil"))
warn_check("pyngrok", lambda: __import__("pyngrok"))
warn_check("requests", lambda: __import__("requests"))

# -- 2. MODULE IMPORT CHECKS --
print("\n--- 5. Core Module Imports ---")

check("core.data_profile", lambda: __import__("core.data_profile"))
check("core.preprocess", lambda: __import__("core.preprocess"))
check("core.models_supervised", lambda: __import__("core.models_supervised"))
check("core.models_clustering", lambda: __import__("core.models_clustering"))
check("core.tuning", lambda: __import__("core.tuning"))
check("core.evaluate_cls", lambda: __import__("core.evaluate_cls"))
check("core.evaluate_clu", lambda: __import__("core.evaluate_clu"))
check("core.visualize", lambda: __import__("core.visualize"))
check("core.explain", lambda: __import__("core.explain"))
check("core.meta_selector", lambda: __import__("core.meta_selector"))
check("core.ensemble", lambda: __import__("core.ensemble"))
check("core.advanced_metrics", lambda: __import__("core.advanced_metrics"))
check("core.advanced_optimization", lambda: __import__("core.advanced_optimization"))
check("core.ai_insights", lambda: __import__("core.ai_insights"))
check("core.ai_insights_enhanced", lambda: __import__("core.ai_insights_enhanced"))
check("core.background_analyzer", lambda: __import__("core.background_analyzer"))
check("core.bias_variance_analyzer", lambda: __import__("core.bias_variance_analyzer"))
check("core.dimred", lambda: __import__("core.dimred"))
check("core.dimred_evaluator", lambda: __import__("core.dimred_evaluator"))
check("core.overfitting_detector", lambda: __import__("core.overfitting_detector"))
check("core.statistical_tests", lambda: __import__("core.statistical_tests"))

print("\n--- 6. Utils Module Imports ---")

check("utils.seed_utils", lambda: __import__("utils.seed_utils"))
check("utils.logging_utils", lambda: __import__("utils.logging_utils"))
check("utils.metrics_utils", lambda: __import__("utils.metrics_utils"))
check("utils.cache_utils", lambda: __import__("utils.cache_utils"))
check("utils.cloud_executor", lambda: __import__("utils.cloud_executor"))
check("utils.jupyter_client", lambda: __import__("utils.jupyter_client"))
check("utils.performance_monitor", lambda: __import__("utils.performance_monitor"))
check("utils.visualization_helpers", lambda: __import__("utils.visualization_helpers"))

print("\n--- 7. App Module Imports ---")

check("app.main", lambda: __import__("app.main"))
check("app.report_builder", lambda: __import__("app.report_builder"))
# ui_dashboard is huge (438KB), just check it's parseable
def check_ui_dashboard():
    import ast
    with open("app/ui_dashboard.py", "r", encoding="utf-8") as f:
        ast.parse(f.read())
check("app.ui_dashboard (syntax)", check_ui_dashboard)

# -- 3. CONFIG CHECKS --
print("\n--- 8. Configuration Files ---")

def check_config():
    import yaml
    with open("app/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    assert "preprocessing" in cfg, "Missing preprocessing section"
    assert "training" in cfg, "Missing training section"
    assert "tuning" in cfg, "Missing tuning section"
check("app/config.yaml valid", check_config)

def check_root_config():
    import yaml
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg is not None, "Root config.yaml is empty"
check("config.yaml (root) valid", check_root_config)

def check_streamlit_config():
    import tomllib
    with open(".streamlit/config.toml", "rb") as f:
        cfg = tomllib.load(f)
    assert "theme" in cfg, "Missing theme section"
check(".streamlit/config.toml valid", check_streamlit_config)

# -- 4. ENV CHECKS --
print("\n--- 9. Environment Variables (.env) ---")

def check_env():
    from dotenv import load_dotenv
    load_dotenv()
    keys = ["GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise Exception(f"Missing keys: {', '.join(missing)}")
warn_check(".env API keys present", check_env)

# -- 5. FUNCTIONAL CHECKS --
print("\n--- 10. Functional Integration Checks ---")

def check_data_profiling():
    import pandas as pd
    from core.data_profile import DataProfiler
    df = pd.DataFrame({"a": [1,2,3,4,5], "b": [10,20,30,40,50], "c": ["x","y","x","y","x"]})
    profiler = DataProfiler()
    profile = profiler.profile_dataset(df)
    assert profile is not None, "Profiler returned None"
check("DataProfiler.profile_dataset()", check_data_profiling)

def check_preprocessing():
    import pandas as pd
    import numpy as np
    from core.preprocess import DataPreprocessor
    df = pd.DataFrame({
        "num1": [1, 2, np.nan, 4, 5],
        "num2": [10, 20, 30, 40, 50],
        "cat1": ["a", "b", "a", "b", "a"],
        "target": [0, 1, 0, 1, 0]
    })
    p = DataPreprocessor()
    X_t, y_t = p.fit_transform(df.drop(columns=['target']), df['target'])
    assert X_t is not None, "DataPreprocessor returned None"
check("DataPreprocessor.fit_transform()", check_preprocessing)

def check_supervised_models():
    from core.models_supervised import get_supervised_models
    models = get_supervised_models()
    assert len(models) > 0, "No supervised models returned"
check("get_supervised_models()", check_supervised_models)

def check_clustering_models():
    from core.models_clustering import get_clustering_models
    models = get_clustering_models()
    assert len(models) > 0, "No clustering models returned"
check("get_clustering_models()", check_clustering_models)

def check_seed_utils():
    from utils.seed_utils import set_seed
    set_seed(42)
check("set_seed()", check_seed_utils)

def check_logging():
    from utils.logging_utils import setup_logger
    logger = setup_logger("health_check_test")
    assert logger is not None
check("setup_logger()", check_logging)

def check_confidence_interval():
    import numpy as np
    from utils.metrics_utils import compute_confidence_interval
    scores = np.array([0.8, 0.85, 0.82, 0.88, 0.84])
    result = compute_confidence_interval(scores)
    assert result is not None
check("compute_confidence_interval()", check_confidence_interval)

def check_train_evaluate_pipeline():
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification
    from core.preprocess import DataPreprocessor
    from core.models_supervised import get_supervised_models
    
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    df_X = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    y_series = pd.Series(y)
    
    p = DataPreprocessor()
    X_proc, y_proc = p.fit_transform(df_X, y_series)
    
    models = get_supervised_models()
    first_model_name = list(models.keys())[0]
    model = models[first_model_name]
    
    model.fit(X_proc[:80], y_proc[:80])
    score = model.score(X_proc[80:], y_proc[80:])
    assert 0 <= score <= 1, f"Invalid score: {score}"
check("Train->Evaluate pipeline", check_train_evaluate_pipeline)

# -- 6. REPORT BUILDER CHECK --
print("\n--- 11. Report Builder ---")

def check_report_builder():
    from app.report_builder import ReportBuilder
    rb = ReportBuilder.__new__(ReportBuilder)
    assert hasattr(rb, '__class__'), "ReportBuilder class not importable"
warn_check("ReportBuilder importable", check_report_builder)

# -- SUMMARY --
print("\n" + "=" * 70)
print("  HEALTH CHECK SUMMARY")
print("=" * 70)
print(f"  [PASS] Passed:   {len(results['pass'])}")
print(f"  [WARN] Warnings: {len(results['warn'])}")
print(f"  [FAIL] Failed:   {len(results['fail'])}")
print()

if results["warn"]:
    print("  Warnings:")
    for name, err in results["warn"]:
        print(f"    [WARN] {name}: {err}")
    print()

if results["fail"]:
    print("  Failures:")
    for name, err in results["fail"]:
        print(f"    [FAIL] {name}: {err}")
    print()

if not results["fail"]:
    print("  ALL CRITICAL CHECKS PASSED!")
else:
    print(f"  {len(results['fail'])} critical check(s) failed -- needs attention")

print("=" * 70)
