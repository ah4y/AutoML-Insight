"""Self-check for the target-column heuristic used to default the Task & Analysis tab."""

import pandas as pd

from app.ui_dashboard import AutoMLDashboard


def test_prefers_conventionally_named_column():
    df = pd.DataFrame({"id": range(100), "feature": range(100), "target": [0, 1] * 50})
    assert AutoMLDashboard._guess_target_column(None, df) == "target"


def test_prefers_lowest_cardinality_when_no_convention():
    df = pd.DataFrame({"a": range(50), "b": [0, 1] * 25, "c": ["x", "y", "z"] * 16 + ["x", "y"]})
    assert AutoMLDashboard._guess_target_column(None, df) == "b"


def test_falls_back_to_last_column_when_nothing_eligible():
    df = pd.DataFrame({"a": range(50), "b": range(50, 100)})
    assert AutoMLDashboard._guess_target_column(None, df) == "b"
