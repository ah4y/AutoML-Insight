"""Self-check for the class-filter re-encoding fix in AutoMLDashboard.run_automl.

Filtering out rare classes after LabelEncoder has already run can leave gaps in the
encoded label space (e.g. removing class 20 out of 0..21 leaves 0..19,21), which
XGBoost rejects as non-contiguous. This checks the re-encode pattern used in the fix.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder


def test_filtering_then_reencoding_stays_contiguous():
    raw = np.array(["a", "b", "d", "e"] * 5 + ["c"])  # "c" has only 1 sample
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(raw)

    rare_code = encoder.transform(["c"])[0]
    keep_mask = y_encoded != rare_code
    y_filtered = y_encoded[keep_mask]

    # Naive filtering alone leaves a gap where "rare"'s code used to be.
    assert sorted(set(y_filtered)) != list(range(len(set(y_filtered))))

    # The fix: map back to raw labels, then re-fit a fresh encoder.
    raw_filtered = encoder.inverse_transform(y_filtered)
    new_encoder = LabelEncoder()
    y_reencoded = new_encoder.fit_transform(raw_filtered)

    assert sorted(set(y_reencoded)) == list(range(len(set(y_reencoded))))


def test_raw_mask_uses_raw_labels_not_encoded_ints():
    raw = np.array(["cat", "dog", "cat", "bird", "dog"])
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(raw)  # bird=0, cat=1, dog=2 (alphabetical)

    valid_codes = [1, 2]  # keep "cat" and "dog"
    valid_raw_labels = set(encoder.inverse_transform(valid_codes))
    assert valid_raw_labels == {"cat", "dog"}

    # Filtering raw values by raw labels (the fix) - correct.
    raw_mask = np.isin(raw, list(valid_raw_labels))
    assert raw_mask.tolist() == [True, True, True, False, True]

    # The old bug: comparing raw string values against encoded integers never matches.
    buggy_mask = np.isin(raw, list(valid_codes))
    assert not buggy_mask.any()
