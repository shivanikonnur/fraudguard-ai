import pandas as pd
import pytest
from fraudguard.data import validate_feature_frame


def test_validate_feature_frame_accepts_valid_numeric_data():
    frame = pd.DataFrame({"Time": [1.0], "Amount": [10.0]})
    result = validate_feature_frame(frame, ["Time", "Amount"])
    assert list(result.columns) == ["Time", "Amount"]


def test_validate_feature_frame_rejects_missing_columns():
    frame = pd.DataFrame({"Time": [1.0]})
    with pytest.raises(ValueError, match="Feature columns mismatch"):
        validate_feature_frame(frame, ["Time", "Amount"])


def test_validate_feature_frame_rejects_missing_values():
    frame = pd.DataFrame({"Time": [1.0], "Amount": [float("nan")]})
    with pytest.raises(ValueError, match="missing values"):
        validate_feature_frame(frame)
