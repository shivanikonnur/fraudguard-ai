from pathlib import Path
import pandas as pd

TARGET = "Class"
REQUIRED_COLUMNS = {"Time", "Amount", TARGET}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the public credit-card fraud CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download creditcard.csv and place it "
            "inside data/raw/ (see README.md)."
        )

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if df[TARGET].isna().any():
        raise ValueError("Target column 'Class' contains missing values.")

    labels = set(df[TARGET].unique())
    if not labels.issubset({0, 1}):
        raise ValueError("Target 'Class' must contain only 0 (genuine) and 1 (fraud).")
    if len(labels) != 2:
        raise ValueError("Dataset must contain both target classes (0 and 1).")

    df = df.copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


def validate_feature_frame(df: pd.DataFrame, expected_columns: list[str] | None = None):
    """Validate inference features and return a numeric-only frame."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame.")
    if expected_columns is not None:
        missing = set(expected_columns) - set(df.columns)
        extra = set(df.columns) - set(expected_columns)
        if missing or extra:
            raise ValueError(
                f"Feature columns mismatch. Missing: {sorted(missing)}; "
                f"unexpected: {sorted(extra)}"
            )
        df = df[expected_columns]
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"All model features must be numeric. Invalid: {non_numeric}")
    if df.isna().any().any():
        raise ValueError("Input features contain missing values.")
    return df
