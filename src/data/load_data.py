from pathlib import Path
import pandas as pd


def train():
    input_path = Path("/workspace", "data", "raw")
    return pd.read_csv(input_path / "train.csv")


def test():
    input_path = Path("/workspace", "data", "raw")
    return pd.read_csv(input_path / "test.csv")


def sample_submission():
    input_path = Path("/workspace", "data", "raw")
    return pd.read_csv(input_path / "sample_submission.csv")
